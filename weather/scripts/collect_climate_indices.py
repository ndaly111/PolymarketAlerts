#!/usr/bin/env python3
"""
Collect climate teleconnection indices from NOAA and PSL.

Fetches monthly values for:
- NAO (North Atlantic Oscillation) — explains ~30% of NE winter variance
- AO (Arctic Oscillation) — polar vortex signal
- PNA (Pacific North American pattern) — West Coast temperatures
- MEI/ENSO (Multivariate ENSO Index) — El Niño / La Niña
- PDO (Pacific Decadal Oscillation) — longer-term Pacific pattern
- MJO (Madden-Julian Oscillation) — daily amplitude and phase

Monthly values are forward-filled to daily records in the DB.
MJO is stored daily.

Data sources (all free NOAA/NOAA-PSL text files):
- NAO: CPC monthly ascii table
- AO: CPC monthly ascii table
- PNA: CPC monthly ascii table
- MEI: NOAA PSL meiv2 data file
- PDO: NCEI ERSST v5 PDO index
- MJO: CPC RMM daily index (scraped from HTML)

Usage:
    # Update all indices (runs quickly, data is tiny)
    python -m weather.scripts.collect_climate_indices

    # Dry run — shows what would be written
    python -m weather.scripts.collect_climate_indices --dry-run

    # Verbose mode
    python -m weather.scripts.collect_climate_indices -v
"""

from __future__ import annotations

import argparse
import io
import math
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from weather.lib import db as db_lib

DEFAULT_DB = Path(os.getenv(
    "WEATHER_DB_PATH",
    str(ROOT / "weather" / "data" / "weather_forecast_accuracy.db")
))

# NOAA data URLs
NAO_URL = "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.nao.monthly.b5001.current.ascii.table"
AO_URL = "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/monthly.ao.index.b50.current.ascii.table"
PNA_URL = "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.pna.monthly.b5001.current.ascii.table"
MEI_URL = "https://psl.noaa.gov/enso/mei/data/meiv2.data"
PDO_URL = "https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/index/ersst.v5.pdo.dat"
# BOM (Australian Bureau of Meteorology) provides the Wheeler-Hendon RMM daily index
# Requires browser-like User-Agent; format: year month day RMM1 RMM2 phase amplitude
MJO_RMM_URL = "https://www.bom.gov.au/climate/mjo/graphics/rmm.74toRealtime.txt"


def make_session(timeout: float = 30.0) -> requests.Session:
    s = requests.Session()
    # Use a browser-like User-Agent; BOM requires it
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (compatible; weather-trading-bot/1.0)",
        "Accept": "text/plain, text/html, */*",
    })
    adapter = requests.adapters.HTTPAdapter(max_retries=3)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    # Wrap with timeout
    orig = s.request
    def _req(method, url, **kwargs):
        kwargs.setdefault("timeout", timeout)
        return orig(method, url, **kwargs)
    s.request = _req
    return s


def parse_cpc_monthly_table(text: str) -> Dict[Tuple[int, int], float]:
    """
    Parse CPC monthly ascii table format.

    Format: rows = years, columns = months Jan-Dec
    First column is year. Missing values are -99.90 or similar.

    Returns dict of {(year, month): value}.
    """
    values: Dict[Tuple[int, int], float] = {}
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            year = int(parts[0])
        except ValueError:
            continue
        for m, val_str in enumerate(parts[1:13], start=1):
            try:
                val = float(val_str)
                if abs(val) < 90:  # Filter missing (-99.90 etc)
                    values[(year, m)] = val
            except ValueError:
                continue
    return values


def parse_ao_table(text: str) -> Dict[Tuple[int, int], float]:
    """
    Parse AO monthly index table.

    Format:
    YEAR JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC
    1950  0.99  0.12  ...
    Or alternatively rows of: year month value
    """
    values: Dict[Tuple[int, int], float] = {}
    lines = text.strip().splitlines()

    # Try to detect format by looking at the first non-header line
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or line.lower().startswith("year"):
            continue
        parts = line.split()
        if not parts:
            continue
        try:
            year = int(parts[0])
        except ValueError:
            continue

        if len(parts) == 13:
            # Year + 12 monthly values
            for m, val_str in enumerate(parts[1:13], start=1):
                try:
                    val = float(val_str)
                    if abs(val) < 90:
                        values[(year, m)] = val
                except ValueError:
                    continue
        elif len(parts) == 3:
            # year month value
            try:
                month = int(parts[1])
                val = float(parts[2])
                if abs(val) < 90:
                    values[(year, month)] = val
            except ValueError:
                continue

    return values


def parse_mei_table(text: str) -> Dict[Tuple[int, int], float]:
    """
    Parse MEI v2 data file format.

    Format: year col1 col2 ... (bi-monthly periods Dec-Jan, Jan-Feb, ...)
    We use the column closest to midmonth. The 12 columns represent:
    DJFM JFMA FMAM MAMJ AMJJ MJJA JJAS JASO ASON NDJF ONDJ NDJF
    Actually MEI v2 has 12 2-month windows. We assign each to the second month.
    """
    values: Dict[Tuple[int, int], float] = {}
    # MEI bi-monthly: DJ JF FM MA AM MJ JJ JA AS SO ON ND
    # We'll map each 2-month period to the second month (1-indexed)
    second_months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    for line in text.strip().splitlines():
        line = line.strip()
        if not line or not line[0].isdigit():
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            year = int(parts[0])
        except ValueError:
            continue
        for i, val_str in enumerate(parts[1:13]):
            if i >= len(second_months):
                break
            try:
                val = float(val_str)
                if abs(val) < 90 and val != -999.0:
                    month = second_months[i]
                    values[(year, month)] = val
            except ValueError:
                continue

    return values


def parse_pdo_table(text: str) -> Dict[Tuple[int, int], float]:
    """
    Parse NCEI PDO index file.

    Format varies but typically: Year Jan Feb Mar ... Dec
    with missing values as 99.99 or -99.99
    """
    values: Dict[Tuple[int, int], float] = {}
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if not parts:
            continue
        try:
            year = int(parts[0])
        except ValueError:
            continue
        for m, val_str in enumerate(parts[1:13], start=1):
            try:
                val = float(val_str)
                if abs(val) < 50:  # PDO ranges roughly -5 to +5
                    values[(year, m)] = val
            except ValueError:
                continue
    return values


def parse_mjo_rmm_bom(text: str) -> List[Tuple[str, float, int]]:
    """
    Parse MJO RMM index from BOM (Australian Bureau of Meteorology) text file.

    Format: year month day RMM1 RMM2 phase amplitude [method]
    Missing value: 1.E36 or 999

    Returns list of (date_str, amplitude, phase).
    """
    results = []
    pattern = re.compile(
        r'^\s*(\d{4})\s+(\d{1,2})\s+(\d{1,2})\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)\s+(\d+)\s+([\d.Ee+]+)',
        re.MULTILINE,
    )
    for match in pattern.finditer(text):
        try:
            year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
            rmm1, rmm2 = float(match.group(4)), float(match.group(5))
            phase = int(match.group(6))
            amplitude_val = float(match.group(7))

            # Skip missing values
            if abs(rmm1) > 100 or abs(rmm2) > 100 or amplitude_val > 100:
                continue

            # Compute amplitude from RMM1/RMM2 as backup
            computed_amp = math.sqrt(rmm1**2 + rmm2**2)
            amp = amplitude_val if 0 < amplitude_val < 20 else computed_amp

            date_str = f"{year:04d}-{month:02d}-{day:02d}"
            results.append((date_str, amp, phase))
        except (ValueError, OverflowError):
            continue

    return results


def forward_fill_monthly(
    monthly_values: Dict[Tuple[int, int], float],
    start_date: str = "2020-01-01",
) -> Dict[str, float]:
    """
    Forward-fill monthly values to daily records.

    A monthly value (year, month) applies to all days in that month.
    Returns dict of {date_str: value}.
    """
    if not monthly_values:
        return {}

    daily: Dict[str, float] = {}
    # Sort monthly data
    sorted_months = sorted(monthly_values.items())

    # Build day-by-day from min year/month to today
    if sorted_months:
        min_year, min_month = sorted_months[0][0]
        today = datetime.now()

        current = datetime(min_year, min_month, 1)
        last_val = None

        while current <= today:
            key = (current.year, current.month)
            if key in monthly_values:
                last_val = monthly_values[key]
            if last_val is not None:
                daily[current.strftime("%Y-%m-%d")] = last_val
            current += timedelta(days=1)

    return daily


def fetch_and_parse_index(
    session: requests.Session,
    url: str,
    parser_fn,
    name: str,
    verbose: bool = False,
) -> Dict[Tuple[int, int], float]:
    """Generic fetch-and-parse helper."""
    if verbose:
        print(f"  Fetching {name}...")
    try:
        r = session.get(url)
        r.raise_for_status()
        result = parser_fn(r.text)
        if verbose:
            print(f"  {name}: {len(result)} month records")
        return result
    except Exception as e:
        print(f"  [warn] Failed to fetch {name}: {e}")
        return {}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect climate teleconnection indices (NAO, AO, PNA, MEI, PDO, MJO)."
    )
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    db_path = Path(args.db)
    db_lib.ensure_schema(db_path)

    session = make_session()
    fetched_at = datetime.utcnow().isoformat()

    print("Collecting climate teleconnection indices...")

    # --- Monthly indices ---
    nao_monthly = fetch_and_parse_index(session, NAO_URL, parse_cpc_monthly_table, "NAO", args.verbose)
    ao_monthly = fetch_and_parse_index(session, AO_URL, parse_ao_table, "AO", args.verbose)
    pna_monthly = fetch_and_parse_index(session, PNA_URL, parse_cpc_monthly_table, "PNA", args.verbose)
    mei_monthly = fetch_and_parse_index(session, MEI_URL, parse_mei_table, "MEI", args.verbose)
    pdo_monthly = fetch_and_parse_index(session, PDO_URL, parse_pdo_table, "PDO", args.verbose)

    # Forward-fill monthly to daily
    nao_daily = forward_fill_monthly(nao_monthly)
    ao_daily = forward_fill_monthly(ao_monthly)
    pna_daily = forward_fill_monthly(pna_monthly)
    mei_daily = forward_fill_monthly(mei_monthly)
    pdo_daily = forward_fill_monthly(pdo_monthly)

    # Get union of all dates covered
    all_dates = set(nao_daily) | set(ao_daily) | set(pna_daily) | set(mei_daily) | set(pdo_daily)

    # --- MJO daily (BOM Wheeler-Hendon RMM index) ---
    mjo_by_date: Dict[str, Tuple[float, int]] = {}
    if args.verbose:
        print("  Fetching MJO RMM (BOM)...")
    try:
        r = session.get(MJO_RMM_URL)
        r.raise_for_status()
        mjo_records = parse_mjo_rmm_bom(r.text)
        for date_str, amp, phase in mjo_records:
            mjo_by_date[date_str] = (amp, phase)
        if args.verbose:
            print(f"  MJO: {len(mjo_by_date)} daily records")
    except Exception as e:
        print(f"  [warn] Failed to fetch MJO: {e}")

    all_dates |= set(mjo_by_date)

    if not all_dates:
        print("[error] No data fetched for any index")
        return 1

    # Upsert all dates
    written = 0
    for date_str in sorted(all_dates):
        mjo_amp, mjo_phase = None, None
        if date_str in mjo_by_date:
            mjo_amp, mjo_phase = mjo_by_date[date_str]

        if args.dry_run:
            if args.verbose:
                print(
                    f"  [dry-run] {date_str}: "
                    f"nao={nao_daily.get(date_str):.2f if date_str in nao_daily else 'N/A'}  "
                    f"ao={ao_daily.get(date_str):.2f if date_str in ao_daily else 'N/A'}"
                )
            written += 1
            continue

        try:
            db_lib.upsert_climate_indices(
                db_path,
                date_local=date_str,
                nao=nao_daily.get(date_str),
                ao=ao_daily.get(date_str),
                pna=pna_daily.get(date_str),
                mei_enso=mei_daily.get(date_str),
                pdo=pdo_daily.get(date_str),
                mjo_amplitude=mjo_amp,
                mjo_phase=mjo_phase,
                fetched_at_utc=fetched_at,
            )
            written += 1
        except Exception as e:
            print(f"  [error] DB write failed for {date_str}: {e}")

    print(f"Done: {written} date records written to climate_indices")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
