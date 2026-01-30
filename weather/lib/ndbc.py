"""
NDBC (National Data Buoy Center) client for sea surface temperature data.

NDBC provides real-time and historical data from ocean buoys across US coasts.

Real-time data: https://www.ndbc.noaa.gov/data/realtime2/
Historical data: https://www.ndbc.noaa.gov/data/historical/stdmet/

Usage:
    from weather.lib.ndbc import fetch_buoy_sst, BUOY_MAPPING

    sst = fetch_buoy_sst(buoy_id="46025", date="2024-01-15")
    if sst:
        print(f"SST: {sst.sst_f}°F")
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError


@dataclass
class SSTObservation:
    """Sea surface temperature observation from NDBC buoy."""
    date: str  # YYYY-MM-DD
    observation_time_utc: str  # ISO timestamp
    buoy_id: str
    sst_f: float  # Sea surface temperature in Fahrenheit
    sst_anomaly_f: Optional[float]  # Deviation from seasonal average (optional)
    source: str = "ndbc"


# Buoy mapping for coastal cities
# Each city is mapped to the nearest relevant NDBC buoy
BUOY_MAPPING: Dict[str, str] = {
    "LAXHIGH": "46025",  # Santa Monica Basin
    "SFOHIGH": "46026",  # San Francisco
    "SEAHIGH": "46029",  # Columbia River Bar
    "MIAHIGH": "FWYF1",  # Fowey Rocks, FL
    "NOLHIGH": "42040",  # Gulf of Mexico
}

# Seasonal SST climatology (approximate, in Fahrenheit)
# Used to calculate anomalies
SST_CLIMATOLOGY: Dict[str, Dict[int, float]] = {
    "46025": {  # LA
        1: 58, 2: 57, 3: 58, 4: 59, 5: 61, 6: 64,
        7: 67, 8: 68, 9: 68, 10: 66, 11: 63, 12: 59,
    },
    "46026": {  # SF
        1: 52, 2: 52, 3: 52, 4: 53, 5: 54, 6: 55,
        7: 56, 8: 57, 9: 58, 10: 57, 11: 55, 12: 53,
    },
    "46029": {  # Seattle
        1: 48, 2: 47, 3: 48, 4: 50, 5: 52, 6: 55,
        7: 58, 8: 60, 9: 58, 10: 55, 11: 51, 12: 49,
    },
    "FWYF1": {  # Miami
        1: 75, 2: 74, 3: 76, 4: 78, 5: 81, 6: 84,
        7: 85, 8: 86, 9: 85, 10: 82, 11: 79, 12: 76,
    },
    "42040": {  # New Orleans (Gulf)
        1: 64, 2: 63, 3: 66, 4: 72, 5: 78, 6: 84,
        7: 86, 8: 87, 9: 85, 10: 79, 11: 72, 12: 66,
    },
}


def _celsius_to_fahrenheit(c: float) -> float:
    """Convert Celsius to Fahrenheit."""
    return c * 9 / 5 + 32


def _parse_realtime_data(data: str, buoy_id: str) -> List[tuple[datetime, float]]:
    """
    Parse NDBC realtime2 .txt file format.

    Format: YY MM DD hh mm WDIR WSPD GST WVHT DPD APD MWD PRES ATMP WTMP DEWP VIS PTDY TIDE
    WTMP is water temperature in Celsius
    """
    lines = data.strip().split("\n")
    observations = []

    for line in lines:
        if line.startswith("#") or line.startswith("YY"):
            continue

        parts = line.split()
        if len(parts) < 15:
            continue

        try:
            year = int(parts[0])
            if year < 100:
                year += 2000
            month = int(parts[1])
            day = int(parts[2])
            hour = int(parts[3])
            minute = int(parts[4])

            # WTMP is at index 14 (0-indexed)
            wtmp = parts[14]
            if wtmp == "MM" or wtmp == "99.0":
                continue

            sst_c = float(wtmp)
            if sst_c < -5 or sst_c > 40:  # Sanity check
                continue

            dt = datetime(year, month, day, hour, minute)
            observations.append((dt, sst_c))
        except (ValueError, IndexError):
            continue

    return observations


def _parse_historical_data(data: str, buoy_id: str) -> List[tuple[datetime, float]]:
    """
    Parse NDBC historical stdmet file format.

    Format varies by year, but generally:
    #YY  MM DD hh mm WDIR WSPD GST  WVHT   DPD   APD MWD   PRES  ATMP  WTMP  DEWP  VIS PTDY  TIDE
    """
    lines = data.strip().split("\n")
    observations = []

    for line in lines:
        if line.startswith("#"):
            continue

        parts = line.split()
        if len(parts) < 15:
            continue

        try:
            year = int(parts[0])
            if year < 100:
                year += 1900 if year > 50 else 2000
            month = int(parts[1])
            day = int(parts[2])
            hour = int(parts[3])
            minute = int(parts[4]) if len(parts[4]) <= 2 else 0

            # Find WTMP - position varies, typically around index 14
            # Look for water temp column (should be reasonable value)
            wtmp_idx = 14
            if len(parts) > wtmp_idx:
                wtmp = parts[wtmp_idx]
                if wtmp == "MM" or wtmp == "99.0" or wtmp == "999.0":
                    continue

                sst_c = float(wtmp)
                if sst_c < -5 or sst_c > 40:
                    continue

                dt = datetime(year, month, day, hour, minute)
                observations.append((dt, sst_c))
        except (ValueError, IndexError):
            continue

    return observations


def fetch_buoy_sst_realtime(
    buoy_id: str,
    date: str,
    timeout: int = 30,
) -> Optional[SSTObservation]:
    """
    Fetch SST from NDBC realtime data (last 45 days).

    Args:
        buoy_id: NDBC buoy station ID
        date: Date string YYYY-MM-DD
        timeout: Request timeout

    Returns:
        SSTObservation or None
    """
    url = f"https://www.ndbc.noaa.gov/data/realtime2/{buoy_id}.txt"

    try:
        req = Request(url, headers={"User-Agent": "weather-ml-model/1.0"})
        with urlopen(req, timeout=timeout) as resp:
            data = resp.read().decode("utf-8", errors="ignore")
    except (HTTPError, URLError, TimeoutError):
        return None

    observations = _parse_realtime_data(data, buoy_id)
    if not observations:
        return None

    # Find observations for the target date
    target_dt = datetime.strptime(date, "%Y-%m-%d")
    target_end = target_dt + timedelta(days=1)

    day_obs = [(dt, sst) for dt, sst in observations
               if target_dt <= dt < target_end]

    if not day_obs:
        return None

    # Use daily average
    avg_sst_c = sum(sst for _, sst in day_obs) / len(day_obs)
    avg_sst_f = _celsius_to_fahrenheit(avg_sst_c)

    # Latest observation time
    latest_dt = max(dt for dt, _ in day_obs)

    # Calculate anomaly
    anomaly = None
    if buoy_id in SST_CLIMATOLOGY:
        climo = SST_CLIMATOLOGY[buoy_id].get(target_dt.month, 60)
        anomaly = avg_sst_f - climo

    return SSTObservation(
        date=date,
        observation_time_utc=latest_dt.isoformat() + "Z",
        buoy_id=buoy_id,
        sst_f=round(avg_sst_f, 1),
        sst_anomaly_f=round(anomaly, 1) if anomaly is not None else None,
        source="ndbc_realtime",
    )


def fetch_buoy_sst_historical(
    buoy_id: str,
    year: int,
    timeout: int = 60,
) -> List[tuple[str, float, Optional[float]]]:
    """
    Fetch SST from NDBC historical archive for a full year.

    Returns list of (date_str, sst_f, anomaly_f) tuples.
    """
    url = f"https://www.ndbc.noaa.gov/data/historical/stdmet/{buoy_id}h{year}.txt.gz"

    try:
        import gzip
        import io

        req = Request(url, headers={"User-Agent": "weather-ml-model/1.0"})
        with urlopen(req, timeout=timeout) as resp:
            compressed = resp.read()

        with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as gz:
            data = gz.read().decode("utf-8", errors="ignore")
    except (HTTPError, URLError, TimeoutError):
        return []
    except Exception:
        return []

    observations = _parse_historical_data(data, buoy_id)
    if not observations:
        return []

    # Group by date and compute daily averages
    daily: Dict[str, List[float]] = {}
    for dt, sst_c in observations:
        date_str = dt.strftime("%Y-%m-%d")
        if date_str not in daily:
            daily[date_str] = []
        daily[date_str].append(sst_c)

    results = []
    for date_str, sst_list in sorted(daily.items()):
        avg_sst_c = sum(sst_list) / len(sst_list)
        avg_sst_f = _celsius_to_fahrenheit(avg_sst_c)

        # Calculate anomaly
        anomaly = None
        month = int(date_str.split("-")[1])
        if buoy_id in SST_CLIMATOLOGY:
            climo = SST_CLIMATOLOGY[buoy_id].get(month, 60)
            anomaly = avg_sst_f - climo

        results.append((
            date_str,
            round(avg_sst_f, 1),
            round(anomaly, 1) if anomaly is not None else None,
        ))

    return results


def fetch_buoy_sst(
    buoy_id: str,
    date: str,
    timeout: int = 30,
) -> Optional[SSTObservation]:
    """
    Fetch SST for a specific buoy and date.

    Tries realtime data first, falls back to historical archive.

    Args:
        buoy_id: NDBC buoy station ID
        date: Date string YYYY-MM-DD
        timeout: Request timeout

    Returns:
        SSTObservation or None
    """
    # Try realtime first (last 45 days)
    obs = fetch_buoy_sst_realtime(buoy_id, date, timeout)
    if obs is not None:
        return obs

    # Fall back to historical archive
    date_dt = datetime.strptime(date, "%Y-%m-%d")
    year = date_dt.year

    historical = fetch_buoy_sst_historical(buoy_id, year, timeout)
    for hist_date, sst_f, anomaly in historical:
        if hist_date == date:
            return SSTObservation(
                date=date,
                observation_time_utc=f"{date}T12:00:00Z",  # Approximate
                buoy_id=buoy_id,
                sst_f=sst_f,
                sst_anomaly_f=anomaly,
                source="ndbc_historical",
            )

    return None


def get_buoy_for_city(city_key: str) -> Optional[str]:
    """Get the appropriate buoy ID for a city, or None if not coastal."""
    return BUOY_MAPPING.get(city_key)
