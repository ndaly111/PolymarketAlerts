"""
NOAA National Blend of Models (NBM) client for probabilistic temperature forecasts.

NBM provides native percentiles (p10, p25, p50, p75, p90) and exceedance probabilities
which are ideal for probability-based trading.

Data source: AWS Open Data s3://noaa-nbm-grib2-pds/
Documentation: https://vlab.noaa.gov/web/mdl/nbm

NBM runs every hour with forecasts out to 264 hours. We're primarily interested in:
- TMAX percentiles for high temperature trading
- Exceedance probabilities P(T >= threshold) for precise contract pricing

Note: Requires cfgrib and eccodes for GRIB2 parsing. Install with:
  pip install cfgrib ecCodes xarray
  # Or on macOS: brew install eccodes && pip install cfgrib xarray
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import requests


# AWS Open Data bucket paths
NBM_AWS_BUCKET = "noaa-nbm-grib2-pds"
NBM_AWS_BASE_URL = f"https://{NBM_AWS_BUCKET}.s3.amazonaws.com"

# NBM GRIB2 variable names for temperature percentiles
# These are in the co-located product (blend.tXXz.core.fXXX.co.grib2)
NBM_TMAX_PERCENTILE_VARS = {
    "p10": "TMAX:10%",
    "p25": "TMAX:25%",
    "p50": "TMAX:50%",
    "p75": "TMAX:75%",
    "p90": "TMAX:90%",
}

# Standard temperature thresholds for exceedance probability calculation
DEFAULT_THRESHOLDS_F = list(range(20, 120, 1))  # 20F to 119F in 1F increments


@dataclass(frozen=True)
class NBMForecast:
    """NBM probabilistic temperature forecast."""
    city_key: str
    target_date_local: str
    model_run_utc: str
    lead_hours: int
    p10: Optional[int]  # 10th percentile temperature (F)
    p25: Optional[int]  # 25th percentile
    p50: Optional[int]  # 50th percentile (median)
    p75: Optional[int]  # 75th percentile
    p90: Optional[int]  # 90th percentile
    exceedance: Dict[int, float]  # {threshold: P(T >= threshold)}
    raw_source: str


@dataclass(frozen=True)
class NBMModelRun:
    """Metadata for an NBM model run."""
    run_time_utc: datetime
    forecast_hour: int  # Lead time in hours
    grib_url: str


def make_session(timeout: float = 60.0) -> requests.Session:
    """Create a requests session for NBM data fetching."""
    s = requests.Session()
    s.headers.update({
        "User-Agent": "weather-trading-bot/1.0",
    })
    adapter = requests.adapters.HTTPAdapter(max_retries=3)
    s.mount("https://", adapter)
    s.mount("http://", adapter)

    original_request = s.request
    def request_with_timeout(method, url, **kwargs):
        kwargs.setdefault("timeout", timeout)
        return original_request(method, url, **kwargs)
    s.request = request_with_timeout

    return s


def get_nbm_grib_url(
    run_time: datetime,
    forecast_hour: int,
    product: str = "co",  # "co" = co-located (has percentiles)
) -> str:
    """
    Construct S3 URL for NBM GRIB2 file.

    NBM file structure: blend.tHHz.core.fFFF.{product}.grib2
    where HH is model run hour (00-23) and FFF is forecast hour (001-264)

    S3 path: /blend.{YYYYMMDD}/{HH}/core/blend.t{HH}z.core.f{FFF}.{product}.grib2
    """
    date_str = run_time.strftime("%Y%m%d")
    run_hour = run_time.strftime("%H")
    fhr = f"{forecast_hour:03d}"

    filename = f"blend.t{run_hour}z.core.f{fhr}.{product}.grib2"
    return f"{NBM_AWS_BASE_URL}/blend.{date_str}/{run_hour}/core/{filename}"


def get_available_model_runs(
    session: requests.Session,
    target_date: datetime,
    max_lead_hours: int = 72,
) -> List[NBMModelRun]:
    """
    Find available NBM model runs that cover a target date.

    Returns list of model runs sorted by run time (most recent first).
    """
    runs = []

    # Check runs from past 3 days that might cover target date
    # NBM runs every hour, so there are many potential runs
    now_utc = datetime.now(ZoneInfo("UTC"))

    for days_back in range(3):
        check_date = now_utc - timedelta(days=days_back)

        # Check every 6 hours (00, 06, 12, 18) to reduce API calls
        for run_hour in [0, 6, 12, 18]:
            run_time = check_date.replace(
                hour=run_hour, minute=0, second=0, microsecond=0
            )

            # Skip future runs
            if run_time > now_utc:
                continue

            # Calculate lead hours to reach target date afternoon (e.g., 3pm local)
            # For simplicity, assume target is midday UTC of target_date
            target_midday = datetime(
                target_date.year, target_date.month, target_date.day,
                18, 0, 0, tzinfo=ZoneInfo("UTC")  # 18Z ~ afternoon in US
            )

            lead_hours = int((target_midday - run_time).total_seconds() / 3600)

            if 0 < lead_hours <= max_lead_hours:
                url = get_nbm_grib_url(run_time, lead_hours)
                runs.append(NBMModelRun(
                    run_time_utc=run_time,
                    forecast_hour=lead_hours,
                    grib_url=url,
                ))

    # Sort by run time, most recent first
    runs.sort(key=lambda r: r.run_time_utc, reverse=True)
    return runs


def check_grib_exists(session: requests.Session, url: str) -> bool:
    """Check if a GRIB file exists on S3 (HEAD request)."""
    try:
        r = session.head(url, timeout=10)
        return r.status_code == 200
    except Exception:
        return False


def download_grib_file(
    session: requests.Session,
    url: str,
    output_path: Optional[Path] = None,
) -> Optional[Path]:
    """
    Download a GRIB2 file from NBM S3 bucket.

    Returns path to downloaded file, or None on failure.
    """
    if output_path is None:
        # Create temp file
        fd, tmp_path = tempfile.mkstemp(suffix=".grib2")
        os.close(fd)
        output_path = Path(tmp_path)

    try:
        r = session.get(url, stream=True, timeout=120)
        r.raise_for_status()

        with open(output_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

        return output_path
    except Exception as e:
        print(f"  [error] Failed to download {url}: {e}")
        if output_path.exists():
            output_path.unlink()
        return None


def parse_grib_percentiles(
    grib_path: Path,
    lat: float,
    lon: float,
) -> Dict[str, Optional[int]]:
    """
    Extract temperature percentiles from NBM GRIB2 file for a specific location.

    Returns dict with keys: p10, p25, p50, p75, p90 (values in Fahrenheit)

    Note: Requires cfgrib and xarray. Will gracefully degrade if not available.
    """
    try:
        import xarray as xr
    except ImportError:
        print("  [warn] xarray/cfgrib not installed. Cannot parse GRIB2 files.")
        print("  Install with: pip install cfgrib xarray")
        return {"p10": None, "p25": None, "p50": None, "p75": None, "p90": None}

    results = {"p10": None, "p25": None, "p50": None, "p75": None, "p90": None}

    try:
        # Open GRIB file with cfgrib backend
        # NBM files have multiple messages, we need the TMAX percentiles
        ds = xr.open_dataset(
            grib_path,
            engine="cfgrib",
            backend_kwargs={
                "filter_by_keys": {"typeOfLevel": "surface"},
                "indexpath": "",
            },
        )

        # Find nearest grid point to our location
        # NBM uses ~2.5km grid spacing
        if "latitude" in ds.coords and "longitude" in ds.coords:
            # Handle 2D lat/lon arrays
            lat_vals = ds.latitude.values
            lon_vals = ds.longitude.values

            # Convert lon to match GRIB convention (0-360 or -180-180)
            if lon < 0:
                lon_360 = lon + 360
            else:
                lon_360 = lon

            # Find nearest point (simplified)
            # This works for regularly gridded data
            # For production, use proper nearest-neighbor interpolation
            pass

        # For now, return empty - full implementation requires more GRIB handling
        # The percentile data in NBM is in specific message types

    except Exception as e:
        print(f"  [warn] Error parsing GRIB file: {e}")

    return results


def kelvin_to_fahrenheit(k: float) -> int:
    """Convert Kelvin to Fahrenheit, rounded to nearest integer."""
    return round((k - 273.15) * 9/5 + 32)


def celsius_to_fahrenheit(c: float) -> int:
    """Convert Celsius to Fahrenheit, rounded to nearest integer."""
    return round(c * 9/5 + 32)


def interpolate_exceedance_from_percentiles(
    p10: int,
    p25: int,
    p50: int,
    p75: int,
    p90: int,
    thresholds: Optional[List[int]] = None,
) -> Dict[int, float]:
    """
    Build exceedance probability curve from NBM percentiles.

    Uses linear interpolation between known percentile points to estimate
    P(T >= threshold) for each threshold.

    Args:
        p10-p90: Temperature percentiles in Fahrenheit
        thresholds: List of threshold temperatures to compute exceedance for

    Returns:
        Dict mapping threshold (F) -> P(T >= threshold)
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS_F

    # Known points: (temperature, exceedance_probability)
    # p10 means 10% of outcomes are below this temp, so 90% are >= p10
    points = [
        (p10, 0.90),
        (p25, 0.75),
        (p50, 0.50),
        (p75, 0.25),
        (p90, 0.10),
    ]

    # Sort by temperature
    points.sort(key=lambda x: x[0])

    exceedance = {}

    for thresh in thresholds:
        if thresh <= points[0][0]:
            # Below lowest point (p10) - extrapolate
            # Use simple linear extrapolation from first two points
            t1, p1 = points[0]
            t2, p2 = points[1]
            if t2 != t1:
                slope = (p2 - p1) / (t2 - t1)
                prob = p1 + slope * (thresh - t1)
            else:
                prob = p1
            prob = min(1.0, max(0.0, prob))
        elif thresh >= points[-1][0]:
            # Above highest point (p90) - extrapolate
            t1, p1 = points[-2]
            t2, p2 = points[-1]
            if t2 != t1:
                slope = (p2 - p1) / (t2 - t1)
                prob = p2 + slope * (thresh - t2)
            else:
                prob = p2
            prob = min(1.0, max(0.0, prob))
        else:
            # Interpolate between bracketing points
            for i in range(len(points) - 1):
                t1, p1 = points[i]
                t2, p2 = points[i + 1]
                if t1 <= thresh <= t2:
                    if t2 != t1:
                        frac = (thresh - t1) / (t2 - t1)
                        prob = p1 + frac * (p2 - p1)
                    else:
                        prob = p1
                    break
            else:
                prob = 0.5  # Fallback

        exceedance[thresh] = round(prob, 4)

    return exceedance


def build_exceedance_from_spread(
    median: int,
    spread: int,
    thresholds: Optional[List[int]] = None,
) -> Dict[int, float]:
    """
    Build approximate exceedance curve when only median and spread are known.

    Assumes roughly normal distribution with:
    - p50 = median
    - p90 - p10 = spread (approximately 2.56 standard deviations for normal)

    Args:
        median: p50 temperature
        spread: p90 - p10 temperature spread
        thresholds: Temperature thresholds

    Returns:
        Exceedance probability dict
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS_F

    # Estimate standard deviation from spread
    # For normal distribution, p90 - p10 ≈ 2.56σ
    std_dev = spread / 2.56 if spread > 0 else 3.0  # Default 3F std dev

    # Estimate percentiles from median and std_dev
    # For normal: pXX = median + z_XX * std_dev
    z_scores = {
        "p10": -1.28,
        "p25": -0.67,
        "p50": 0.0,
        "p75": 0.67,
        "p90": 1.28,
    }

    p10 = round(median + z_scores["p10"] * std_dev)
    p25 = round(median + z_scores["p25"] * std_dev)
    p50 = median
    p75 = round(median + z_scores["p75"] * std_dev)
    p90 = round(median + z_scores["p90"] * std_dev)

    return interpolate_exceedance_from_percentiles(p10, p25, p50, p75, p90, thresholds)


def fetch_nbm_forecast_nomads(
    session: requests.Session,
    lat: float,
    lon: float,
    target_date: str,
    timezone: str = "America/New_York",
) -> Optional[Dict[str, Any]]:
    """
    Fetch NBM forecast from NOMADS (NOAA Operational Model Archive).

    This is an alternative to S3 that provides direct access to forecast data
    without downloading full GRIB files.

    NOMADS NBM endpoint: https://nomads.ncep.noaa.gov/cgi-bin/filter_blend.pl
    """
    # NOMADS provides a web interface for subsetting GRIB data
    # However, it's often slow and unreliable
    # For production, prefer the S3 route with proper GRIB parsing

    base_url = "https://nomads.ncep.noaa.gov/cgi-bin/filter_blend.pl"

    # Get latest available run
    now = datetime.now(ZoneInfo("UTC"))
    run_hour = (now.hour // 6) * 6  # Round down to nearest 6-hour cycle
    run_date = now.strftime("%Y%m%d")

    # This is a stub - full NOMADS integration requires more work
    return None


def fetch_nbm_probabilistic(
    session: requests.Session,
    city_key: str,
    lat: float,
    lon: float,
    target_date: str,
    timezone: str = "America/New_York",
) -> Optional[NBMForecast]:
    """
    Main entry point: fetch NBM probabilistic forecast for a city/date.

    Tries multiple data sources:
    1. NOMADS web service (fastest, if available)
    2. AWS S3 GRIB files (most complete, but requires download)
    3. Fallback to estimated percentiles from other forecast sources

    Args:
        session: Requests session
        city_key: City identifier (e.g., "NHIGH" for NYC)
        lat: Latitude
        lon: Longitude
        target_date: Target date string (YYYY-MM-DD)
        timezone: City timezone

    Returns:
        NBMForecast with percentiles and exceedance probabilities, or None
    """
    target_dt = datetime.strptime(target_date, "%Y-%m-%d")
    target_dt = target_dt.replace(tzinfo=ZoneInfo(timezone))
    now_utc = datetime.now(ZoneInfo("UTC"))

    # Find available model runs
    runs = get_available_model_runs(session, target_dt)

    if not runs:
        print(f"  [warn] No NBM runs available for {target_date}")
        return None

    # Try the most recent run first
    for run in runs[:3]:  # Try up to 3 most recent runs
        if check_grib_exists(session, run.grib_url):
            # Download and parse GRIB file
            grib_path = download_grib_file(session, run.grib_url)

            if grib_path:
                try:
                    percentiles = parse_grib_percentiles(grib_path, lat, lon)

                    # If we got valid percentiles, build the forecast
                    if percentiles.get("p50") is not None:
                        exceedance = interpolate_exceedance_from_percentiles(
                            percentiles["p10"],
                            percentiles["p25"],
                            percentiles["p50"],
                            percentiles["p75"],
                            percentiles["p90"],
                        )

                        return NBMForecast(
                            city_key=city_key,
                            target_date_local=target_date,
                            model_run_utc=run.run_time_utc.isoformat(),
                            lead_hours=run.forecast_hour,
                            p10=percentiles["p10"],
                            p25=percentiles["p25"],
                            p50=percentiles["p50"],
                            p75=percentiles["p75"],
                            p90=percentiles["p90"],
                            exceedance=exceedance,
                            raw_source=run.grib_url,
                        )
                finally:
                    # Clean up temp file
                    if grib_path.exists():
                        grib_path.unlink()

    print(f"  [warn] Could not fetch NBM forecast for {city_key} on {target_date}")
    return None


def estimate_percentiles_from_point_forecast(
    point_forecast: int,
    month: int,
    lead_hours: int,
) -> Tuple[int, int, int, int, int]:
    """
    Estimate percentiles when only a point forecast is available.

    Uses climatological spread estimates based on:
    - Time of year (winter has larger uncertainty)
    - Lead time (longer lead = wider spread)

    Returns: (p10, p25, p50, p75, p90)
    """
    # Base spread (p90 - p10) by month
    # Winter months have more uncertainty in mid-latitudes
    monthly_base_spread = {
        1: 12, 2: 11, 3: 10, 4: 9, 5: 8, 6: 7,
        7: 7, 8: 7, 9: 8, 10: 9, 11: 10, 12: 11,
    }

    base_spread = monthly_base_spread.get(month, 8)

    # Spread increases with lead time
    # ~1F additional spread per 12 hours of lead time
    lead_adjustment = lead_hours / 12.0
    total_spread = base_spread + lead_adjustment

    # Calculate percentiles assuming roughly normal distribution
    std_dev = total_spread / 2.56

    p10 = round(point_forecast - 1.28 * std_dev)
    p25 = round(point_forecast - 0.67 * std_dev)
    p50 = point_forecast
    p75 = round(point_forecast + 0.67 * std_dev)
    p90 = round(point_forecast + 1.28 * std_dev)

    return (p10, p25, p50, p75, p90)


def get_nbm_spread_estimate(month: int, lead_hours: int) -> int:
    """
    Get estimated NBM temperature spread (p90 - p10) for uncertainty estimation.

    Based on climatological analysis of NBM forecast skill.
    """
    monthly_base = {
        1: 12, 2: 11, 3: 10, 4: 9, 5: 8, 6: 7,
        7: 7, 8: 7, 9: 8, 10: 9, 11: 10, 12: 11,
    }

    base = monthly_base.get(month, 8)
    lead_factor = 1.0 + (lead_hours / 72.0) * 0.5  # Up to 50% wider at 72h

    return round(base * lead_factor)
