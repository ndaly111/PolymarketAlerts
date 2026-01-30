"""
METAR parser for visibility and fog data.

METAR (Meteorological Aerodrome Report) provides hourly weather observations
from airports, including visibility and ceiling height.

Sources:
- Aviation Weather Center: https://aviationweather.gov/api/data/metar
- IEM ASOS Archive: https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py

Usage:
    from weather.lib.metar import fetch_metar_visibility

    vis = fetch_metar_visibility(station_id="KSFO", date="2024-01-15")
    if vis:
        print(f"Min visibility: {vis.visibility_min_miles} miles")
        print(f"Fog hours: {vis.fog_hours}")
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode


@dataclass
class VisibilityObservation:
    """Single METAR visibility observation."""
    date: str  # YYYY-MM-DD
    observation_time_utc: str  # ISO timestamp
    station_id: str
    visibility_miles: float
    ceiling_ft: Optional[int]  # Ceiling height in feet
    fog_present: bool
    source: str = "metar"


@dataclass
class DailyVisibilitySummary:
    """Daily summary of visibility observations."""
    date: str
    station_id: str
    visibility_min_miles: float  # Minimum visibility during day
    visibility_avg_miles: float  # Average visibility
    fog_hours: int  # Hours with visibility < 1 mile
    low_visibility_hours: int  # Hours with visibility < 3 miles
    obs_count: int  # Number of observations
    source: str = "metar"


def _parse_visibility(vis_str: str) -> Optional[float]:
    """
    Parse visibility from METAR format.

    Examples:
        "10SM" -> 10.0
        "1/2SM" -> 0.5
        "1 1/2SM" -> 1.5
        "P6SM" -> 10.0 (greater than 6)
        "M1/4SM" -> 0.125 (less than 1/4)
    """
    if not vis_str:
        return None

    vis_str = vis_str.strip().upper()

    # Greater than 6 miles
    if vis_str.startswith("P"):
        return 10.0

    # Less than (use half the value)
    if vis_str.startswith("M"):
        vis_str = vis_str[1:]
        multiplier = 0.5
    else:
        multiplier = 1.0

    # Remove SM suffix
    vis_str = vis_str.replace("SM", "").strip()

    # Handle fractions
    if "/" in vis_str:
        parts = vis_str.split()
        if len(parts) == 2:
            # "1 1/2" -> 1.5
            whole = int(parts[0])
            frac = parts[1].split("/")
            value = whole + int(frac[0]) / int(frac[1])
        else:
            # "1/2" -> 0.5
            frac = vis_str.split("/")
            value = int(frac[0]) / int(frac[1])
    else:
        try:
            value = float(vis_str)
        except ValueError:
            return None

    return value * multiplier


def _parse_ceiling(metar_str: str) -> Optional[int]:
    """
    Parse ceiling height from METAR.

    Ceiling is the lowest broken (BKN) or overcast (OVC) layer.
    Height is in hundreds of feet (e.g., "OVC015" = 1500 ft).
    """
    # Look for BKN or OVC with height
    pattern = r"(BKN|OVC)(\d{3})"
    matches = re.findall(pattern, metar_str.upper())

    if not matches:
        return None

    # Find lowest BKN/OVC layer
    lowest = min(int(m[1]) * 100 for m in matches)
    return lowest


def _is_fog(visibility_miles: float, metar_str: str) -> bool:
    """
    Determine if fog is present.

    Fog (FG) is defined as visibility < 1 mile due to water droplets.
    """
    # Check for explicit fog codes
    if "FG" in metar_str.upper() or "FZFG" in metar_str.upper():
        return True

    # Also consider mist (BR) with very low visibility as fog-like
    if visibility_miles < 1.0 and "BR" in metar_str.upper():
        return True

    # Very low visibility without precipitation is likely fog
    if visibility_miles < 0.5:
        # Check it's not rain/snow causing low visibility
        if not any(code in metar_str.upper() for code in ["RA", "SN", "PL", "GR"]):
            return True

    return False


def _parse_metar_line(line: str, station_id: str) -> Optional[VisibilityObservation]:
    """Parse a single METAR observation line."""
    # METAR format varies, but typically:
    # KSFO 151256Z 27005KT 10SM FEW020 BKN200 12/07 A3012
    parts = line.split()
    if len(parts) < 5:
        return None

    # Find station ID
    station_idx = -1
    for i, p in enumerate(parts):
        if p.upper() == station_id.upper():
            station_idx = i
            break

    if station_idx < 0:
        return None

    # Time is next part (DDHHMM format)
    if station_idx + 1 >= len(parts):
        return None

    time_str = parts[station_idx + 1]
    time_match = re.match(r"(\d{2})(\d{2})(\d{2})Z", time_str)
    if not time_match:
        return None

    day = int(time_match.group(1))
    hour = int(time_match.group(2))
    minute = int(time_match.group(3))

    # Find visibility (ends with SM)
    vis_miles = None
    for part in parts[station_idx + 2:]:
        if "SM" in part.upper():
            vis_miles = _parse_visibility(part)
            break

    if vis_miles is None:
        return None

    # Parse ceiling
    ceiling = _parse_ceiling(line)

    # Check for fog
    fog = _is_fog(vis_miles, line)

    # We need to get the date from context
    return VisibilityObservation(
        date="",  # Will be filled in by caller
        observation_time_utc="",  # Will be filled in
        station_id=station_id,
        visibility_miles=vis_miles,
        ceiling_ft=ceiling,
        fog_present=fog,
        source="metar",
    )


def fetch_metar_awc(
    station_id: str,
    date: str,
    timeout: int = 30,
) -> List[VisibilityObservation]:
    """
    Fetch METAR data from Aviation Weather Center API.

    Note: AWC only provides recent data (last 3-5 days typically).
    """
    # AWC API endpoint
    date_dt = datetime.strptime(date, "%Y-%m-%d")
    start_time = date_dt.strftime("%Y-%m-%dT00:00:00Z")
    end_time = (date_dt + timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")

    params = {
        "ids": station_id.upper(),
        "format": "raw",
        "taf": "false",
        "hours": "24",
        "date": start_time,
    }

    url = f"https://aviationweather.gov/api/data/metar?{urlencode(params)}"

    try:
        req = Request(url, headers={"User-Agent": "weather-ml-model/1.0"})
        with urlopen(req, timeout=timeout) as resp:
            data = resp.read().decode("utf-8", errors="ignore")
    except (HTTPError, URLError, TimeoutError):
        return []

    observations = []
    for line in data.strip().split("\n"):
        if not line.strip():
            continue

        obs = _parse_metar_line(line, station_id)
        if obs:
            obs.date = date
            # TODO: Parse actual observation time
            observations.append(obs)

    return observations


def fetch_metar_iem(
    station_id: str,
    date: str,
    timeout: int = 60,
) -> List[VisibilityObservation]:
    """
    Fetch METAR data from Iowa Environmental Mesonet (IEM) ASOS archive.

    IEM has historical data going back many years.
    """
    # IEM ASOS request
    date_dt = datetime.strptime(date, "%Y-%m-%d")
    end_dt = date_dt + timedelta(days=1)

    # Remove K prefix if present for IEM
    iem_station = station_id.upper()
    if iem_station.startswith("K") and len(iem_station) == 4:
        iem_station = iem_station[1:]

    params = {
        "station": iem_station,
        "data": "all",
        "year1": date_dt.year,
        "month1": date_dt.month,
        "day1": date_dt.day,
        "year2": end_dt.year,
        "month2": end_dt.month,
        "day2": end_dt.day,
        "tz": "Etc/UTC",
        "format": "onlycomma",
        "latlon": "no",
        "elev": "no",
        "missing": "M",
        "trace": "T",
        "direct": "no",
        "report_type": "3",  # METAR only
    }

    url = f"https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?{urlencode(params)}"

    try:
        req = Request(url, headers={"User-Agent": "weather-ml-model/1.0"})
        with urlopen(req, timeout=timeout) as resp:
            data = resp.read().decode("utf-8", errors="ignore")
    except (HTTPError, URLError, TimeoutError):
        return []

    observations = []
    lines = data.strip().split("\n")

    # First line is header
    if len(lines) < 2:
        return []

    header = lines[0].split(",")
    try:
        vis_idx = header.index("vsby")
        time_idx = header.index("valid")
        metar_idx = header.index("metar") if "metar" in header else -1
    except ValueError:
        return []

    for line in lines[1:]:
        if not line.strip():
            continue

        parts = line.split(",")
        if len(parts) <= max(vis_idx, time_idx):
            continue

        # Parse visibility
        vis_str = parts[vis_idx].strip()
        if vis_str == "M" or not vis_str:
            continue

        try:
            vis_miles = float(vis_str)
        except ValueError:
            continue

        # Parse time
        time_str = parts[time_idx].strip()
        try:
            obs_dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M")
        except ValueError:
            continue

        # Check for fog in METAR text if available
        metar_text = parts[metar_idx] if metar_idx >= 0 and len(parts) > metar_idx else ""
        fog = _is_fog(vis_miles, metar_text)

        # Parse ceiling if METAR available
        ceiling = _parse_ceiling(metar_text) if metar_text else None

        observations.append(VisibilityObservation(
            date=date,
            observation_time_utc=obs_dt.isoformat() + "Z",
            station_id=station_id,
            visibility_miles=vis_miles,
            ceiling_ft=ceiling,
            fog_present=fog,
            source="iem_asos",
        ))

    return observations


def fetch_metar_visibility(
    station_id: str,
    date: str,
    timeout: int = 60,
) -> Optional[DailyVisibilitySummary]:
    """
    Fetch visibility observations for a station and date.

    Tries IEM archive first (has historical data), falls back to AWC for recent.

    Returns daily summary with min/avg visibility and fog hours.
    """
    # Try IEM first (better historical coverage)
    observations = fetch_metar_iem(station_id, date, timeout)

    # Fall back to AWC for recent data
    if not observations:
        observations = fetch_metar_awc(station_id, date, timeout)

    if not observations:
        return None

    # Compute summary statistics
    visibilities = [o.visibility_miles for o in observations]
    fog_count = sum(1 for o in observations if o.fog_present)
    low_vis_count = sum(1 for o in observations if o.visibility_miles < 3.0)

    return DailyVisibilitySummary(
        date=date,
        station_id=station_id,
        visibility_min_miles=min(visibilities),
        visibility_avg_miles=sum(visibilities) / len(visibilities),
        fog_hours=fog_count,
        low_visibility_hours=low_vis_count,
        obs_count=len(observations),
        source=observations[0].source if observations else "metar",
    )


def get_station_for_city(city_config: Dict) -> Optional[str]:
    """Get METAR station ID for a city from its config."""
    return city_config.get("mos_station")
