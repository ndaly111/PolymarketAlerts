"""
Open-Meteo API client for weather forecasts.

Open-Meteo provides free access to high-quality weather models including:
- HRRR (High-Resolution Rapid Refresh) - Best for 0-48 hour forecasts in the US
- ECMWF IFS - Most accurate global model
- GFS - NOAA's global model
- "best_match" - Automatically selects the best model for the location

Open-Meteo typically provides more accurate forecasts than NWS for short-term
high temperature predictions.

API Documentation: https://open-meteo.com/en/docs
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import requests


BASE_URL = "https://api.open-meteo.com/v1/forecast"


@dataclass(frozen=True)
class OpenMeteoForecast:
    """Daily high temperature forecast from Open-Meteo."""
    date_local: str
    high_f: int
    model: str
    raw: Dict[str, Any]


def make_session(timeout: float = 25.0) -> requests.Session:
    """Create a requests session for Open-Meteo API calls."""
    s = requests.Session()
    s.headers.update({
        "User-Agent": "weather-trading-bot/1.0",
        "Accept": "application/json",
    })
    adapter = requests.adapters.HTTPAdapter(max_retries=3)
    s.mount("https://", adapter)
    s.mount("http://", adapter)

    # Wrap request method with default timeout
    original_request = s.request
    def request_with_timeout(method, url, **kwargs):
        kwargs.setdefault("timeout", timeout)
        return original_request(method, url, **kwargs)
    s.request = request_with_timeout

    return s


def get_daily_high_forecast(
    session: requests.Session,
    lat: float,
    lon: float,
    target_date: str,
    timezone: str = "America/New_York",
    model: str = "best_match",
) -> Optional[OpenMeteoForecast]:
    """
    Fetch the daily high temperature forecast from Open-Meteo.

    Args:
        session: Requests session
        lat: Latitude
        lon: Longitude
        target_date: Date in YYYY-MM-DD format
        timezone: Timezone for the forecast
        model: Weather model to use. Options:
            - "best_match" (default) - Auto-selects best model for location
            - "hrrr" - HRRR 3km (best for US 0-48h)
            - "gfs" - NOAA GFS global
            - "ecmwf" - ECMWF IFS (most accurate global)

    Returns:
        OpenMeteoForecast or None if unavailable
    """
    # Map friendly model names to API model names
    model_map = {
        "best_match": None,  # Don't specify - use default
        "hrrr": "ncep_hrrr_conus",
        "gfs": "ncep_gfs_graphcast025",
        "ecmwf": "ecmwf_ifs025",
    }

    api_model = model_map.get(model.lower(), model)

    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max",
        "temperature_unit": "fahrenheit",
        "timezone": timezone,
        "forecast_days": 7,  # Get a week of forecasts
    }

    if api_model:
        params["models"] = api_model

    try:
        r = session.get(BASE_URL, params=params)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"  [warn] Open-Meteo API error: {e}")
        return None

    # Parse response
    daily = data.get("daily", {})
    dates = daily.get("time", [])
    temps = daily.get("temperature_2m_max", [])

    if not dates or not temps:
        return None

    # Find the target date
    for i, date in enumerate(dates):
        if date == target_date:
            high_f = round(temps[i])
            return OpenMeteoForecast(
                date_local=target_date,
                high_f=high_f,
                model=api_model or "best_match",
                raw=data,
            )

    return None


def get_multi_model_forecast(
    session: requests.Session,
    lat: float,
    lon: float,
    target_date: str,
    timezone: str = "America/New_York",
) -> Dict[str, Optional[int]]:
    """
    Fetch forecasts from multiple models for comparison.

    Returns dict of model -> high_f (or None if unavailable)
    """
    models = ["best_match", "hrrr", "gfs", "ecmwf"]
    results = {}

    for model in models:
        forecast = get_daily_high_forecast(
            session, lat, lon, target_date, timezone, model
        )
        results[model] = forecast.high_f if forecast else None

    return results


def get_hourly_forecast(
    session: requests.Session,
    lat: float,
    lon: float,
    target_date: str,
    timezone: str = "America/New_York",
    model: str = "best_match",
) -> Optional[Dict[str, Any]]:
    """
    Fetch hourly forecast for a specific date.

    This can be used to compute the hourly max (similar to NWS approach)
    or for intraday analysis.

    Returns dict with 'hours' (list of timestamps) and 'temps' (list of temps in F)
    """
    model_map = {
        "best_match": None,
        "hrrr": "ncep_hrrr_conus",
        "gfs": "ncep_gfs_graphcast025",
        "ecmwf": "ecmwf_ifs025",
    }

    api_model = model_map.get(model.lower(), model)

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m",
        "temperature_unit": "fahrenheit",
        "timezone": timezone,
        "forecast_days": 7,
    }

    if api_model:
        params["models"] = api_model

    try:
        r = session.get(BASE_URL, params=params)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return None

    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    temps = hourly.get("temperature_2m", [])

    if not times or not temps:
        return None

    # Filter to target date
    tz = ZoneInfo(timezone)
    target_hours = []
    target_temps = []

    for time_str, temp in zip(times, temps):
        try:
            dt = datetime.fromisoformat(time_str)
            if dt.date().isoformat() == target_date:
                target_hours.append(time_str)
                target_temps.append(temp)
        except Exception:
            continue

    if not target_temps:
        return None

    return {
        "hours": target_hours,
        "temps": target_temps,
        "max_temp": round(max(target_temps)),
        "model": api_model or "best_match",
    }
