from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests


@dataclass(frozen=True)
class NwsEndpoints:
    points_url: str
    forecast_hourly_url: str
    time_zone: Optional[str]


def make_session(user_agent: str, timeout: float = 25.0) -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": user_agent,
            "Accept": "application/geo+json, application/json;q=0.9, */*;q=0.5",
        }
    )
    adapter = requests.adapters.HTTPAdapter(max_retries=3)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.request = _wrap_timeout(s.request, timeout)
    return s


def _wrap_timeout(fn, timeout):
    def _inner(method, url, **kwargs):
        kwargs.setdefault("timeout", timeout)
        return fn(method, url, **kwargs)

    return _inner


def get_points_endpoints(s: requests.Session, lat: float, lon: float) -> NwsEndpoints:
    points_url = f"https://api.weather.gov/points/{lat:.4f},{lon:.4f}"
    r = s.get(points_url)
    r.raise_for_status()
    j = r.json()
    props = (j or {}).get("properties") or {}
    forecast_hourly = props.get("forecastHourly")
    if not forecast_hourly:
        raise RuntimeError("NWS points response missing properties.forecastHourly")
    tz = props.get("timeZone")
    return NwsEndpoints(points_url=points_url, forecast_hourly_url=forecast_hourly, time_zone=tz)


def fetch_json(s: requests.Session, url: str) -> Dict[str, Any]:
    r = s.get(url)
    r.raise_for_status()
    return r.json()


def polite_sleep():
    time.sleep(0.2)


# ---------------------------------------------------------------------------
# Station and observation functions for intraday tracking
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NwsStation:
    station_id: str
    name: str
    distance_km: Optional[float]


@dataclass(frozen=True)
class NwsObservation:
    station_id: str
    timestamp_utc: str
    temperature_f: int
    raw: Dict[str, Any]


def get_nearby_stations(
    s: requests.Session,
    lat: float,
    lon: float,
    limit: int = 5,
) -> List[NwsStation]:
    """
    Fetch nearby observation stations for a lat/lon point.

    Uses: https://api.weather.gov/points/{lat},{lon}/stations

    Returns list of stations ordered by distance (nearest first).
    """
    url = f"https://api.weather.gov/points/{lat:.4f},{lon:.4f}/stations"
    try:
        r = s.get(url)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return []

    features = data.get("features", [])
    stations: List[NwsStation] = []
    for feat in features[:limit]:
        props = feat.get("properties", {})
        station_id = props.get("stationIdentifier", "")
        name = props.get("name", "")
        if station_id:
            stations.append(NwsStation(
                station_id=station_id,
                name=name,
                distance_km=None,  # NWS returns ordered by proximity
            ))
    return stations


def get_latest_observation(
    s: requests.Session,
    station_id: str,
) -> Optional[NwsObservation]:
    """
    Fetch the latest observation from a station.

    Uses: https://api.weather.gov/stations/{stationId}/observations/latest

    Returns NwsObservation or None if unavailable.
    Temperature is converted from Celsius to Fahrenheit.
    """
    url = f"https://api.weather.gov/stations/{station_id}/observations/latest"
    try:
        r = s.get(url)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return None

    props = data.get("properties", {})
    timestamp = props.get("timestamp")

    # Temperature in NWS API is in Celsius under "temperature.value"
    temp_obj = props.get("temperature", {})
    temp_c = temp_obj.get("value")

    if temp_c is None or timestamp is None:
        return None

    # Convert C to F
    temp_f = int(round(temp_c * 9 / 5 + 32))

    return NwsObservation(
        station_id=station_id,
        timestamp_utc=timestamp,
        temperature_f=temp_f,
        raw=data,
    )
