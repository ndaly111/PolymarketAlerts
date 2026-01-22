from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

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
