"""
NOAA HMS (Hazard Mapping System) Fire and Smoke Product client.

HMS provides smoke plume analysis from satellite data.
Data is available near real-time and in archive.

Source: https://satepsanone.nesdis.noaa.gov/pub/FIRE/web/HMS/

Usage:
    from weather.lib.hms_smoke import fetch_smoke_for_location

    smoke = fetch_smoke_for_location(lat=34.05, lon=-118.24, date="2024-08-15")
    if smoke and smoke.smoke_present:
        print(f"Smoke detected! Density: {smoke.smoke_density}")
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError
import xml.etree.ElementTree as ET


@dataclass
class SmokeObservation:
    """Smoke observation from HMS."""
    date: str  # YYYY-MM-DD
    observation_time_utc: str  # ISO timestamp
    lat: float
    lon: float
    smoke_present: bool
    smoke_density: Optional[str]  # "light", "medium", "heavy"
    source: str = "hms"


# West Coast cities most affected by wildfire smoke
SMOKE_RELEVANT_CITIES = ["LAXHIGH", "SFOHIGH", "SEAHIGH", "LVHIGH", "DENHIGH"]

# Fire season months (when to actively collect smoke data)
FIRE_SEASON_MONTHS = [7, 8, 9, 10, 11]  # Jul-Nov


def is_fire_season(date: datetime) -> bool:
    """Check if date is during typical Western US fire season."""
    return date.month in FIRE_SEASON_MONTHS


def is_smoke_relevant_city(city_key: str) -> bool:
    """Check if city is in smoke-relevant region."""
    return city_key in SMOKE_RELEVANT_CITIES


def _parse_kml_smoke_polygons(kml_data: str) -> List[Tuple[str, List[Tuple[float, float]]]]:
    """
    Parse HMS KML file to extract smoke polygons.

    Returns list of (density, polygon_coords) tuples.
    """
    polygons = []

    try:
        # Handle namespace
        kml_data = kml_data.replace('xmlns=', 'xmlns_removed=')
        root = ET.fromstring(kml_data)

        for placemark in root.iter("Placemark"):
            # Get density from name or description
            name_elem = placemark.find("name")
            desc_elem = placemark.find("description")

            density = "unknown"
            if name_elem is not None and name_elem.text:
                name_lower = name_elem.text.lower()
                if "light" in name_lower:
                    density = "light"
                elif "medium" in name_lower:
                    density = "medium"
                elif "heavy" in name_lower or "dense" in name_lower:
                    density = "heavy"

            # Get polygon coordinates
            for coords_elem in placemark.iter("coordinates"):
                if coords_elem.text:
                    coords = []
                    for point in coords_elem.text.strip().split():
                        parts = point.split(",")
                        if len(parts) >= 2:
                            try:
                                lon = float(parts[0])
                                lat = float(parts[1])
                                coords.append((lat, lon))
                            except ValueError:
                                continue
                    if coords:
                        polygons.append((density, coords))

    except ET.ParseError:
        pass

    return polygons


def _point_in_polygon(lat: float, lon: float, polygon: List[Tuple[float, float]]) -> bool:
    """Check if point is inside polygon using ray casting algorithm."""
    n = len(polygon)
    inside = False

    p1_lat, p1_lon = polygon[0]
    for i in range(1, n + 1):
        p2_lat, p2_lon = polygon[i % n]
        if lat > min(p1_lat, p2_lat):
            if lat <= max(p1_lat, p2_lat):
                if lon <= max(p1_lon, p2_lon):
                    if p1_lat != p2_lat:
                        lon_intersect = (lat - p1_lat) * (p2_lon - p1_lon) / (p2_lat - p1_lat) + p1_lon
                    if p1_lon == p2_lon or lon <= lon_intersect:
                        inside = not inside
        p1_lat, p1_lon = p2_lat, p2_lon

    return inside


def fetch_hms_smoke_kml(date: datetime, timeout: int = 30) -> Optional[str]:
    """
    Fetch HMS smoke KML data for a specific date.

    HMS provides smoke analysis typically 1-2 times per day.
    """
    # HMS archive URL pattern
    # https://satepsanone.nesdis.noaa.gov/pub/FIRE/web/HMS/Smoke_Polygons/Shapefile/YYYY/MM/hms_smokeYYYYMMDD.kml
    url = (
        f"https://satepsanone.nesdis.noaa.gov/pub/FIRE/web/HMS/"
        f"Smoke_Polygons/KML/{date.year}/{date.month:02d}/"
        f"hms_smoke{date.strftime('%Y%m%d')}.kml"
    )

    try:
        req = Request(url, headers={"User-Agent": "weather-ml-model/1.0"})
        with urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="ignore")
    except HTTPError as e:
        if e.code == 404:
            return None
        raise
    except (URLError, TimeoutError):
        return None


def fetch_smoke_for_location(
    lat: float,
    lon: float,
    date: str,
    timeout: int = 30,
) -> Optional[SmokeObservation]:
    """
    Fetch smoke observation for a specific location and date.

    Args:
        lat: Latitude
        lon: Longitude
        date: Date string YYYY-MM-DD
        timeout: Request timeout

    Returns:
        SmokeObservation or None if data unavailable
    """
    date_dt = datetime.strptime(date, "%Y-%m-%d")

    # Skip non-fire season for efficiency (smoke is rare outside season)
    if not is_fire_season(date_dt):
        return SmokeObservation(
            date=date,
            observation_time_utc=f"{date}T12:00:00Z",
            lat=lat,
            lon=lon,
            smoke_present=False,
            smoke_density=None,
            source="hms_offseason",
        )

    kml_data = fetch_hms_smoke_kml(date_dt, timeout)

    if kml_data is None:
        # No data available - assume no smoke
        return SmokeObservation(
            date=date,
            observation_time_utc=f"{date}T12:00:00Z",
            lat=lat,
            lon=lon,
            smoke_present=False,
            smoke_density=None,
            source="hms_no_data",
        )

    polygons = _parse_kml_smoke_polygons(kml_data)

    # Check if location is in any smoke polygon
    smoke_present = False
    smoke_density = None

    # Priority: heavy > medium > light
    density_priority = {"heavy": 3, "medium": 2, "light": 1, "unknown": 0}

    for density, coords in polygons:
        if _point_in_polygon(lat, lon, coords):
            smoke_present = True
            if smoke_density is None or density_priority.get(density, 0) > density_priority.get(smoke_density, 0):
                smoke_density = density

    return SmokeObservation(
        date=date,
        observation_time_utc=f"{date}T18:00:00Z",  # Approximate analysis time
        lat=lat,
        lon=lon,
        smoke_present=smoke_present,
        smoke_density=smoke_density,
        source="hms",
    )


def fetch_smoke_batch(
    locations: List[Tuple[str, float, float]],  # (city_key, lat, lon)
    date: str,
    timeout: int = 30,
) -> Dict[str, SmokeObservation]:
    """
    Fetch smoke observations for multiple locations on the same date.

    More efficient than calling fetch_smoke_for_location repeatedly
    since it only downloads the KML once.

    Args:
        locations: List of (city_key, lat, lon) tuples
        date: Date string YYYY-MM-DD
        timeout: Request timeout

    Returns:
        Dict mapping city_key -> SmokeObservation
    """
    date_dt = datetime.strptime(date, "%Y-%m-%d")
    results = {}

    # Skip non-fire season
    if not is_fire_season(date_dt):
        for city_key, lat, lon in locations:
            results[city_key] = SmokeObservation(
                date=date,
                observation_time_utc=f"{date}T12:00:00Z",
                lat=lat,
                lon=lon,
                smoke_present=False,
                smoke_density=None,
                source="hms_offseason",
            )
        return results

    kml_data = fetch_hms_smoke_kml(date_dt, timeout)

    if kml_data is None:
        for city_key, lat, lon in locations:
            results[city_key] = SmokeObservation(
                date=date,
                observation_time_utc=f"{date}T12:00:00Z",
                lat=lat,
                lon=lon,
                smoke_present=False,
                smoke_density=None,
                source="hms_no_data",
            )
        return results

    polygons = _parse_kml_smoke_polygons(kml_data)
    density_priority = {"heavy": 3, "medium": 2, "light": 1, "unknown": 0}

    for city_key, lat, lon in locations:
        smoke_present = False
        smoke_density = None

        for density, coords in polygons:
            if _point_in_polygon(lat, lon, coords):
                smoke_present = True
                if smoke_density is None or density_priority.get(density, 0) > density_priority.get(smoke_density, 0):
                    smoke_density = density

        results[city_key] = SmokeObservation(
            date=date,
            observation_time_utc=f"{date}T18:00:00Z",
            lat=lat,
            lon=lon,
            smoke_present=smoke_present,
            smoke_density=smoke_density,
            source="hms",
        )

    return results
