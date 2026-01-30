"""
SNODAS (SNOw Data Assimilation System) client for snow depth data.

SNODAS provides daily 1km resolution snow data across the continental US.
Data is available from 2003-present.

Source: https://noaadata.apps.nsidc.org/NOAA/G02158/

Usage:
    from weather.lib.snodas import fetch_snow_depth

    snow = fetch_snow_depth(lat=40.7128, lon=-74.0060, date="2024-01-15")
    if snow:
        print(f"Snow depth: {snow.depth_inches} inches")
"""

from __future__ import annotations

import gzip
import io
import ssl
import struct
import tarfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError

import numpy as np

# Create SSL context that doesn't verify certificates (needed for NSIDC)
_SSL_CONTEXT = ssl.create_default_context()
_SSL_CONTEXT.check_hostname = False
_SSL_CONTEXT.verify_mode = ssl.CERT_NONE


@dataclass
class SnowObservation:
    """Snow observation from SNODAS."""
    date: str  # YYYY-MM-DD
    lat: float
    lon: float
    depth_inches: float  # Snow depth in inches
    swe_inches: Optional[float]  # Snow water equivalent in inches
    source: str = "snodas"


# SNODAS grid parameters (continental US)
# SNODAS uses a 1km x 1km grid
SNODAS_NROWS = 3351
SNODAS_NCOLS = 6935
SNODAS_XLLCORNER = -124.729583333333
SNODAS_YLLCORNER = 24.9495833333333
SNODAS_CELLSIZE = 0.00833333333333333  # ~1km in degrees


def _lat_lon_to_snodas_indices(lat: float, lon: float) -> tuple[int, int]:
    """Convert lat/lon to SNODAS grid row/col indices."""
    # SNODAS grid starts from lower-left corner
    col = int((lon - SNODAS_XLLCORNER) / SNODAS_CELLSIZE)
    row = int((lat - SNODAS_YLLCORNER) / SNODAS_CELLSIZE)
    # SNODAS data is stored with top row first, so invert
    row = SNODAS_NROWS - 1 - row
    return row, col


def _build_snodas_url(date: datetime) -> str:
    """Build URL for SNODAS data for a given date."""
    # URL format: https://noaadata.apps.nsidc.org/NOAA/G02158/masked/YYYY/MM_Mon/
    # File format: SNODAS_YYYYMMDD.tar
    month_abbr = date.strftime("%b")
    return (
        f"https://noaadata.apps.nsidc.org/NOAA/G02158/masked/"
        f"{date.year}/{date.month:02d}_{month_abbr}/"
        f"SNODAS_{date.strftime('%Y%m%d')}.tar"
    )


def _extract_snow_depth_from_tar(tar_data: bytes, date: datetime) -> Optional[np.ndarray]:
    """Extract snow depth grid from SNODAS tar archive.

    SNODAS tar contains multiple .dat.gz files for different variables.
    Snow depth file pattern: us_ssmv11036tS__T0001TTNATSYYYYMMDD05HP001.dat.gz
    """
    try:
        with tarfile.open(fileobj=io.BytesIO(tar_data), mode="r:") as tar:
            # Look for snow depth file (code 1036)
            for member in tar.getnames():
                # Snow depth file has '1036' in the name
                if "1036" in member and member.endswith(".dat.gz"):
                    f = tar.extractfile(member)
                    if f is None:
                        continue

                    # Decompress and read binary data
                    with gzip.GzipFile(fileobj=io.BytesIO(f.read())) as gz:
                        data = gz.read()

                    # SNODAS uses big-endian 16-bit signed integers
                    # Scale factor is 1 (units are meters * 1000)
                    arr = np.frombuffer(data, dtype=">i2").reshape(SNODAS_NROWS, SNODAS_NCOLS)

                    # Convert from meters * 1000 to inches
                    # -9999 is the no-data value
                    arr = arr.astype(float)
                    arr[arr == -9999] = np.nan
                    arr = arr / 1000.0 * 39.3701  # meters to inches

                    return arr
    except Exception as e:
        print(f"[snodas] Error extracting snow depth: {e}")

    return None


def _extract_swe_from_tar(tar_data: bytes, date: datetime) -> Optional[np.ndarray]:
    """Extract SWE grid from SNODAS tar archive.

    SWE file pattern: us_ssmv11034tS__T0001TTNATSYYYYMMDD05HP001.dat.gz (code 1034)
    """
    try:
        with tarfile.open(fileobj=io.BytesIO(tar_data), mode="r:") as tar:
            for member in tar.getnames():
                # SWE file has '1034' in the name
                if "1034" in member and member.endswith(".dat.gz"):
                    f = tar.extractfile(member)
                    if f is None:
                        continue

                    with gzip.GzipFile(fileobj=io.BytesIO(f.read())) as gz:
                        data = gz.read()

                    arr = np.frombuffer(data, dtype=">i2").reshape(SNODAS_NROWS, SNODAS_NCOLS)
                    arr = arr.astype(float)
                    arr[arr == -9999] = np.nan
                    # SWE is in meters * 1000, convert to inches
                    arr = arr / 1000.0 * 39.3701

                    return arr
    except Exception as e:
        print(f"[snodas] Error extracting SWE: {e}")

    return None


def fetch_snow_depth(
    lat: float,
    lon: float,
    date: str,
    timeout: int = 30,
) -> Optional[SnowObservation]:
    """
    Fetch snow depth from SNODAS for a specific location and date.

    Args:
        lat: Latitude
        lon: Longitude
        date: Date string YYYY-MM-DD
        timeout: Request timeout in seconds

    Returns:
        SnowObservation or None if data unavailable
    """
    date_dt = datetime.strptime(date, "%Y-%m-%d")

    # Check if location is within SNODAS coverage
    if not (24.0 <= lat <= 58.0 and -125.0 <= lon <= -66.0):
        # Outside continental US - no snow data available
        return SnowObservation(
            date=date,
            lat=lat,
            lon=lon,
            depth_inches=0.0,
            swe_inches=0.0,
            source="snodas_outside_coverage",
        )

    url = _build_snodas_url(date_dt)

    try:
        req = Request(url, headers={"User-Agent": "weather-ml-model/1.0"})
        with urlopen(req, timeout=timeout, context=_SSL_CONTEXT) as resp:
            tar_data = resp.read()
    except HTTPError as e:
        if e.code == 404:
            # Data not available for this date
            return None
        raise
    except (URLError, TimeoutError):
        return None

    # Extract grids
    row, col = _lat_lon_to_snodas_indices(lat, lon)

    # Bounds check
    if not (0 <= row < SNODAS_NROWS and 0 <= col < SNODAS_NCOLS):
        return None

    depth_grid = _extract_snow_depth_from_tar(tar_data, date_dt)
    swe_grid = _extract_swe_from_tar(tar_data, date_dt)

    if depth_grid is None:
        return None

    depth_value = depth_grid[row, col]
    swe_value = swe_grid[row, col] if swe_grid is not None else None

    # Handle NaN (no data)
    if np.isnan(depth_value):
        depth_value = 0.0
    if swe_value is not None and np.isnan(swe_value):
        swe_value = 0.0

    return SnowObservation(
        date=date,
        lat=lat,
        lon=lon,
        depth_inches=max(0.0, depth_value),
        swe_inches=max(0.0, swe_value) if swe_value is not None else None,
        source="snodas",
    )


def fetch_snow_depth_cached(
    lat: float,
    lon: float,
    date: str,
    cache_dir: Optional[Path] = None,
    timeout: int = 30,
) -> Optional[SnowObservation]:
    """
    Fetch snow depth with local caching of SNODAS tar files.

    This is more efficient for backfilling multiple cities on the same date.

    Args:
        lat: Latitude
        lon: Longitude
        date: Date string YYYY-MM-DD
        cache_dir: Directory to cache downloaded tar files
        timeout: Request timeout

    Returns:
        SnowObservation or None
    """
    date_dt = datetime.strptime(date, "%Y-%m-%d")

    # Check coverage
    if not (24.0 <= lat <= 58.0 and -125.0 <= lon <= -66.0):
        return SnowObservation(
            date=date,
            lat=lat,
            lon=lon,
            depth_inches=0.0,
            swe_inches=0.0,
            source="snodas_outside_coverage",
        )

    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "snodas"
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_file = cache_dir / f"SNODAS_{date_dt.strftime('%Y%m%d')}.tar"

    # Check cache first
    if cache_file.exists():
        tar_data = cache_file.read_bytes()
    else:
        url = _build_snodas_url(date_dt)
        try:
            req = Request(url, headers={"User-Agent": "weather-ml-model/1.0"})
            with urlopen(req, timeout=timeout, context=_SSL_CONTEXT) as resp:
                tar_data = resp.read()
            # Save to cache
            cache_file.write_bytes(tar_data)
        except HTTPError as e:
            if e.code == 404:
                return None
            raise
        except (URLError, TimeoutError):
            return None

    row, col = _lat_lon_to_snodas_indices(lat, lon)

    if not (0 <= row < SNODAS_NROWS and 0 <= col < SNODAS_NCOLS):
        return None

    depth_grid = _extract_snow_depth_from_tar(tar_data, date_dt)
    swe_grid = _extract_swe_from_tar(tar_data, date_dt)

    if depth_grid is None:
        return None

    depth_value = depth_grid[row, col]
    swe_value = swe_grid[row, col] if swe_grid is not None else None

    if np.isnan(depth_value):
        depth_value = 0.0
    if swe_value is not None and np.isnan(swe_value):
        swe_value = 0.0

    return SnowObservation(
        date=date,
        lat=lat,
        lon=lon,
        depth_inches=max(0.0, depth_value),
        swe_inches=max(0.0, swe_value) if swe_value is not None else None,
        source="snodas",
    )
