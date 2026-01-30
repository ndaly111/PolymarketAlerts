"""
HTTP session utilities with retry logic.

Consolidates session creation from props_value.py and other scripts.
"""

from __future__ import annotations

from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def make_session(
    total_retries: int = 6,
    backoff_factor: float = 1.0,
    status_forcelist: tuple = (429, 500, 502, 503, 504),
    user_agent: str = "trading-pipeline/1.0",
) -> requests.Session:
    """
    Create a requests session with retry logic.

    Args:
        total_retries: Maximum number of retries
        backoff_factor: Exponential backoff factor (seconds)
        status_forcelist: HTTP status codes to retry
        user_agent: User-Agent header value

    Returns:
        Configured requests.Session

    Example:
        >>> session = make_session()
        >>> response = session.get("https://api.example.com/data")
    """
    session = requests.Session()

    retries = Retry(
        total=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=frozenset(["GET", "POST", "PUT", "DELETE"]),
        raise_on_status=False,
    )

    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    session.headers.update({
        "Accept": "application/json",
        "User-Agent": user_agent,
    })

    return session


# Singleton session for reuse
_default_session: Optional[requests.Session] = None


def get_session() -> requests.Session:
    """
    Get the default shared session (singleton pattern).

    Returns:
        Shared requests.Session with retry logic
    """
    global _default_session
    if _default_session is None:
        _default_session = make_session()
    return _default_session


def fetch_json(
    url: str,
    params: Optional[dict] = None,
    session: Optional[requests.Session] = None,
    timeout: int = 30,
) -> dict:
    """
    Fetch JSON from URL with error handling.

    Args:
        url: URL to fetch
        params: Query parameters
        session: Optional session (uses default if not provided)
        timeout: Request timeout in seconds

    Returns:
        Parsed JSON response

    Raises:
        requests.RequestException: On HTTP errors
        ValueError: On JSON parse errors
    """
    sess = session or get_session()
    response = sess.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def post_json(
    url: str,
    data: dict,
    headers: Optional[dict] = None,
    session: Optional[requests.Session] = None,
    timeout: int = 30,
) -> dict:
    """
    POST JSON to URL with error handling.

    Args:
        url: URL to post to
        data: JSON data to send
        headers: Additional headers
        session: Optional session (uses default if not provided)
        timeout: Request timeout in seconds

    Returns:
        Parsed JSON response

    Raises:
        requests.RequestException: On HTTP errors
        ValueError: On JSON parse errors
    """
    sess = session or get_session()
    response = sess.post(url, json=data, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.json()
