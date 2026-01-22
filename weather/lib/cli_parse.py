from __future__ import annotations

import re
from datetime import datetime
from typing import Dict


_MONTHS: Dict[str, int] = {
    "JANUARY": 1,
    "FEBRUARY": 2,
    "MARCH": 3,
    "APRIL": 4,
    "MAY": 5,
    "JUNE": 6,
    "JULY": 7,
    "AUGUST": 8,
    "SEPTEMBER": 9,
    "OCTOBER": 10,
    "NOVEMBER": 11,
    "DECEMBER": 12,
}


def parse_report_date_local(text: str) -> str:
    match = re.search(r"CLIMATE SUMMARY FOR\s+([A-Z]+)\s+(\d{1,2})\s+(\d{4})", text)
    if not match:
        match = re.search(r"\.{3,}\s*THE\s+.*CLIMATE SUMMARY FOR\s+([A-Z]+)\s+(\d{1,2})\s+(\d{4})", text)
    if not match:
        raise ValueError("Could not parse report date from CLI text")
    mon_str, day_str, year_str = match.groups()
    month = _MONTHS.get(mon_str.upper())
    if not month:
        raise ValueError(f"Unknown month name: {mon_str}")
    dt = datetime(int(year_str), month, int(day_str))
    return dt.date().isoformat()


def parse_tmax_f(text: str) -> int:
    match = re.search(r"^\s*MAXIMUM\s+(-?\d+)", text, re.MULTILINE)
    if not match:
        raise ValueError("Could not parse MAXIMUM temperature from CLI text")
    return int(match.group(1))


def detect_preliminary(text: str) -> bool:
    return "PRELIMINARY" in text.upper()
