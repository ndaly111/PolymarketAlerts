#!/usr/bin/env python3
"""
Settle weather trades using observed temperatures from the database.

This script settles trades based on actual observed temperatures
without needing Kalshi API credentials.
"""

import re
import sqlite3
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEATHER_DB = ROOT / "weather_trades.db"
ACCURACY_DB = ROOT / "weather" / "data" / "weather_forecast_accuracy.db"


def parse_ticker(ticker: str) -> tuple[str, str, float]:
    """
    Parse market ticker to extract city, date, and bracket.

    Example: KXHIGHAUS-26JAN27-B49.5 -> ("AUSHIGH", "2026-01-27", 49.5)
    """
    # Pattern: KXHIGH<CITY>-<YY><MON><DD>-B<BRACKET>
    match = re.match(r"KXHIGH([A-Z]+)-(\d{2})([A-Z]{3})(\d{2})-B(\d+\.?\d*)", ticker)
    if not match:
        # Try alternative format: KXHIGHT<CITY>-...
        match = re.match(r"KXHIGHT([A-Z]+)-(\d{2})([A-Z]{3})(\d{2})-B(\d+\.?\d*)", ticker)

    if not match:
        return None, None, None

    city_code = match.group(1)
    year = int(match.group(2))
    month_str = match.group(3).upper()
    day = int(match.group(4))
    bracket = float(match.group(5))

    # Map city codes to city keys
    city_map = {
        "AUS": "AUSHIGH",
        "CHI": "CHIHIGH",
        "DC": "DCHIGH",
        "TDC": "DCHIGH",  # Alternative format
        "DEN": "DENHIGH",
        "LAX": "LAXHIGH",
        "LV": "LVHIGH",
        "TLV": "LVHIGH",  # Alternative format
        "MIA": "MIAHIGH",
        "NY": "NHIGH",
        "NOL": "NOLHIGH",
        "NOLA": "NOLHIGH",
        "TNOLA": "NOLHIGH",  # Alternative format
        "PHIL": "PHLHIGH",
        "SEA": "SEAHIGH",
        "TSEA": "SEAHIGH",  # Alternative format
        "SFO": "SFOHIGH",
        "TSFO": "SFOHIGH",  # Alternative format
    }
    city_key = city_map.get(city_code, f"{city_code}HIGH")

    # Parse month
    months = {
        "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
        "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12
    }
    month = months.get(month_str, 1)

    # Assume 20XX century
    full_year = 2000 + year
    date_str = f"{full_year}-{month:02d}-{day:02d}"

    return city_key, date_str, bracket


def get_observed_temp(accuracy_db: Path, city_key: str, date_str: str) -> float | None:
    """Get observed temperature for a city/date."""
    if not accuracy_db.exists():
        return None

    conn = sqlite3.connect(str(accuracy_db))
    cur = conn.cursor()
    cur.execute(
        "SELECT tmax_f FROM observed_cli WHERE city_key = ? AND date_local = ?",
        (city_key, date_str)
    )
    row = cur.fetchone()
    conn.close()

    return row[0] if row else None


def is_bracket_winner(observed_temp: float, bracket: float) -> bool:
    """
    Check if YES wins the bracket.

    Bracket B49.5 means the range 49 ≤ TMAX ≤ 50.
    YES wins if observed temp falls in this 2-degree range.
    """
    lower = int(bracket)
    upper = lower + 1
    return lower <= observed_temp <= upper


def settle_trades():
    """Settle all unsettled weather trades using observed data."""
    if not WEATHER_DB.exists():
        print("Weather trades database not found")
        return

    conn = sqlite3.connect(str(WEATHER_DB))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Get unsettled trades
    cur.execute("""
        SELECT id, trade_date, city_key, market_ticker, side,
               fill_price_cents, limit_price_cents
        FROM weather_trades
        WHERE (settled = 0 OR settled IS NULL)
          AND status IN ('filled', 'FILLED', 'PLACED')
    """)
    trades = [dict(r) for r in cur.fetchall()]

    print(f"Found {len(trades)} unsettled trades")

    settled_count = 0
    wins = 0
    losses = 0
    total_pnl = 0

    for trade in trades:
        ticker = trade["market_ticker"]
        side = trade["side"].upper()
        cost = trade["fill_price_cents"] or trade["limit_price_cents"] or 0

        city_key, date_str, bracket = parse_ticker(ticker)
        if not city_key or not date_str:
            print(f"  Cannot parse ticker: {ticker}")
            continue

        observed = get_observed_temp(ACCURACY_DB, city_key, date_str)
        if observed is None:
            print(f"  No observed data for {city_key} {date_str}")
            continue

        # Check if bracket won
        bracket_won = is_bracket_winner(observed, bracket)

        # Did OUR side win?
        if side == "YES":
            won = bracket_won
        else:  # NO
            won = not bracket_won

        payout = 100 if won else 0
        pnl = payout - cost

        # Update the trade
        now = datetime.utcnow().isoformat()
        cur.execute("""
            UPDATE weather_trades
            SET settled = 1, won = ?, payout_cents = ?, settled_at_utc = ?, updated_at = ?
            WHERE id = ?
        """, (1 if won else 0, payout, now, now, trade["id"]))

        icon = "✓" if won else "✗"
        print(f"  {icon} {city_key} {date_str} B{bracket} {side} | Obs={observed}°F | P/L: {pnl:+d}¢")

        settled_count += 1
        if won:
            wins += 1
        else:
            losses += 1
        total_pnl += pnl

    conn.commit()
    conn.close()

    print(f"\nSettled: {settled_count} trades")
    print(f"Record: {wins}W-{losses}L")
    print(f"P/L: {total_pnl:+d}¢ (${total_pnl / 100:+.2f})")


if __name__ == "__main__":
    settle_trades()
