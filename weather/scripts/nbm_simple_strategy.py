#!/usr/bin/env python3
"""
Simple NBM-based trading strategy for comparison with ML model.

Strategy Rules:
1. Use NBM forecast as center point
2. Assume ±2°F variation window
3. For strikes WITHIN forecast ±2°F: Buy YES if ask < 90¢ (with 2¢ fee = 88¢ max)
4. For strikes OUTSIDE forecast ±2°F: Buy NO if ask between 25¢-85¢

This is a simple heuristic to benchmark against the ML quantile model.

Usage:
    python weather/scripts/nbm_simple_strategy.py
    python weather/scripts/nbm_simple_strategy.py --city LAXHIGH --date 2026-01-31
    python weather/scripts/nbm_simple_strategy.py --live  # Fetch live Kalshi prices
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = Path(os.getenv("WEATHER_DB_PATH", str(ROOT / "weather" / "data" / "weather_forecast_accuracy.db")))
TRADES_DB = Path(os.getenv("WEATHER_TRADES_DB_PATH", str(ROOT / "weather_trades.db")))
CONFIG_PATH = ROOT / "weather" / "config" / "cities.yml"

# Strategy parameters
VARIATION_DEGREES = 2  # ±2°F window around forecast
MAX_YES_ASK_CENTS = 88  # 90¢ - 2¢ fee
MIN_NO_ASK_CENTS = 25
MAX_NO_ASK_CENTS = 85
FEE_CENTS = 2


@dataclass
class City:
    key: str
    label: str
    tz: str
    kalshi_series: str


@dataclass
class TradeSignal:
    city_key: str
    market_ticker: str
    strike: int
    side: str  # "YES" or "NO"
    ask_cents: int
    forecast_high: int
    distance_from_forecast: int
    reason: str
    ev_estimate: float  # Rough EV estimate


def load_cities() -> List[City]:
    """Load city configurations."""
    import yaml
    data = yaml.safe_load(CONFIG_PATH.read_text())
    cities = []
    for row in data.get("cities", []):
        cities.append(City(
            key=row["key"],
            label=row.get("label", row["key"]),
            tz=row.get("tz", "America/New_York"),
            kalshi_series=row.get("kalshi_series", f"KXHIGH{row['key'][:3]}"),
        ))
    return cities


def get_nbm_forecast(db_path: Path, city_key: str, target_date: str) -> Optional[int]:
    """Get NBM forecast high for a city and date."""
    conn = sqlite3.connect(str(db_path))

    # Try NBM probabilistic forecast first (p50)
    row = conn.execute("""
        SELECT p50 FROM nbm_probabilistic_forecasts
        WHERE city_key = ? AND target_date_local = ? AND metric = 'tmax'
        ORDER BY model_run_utc DESC LIMIT 1
    """, (city_key, target_date)).fetchone()

    if row and row[0]:
        conn.close()
        return int(round(row[0]))

    # Fallback to forecast_snapshots
    row = conn.execute("""
        SELECT forecast_high_f FROM forecast_snapshots
        WHERE city_key = ? AND target_date_local = ?
        ORDER BY snapshot_time_utc DESC LIMIT 1
    """, (city_key, target_date)).fetchone()

    conn.close()
    return int(row[0]) if row else None


def get_kalshi_markets_from_db(
    db_path: Path,
    city_key: str,
    target_date: str,
    series_ticker: str,
) -> List[Dict[str, Any]]:
    """Get Kalshi market data from database."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Build event ticker pattern
    date_obj = datetime.strptime(target_date, "%Y-%m-%d")
    event_pattern = f"{series_ticker}-{date_obj.strftime('%y%b%d').upper()}"

    rows = conn.execute("""
        SELECT DISTINCT
            market_ticker,
            yes_ask,
            no_ask,
            raw_json
        FROM kalshi_weather_market_snapshots
        WHERE event_ticker = ? AND status = 'active'
        ORDER BY snapshot_time_utc DESC
    """, (event_pattern,)).fetchall()

    conn.close()

    # Dedupe by market ticker (keep latest)
    markets = {}
    for row in rows:
        ticker = row["market_ticker"]
        if ticker not in markets:
            market = dict(row)
            try:
                raw = json.loads(row["raw_json"])
                market["floor_strike"] = raw.get("floor_strike")
            except:
                market["floor_strike"] = None
            markets[ticker] = market

    return list(markets.values())


def fetch_live_kalshi_markets(city_key: str, series_ticker: str) -> List[Dict[str, Any]]:
    """Fetch live Kalshi markets via API."""
    try:
        from kalshi_auth_client import KalshiAuthClient
        client = KalshiAuthClient.from_env()

        # Get markets for series
        resp = client.get(f"/markets?series_ticker={series_ticker}&status=active&limit=100")
        markets = resp.get("markets", [])

        result = []
        for m in markets:
            result.append({
                "market_ticker": m.get("ticker"),
                "yes_ask": m.get("yes_ask"),
                "no_ask": m.get("no_ask"),
                "floor_strike": m.get("floor_strike"),
            })
        return result
    except Exception as e:
        print(f"  [error] Failed to fetch live prices: {e}")
        return []


def generate_signals(
    forecast_high: int,
    markets: List[Dict[str, Any]],
    city_key: str,
) -> List[TradeSignal]:
    """Generate trade signals based on NBM strategy."""
    signals = []

    low_bound = forecast_high - VARIATION_DEGREES
    high_bound = forecast_high + VARIATION_DEGREES

    for market in markets:
        strike = market.get("floor_strike")
        if not strike:
            continue

        yes_ask = market.get("yes_ask") or 0
        no_ask = market.get("no_ask") or 0
        ticker = market.get("market_ticker", "")

        distance = abs(strike - forecast_high)

        # Rule 1: Within ±2°F of forecast -> Buy YES if cheap enough
        if low_bound <= strike <= high_bound:
            if 0 < yes_ask <= MAX_YES_ASK_CENTS:
                # Estimate EV: assume ~60% win rate for "in range" bets
                # EV = 0.60 * (100 - yes_ask) - 0.40 * yes_ask - FEE
                win_prob = 0.60
                ev = win_prob * (100 - yes_ask) - (1 - win_prob) * yes_ask - FEE_CENTS

                signals.append(TradeSignal(
                    city_key=city_key,
                    market_ticker=ticker,
                    strike=strike,
                    side="YES",
                    ask_cents=yes_ask,
                    forecast_high=forecast_high,
                    distance_from_forecast=distance,
                    reason=f"Within ±{VARIATION_DEGREES}°F of {forecast_high}°F forecast",
                    ev_estimate=ev / 100.0,
                ))

        # Rule 2: Outside ±2°F -> Buy NO if in price range
        else:
            if MIN_NO_ASK_CENTS <= no_ask <= MAX_NO_ASK_CENTS:
                # Estimate EV: assume ~70% win rate for "out of range" NO bets
                win_prob = 0.70
                ev = win_prob * (100 - no_ask) - (1 - win_prob) * no_ask - FEE_CENTS

                signals.append(TradeSignal(
                    city_key=city_key,
                    market_ticker=ticker,
                    strike=strike,
                    side="NO",
                    ask_cents=no_ask,
                    forecast_high=forecast_high,
                    distance_from_forecast=distance,
                    reason=f"Outside ±{VARIATION_DEGREES}°F of {forecast_high}°F forecast",
                    ev_estimate=ev / 100.0,
                ))

    return signals


def ensure_trades_table(db_path: Path) -> None:
    """Ensure the nbm_strategy_trades table exists."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS nbm_strategy_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_date TEXT NOT NULL,
            city_key TEXT NOT NULL,
            market_ticker TEXT NOT NULL,
            strike INTEGER NOT NULL,
            side TEXT NOT NULL,
            ask_cents INTEGER NOT NULL,
            forecast_high INTEGER NOT NULL,
            distance_from_forecast INTEGER NOT NULL,
            reason TEXT,
            ev_estimate REAL,
            status TEXT DEFAULT 'PAPER',
            created_at TEXT NOT NULL,
            settled INTEGER DEFAULT 0,
            won INTEGER,
            payout_cents INTEGER,
            settled_at TEXT,
            UNIQUE(trade_date, market_ticker)
        )
    """)
    conn.commit()
    conn.close()


def log_signal(db_path: Path, signal: TradeSignal, target_date: str) -> None:
    """Log a trade signal to the database."""
    conn = sqlite3.connect(str(db_path))
    now = datetime.now(ZoneInfo("UTC")).isoformat()

    try:
        conn.execute("""
            INSERT OR REPLACE INTO nbm_strategy_trades
            (trade_date, city_key, market_ticker, strike, side, ask_cents,
             forecast_high, distance_from_forecast, reason, ev_estimate, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'PAPER', ?)
        """, (
            target_date,
            signal.city_key,
            signal.market_ticker,
            signal.strike,
            signal.side,
            signal.ask_cents,
            signal.forecast_high,
            signal.distance_from_forecast,
            signal.reason,
            signal.ev_estimate,
            now,
        ))
        conn.commit()
    except Exception as e:
        print(f"  [warn] Failed to log signal: {e}")
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="NBM Simple Strategy")
    parser.add_argument("--city", default="", help="Specific city (default: all)")
    parser.add_argument("--date", default="", help="Target date YYYY-MM-DD (default: today)")
    parser.add_argument("--live", action="store_true", help="Fetch live Kalshi prices")
    parser.add_argument("--dry-run", action="store_true", help="Don't log to database")
    args = parser.parse_args()

    # Load cities
    import yaml
    cities = load_cities()
    if args.city:
        cities = [c for c in cities if c.key == args.city.upper()]
        if not cities:
            print(f"City not found: {args.city}")
            return 1

    # Determine target date
    if args.date:
        target_date = args.date
    else:
        # Default to today in ET
        target_date = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")

    print("=" * 70)
    print("NBM SIMPLE STRATEGY")
    print(f"Target date: {target_date}")
    print(f"Rules:")
    print(f"  - Within ±{VARIATION_DEGREES}°F of forecast: Buy YES if ask ≤ {MAX_YES_ASK_CENTS}¢")
    print(f"  - Outside ±{VARIATION_DEGREES}°F: Buy NO if {MIN_NO_ASK_CENTS}¢ ≤ ask ≤ {MAX_NO_ASK_CENTS}¢")
    print("=" * 70)
    print()

    if not args.dry_run:
        ensure_trades_table(TRADES_DB)

    all_signals = []

    for city in cities:
        print(f"\n--- {city.label} ({city.key}) ---")

        # Get forecast
        forecast = get_nbm_forecast(DB_PATH, city.key, target_date)
        if not forecast:
            print(f"  No forecast available")
            continue

        print(f"  Forecast: {forecast}°F")
        print(f"  YES zone: {forecast - VARIATION_DEGREES}°F to {forecast + VARIATION_DEGREES}°F")

        # Get markets
        if args.live:
            markets = fetch_live_kalshi_markets(city.key, city.kalshi_series)
        else:
            markets = get_kalshi_markets_from_db(DB_PATH, city.key, target_date, city.kalshi_series)

        if not markets:
            print(f"  No markets found")
            continue

        print(f"  Markets: {len(markets)}")

        # Generate signals
        signals = generate_signals(forecast, markets, city.key)

        if signals:
            print(f"  Signals:")
            for sig in sorted(signals, key=lambda x: x.strike):
                print(f"    {sig.side} @ {sig.ask_cents}¢ on {sig.strike}°F+ | EV: {sig.ev_estimate*100:+.1f}¢ | {sig.reason}")

                if not args.dry_run:
                    log_signal(TRADES_DB, sig, target_date)

            all_signals.extend(signals)
        else:
            print(f"  No signals")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    yes_signals = [s for s in all_signals if s.side == "YES"]
    no_signals = [s for s in all_signals if s.side == "NO"]

    print(f"Total signals: {len(all_signals)}")
    print(f"  YES bets: {len(yes_signals)}")
    print(f"  NO bets: {len(no_signals)}")

    if all_signals:
        total_cost = sum(s.ask_cents for s in all_signals)
        avg_ev = sum(s.ev_estimate for s in all_signals) / len(all_signals) * 100
        print(f"  Total cost: {total_cost}¢ (${total_cost/100:.2f})")
        print(f"  Avg EV estimate: {avg_ev:+.1f}¢")

    if not args.dry_run:
        print(f"\nSignals logged to: {TRADES_DB}")
        print("Table: nbm_strategy_trades")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
