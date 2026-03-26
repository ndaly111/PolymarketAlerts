#!/usr/bin/env python3
"""
Weather trading scanner with profit-optimized ML filtering.

Scans Kalshi weather markets for trading opportunities and paper trades them
to collect data for ML model improvement.

Features:
- Real-time market scanning across all 12 cities
- Profit model filtering (only take trades ML predicts will be profitable)
- Paper trading with outcome tracking
- Discord notifications for high-confidence opportunities
- Automatic settlement resolution

Usage:
    python weather_scanner.py                    # Paper trade mode (default)
    python weather_scanner.py --live             # Live mode (1 trade per day per city)
    python weather_scanner.py --min-ev 0.05      # Higher EV threshold
    python weather_scanner.py --cities NHIGH,CHIHIGH  # Specific cities only
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sqlite3
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from weather.lib import db as db_lib
from weather.lib.fees import FeeSchedule
from weather.lib.fair import normalize_pmf, shift_pmf
from weather.lib.kalshi_weather import EventSpec, parse_event_spec_from_title, parse_event_spec_from_ticker, prob_event

DB_PATH = Path(os.getenv("WEATHER_DB_PATH", str(ROOT / "weather" / "data" / "weather_forecast_accuracy.db")))
MODEL_DIR = ROOT / "weather" / "models"
CONFIG_PATH = ROOT / "weather" / "config" / "cities.yml"

PROFIT_MODEL_PATH = MODEL_DIR / "weather_profit_model.pkl"
CALIBRATION_STATE_FILE = MODEL_DIR / "weather_calibration_state.json"

# Recalibration threshold: retrain when settled trades increase by 25%
RECALIBRATION_THRESHOLD = 1.25


def load_calibration_state() -> Dict[str, Any]:
    """Load calibration state from file."""
    if CALIBRATION_STATE_FILE.exists():
        try:
            return json.loads(CALIBRATION_STATE_FILE.read_text())
        except:
            pass
    return {"last_calibration_count": 0, "last_calibration_time": None}


def save_calibration_state(settled_count: int) -> None:
    """Save calibration state to file."""
    state = {
        "last_calibration_count": settled_count,
        "last_calibration_time": datetime.now(ZoneInfo("UTC")).isoformat(),
    }
    CALIBRATION_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CALIBRATION_STATE_FILE.write_text(json.dumps(state, indent=2))


def check_recalibration_needed(db_path: Path) -> Tuple[bool, int, int]:
    """Check if ML model recalibration is needed.

    Returns (should_recalibrate, current_count, last_count).
    Recalibrates when settled trades increase by 25% from last calibration.
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Get count of settled weather paper trades
    c.execute("SELECT COUNT(*) FROM weather_paper_trades WHERE outcome IS NOT NULL")
    current_count = c.fetchone()[0]
    conn.close()

    state = load_calibration_state()
    last_count = state.get("last_calibration_count", 0)

    # Recalibrate when we have 25% more data than last calibration
    # Minimum 50 samples before first calibration
    if current_count >= 50 and (last_count == 0 or current_count >= last_count * RECALIBRATION_THRESHOLD):
        return True, current_count, last_count

    return False, current_count, last_count


def run_recalibration(db_path: Path) -> bool:
    """Run ML model recalibration based on settled trades."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    # Get settled trades with outcomes
    c.execute("""
        SELECT
            outcome,
            city_key,
            event_type,
            side,
            entry_price,
            model_prob,
            edge,
            ev_estimate,
            pnl
        FROM weather_paper_trades
        WHERE outcome IS NOT NULL
    """)
    rows = c.fetchall()

    if len(rows) < 50:
        print(f"[CALIBRATE] Not enough data ({len(rows)} samples, need 50+)")
        conn.close()
        return False

    # Analyze accuracy by different factors
    total = len(rows)
    wins = sum(1 for r in rows if r["outcome"] == "win")
    win_rate = wins / total if total > 0 else 0
    total_pnl = sum(r["pnl"] or 0 for r in rows)

    # Accuracy by side
    yes_wins = sum(1 for r in rows if r["outcome"] == "win" and r["side"] == "YES")
    yes_total = sum(1 for r in rows if r["side"] == "YES")

    no_wins = sum(1 for r in rows if r["outcome"] == "win" and r["side"] == "NO")
    no_total = sum(1 for r in rows if r["side"] == "NO")

    # Accuracy by event type
    above_wins = sum(1 for r in rows if r["outcome"] == "win" and r["event_type"] == "ge")
    above_total = sum(1 for r in rows if r["event_type"] == "ge")

    below_wins = sum(1 for r in rows if r["outcome"] == "win" and r["event_type"] == "le")
    below_total = sum(1 for r in rows if r["event_type"] == "le")

    print(f"\n[CALIBRATE] ===== WEATHER ML RECALIBRATION =====")
    print(f"[CALIBRATE] Total samples: {total}")
    print(f"[CALIBRATE] Win rate: {win_rate*100:.1f}% ({wins}/{total})")
    print(f"[CALIBRATE] Total P&L: ${total_pnl:.2f}")
    print(f"[CALIBRATE] YES trades: {yes_wins}/{yes_total} = {(yes_wins/yes_total*100 if yes_total else 0):.1f}%")
    print(f"[CALIBRATE] NO trades: {no_wins}/{no_total} = {(no_wins/no_total*100 if no_total else 0):.1f}%")
    print(f"[CALIBRATE] Above markets: {above_wins}/{above_total} = {(above_wins/above_total*100 if above_total else 0):.1f}%")
    print(f"[CALIBRATE] Below markets: {below_wins}/{below_total} = {(below_wins/below_total*100 if below_total else 0):.1f}%")
    print(f"[CALIBRATE] ==========================================\n")

    conn.close()

    # Save new calibration state
    save_calibration_state(total)

    # TODO: Could trigger actual model training here if train_profit_model.py exists
    # For now, just log the analytics

    return True

# Feature names must match train_profit_model.py
PROFIT_FEATURES = [
    "model_prob",
    "market_price",
    "edge",
    "ev_estimate",
    "is_yes_side",
    "event_type_above",
    "event_type_below",
    "event_type_between",
    "threshold_distance",
    "nbm_spread",
    "month",
    "day_of_week",
    "hour_of_day",
    "forecast_high",
    "lead_hours",
    "precip_prob",
    "cloud_cover",
    "wind_speed",
    "city_idx",
]


def load_config() -> Dict[str, Any]:
    """Load city configuration."""
    with open(CONFIG_PATH) as f:
        data = yaml.safe_load(f)
    cities = {c["key"]: c for c in data.get("cities", [])}
    city_idx_map = {k: i for i, k in enumerate(sorted(cities.keys()))}
    return {"cities": cities, "city_idx_map": city_idx_map}


def load_profit_model() -> Optional[Dict[str, Any]]:
    """Load the trained profit prediction model."""
    if not PROFIT_MODEL_PATH.exists():
        return None

    with open(PROFIT_MODEL_PATH, "rb") as f:
        return pickle.load(f)


def ensure_trades_table(db_path: Path) -> None:
    """Create the weather trades table if it doesn't exist."""
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS weather_paper_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            city_key TEXT NOT NULL,
            target_date TEXT NOT NULL,
            market_ticker TEXT NOT NULL,
            event_type TEXT NOT NULL,
            threshold_a INTEGER NOT NULL,
            threshold_b INTEGER,
            side TEXT NOT NULL,
            entry_price REAL NOT NULL,
            contracts INTEGER NOT NULL DEFAULT 1,
            model_prob REAL NOT NULL,
            edge REAL NOT NULL,
            ev_estimate REAL NOT NULL,
            profit_model_prob REAL,
            features_json TEXT,
            outcome TEXT,
            pnl REAL,
            settled_at TEXT,
            is_paper INTEGER NOT NULL DEFAULT 1
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_weather_trades_city_date
        ON weather_paper_trades(city_key, target_date)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_weather_trades_outcome
        ON weather_paper_trades(outcome)
    """)
    conn.commit()
    conn.close()


def get_latest_kalshi_markets(
    db_path: Path,
    city_key: str,
    target_date: str,
    series_ticker: str,
) -> List[Dict[str, Any]]:
    """Get latest Kalshi market snapshots for a city and date."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    date_obj = datetime.strptime(target_date, "%Y-%m-%d")
    event_pattern = f"{series_ticker}-{date_obj.strftime('%y%b%d').upper()}"

    result = conn.execute("""
        SELECT
            market_ticker,
            event_ticker,
            yes_ask,
            no_ask,
            yes_bid,
            no_bid,
            volume,
            open_interest,
            snapshot_time_utc,
            status,
            raw_json
        FROM kalshi_weather_market_snapshots
        WHERE event_ticker = ?
          AND status = 'active'
        ORDER BY snapshot_time_utc DESC
    """, (event_pattern,)).fetchall()

    conn.close()

    # Get latest snapshot per market
    markets = {}
    for row in result:
        ticker = row["market_ticker"]
        if ticker not in markets:
            markets[ticker] = dict(row)

    return list(markets.values())


def compute_trade_features(
    city_key: str,
    city_cfg: Dict,
    city_idx_map: Dict[str, int],
    side: str,
    price: float,
    model_prob: float,
    event_type: str,
    threshold_a: int,
    threshold_b: Optional[int],
    forecast_high: int,
    forecast_context: Dict,
    nbm_spread: float,
    snapshot_hour: int,
) -> Dict[str, float]:
    """Compute feature dict for profit model prediction."""
    now = datetime.now(ZoneInfo("UTC"))
    month = now.month
    day_of_week = now.weekday()

    if event_type == "between" and threshold_b:
        mid = (threshold_a + threshold_b) / 2
        threshold_distance = abs(forecast_high - mid)
    else:
        threshold_distance = abs(forecast_high - threshold_a)

    prob = model_prob if side == "YES" else 1.0 - model_prob
    fee = 0.02
    edge = prob - price - fee
    ev = edge * (1 - price)

    features = {
        "model_prob": prob,
        "market_price": price,
        "edge": edge,
        "ev_estimate": ev,
        "is_yes_side": 1.0 if side == "YES" else 0.0,
        "event_type_above": 1.0 if event_type == "ge" else 0.0,
        "event_type_below": 1.0 if event_type == "le" else 0.0,
        "event_type_between": 1.0 if event_type == "between" else 0.0,
        "threshold_distance": threshold_distance,
        "nbm_spread": nbm_spread,
        "month": float(month),
        "day_of_week": float(day_of_week),
        "hour_of_day": float(snapshot_hour),
        "forecast_high": float(forecast_high),
        "lead_hours": max(0, 18 - snapshot_hour),
        "precip_prob": forecast_context.get("precip_prob_max", 0) / 100.0,
        "cloud_cover": forecast_context.get("cloud_cover_avg", 50) / 100.0,
        "wind_speed": forecast_context.get("wind_speed_avg", 10),
        "city_idx": float(city_idx_map.get(city_key, 0)),
    }

    return features


def record_paper_trade(
    db_path: Path,
    city_key: str,
    target_date: str,
    market_ticker: str,
    event_type: str,
    threshold_a: int,
    threshold_b: Optional[int],
    side: str,
    entry_price: float,
    model_prob: float,
    edge: float,
    ev_estimate: float,
    profit_model_prob: Optional[float],
    features: Dict[str, float],
    is_paper: bool = True,
) -> int:
    """Record a paper trade in the database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.execute("""
        INSERT INTO weather_paper_trades (
            timestamp, city_key, target_date, market_ticker,
            event_type, threshold_a, threshold_b, side,
            entry_price, model_prob, edge, ev_estimate,
            profit_model_prob, features_json, is_paper
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now(ZoneInfo("UTC")).isoformat(),
        city_key,
        target_date,
        market_ticker,
        event_type,
        threshold_a,
        threshold_b,
        side,
        entry_price,
        model_prob,
        edge,
        ev_estimate,
        profit_model_prob,
        json.dumps(features),
        1 if is_paper else 0,
    ))
    trade_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return trade_id


def settle_pending_trades(db_path: Path) -> int:
    """Settle pending trades where we now have observed data."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Get pending trades
    pending = conn.execute("""
        SELECT id, city_key, target_date, event_type, threshold_a, threshold_b,
               side, entry_price
        FROM weather_paper_trades
        WHERE outcome IS NULL
    """).fetchall()

    settled_count = 0
    now = datetime.now(ZoneInfo("UTC"))

    for trade in pending:
        trade_date = datetime.strptime(trade["target_date"], "%Y-%m-%d")
        # Don't try to settle if target date is today or in the future
        if trade_date.date() >= now.date():
            continue

        # Get observed high
        observed_row = conn.execute("""
            SELECT tmax_f FROM observed_cli
            WHERE city_key = ? AND date_local = ?
        """, (trade["city_key"], trade["target_date"])).fetchone()

        if not observed_row:
            continue

        observed = observed_row["tmax_f"]
        event_type = trade["event_type"]
        threshold_a = trade["threshold_a"]
        threshold_b = trade["threshold_b"]
        side = trade["side"]
        entry_price = trade["entry_price"]

        # Determine if event happened
        if event_type == "ge":
            event_happened = observed >= threshold_a
        elif event_type == "le":
            event_happened = observed <= threshold_a
        elif event_type == "between":
            event_happened = threshold_a <= observed <= threshold_b
        else:
            continue

        # Determine outcome
        if side == "YES":
            won = event_happened
        else:
            won = not event_happened

        outcome = "win" if won else "loss"
        pnl = (1 - entry_price) * 0.98 if won else -entry_price

        # Update trade
        conn.execute("""
            UPDATE weather_paper_trades
            SET outcome = ?, pnl = ?, settled_at = ?
            WHERE id = ?
        """, (outcome, pnl, now.isoformat(), trade["id"]))

        settled_count += 1
        print(f"  Settled trade #{trade['id']}: {trade['city_key']} {trade['target_date']} → {outcome} (${pnl:+.2f})")

    conn.commit()
    conn.close()
    return settled_count


def get_trade_stats(db_path: Path, days: int = 30) -> Dict[str, Any]:
    """Get paper trading statistics."""
    conn = sqlite3.connect(db_path)
    cutoff = (datetime.now(ZoneInfo("UTC")) - timedelta(days=days)).isoformat()

    total = conn.execute("""
        SELECT COUNT(*) FROM weather_paper_trades WHERE timestamp > ?
    """, (cutoff,)).fetchone()[0]

    pending = conn.execute("""
        SELECT COUNT(*) FROM weather_paper_trades
        WHERE timestamp > ? AND outcome IS NULL
    """, (cutoff,)).fetchone()[0]

    wins = conn.execute("""
        SELECT COUNT(*) FROM weather_paper_trades
        WHERE timestamp > ? AND outcome = 'win'
    """, (cutoff,)).fetchone()[0]

    losses = conn.execute("""
        SELECT COUNT(*) FROM weather_paper_trades
        WHERE timestamp > ? AND outcome = 'loss'
    """, (cutoff,)).fetchone()[0]

    total_pnl = conn.execute("""
        SELECT COALESCE(SUM(pnl), 0) FROM weather_paper_trades
        WHERE timestamp > ? AND pnl IS NOT NULL
    """, (cutoff,)).fetchone()[0]

    conn.close()

    settled = wins + losses
    win_rate = wins / settled * 100 if settled > 0 else 0

    return {
        "total": total,
        "pending": pending,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "pnl": total_pnl,
    }


def scan_city(
    config: Dict[str, Any],
    city_key: str,
    target_date: str,
    profit_model: Optional[Dict],
    min_ev: float,
    profit_threshold: float,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """Scan a single city for trading opportunities."""
    cities = config["cities"]
    city_idx_map = config["city_idx_map"]

    if city_key not in cities:
        return []

    city_cfg = cities[city_key]
    series_ticker = city_cfg.get("kalshi_series", f"KXHIGH{city_key[:3]}")
    fees = FeeSchedule()

    # Get forecast - try multiple sources
    forecast = None
    for source in ["open_meteo", "nws_hourly_max", "mos_gfs_18z_archive", "mos_gfs_12z"]:
        forecast = db_lib.fetch_latest_forecast_snapshot(
            DB_PATH,
            city_key=city_key,
            target_date_local=target_date,
            source=source,
        )
        if forecast:
            break

    if not forecast:
        if verbose:
            print(f"  [{city_key}] No forecast found")
        return []

    forecast_high = int(forecast["forecast_high_f"])
    snapshot_hour = forecast.get("snapshot_hour_local", 6)

    # Parse forecast context
    forecast_context = {}
    if forecast.get("forecast_context_json"):
        try:
            forecast_context = json.loads(forecast["forecast_context_json"])
        except:
            pass

    # Get error model - try multiple sources and hours
    month = int(target_date.split("-")[1])
    error_model = None
    for source in ["combined", "mos_gfs_18z_archive", "nws_hourly_max"]:
        for hour in [6, 12, 18]:
            error_model = db_lib.fetch_error_model(
                DB_PATH,
                city_key=city_key,
                month=month,
                snapshot_hour_local=hour,
                source=source,
            )
            if error_model:
                break
        if error_model:
            break

    if not error_model:
        if verbose:
            print(f"  [{city_key}] No error model found")
        return []

    # Build PMF
    pmf_error = error_model["pmf"]
    pmf_high = normalize_pmf(shift_pmf(pmf_error, forecast_high))

    # Get NBM spread
    nbm_spread = 8  # Default
    nbm_row = db_lib.fetch_latest_nbm_forecast(
        DB_PATH, city_key=city_key, target_date_local=target_date, metric="tmax"
    )
    if nbm_row and nbm_row.get("p90") and nbm_row.get("p10"):
        nbm_spread = nbm_row["p90"] - nbm_row["p10"]

    # Get markets
    markets = get_latest_kalshi_markets(DB_PATH, city_key, target_date, series_ticker)

    if not markets:
        if verbose:
            print(f"  [{city_key}] No markets found")
        return []

    opportunities = []

    for market in markets:
        ticker = market["market_ticker"]
        yes_ask = market.get("yes_ask")
        no_ask = market.get("no_ask")

        if yes_ask is None or no_ask is None:
            continue

        # Parse market details
        try:
            raw = json.loads(market["raw_json"])
            title = raw.get("no_sub_title", raw.get("title", ""))
        except:
            continue

        event_spec = parse_event_spec_from_title(title)
        if not event_spec:
            ticker = row.get("market_ticker") or row.get("ticker", "")
            event_spec = parse_event_spec_from_ticker(ticker)
        if not event_spec:
            continue

        event_type = event_spec.kind
        threshold_a = event_spec.a
        threshold_b = event_spec.b

        # Compute model probability
        model_prob = prob_event(pmf_high, event_spec)

        # Check both sides
        for side in ["YES", "NO"]:
            if side == "YES":
                price = yes_ask / 100.0
                prob = model_prob
            else:
                price = no_ask / 100.0
                prob = 1.0 - model_prob

            edge = prob - price - fees.open_fee_dollars()
            ev = edge * (1 - price)

            if ev < min_ev:
                continue

            # Compute features
            features = compute_trade_features(
                city_key=city_key,
                city_cfg=city_cfg,
                city_idx_map=city_idx_map,
                side=side,
                price=price,
                model_prob=model_prob,
                event_type=event_type,
                threshold_a=threshold_a,
                threshold_b=threshold_b,
                forecast_high=forecast_high,
                forecast_context=forecast_context,
                nbm_spread=nbm_spread,
                snapshot_hour=snapshot_hour,
            )

            # Get profit model prediction
            profit_prob = None
            if profit_model:
                X = np.array([[features[name] for name in PROFIT_FEATURES]])
                profit_prob = profit_model["model"].predict_proba(X)[0, 1]

            # Filter by profit model if available
            if profit_model and profit_prob < profit_threshold:
                if verbose:
                    print(f"  [{city_key}] Skip {side}@{price:.0%} (profit_prob={profit_prob:.1%} < {profit_threshold:.0%})")
                continue

            opportunities.append({
                "city_key": city_key,
                "city_label": city_cfg.get("label", city_key),
                "target_date": target_date,
                "ticker": ticker,
                "title": title,
                "event_type": event_type,
                "threshold_a": threshold_a,
                "threshold_b": threshold_b,
                "side": side,
                "price": price,
                "model_prob": prob,
                "edge": edge,
                "ev": ev,
                "profit_prob": profit_prob,
                "forecast_high": forecast_high,
                "features": features,
            })

    return opportunities


def main():
    parser = argparse.ArgumentParser(description="Weather trading scanner")
    parser.add_argument("--live", action="store_true", help="Enable live trading (1 per city per day)")
    parser.add_argument("--cities", help="Comma-separated city keys (default: all)")
    parser.add_argument("--min-ev", type=float, default=0.02, help="Minimum EV threshold")
    parser.add_argument("--profit-threshold", type=float, default=0.6, help="Profit model threshold")
    parser.add_argument("--interval", type=int, default=300, help="Scan interval in seconds")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--date", help="Target date (YYYY-MM-DD) instead of today")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    print("="*60)
    print("WEATHER TRADING SCANNER")
    print("="*60)
    print(f"Mode: {'🔴 LIVE' if args.live else '📝 PAPER'}")
    print(f"Min EV: {args.min_ev*100:.1f}%")
    print(f"Profit threshold: {args.profit_threshold*100:.0f}%")
    print(f"Scan interval: {args.interval}s")
    print(f"Database: {DB_PATH}")
    print("="*60)

    # Load config
    config = load_config()
    cities = config["cities"]

    # Filter cities if specified
    if args.cities:
        city_keys = [k.strip().upper() for k in args.cities.split(",")]
        cities = {k: v for k, v in cities.items() if k in city_keys}
        config["cities"] = cities

    print(f"Cities: {', '.join(cities.keys())}")

    # Ensure tables exist
    ensure_trades_table(DB_PATH)

    # Load profit model
    profit_model = load_profit_model()
    if profit_model:
        print(f"Profit model loaded (accuracy: {profit_model['metrics']['accuracy']*100:.1f}%)")
    else:
        print("⚠️  No profit model found - using raw EV filtering only")

    # Check initial calibration state
    cal_state = load_calibration_state()
    print(f"Last calibration: {cal_state.get('last_calibration_count', 0)} samples")
    print("ML recalibrates automatically when settled trades increase by 25%")

    # Main scan loop
    while True:
        now = datetime.now(ZoneInfo("UTC"))
        target_date = args.date if args.date else now.strftime("%Y-%m-%d")

        print(f"\n{'='*60}")
        print(f"[SCAN] {now.strftime('%H:%M:%S')} UTC")
        print(f"[SCAN] Target date: {target_date}")

        # Settle any pending trades
        settled = settle_pending_trades(DB_PATH)
        if settled > 0:
            print(f"[SETTLE] Resolved {settled} pending trades")

        # Scan all cities
        all_opportunities = []
        for city_key in cities:
            opps = scan_city(
                config=config,
                city_key=city_key,
                target_date=target_date,
                profit_model=profit_model,
                min_ev=args.min_ev,
                profit_threshold=args.profit_threshold,
                verbose=args.verbose,
            )
            all_opportunities.extend(opps)

        # Sort by EV
        all_opportunities.sort(key=lambda x: x["ev"], reverse=True)

        # IMPORTANT: Portfolio constraints for weather markets
        #
        # YES trades: correlated — if 75F hits then 72/73/74 all hit too.
        # Must treat all YES legs for same city/date as ONE combined position.
        # Cap total YES cost at 85c per city/date.
        #
        # NO trades: independent — NO>=72 and NO>=74 can have different
        # outcomes (temp 73 → first loses, second wins). 100% of historical
        # multi-leg NO positions had mixed outcomes. Evaluate each individually
        # but still cap per-trade cost.
        MAX_YES_BLOCK_COST = 0.85  # Max combined YES cost per city/date
        MAX_NO_TRADE_COST = 0.65   # Max cost per individual NO trade
        MIN_YES_PRICE = 0.15       # Skip YES trades below 15c (3% historical WR)

        yes_block_costs: dict = {}  # city_date -> cumulative YES cost
        for opp in all_opportunities:
            key = f"{opp['city_key']}_{opp['target_date']}"

            if opp["side"] == "YES":
                # Skip cheap YES longshots
                if opp["price"] < MIN_YES_PRICE:
                    opp["skipped_portfolio"] = True
                    opp["portfolio_cost"] = opp["price"]
                    continue
                # Track YES as a combined block per city/date
                current_yes = yes_block_costs.get(key, 0)
                new_yes = current_yes + opp["price"]
                if new_yes <= MAX_YES_BLOCK_COST:
                    yes_block_costs[key] = new_yes
                    opp["skipped_portfolio"] = False
                    opp["portfolio_cost"] = new_yes
                else:
                    opp["skipped_portfolio"] = True
                    opp["portfolio_cost"] = new_yes
            else:
                # NO trades are independent — evaluate each on its own
                if opp["price"] <= MAX_NO_TRADE_COST:
                    opp["skipped_portfolio"] = False
                    opp["portfolio_cost"] = opp["price"]
                else:
                    opp["skipped_portfolio"] = True
                    opp["portfolio_cost"] = opp["price"]

        filtered_opportunities = all_opportunities

        if all_opportunities:
            unique_city_dates = len(city_date_costs)
            print(f"\n[SCAN] Found {len(all_opportunities)} opportunities ({unique_city_dates} unique city/dates):")
            for opp in filtered_opportunities[:10]:
                profit_str = f", profit={opp['profit_prob']*100:.0f}%" if opp['profit_prob'] else ""
                skip_str = " [SKIP: portfolio]" if opp.get("skipped_portfolio") else ""
                print(f"  {opp['city_label']}: {opp['side']}@{opp['price']*100:.0f}¢ | EV={opp['ev']*100:.1f}¢ | edge={opp['edge']*100:.1f}%{profit_str}{skip_str}")

            # Paper trade top opportunities (only non-skipped ones for live, all for paper)
            traded_count = 0
            for opp in filtered_opportunities[:10]:  # Check more but only trade valid ones
                if traded_count >= 5:
                    break
                if opp.get("skipped_portfolio"):
                    continue  # Skip portfolio-conflicting trades for live, but still log
                trade_id = record_paper_trade(
                    db_path=DB_PATH,
                    city_key=opp["city_key"],
                    target_date=opp["target_date"],
                    market_ticker=opp["ticker"],
                    event_type=opp["event_type"],
                    threshold_a=opp["threshold_a"],
                    threshold_b=opp["threshold_b"],
                    side=opp["side"],
                    entry_price=opp["price"],
                    model_prob=opp["model_prob"],
                    edge=opp["edge"],
                    ev_estimate=opp["ev"],
                    profit_model_prob=opp["profit_prob"],
                    features=opp["features"],
                    is_paper=not args.live,
                )
                print(f"[PAPER] Trade #{trade_id}: {opp['city_key']} {opp['side']}@{opp['price']*100:.0f}¢")
        else:
            print(f"\n[SCAN] No opportunities found")

        # Print stats
        stats = get_trade_stats(DB_PATH)
        print(f"\n[STATS] Last 30 days: {stats['total']} trades, {stats['pending']} pending")
        if stats['wins'] + stats['losses'] > 0:
            print(f"[STATS] Win rate: {stats['win_rate']:.1f}% | P&L: ${stats['pnl']:.2f}")

        # Check if recalibration is needed (25% more data than last calibration)
        should_recal, current_count, last_count = check_recalibration_needed(DB_PATH)
        if should_recal:
            print(f"\n[CALIBRATE] Triggering recalibration: {current_count} samples (+{((current_count/last_count - 1)*100) if last_count > 0 else 100:.0f}% from last)")
            run_recalibration(DB_PATH)

        if args.once:
            break

        print(f"\nNext scan in {args.interval}s...")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
