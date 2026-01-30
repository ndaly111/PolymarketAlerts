#!/usr/bin/env python3
"""Compute trading edges using ML-based probability models.

This script uses trained LightGBM quantile regression models to estimate
temperature probability distributions, then finds positive EV opportunities
against Kalshi weather market prices.

Usage:
    # Compute edges for NYC using ML model
    python compute_ml_edges.py --city NHIGH

    # Compute for specific date
    python compute_ml_edges.py --city NHIGH --date 2026-01-30

    # Save results to JSON
    python compute_ml_edges.py --city NHIGH --save

    # Verbose output with all details
    python compute_ml_edges.py --city NHIGH -v
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "weather" / "models"
CONFIG_PATH = ROOT / "weather" / "config" / "cities.yml"
DB_PATH = Path(os.getenv("WEATHER_DB_PATH", str(ROOT / "weather" / "data" / "weather_forecast_accuracy.db")))
OUT_BASE = ROOT / "weather" / "outputs" / "ml_edges"


@dataclass(frozen=True)
class City:
    key: str
    label: str
    tz: str
    series_ticker: str


@dataclass
class TradeCandidate:
    market_ticker: str
    title: str
    event_type: str  # "above", "below", "between"
    threshold_a: int
    threshold_b: Optional[int]
    model_prob: float
    yes_ask: float
    no_ask: float
    best_side: str  # "YES" or "NO"
    best_price: float
    ev: float
    volume: int
    open_interest: int


def load_cities(config_path: Path) -> Dict[str, City]:
    """Load city configuration."""
    with open(config_path) as f:
        data = yaml.safe_load(f)

    cities = {}
    for cfg in data.get("cities", []):
        key = cfg.get("key", "")
        cities[key] = City(
            key=key,
            label=cfg.get("label", key),
            tz=cfg.get("tz", "America/New_York"),
            series_ticker=cfg.get("kalshi_series", f"KXHIGH{key[:3]}"),
        )
    return cities


def load_ml_model(city_key: str) -> Optional[Dict[str, Any]]:
    """Load trained ML model for a city."""
    model_path = MODELS_DIR / f"{city_key.lower()}_quantile.pkl"
    if not model_path.exists():
        return None

    with open(model_path, "rb") as f:
        return pickle.load(f)


def get_latest_forecast(db_path: Path, city_key: str, target_date: str) -> Optional[Dict[str, Any]]:
    """Get the latest forecast for a city and date from the database."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Get latest forecast snapshot including weather context
    result = conn.execute("""
        SELECT
            forecast_high_f,
            source,
            snapshot_time_utc,
            snapshot_hour_local,
            forecast_context_json
        FROM forecast_snapshots
        WHERE city_key = ? AND target_date_local = ?
        ORDER BY snapshot_time_utc DESC
        LIMIT 1
    """, (city_key, target_date)).fetchone()

    conn.close()

    if result:
        data = dict(result)
        # Parse forecast context JSON if present
        context_json = data.get("forecast_context_json")
        if context_json:
            try:
                data["forecast_context"] = json.loads(context_json)
            except (json.JSONDecodeError, TypeError):
                data["forecast_context"] = None
        else:
            data["forecast_context"] = None
        return data
    return None


def get_kalshi_markets(
    db_path: Path,
    city_key: str,
    target_date: str,
    series_ticker: str,
) -> List[Dict[str, Any]]:
    """Get Kalshi market snapshots for the city and date."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Build event ticker pattern: KXHIGHNY-26JAN30
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

    # Get only the latest snapshot for each market
    markets = {}
    for row in result:
        ticker = row["market_ticker"]
        if ticker not in markets:
            market = dict(row)
            # Extract title from raw_json
            try:
                raw = json.loads(row["raw_json"])
                # Construct title from no_sub_title or floor_strike
                if "no_sub_title" in raw:
                    market["title"] = raw["no_sub_title"]
                elif "floor_strike" in raw:
                    floor = raw.get("floor_strike", 0)
                    market["title"] = f"{floor} or above"
            except Exception:
                market["title"] = ticker
            markets[ticker] = market

    return list(markets.values())


def parse_market_event(title: str, market_ticker: str = "") -> Optional[Tuple[str, int, Optional[int]]]:
    """Parse Kalshi market title to extract event type and thresholds.

    Returns: (event_type, threshold_a, threshold_b)
        event_type: "above", "below", "between"
        threshold_a: first threshold
        threshold_b: second threshold (for "between" events)
    """
    if not title:
        return None

    title_lower = title.lower()
    import re

    # Match "X° or above" or "X or above" (from no_sub_title)
    above_match = re.search(r"(\d+)°?\s*(or above|and above)", title_lower)
    if above_match:
        return ("above", int(above_match.group(1)), None)

    # Match "X° or below" or "X or below"
    below_match = re.search(r"(\d+)°?\s*(or below|and below)", title_lower)
    if below_match:
        return ("below", int(below_match.group(1)), None)

    # Match "be 21-22°" or similar
    between_match = re.search(r"be\s+(\d+)[–-](\d+)°", title)
    if between_match:
        a, b = int(between_match.group(1)), int(between_match.group(2))
        return ("between", min(a, b), max(a, b))

    # Try to parse from market ticker (e.g., KXHIGHNY-26JAN30-T21 means >= 21)
    # T = threshold (above), B = between
    if market_ticker:
        t_match = re.search(r"-T(\d+)$", market_ticker)
        if t_match:
            return ("above", int(t_match.group(1)), None)

        b_match = re.search(r"-B(\d+\.?\d*)$", market_ticker)
        if b_match:
            # B21.5 means between 21-22
            mid = float(b_match.group(1))
            a = int(mid - 0.5)
            b = int(mid + 0.5)
            return ("between", a, b)

    return None


def get_weather_features(
    db_path: Path,
    city_key: str,
    target_date: str,
) -> Dict[str, Any]:
    """Load additional weather features (snow, SST, visibility, smoke) for prediction."""
    conn = sqlite3.connect(db_path)
    features = {}

    # Snow
    row = conn.execute(
        """
        SELECT snow_depth_inches FROM snow_observations
        WHERE city_key = ? AND date_local = ?
        ORDER BY snapshot_time_utc DESC LIMIT 1
        """,
        (city_key, target_date),
    ).fetchone()
    if row and row[0] is not None:
        features["snow_depth_inches"] = row[0]

    # SST
    row = conn.execute(
        """
        SELECT sst_f, sst_anomaly_f FROM sst_observations
        WHERE city_key = ? AND date_local = ?
        ORDER BY observation_time_utc DESC LIMIT 1
        """,
        (city_key, target_date),
    ).fetchone()
    if row:
        features["sst_f"] = row[0]
        features["sst_anomaly_f"] = row[1]

    # Visibility
    row = conn.execute(
        """
        SELECT MIN(visibility_miles) FROM visibility_observations
        WHERE city_key = ? AND date_local = ?
        """,
        (city_key, target_date),
    ).fetchone()
    if row and row[0] is not None:
        features["visibility_min_miles"] = row[0]

    # Smoke
    row = conn.execute(
        """
        SELECT smoke_present FROM smoke_observations
        WHERE city_key = ? AND date_local = ?
        ORDER BY observation_time_utc DESC LIMIT 1
        """,
        (city_key, target_date),
    ).fetchone()
    if row:
        features["smoke_present"] = bool(row[0])

    conn.close()
    return features


def predict_temperature_distribution(
    model_data: Dict[str, Any],
    forecast_high: int,
    target_date: datetime,
    forecast_context: Optional[Dict[str, Any]] = None,
    snapshot_hour: Optional[int] = None,
    city_key: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> Dict[float, float]:
    """Predict temperature quantiles using ML model.

    Args:
        model_data: Loaded ML model data
        forecast_high: Point forecast temperature
        target_date: Target date
        forecast_context: Weather context dict from forecast_context_json
        snapshot_hour: Hour of forecast snapshot (for lead time calculation)
        city_key: City key for loading additional features
        db_path: Database path for loading additional features

    Returns: Dict mapping quantile (0.05-0.95) to temperature value.
    """
    from weather.lib.ml_models import FEATURE_NAMES, prepare_features

    models = model_data["models"]

    # Extract weather context features if available
    precip_prob = 0.0
    cloud_cover = 0.5
    wind_speed = 10.0
    if forecast_context:
        precip_prob = forecast_context.get("precip_prob_max", 0) / 100.0
        cloud_cover = forecast_context.get("cloud_cover_avg", 50) / 100.0
        wind_speed = forecast_context.get("wind_speed_avg", 10)

    # Calculate lead hours if snapshot_hour provided
    lead_hours = 18  # Default
    if snapshot_hour is not None:
        lead_hours = max(0, 18 - snapshot_hour)

    # Load additional weather features from database
    weather_features = {}
    if city_key and db_path:
        target_date_str = target_date.strftime("%Y-%m-%d")
        weather_features = get_weather_features(db_path, city_key, target_date_str)

    # Prepare features with weather context
    features = prepare_features(
        forecast_high=forecast_high,
        target_date=target_date,
        lead_hours=lead_hours,
        precip_prob=precip_prob,
        cloud_cover=cloud_cover,
        wind_speed=wind_speed,
        # New weather features (use defaults if not available)
        snow_depth_inches=weather_features.get("snow_depth_inches", 0.0),
        sst_f=weather_features.get("sst_f"),
        sst_anomaly_f=weather_features.get("sst_anomaly_f"),
        visibility_min_miles=weather_features.get("visibility_min_miles"),
        smoke_present=weather_features.get("smoke_present", False),
    )

    # Convert to numpy array
    X = np.zeros((1, len(FEATURE_NAMES)))
    for i, name in enumerate(FEATURE_NAMES):
        X[0, i] = features.get(name, 0.0)

    # Predict error quantiles and convert to temperature quantiles
    temp_quantiles = {}
    for q in sorted(models.keys()):
        model = models[q]
        error_pred = model.predict(X)[0]
        temp_quantiles[q] = forecast_high + error_pred

    return temp_quantiles


def prob_above_threshold(threshold: float, temp_quantiles: Dict[float, float]) -> float:
    """Compute P(temperature >= threshold) from quantile distribution."""
    qs = sorted(temp_quantiles.keys())
    temps = [temp_quantiles[q] for q in qs]

    # Edge cases
    if threshold <= temps[0]:
        return 1.0 - qs[0]  # Very high probability
    if threshold >= temps[-1]:
        return 1.0 - qs[-1]  # Very low probability

    # Linear interpolation between quantiles
    for i in range(len(temps) - 1):
        if temps[i] <= threshold <= temps[i + 1]:
            frac = (threshold - temps[i]) / (temps[i + 1] - temps[i])
            return 1.0 - (qs[i] + frac * (qs[i + 1] - qs[i]))

    return 0.5  # Fallback


def prob_between_thresholds(a: int, b: int, temp_quantiles: Dict[float, float]) -> float:
    """Compute P(a <= temperature <= b) from quantile distribution."""
    return prob_above_threshold(a, temp_quantiles) - prob_above_threshold(b + 1, temp_quantiles)


def compute_ev(prob: float, price: float, fee: float = 0.02) -> float:
    """Compute expected value of buying at a price given true probability."""
    # EV = prob * (1 - price) - (1 - prob) * price - fee
    # Simplified: EV = prob - price - fee
    return prob - price - fee


def find_trading_edges(
    markets: List[Dict[str, Any]],
    temp_quantiles: Dict[float, float],
    min_ev: float = 0.02,
    fee_cents: int = 2,
) -> List[TradeCandidate]:
    """Find positive EV trading opportunities."""
    candidates = []
    fee = fee_cents / 100.0

    for market in markets:
        title = market.get("title", "")
        market_ticker = market.get("market_ticker", "")
        event = parse_market_event(title, market_ticker)
        if not event:
            continue

        event_type, a, b = event

        # Compute model probability
        if event_type == "between":
            prob = prob_between_thresholds(a, b, temp_quantiles)
        elif event_type == "above":
            prob = prob_above_threshold(a, temp_quantiles)
        elif event_type == "below":
            prob = 1.0 - prob_above_threshold(a, temp_quantiles)
        else:
            continue

        # Get market prices (stored as cents, convert to dollars)
        yes_ask_cents = market.get("yes_ask")
        no_ask_cents = market.get("no_ask")

        if yes_ask_cents is None or no_ask_cents is None:
            continue

        yes_ask = float(yes_ask_cents) / 100.0  # Convert cents to dollars
        no_ask = float(no_ask_cents) / 100.0

        # Compute EV for both sides
        ev_yes = compute_ev(prob, yes_ask, fee)
        ev_no = compute_ev(1 - prob, no_ask, fee)

        # Take the better side
        if ev_yes >= ev_no and ev_yes >= min_ev:
            candidates.append(TradeCandidate(
                market_ticker=market.get("market_ticker", ""),
                title=title,
                event_type=event_type,
                threshold_a=a,
                threshold_b=b,
                model_prob=prob,
                yes_ask=yes_ask,
                no_ask=no_ask,
                best_side="YES",
                best_price=yes_ask,
                ev=ev_yes,
                volume=int(market.get("volume", 0)),
                open_interest=int(market.get("open_interest", 0)),
            ))
        elif ev_no >= min_ev:
            candidates.append(TradeCandidate(
                market_ticker=market.get("market_ticker", ""),
                title=title,
                event_type=event_type,
                threshold_a=a,
                threshold_b=b,
                model_prob=prob,
                yes_ask=yes_ask,
                no_ask=no_ask,
                best_side="NO",
                best_price=no_ask,
                ev=ev_no,
                volume=int(market.get("volume", 0)),
                open_interest=int(market.get("open_interest", 0)),
            ))

    # Sort by EV descending
    candidates.sort(key=lambda x: x.ev, reverse=True)
    return candidates


def process_city(
    city: City,
    target_date: Optional[str],
    min_ev: float,
    fee_cents: int,
    save: bool,
    verbose: bool,
) -> List[TradeCandidate]:
    """Process a single city and return trading candidates."""
    # Load ML model
    model_data = load_ml_model(city.key)
    if not model_data:
        if verbose:
            print(f"[{city.key}] No ML model found, skipping")
        return []

    # Determine target date
    tz = ZoneInfo(city.tz)
    if target_date:
        target_date_str = target_date
        target_dt = datetime.strptime(target_date_str, "%Y-%m-%d").replace(tzinfo=tz)
    else:
        now = datetime.now(tz)
        target_dt = now
        target_date_str = now.strftime("%Y-%m-%d")

    # Get latest forecast
    forecast = get_latest_forecast(DB_PATH, city.key, target_date_str)
    if not forecast:
        if verbose:
            print(f"[{city.key}] No forecast found for {target_date_str}")
        return []

    forecast_high = int(forecast["forecast_high_f"])
    forecast_context = forecast.get("forecast_context")
    snapshot_hour = forecast.get("snapshot_hour_local")

    # Predict temperature distribution using ML model with weather context
    temp_quantiles = predict_temperature_distribution(
        model_data,
        forecast_high,
        target_dt,
        forecast_context=forecast_context,
        snapshot_hour=snapshot_hour,
        city_key=city.key,
        db_path=DB_PATH,
    )

    # Get Kalshi markets
    markets = get_kalshi_markets(DB_PATH, city.key, target_date_str, city.series_ticker)
    if not markets:
        if verbose:
            print(f"[{city.key}] No Kalshi markets for {target_date_str}")
        return []

    # Find trading edges
    candidates = find_trading_edges(
        markets,
        temp_quantiles,
        min_ev=min_ev,
        fee_cents=fee_cents,
    )

    # Save results if requested
    if save and candidates:
        out_dir = OUT_BASE / target_date_str
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{city.key}.json"

        result = {
            "city_key": city.key,
            "city_label": city.label,
            "target_date": target_date_str,
            "forecast_high_f": forecast_high,
            "forecast_source": forecast["source"],
            "model_source": "ml_quantile",
            "temp_quantiles": {str(k): v for k, v in temp_quantiles.items()},
            "min_ev": min_ev,
            "fee_cents": fee_cents,
            "candidates": [
                {
                    "market_ticker": c.market_ticker,
                    "title": c.title,
                    "event_type": c.event_type,
                    "threshold_a": c.threshold_a,
                    "threshold_b": c.threshold_b,
                    "model_prob": c.model_prob,
                    "yes_ask": c.yes_ask,
                    "no_ask": c.no_ask,
                    "best_side": c.best_side,
                    "best_price": c.best_price,
                    "ev": c.ev,
                    "volume": c.volume,
                    "open_interest": c.open_interest,
                }
                for c in candidates
            ],
            "computed_at_utc": datetime.now(ZoneInfo("UTC")).isoformat(),
        }

        with open(out_file, "w") as f:
            json.dump(result, f, indent=2)

    return candidates


def main():
    parser = argparse.ArgumentParser(
        description="Compute trading edges using ML-based probability models"
    )
    parser.add_argument(
        "--city",
        default="",
        help="City key (e.g., NHIGH). Use --all for all cities.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all cities with trained models",
    )
    parser.add_argument(
        "--date",
        help="Target date YYYY-MM-DD (default: today in city's timezone)",
    )
    parser.add_argument(
        "--min-ev",
        type=float,
        default=0.02,
        help="Minimum EV threshold (default: 0.02 = 2 cents)",
    )
    parser.add_argument(
        "--fee-cents",
        type=int,
        default=2,
        help="Kalshi fee in cents (default: 2)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save results to JSON file",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )
    args = parser.parse_args()

    # Load city config
    cities = load_cities(CONFIG_PATH)

    # Determine which cities to process
    if args.all:
        # Process all cities that have trained models
        cities_to_process = []
        for key, city in cities.items():
            model_path = MODELS_DIR / f"{key.lower()}_quantile.pkl"
            if model_path.exists():
                cities_to_process.append(city)
        print(f"Processing {len(cities_to_process)} cities with ML models...")
        print()
    elif args.city:
        if args.city not in cities:
            print(f"ERROR: Unknown city '{args.city}'")
            print(f"Available: {', '.join(cities.keys())}")
            return 1
        cities_to_process = [cities[args.city]]
    else:
        # Default to NHIGH
        cities_to_process = [cities.get("NHIGH", list(cities.values())[0])]

    # Process each city
    all_candidates = []
    for city in cities_to_process:
        candidates = process_city(
            city=city,
            target_date=args.date,
            min_ev=args.min_ev,
            fee_cents=args.fee_cents,
            save=args.save,
            verbose=args.verbose,
        )

        if candidates:
            print(f"=== {city.label} ({city.key}) ===")
            for c in candidates[:3]:  # Show top 3 per city
                print(f"  {c.best_side} @ ${c.best_price:.2f} | EV: +{c.ev*100:.1f}¢ | {c.title}")
            if len(candidates) > 3:
                print(f"  ... and {len(candidates) - 3} more")
            print()
            all_candidates.extend(candidates)

    # Summary
    if all_candidates:
        all_candidates.sort(key=lambda x: x.ev, reverse=True)
        print(f"=== TOP 10 OPPORTUNITIES (All Cities) ===")
        for c in all_candidates[:10]:
            # Find city label
            city_label = next((ct.label for ct in cities_to_process if ct.series_ticker in c.market_ticker), "?")
            print(f"  {c.best_side} @ ${c.best_price:.2f} | EV: +{c.ev*100:.1f}¢ | {city_label}: {c.title}")
        print()
        print(f"Total opportunities: {len(all_candidates)}")
    else:
        print("No positive EV opportunities found across all cities.")

    return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
