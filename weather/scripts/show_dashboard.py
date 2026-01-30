#!/usr/bin/env python3
"""Weather Trading Dashboard - View ML model status, predictions, and trading edges.

Usage:
    # Show current status
    python show_dashboard.py

    # Show specific city details
    python show_dashboard.py --city NHIGH

    # Watch mode - refresh every N seconds
    python show_dashboard.py --watch 30

    # JSON output for integration
    python show_dashboard.py --json
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import yaml

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "weather" / "models"
CONFIG_PATH = ROOT / "weather" / "config" / "cities.yml"
DB_PATH = Path(os.getenv("WEATHER_DB_PATH", str(ROOT / "weather" / "data" / "weather_forecast_accuracy.db")))
ML_EDGES_DIR = ROOT / "weather" / "outputs" / "ml_edges"


def load_cities() -> Dict[str, Dict[str, Any]]:
    """Load city configuration."""
    with open(CONFIG_PATH) as f:
        data = yaml.safe_load(f)

    cities = {}
    for cfg in data.get("cities", []):
        key = cfg.get("key", "")
        cities[key] = {
            "key": key,
            "label": cfg.get("label", key),
            "tz": cfg.get("tz", "America/New_York"),
            "series_ticker": cfg.get("kalshi_series", f"KXHIGH{key[:3]}"),
        }
    return cities


def get_model_info(city_key: str) -> Optional[Dict[str, Any]]:
    """Get ML model metadata for a city."""
    model_path = MODELS_DIR / f"{city_key.lower()}_quantile.pkl"
    if not model_path.exists():
        return None

    try:
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)

        metadata = model_data.get("metadata")
        if metadata:
            # Use pinball loss at q=0.50 as a proxy for accuracy (lower is better)
            pinball_50 = metadata.validation_metrics.get("pinball_q0.50", 0)
            return {
                "trained_at": metadata.trained_at_utc,
                "training_samples": metadata.training_samples,
                "validation_samples": metadata.validation_samples,
                "pinball_50": pinball_50,  # Lower is better, ~1.0 is typical
            }
    except Exception as e:
        return {"error": str(e)}

    return None


def get_latest_forecast(city_key: str, target_date: str) -> Optional[Dict[str, Any]]:
    """Get the latest forecast for a city."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    result = conn.execute("""
        SELECT forecast_high_f, source, snapshot_time_utc
        FROM forecast_snapshots
        WHERE city_key = ? AND target_date_local = ?
        ORDER BY snapshot_time_utc DESC
        LIMIT 1
    """, (city_key, target_date)).fetchone()

    conn.close()
    return dict(result) if result else None


def get_latest_kalshi(city_key: str, target_date: str) -> Optional[Dict[str, Any]]:
    """Get the latest Kalshi market snapshot info."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    result = conn.execute("""
        SELECT COUNT(*) as market_count, MAX(snapshot_time_utc) as latest_snapshot
        FROM kalshi_weather_market_snapshots
        WHERE city_key = ? AND target_date_local = ?
    """, (city_key, target_date)).fetchone()

    conn.close()
    return dict(result) if result else None


def get_ml_edges(target_date: str) -> Dict[str, List[Dict[str, Any]]]:
    """Get ML edge results for a date."""
    edges = {}
    edge_dir = ML_EDGES_DIR / target_date

    if not edge_dir.exists():
        return edges

    for f in edge_dir.glob("*.json"):
        city_key = f.stem
        try:
            with open(f) as fp:
                data = json.load(fp)
                edges[city_key] = data.get("candidates", [])
        except Exception:
            pass

    return edges


def format_time_ago(timestamp_str: str) -> str:
    """Format a UTC timestamp as 'X minutes ago'."""
    if not timestamp_str:
        return "never"

    try:
        # Parse ISO format
        if "+" in timestamp_str:
            ts = datetime.fromisoformat(timestamp_str)
        else:
            ts = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))

        now = datetime.now(ZoneInfo("UTC"))
        diff = now - ts.astimezone(ZoneInfo("UTC"))

        mins = int(diff.total_seconds() / 60)
        if mins < 1:
            return "just now"
        if mins < 60:
            return f"{mins}m ago"
        hours = mins // 60
        if hours < 24:
            return f"{hours}h ago"
        days = hours // 24
        return f"{days}d ago"
    except Exception:
        return timestamp_str[:16]


def print_dashboard(cities: Dict[str, Dict[str, Any]], target_date: str, verbose: bool = False):
    """Print the dashboard to console."""
    print()
    print("=" * 70)
    print(f"  WEATHER TRADING DASHBOARD - {target_date}")
    print("=" * 70)
    print()

    # Get all ML edges
    ml_edges = get_ml_edges(target_date)

    # Summary row
    total_opps = sum(len(edges) for edges in ml_edges.values())
    models_trained = sum(1 for k in cities if (MODELS_DIR / f"{k.lower()}_quantile.pkl").exists())

    print(f"  Models: {models_trained}/{len(cities)} | Opportunities: {total_opps} | Date: {target_date}")
    print()

    # Per-city status
    print("-" * 70)
    print(f"{'City':<15} {'Model':<8} {'Forecast':<10} {'Kalshi':<10} {'Edges':<8} {'Top EV':<10}")
    print("-" * 70)

    for key, city in sorted(cities.items()):
        # Model status
        model_info = get_model_info(key)
        if model_info and "error" not in model_info:
            samples = model_info.get('training_samples', 0)
            model_status = f"{samples} smp"
        elif model_info:
            model_status = "ERROR"
        else:
            model_status = "NONE"

        # Forecast status
        forecast = get_latest_forecast(key, target_date)
        if forecast:
            forecast_status = f"{forecast['forecast_high_f']}°F"
        else:
            forecast_status = "-"

        # Kalshi status
        kalshi = get_latest_kalshi(key, target_date)
        if kalshi and kalshi.get("market_count", 0) > 0:
            kalshi_status = f"{kalshi['market_count']} mkts"
        else:
            kalshi_status = "-"

        # Edges
        city_edges = ml_edges.get(key, [])
        edge_count = len(city_edges)
        if city_edges:
            top_ev = max(e.get("ev", 0) for e in city_edges)
            top_ev_str = f"+{top_ev*100:.1f}¢"
        else:
            top_ev_str = "-"

        print(f"{city['label']:<15} {model_status:<8} {forecast_status:<10} {kalshi_status:<10} {edge_count:<8} {top_ev_str:<10}")

    print("-" * 70)
    print()

    # Top opportunities across all cities
    all_edges = []
    for city_key, edges in ml_edges.items():
        city_label = cities.get(city_key, {}).get("label", city_key)
        for edge in edges:
            edge["city_label"] = city_label
            all_edges.append(edge)

    if all_edges:
        all_edges.sort(key=lambda x: x.get("ev", 0), reverse=True)
        print("  TOP 5 OPPORTUNITIES:")
        print()
        for edge in all_edges[:5]:
            side = edge.get("best_side", "?")
            price = edge.get("best_price", 0)
            ev = edge.get("ev", 0)
            title = edge.get("title", "?")[:30]
            city = edge.get("city_label", "?")

            print(f"    {side} @ ${price:.2f} | EV: +{ev*100:.1f}¢ | {city}: {title}")
        print()

    # Data freshness
    print("  DATA FRESHNESS:")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # Latest forecast snapshot
    result = conn.execute("""
        SELECT MAX(snapshot_time_utc) as latest FROM forecast_snapshots
    """).fetchone()
    forecast_age = format_time_ago(result["latest"]) if result else "never"

    # Latest Kalshi snapshot
    result = conn.execute("""
        SELECT MAX(snapshot_time_utc) as latest FROM kalshi_weather_market_snapshots
    """).fetchone()
    kalshi_age = format_time_ago(result["latest"]) if result else "never"

    conn.close()

    print(f"    Forecasts: {forecast_age} | Kalshi: {kalshi_age}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Weather Trading Dashboard")
    parser.add_argument("--city", help="Show details for specific city")
    parser.add_argument("--date", help="Target date YYYY-MM-DD (default: today)")
    parser.add_argument("--watch", type=int, help="Refresh every N seconds")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    cities = load_cities()

    # Default to today in Eastern time
    if args.date:
        target_date = args.date
    else:
        target_date = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")

    if args.json:
        # JSON output for integration
        result = {
            "target_date": target_date,
            "generated_at": datetime.now(ZoneInfo("UTC")).isoformat(),
            "cities": {},
        }

        ml_edges = get_ml_edges(target_date)

        for key, city in cities.items():
            model_info = get_model_info(key)
            forecast = get_latest_forecast(key, target_date)
            kalshi = get_latest_kalshi(key, target_date)

            result["cities"][key] = {
                "label": city["label"],
                "model": model_info,
                "forecast": forecast,
                "kalshi": kalshi,
                "edges": ml_edges.get(key, []),
            }

        print(json.dumps(result, indent=2, default=str))
        return 0

    if args.watch:
        # Watch mode - refresh periodically
        try:
            while True:
                os.system("clear" if os.name != "nt" else "cls")
                print_dashboard(cities, target_date, args.verbose)
                print(f"  [Refreshing every {args.watch}s - Ctrl+C to exit]")
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\nExiting...")
            return 0
    else:
        print_dashboard(cities, target_date, args.verbose)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
