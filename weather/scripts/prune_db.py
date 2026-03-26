"""Prune old data from weather_forecast_accuracy.db to keep it under GitHub's 100MB limit.

Usage:
    python -m weather.scripts.prune_db --db weather/data/weather_forecast_accuracy.db [--keep-years 5]
"""
import argparse
import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

DEFAULT_DB = Path(os.getenv(
    "WEATHER_DB_PATH",
    str(Path(__file__).resolve().parents[2] / "weather" / "data" / "weather_forecast_accuracy.db"),
))


def prune(db_path: Path, keep_years: int = 5) -> None:
    cutoff = (datetime.now() - timedelta(days=keep_years * 365)).strftime("%Y-%m-%d")
    conn = sqlite3.connect(str(db_path))

    size_before = db_path.stat().st_size
    print(f"DB size before: {size_before / 1_048_576:.1f} MB")

    tables_to_prune = {
        "observed_cli": "date_local",
        "climate_indices": "date_local",
        "snow_observations": "date_local",
    }

    for table, date_col in tables_to_prune.items():
        try:
            before = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            conn.execute(f"DELETE FROM {table} WHERE {date_col} < ?", (cutoff,))
            after = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            removed = before - after
            if removed > 0:
                print(f"  {table}: removed {removed} rows older than {cutoff} ({before} -> {after})")
        except Exception as e:
            print(f"  {table}: skip ({e})")

    conn.commit()
    print("Running VACUUM...")
    conn.execute("VACUUM")
    conn.close()

    size_after = db_path.stat().st_size
    print(f"DB size after:  {size_after / 1_048_576:.1f} MB (saved {(size_before - size_after) / 1_048_576:.1f} MB)")


def main():
    p = argparse.ArgumentParser(description="Prune old data from weather DB")
    p.add_argument("--db", default=str(DEFAULT_DB))
    p.add_argument("--keep-years", type=int, default=5, help="Keep data from the last N years (default 5)")
    args = p.parse_args()
    prune(Path(args.db), args.keep_years)


if __name__ == "__main__":
    main()
