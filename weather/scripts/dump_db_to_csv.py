#!/usr/bin/env python3
"""
Dump every SQLite table into CSV files.

Output:
  weather/db_dump/<table>.csv

Purpose:
  - Inspect what data actually exists
  - Verify archive backfills
  - Debug missing models or snapshots
  - Phone-friendly review via CSV artifacts
"""

from __future__ import annotations

import csv
import sqlite3
from pathlib import Path


DB_PATH = Path("weather/data/weather.db")
OUT_DIR = Path("weather/db_dump")


def main() -> int:
    if not DB_PATH.exists():
        raise SystemExit(f"DB not found: {DB_PATH}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
    ).fetchall()

    print(f"Found {len(tables)} tables\n")

    for row in tables:
        table = row["name"]
        print(f"Dumping table: {table}")

        cur = conn.execute(f"SELECT * FROM {table};")
        rows = cur.fetchall()

        if not rows:
            print("  → empty")
            continue

        csv_path = OUT_DIR / f"{table}.csv"

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # header
            writer.writerow(rows[0].keys())

            # rows
            for r in rows:
                writer.writerow(list(r))

        print(f"  → wrote {len(rows)} rows")

    conn.close()

    print("\nDone.")
    print(f"CSV output directory: {OUT_DIR.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
