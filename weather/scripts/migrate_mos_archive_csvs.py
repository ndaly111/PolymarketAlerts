#!/usr/bin/env python3
"""
Move MOS archive CSVs into the canonical repo location:
  weather/data/forecast_archive/

Why:
  - GitHub Actions backfill workflow expects MOS CSVs under weather/data/forecast_archive/
  - Users sometimes add the CSVs at repo root (kmdw_*.csv, kmia_*.csv, klax_*.csv, knyc_*.csv)

This script:
  - finds matching files in repo root (and optionally anywhere under repo)
  - moves them into weather/data/forecast_archive/
  - prints a summary

Safe behavior:
  - if destination file already exists, it will NOT overwrite (it will skip)
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import List


ROOT = Path(__file__).resolve().parents[2]
DEST_DIR = ROOT / "weather" / "data" / "forecast_archive"

PREFIXES = ("kmdw_", "kmia_", "klax_", "knyc_")


def _is_target_csv(path: Path) -> bool:
    if not path.is_file():
        return False
    name = path.name
    low = name.lower()
    return low.endswith(".csv") and low.startswith(PREFIXES)


def _collect_candidates(search_root: Path, recursive: bool) -> List[Path]:
    if recursive:
        files = [p for p in search_root.rglob("*.csv") if _is_target_csv(p)]
    else:
        files = [p for p in search_root.glob("*.csv") if _is_target_csv(p)]
    return sorted({p.resolve() for p in files})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search recursively from repo root (default: root only)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be moved without changing files",
    )
    args = parser.parse_args()

    DEST_DIR.mkdir(parents=True, exist_ok=True)

    candidates = _collect_candidates(ROOT, recursive=bool(args.recursive))

    moved = 0
    skipped = 0
    for src in candidates:
        if DEST_DIR in src.parents:
            continue

        dst = DEST_DIR / src.name
        if dst.exists():
            skipped += 1
            continue

        if args.dry_run:
            print(f"[dry-run] would move: {src} -> {dst}")
        else:
            shutil.move(str(src), str(dst))
            print(f"[moved] {src} -> {dst}")
        moved += 1

    print(f"[done] moved={moved} skipped_existing={skipped} dest={DEST_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
