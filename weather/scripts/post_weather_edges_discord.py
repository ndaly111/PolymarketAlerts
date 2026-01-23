#!/usr/bin/env python3
"""
Post the daily weather edges summary to Discord.

Looks for:
  weather/outputs/edges/{forecast_source}/{YYYY-MM-DD}/SUMMARY.md

If missing, posts a small fallback message.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

ROOT = Path(__file__).resolve().parents[2]
OUT_BASE = ROOT / "weather" / "outputs" / "edges"


def _today_local() -> str:
    return datetime.now(tz=ZoneInfo("America/New_York")).date().isoformat()


def _truncate(s: str, max_len: int = 1900) -> str:
    s = (s or "").strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 20].rstrip() + "\n\n...(truncated)"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--forecast-source", default="nws_hourly_max")
    p.add_argument("--date", default="", help="YYYY-MM-DD (default: today America/New_York)")
    args = p.parse_args()

    webhook = (os.getenv("DISCORD_WEBHOOK_URL") or "").strip()
    if not webhook:
        raise SystemExit(
            "DISCORD_WEBHOOK_URL is not set (env). Add a repo Secret/Variable and map it in the workflow."
        )

    src = str(args.forecast_source).replace("/", "_")
    day = args.date.strip() or _today_local()

    summary_path = OUT_BASE / src / day / "SUMMARY.md"
    if summary_path.exists():
        txt = summary_path.read_text(encoding="utf-8", errors="ignore")
        content = _truncate(txt)
    else:
        content = f"Weather edges: no SUMMARY.md found for {day} ({src})."

    # Discord hard limit is 2000 chars; keep room.
    payload = {"content": content[:1990]}
    r = requests.post(webhook, json=payload, timeout=20)
    if r.status_code >= 300:
        raise SystemExit(f"Discord post failed: HTTP {r.status_code} {r.text[:200]}")
    print("[ok] posted weather edges to Discord")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
