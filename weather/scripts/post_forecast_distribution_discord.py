#!/usr/bin/env python3
"""Post forecast distribution summaries to Discord."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import requests

ROOT = Path(__file__).resolve().parents[2]
OUT_BASE = ROOT / "weather" / "outputs" / "fair_prices"


def _today_local() -> str:
    return datetime.now(tz=ZoneInfo("America/New_York")).date().isoformat()


def _truncate(s: str, max_len: int = 1900) -> str:
    s = (s or "").strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 20].rstrip() + "\n\n...(truncated)"


def _format_date_mdy(date_str: str) -> str:
    if not date_str:
        return ""
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return date_str
    return f"{dt.month}-{dt.day}-{dt.year}"


def _join_blocks_with_limit(blocks: List[str], limit: int = 1990) -> str:
    out: List[str] = []
    used = 0
    sep = "\n\n"
    for b in blocks:
        b = (b or "").strip()
        if not b:
            continue
        add = b if not out else (sep + b)
        if used + len(add) > limit:
            break
        out.append(b)
        used += len(add)
    if not out:
        return ""
    txt = sep.join(out).strip()
    return txt if len(txt) <= limit else _truncate(txt, max_len=limit)


def _prob_to_american_odds(p: float) -> str:
    try:
        p = float(p)
    except Exception:
        return "N/A"
    if not (0.0 < p < 1.0):
        return "N/A"
    if p == 0.5:
        return "+100"
    if p > 0.5:
        odds = -100.0 * p / (1.0 - p)
    else:
        odds = 100.0 * (1.0 - p) / p
    oi = int(round(odds))
    return f"{oi:+d}" if oi > 0 else str(oi)


def _parse_pmf(pmf_raw: Dict[str, float]) -> List[Tuple[int, float]]:
    pmf: List[Tuple[int, float]] = []
    for k, v in pmf_raw.items():
        try:
            temp = int(k)
            prob = float(v)
        except Exception:
            continue
        pmf.append((temp, prob))
    return sorted(pmf, key=lambda item: item[0])


def _select_entries(pmf: List[Tuple[int, float]], min_prob: float, fallback: int = 3) -> List[Tuple[int, float]]:
    selected = [(t, p) for t, p in pmf if p >= min_prob]
    if selected:
        return selected
    if not pmf:
        return []
    top = sorted(pmf, key=lambda item: item[1], reverse=True)[:fallback]
    return sorted(top, key=lambda item: item[0])


def _render_city_blocks(out_dir: Path, min_prob: float) -> Optional[str]:
    if not out_dir.exists():
        return None

    city_files = sorted([p for p in out_dir.glob("*.json") if p.is_file()], key=lambda p: p.name)
    if not city_files:
        return None

    blocks: List[str] = []
    any_city = False

    for p in city_files:
        try:
            obj = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue

        pmf_raw = obj.get("pmf_high_f") or {}
        pmf = _parse_pmf(pmf_raw)
        entries = _select_entries(pmf, min_prob=min_prob)
        if not entries:
            continue

        label = str(obj.get("label") or obj.get("city_key") or p.stem)
        target_date = _format_date_mdy(str(obj.get("target_date_local") or "").strip())
        forecast_high = obj.get("forecast_high_f")

        if target_date:
            header = f"{label} {target_date}"
        else:
            header = label
        lines: List[str] = [header]
        if forecast_high is not None:
            lines.append(f"Forecast H: {forecast_high}")
        lines.append("")

        for temp, prob in entries:
            odds = _prob_to_american_odds(prob)
            pct = int(round(prob * 100.0))
            lines.append(f"{temp} - {pct}% ({odds})")

        blocks.append("\n".join(lines).rstrip())
        any_city = True

    if not any_city:
        return None

    txt = _join_blocks_with_limit(blocks, limit=1990).strip()
    return txt or None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--forecast-source", default="nws_hourly_max")
    p.add_argument("--date", default="", help="YYYY-MM-DD (default: today America/New_York)")
    p.add_argument("--min-prob", type=float, default=0.05, help="Minimum probability to include in output")
    p.add_argument(
        "--require-data",
        action="store_true",
        help="Skip posting when no fair-price distributions are available.",
    )
    args = p.parse_args()

    webhook = (os.getenv("WEATHER_DISCORD_WEBHOOK") or os.getenv("DISCORD_WEBHOOK_URL") or "").strip()
    if not webhook:
        raise SystemExit(
            "WEATHER_DISCORD_WEBHOOK is not set (env). Add a repo Secret/Variable and map it in the workflow."
        )

    src = str(args.forecast_source).replace("/", "_")
    day = args.date.strip() or _today_local()
    out_dir = OUT_BASE / src / day

    content = _render_city_blocks(out_dir, min_prob=float(args.min_prob))
    if not content:
        if args.require_data:
            print(f"[skip] no fair-price distributions for {day} ({src}); skipping Discord post.")
            return 0
        content = f"Forecast distribution: no fair-price outputs for {day} ({src})."

    payload = {"content": content if len(content) <= 1990 else _truncate(content, max_len=1990)}
    r = requests.post(webhook, json=payload, timeout=20)
    if r.status_code >= 300:
        raise SystemExit(f"Discord post failed: HTTP {r.status_code} {r.text[:500]}")
    print(f"[ok] posted forecast distribution to Discord (status={r.status_code})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
