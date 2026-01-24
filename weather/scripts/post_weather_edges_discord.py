#!/usr/bin/env python3
"""Post daily weather edges to Discord.

Preferred output format grouped by city:

    New York
    H: 34

    - 34-35
    Fair Odds -120 | Kalshi YES +100
    EV (12.3%)

    - 36-37
    Fair Odds +300 | Kalshi YES +290
    EV (0.8%)

Falls back to SUMMARY.md if per-city JSON artifacts aren't present.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional
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


def _join_blocks_with_limit(blocks: List[str], limit: int = 1990) -> str:
    """
    Join blocks separated by blank lines but NEVER exceed the limit.
    If it won't fit, stop at a block boundary (no mid-city slicing).
    """
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
    """Convert probability (0..1) to American odds string like -120 or +145."""
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


def _bucket_kelly_percent(pct: float) -> float:
    """
    Floor Kelly percent into:
      0, 2.5, 5, 10, 15, 20, 25, ...

    Rules (conservative):
      - < 2.5%  -> 0
      - 2.5–<5  -> 2.5
      - 5–<10   -> 5
      - 10–<15  -> 10
      - 15–<20  -> 15
      - etc.
    """
    try:
        pct = float(pct)
    except Exception:
        return 0.0

    if pct < 2.5:
        return 0.0

    if pct < 5.0:
        return 2.5

    return float(int(pct // 5) * 5)


def _half_kelly_percent(win_prob: float, cost: float) -> float:
    """Return half-Kelly percent for a binary payout (win pays 1, lose pays 0)."""
    try:
        p = float(win_prob)
        c = float(cost)
    except Exception:
        return 0.0
    if not (0.0 < p < 1.0) or not (0.0 < c < 1.0):
        return 0.0
    kelly_fraction = (p - c) / (1.0 - c)
    if kelly_fraction <= 0:
        return 0.0
    return 100.0 * 0.5 * kelly_fraction


def _event_display(kind: str, a: Any, b: Any) -> str:
    k = (kind or "").strip().lower()
    try:
        a_i = int(a)
    except Exception:
        a_i = a
    try:
        b_i = int(b) if b is not None else None
    except Exception:
        b_i = b
    if k in ("between", "range", "in") and b_i is not None:
        lo = int(min(a_i, b_i))
        hi = int(max(a_i, b_i))
        return f"{lo}-{hi}"
    if k in ("ge", "gte"):
        return f">={a_i}"
    if k in ("gt",):
        return f">{a_i}"
    if k in ("le", "lte"):
        return f"<={a_i}"
    if k in ("lt",):
        return f"<{a_i}"
    if k in ("exact", "eq"):
        return str(a_i)
    return str(a_i) if b_i is None else f"{a_i}-{b_i}"


def _render_city_blocks(edge_dir: Path) -> Optional[str]:
    """Render grouped-by-city summary from per-city JSON artifacts."""
    if not edge_dir.exists():
        return None

    city_files = sorted([p for p in edge_dir.glob("*.json") if p.is_file()], key=lambda p: p.name)
    if not city_files:
        return None

    blocks: List[str] = []
    any_edges = False

    for p in city_files:
        try:
            obj = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue

        label = str(obj.get("label") or obj.get("city_key") or p.stem)
        forecast_high_f = obj.get("forecast_high_f")
        fee_open = float(((obj.get("fee") or {}).get("open_fee_dollars") or 0.0))

        candidates = obj.get("candidates") or []
        if not candidates:
            continue
        any_edges = True

        lines: List[str] = [label]
        if forecast_high_f is not None:
            lines.append(f"H: {forecast_high_f}")
        lines.append("")

        for r in candidates:
            event = r.get("event") or {}
            display = _event_display(str(event.get("kind") or ""), event.get("a"), event.get("b"))

            q = float(r.get("model_q") or 0.0)
            decision = r.get("decision") or {}
            side = str(decision.get("side_to_buy") or "").upper()
            buy_price = float(decision.get("buy_price") or 0.0)
            ev = float(decision.get("ev") or 0.0)

            fair_p = q if side == "YES" else (1.0 - q)
            fair_odds = _prob_to_american_odds(fair_p)
            kalshi_odds = _prob_to_american_odds(buy_price)

            cost = max(1e-9, buy_price + fee_open)
            ev_pct = 100.0 * (ev / cost)
            kelly_pct = _bucket_kelly_percent(_half_kelly_percent(fair_p, cost))

            lines.append(f"- {display}")
            fee_cents = int(round(100.0 * fee_open))
            fee_txt = f" (fee +{fee_cents}¢)" if fee_cents > 0 else ""
            lines.append(f"Fair Odds {fair_odds} | Kalshi {side} {kalshi_odds}{fee_txt}")
            lines.append(f"EV ({ev_pct:.1f}%) | Kelly {kelly_pct:g}%")
            lines.append("")

        blocks.append("\n".join(lines).rstrip())

    if not any_edges:
        return None

    # Respect Discord limits without chopping mid-city.
    txt = _join_blocks_with_limit(blocks, limit=1990).strip()
    return txt or None

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--forecast-source", default="nws_hourly_max")
    p.add_argument("--date", default="", help="YYYY-MM-DD (default: today America/New_York)")
    args = p.parse_args()

    webhook = (os.getenv("WEATHER_DISCORD_WEBHOOK") or os.getenv("DISCORD_WEBHOOK_URL") or "").strip()
    if not webhook:
        raise SystemExit(
            "WEATHER_DISCORD_WEBHOOK is not set (env). Add a repo Secret/Variable and map it in the workflow."
        )

    src = str(args.forecast_source).replace("/", "_")
    day = args.date.strip() or _today_local()

    out_dir = OUT_BASE / src / day
    content = _render_city_blocks(out_dir)

    if not content:
        summary_path = out_dir / "SUMMARY.md"
        if summary_path.exists():
            txt = summary_path.read_text(encoding="utf-8", errors="ignore")
            content = _truncate(txt)
        else:
            content = f"Weather edges: no edges found for {day} ({src})."

    # Discord hard limit is 2000 chars; keep room.
    payload = {"content": content if len(content) <= 1990 else _truncate(content, max_len=1990)}
    r = requests.post(webhook, json=payload, timeout=20)
    if r.status_code >= 300:
        raise SystemExit(f"Discord post failed: HTTP {r.status_code} {r.text[:500]}")
    print(f"[ok] posted weather edges to Discord (status={r.status_code})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
