#!/usr/bin/env python3
"""Compute fee-adjusted EV (YES/NO) for Kalshi weather markets.

Inputs:
- Fair-price artifacts from weather/scripts/compute_fair_prices.py
- Kalshi market snapshots stored in weather.db (weather/scripts/collect_kalshi_weather_markets.py)

Outputs:
- JSON per city:
    weather/outputs/edges/{forecast_source}/{YYYY-MM-DD}/{CITY_KEY}.json
- Optional combined markdown summary:
    weather/outputs/edges/{forecast_source}/{YYYY-MM-DD}/SUMMARY.md
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import yaml

from utils.logging_utils import configure_logging

from weather.lib import db as db_lib
from weather.lib.fees import FeeSchedule, ev_no, ev_yes
from weather.lib.kalshi_weather import (
    best_buy_prices_from_snapshot_row,
    best_bid_ask_from_snapshot_row,
    parse_event_spec_from_title,
    prob_event,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "weather" / "config" / "cities.yml"
DEFAULT_DB = ROOT / "weather" / "data" / "weather.db"
FAIR_BASE = ROOT / "weather" / "outputs" / "fair_prices"
OUT_BASE = ROOT / "weather" / "outputs" / "edges"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class City:
    key: str
    label: str
    tz: str


def _now_utc() -> datetime:
    return datetime.now(tz=ZoneInfo("UTC"))


def _to_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def _central_support_interval(pmf: Dict[int, float], tail_mass: float) -> Tuple[int, int]:
    """Return (lo, hi) temperature bounds that contain ~1 - tail_mass of PMF mass.

    We implement a simple central interval by removing tail_mass/2 from each tail.
    """
    if not pmf:
        return (0, 0)
    tail = max(0.0, min(0.49, float(tail_mass) / 2.0))
    items = sorted((int(k), float(v)) for k, v in pmf.items())
    total = sum(v for _, v in items)
    if total <= 0:
        ks = [k for k, _ in items]
        return (min(ks), max(ks))

    # normalize defensively
    items = [(k, v / total) for k, v in items]

    c = 0.0
    lo = items[0][0]
    for k, v in items:
        c += v
        if c >= tail:
            lo = k
            break

    c = 0.0
    hi = items[-1][0]
    for k, v in reversed(items):
        c += v
        if c >= tail:
            hi = k
            break

    if lo > hi:
        lo, hi = hi, lo
    return lo, hi


def _pmf_min_max(pmf: Dict[int, float]) -> Tuple[int, int]:
    if not pmf:
        return (0, 0)
    ks = sorted(int(k) for k in pmf.keys())
    return (ks[0], ks[-1])


def _parse_optional_float(s: Any) -> Optional[float]:
    if s is None:
        return None
    txt = str(s).strip()
    if txt == "":
        return None
    try:
        return float(txt)
    except Exception:
        return None


def _spread_cents_for_side(row: Dict[str, Any], side: str) -> Optional[int]:
    """Return bid/ask spread in cents for YES or NO. None if bid or ask missing."""
    yes_bid, yes_ask, no_bid, no_ask = best_bid_ask_from_snapshot_row(row)
    side = (side or "").upper().strip()
    if side == "YES":
        bid, ask = yes_bid, yes_ask
    else:
        bid, ask = no_bid, no_ask
    if bid is None or ask is None:
        return None
    return ask - bid


def _strike_overlaps_support(spec_kind: str, a: int, b: Optional[int], support_lo: int, support_hi: int) -> bool:
    """
    Return True if the strike definition has non-trivial overlap with the model support interval.

    This implements the blueprint idea of ignoring "weird" strikes that are outside PMF support,
    rather than filtering on event probability q.
    """
    k = (spec_kind or "").strip().lower()

    # Common aliases / defensive normalization (depends on parse_event_spec_from_title implementation)
    if k in ("gte", "geq", "≥"):
        k = "ge"
    elif k in ("lte", "leq", "≤"):
        k = "le"
    elif k in ("range", "in", "within", "between_inclusive"):
        k = "between"
    elif k in ("eq", "=", "=="):
        k = "exact"
    elif k in ("above",):
        k = "gt"
    elif k in ("below",):
        k = "lt"

    # Normalize support bounds
    lo = int(min(support_lo, support_hi))
    hi = int(max(support_lo, support_hi))

    if k == "exact":
        return lo <= int(a) <= hi

    if k == "between" and b is not None:
        left = int(min(a, b))
        right = int(max(a, b))
        return not (right < lo or left > hi)  # interval intersection

    # Semi-infinite strikes:
    if k in ("ge",):  # TMAX >= a
        return int(a) <= hi
    if k in ("gt",):  # TMAX > a
        return int(a) < hi
    if k in ("le",):  # TMAX <= a
        return int(a) >= lo
    if k in ("lt",):  # TMAX < a
        return int(a) > lo

    # Unknown kind: be conservative and drop
    return False


def load_cities(config_path: Path) -> List[City]:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    out: List[City] = []
    for row in data.get("cities", []):
        out.append(
            City(
                key=str(row["key"]).strip(),
                label=str(row.get("label", row["key"])).strip(),
                tz=str(row.get("tz", "America/New_York")).strip(),
            )
        )
    return out



def _read_fair_artifact(
    forecast_source: str,
    target_date_local: str,
    city_key: str,
) -> Optional[Dict[str, Any]]:
    src = str(forecast_source).replace("/", "_")
    path = FAIR_BASE / src / target_date_local / f"{city_key}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None



def _market_title_from_raw(raw: Dict[str, Any]) -> str:
    for k in ("title", "market_title", "marketTitle", "name"):
        v = raw.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    m = raw.get("market") if isinstance(raw.get("market"), dict) else {}
    for k in ("title", "market_title", "marketTitle", "name"):
        v = m.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""



def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--db", default=str(DEFAULT_DB))
    p.add_argument("--city", default="", help="Optional city_key filter (default: all)")
    p.add_argument("--date", default="", help="Optional YYYY-MM-DD (default: today in each city TZ)")
    p.add_argument("--forecast-source", default="nws_hourly_max")
    p.add_argument("--fee-cents", default=os.getenv("WEATHER_BUY_FEE_CENTS", "2"))
    p.add_argument("--top-n", default=os.getenv("WEATHER_TOP_N", "25"))
    p.add_argument("--min-ev", default=os.getenv("WEATHER_MIN_EV", "0.02"))
    p.add_argument(
        "--min-q",
        default=os.getenv("WEATHER_MIN_Q", "0.05"),
        help=(
            "Drop markets where q < min_q or q > (1-min_q). "
            "Default 0.05 => drop <5 percent and >95 percent."
        ),
    )
    # Optional guardrail: strike support-overlap filtering derived from PMF central interval.
    # Default is disabled (blank). Kept separate from min_q, which enforces your true “<5% odds” rule.
    p.add_argument("--support-tail-mass", default=os.getenv("WEATHER_SUPPORT_TAIL_MASS", ""))
    # Back-compat: previous name (deprecated). If support-tail-mass is blank, we will fall back to this.
    p.add_argument("--tail-mass", default=os.getenv("WEATHER_TAIL_MASS", ""))
    p.add_argument("--min-volume", default=os.getenv("WEATHER_MIN_VOLUME", "0"))
    p.add_argument("--min-open-interest", default=os.getenv("WEATHER_MIN_OPEN_INTEREST", "0"))
    p.add_argument("--max-spread-cents", default=os.getenv("WEATHER_MAX_SPREAD_CENTS", ""))
    p.add_argument("--require-ask", action="store_true")
    p.add_argument("--debug", action="store_true", help="Write per-market audit CSV showing q/EV and drop reason")
    args = p.parse_args()

    debug_enabled = bool(args.debug) or os.getenv("WEATHER_DEBUG", "").strip().lower() in {"1", "true", "yes", "y"}
    configure_logging(
        name="weather_compute_edges",
        log_dir=ROOT / "weather" / "outputs" / "logs",
        debug=debug_enabled,
    )
    if debug_enabled:
        logger.info("compute_edges debug enabled (will write per-market audit CSV)")

    fee_cents = int(str(args.fee_cents).strip())
    fees = FeeSchedule(open_fee_cents=fee_cents)
    fee_open = fees.open_fee_dollars()

    top_n = int(str(args.top_n).strip())
    min_ev = float(str(args.min_ev).strip())
    min_q = _clamp(float(str(args.min_q).strip()), 0.0, 0.49)
    # support-tail-mass: prefer new flag, fall back to deprecated tail-mass, default disabled
    stm = _parse_optional_float(args.support_tail_mass)
    if stm is None:
        stm = _parse_optional_float(args.tail_mass)
    support_tail_mass: Optional[float] = None
    if stm is not None:
        support_tail_mass = _clamp(float(stm), 0.0, 0.98)
    min_volume = int(str(args.min_volume).strip())
    if min_volume < 0:
        min_volume = 0
    min_open_interest = int(str(args.min_open_interest).strip())
    if min_open_interest < 0:
        min_open_interest = 0
    max_spread_raw = str(args.max_spread_cents).strip()
    max_spread_cents: Optional[int] = None
    if max_spread_raw != "":
        try:
            max_spread_cents = int(max_spread_raw)
        except Exception:
            # Treat invalid input as disabled rather than failing the run.
            max_spread_cents = None
    if max_spread_cents is not None and max_spread_cents < 0:
        max_spread_cents = None

    db_path = Path(args.db)
    cities = load_cities(Path(args.config))

    wrote = 0
    all_rows_for_md: List[Dict[str, Any]] = []
    summary_date_local: Optional[str] = None

    for c in cities:
        if args.city and c.key != args.city:
            continue
        audit_rows: List[Dict[str, Any]] = []

        now_local = _now_utc().astimezone(ZoneInfo(c.tz))
        target_date_local = args.date.strip() or now_local.date().isoformat()

        fair = _read_fair_artifact(args.forecast_source, target_date_local, c.key)
        if not fair:
            continue
        # Pick a consistent folder for SUMMARY.md: the first city's local target date we actually process.
        if summary_date_local is None:
            summary_date_local = target_date_local

        # Support-overlap guardrail is OPTIONAL. If disabled, we do not drop based on overlap.
        snap_time = db_lib.fetch_latest_kalshi_weather_snapshot_time(
            db_path,
            city_key=c.key,
            target_date_local=target_date_local,
        )
        if not snap_time:
            continue

        markets = list(
            db_lib.fetch_kalshi_weather_markets_at_snapshot(
                db_path,
                snapshot_time_utc=snap_time,
                city_key=c.key,
                target_date_local=target_date_local,
            )
        )
        if not markets:
            if debug_enabled:
                audit_rows.append(
                    {"city_key": c.key, "reason": "no_markets_at_snapshot", "snapshot_time_utc": snap_time}
                )
            markets = []

        scored: List[Dict[str, Any]] = []
        drops = {
            "no_title": 0,
            "unparsed": 0,
            "no_prices": 0,
            "no_positive_ev": 0,
            "no_pmf": 0,
            # strike doesn't overlap modeled support interval (blueprint "weird strikes outside PMF support")
            "outside_support": 0,
            # model probability in tails (q < 5% or q > 95% by default)
            "q_tail": 0,
            "min_volume": 0,
            "min_open_interest": 0,
            "max_spread_missing": 0,
            "max_spread_too_wide": 0,
        }

        pmf_raw = fair.get("pmf_high_f") or {}
        pmf: Dict[int, float] = {}
        for k, v in pmf_raw.items():
            try:
                pmf[int(k)] = float(v)
            except Exception:
                continue
        if not pmf:
            drops["no_pmf"] += 1
            out = {
                "city_key": c.key,
                "label": c.label,
                "tz": c.tz,
                "target_date_local": target_date_local,
                "forecast_source": str(args.forecast_source),
                "fee": {"open_fee_cents": fee_cents, "open_fee_dollars": fee_open},
                "kalshi_snapshot_time_utc": snap_time,
                "filters": {
                    "min_ev": min_ev,
                    "min_q": min_q,
                    "q_interval": [min_q, 1.0 - min_q],
                    # Optional guardrail: support overlap filter
                    "support_tail_mass": support_tail_mass,
                    "support_overlap_enabled": False,
                    # retained for reference/debugging
                    "tail_prob_min": 0.0,
                    "support_interval_f": [0, 0],
                    "min_volume": min_volume,
                    "min_open_interest": min_open_interest,
                    "max_spread_cents": max_spread_cents,
                },
                "drops": drops,
                "candidates": [],
            }
            src = str(args.forecast_source).replace("/", "_")
            out_dir = OUT_BASE / src / target_date_local
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"{c.key}.json").write_text(
                json.dumps(out, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            wrote += 1
            continue

        # Support-overlap guardrail is OPTIONAL. If disabled, we do not drop based on overlap.
        if support_tail_mass is not None and support_tail_mass > 0.0:
            support_lo, support_hi = _central_support_interval(pmf, tail_mass=float(support_tail_mass))
            support_enabled = True
        else:
            support_lo, support_hi = _pmf_min_max(pmf)
            support_enabled = False
        tail_prob_min = max(0.0, min(0.49, float(support_tail_mass or 0.0) / 2.0))  # metadata only
        q_lo = float(min_q)
        q_hi = float(1.0 - min_q)

        for row in markets:
            drop_reason = ""
            # liquidity filters (raw markets)
            vol = _to_int(row.get("volume"), default=0)
            oi = _to_int(row.get("open_interest"), default=0)
            if min_volume > 0 and vol < min_volume:
                drops["min_volume"] += 1
                drop_reason = "min_volume"
            if min_open_interest > 0 and oi < min_open_interest:
                drops["min_open_interest"] += 1
                drop_reason = drop_reason or "min_open_interest"

            raw = row.get("raw") if isinstance(row.get("raw"), dict) else {}
            title = _market_title_from_raw(raw)
            if not title:
                drops["no_title"] += 1
                drop_reason = drop_reason or "no_title"

            spec = parse_event_spec_from_title(title)
            if not spec:
                drops["unparsed"] += 1
                drop_reason = drop_reason or "unparsed"

            q = None
            if spec:
                q = float(prob_event(pmf, spec))
            # OPTIONAL guardrail: ignore strikes that do not overlap modeled support.
            # Default: disabled to avoid double-filtering beyond the min_q policy.
            if support_enabled:
                if spec and (not _strike_overlaps_support(spec.kind, spec.a, spec.b, support_lo, support_hi)):
                    drops["outside_support"] += 1
                    drop_reason = drop_reason or "outside_support"
            # Hard tail cutoff: ignore events with <5% chance (or >95% on the other tail).
            if q is not None and (q < q_lo or q > q_hi):
                drops["q_tail"] += 1
                drop_reason = drop_reason or "q_tail"

            p_yes, p_no = best_buy_prices_from_snapshot_row(row)

            if args.require_ask and (p_yes is None and p_no is None):
                drops["no_prices"] += 1
                drop_reason = drop_reason or "no_prices"

            best_side = None
            best_ev = -1e9
            chosen_price = None
            ev_yes_val = None
            ev_no_val = None

            if p_yes is not None and q is not None:
                ev = ev_yes(q, float(p_yes), fee_open)
                ev_yes_val = float(ev)
                if ev > best_ev:
                    best_ev = ev
                    best_side = "YES"
                    chosen_price = float(p_yes)

            if p_no is not None and q is not None:
                ev = ev_no(q, float(p_no), fee_open)
                ev_no_val = float(ev)
                if ev > best_ev:
                    best_ev = ev
                    best_side = "NO"
                    chosen_price = float(p_no)

            if best_side is None or chosen_price is None:
                drops["no_prices"] += 1
                drop_reason = drop_reason or "no_prices"

            # spread filter applied to chosen side (requires bid + ask on that side)
            if max_spread_cents is not None:
                sp = _spread_cents_for_side(row, best_side)
                if sp is None:
                    drops["max_spread_missing"] += 1
                    drop_reason = drop_reason or "max_spread_missing"
                if sp > max_spread_cents:
                    drops["max_spread_too_wide"] += 1
                    drop_reason = drop_reason or "max_spread_too_wide"

            if q is not None and best_ev < min_ev:
                drops["no_positive_ev"] += 1
                drop_reason = drop_reason or "no_positive_ev"

            if debug_enabled:
                audit_rows.append(
                    {
                        "city_key": c.key,
                        "target_date_local": target_date_local,
                        "market_ticker": row.get("market_ticker"),
                        "title": title,
                        "snapshot_time_utc": row.get("snapshot_time_utc"),
                        "volume": vol,
                        "open_interest": oi,
                        "q": q,
                        "yes_ask": p_yes,
                        "no_ask": p_no,
                        "ev_yes": ev_yes_val,
                        "ev_no": ev_no_val,
                        "best_side": best_side,
                        "best_ev": float(best_ev) if best_ev > -1e8 else None,
                        "drop_reason": drop_reason,
                    }
                )

            if drop_reason:
                continue

            scored.append(
                {
                    "market_ticker": row.get("market_ticker"),
                    "title": title,
                    "event": {"kind": spec.kind, "a": spec.a, "b": spec.b, "desc": spec.describe()},
                    "model_q": float(q),
                    "prices": {"yes_ask": p_yes, "no_ask": p_no},
                    "decision": {
                        "side_to_buy": best_side,
                        "buy_price": chosen_price,
                        "fee_open": fee_open,
                        "ev": best_ev,
                    },
                    "liquidity": {"volume": vol, "open_interest": oi, "support_lo": support_lo, "support_hi": support_hi},
                    "snapshot": {
                        "snapshot_time_utc": row.get("snapshot_time_utc"),
                        "series_ticker": row.get("series_ticker"),
                        "event_ticker": row.get("event_ticker"),
                        "status": row.get("status"),
                    },
                }
            )

        scored.sort(key=lambda r: float(((r.get("decision") or {}).get("ev") or 0.0)), reverse=True)
        scored = scored[: max(1, top_n)]

        out = {
            "city_key": c.key,
            "label": c.label,
            "tz": c.tz,
            "target_date_local": target_date_local,
            "forecast_source": str(args.forecast_source),
            "fee": {"open_fee_cents": fee_cents, "open_fee_dollars": fee_open},
            "kalshi_snapshot_time_utc": snap_time,
            "filters": {
                "min_ev": min_ev,
                "min_q": min_q,
                "q_interval": [q_lo, q_hi],
                # Optional guardrail: support overlap filter
                "support_tail_mass": support_tail_mass,
                "support_overlap_enabled": support_enabled,
                # retained for reference/debugging
                "tail_prob_min": tail_prob_min,
                "support_interval_f": [support_lo, support_hi],
                "min_volume": min_volume,
                "min_open_interest": min_open_interest,
                "max_spread_cents": max_spread_cents,
            },
            "drops": drops,
            "candidates": scored,
        }

        src = str(args.forecast_source).replace("/", "_")
        out_dir = OUT_BASE / src / target_date_local
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{c.key}.json").write_text(
            json.dumps(out, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if debug_enabled:
            audit_path = out_dir / f"{c.key}_audit.csv"
            cols = [
                "city_key",
                "target_date_local",
                "market_ticker",
                "title",
                "snapshot_time_utc",
                "volume",
                "open_interest",
                "q",
                "yes_ask",
                "no_ask",
                "ev_yes",
                "ev_no",
                "best_side",
                "best_ev",
                "drop_reason",
            ]
            with audit_path.open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=cols)
                w.writeheader()
                for r in audit_rows:
                    w.writerow({k: r.get(k) for k in cols})
            logger.info("Wrote audit CSV: %s (%d rows)", audit_path, len(audit_rows))
        wrote += 1

        for r in scored:
            all_rows_for_md.append(
                {
                    "city": c.key,
                    "title": r.get("title", ""),
                    "side": (r.get("decision") or {}).get("side_to_buy", ""),
                    "ev": (r.get("decision") or {}).get("ev", 0.0),
                    "price": (r.get("decision") or {}).get("buy_price", 0.0),
                    "q": r.get("model_q", 0.0),
                }
            )

    if wrote > 0 and all_rows_for_md:
        all_rows_for_md.sort(key=lambda r: float(r.get("ev", 0.0)), reverse=True)
        src = str(args.forecast_source).replace("/", "_")
        # Keep SUMMARY.md alongside per-city artifacts (same date folder).
        target_date = args.date.strip() or (summary_date_local or _now_utc().date().isoformat())
        out_dir = OUT_BASE / src / str(target_date)
        out_dir.mkdir(parents=True, exist_ok=True)
        md_lines = [
            f"# Weather edges — {target_date} ({src})",
            "",
            (
                f"Fee (open): {fee_cents}¢ (${fee_open:.2f}) | min EV: {min_ev:.2f} | min_q: {min_q:.2f} | "
                f"support_tail_mass: {support_tail_mass} | support_overlap: {support_enabled} | min_vol: {min_volume} | "
                f"min_oi: {min_open_interest} | max_spread_cents: {max_spread_cents} | top_n per city: {top_n}"
            ),
            "",
            "| City | Side | EV ($/contract) | Price | q(model) | Market |",
            "|---|---|---:|---:|---:|---|",
        ]
        for r in all_rows_for_md[:50]:
            md_lines.append(
                "| {city} | {side} | {ev:.3f} | {price:.2f} | {q:.3f} | {title} |".format(
                    city=r["city"],
                    side=r["side"],
                    ev=float(r["ev"]),
                    price=float(r["price"]),
                    q=float(r["q"]),
                    title=str(r["title"]).replace("|", "\\|"),
                )
            )
        (out_dir / "SUMMARY.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"[done] wrote {wrote} edge artifacts under {OUT_BASE}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
