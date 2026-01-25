#!/usr/bin/env python3
"""Debug script to show all Kalshi weather markets with fair values and edges.

Posts separate Discord messages per city showing ALL markets regardless of edge value.
Useful for debugging when you want to see what data is available.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import requests
import yaml

from weather.lib import db as db_lib
from weather.lib.fees import FeeSchedule, ev_no, ev_yes
from weather.lib.kalshi_weather import (
    best_bid_ask_from_snapshot_row,
    best_buy_prices_from_snapshot_row,
    parse_event_spec_from_title,
    prob_event,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "weather" / "config" / "cities.yml"
DEFAULT_DB = Path(os.getenv("WEATHER_DB_PATH", str(ROOT / "weather" / "data" / "weather_forecast_accuracy.db")))
FAIR_BASE = ROOT / "weather" / "outputs" / "fair_prices"


@dataclass(frozen=True)
class City:
    key: str
    label: str
    tz: str


def _now_utc() -> datetime:
    return datetime.now(tz=ZoneInfo("UTC"))


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


def _prob_to_american_odds(p: float) -> str:
    """Convert probability to American odds string."""
    if not (0.0 < p < 1.0):
        return "N/A"
    if p == 0.5:
        return "+100"
    if p > 0.5:
        odds = -100.0 * p / (1.0 - p)
    else:
        odds = 100.0 * (1.0 - p) / p
    return f"{int(round(odds)):+d}" if odds > 0 else str(int(round(odds)))


def _cents_to_prob(cents: int) -> float:
    """Convert cents (0-100) to probability."""
    return max(0.01, min(0.99, cents / 100.0))


def _event_display(kind: str, a: Any, b: Any) -> str:
    """Format event for display."""
    k = (kind or "").strip().lower()
    try:
        a_i = int(a) if a is not None else None
        b_i = int(b) if b is not None else None
    except Exception:
        return str(a)

    if k in ("between", "range") and a_i is not None and b_i is not None:
        return f"{min(a_i, b_i)}-{max(a_i, b_i)}°F"
    if k in ("ge", "gte") and a_i is not None:
        return f"≥{a_i}°F"
    if k in ("le", "lte") and a_i is not None:
        return f"≤{a_i}°F"
    if k == "exact" and a_i is not None:
        return f"={a_i}°F"
    return f"{a_i}°F" if a_i else "?"


def get_forecast_for_city(db_path: Path, city_key: str, target_date: str, source: str) -> Optional[int]:
    """Get forecast high for a city on a given date."""
    snap = db_lib.fetch_latest_forecast_snapshot(db_path, city_key=city_key, target_date_local=target_date, source=source)
    if snap:
        return int(snap.get("forecast_high_f", 0))
    return None


def _read_fair_artifact(forecast_source: str, target_date_local: str, city_key: str) -> Optional[Dict[str, Any]]:
    """Read fair price artifact from disk (computed by compute_fair_prices.py)."""
    src = str(forecast_source).replace("/", "_")
    path = FAIR_BASE / src / target_date_local / f"{city_key}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _market_title_from_raw(raw: Dict[str, Any]) -> str:
    """Extract market title from raw JSON."""
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


def build_simple_pmf(forecast_high: int, std: float = 4.0, min_tail_prob: float = 0.01) -> Dict[int, float]:
    """Build a simple Gaussian PMF centered on forecast high (fallback if no fair artifact)."""
    import math

    center = forecast_high
    denom = 2.0 * (std ** 2)

    def gaussian_density(k: int) -> float:
        return math.exp(-((float(k) - center) ** 2) / denom)

    out: Dict[int, float] = {}

    # Extend in both directions until tail < min_tail_prob
    for direction in [-1, 1]:
        k = center if direction == -1 else center + 1
        while True:
            density = gaussian_density(k)
            out[k] = density
            total = sum(out.values())
            if direction == -1:
                tail = sum(out[j] for j in out if j <= k) / total
            else:
                tail = sum(out[j] for j in out if j >= k) / total
            if tail < min_tail_prob:
                break
            k += direction
            if abs(k - center) > 30:
                break

    # Normalize
    total = sum(out.values())
    return {k: v / total for k, v in sorted(out.items())}


def format_city_message(
    city: City,
    forecast_high: Optional[int],
    markets: List[Dict[str, Any]],
    pmf: Dict[int, float],
    fee_cents: int,
    target_date: str,
) -> str:
    """Format a Discord message for a single city showing ALL markets."""
    lines = [f"**{city.label}** — {target_date}"]

    if forecast_high:
        lines.append(f"Forecast High: {forecast_high}°F")
    else:
        lines.append("Forecast High: N/A (using default)")

    lines.append(f"Fee: {fee_cents}¢")
    lines.append("")

    if not markets:
        lines.append("_No markets found_")
        return "\n".join(lines)

    fee_open = fee_cents / 100.0

    # Collect all market data for sorting
    market_rows: List[Dict[str, Any]] = []

    for m in markets:
        raw = m.get("raw") if isinstance(m.get("raw"), dict) else {}
        title = _market_title_from_raw(raw) or m.get("market_ticker") or "Unknown"
        market_ticker = m.get("market_ticker") or ""

        # Strip markdown formatting from title before parsing
        clean_title = title.replace("**", "").replace("*", "")

        # Parse event spec from title
        spec = parse_event_spec_from_title(clean_title)
        if not spec:
            # Still show unparseable markets in debug mode
            market_rows.append({
                "event_str": clean_title[:30] + "..." if len(clean_title) > 30 else clean_title,
                "market_ticker": market_ticker,
                "q": None,
                "best_side": None,
                "best_ev": None,
                "buy_price": None,
                "yes_bid": m.get("yes_bid"),
                "yes_ask": m.get("yes_ask"),
                "no_bid": m.get("no_bid"),
                "no_ask": m.get("no_ask"),
                "status": "unparsed",
            })
            continue

        event_str = _event_display(spec.kind, spec.a, spec.b)

        # Get prices from snapshot row
        yes_bid, yes_ask, no_bid, no_ask = best_bid_ask_from_snapshot_row(m)

        # Calculate fair probability (use actual model value, even if 0%)
        q = prob_event(pmf, spec) or 0.0

        # Calculate EV for both sides
        ev_y = ev_yes(q, yes_ask / 100.0, fee_open) if yes_ask else None
        ev_n = ev_no(q, no_ask / 100.0, fee_open) if no_ask else None

        # Determine best side
        best_side = None
        best_ev = None
        buy_price = None

        if ev_y is not None and (ev_n is None or ev_y >= ev_n):
            best_side = "YES"
            best_ev = ev_y
            buy_price = yes_ask
        elif ev_n is not None:
            best_side = "NO"
            best_ev = ev_n
            buy_price = no_ask

        market_rows.append({
            "event_str": event_str,
            "market_ticker": market_ticker,
            "q": q,
            "best_side": best_side,
            "best_ev": best_ev,
            "buy_price": buy_price,
            "yes_bid": yes_bid,
            "yes_ask": yes_ask,
            "no_bid": no_bid,
            "no_ask": no_ask,
            "status": "ok" if best_ev is not None else "no_price",
        })

    # Sort by EV descending (None values at end)
    market_rows.sort(key=lambda r: (r["best_ev"] is not None, r["best_ev"] or -999), reverse=True)

    for row in market_rows:
        event_str = row["event_str"]
        market_ticker = row.get("market_ticker", "")
        q = row["q"]
        best_side = row["best_side"]
        best_ev = row["best_ev"]
        buy_price = row["buy_price"]
        yes_bid = row["yes_bid"]
        yes_ask = row["yes_ask"]
        no_bid = row["no_bid"]
        no_ask = row["no_ask"]

        if row["status"] == "unparsed":
            lines.append(f"⚠️ **{event_str}** (unparsed)")
            if yes_ask is not None:
                lines.append(f"  YES: {yes_ask}¢")
            if no_ask is not None:
                lines.append(f"  NO: {no_ask}¢")
            if market_ticker:
                lines.append(f"  `{market_ticker}`")
            lines.append("")
            continue

        if best_ev is None:
            lines.append(f"⚫ **{event_str}** (no price)")
            lines.append(f"  Fair: {q*100:.1f}%")
            if market_ticker:
                lines.append(f"  `{market_ticker}`")
            lines.append("")
            continue

        # Format output
        fair_p = q if best_side == "YES" else (1.0 - q)
        fair_odds = _prob_to_american_odds(fair_p)
        kalshi_odds = _prob_to_american_odds(buy_price / 100.0) if buy_price else "N/A"

        ev_cents = best_ev * 100  # Convert dollars to cents
        if ev_cents >= 10:
            edge_emoji = "🟢"  # Large positive edge (10¢+)
        elif ev_cents >= 5:
            edge_emoji = "🟡"  # Small positive edge (5-10¢)
        elif ev_cents >= -5:
            edge_emoji = "⚪"  # Near zero (-5¢ to 5¢)
        else:
            edge_emoji = "🔴"  # Negative edge (< -5¢)

        lines.append(f"{edge_emoji} **{event_str}**")
        lines.append(f"  Fair: {q*100:.1f}% ({fair_odds}) | Kalshi {best_side}: {buy_price}¢ ({kalshi_odds})")
        lines.append(f"  EV: {ev_cents:+.1f}¢ per contract")

        # Show bid/ask spread
        if best_side == "YES" and yes_bid is not None and yes_ask is not None:
            spread = yes_ask - yes_bid
            lines.append(f"  Spread: {spread}¢ (bid {yes_bid}¢ / ask {yes_ask}¢)")
        elif best_side == "NO" and no_bid is not None and no_ask is not None:
            spread = no_ask - no_bid
            lines.append(f"  Spread: {spread}¢ (bid {no_bid}¢ / ask {no_ask}¢)")

        if market_ticker:
            lines.append(f"  `{market_ticker}`")

        lines.append("")

    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--db", default=str(DEFAULT_DB))
    p.add_argument("--date", default="", help="Target date YYYY-MM-DD (default: today)")
    p.add_argument("--forecast-source", default="nws_hourly_max")
    p.add_argument("--fee-cents", type=int, default=2, help="Trading fee in cents")
    p.add_argument("--default-forecast", type=int, default=70, help="Default forecast if none found")
    p.add_argument("--std", type=float, default=4.0, help="Std dev for fallback probability distribution")
    p.add_argument("--dry-run", action="store_true", help="Print to stdout instead of Discord")
    p.add_argument("--city", default="", help="Filter to a single city key")
    args = p.parse_args()

    webhook = (os.getenv("WEATHER_DISCORD_WEBHOOK") or os.getenv("DISCORD_WEBHOOK_URL") or "").strip()
    if not webhook and not args.dry_run:
        print("WEATHER_DISCORD_WEBHOOK not set; use --dry-run to print to stdout")
        return 1

    db_path = Path(args.db)
    cities = load_cities(Path(args.config))

    target_date = args.date.strip() or datetime.now(tz=ZoneInfo("America/New_York")).date().isoformat()

    print(f"[debug] Target date: {target_date}")
    print(f"[debug] Cities: {[c.key for c in cities]}")
    print(f"[debug] Fair price dir: {FAIR_BASE}")

    messages_sent = 0

    for city in cities:
        # Filter by city if specified
        if args.city and city.key != args.city:
            continue

        # Try to read fair price artifact first (preferred - uses error model)
        fair = _read_fair_artifact(args.forecast_source, target_date, city.key)
        if fair:
            pmf_raw = fair.get("pmf_high_f") or {}
            pmf: Dict[int, float] = {}
            for k, v in pmf_raw.items():
                try:
                    pmf[int(k)] = float(v)
                except Exception:
                    continue
            forecast_high = fair.get("forecast_high_f")
            print(f"[debug] {city.key}: Using fair price artifact (forecast {forecast_high}°F, PMF size {len(pmf)})")
        else:
            # Fallback: get forecast and build simple Gaussian PMF
            forecast_high = get_forecast_for_city(db_path, city.key, target_date, args.forecast_source)
            if not forecast_high:
                forecast_high = args.default_forecast
                print(f"[debug] {city.key}: No forecast found, using default {forecast_high}°F")
            else:
                print(f"[debug] {city.key}: Forecast {forecast_high}°F (using fallback Gaussian PMF)")
            pmf = build_simple_pmf(forecast_high, std=args.std)

        if not pmf:
            print(f"[warn] {city.key}: No PMF available, skipping")
            continue

        # Get latest Kalshi snapshot time for this city/date
        snap_time = db_lib.fetch_latest_kalshi_weather_snapshot_time(
            db_path,
            city_key=city.key,
            target_date_local=target_date,
        )
        if not snap_time:
            print(f"[debug] {city.key}: No Kalshi snapshot found")
            continue

        # Get Kalshi markets at that snapshot time
        markets = list(db_lib.fetch_kalshi_weather_markets_at_snapshot(
            db_path,
            snapshot_time_utc=snap_time,
            city_key=city.key,
            target_date_local=target_date,
        ))
        print(f"[debug] {city.key}: Found {len(markets)} markets at {snap_time}")

        if not markets:
            continue

        # Format message
        msg = format_city_message(city, forecast_high, markets, pmf, args.fee_cents, target_date)

        if args.dry_run:
            print("\n" + "=" * 50)
            print(msg)
            print("=" * 50)
        else:
            # Post to Discord
            if len(msg) > 1990:
                msg = msg[:1970] + "\n...(truncated)"

            payload = {"content": msg}
            r = requests.post(webhook, json=payload, timeout=20)
            if r.status_code >= 300:
                print(f"[error] Discord post failed for {city.key}: HTTP {r.status_code}")
            else:
                print(f"[ok] Posted {city.key} to Discord")
                messages_sent += 1

    print(f"[done] Sent {messages_sent} messages to Discord")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
