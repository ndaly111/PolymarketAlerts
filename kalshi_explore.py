#!/usr/bin/env python3
"""Explore Kalshi Weather/Sports data shapes.

This script is intentionally *schema-first*:
- pull a small sample of Series → Events → Markets (+ optional Orderbooks)
- write raw JSON (so nothing is lost)
- write flat CSVs + an overview (so it’s easy to reason about matching & pricing)

Run locally:
  python kalshi_explore.py --category Weather --depth quick

Or use the included GitHub Actions workflow: .github/workflows/kalshi_explore.yml
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from kalshi_explore_client import KalshiClient, paginate


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_json(path: str, obj: Any) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, sort_keys=True)


def write_csv(path: str, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: _safe_str(r.get(k, "")) for k in fieldnames})


def _safe_str(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False, sort_keys=True)
    return str(v)


def _compact_text(v: Any, max_len: int = 600) -> str:
    s = _safe_str(v)
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def _trim_metadata(pm: Dict[str, Any], max_keys: int = 12) -> Dict[str, Any]:
    if not isinstance(pm, dict):
        return {}
    return dict(list(pm.items())[:max_keys])


def _as_list_csv(s: str) -> List[str]:
    s = (s or "").strip()
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def _pm_keys_counter(rows: Iterable[Dict[str, Any]], pm_field: str = "product_metadata") -> Counter:
    c: Counter = Counter()
    for r in rows:
        pm = r.get(pm_field) or {}
        if isinstance(pm, dict):
            c.update(pm.keys())
    return c


def _extract_tags_map(tags_by_categories: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(tags_by_categories, dict):
        return {}
    for key in ("tags_by_categories", "data"):
        if isinstance(tags_by_categories.get(key), dict):
            return tags_by_categories[key]
    return tags_by_categories


def _choose_category_ticker(
    tags_by_categories: Dict[str, Any],
    want_name: str,
) -> Tuple[Optional[str], List[str]]:
    tags_map = _extract_tags_map(tags_by_categories)
    available = list(tags_map.keys()) if isinstance(tags_map, dict) else []
    want = (want_name or "").strip().lower()
    if not available:
        return None, []
    if want == "sports":
        for key in available:
            if "sport" in key.lower():
                return key, available
    if want == "weather":
        for key in available:
            if "weather" in key.lower() or "climate" in key.lower():
                return key, available
    return available[0], available


def _cents_to_dollars(value: Any) -> Optional[str]:
    try:
        cents = float(value)
    except (TypeError, ValueError):
        return None
    return f"{cents / 100:.4f}"


def fetch_discovery(client: KalshiClient, out_dir: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    tags_by_categories: Dict[str, Any] = {}
    filters_by_sport: Dict[str, Any] = {}

    try:
        code, data = client.get("/search/tags_by_categories")
        if code == 200:
            tags_by_categories = data or {}
        else:
            tags_by_categories = {"_error": f"status={code}", "_response": data}
    except Exception as e:
        tags_by_categories = {"_error": f"{type(e).__name__}: {e}"}

    try:
        code, data = client.get("/search/filters_by_sport")
        if code == 200:
            filters_by_sport = data or {}
        else:
            filters_by_sport = {"_error": f"status={code}", "_response": data}
    except Exception as e:
        filters_by_sport = {"_error": f"{type(e).__name__}: {e}"}

    write_json(os.path.join(out_dir, "tags_by_categories.json"), tags_by_categories)
    write_json(os.path.join(out_dir, "filters_by_sport.json"), filters_by_sport)
    return tags_by_categories, filters_by_sport


def list_series(
    client: KalshiClient,
    category_ticker: Optional[str],
    series_tickers: List[str],
    max_series: int,
) -> Tuple[List[Dict[str, Any]], List[str], Dict[str, Any]]:
    if series_tickers:
        out: List[Dict[str, Any]] = []
        errors: List[str] = []
        for t in series_tickers[:max_series]:
            try:
                code, data = client.get(f"/series/{t}")
                if code == 200:
                    out.append(data or {"ticker": t, "_error": "empty"})
                else:
                    out.append({"ticker": t, "_error": f"status={code}", "_response": data})
                    errors.append(f"/series/{t} status={code}")
            except Exception as e:
                out.append({"ticker": t, "_error": f"{type(e).__name__}: {e}"})
                errors.append(f"/series/{t} error={type(e).__name__}")
        return out, errors, {}

    params: Dict[str, Any] = {
        "limit": 200,
        "include_product_metadata": True,
        "include_volume": True,
    }
    if category_ticker:
        params["category"] = category_ticker
    items, meta, errors = paginate(client, "/series", "series", params, hard_limit=max_series)
    return items, errors, meta


def list_events_for_series(
    client: KalshiClient,
    series_ticker: str,
    status: str,
    max_events: int,
) -> Tuple[List[Dict[str, Any]], List[str], Dict[str, Any]]:
    params: Dict[str, Any] = {
        "series_ticker": series_ticker,
        "limit": min(200, max_events),
        "with_nested_markets": True,
    }
    if status and status.lower() != "any":
        params["status"] = status.lower()
    items, meta, errors = paginate(client, "/events", "events", params, hard_limit=max_events)
    return items, errors, meta


def markets_from_event_or_fallback(
    client: KalshiClient,
    event: Dict[str, Any],
    status: str,
    max_markets: int,
) -> Tuple[List[Dict[str, Any]], List[str], Dict[str, Any]]:
    nested = event.get("markets")
    if isinstance(nested, list) and nested:
        return nested[:max_markets], [], {}

    et = event.get("ticker") or event.get("event_ticker")
    if not et:
        return [], ["event missing ticker"], {}
    params: Dict[str, Any] = {"event_ticker": et, "limit": min(200, max_markets)}
    if status and status.lower() != "any":
        params["status"] = status.lower()
    items, meta, errors = paginate(client, "/markets", "markets", params, hard_limit=max_markets)
    return items, errors, meta


def fetch_orderbook_top(client: KalshiClient, market_ticker: str, depth: int) -> Dict[str, Any]:
    code, data = client.get(f"/markets/{market_ticker}/orderbook", params={"depth": depth})
    if code != 200:
        return {"market_ticker": market_ticker, "_error": f"status={code}", "_response": data}
    ob = data or {}
    orderbook = ob.get("orderbook") if isinstance(ob.get("orderbook"), dict) else {}
    orderbook_fp = ob.get("orderbook_fp") if isinstance(ob.get("orderbook_fp"), dict) else {}
    yes = orderbook.get("yes_dollars") or orderbook.get("yes") or ob.get("yes_dollars") or ob.get("yes")
    no = orderbook.get("no_dollars") or orderbook.get("no") or ob.get("no_dollars") or ob.get("no")
    yes_fp = orderbook_fp.get("yes") if isinstance(orderbook_fp.get("yes"), list) else None
    no_fp = orderbook_fp.get("no") if isinstance(orderbook_fp.get("no"), list) else None

    def best(levels: Any) -> Tuple[Optional[float], Optional[int]]:
        if not isinstance(levels, list) or not levels:
            return None, None
        try:
            p, q = max(levels, key=lambda x: float(x[0]))
            return float(p), int(q)
        except Exception:
            return None, None

    yes_bid, yes_bid_q = best(yes)
    no_bid, no_bid_q = best(no)
    yes_bid_fp, yes_bid_fp_q = best(yes_fp) if yes_fp else (None, None)
    no_bid_fp, no_bid_fp_q = best(no_fp) if no_fp else (None, None)

    yes_ask_implied = (1.0 - no_bid) if isinstance(no_bid, float) else None
    no_ask_implied = (1.0 - yes_bid) if isinstance(yes_bid, float) else None

    return {
        "market_ticker": market_ticker,
        "yes_bid_dollars": yes_bid,
        "yes_bid_qty": yes_bid_q,
        "yes_ask_implied_dollars": yes_ask_implied,
        "no_bid_dollars": no_bid,
        "no_bid_qty": no_bid_q,
        "no_ask_implied_dollars": no_ask_implied,
        "yes_bid_fp": yes_bid_fp,
        "yes_bid_fp_qty": yes_bid_fp_q,
        "no_bid_fp": no_bid_fp,
        "no_bid_fp_qty": no_bid_fp_q,
        "raw": ob,
    }


def flatten_series(s: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "series_ticker": s.get("ticker"),
        "title": s.get("title"),
        "category": s.get("category"),
        "tags": s.get("tags"),
        "volume": s.get("volume"),
        "product_metadata": s.get("product_metadata"),
    }


def flatten_event(e: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "event_ticker": e.get("event_ticker") or e.get("ticker"),
        "series_ticker": e.get("series_ticker"),
        "category": e.get("category"),
        "title": e.get("title"),
        "subtitle": e.get("sub_title") or e.get("subtitle"),
        "status": e.get("status"),
        "open_time": e.get("open_time"),
        "close_time": e.get("close_time"),
        "product_metadata": e.get("product_metadata"),
    }


def flatten_market(m: Dict[str, Any], event_ticker: Optional[str], series_ticker: Optional[str]) -> Dict[str, Any]:
    yes_bid_dollars = m.get("yes_bid_dollars") or _cents_to_dollars(m.get("yes_bid"))
    yes_ask_dollars = m.get("yes_ask_dollars") or _cents_to_dollars(m.get("yes_ask"))
    no_bid_dollars = m.get("no_bid_dollars") or _cents_to_dollars(m.get("no_bid"))
    no_ask_dollars = m.get("no_ask_dollars") or _cents_to_dollars(m.get("no_ask"))
    return {
        "market_ticker": m.get("ticker"),
        "event_ticker": m.get("event_ticker") or event_ticker,
        "series_ticker": m.get("series_ticker") or series_ticker,
        "title": m.get("title"),
        "subtitle": m.get("subtitle"),
        "status": m.get("status"),
        "yes_bid_dollars": yes_bid_dollars,
        "yes_ask_dollars": yes_ask_dollars,
        "no_bid_dollars": no_bid_dollars,
        "no_ask_dollars": no_ask_dollars,
        "response_price_units": m.get("response_price_units"),
        "volume": m.get("volume"),
        "open_interest": m.get("open_interest"),
        "product_metadata": m.get("product_metadata"),
        "rules_primary": _compact_text(m.get("rules_primary")),
        "payout_type": m.get("payout_type"),
    }


def render_overview(
    category: str,
    stamp: str,
    category_ticker: Optional[str],
    available_category_tickers: List[str],
    category_warning: Optional[str],
    series_param_category: Optional[str],
    warnings: List[str],
    series: List[Dict[str, Any]],
    events: List[Dict[str, Any]],
    markets: List[Dict[str, Any]],
    orderbooks_top: List[Dict[str, Any]],
    out_dir: str,
) -> None:
    lines: List[str] = []
    lines.append(f"# Kalshi Explore ({category})")
    lines.append("")
    lines.append(f"Generated (UTC): `{stamp}`")
    lines.append("")
    lines.append("## What you’ll use for matching + pricing")
    lines.append("- **series_ticker** groups related contracts (e.g., a city or a league).")
    lines.append("- **event_ticker** is the real-world instance (e.g., a specific game or a specific day’s weather).")
    lines.append("- **market_ticker** is the tradable instrument you price + trade.")
    lines.append("- Top-of-book pricing is usually on the market object (`*_dollars` fields).")
    lines.append("- Orderbook depth (optional): `orderbooks_raw.json` + `orderbooks_top.csv`.")
    lines.append("- Orderbooks do **not** include explicit asks; implied asks are labeled as such.")
    lines.append("")

    lines.append("## Outputs")
    lines.append("- `overview.md`: this file")
    lines.append("- `schema_summary.json`: which `product_metadata` keys show up (most important for matching)")
    lines.append("- `series_flat.csv`, `events_flat.csv`, `markets_flat.csv`, `orderbooks_top.csv`")
    lines.append("- Raw dumps: `*_raw.json`, `tags_by_categories.json`, `filters_by_sport.json`")
    lines.append("")

    lines.append("## Sample size")
    lines.append(f"- Series: {len(series)}")
    lines.append(f"- Events: {len(events)}")
    lines.append(f"- Markets: {len(markets)}")
    lines.append(f"- Orderbooks fetched: {len(orderbooks_top)}")
    lines.append("")

    lines.append("## Series category selection")
    lines.append(f"- Selected category ticker: `{category_ticker or 'unknown'}`")
    lines.append(f"- Available category tickers: {', '.join(available_category_tickers) or '(none)'}")
    lines.append(f"- Category param sent to /series: `{series_param_category or '(none)'}`")
    if category_warning:
        lines.append(f"- Warning: {category_warning}")
    lines.append("")

    if series:
        lines.append("## Sample series (sanity check)")
        for s in series[:5]:
            lines.append(f"- {s.get('ticker')}: {s.get('title')}")
        lines.append("")

    s_keys = _pm_keys_counter(series)
    e_keys = _pm_keys_counter(events)
    m_keys = _pm_keys_counter(markets)

    def top_keys(c: Counter, n: int = 25) -> str:
        items = c.most_common(n)
        if not items:
            return "(none)"
        return ", ".join([f"{k}({v})" for k, v in items])

    lines.append("## Most common `product_metadata` keys")
    lines.append(f"- Series: {top_keys(s_keys)}")
    lines.append(f"- Events: {top_keys(e_keys)}")
    lines.append(f"- Markets: {top_keys(m_keys)}")
    lines.append("")

    def example_metadata(rows: List[Dict[str, Any]], label: str) -> None:
        for r in rows:
            pm = r.get("product_metadata")
            if pm:
                lines.append(f"### Example {label} product_metadata")
                lines.append("```json")
                lines.append(json.dumps(_trim_metadata(pm), indent=2, ensure_ascii=False))
                lines.append("```")
                lines.append("")
                return
        lines.append(f"### Example {label} product_metadata")
        lines.append("(none found)")
        lines.append("")

    example_metadata(series, "series")
    example_metadata(events, "events")
    example_metadata(markets, "markets")

    def example_event_raw(rows: List[Dict[str, Any]]) -> None:
        if not rows:
            lines.append("## Example event (raw)")
            lines.append("(none found)")
            lines.append("")
            return
        sample = dict(list(rows[0].items())[:20])
        lines.append("## Example event (raw)")
        lines.append("```json")
        lines.append(json.dumps(sample, indent=2, ensure_ascii=False))
        lines.append("```")
        lines.append("")

    example_event_raw(events)

    def recommend_keys(c: Counter, total: int) -> List[str]:
        if total <= 0:
            return []
        likely = []
        for key, count in c.items():
            if count / total >= 0.8:
                if any(token in key.lower() for token in ("id", "team", "league", "city", "station", "date")):
                    likely.append(key)
        return sorted(likely)

    lines.append("## Recommended match keys (data-driven)")
    lines.append(
        f"- Series: {', '.join(recommend_keys(s_keys, len(series))) or '(insufficient coverage)'}"
    )
    lines.append(
        f"- Events: {', '.join(recommend_keys(e_keys, len(events))) or '(insufficient coverage)'}"
    )
    lines.append(
        f"- Markets: {', '.join(recommend_keys(m_keys, len(markets))) or '(insufficient coverage)'}"
    )
    lines.append("")

    lines.append("## Next step (how this makes your scanners work)")
    lines.append("1. Open `events_flat.csv` and look at the `product_metadata` JSON.")
    lines.append("2. Pick the smallest set of stable keys that uniquely identify the thing you’re matching:")
    lines.append("   - Sports: home/away/team IDs + league/competition + start time/date")
    lines.append("   - Weather: location/city ID + date + threshold (if any)")
    lines.append("3. Use those keys in your glossary/matcher instead of parsing titles.")
    lines.append("4. `/events?with_nested_markets=true` is the best shape pull to model series→event→market.")
    if warnings:
        lines.append("")
        lines.append("## Warnings")
        for warning in warnings:
            lines.append(f"- {warning}")
        lines.append("")

    path = os.path.join(out_dir, "overview.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _self_check_client(client: KalshiClient) -> Optional[str]:
    try:
        status, data = client.get("/series", params={"limit": 1})
    except Exception as exc:
        return f"Self-check failed: {type(exc).__name__}: {exc}"
    if not isinstance(status, int):
        return "Self-check failed: client.get did not return a status code."
    if not isinstance(data, dict):
        return "Self-check failed: client.get did not return a dict payload."
    if status != 200:
        return f"Self-check warning: /series returned status {status}."
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", choices=["Weather", "Sports"], default="Weather")
    ap.add_argument("--depth", choices=["quick", "deep"], default="quick")
    ap.add_argument("--status", default="open", help="open/closed/settled/any")
    ap.add_argument("--series-tickers", default="", help="Comma-separated series tickers to restrict")
    ap.add_argument("--max-series", type=int, default=5)
    ap.add_argument("--max-events-per-series", type=int, default=6)
    ap.add_argument("--max-markets-per-event", type=int, default=25)
    ap.add_argument("--max-orderbooks", type=int, default=50)
    ap.add_argument("--category-param", default="", help="Exact /series category param override")
    ap.add_argument("--auth-mode", choices=["public", "signed"], default="public")
    ap.add_argument("--orderbook-depth", type=int, default=1)
    ap.add_argument("--out-dir", default="artifacts/kalshi_explore")
    args = ap.parse_args()

    if args.depth == "quick":
        args.max_series = min(args.max_series, 5)
        args.max_events_per_series = min(args.max_events_per_series, 6)
        args.max_markets_per_event = min(args.max_markets_per_event, 25)
        args.max_orderbooks = min(args.max_orderbooks, 50)
    else:
        args.max_series = min(args.max_series, 20)
        args.max_events_per_series = min(args.max_events_per_series, 40)
        args.max_markets_per_event = min(args.max_markets_per_event, 200)
        args.max_orderbooks = min(args.max_orderbooks, 300)

    out_dir = args.out_dir
    ensure_dir(out_dir)
    stamp = utc_stamp()

    if args.auth_mode == "public":
        client = KalshiClient.from_env(auth_mode="public")
    else:
        raise SystemExit(
            "Signed auth not supported in this minimal client yet "
            "(requires RSA-PSS with millisecond timestamps)."
        )

    self_check_warning = _self_check_client(client)
    tags_by_categories, _ = fetch_discovery(client, out_dir)
    category_ticker, available_category_tickers = _choose_category_ticker(tags_by_categories, args.category)
    category_warning = None
    series_param_category = args.category_param.strip() or category_ticker
    if not series_param_category and not args.category_param:
        raise SystemExit(
            "No category parameter could be determined.\n"
            f"Available category keys: {', '.join(available_category_tickers) or '(none)'}\n"
            "Provide --category-param to set the exact /series category."
        )
    if not category_ticker:
        category_warning = "No category tickers discovered; category selection may be unreliable."
    elif (
        args.category.lower() == "sports"
        and "sport" not in category_ticker.lower()
        or args.category.lower() == "weather"
        and "weather" not in category_ticker.lower()
        and "climate" not in category_ticker.lower()
    ):
        category_warning = f"Category ticker auto-match may be incorrect: {category_ticker}"

    restrict_series = _as_list_csv(args.series_tickers)
    series, series_errors, series_meta = list_series(
        client,
        series_param_category,
        restrict_series,
        args.max_series,
    )
    write_json(os.path.join(out_dir, "series_raw.json"), series)
    series_flat = [flatten_series(s) for s in series]
    write_csv(
        os.path.join(out_dir, "series_flat.csv"),
        series_flat,
        ["series_ticker", "title", "category", "tags", "volume", "product_metadata"],
    )

    all_events: List[Dict[str, Any]] = []
    all_markets: List[Dict[str, Any]] = []

    pagination_warnings: List[str] = []
    for s in series:
        st = s.get("ticker")
        if not st:
            continue
        evs, ev_errors, ev_meta = list_events_for_series(client, st, args.status, args.max_events_per_series)
        all_events.extend(evs)
        if ev_errors:
            pagination_warnings.append(f"/events series={st} errors={ev_errors} meta={ev_meta}")
        for e in evs:
            ms, m_errors, m_meta = markets_from_event_or_fallback(
                client, e, args.status, args.max_markets_per_event
            )
            for m in ms:
                m.setdefault("event_ticker", e.get("event_ticker") or e.get("ticker"))
                m.setdefault("series_ticker", st)
            all_markets.extend(ms)
            if m_errors:
                pagination_warnings.append(
                    f"/markets event={e.get('event_ticker') or e.get('ticker')} errors={m_errors} "
                    f"meta={m_meta}"
                )

    write_json(os.path.join(out_dir, "events_raw.json"), all_events)
    write_json(os.path.join(out_dir, "markets_raw.json"), all_markets)

    events_flat = [flatten_event(e) for e in all_events]
    write_csv(
        os.path.join(out_dir, "events_flat.csv"),
        events_flat,
        [
            "event_ticker",
            "series_ticker",
            "category",
            "title",
            "subtitle",
            "status",
            "open_time",
            "close_time",
            "product_metadata",
        ],
    )

    markets_flat = [
        flatten_market(m, m.get("event_ticker"), m.get("series_ticker")) for m in all_markets
    ]
    write_csv(
        os.path.join(out_dir, "markets_flat.csv"),
        markets_flat,
        [
            "market_ticker",
            "event_ticker",
            "series_ticker",
            "title",
            "subtitle",
            "status",
            "yes_bid_dollars",
            "yes_ask_dollars",
            "no_bid_dollars",
            "no_ask_dollars",
            "response_price_units",
            "volume",
            "open_interest",
            "payout_type",
            "rules_primary",
            "product_metadata",
        ],
    )

    orderbooks_top: List[Dict[str, Any]] = []
    orderbooks_raw: List[Dict[str, Any]] = []
    orderbook_warning: Optional[str] = None
    if args.max_orderbooks > 0:
        for m in all_markets:
            mt = m.get("ticker")
            if not mt:
                continue
            if len(orderbooks_top) >= args.max_orderbooks:
                break
            try:
                top = fetch_orderbook_top(client, mt, args.orderbook_depth)
                if isinstance(top.get("_error"), str) and top.get("_error", "").startswith("status=429"):
                    orderbook_warning = f"Received 429 rate limit at market {mt}; stopped fetching orderbooks."
                    break
                if isinstance(top.get("_error"), str) and top.get("_error", "").startswith("status=401"):
                    orderbook_warning = "Orderbooks require auth; run with auth_mode=signed once implemented."
                    break
                if isinstance(top.get("_error"), str) and top.get("_error", "").startswith("status=403"):
                    orderbook_warning = "Orderbooks require auth; run with auth_mode=signed once implemented."
                    break
                orderbooks_top.append({k: v for k, v in top.items() if k != "raw"})
                orderbooks_raw.append(top)
            except Exception as e:
                orderbooks_top.append({"market_ticker": mt, "_error": f"{type(e).__name__}: {e}"})
            time.sleep(0.2)

    write_json(os.path.join(out_dir, "orderbooks_raw.json"), orderbooks_raw)
    write_csv(
        os.path.join(out_dir, "orderbooks_top.csv"),
        orderbooks_top,
        [
            "market_ticker",
            "yes_bid_dollars",
            "yes_bid_qty",
            "yes_ask_implied_dollars",
            "no_bid_dollars",
            "no_bid_qty",
            "no_ask_implied_dollars",
            "yes_bid_fp",
            "yes_bid_fp_qty",
            "no_bid_fp",
            "no_bid_fp_qty",
        ],
    )

    schema_summary = {
        "generated_utc": stamp,
        "category": args.category,
        "series_product_metadata_keys": _pm_keys_counter(series).most_common(),
        "events_product_metadata_keys": _pm_keys_counter(all_events).most_common(),
        "markets_product_metadata_keys": _pm_keys_counter(all_markets).most_common(),
    }
    write_json(os.path.join(out_dir, "schema_summary.json"), schema_summary)

    warning_list = [warning for warning in (category_warning, self_check_warning, orderbook_warning) if warning]
    warning_list.extend(pagination_warnings)
    if series_errors:
        warning_list.append(f"/series errors={series_errors} meta={series_meta}")
    render_overview(
        args.category,
        stamp,
        category_ticker,
        available_category_tickers,
        category_warning,
        series_param_category,
        warning_list,
        series,
        all_events,
        all_markets,
        orderbooks_top,
        out_dir,
    )

    print(f"Wrote Kalshi explore artifacts to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
