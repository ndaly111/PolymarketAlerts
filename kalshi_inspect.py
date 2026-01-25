"""Kalshi Inspect (one-off)

This script is meant to answer a practical question:
  "What does Kalshi data look like in the trade-api/v2 endpoints?"

It performs signed GET requests, then dumps *raw* JSON responses and small summaries
into OUTDIR (default: out/). It does NOT require The Odds API.

Outputs (examples):
  - kalshi_inspect_meta_<stamp>.json
  - kalshi_series_raw_<stamp>.json + kalshi_series_sample_<stamp>.csv
  - kalshi_markets_sample_<stamp>.json
  - kalshi_events_sample_<stamp>.json
  - kalshi_inspect_summary_<stamp>.md
and also *_latest.* convenience copies.
"""

from __future__ import annotations

import base64
import csv
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding


# -----------------------------
# Helpers / env
# -----------------------------


def env_str(name: str, default: str = "") -> str:
    v = os.getenv(name)
    if v is None:
        return default
    v = str(v)
    if not v.strip():
        return default
    return v.strip()


def env_int(name: str, default: int) -> int:
    v = env_str(name, "")
    if not v:
        return default
    try:
        return int(v)
    except Exception:
        return default


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def write_csv(path: str, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def write_jsonl(path: str, items: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False))
            f.write("\n")


def copy_latest(src: str, dst: str) -> None:
    try:
        with open(src, "rb") as fsrc:
            data = fsrc.read()
        with open(dst, "wb") as fdst:
            fdst.write(data)
    except Exception:
        # Don't fail the run for convenience copies.
        pass


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def shallow_keys(items: List[Dict[str, Any]]) -> List[str]:
    keys = set()
    for it in items:
        if isinstance(it, dict):
            keys.update(it.keys())
    return sorted(keys)


def summarize_schema(items: List[Dict[str, Any]], title: str, max_keys: int = 80) -> str:
    keys = shallow_keys(items)
    if not keys:
        return f"### {title}\n\n(no items)\n"
    shown = keys[:max_keys]
    extra = len(keys) - len(shown)
    lines = [f"### {title}", "", f"Count: {len(items)}", "", "Top-level keys:"]
    for k in shown:
        lines.append(f"- `{k}`")
    if extra > 0:
        lines.append(f"- … ({extra} more)")
    lines.append("")
    return "\n".join(lines)


# -----------------------------
# Kalshi auth + HTTP
# -----------------------------


def _session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://", HTTPAdapter(max_retries=retry))
    s.headers.update(
        {
            "User-Agent": "kalshi-inspect/1.0 (+github-actions)",
            "Accept": "application/json",
        }
    )
    return s


def kalshi_private_key_pem() -> str:
    pem = env_str("KALSHI_PRIVATE_KEY", "")
    if not pem:
        raise RuntimeError("Missing KALSHI_PRIVATE_KEY.")

    # Common GitHub Secret input: pasted with literal \\n sequences
    if "\\n" in pem and "\n" not in pem:
        pem = pem.replace("\\n", "\n")
    return pem


def kalshi_key_id() -> str:
    kid = env_str("KALSHI_API_KEY_ID", "") or env_str("KALSHI_KEY_ID", "")
    if not kid:
        raise RuntimeError("Missing KALSHI_API_KEY_ID (or KALSHI_KEY_ID).")
    return kid


def kalshi_base_url() -> str:
    base = env_str("KALSHI_BASE", "https://api.elections.kalshi.com/trade-api/v2").rstrip("/")
    if "/trade-api/" not in base:
        base = base + "/trade-api/v2"
    return base.rstrip("/")


@dataclass(frozen=True)
class KalshiClient:
    base: str
    key_id: str
    priv: Any
    session: requests.Session

    @staticmethod
    def from_env() -> "KalshiClient":
        priv = serialization.load_pem_private_key(
            kalshi_private_key_pem().encode("utf-8"),
            password=None,
        )
        return KalshiClient(
            base=kalshi_base_url(),
            key_id=kalshi_key_id(),
            priv=priv,
            session=_session(),
        )

    def signed_headers(self, method: str, full_path: str) -> Dict[str, str]:
        # message = timestamp_ms + method + path
        ts = str(int(time.time() * 1000))
        m = method.upper().strip()
        path = str(full_path).strip()
        if not path.startswith("/"):
            path = "/" + path

        msg = (ts + m + path).encode("utf-8")
        sig = self.priv.sign(
            msg,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode("ascii"),
            "KALSHI-ACCESS-TIMESTAMP": ts,
        }

    def get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Tuple[int, Dict[str, Any]]:
        path = path if path.startswith("/") else "/" + path
        url = self.base + path
        headers = self.signed_headers("GET", path)
        try:
            r = self.session.get(url, params=params or {}, headers=headers, timeout=30)
        except requests.RequestException as e:
            return 0, {
                "_request_error": True,
                "error": str(e),
                "url": url,
                "path": path,
                "params": params or {},
            }
        try:
            data = r.json()
        except Exception:
            data = {"_non_json": True, "text": r.text[:2000]}
        # Always attach a short raw snippet for debugging on errors.
        if r.status_code >= 400:
            try:
                txt = r.text or ""
            except Exception:
                txt = ""
            if isinstance(data, dict):
                data.setdefault("_http_status", r.status_code)
                if txt:
                    data.setdefault("_raw_text_snippet", txt[:2000])
        return r.status_code, data


# -----------------------------
# Kalshi endpoints (best-effort)
# -----------------------------


def paginate(
    client: KalshiClient,
    path: str,
    list_key: str,
    params: Dict[str, Any],
    hard_limit: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    """
    Paginate if the response includes a cursor-like token.
    We don't assume a specific field name; we check common patterns.
    """
    out: List[Dict[str, Any]] = []
    pages_meta: List[Dict[str, Any]] = []
    last_resp: Dict[str, Any] = {}

    cursor_param_names = ["cursor", "next_cursor", "page_token"]
    cursor_resp_keys = ["cursor", "next_cursor", "nextCursor", "next_page_token", "nextPageToken"]

    # We'll store any cursor in params using "cursor" (common) unless caller already used something else.
    cursor_param = "cursor" if "cursor" not in params else "cursor"

    while True:
        if len(out) >= hard_limit:
            break

        status, data = client.get(path, params=params)
        last_resp = data
        pages_meta.append(
            {
                "path": path,
                "status": status,
                "params": dict(params),
                "received_keys": sorted(list(data.keys())) if isinstance(data, dict) else [],
            }
        )

        if status >= 400:
            break

        items = data.get(list_key, [])
        if isinstance(items, list):
            out.extend([it for it in items if isinstance(it, dict)])
        else:
            # Unexpected shape
            break

        # Find cursor in response
        cursor = None
        for k in cursor_resp_keys:
            v = data.get(k)
            if isinstance(v, str) and v.strip():
                cursor = v.strip()
                break

        if not cursor:
            # Some APIs nest pagination under "pagination" or similar
            pag = data.get("pagination")
            if isinstance(pag, dict):
                for k in cursor_resp_keys:
                    v = pag.get(k)
                    if isinstance(v, str) and v.strip():
                        cursor = v.strip()
                        break

        if not cursor:
            break

        # Put cursor into params for next request
        params[cursor_param] = cursor

        # If API expects a different param name, try a best-effort swap next iteration
        for alt in cursor_param_names:
            if alt in params and alt != cursor_param:
                cursor_param = alt
                break

    # Trim to hard limit
    out = out[:hard_limit]
    meta = {"pages": pages_meta}
    return out, meta, [last_resp] if isinstance(last_resp, dict) else []


def fetch_series(client: KalshiClient, category: Optional[str], limit: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    params: Dict[str, Any] = {"limit": min(limit, 500)}
    if category:
        params["category"] = category

    items, meta, last = paginate(client, "/series", "series", params, hard_limit=limit)
    return items, meta, (last[0] if last else {})


def fetch_markets_for_series(client: KalshiClient, series_ticker: str, limit: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    params: Dict[str, Any] = {"series_ticker": series_ticker, "limit": min(limit, 500)}
    items, meta, last = paginate(client, "/markets", "markets", params, hard_limit=limit)
    return items, meta, (last[0] if last else {})


def fetch_event(client: KalshiClient, event_ticker: str) -> Tuple[int, Dict[str, Any]]:
    return client.get(f"/events/{event_ticker}", params={})


def fetch_filters_by_sport(client: KalshiClient) -> Tuple[int, Dict[str, Any]]:
    return client.get("/search/filters_by_sport", params={})


def fetch_tags_by_categories(client: KalshiClient) -> Tuple[int, Dict[str, Any]]:
    return client.get("/search/tags_by_categories", params={})


# -----------------------------
# Main
# -----------------------------


def main() -> int:
    outdir = env_str("OUTDIR", "out")
    ensure_dir(outdir)
    stamp = utc_stamp()

    meta: Dict[str, Any] = {
        "stamp_utc": stamp,
        "base": env_str("KALSHI_BASE", ""),
        "base_normalized": kalshi_base_url(),
        "has_key_id": bool(env_str("KALSHI_KEY_ID", "") or env_str("KALSHI_API_KEY_ID", "")),
        "has_private_key": bool(env_str("KALSHI_PRIVATE_KEY", "")),
        "category": env_str("KALSHI_CATEGORY", "Sports"),
        "series_limit": env_int("KALSHI_SERIES_LIMIT", 200),
        "market_limit": env_int("KALSHI_MARKET_LIMIT", 120),
        "sample_series": env_str("KALSHI_SAMPLE_SERIES", ""),
        "sample_series_count": env_int("KALSHI_SAMPLE_SERIES_COUNT", 6),
        "markets_per_series": env_int("KALSHI_MARKETS_PER_SERIES", 12),
        "events_per_series": env_int("KALSHI_EVENTS_PER_SERIES", 6),
        "status": "init",
        "notes": [],
    }

    meta_path = os.path.join(outdir, f"kalshi_inspect_meta_{stamp}.json")

    # Fail fast if secrets are missing
    if not meta["has_key_id"] or not meta["has_private_key"]:
        meta["status"] = "error_missing_auth"
        meta["notes"].append("Missing KALSHI_API_KEY_ID/KALSHI_KEY_ID and/or KALSHI_PRIVATE_KEY.")
        write_json(meta_path, meta)
        copy_latest(meta_path, os.path.join(outdir, "kalshi_inspect_meta_latest.json"))
        print("Missing Kalshi auth env vars. See out/kalshi_inspect_meta_latest.json")
        return 2

    client = KalshiClient.from_env()

    # 0) Discovery helpers (very useful for sports)
    f_status, filters = fetch_filters_by_sport(client)
    t_status, tags = fetch_tags_by_categories(client)
    meta["filters_by_sport_status"] = f_status
    meta["tags_by_categories_status"] = t_status
    filters_path = os.path.join(outdir, f"kalshi_filters_by_sport_{stamp}.json")
    tags_path = os.path.join(outdir, f"kalshi_tags_by_categories_{stamp}.json")
    write_json(filters_path, {"status": f_status, "data": filters})
    write_json(tags_path, {"status": t_status, "data": tags})

    category = meta["category"]
    series_limit = int(meta["series_limit"])

    # 1) Series (with category)
    series_items, series_meta, series_last = fetch_series(client, category=category, limit=series_limit)
    meta["series_request"] = series_meta

    series_raw_path = os.path.join(outdir, f"kalshi_series_raw_{stamp}.json")
    write_json(series_raw_path, {"category": category, "series": series_items, "last_page": series_last})

    if not series_items:
        # 1b) Fallback: try common casing variants, then no filter
        base_cat = str(category).strip()
        tried = {base_cat}
        for alt in [base_cat.title(), base_cat.upper(), base_cat.lower()]:
            if not alt or alt in tried:
                continue
            tried.add(alt)
            series_items2, series_meta2, series_last2 = fetch_series(client, category=alt, limit=series_limit)
            meta[f"series_request_retry_{alt}"] = series_meta2
            if series_items2:
                meta["notes"].append(f"Category '{category}' returned 0; retry '{alt}' returned {len(series_items2)}.")
                series_items, series_last = series_items2, series_last2
                break
        if not series_items:
            meta["notes"].append(f"No series returned for category='{category}'. Falling back to /series without category filter.")
        series_items2, series_meta2, series_last2 = fetch_series(client, category=None, limit=series_limit)
        meta["series_request_fallback"] = series_meta2
        series_items = series_items2
        series_last = series_last2
        write_json(series_raw_path, {"category": None, "series": series_items, "last_page": series_last})

    # Write CSV sample for readability
    series_csv_path = os.path.join(outdir, f"kalshi_series_sample_{stamp}.csv")
    series_rows: List[Dict[str, Any]] = []
    for s in series_items:
        series_rows.append(
            {
                "ticker": s.get("ticker", ""),
                "title": s.get("title", ""),
                "category": s.get("category", ""),
                "sub_category": s.get("sub_category", "") or s.get("subcategory", ""),
                "active": s.get("active", ""),
            }
        )
    write_csv(series_csv_path, series_rows, ["ticker", "title", "category", "sub_category", "active"])

    # Decide which series to inspect deeper
    sample_series_override = env_str("KALSHI_SAMPLE_SERIES", "")
    if sample_series_override:
        chosen_series = [x.strip() for x in sample_series_override.split(",") if x.strip()]
    else:
        n = env_int("KALSHI_SAMPLE_SERIES_COUNT", 6)
        chosen_series = []
        for s in series_items:
            t = str(s.get("ticker") or "").strip()
            if not t:
                continue
            chosen_series.append(t)
            if len(chosen_series) >= n:
                break

    meta["chosen_series"] = chosen_series

    # 2) Markets for chosen series
    market_limit = env_int("KALSHI_MARKET_LIMIT", 120)
    markets_per_series = env_int("KALSHI_MARKETS_PER_SERIES", 12)
    events_per_series = env_int("KALSHI_EVENTS_PER_SERIES", 6)

    market_samples: List[Dict[str, Any]] = []
    market_pages_meta: Dict[str, Any] = {}
    market_last_pages: Dict[str, Any] = {}

    event_samples: List[Dict[str, Any]] = []
    events_seen: set[str] = set()

    for st in chosen_series:
        markets, mmeta, mlast = fetch_markets_for_series(client, series_ticker=st, limit=market_limit)
        market_pages_meta[st] = mmeta
        market_last_pages[st] = mlast

        # sample markets for schema
        for mk in markets[:markets_per_series]:
            mk2 = dict(mk)
            mk2["_series_ticker"] = st
            market_samples.append(mk2)

        # derive some event tickers from the sampled markets
        picked = 0
        for mk in markets:
            if picked >= events_per_series:
                break
            et = str(mk.get("event_ticker") or mk.get("eventTicker") or "").strip()
            if not et or et in events_seen:
                continue
            status, ev = fetch_event(client, et)
            ev2 = dict(ev) if isinstance(ev, dict) else {"raw": ev}
            ev2["_event_ticker"] = et
            ev2["_status"] = status
            ev2["_from_series"] = st
            event_samples.append(ev2)
            events_seen.add(et)
            picked += 1

    markets_path = os.path.join(outdir, f"kalshi_markets_sample_{stamp}.json")
    write_json(markets_path, {"markets": market_samples, "last_pages": market_last_pages, "pages": market_pages_meta})

    events_path = os.path.join(outdir, f"kalshi_events_sample_{stamp}.json")
    write_json(events_path, {"events": event_samples})

    # 3) Human-readable summary
    summary_lines: List[str] = []
    summary_lines.append("# Kalshi Inspect Summary")
    summary_lines.append("")
    summary_lines.append(f"- Stamp (UTC): `{stamp}`")
    summary_lines.append(f"- Base: `{kalshi_base_url()}`")
    summary_lines.append(f"- Category requested: `{category}`")
    summary_lines.append(f"- filters_by_sport status: **{f_status}**")
    summary_lines.append(f"- tags_by_categories status: **{t_status}**")
    summary_lines.append(f"- Series returned: **{len(series_items)}**")
    summary_lines.append(f"- Chosen series: `{', '.join(chosen_series) if chosen_series else '(none)'}`")
    summary_lines.append(f"- Market samples written: **{len(market_samples)}**")
    summary_lines.append(f"- Event samples written: **{len(event_samples)}**")
    summary_lines.append("")
    if meta.get("notes"):
        summary_lines.append("## Notes")
        for n in meta["notes"]:
            summary_lines.append(f"- {n}")
        summary_lines.append("")

    summary_lines.append("## Schemas (top-level keys)")
    summary_lines.append("")
    summary_lines.append(summarize_schema(series_items, "Series objects"))
    summary_lines.append(summarize_schema(market_samples, "Market objects (sampled)"))
    summary_lines.append(summarize_schema(event_samples, "Event objects (sampled)"))

    # Extra: quick heuristics that help the main scanner later
    summary_lines.append("## Quick heuristics")
    summary_lines.append("")
    if series_items:
        tickers = [str(s.get("ticker") or "").strip() for s in series_items[:50]]
        tickers = [t for t in tickers if t]
        if tickers:
            summary_lines.append("First 20 series tickers:")
            summary_lines.append("")
            summary_lines.append("- " + ", ".join(tickers[:20]))
            summary_lines.append("")

    # Look for common fields that might drive filtering
    def count_field(items: List[Dict[str, Any]], field: str) -> int:
        c = 0
        for it in items:
            if isinstance(it, dict) and it.get(field) not in (None, ""):
                c += 1
        return c

    summary_lines.append("Market field presence (sample):")
    for f in [
        "ticker",
        "title",
        "series_ticker",
        "event_ticker",
        "close_time",
        "open_time",
        "status",
        "volume",
        "volume_24h",
        "yes_bid",
        "yes_ask",
        "no_bid",
        "no_ask",
    ]:
        summary_lines.append(f"- `{f}`: {count_field(market_samples, f)}/{len(market_samples)}")
    summary_lines.append("")

    summary_path = os.path.join(outdir, f"kalshi_inspect_summary_{stamp}.md")
    write_text(summary_path, "\n".join(summary_lines))

    # 4) Meta write
    meta["status"] = "ok"
    write_json(meta_path, meta)

    # Convenience: keep non-timestamped copies for the most important outputs.
    copy_latest(meta_path, os.path.join(outdir, "kalshi_inspect_meta_latest.json"))
    copy_latest(summary_path, os.path.join(outdir, "kalshi_inspect_summary_latest.md"))
    copy_latest(series_raw_path, os.path.join(outdir, "kalshi_series_raw_latest.json"))
    copy_latest(series_csv_path, os.path.join(outdir, "kalshi_series_sample_latest.csv"))
    copy_latest(markets_path, os.path.join(outdir, "kalshi_markets_sample_latest.json"))
    copy_latest(events_path, os.path.join(outdir, "kalshi_events_sample_latest.json"))
    copy_latest(filters_path, os.path.join(outdir, "kalshi_filters_by_sport_latest.json"))
    copy_latest(tags_path, os.path.join(outdir, "kalshi_tags_by_categories_latest.json"))

    print(
        f"Wrote artifacts to {outdir}/ (series={len(series_items)}, market_samples={len(market_samples)}, event_samples={len(event_samples)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
