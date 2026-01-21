#!/usr/bin/env python3
"""
Runs once (for GitHub Actions schedule).

Behavior:
- First run seeds state and exits (no alerts).
- Subsequent runs:
  - fetches new TRADE activity from Polymarket Data API /activity since last_ts
  - tracks per-position share size (conditionId + outcomeIndex)
  - posts Discord alerts ONLY for:
      OPEN  = BUY when prior size == 0
      CLOSE = SELL that brings size to 0
  - persists state to a JSON file so runs don't duplicate alerts
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DATA_API_BASE = "https://data-api.polymarket.com"
GAMMA_API_BASE = "https://gamma-api.polymarket.com"

# Default target = Polymarket profile @0x8dxd (profile address)
DEFAULT_TARGET_ADDRESS = "0x63ce342161250d705dc0b16df89036c8e5f9ba9a"

TARGET_ADDRESS = os.getenv("TARGET_ADDRESS", DEFAULT_TARGET_ADDRESS).strip()
TARGET_USERNAME = os.getenv("TARGET_USERNAME", "").strip().lstrip("@")

# IMPORTANT: This must come from GitHub Actions secret, not hardcoded.
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "").strip()

# Optional: mention a user or role for stronger notifications in Discord
# Examples: "<@123456789012345678>" or "<@&ROLE_ID>"
DISCORD_MENTION = os.getenv("DISCORD_MENTION", "").strip()

STATE_FILE = Path(os.getenv("STATE_FILE", ".state/polymarket_state.json"))
SEED_LIMIT = int(os.getenv("SEED_LIMIT", "50"))
SEEN_CACHE_MAX = int(os.getenv("SEEN_CACHE_MAX", "2000"))

# Polling behavior
BATCH_LIMIT = int(os.getenv("BATCH_LIMIT", "100"))
SLEEP_BETWEEN_PAGES = float(os.getenv("SLEEP_BETWEEN_PAGES", "0.5"))

# Resync position sizes after processing trades (helps avoid drift).
RESYNC_POSITIONS = os.getenv("RESYNC_POSITIONS", "true").lower() in ("1", "true", "yes", "y")

# Optional: persist trade log for later display (JSONL)
TRADE_LOG_FILE = os.getenv("TRADE_LOG_FILE", "").strip()
MAX_TRADE_LOG_LINES = int(os.getenv("MAX_TRADE_LOG_LINES", "10000"))

# Optional: emit quick reports (useful for GitHub Actions artifacts)
REPORT_JSON = os.getenv("REPORT_JSON", "").strip()
REPORT_HTML = os.getenv("REPORT_HTML", "").strip()
REPORT_TAIL = int(os.getenv("REPORT_TAIL", "250"))

# Which lifecycle events should trigger Discord alerts
# - OPEN: first time a position becomes >0
# - CLOSE: position goes to 0 from >0
_aa = os.getenv("ALERT_ACTIONS", "OPEN,CLOSE")
ALERT_ACTIONS = {s.strip().upper() for s in _aa.split(",") if s.strip()}


def make_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=6,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST"]),
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.headers.update(
        {"Accept": "application/json", "User-Agent": "polymarket-open-close-watcher/1.0"}
    )
    return s


SESSION = make_session()


def _ensure_parent_dir(path: str) -> None:
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)


def _append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    if not path:
        return
    _ensure_parent_dir(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _read_jsonl_tail(path: str, n: int) -> List[Dict[str, Any]]:
    if not path or not Path(path).exists() or n <= 0:
        return []
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    out: List[Dict[str, Any]] = []
    for line in lines[-n:]:
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _trim_jsonl(path: str, max_lines: int) -> None:
    if not path or max_lines <= 0:
        return
    p = Path(path)
    if not p.exists():
        return
    with open(p, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if len(lines) <= max_lines:
        return
    keep = lines[-max_lines:]
    with open(p, "w", encoding="utf-8") as f:
        f.writelines(keep)


def _get_json_base(base: str, path: str, params: Dict[str, Any]) -> Any:
    url = f"{base.rstrip('/')}{path}"
    print(f"[fetch] GET {url} params={params}")
    r = SESSION.get(url, params=params, timeout=25)
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code} from API {path}: {r.text[:300]}")
    return r.json()


def resolve_username_to_wallet(username: str) -> Optional[str]:
    """
    Resolve a Polymarket username -> proxy wallet address using Gamma's public-search endpoint.
    If resolution isn't unique, returns None.
    """
    u = username.strip().lstrip("@")
    if not u:
        return None
    try:
        data = _get_json_base(
            GAMMA_API_BASE,
            "/public-search",
            {"q": u, "search_profiles": "true", "limit_per_type": 10},
        )
    except Exception as exc:
        print(f"[resolve] Failed to resolve @{u}: {exc}")
        return None

    profiles = data.get("profiles") or []
    if not isinstance(profiles, list):
        profiles = []

    def match(profile: Dict[str, Any]) -> bool:
        name = str(profile.get("name", "") or "").strip().lstrip("@").lower()
        pseudonym = str(profile.get("pseudonym", "") or "").strip().lstrip("@").lower()
        return name == u.lower() or pseudonym == u.lower()

    for profile in profiles:
        if isinstance(profile, dict) and match(profile):
            wallet = str(profile.get("proxyWallet", "") or "").strip()
            return wallet or None

    if len(profiles) == 1 and isinstance(profiles[0], dict):
        wallet = str(profiles[0].get("proxyWallet", "") or "").strip()
        return wallet or None

    print(f"[resolve] Could not uniquely resolve @{u}. Profiles returned={len(profiles)}")
    return None


def resolve_target_address() -> str:
    if TARGET_USERNAME:
        wallet = resolve_username_to_wallet(TARGET_USERNAME)
        if wallet:
            return wallet
        print(
            f"[resolve] Falling back to TARGET_ADDRESS because @{TARGET_USERNAME} could not be resolved."
        )
    return TARGET_ADDRESS


def unix_to_dt(ts: int) -> datetime:
    # Handle seconds or ms defensively.
    if ts > 1_000_000_000_000:
        return datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def get_json(path: str, params: Dict[str, Any]) -> Any:
    return _get_json_base(DATA_API_BASE, path, params)


def fetch_activity(
    user: str,
    start: int | None,
    limit: int,
    offset: int,
    sort_direction: str,
) -> List[Dict[str, Any]]:
    params: Dict[str, Any] = {
        "user": user,
        "type": "TRADE",
        "limit": limit,
        "offset": offset,
        "sortDirection": sort_direction,  # ASC for chronological processing
    }
    if start is not None:
        params["start"] = start
    return get_json("/activity", params)


def fetch_positions(user: str, limit: int = 500, offset: int = 0) -> List[Dict[str, Any]]:
    return get_json("/positions", {"user": user, "limit": limit, "offset": offset})


def trade_key(evt: Dict[str, Any]) -> str:
    def normalize_decimal(value: Any) -> str:
        if value is None:
            return ""
        try:
            d = Decimal(str(value))
        except (InvalidOperation, ValueError):
            return str(value)
        normalized = format(d.normalize(), "f")
        if "." in normalized:
            normalized = normalized.rstrip("0").rstrip(".")
        return normalized

    # Dedup key: combine tx hash + core fields to be safe.
    tx = str(evt.get("transactionHash", ""))
    asset = str(evt.get("asset", ""))
    side = str(evt.get("side", ""))
    ts = str(int(evt.get("timestamp", 0) or 0))
    size = normalize_decimal(evt.get("size", ""))
    price = normalize_decimal(evt.get("price", ""))
    outcome_index = str(evt.get("outcomeIndex", ""))
    condition_id = str(evt.get("conditionId", ""))
    return f"{tx}|{asset}|{side}|{ts}|{condition_id}|{outcome_index}|{size}|{price}"


def position_key(condition_id: str, outcome_index: int) -> str:
    return f"{condition_id}:{outcome_index}"


@dataclass
class State:
    last_ts: int = 0
    seen_keys: List[str] = field(default_factory=list)
    pos_sizes: Dict[str, str] = field(default_factory=dict)

    @staticmethod
    def load(path: Path) -> "State":
        if not path.exists():
            return State()
        data = json.loads(path.read_text())
        return State(
            last_ts=int(data.get("last_ts", 0) or 0),
            seen_keys=list(data.get("seen_keys", [])),
            pos_sizes={k: str(v) for k, v in dict(data.get("pos_sizes", {})).items()},
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.seen_keys = self.seen_keys[-SEEN_CACHE_MAX:]
        path.write_text(
            json.dumps(
                {"last_ts": self.last_ts, "seen_keys": self.seen_keys, "pos_sizes": self.pos_sizes},
                indent=2,
                sort_keys=True,
            )
        )


def post_discord(text: str) -> None:
    if not DISCORD_WEBHOOK_URL:
        raise RuntimeError("DISCORD_WEBHOOK_URL is not set (configure as a GitHub Actions secret).")

    content = f"{DISCORD_MENTION}\n{text}" if DISCORD_MENTION else text
    payload = {
        "content": content[:1900],
        "allowed_mentions": {"parse": ["users", "roles", "everyone"]},
    }

    resp = SESSION.post(DISCORD_WEBHOOK_URL, json=payload, timeout=20)
    if resp.status_code >= 400:
        # Fail the run so we don’t advance state and miss notifications.
        raise RuntimeError(f"Discord webhook returned HTTP {resp.status_code}: {resp.text[:200]}")


def seed_state(state: State, target_address: str) -> None:
    """
    First run: record recent trades so we don't spam history, and snapshot positions.
    No alerts on seed run.
    """
    print(
        f"[seed] No prior state found. Seeding from recent activity (limit={SEED_LIMIT}) and snapshotting positions."
    )
    recent = fetch_activity(
        target_address,
        start=None,
        limit=SEED_LIMIT,
        offset=0,
        sort_direction="DESC",
    )
    if recent:
        state.seen_keys = [trade_key(e) for e in recent]
        state.last_ts = max(int(e.get("timestamp", 0) or 0) for e in recent)

    # Snapshot current positions so OPEN/CLOSE begins from a sane baseline.
    try:
        positions = fetch_positions(target_address)
        print(f"[seed] Snapshot positions: {len(positions)} entries.")
        state.pos_sizes = {}
        for p in positions:
            cid = str(p.get("conditionId", ""))
            oi = int(p.get("outcomeIndex", -1))
            state.pos_sizes[position_key(cid, oi)] = str(p.get("size", 0.0) or 0.0)
    except Exception:
        print("[seed] Position snapshot failed; continuing with empty baseline.")
        pass

    state.save(STATE_FILE)
    print(f"[seed] Created {STATE_FILE}. No alerts sent on seed run.")


def run_once() -> None:
    target_address = resolve_target_address()
    who = f"@{TARGET_USERNAME}" if TARGET_USERNAME else target_address

    state = State.load(STATE_FILE)
    print(
        f"[run] Starting watch for {who} ({target_address}). State file: {STATE_FILE} (last_ts={state.last_ts})."
    )

    # Seed and exit on first run
    if state.last_ts == 0:
        seed_state(state, target_address)
        return

    seen = set(state.seen_keys)
    max_ts_seen = state.last_ts

    limit = BATCH_LIMIT
    offset = 0

    total_processed = 0
    total_alerts = 0
    opened_positions_run: List[Dict[str, Any]] = []
    while True:
        batch = fetch_activity(
            target_address,
            start=state.last_ts,
            limit=limit,
            offset=offset,
            sort_direction="ASC",
        )
        print(
            f"[run] Activity batch offset={offset} limit={limit} received={len(batch)} items."
        )
        if not batch:
            break

        for evt in batch:
            ts = int(evt.get("timestamp", 0) or 0)
            max_ts_seen = max(max_ts_seen, ts)

            k = trade_key(evt)
            if k in seen:
                continue

            seen.add(k)
            state.seen_keys.append(k)
            condition_id = str(evt.get("conditionId", "")).strip()
            outcome_index = int(evt.get("outcomeIndex", -1))
            outcome_index_str = str(evt.get("outcomeIndex", "")).strip()
            pk = position_key(condition_id, outcome_index)

            side = str(evt.get("side", "")).upper().strip()
            try:
                size_tokens = Decimal(str(evt.get("size", 0.0) or 0.0))
            except (InvalidOperation, ValueError):
                size_tokens = Decimal(0)

            try:
                before = Decimal(state.pos_sizes.get(pk, "0"))
            except (InvalidOperation, ValueError):
                before = Decimal(0)
            after = before
            action = "UNKNOWN"
            alert_label: Optional[str] = None

            if side == "BUY":
                after = before + size_tokens
                action = "OPEN" if (before == 0 and after > 0) else "ADD"
                if action == "OPEN":
                    alert_label = "OPEN"
            elif side == "SELL":
                after = max(before - size_tokens, Decimal(0))
                action = "CLOSE" if (before > 0 and after == 0) else "REDUCE"
                if action == "CLOSE":
                    alert_label = "CLOSE"

            state.pos_sizes[pk] = format(after, "f")

            title = str(evt.get("title", "(unknown market)")).strip()
            outcome = str(evt.get("outcome", "")).strip()

            usdc = evt.get("usdcSize", None)
            try:
                price = Decimal(str(evt.get("price", 0.0) or 0.0))
            except (InvalidOperation, ValueError):
                price = Decimal(0)
            if usdc is not None:
                try:
                    usdc_amt = float(Decimal(str(usdc)))
                except (InvalidOperation, ValueError):
                    usdc_amt = float(size_tokens * price)
            else:
                usdc_amt = float(size_tokens * price)

            when_utc = unix_to_dt(ts).strftime("%Y-%m-%d %H:%M:%S UTC") if ts else "unknown time"
            tx = str(evt.get("transactionHash", "")).strip()
            tx_url = f"https://polygonscan.com/tx/{tx}" if tx else ""

            if action == "OPEN":
                opened_positions_run.append(
                    {
                        "when_utc": when_utc,
                        "timestamp": ts,
                        "action": "OPEN",
                        "side": side,
                        "conditionId": condition_id,
                        "outcomeIndex": outcome_index_str,
                        "outcome": outcome,
                        "title": title,
                        "size": str(size_tokens),
                        "usdcSize": usdc_amt,
                        "before": str(before),
                        "after": str(after),
                        "tx": tx,
                        "tx_url": tx_url,
                        "event_key": k,
                    }
                )

            if TRADE_LOG_FILE:
                _append_jsonl(
                    TRADE_LOG_FILE,
                    {
                        "when_utc": when_utc,
                        "timestamp": ts,
                        "address": target_address,
                        "username": TARGET_USERNAME,
                        "action": action,
                        "side": side,
                        "conditionId": condition_id,
                        "outcomeIndex": outcome_index_str,
                        "outcome": outcome,
                        "title": title,
                        "size": str(size_tokens),
                        "usdcSize": usdc_amt,
                        "before": str(before),
                        "after": str(after),
                        "price": str(price),
                        "tx": tx,
                        "tx_url": tx_url,
                        "event_key": k,
                    },
                )

            if alert_label and alert_label in ALERT_ACTIONS:
                when = when_utc

                shares_str = f"{size_tokens.quantize(Decimal('0.0001')):,}"
                before_str = f"{before.quantize(Decimal('0.0001')):,}"
                after_str = f"{after.quantize(Decimal('0.0001')):,}"
                msg = (
                    f"**{alert_label}** — {side} {outcome}\n"
                    f"{title}\n"
                    f"{when}\n"
                    f"Amount: **${usdc_amt:,.2f}** | Shares: {shares_str}\n"
                    f"Position: {before_str} → **{after_str}**\n"
                    f"{tx_url}"
                )
                print(
                    f"[alert] {alert_label} {side} outcome={outcome} shares={shares_str} amount=${usdc_amt:,.2f}."
                )
                post_discord(msg)
                total_alerts += 1

            total_processed += 1

        if len(batch) < limit:
            break
        offset += limit
        time.sleep(SLEEP_BETWEEN_PAGES)

    state.last_ts = max_ts_seen

    if TRADE_LOG_FILE:
        _trim_jsonl(TRADE_LOG_FILE, MAX_TRADE_LOG_LINES)

    if REPORT_JSON or REPORT_HTML:
        report_obj = {
            "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "target": {"username": TARGET_USERNAME or None, "address": target_address},
            "opened_positions": opened_positions_run,
        }

        if REPORT_JSON:
            _ensure_parent_dir(REPORT_JSON)
            Path(REPORT_JSON).write_text(json.dumps(report_obj, indent=2), encoding="utf-8")

        if REPORT_HTML:
            tail = _read_jsonl_tail(TRADE_LOG_FILE, REPORT_TAIL) if TRADE_LOG_FILE else []
            html_parts = [
                "<html><head><meta charset='utf-8'><title>Polymarket Watch</title></head><body>",
                f"<h2>Polymarket Watch — {who}</h2>",
                f"<p>Generated: {report_obj['generated_at_utc']}</p>",
                "<h3>Newly opened positions (since last run)</h3>",
            ]

            if opened_positions_run:
                html_parts.append("<ul>")
                for op in opened_positions_run:
                    tx = (op.get("tx") or "").strip()
                    tx_url = f"https://polygonscan.com/tx/{tx}" if tx else ""
                    tx_link = f"<a href=\"{tx_url}\">tx</a>" if tx_url else ""
                    html_parts.append(
                        f"<li><b>{op.get('outcome','')}</b> — {op.get('title','')} "
                        f"({op.get('when_utc','')}) {tx_link}</li>"
                    )
                html_parts.append("</ul>")
            else:
                html_parts.append("<p><i>No new opens in this run.</i></p>")

            html_parts.append("<h3>Recent trades (tail)</h3>")
            if tail:
                html_parts.append("<table border='1' cellspacing='0' cellpadding='4'>")
                html_parts.append(
                    "<tr><th>When (UTC)</th><th>Action</th><th>Outcome</th><th>Title</th>"
                    "<th>Shares</th><th>USDC</th><th>Tx</th></tr>"
                )
                for trade in tail:
                    tx = (trade.get("tx") or "").strip()
                    tx_url = f"https://polygonscan.com/tx/{tx}" if tx else ""
                    tx_link = f"<a href=\"{tx_url}\">link</a>" if tx_url else ""
                    html_parts.append(
                        "<tr>"
                        f"<td>{trade.get('when_utc','')}</td>"
                        f"<td>{trade.get('action','')}</td>"
                        f"<td>{trade.get('outcome','')}</td>"
                        f"<td>{trade.get('title','')}</td>"
                        f"<td>{trade.get('size','')}</td>"
                        f"<td>{trade.get('usdcSize','')}</td>"
                        f"<td>{tx_link}</td>"
                        "</tr>"
                    )
                html_parts.append("</table>")
            else:
                html_parts.append("<p><i>No trade log available (set TRADE_LOG_FILE).</i></p>")

            html_parts.append("</body></html>")
            _ensure_parent_dir(REPORT_HTML)
            Path(REPORT_HTML).write_text("\n".join(html_parts), encoding="utf-8")

    if RESYNC_POSITIONS:
        try:
            positions = fetch_positions(target_address)
            print(f"[resync] Positions resynced: {len(positions)} entries.")
            new_map: Dict[str, str] = {}
            for p in positions:
                cid = str(p.get("conditionId", ""))
                oi = int(p.get("outcomeIndex", -1))
                new_map[position_key(cid, oi)] = str(p.get("size", 0.0) or 0.0)
            state.pos_sizes = new_map
        except Exception:
            print("[resync] Position resync failed; keeping computed sizes.")
            pass

    state.save(STATE_FILE)
    print(
        f"[run] Completed. New last_ts={state.last_ts}. Trades processed={total_processed}. Alerts sent={total_alerts}."
    )


if __name__ == "__main__":
    run_once()
