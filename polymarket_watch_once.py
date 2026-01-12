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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DATA_API_BASE = "https://data-api.polymarket.com"

# Default target = Polymarket profile @0x8dxd (profile address)
DEFAULT_TARGET_ADDRESS = "0x63ce342161250d705dc0b16df89036c8e5f9ba9a"

TARGET_ADDRESS = os.getenv("TARGET_ADDRESS", DEFAULT_TARGET_ADDRESS).strip()

# IMPORTANT: This must come from GitHub Actions secret, not hardcoded.
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "").strip()

# Optional: mention a user or role for stronger notifications in Discord
# Examples: "<@123456789012345678>" or "<@&ROLE_ID>"
DISCORD_MENTION = os.getenv("DISCORD_MENTION", "").strip()

STATE_FILE = Path(os.getenv("STATE_FILE", ".state/polymarket_state.json"))
SEED_LIMIT = int(os.getenv("SEED_LIMIT", "50"))
SEEN_CACHE_MAX = int(os.getenv("SEEN_CACHE_MAX", "2000"))

# Resync position sizes after processing trades (helps avoid drift).
RESYNC_POSITIONS = os.getenv("RESYNC_POSITIONS", "1").strip() == "1"


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


def unix_to_dt(ts: int) -> datetime:
    # Handle seconds or ms defensively.
    if ts > 1_000_000_000_000:
        return datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def get_json(path: str, params: Dict[str, Any]) -> Any:
    url = f"{DATA_API_BASE}{path}"
    r = SESSION.get(url, params=params, timeout=25)
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code} from Data API {path}: {r.text[:300]}")
    return r.json()


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


def seed_state(state: State) -> None:
    """
    First run: record recent trades so we don't spam history, and snapshot positions.
    No alerts on seed run.
    """
    recent = fetch_activity(
        TARGET_ADDRESS,
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
        positions = fetch_positions(TARGET_ADDRESS)
        state.pos_sizes = {}
        for p in positions:
            cid = str(p.get("conditionId", ""))
            oi = int(p.get("outcomeIndex", -1))
            state.pos_sizes[position_key(cid, oi)] = str(p.get("size", 0.0) or 0.0)
    except Exception:
        pass

    state.save(STATE_FILE)
    print(f"[seed] Created {STATE_FILE}. No alerts sent on seed run.")


def run_once() -> None:
    state = State.load(STATE_FILE)

    # Seed and exit on first run
    if state.last_ts == 0:
        seed_state(state)
        return

    seen = set(state.seen_keys)
    max_ts_seen = state.last_ts

    limit = 500
    offset = 0

    while True:
        batch = fetch_activity(
            TARGET_ADDRESS,
            start=state.last_ts,
            limit=limit,
            offset=offset,
            sort_direction="ASC",
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
            label = None

            if side == "BUY":
                after = before + size_tokens
                if before == 0 and after > 0:
                    label = "OPEN"
            elif side == "SELL":
                after = max(before - size_tokens, Decimal(0))
                if before > 0 and after == 0:
                    label = "CLOSE"

            state.pos_sizes[pk] = format(after, "f")

            if label:
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

                when = unix_to_dt(ts).strftime("%Y-%m-%d %H:%M:%S UTC") if ts else "unknown time"
                tx = str(evt.get("transactionHash", "")).strip()
                tx_url = f"https://polygonscan.com/tx/{tx}" if tx else ""

                shares_str = f"{size_tokens.quantize(Decimal('0.0001')):,}"
                before_str = f"{before.quantize(Decimal('0.0001')):,}"
                after_str = f"{after.quantize(Decimal('0.0001')):,}"
                msg = (
                    f"**{label}** — {side} {outcome}\n"
                    f"{title}\n"
                    f"{when}\n"
                    f"Amount: **${usdc_amt:,.2f}** | Shares: {shares_str}\n"
                    f"Position: {before_str} → {after_str}\n"
                    f"{tx_url}"
                )
                post_discord(msg)

        if len(batch) < limit:
            break
        offset += limit

    state.last_ts = max_ts_seen

    if RESYNC_POSITIONS:
        try:
            positions = fetch_positions(TARGET_ADDRESS)
            new_map: Dict[str, str] = {}
            for p in positions:
                cid = str(p.get("conditionId", ""))
                oi = int(p.get("outcomeIndex", -1))
                new_map[position_key(cid, oi)] = str(p.get("size", 0.0) or 0.0)
            state.pos_sizes = new_map
        except Exception:
            pass

    state.save(STATE_FILE)


if __name__ == "__main__":
    run_once()
