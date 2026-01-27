#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from kalshi_auth_client import KalshiAuthClient
from weather.lib.db import ensure_weather_trades_schema


TRADES_DB_PATH = Path(os.getenv("WEATHER_TRADES_DB_PATH", "weather_trades.db"))
DISCORD_WEBHOOK = os.getenv(
    "DISCORD_WEATHER_TRADES_WEBHOOK",
    os.getenv("DISCORD_WEATHER_ALERTS", os.getenv("DISCORD_WEBHOOK_URL", "")),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cancel stale weather orders.")
    parser.add_argument("--date", default="", help="Target date YYYY-MM-DD (ET).")
    parser.add_argument("--ttl-minutes", type=int, default=5, help="Stale order TTL in minutes.")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without canceling.")
    return parser.parse_args()


def post_discord(message: str) -> None:
    if not DISCORD_WEBHOOK:
        return
    try:
        import requests

        requests.post(DISCORD_WEBHOOK, json={"content": message[:1990]}, timeout=10)
    except Exception as exc:
        print(f"[warn] Discord post failed: {exc}")


def parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def fetch_placed_trades(db_path: Path, trade_date: str) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT id, market_ticker, event_display, side, limit_price_cents,
               fair_q, ev, forecast_high_f, order_id, placed_at_utc
        FROM weather_trades
        WHERE trade_date = ? AND status = 'PLACED'
        """,
        (trade_date,),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def update_trade_status(
    db_path: Path,
    trade_id: int,
    *,
    status: str,
    fill_price_cents: Optional[int] = None,
    filled_at_utc: Optional[str] = None,
    last_checked_at_utc: Optional[str] = None,
) -> None:
    now = datetime.now(ZoneInfo("UTC")).isoformat()
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        UPDATE weather_trades
        SET status = ?,
            fill_price_cents = COALESCE(?, fill_price_cents),
            filled_at_utc = COALESCE(?, filled_at_utc),
            last_checked_at_utc = COALESCE(?, last_checked_at_utc),
            updated_at = ?
        WHERE id = ?
        """,
        (status, fill_price_cents, filled_at_utc, last_checked_at_utc, now, trade_id),
    )
    conn.commit()
    conn.close()


def format_summary(
    trade_date: str,
    ttl_minutes: int,
    filled: List[Dict[str, Any]],
    cancelled: List[Dict[str, Any]],
    still_open: List[Dict[str, Any]],
    failed: List[Dict[str, Any]],
    dry_run: bool,
) -> str:
    header = [
        f"**Weather Order Reaper — {trade_date}**",
        f"TTL: {ttl_minutes}m | Checked: {len(filled) + len(cancelled) + len(still_open) + len(failed)}",
        f"Filled: {len(filled)} | Cancelled: {len(cancelled)} | Still open: {len(still_open)} | Failed: {len(failed)}",
    ]
    if dry_run:
        header.append("Mode: DRY_RUN (no cancels executed)")

    sections: List[str] = ["\n".join(header)]

    def add_section(title: str, rows: List[str]) -> None:
        if rows:
            sections.append(f"\n{title}\n" + "\n".join(rows))

    add_section(
        "✅ FILLED",
        [
            f"- {row['market_ticker']} {row['side']} | {row['fill_price']}¢ | Order {row['order_id']}"
            for row in filled
        ],
    )
    add_section(
        "🛑 CANCELLED",
        [
            f"- {row['market_ticker']} {row['side']} | {row['limit']}¢ | Order {row['order_id']}"
            for row in cancelled
        ],
    )
    add_section(
        "⏳ STILL OPEN",
        [
            f"- {row['market_ticker']} {row['side']} | {row.get('reason', 'waiting')}"
            for row in still_open
        ],
    )
    add_section(
        "⚠️ FAILED",
        [
            f"- {row['market_ticker']} {row['side']} | {row.get('error', 'unknown error')}"
            for row in failed
        ],
    )

    message = "\n".join(sections)
    return message[:1990]


def main() -> int:
    args = parse_args()
    ensure_weather_trades_schema(TRADES_DB_PATH)

    now_et = datetime.now(ZoneInfo("America/New_York"))
    trade_date = args.date or now_et.strftime("%Y-%m-%d")
    ttl_minutes = max(args.ttl_minutes, 1)
    cutoff = datetime.now(ZoneInfo("UTC")) - timedelta(minutes=ttl_minutes)

    placed_trades = fetch_placed_trades(TRADES_DB_PATH, trade_date)
    if not placed_trades:
        # Only post when there is something to act on.
        print("No placed trades found.")
        return 0

    client = KalshiAuthClient.from_env()

    filled: List[Dict[str, Any]] = []
    cancelled: List[Dict[str, Any]] = []
    still_open: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []

    for trade in placed_trades:
        placed_at = parse_dt(trade.get("placed_at_utc"))
        if placed_at is None:
            still_open.append(
                {
                    "market_ticker": trade["market_ticker"],
                    "side": trade["side"],
                    "reason": "missing placed_at_utc",
                }
            )
            continue
        if placed_at > cutoff:
            still_open.append(
                {
                    "market_ticker": trade["market_ticker"],
                    "side": trade["side"],
                    "reason": "within TTL",
                }
            )
            continue

        order_id = trade.get("order_id")
        if not order_id:
            failed.append(
                {
                    "market_ticker": trade["market_ticker"],
                    "side": trade["side"],
                    "error": "missing order_id",
                }
            )
            update_trade_status(
                TRADES_DB_PATH,
                trade["id"],
                status="PLACED",
                last_checked_at_utc=datetime.now(ZoneInfo("UTC")).isoformat(),
            )
            continue

        try:
            order_status = client.get_order(order_id)
            status = order_status.get("status", "").lower()
            filled_count = order_status.get("filled_count", 0)
            avg_fill_price = order_status.get("avg_fill_price")
            if filled_count > 0 or status in {"filled", "executed"}:
                fill_price = int(avg_fill_price or trade.get("limit_price_cents") or 0)
                update_trade_status(
                    TRADES_DB_PATH,
                    trade["id"],
                    status="FILLED",
                    fill_price_cents=fill_price,
                    filled_at_utc=datetime.now(ZoneInfo("UTC")).isoformat(),
                    last_checked_at_utc=datetime.now(ZoneInfo("UTC")).isoformat(),
                )
                filled.append(
                    {
                        "market_ticker": trade["market_ticker"],
                        "side": trade["side"],
                        "fill_price": fill_price,
                        "order_id": order_id,
                    }
                )
                continue

            if args.dry_run:
                cancelled.append(
                    {
                        "market_ticker": trade["market_ticker"],
                        "side": trade["side"],
                        "limit": trade.get("limit_price_cents"),
                        "order_id": order_id,
                    }
                )
                continue

            client.cancel_order(order_id)
            update_trade_status(
                TRADES_DB_PATH,
                trade["id"],
                status="CANCELLED_TIMEOUT",
                last_checked_at_utc=datetime.now(ZoneInfo("UTC")).isoformat(),
            )
            cancelled.append(
                {
                    "market_ticker": trade["market_ticker"],
                    "side": trade["side"],
                    "limit": trade.get("limit_price_cents"),
                    "order_id": order_id,
                }
            )
        except Exception as exc:
            failed.append(
                {
                    "market_ticker": trade["market_ticker"],
                    "side": trade["side"],
                    "error": str(exc)[:120],
                }
            )
            update_trade_status(
                TRADES_DB_PATH,
                trade["id"],
                status="PLACED",
                last_checked_at_utc=datetime.now(ZoneInfo("UTC")).isoformat(),
            )

    post_discord(
        format_summary(
            trade_date,
            ttl_minutes,
            filled,
            cancelled,
            still_open,
            failed,
            args.dry_run,
        )
    )

    print(
        f"Reaper complete: filled={len(filled)} cancelled={len(cancelled)} "
        f"still_open={len(still_open)} failed={len(failed)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
