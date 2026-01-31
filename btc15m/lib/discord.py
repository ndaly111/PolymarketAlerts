"""Discord webhook integration for BTC 15-min alerts."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Optional

import requests


def get_webhook_url() -> Optional[str]:
    """Get Discord webhook URL from environment."""
    return os.getenv("BTC_15MIN_DISCORD")


def send_discord_alert(
    title: str,
    description: str,
    color: int = 0x00FF00,  # Green
    fields: Optional[list[dict]] = None,
    webhook_url: Optional[str] = None,
) -> bool:
    """Send an alert to Discord via webhook.

    Args:
        title: Embed title
        description: Embed description
        color: Embed color (hex int)
        fields: List of field dicts with 'name' and 'value'
        webhook_url: Webhook URL (uses env var if not provided)

    Returns:
        True if successful, False otherwise
    """
    url = webhook_url or get_webhook_url()
    if not url:
        print("[discord] No webhook URL configured")
        return False

    embed = {
        "title": title,
        "description": description,
        "color": color,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if fields:
        embed["fields"] = [
            {"name": f["name"], "value": str(f["value"]), "inline": f.get("inline", True)}
            for f in fields
        ]

    payload = {"embeds": [embed]}

    try:
        resp = requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        resp.raise_for_status()
        return True
    except Exception as e:
        print(f"[discord] Failed to send alert: {e}")
        return False


def send_trade_alert(
    side: str,
    market_ticker: str,
    strike_price: float,
    model_prob: float,
    market_prob: float,
    edge: float,
    btc_price: float,
    time_to_expiry_sec: int,
    webhook_url: Optional[str] = None,
) -> bool:
    """Send a trade signal alert."""
    color = 0x00FF00 if side == "UP" else 0xFF0000  # Green for UP, Red for DOWN
    emoji = "🟢" if side == "UP" else "🔴"

    title = f"{emoji} BTC 15-Min Signal: {side}"
    description = f"**{market_ticker}**\nStrike: ${strike_price:,.0f}"

    fields = [
        {"name": "BTC Price", "value": f"${btc_price:,.2f}"},
        {"name": "Model Prob", "value": f"{model_prob*100:.1f}%"},
        {"name": "Market Prob", "value": f"{market_prob*100:.1f}%"},
        {"name": "Edge", "value": f"{edge*100:+.1f}%"},
        {"name": "Expires In", "value": f"{time_to_expiry_sec//60}m {time_to_expiry_sec%60}s"},
    ]

    return send_discord_alert(
        title=title,
        description=description,
        color=color,
        fields=fields,
        webhook_url=webhook_url,
    )


def send_settlement_alert(
    market_ticker: str,
    side: str,
    outcome: str,
    pnl: float,
    strike_price: float,
    settlement_price: float,
    webhook_url: Optional[str] = None,
) -> bool:
    """Send a trade settlement alert."""
    if outcome == "WIN":
        color = 0x00FF00
        emoji = "✅"
    else:
        color = 0xFF0000
        emoji = "❌"

    title = f"{emoji} Trade Settled: {outcome}"
    description = f"**{market_ticker}**"

    fields = [
        {"name": "Side", "value": side},
        {"name": "Strike", "value": f"${strike_price:,.0f}"},
        {"name": "Settlement", "value": f"${settlement_price:,.2f}"},
        {"name": "P&L", "value": f"${pnl:+.2f}"},
    ]

    return send_discord_alert(
        title=title,
        description=description,
        color=color,
        fields=fields,
        webhook_url=webhook_url,
    )


def send_summary_alert(
    total_trades: int,
    wins: int,
    losses: int,
    total_pnl: float,
    win_rate: Optional[float] = None,
    webhook_url: Optional[str] = None,
) -> bool:
    """Send a daily/session summary alert."""
    color = 0x00FF00 if total_pnl >= 0 else 0xFF0000

    title = "📊 BTC 15-Min Trading Summary"
    description = f"Session results"

    fields = [
        {"name": "Total Trades", "value": str(total_trades)},
        {"name": "Wins", "value": str(wins)},
        {"name": "Losses", "value": str(losses)},
        {"name": "Win Rate", "value": f"{win_rate*100:.1f}%" if win_rate else "N/A"},
        {"name": "Total P&L", "value": f"${total_pnl:+.2f}", "inline": False},
    ]

    return send_discord_alert(
        title=title,
        description=description,
        color=color,
        fields=fields,
        webhook_url=webhook_url,
    )
