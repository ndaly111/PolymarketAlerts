"""
Discord notification utilities.

Consolidates Discord posting from props_value.py, props_auto_trade.py,
and sports_auto_trade.py.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

import requests

# Timezone for timestamps
ET = ZoneInfo("America/New_York")


def post_discord(
    message: str,
    webhook_url: Optional[str] = None,
    mention: Optional[str] = None,
    dry_run: bool = False,
    max_length: int = 1990,
) -> bool:
    """
    Post a message to Discord via webhook.

    Args:
        message: Message content
        webhook_url: Discord webhook URL (or uses DISCORD_WEBHOOK_URL env var)
        mention: Optional role/user mention to prepend (e.g., "<@&123456>")
        dry_run: If True, print message instead of posting
        max_length: Maximum message length (Discord limit is 2000)

    Returns:
        True if message was sent successfully (or dry_run), False otherwise
    """
    webhook = webhook_url or os.getenv("DISCORD_WEBHOOK_URL", "").strip()

    if not webhook:
        return False

    # Prepend mention if provided
    content = f"{mention}\n{message}" if mention else message

    # Truncate if too long
    if len(content) > max_length:
        content = content[:max_length - 10] + "\n...(trimmed)"

    if dry_run:
        print("[DRY_RUN] Would post to Discord:")
        print(content)
        return True

    try:
        payload = {
            "content": content,
            "allowed_mentions": {"parse": ["users", "roles", "everyone"]},
        }
        r = requests.post(webhook, json=payload, timeout=20)
        return r.status_code < 400
    except Exception as e:
        print(f"[warn] Discord post failed: {e}")
        return False


def format_timestamp() -> str:
    """Get current ET timestamp string."""
    return datetime.now(ET).strftime("%I:%M %p ET")


def format_trade_notification(
    status: str,
    player_name: str,
    stat_type: str,
    line: float,
    side: str,
    order_id: str,
    limit_price: int,
    fair_prob: float,
    edge: float,
    books_count: int,
    fill_price: Optional[int] = None,
    net_edge: Optional[float] = None,
) -> str:
    """
    Format a trade notification message for Discord.

    Args:
        status: Trade status (e.g., "Order Placed", "FILLED", "Cancelled")
        player_name: Player name
        stat_type: Stat type (e.g., "points")
        line: Line value (e.g., 25.5)
        side: Side (e.g., "OVER")
        order_id: Kalshi order ID
        limit_price: Limit price in cents
        fair_prob: Fair probability
        edge: Raw edge
        books_count: Number of books in consensus
        fill_price: Fill price in cents (if filled)
        net_edge: Net edge after fees (if available)
    """
    timestamp = format_timestamp()

    # Calculate EV
    price_to_use = fill_price if fill_price else limit_price
    actual_edge = fair_prob - (price_to_use / 100.0)
    ev_cents = actual_edge * 100

    edge_display = net_edge if net_edge is not None else edge

    lines = [
        f"**{status}** | {player_name} {stat_type} {line:.0f}+ {side}",
        f"Order ID: {order_id}",
    ]

    if fill_price:
        lines.append(f"Fill: {fill_price}¢ | Fair: {fair_prob:.1%} | Edge: {edge_display:.1%} | EV: {ev_cents:.1f}¢")
    else:
        lines.append(f"Limit: {limit_price}¢ | Fair: {fair_prob:.1%} | Edge: {edge_display:.1%} | EV: {ev_cents:.1f}¢")

    lines.append(f"Books: {books_count}")
    lines.append(f"_{timestamp}_")

    return "\n".join(lines)


def format_sports_notification(
    status: str,
    line_type: str,
    line_label: str,
    event_title: str,
    order_id: str,
    side: str,
    limit_price: int,
    fair_prob: float,
    edge: float,
    books_count: int,
    fill_price: Optional[int] = None,
    net_edge: Optional[float] = None,
) -> str:
    """
    Format a sports trade notification message for Discord.

    Args:
        status: Trade status
        line_type: Line type (e.g., "moneyline", "spread", "total")
        line_label: Line label (e.g., "Lakers ML", "Over 215.5")
        event_title: Event title
        order_id: Kalshi order ID
        side: Side (e.g., "YES", "NO")
        limit_price: Limit price in cents
        fair_prob: Fair probability
        edge: Raw edge
        books_count: Number of books
        fill_price: Fill price in cents (if filled)
        net_edge: Net edge after fees
    """
    timestamp = format_timestamp()

    price_to_use = fill_price if fill_price else limit_price
    actual_edge = fair_prob - (price_to_use / 100.0)
    ev_cents = actual_edge * 100

    edge_display = net_edge if net_edge is not None else edge

    lines = [
        f"**Sports {status}** | {line_type.upper()} {line_label}",
        f"Event: {event_title}",
        f"Order ID: {order_id}",
    ]

    if fill_price:
        lines.append(f"Side: {side} | Fill: {fill_price}¢ | Fair: {fair_prob:.1%} | Edge: {edge_display:.1%} | EV: {ev_cents:.1f}¢")
    else:
        lines.append(f"Side: {side} | Limit: {limit_price}¢ | Fair: {fair_prob:.1%} | Edge: {edge_display:.1%} | EV: {ev_cents:.1f}¢")

    lines.append(f"Books: {books_count}")
    lines.append(f"_{timestamp}_")

    return "\n".join(lines)


def format_edge_warning(
    player_name: str,
    stat_type: str,
    line: float,
    side: str,
    edge: float,
    net_edge: float,
    threshold: float = 0.30,
) -> str:
    """Format a suspicious edge warning for Discord."""
    timestamp = format_timestamp()

    return (
        f"**\u26a0\ufe0f Suspicious Edge** | {player_name} {stat_type} {line:.0f}+ {side}\n"
        f"Raw edge: {edge:.1%} | Net edge: {net_edge:.1%}\n"
        f"Edge exceeds {threshold:.0%} threshold - may be bad data\n"
        f"_{timestamp}_"
    )


def format_scan_summary(
    markets_scanned: int,
    opportunities_found: int,
    trades_filled: int,
    trades_attempted: int,
    trades_today: int,
    max_trades_per_day: int,
    top_opportunities: list,
    pipeline_type: str = "Props",
) -> str:
    """
    Format a scan summary message for Discord.

    Args:
        markets_scanned: Number of markets scanned
        opportunities_found: Number of opportunities found
        trades_filled: Trades filled this session
        trades_attempted: Trades attempted this session
        trades_today: Total trades today
        max_trades_per_day: Max trades allowed per day
        top_opportunities: List of top opportunity dicts
        pipeline_type: "Props" or "Sports"
    """
    timestamp = format_timestamp()

    lines = [
        f"**{pipeline_type} Scan Summary** | {timestamp}",
        f"Trades today: {trades_today}/{max_trades_per_day} (filled)",
        f"Markets scanned: {markets_scanned}",
        f"Opportunities: {opportunities_found}",
        f"This session: {trades_filled} filled ({trades_attempted} attempted)",
    ]

    if top_opportunities:
        lines.append("\nTop opportunities:")
        for opp in top_opportunities[:5]:
            if pipeline_type == "Sports":
                lines.append(
                    f"- {opp.get('line_type', '')} {opp.get('line_label', '')}: "
                    f"{opp.get('edge', 0)*100:.1f}% edge @ {opp.get('kalshi_ask', 0)}¢"
                )
            else:
                lines.append(
                    f"- {opp.get('player_name', '')} {opp.get('stat_type', '')} {opp.get('line', 0):.0f}+: "
                    f"{opp.get('edge', 0)*100:.1f}% edge @ {opp.get('kalshi_ask', 0)}¢"
                )

    return "\n".join(lines)
