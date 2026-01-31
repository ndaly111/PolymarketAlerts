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
    cumulative_pnl: Optional[float] = None,
    total_trades: Optional[int] = None,
    win_rate: Optional[float] = None,
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

    # Add cumulative stats if provided
    if cumulative_pnl is not None:
        fields.append({"name": "Cumulative P&L", "value": f"${cumulative_pnl:+.2f}", "inline": False})
    if total_trades is not None and win_rate is not None:
        fields.append({"name": "Record", "value": f"{total_trades} trades | {win_rate*100:.1f}% win rate", "inline": False})

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
    cumulative_pnl: Optional[float] = None,
    total_record: Optional[str] = None,
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

    # Add cumulative stats if provided
    if cumulative_pnl is not None:
        fields.append({"name": "Cumulative P&L", "value": f"${cumulative_pnl:+.2f}", "inline": False})
    if total_record:
        fields.append({"name": "Record", "value": total_record, "inline": False})

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


def send_edge_analysis_alert(
    prediction_direction: str,
    prediction_confidence: float,
    market_ask_price: float,
    edge: float,
    threshold: float,
    btc_price: float,
    trade_executed: bool,
    no_trade_reason: Optional[str] = None,
    market_ticker: Optional[str] = None,
    strike_price: Optional[float] = None,
    cumulative_pnl: Optional[float] = None,
    total_record: Optional[str] = None,
    webhook_url: Optional[str] = None,
) -> bool:
    """Send edge analysis update every run.

    Shows whether edge was found and the numbers behind the decision.
    """
    # Determine if edge was found
    edge_found = edge > 0 and prediction_confidence >= threshold

    if trade_executed:
        title = "📈 EDGE FOUND - Trade Executed"
        color = 0x00FF00  # Green
        status = "✅ Trade placed"
    elif edge_found and no_trade_reason:
        title = "⚠️ EDGE FOUND - No Trade"
        color = 0xFFA500  # Orange
        status = f"⚠️ {no_trade_reason}"
    elif edge > 0:
        title = "📊 Edge Analysis - Marginal Edge"
        color = 0xFFFF00  # Yellow
        status = f"❌ Below threshold ({prediction_confidence*100:.1f}% < {threshold*100:.0f}%)"
    else:
        title = "📊 Edge Analysis - No Edge"
        color = 0x808080  # Gray
        status = f"❌ {no_trade_reason or 'Negative edge'}"

    description = f"BTC: ${btc_price:,.2f}"
    if market_ticker:
        description += f"\nMarket: {market_ticker}"
    if strike_price:
        description += f"\nStrike: ${strike_price:,.0f}"

    fields = [
        {"name": "Prediction", "value": f"{prediction_direction}", "inline": True},
        {"name": "ML Confidence", "value": f"{prediction_confidence*100:.1f}%", "inline": True},
        {"name": "Threshold", "value": f"{threshold*100:.0f}%", "inline": True},
        {"name": "Market Ask", "value": f"{market_ask_price*100:.1f}%", "inline": True},
        {"name": "Edge", "value": f"{edge*100:+.2f}%", "inline": True},
        {"name": "Decision", "value": status, "inline": True},
    ]

    # Edge breakdown
    edge_calc = f"**Edge Calculation:**\nML Confidence - Market Ask = Edge\n{prediction_confidence*100:.1f}% - {market_ask_price*100:.1f}% = **{edge*100:+.2f}%**"
    if edge > 0:
        edge_calc += f"\n\n✅ Positive edge (we think it's worth more)"
    else:
        edge_calc += f"\n\n❌ Negative edge (market is right or we'd overpay)"

    fields.append({"name": "📐 Analysis", "value": edge_calc, "inline": False})

    # Add cumulative stats
    if cumulative_pnl is not None:
        fields.append({"name": "Cumulative P&L", "value": f"${cumulative_pnl:+.2f}", "inline": True})
    if total_record:
        fields.append({"name": "Record", "value": total_record, "inline": True})

    return send_discord_alert(
        title=title,
        description=description,
        color=color,
        fields=fields,
        webhook_url=webhook_url,
    )


def send_pending_status_alert(
    pending_trades: list,
    current_price: float,
    webhook_url: Optional[str] = None,
) -> bool:
    """Send status update for pending trades showing preliminary win/loss."""
    if not pending_trades:
        return False

    title = "⏳ Pending Trades Status"
    description = f"Current BTC: ${current_price:,.2f}"

    fields = []
    total_preliminary_pnl = 0

    for trade in pending_trades[:5]:  # Limit to 5 most recent
        strike = trade.get('strike_price', 0)
        side = trade.get('side', 'UP')
        entry_price = trade.get('entry_price', 0.5)
        stake = trade.get('stake', 1)

        # Determine if currently winning
        if side == "UP":
            currently_winning = current_price > strike
        else:
            currently_winning = current_price < strike

        # Calculate preliminary P&L
        if currently_winning:
            prelim_pnl = (1.0 - entry_price) * stake
            status = "🟢 WINNING"
        else:
            prelim_pnl = -entry_price * stake
            status = "🔴 LOSING"

        total_preliminary_pnl += prelim_pnl

        # Distance from strike
        diff = current_price - strike
        diff_pct = (diff / strike) * 100

        fields.append({
            "name": f"{side} @ ${strike:,.0f}",
            "value": f"{status}\nPrice diff: ${diff:+,.0f} ({diff_pct:+.2f}%)\nPrelim P&L: ${prelim_pnl:+.2f}",
            "inline": True,
        })

    # Add total
    fields.append({
        "name": "📊 Total Preliminary",
        "value": f"${total_preliminary_pnl:+.2f}",
        "inline": False,
    })

    color = 0x00FF00 if total_preliminary_pnl >= 0 else 0xFF0000

    return send_discord_alert(
        title=title,
        description=description,
        color=color,
        fields=fields,
        webhook_url=webhook_url,
    )


def send_ml_training_alert(
    total_samples: int,
    live_samples: int,
    historical_samples: int,
    model_name: str,
    accuracy: float,
    profit_weighted: bool,
    weight_range: Optional[tuple] = None,
    cumulative_pnl: Optional[float] = None,
    total_record: Optional[str] = None,
    live_accuracy: Optional[float] = None,
    webhook_url: Optional[str] = None,
) -> bool:
    """Send ML model training/retraining update.

    Args:
        total_samples: Total training samples used
        live_samples: Number of live trading samples
        historical_samples: Number of historical samples
        model_name: Name of the best model (e.g. "LightGBM")
        accuracy: Test accuracy
        profit_weighted: Whether profit-weighted training was used
        weight_range: (min, max) weight range if profit_weighted
        cumulative_pnl: Total P&L so far
        total_record: Win/loss record string
        live_accuracy: Accuracy on live trades only
    """
    title = "🤖 ML Model Retrained"
    color = 0x9B59B6  # Purple

    description = f"**{model_name}** model updated"
    if profit_weighted:
        description += "\n💰 Profit-weighted training enabled"

    fields = [
        {"name": "Total Samples", "value": f"{total_samples:,}", "inline": True},
        {"name": "Historical", "value": f"{historical_samples:,}", "inline": True},
        {"name": "Live", "value": f"{live_samples:,}", "inline": True},
        {"name": "Test Accuracy", "value": f"{accuracy*100:.1f}%", "inline": True},
    ]

    if profit_weighted and weight_range:
        fields.append({
            "name": "Weight Range",
            "value": f"{weight_range[0]:.1f}x - {weight_range[1]:.1f}x",
            "inline": True
        })

    if live_accuracy is not None:
        fields.append({
            "name": "Live Accuracy",
            "value": f"{live_accuracy*100:.1f}%",
            "inline": True
        })

    # Performance stats
    if cumulative_pnl is not None:
        emoji = "📈" if cumulative_pnl >= 0 else "📉"
        fields.append({
            "name": f"{emoji} Cumulative P&L",
            "value": f"${cumulative_pnl:+.2f}",
            "inline": True
        })

    if total_record:
        fields.append({
            "name": "Record",
            "value": total_record,
            "inline": True
        })

    # Training mode explanation
    if profit_weighted:
        fields.append({
            "name": "📊 Training Mode",
            "value": "Samples weighted by |P&L| - big wins/losses have more influence on model",
            "inline": False
        })

    return send_discord_alert(
        title=title,
        description=description,
        color=color,
        fields=fields,
        webhook_url=webhook_url,
    )
