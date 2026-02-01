"""Discord webhook integration for BTC 15-min alerts."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict

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
    profit_predictor_prob: Optional[float] = None,
    contracts: int = 1,
    entry_cost: Optional[float] = None,
    potential_profit: Optional[float] = None,
    key_indicators: Optional[dict] = None,
    webhook_url: Optional[str] = None,
) -> bool:
    """Send a trade signal alert with detailed reasoning."""
    color = 0x00FF00 if side == "UP" else 0xFF0000
    emoji = "🟢" if side == "UP" else "🔴"

    # Calculate distance to strike
    distance = btc_price - strike_price
    distance_pct = (distance / strike_price) * 100

    title = f"{emoji} BTC 15-Min Trade: {side}"

    # Build insightful description
    if side == "UP":
        position_desc = f"above ${strike_price:,.0f}" if distance > 0 else f"${abs(distance):,.0f} below strike"
    else:
        position_desc = f"below ${strike_price:,.0f}" if distance < 0 else f"${abs(distance):,.0f} above strike"

    description = f"**{market_ticker}**\nBTC is currently {position_desc}"

    fields = [
        {"name": "💰 Entry", "value": f"{contracts}x @ {market_prob*100:.0f}¢", "inline": True},
        {"name": "🎯 Strike", "value": f"${strike_price:,.0f}", "inline": True},
        {"name": "⏱️ Expires", "value": f"{time_to_expiry_sec//60}m", "inline": True},
    ]

    # Risk/Reward section
    if entry_cost and potential_profit:
        rr_ratio = potential_profit / entry_cost if entry_cost > 0 else 0
        fields.append({"name": "💵 Risk", "value": f"${entry_cost:.2f}", "inline": True})
        fields.append({"name": "🎁 Reward", "value": f"${potential_profit:.2f}", "inline": True})
        fields.append({"name": "📊 R:R", "value": f"1:{rr_ratio:.1f}", "inline": True})

    # Why we're taking this trade
    why_trading = f"• ML predicts **{model_prob*100:.0f}%** chance of {side}\n"
    why_trading += f"• Market prices at **{market_prob*100:.0f}%**\n"
    why_trading += f"• Edge: **{edge*100:+.1f}%** (we think market is wrong)"

    if profit_predictor_prob:
        profit_emoji = "✅" if profit_predictor_prob > 0.6 else "⚠️"
        why_trading += f"\n• Profit model: {profit_emoji} **{profit_predictor_prob*100:.0f}%** profitable"

    fields.append({"name": "📈 Why This Trade", "value": why_trading, "inline": False})

    # Key indicators if provided
    if key_indicators:
        ind_text = ""
        for k, v in list(key_indicators.items())[:4]:
            ind_text += f"• {k}: {v}\n"
        if ind_text:
            fields.append({"name": "🔬 Key Signals", "value": ind_text.strip(), "inline": False})

    # Cumulative stats
    if cumulative_pnl is not None and total_trades:
        pnl_emoji = "📈" if cumulative_pnl >= 0 else "📉"
        stats = f"{pnl_emoji} **${cumulative_pnl:+.2f}** P&L"
        if win_rate:
            stats += f" | {win_rate*100:.0f}% win rate ({total_trades} trades)"
        fields.append({"name": "📊 Session Stats", "value": stats, "inline": False})

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
    entry_price: Optional[float] = None,
    model_confidence: Optional[float] = None,
    edge_at_entry: Optional[float] = None,
    contracts: int = 1,
    streak: Optional[int] = None,
    webhook_url: Optional[str] = None,
) -> bool:
    """Send a trade settlement alert with analysis."""
    if outcome == "WIN":
        color = 0x2ECC71  # Nice green
        emoji = "✅"
        result_text = "WINNER"
    else:
        color = 0xE74C3C  # Nice red
        emoji = "❌"
        result_text = "LOSS"

    title = f"{emoji} Trade Settled: {result_text}"

    # Calculate how close it was
    diff = settlement_price - strike_price
    diff_pct = (diff / strike_price) * 100

    # Determine if it was a close call
    if abs(diff_pct) < 0.1:
        closeness = "🎯 **RAZOR CLOSE** - within 0.1%!"
    elif abs(diff_pct) < 0.3:
        closeness = "😰 Close call - within 0.3%"
    elif abs(diff_pct) < 1:
        closeness = "Moderate margin"
    else:
        closeness = "Clear result" if outcome == "WIN" else "Missed by wide margin"

    # Build description
    if side == "UP":
        needed = f"BTC needed to be **above** ${strike_price:,.0f}"
        actual = f"Settled at ${settlement_price:,.2f} ({diff_pct:+.2f}%)"
    else:
        needed = f"BTC needed to be **below** ${strike_price:,.0f}"
        actual = f"Settled at ${settlement_price:,.2f} ({diff_pct:+.2f}%)"

    description = f"{needed}\n{actual}\n\n{closeness}"

    fields = [
        {"name": "💵 P&L", "value": f"**${pnl:+.2f}**", "inline": True},
        {"name": "📊 Contracts", "value": str(contracts), "inline": True},
    ]

    if entry_price:
        fields.append({"name": "🎫 Entry", "value": f"{entry_price*100:.0f}¢", "inline": True})

    # Trade analysis
    analysis = ""
    if model_confidence:
        if outcome == "WIN":
            analysis += f"• ML was {model_confidence*100:.0f}% confident ✓\n"
        else:
            analysis += f"• ML was {model_confidence*100:.0f}% confident ✗\n"

    if edge_at_entry:
        if outcome == "WIN":
            analysis += f"• {edge_at_entry*100:+.1f}% edge paid off\n"
        else:
            analysis += f"• {edge_at_entry*100:+.1f}% edge didn't hold\n"

    if abs(diff_pct) < 0.5 and outcome == "LOSS":
        analysis += "• 💡 Was close - model direction was right\n"

    if analysis:
        fields.append({"name": "🔍 Analysis", "value": analysis.strip(), "inline": False})

    # Running totals
    if cumulative_pnl is not None:
        pnl_emoji = "📈" if cumulative_pnl >= 0 else "📉"
        total_text = f"{pnl_emoji} **${cumulative_pnl:+.2f}**"
        if total_record:
            total_text += f" ({total_record})"
        fields.append({"name": "📊 Total P&L", "value": total_text, "inline": True})

    # Streak info
    if streak:
        if streak > 0:
            fields.append({"name": "🔥 Streak", "value": f"{streak}W in a row!", "inline": True})
        elif streak < -2:
            fields.append({"name": "❄️ Streak", "value": f"{abs(streak)}L in a row", "inline": True})

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


def send_scan_log(
    message: str,
    btc_price: Optional[float] = None,
    prediction: Optional[str] = None,
    probability: Optional[float] = None,
    webhook_url: Optional[str] = None,
) -> bool:
    """Send a scan cycle log to Discord."""
    url = webhook_url or get_webhook_url()
    if not url:
        return False

    # Simple text message for logs - include AWS tag
    content = f"**[AWS]** {message}"
    if btc_price:
        content += f" | BTC: ${btc_price:,.2f}"
    if prediction and probability:
        content += f" | {prediction} {probability*100:.1f}%"

    try:
        resp = requests.post(
            url,
            json={"content": content},
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        resp.raise_for_status()
        return True
    except Exception:
        return False


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
    title = "🤖 [AWS] ML Model Retrained"
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


def send_threshold_update_alert(
    old_threshold: float,
    new_threshold: float,
    reason: str,
    bucket_analysis: list,
    cumulative_pnl: Optional[float] = None,
    total_record: Optional[str] = None,
    webhook_url: Optional[str] = None,
) -> bool:
    """Send alert when confidence threshold is auto-adjusted."""
    if new_threshold > old_threshold:
        title = "📈 [AWS] Threshold RAISED"
        color = 0xFFA500  # Orange - caution
        emoji = "⬆️"
    else:
        title = "📉 [AWS] Threshold LOWERED"
        color = 0x00FF00  # Green - more trades
        emoji = "⬇️"

    description = f"{emoji} **{old_threshold*100:.0f}% → {new_threshold*100:.0f}%**\n\n{reason}"

    fields = []

    # Show bucket analysis
    if bucket_analysis:
        bucket_text = ""
        for b in bucket_analysis[:6]:  # Limit to 6 buckets
            pnl_emoji = "✅" if b["avg_pnl"] > 0 else "❌"
            bucket_text += f"{pnl_emoji} {b['bucket_min']*100:.0f}-{b['bucket_max']*100:.0f}%: ${b['avg_pnl']:+.2f}/trade ({b['trades']} trades)\n"

        if bucket_text:
            fields.append({
                "name": "📊 Confidence Bucket Analysis",
                "value": bucket_text,
                "inline": False
            })

    if cumulative_pnl is not None:
        emoji = "📈" if cumulative_pnl >= 0 else "📉"
        fields.append({
            "name": f"{emoji} Cumulative P&L",
            "value": f"${cumulative_pnl:+.2f}",
            "inline": True
        })

    if total_record:
        fields.append({
            "name": "📋 Record",
            "value": total_record,
            "inline": True
        })

    return send_discord_alert(
        title=title,
        description=description,
        color=color,
        fields=fields,
        webhook_url=webhook_url,
    )


def send_paper_mode_alert(
    activated: bool,
    reason: str,
    recent_pnl: float,
    consecutive_losses: int,
    cumulative_pnl: Optional[float] = None,
    total_record: Optional[str] = None,
    webhook_url: Optional[str] = None,
) -> bool:
    """Send alert when paper mode is activated or deactivated."""
    if activated:
        title = "🛑 [AWS] PAPER MODE ACTIVATED"
        color = 0xFF0000  # Red
        description = f"**Live trading paused**\n\n{reason}"
    else:
        title = "✅ [AWS] LIVE TRADING RESUMED"
        color = 0x00FF00  # Green
        description = f"**Paper mode deactivated**\n\n{reason}"

    fields = [
        {"name": "Recent P&L (last 10)", "value": f"${recent_pnl:+.2f}", "inline": True},
        {"name": "Consecutive Losses", "value": str(consecutive_losses), "inline": True},
    ]

    if cumulative_pnl is not None:
        fields.append({
            "name": "Total P&L",
            "value": f"${cumulative_pnl:+.2f}",
            "inline": True
        })

    if total_record:
        fields.append({
            "name": "📋 Record",
            "value": total_record,
            "inline": True
        })

    if activated:
        fields.append({
            "name": "📋 What happens now?",
            "value": "• Predictions continue to be logged\n• No real money trades\n• ML continues learning\n• Auto-resumes after 5 profitable paper trades",
            "inline": False
        })

    return send_discord_alert(
        title=title,
        description=description,
        color=color,
        fields=fields,
        webhook_url=webhook_url,
    )


def send_cumulative_results_alert(
    total_pnl: float,
    total_trades: int,
    wins: int,
    losses: int,
    pending: int,
    win_rate: Optional[float] = None,
    avg_edge: Optional[float] = None,
    current_threshold: Optional[float] = None,
    paper_mode: bool = False,
    webhook_url: Optional[str] = None,
) -> bool:
    """Send cumulative trade results summary (settled trades only)."""
    if total_pnl >= 0:
        title = "📊 [AWS] Trade Results Summary"
        color = 0x00FF00  # Green
    else:
        title = "📊 [AWS] Trade Results Summary"
        color = 0xFF6B6B  # Light red

    mode_str = "📝 PAPER" if paper_mode else "💰 LIVE"
    description = f"**{mode_str} MODE**\nSettled trades only"

    fields = [
        {"name": "Total P&L", "value": f"${total_pnl:+.2f}", "inline": True},
        {"name": "Trades", "value": f"{total_trades} ({pending} pending)", "inline": True},
        {"name": "Record", "value": f"{wins}W / {losses}L", "inline": True},
    ]

    if win_rate is not None:
        fields.append({"name": "Win Rate", "value": f"{win_rate*100:.1f}%", "inline": True})

    if avg_edge is not None:
        fields.append({"name": "Avg Edge", "value": f"{avg_edge*100:+.1f}%", "inline": True})

    if current_threshold is not None:
        fields.append({"name": "Threshold", "value": f"{current_threshold*100:.0f}%", "inline": True})

    return send_discord_alert(
        title=title,
        description=description,
        color=color,
        fields=fields,
        webhook_url=webhook_url,
    )


def send_hourly_trade_alert(
    range_lower: float,
    range_upper: float,
    btc_price: float,
    side: str,
    entry_price: float,
    contracts: int,
    edge: float,
    ev_score: float,
    model_prob: float,
    time_to_expiry_min: int,
    position_vs_range: str,
    total_opportunities: int,
    cumulative_pnl: Optional[float] = None,
    total_record: Optional[str] = None,
    bankroll: Optional[float] = None,
    webhook_url: Optional[str] = None,
) -> bool:
    """Send alert for hourly range market trade with full context."""
    color = 0x3498DB  # Blue for hourly

    # Determine position context
    if position_vs_range == "inside":
        position_emoji = "🎯"
        position_text = f"BTC is **inside** the range"
        confidence_context = "High confidence - already in target zone"
    elif position_vs_range == "below":
        position_emoji = "⬆️"
        distance = range_lower - btc_price
        position_text = f"BTC is **${distance:,.0f} below** range"
        confidence_context = "Needs upward move to win"
    else:  # above
        position_emoji = "⬇️"
        distance = btc_price - range_upper
        position_text = f"BTC is **${distance:,.0f} above** range"
        confidence_context = "Needs downward move to win"

    title = f"💰 Hourly Range Trade {position_emoji}"

    range_width = range_upper - range_lower
    description = (
        f"**Range: ${range_lower:,.0f} - ${range_upper:,.0f}**\n"
        f"(${range_width:,.0f} wide bucket)\n\n"
        f"{position_text}"
    )

    # Entry details
    cost = entry_price * contracts
    potential_win = (1 - entry_price) * contracts * 0.93  # After 7% fee
    rr_ratio = potential_win / cost if cost > 0 else 0

    fields = [
        {"name": "📍 BTC Now", "value": f"${btc_price:,.0f}", "inline": True},
        {"name": "🎫 Entry", "value": f"{contracts}x {side.upper()} @ {entry_price*100:.0f}¢", "inline": True},
        {"name": "⏱️ Expires", "value": f"{time_to_expiry_min}m", "inline": True},
        {"name": "💵 Cost", "value": f"${cost:.2f}", "inline": True},
        {"name": "🎁 Potential", "value": f"${potential_win:.2f}", "inline": True},
        {"name": "📊 R:R", "value": f"1:{rr_ratio:.1f}", "inline": True},
    ]

    # Why this trade
    why_text = (
        f"• Model: **{model_prob*100:.0f}%** prob in range\n"
        f"• Market: **{entry_price*100:.0f}%** implied\n"
        f"• Edge: **{edge*100:+.1f}%**\n"
        f"• EV Score: **{ev_score:.3f}** (#{1} of {total_opportunities})"
    )
    fields.append({"name": "📈 Why This Trade", "value": why_text, "inline": False})

    # Context
    fields.append({"name": "🎯 Situation", "value": confidence_context, "inline": False})

    # Position sizing context
    if bankroll:
        pct_of_roll = (cost / bankroll) * 100
        fields.append({"name": "💼 Position", "value": f"{pct_of_roll:.1f}% of ${bankroll:.2f} bankroll", "inline": False})

    # Running stats
    if cumulative_pnl is not None:
        pnl_emoji = "📈" if cumulative_pnl >= 0 else "📉"
        stats = f"{pnl_emoji} ${cumulative_pnl:+.2f}"
        if total_record:
            stats += f" ({total_record})"
        fields.append({"name": "📊 Hourly Stats", "value": stats, "inline": False})

    return send_discord_alert(
        title=title,
        description=description,
        color=color,
        fields=fields,
        webhook_url=webhook_url,
    )


def send_hourly_settlement_alert(
    range_lower: float,
    range_upper: float,
    settlement_price: float,
    side: str,
    outcome: str,
    pnl: float,
    entry_price: float,
    contracts: int,
    model_prob: float,
    cumulative_pnl: float,
    total_wins: int,
    total_losses: int,
    webhook_url: Optional[str] = None,
) -> bool:
    """Send settlement alert for hourly range trade with analysis."""
    if outcome == "WIN":
        color = 0x2ECC71
        emoji = "✅"
        result = "WINNER"
    else:
        color = 0xE74C3C
        emoji = "❌"
        result = "LOSS"

    title = f"{emoji} Hourly Trade: {result}"

    # Analyze the outcome
    in_range = range_lower <= settlement_price <= range_upper

    if in_range:
        range_position = f"**IN** range at ${settlement_price:,.2f}"
        if side == "yes":
            analysis = "✓ Correctly predicted BTC would land in range"
        else:
            analysis = "✗ Predicted out-of-range but BTC landed inside"
    else:
        if settlement_price < range_lower:
            diff = range_lower - settlement_price
            range_position = f"**${diff:,.0f} below** range"
        else:
            diff = settlement_price - range_upper
            range_position = f"**${diff:,.0f} above** range"

        if side == "no":
            analysis = "✓ Correctly predicted BTC would miss the range"
        else:
            analysis = f"✗ BTC missed the range by ${diff:,.0f}"

    description = (
        f"**Range: ${range_lower:,.0f} - ${range_upper:,.0f}**\n"
        f"Settlement: {range_position}"
    )

    fields = [
        {"name": "💵 P&L", "value": f"**${pnl:+.2f}**", "inline": True},
        {"name": "🎫 Position", "value": f"{contracts}x {side.upper()} @ {entry_price*100:.0f}¢", "inline": True},
        {"name": "🎯 Model", "value": f"{model_prob*100:.0f}% confident", "inline": True},
    ]

    fields.append({"name": "🔍 Analysis", "value": analysis, "inline": False})

    # Running totals
    pnl_emoji = "📈" if cumulative_pnl >= 0 else "📉"
    win_rate = total_wins / (total_wins + total_losses) * 100 if (total_wins + total_losses) > 0 else 0

    stats = f"{pnl_emoji} **${cumulative_pnl:+.2f}** | {total_wins}W/{total_losses}L ({win_rate:.0f}%)"
    fields.append({"name": "📊 Hourly Total", "value": stats, "inline": False})

    return send_discord_alert(
        title=title,
        description=description,
        color=color,
        fields=fields,
        webhook_url=webhook_url,
    )


def send_calibration_alert(
    calibration_data: List[Dict],
    is_overconfident: bool,
    adjustment_suggestion: str,
    current_threshold: float,
    webhook_url: Optional[str] = None,
) -> bool:
    """Send alert when model calibration is off."""
    if is_overconfident:
        title = "⚠️ Model Overconfidence Detected"
        color = 0xFFA500  # Orange
        description = "The model is predicting higher confidence than actual win rates justify."
    else:
        title = "💡 Model Underconfidence Detected"
        color = 0x3498DB  # Blue
        description = "The model is more accurate than its confidence suggests - potential edge opportunity."

    fields = []

    # Show calibration buckets
    cal_text = ""
    for bucket in calibration_data[:5]:
        expected = bucket.get('expected', 0)
        actual = bucket.get('actual', 0)
        diff = actual - expected
        trades = bucket.get('trades', 0)

        if abs(diff) < 5:
            emoji = "✅"
        elif diff > 0:
            emoji = "📈"  # Actual > expected (underconfident)
        else:
            emoji = "📉"  # Actual < expected (overconfident)

        cal_text += f"{emoji} **{bucket.get('range', '')}**: {actual:.0f}% actual vs {expected:.0f}% expected ({trades} trades)\n"

    if cal_text:
        fields.append({
            "name": "🎯 Calibration Analysis",
            "value": cal_text.strip(),
            "inline": False
        })

    # Suggestion
    fields.append({
        "name": "💡 Recommendation",
        "value": adjustment_suggestion,
        "inline": False
    })

    fields.append({
        "name": "Current Threshold",
        "value": f"{current_threshold*100:.0f}%",
        "inline": True
    })

    return send_discord_alert(
        title=title,
        description=description,
        color=color,
        fields=fields,
        webhook_url=webhook_url,
    )


def send_daily_summary(
    date_str: str,
    # 15-min stats
    fifteen_min_pnl: float,
    fifteen_min_trades: int,
    fifteen_min_wins: int,
    fifteen_min_win_rate: float,
    # Hourly stats
    hourly_pnl: float,
    hourly_trades: int,
    hourly_wins: int,
    hourly_win_rate: float,
    # Best/worst trades
    best_trade: Optional[Dict] = None,
    worst_trade: Optional[Dict] = None,
    # ML stats
    ml_accuracy: Optional[float] = None,
    profit_predictor_accuracy: Optional[float] = None,
    # Calibration data
    calibration_data: Optional[List[Dict]] = None,
    # Time analysis
    best_hour: Optional[Dict] = None,
    worst_hour: Optional[Dict] = None,
    # All-time stats
    all_time_pnl: Optional[float] = None,
    all_time_trades: Optional[int] = None,
    webhook_url: Optional[str] = None,
) -> bool:
    """Send comprehensive daily trading summary."""
    total_pnl = fifteen_min_pnl + hourly_pnl
    total_trades = fifteen_min_trades + hourly_trades
    total_wins = fifteen_min_wins + hourly_wins

    if total_pnl >= 0:
        color = 0x2ECC71  # Green
        emoji = "📈"
        verdict = "Profitable day!"
    else:
        color = 0xE74C3C  # Red
        emoji = "📉"
        verdict = "Losing day"

    title = f"{emoji} Daily Summary: {date_str}"
    description = f"**{verdict}**\nTotal P&L: **${total_pnl:+.2f}** across {total_trades} trades"

    fields = []

    # 15-min market summary
    if fifteen_min_trades > 0:
        fm_emoji = "✅" if fifteen_min_pnl >= 0 else "❌"
        fields.append({
            "name": "⏱️ 15-Min Markets",
            "value": f"{fm_emoji} ${fifteen_min_pnl:+.2f}\n{fifteen_min_wins}W/{fifteen_min_trades - fifteen_min_wins}L ({fifteen_min_win_rate:.0f}%)",
            "inline": True
        })

    # Hourly market summary
    if hourly_trades > 0:
        hr_emoji = "✅" if hourly_pnl >= 0 else "❌"
        fields.append({
            "name": "🕐 Hourly Markets",
            "value": f"{hr_emoji} ${hourly_pnl:+.2f}\n{hourly_wins}W/{hourly_trades - hourly_wins}L ({hourly_win_rate:.0f}%)",
            "inline": True
        })

    # Combined totals
    overall_win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0
    fields.append({
        "name": "📊 Combined",
        "value": f"**${total_pnl:+.2f}**\n{total_wins}W/{total_trades - total_wins}L ({overall_win_rate:.0f}%)",
        "inline": True
    })

    # Best trade
    if best_trade:
        fields.append({
            "name": "🏆 Best Trade",
            "value": f"+${best_trade.get('pnl', 0):.2f}\n{best_trade.get('market', 'N/A')}\n{best_trade.get('side', '')} @ {best_trade.get('entry', 0)*100:.0f}¢",
            "inline": True
        })

    # Worst trade
    if worst_trade:
        fields.append({
            "name": "💀 Worst Trade",
            "value": f"${worst_trade.get('pnl', 0):.2f}\n{worst_trade.get('market', 'N/A')}\n{worst_trade.get('side', '')} @ {worst_trade.get('entry', 0)*100:.0f}¢",
            "inline": True
        })

    # ML accuracy
    if ml_accuracy is not None or profit_predictor_accuracy is not None:
        ml_text = ""
        if ml_accuracy is not None:
            ml_text += f"Direction: {ml_accuracy:.0f}%\n"
        if profit_predictor_accuracy is not None:
            ml_text += f"Profit Pred: {profit_predictor_accuracy:.0f}%"
        fields.append({
            "name": "🤖 ML Accuracy",
            "value": ml_text.strip(),
            "inline": True
        })

    # Confidence calibration
    if calibration_data:
        cal_text = ""
        for bucket in calibration_data[:4]:
            expected = bucket.get('expected', 0)
            actual = bucket.get('actual', 0)
            diff = actual - expected
            cal_emoji = "✓" if abs(diff) < 10 else ("⬆️" if diff > 0 else "⬇️")
            cal_text += f"{cal_emoji} {bucket.get('range', '')}: {actual:.0f}% (exp {expected:.0f}%)\n"
        if cal_text:
            fields.append({
                "name": "🎯 Calibration",
                "value": cal_text.strip(),
                "inline": False
            })

    # Best/worst hours
    if best_hour or worst_hour:
        time_text = ""
        if best_hour:
            time_text += f"📈 Best: {best_hour.get('hour', 'N/A')} ({best_hour.get('pnl', 0):+.2f}, {best_hour.get('trades', 0)} trades)\n"
        if worst_hour:
            time_text += f"📉 Worst: {worst_hour.get('hour', 'N/A')} ({worst_hour.get('pnl', 0):+.2f}, {worst_hour.get('trades', 0)} trades)"
        if time_text:
            fields.append({
                "name": "⏰ Time Analysis",
                "value": time_text.strip(),
                "inline": False
            })

    # All-time stats
    if all_time_pnl is not None:
        at_emoji = "📈" if all_time_pnl >= 0 else "📉"
        fields.append({
            "name": f"{at_emoji} All-Time",
            "value": f"**${all_time_pnl:+.2f}** ({all_time_trades or 0} trades)",
            "inline": False
        })

    return send_discord_alert(
        title=title,
        description=description,
        color=color,
        fields=fields,
        webhook_url=webhook_url,
    )
