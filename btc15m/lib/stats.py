"""Stats compilation, JSON logging, and Discord reporting."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from lib.db import (
    get_paper_trade_summary,
    get_daily_stats,
    get_all_trades,
    get_pending_trades,
    DEFAULT_DB_PATH,
)
from lib.discord import send_discord_alert


STATS_DIR = Path(__file__).parent.parent / "data" / "stats"


def compile_all_stats(db_path: Path = DEFAULT_DB_PATH) -> dict:
    """Compile all trading statistics."""
    summary = get_paper_trade_summary(db_path)
    daily = get_daily_stats(db_path)
    pending = get_pending_trades(db_path)
    all_trades = get_all_trades(db_path)

    # Calculate streaks
    settled = [t for t in all_trades if t['outcome'] is not None]
    current_streak = 0
    streak_type = None
    max_win_streak = 0
    max_loss_streak = 0
    current_win = 0
    current_loss = 0

    for trade in reversed(settled):
        if trade['outcome'] == 'WIN':
            current_win += 1
            current_loss = 0
            max_win_streak = max(max_win_streak, current_win)
        else:
            current_loss += 1
            current_win = 0
            max_loss_streak = max(max_loss_streak, current_loss)

    if settled:
        last = settled[0]
        streak_type = last['outcome']
        current_streak = 1
        for trade in settled[1:]:
            if trade['outcome'] == streak_type:
                current_streak += 1
            else:
                break

    # Calculate average PnL
    settled_pnl = [t['pnl'] for t in settled if t['pnl'] is not None]
    avg_win = sum(p for p in settled_pnl if p > 0) / len([p for p in settled_pnl if p > 0]) if any(p > 0 for p in settled_pnl) else 0
    avg_loss = sum(p for p in settled_pnl if p < 0) / len([p for p in settled_pnl if p < 0]) if any(p < 0 for p in settled_pnl) else 0

    stats = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "all_time": {
            "total_trades": summary['total_trades'],
            "wins": summary['wins'],
            "losses": summary['losses'],
            "pending": summary['pending'],
            "win_rate": summary['win_rate'],
            "total_pnl": summary['total_pnl'],
            "avg_edge": summary['avg_edge'],
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "max_win_streak": max_win_streak,
            "max_loss_streak": max_loss_streak,
        },
        "today": {
            "date": daily['date'],
            "trades": daily['trades_today'],
            "wins": daily['wins'],
            "losses": daily['losses'],
            "pending": daily['pending'],
            "pnl": daily['pnl_today'],
            "win_rate": daily['win_rate'],
            "avg_confidence": daily['avg_confidence'],
        },
        "current_streak": {
            "type": streak_type,
            "count": current_streak,
        },
        "pending_trades": len(pending),
        "recent_trades": [
            {
                "id": t['id'],
                "timestamp": t['timestamp'],
                "market": t['market_ticker'],
                "side": t['side'],
                "entry_price": t['entry_price'],
                "model_prob": t['model_prob'],
                "edge": t['edge'],
                "outcome": t['outcome'],
                "pnl": t['pnl'],
            }
            for t in all_trades[:20]
        ],
    }

    return stats


def save_stats_json(stats: dict, filename: Optional[str] = None) -> Path:
    """Save stats to JSON file."""
    STATS_DIR.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = f"stats_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"

    filepath = STATS_DIR / filename
    with open(filepath, 'w') as f:
        json.dump(stats, f, indent=2, default=str)

    # Also save as latest.json
    latest = STATS_DIR / "latest.json"
    with open(latest, 'w') as f:
        json.dump(stats, f, indent=2, default=str)

    return filepath


def send_stats_discord(stats: dict) -> bool:
    """Send comprehensive stats to Discord."""
    all_time = stats['all_time']
    today = stats['today']
    streak = stats['current_streak']

    # Determine color based on PnL
    color = 0x00FF00 if all_time['total_pnl'] >= 0 else 0xFF0000

    title = "📊 BTC 15-Min Trading Dashboard"
    description = f"Updated: {stats['timestamp'][:19]} UTC"

    fields = [
        # All-time stats
        {"name": "📈 All-Time P&L", "value": f"${all_time['total_pnl']:+.2f}", "inline": True},
        {"name": "🎯 Win Rate", "value": f"{all_time['win_rate']*100:.1f}%" if all_time['win_rate'] else "N/A", "inline": True},
        {"name": "📊 Total Trades", "value": f"{all_time['total_trades']}", "inline": True},

        # Record
        {"name": "✅ Wins", "value": f"{all_time['wins']}", "inline": True},
        {"name": "❌ Losses", "value": f"{all_time['losses']}", "inline": True},
        {"name": "⏳ Pending", "value": f"{all_time['pending']}", "inline": True},

        # Averages
        {"name": "💰 Avg Win", "value": f"${all_time['avg_win']:+.2f}", "inline": True},
        {"name": "📉 Avg Loss", "value": f"${all_time['avg_loss']:+.2f}", "inline": True},
        {"name": "📐 Avg Edge", "value": f"{all_time['avg_edge']*100:.1f}%" if all_time['avg_edge'] else "N/A", "inline": True},

        # Today
        {"name": "📅 Today's P&L", "value": f"${today['pnl']:+.2f}", "inline": True},
        {"name": "📅 Today's Trades", "value": f"{today['trades']} ({today['wins']}W/{today['losses']}L)", "inline": True},

        # Streaks
        {"name": "🔥 Current Streak", "value": f"{streak['count']} {streak['type']}s" if streak['type'] else "N/A", "inline": True},
        {"name": "🏆 Best Win Streak", "value": f"{all_time['max_win_streak']}", "inline": True},
        {"name": "📉 Worst Loss Streak", "value": f"{all_time['max_loss_streak']}", "inline": True},
    ]

    return send_discord_alert(
        title=title,
        description=description,
        color=color,
        fields=fields,
    )


def generate_dashboard_html(stats: dict) -> str:
    """Generate HTML dashboard."""
    all_time = stats['all_time']
    today = stats['today']

    pnl_color = "green" if all_time['total_pnl'] >= 0 else "red"
    win_rate = all_time['win_rate'] * 100 if all_time['win_rate'] else 0

    trades_html = ""
    for t in stats['recent_trades'][:10]:
        outcome_class = "win" if t['outcome'] == 'WIN' else ("loss" if t['outcome'] == 'LOSS' else "pending")
        pnl_str = f"${t['pnl']:+.2f}" if t['pnl'] is not None else "Pending"
        trades_html += f"""
        <tr class="{outcome_class}">
            <td>{t['timestamp'][:19]}</td>
            <td>{t['side']}</td>
            <td>{t['model_prob']*100:.1f}%</td>
            <td>{t['edge']*100:+.1f}%</td>
            <td>{t['outcome'] or 'PENDING'}</td>
            <td>{pnl_str}</td>
        </tr>
        """

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>BTC 15-Min Trading Dashboard</title>
        <meta http-equiv="refresh" content="60">
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #1a1a2e; color: #eee; padding: 20px; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            h1 {{ color: #f39c12; text-align: center; }}
            .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
            .stat-card {{ background: #16213e; padding: 20px; border-radius: 10px; text-align: center; }}
            .stat-value {{ font-size: 2em; font-weight: bold; }}
            .stat-label {{ color: #888; margin-top: 5px; }}
            .green {{ color: #00ff88; }}
            .red {{ color: #ff4444; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #333; }}
            th {{ background: #16213e; color: #f39c12; }}
            tr.win {{ background: rgba(0,255,136,0.1); }}
            tr.loss {{ background: rgba(255,68,68,0.1); }}
            tr.pending {{ background: rgba(255,193,7,0.1); }}
            .updated {{ text-align: center; color: #666; margin-top: 20px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 BTC 15-Min Trading Dashboard</h1>

            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value {pnl_color}">${all_time['total_pnl']:+.2f}</div>
                    <div class="stat-label">All-Time P&L</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{win_rate:.1f}%</div>
                    <div class="stat-label">Win Rate</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{all_time['total_trades']}</div>
                    <div class="stat-label">Total Trades</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value green">{all_time['wins']}</div>
                    <div class="stat-label">Wins</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value red">{all_time['losses']}</div>
                    <div class="stat-label">Losses</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{all_time['pending']}</div>
                    <div class="stat-label">Pending</div>
                </div>
            </div>

            <h2>📅 Today's Stats</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value {'green' if today['pnl'] >= 0 else 'red'}">${today['pnl']:+.2f}</div>
                    <div class="stat-label">Today's P&L</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{today['trades']}</div>
                    <div class="stat-label">Trades Today</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{today['wins']}W / {today['losses']}L</div>
                    <div class="stat-label">Record</div>
                </div>
            </div>

            <h2>📜 Recent Trades</h2>
            <table>
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Side</th>
                        <th>Confidence</th>
                        <th>Edge</th>
                        <th>Outcome</th>
                        <th>P&L</th>
                    </tr>
                </thead>
                <tbody>
                    {trades_html}
                </tbody>
            </table>

            <p class="updated">Last updated: {stats['timestamp'][:19]} UTC | Auto-refreshes every 60 seconds</p>
        </div>
    </body>
    </html>
    """
    return html


def save_dashboard(stats: dict) -> Path:
    """Save dashboard HTML file."""
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    html = generate_dashboard_html(stats)
    filepath = STATS_DIR / "dashboard.html"
    with open(filepath, 'w') as f:
        f.write(html)
    return filepath


def update_all_stats(db_path: Path = DEFAULT_DB_PATH, send_discord: bool = True) -> dict:
    """Compile stats, save to JSON, update dashboard, and optionally send to Discord."""
    stats = compile_all_stats(db_path)
    save_stats_json(stats)
    save_dashboard(stats)

    if send_discord:
        send_stats_discord(stats)

    return stats
