#!/usr/bin/env python3
"""Local Flask dashboard for BTC 15-min scanner."""

import sqlite3
import json
import os
import sys
import queue
import threading
from datetime import datetime, timezone
from pathlib import Path
from flask import Flask, jsonify, render_template_string, Response

# Global log queue for real-time streaming
LOG_QUEUE = queue.Queue(maxsize=100)
LOG_HISTORY = []  # Keep last 100 logs

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from lib.db import DEFAULT_DB_PATH, get_paper_trade_summary, get_pending_trades, get_prediction_stats, get_recent_logs
from lib.binance import fetch_current_price

app = Flask(__name__)

DB_PATH = DEFAULT_DB_PATH


def push_log(message: str):
    """Push a log message to the dashboard."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {message}"

    # Add to history
    LOG_HISTORY.append(log_entry)
    if len(LOG_HISTORY) > 100:
        LOG_HISTORY.pop(0)

    # Push to queue (non-blocking)
    try:
        LOG_QUEUE.put_nowait(log_entry)
    except queue.Full:
        pass

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>BTC 15-Min Scanner</title>
    <meta charset="utf-8">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, monospace;
            background: #0d1117;
            color: #c9d1d9;
            padding: 20px;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { color: #58a6ff; margin-bottom: 20px; }
        h2 { color: #8b949e; margin: 20px 0 10px; font-size: 14px; text-transform: uppercase; }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }
        .stat-card {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 20px;
        }
        .stat-label { color: #8b949e; font-size: 12px; margin-bottom: 5px; }
        .stat-value { font-size: 28px; font-weight: bold; }
        .stat-value.positive { color: #3fb950; }
        .stat-value.negative { color: #f85149; }
        .stat-value.neutral { color: #58a6ff; }

        .trades-table {
            width: 100%;
            border-collapse: collapse;
            background: #161b22;
            border-radius: 8px;
            overflow: hidden;
        }
        .trades-table th, .trades-table td {
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #30363d;
        }
        .trades-table th {
            background: #21262d;
            color: #8b949e;
            font-size: 12px;
            text-transform: uppercase;
        }
        .trades-table tr:hover { background: #21262d; }

        .badge {
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
        }
        .badge.win { background: #238636; color: #fff; }
        .badge.loss { background: #da3633; color: #fff; }
        .badge.pending { background: #9e6a03; color: #fff; }
        .badge.up { background: #238636; color: #fff; }
        .badge.down { background: #da3633; color: #fff; }

        .price-display {
            font-size: 42px;
            font-weight: bold;
            color: #f0f6fc;
        }

        .refresh-info {
            color: #8b949e;
            font-size: 12px;
            margin-top: 20px;
        }

        .confidence-bar {
            width: 100%;
            height: 8px;
            background: #30363d;
            border-radius: 4px;
            overflow: hidden;
        }
        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, #f85149 0%, #d29922 50%, #3fb950 100%);
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>BTC 15-Min Scanner Dashboard</h1>

        <div class="stats-grid" id="stats">
            <div class="stat-card">
                <div class="stat-label">BTC Price</div>
                <div class="stat-value neutral" id="btc-price">Loading...</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Cumulative P&L</div>
                <div class="stat-value" id="pnl">--</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Win Rate</div>
                <div class="stat-value neutral" id="win-rate">--</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Total Trades</div>
                <div class="stat-value neutral" id="total-trades">--</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Wins / Losses</div>
                <div class="stat-value neutral" id="record">--</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Pending</div>
                <div class="stat-value neutral" id="pending">--</div>
            </div>
        </div>

        <h2>Recent Trades</h2>
        <table class="trades-table" id="trades-table">
            <thead>
                <tr>
                    <th>Time</th>
                    <th>Market</th>
                    <th>Side</th>
                    <th>Strike</th>
                    <th>Entry</th>
                    <th>ML Prob</th>
                    <th>Edge</th>
                    <th>Outcome</th>
                    <th>P&L</th>
                </tr>
            </thead>
            <tbody id="trades-body">
                <tr><td colspan="9">Loading...</td></tr>
            </tbody>
        </table>

        <h2>ML Learning Progress</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Total Predictions</div>
                <div class="stat-value neutral" id="total-predictions">--</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Prediction Accuracy</div>
                <div class="stat-value neutral" id="prediction-accuracy">--</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Pending Settlement</div>
                <div class="stat-value neutral" id="pending-predictions">--</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Traded vs Non-Traded</div>
                <div class="stat-value neutral" id="traded-ratio">--</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Live Samples (Training)</div>
                <div class="stat-value neutral" id="live-samples">--</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Live Accuracy</div>
                <div class="stat-value neutral" id="live-accuracy">--</div>
            </div>
        </div>

        <h2>Live Scanner Logs</h2>
        <div id="log-container" style="background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; height: 300px; overflow-y: auto; font-family: monospace; font-size: 12px;">
            <div id="logs">Connecting to log stream...</div>
        </div>

        <p class="refresh-info">Stats refresh every 10 seconds | Logs stream in real-time</p>
    </div>

    <script>
        async function fetchData() {
            try {
                const resp = await fetch('/api/stats');
                const data = await resp.json();

                // Update stats
                document.getElementById('btc-price').textContent = '$' + data.btc_price.toLocaleString(undefined, {minimumFractionDigits: 2});

                const pnlEl = document.getElementById('pnl');
                pnlEl.textContent = '$' + data.cumulative_pnl.toFixed(2);
                pnlEl.className = 'stat-value ' + (data.cumulative_pnl >= 0 ? 'positive' : 'negative');

                document.getElementById('win-rate').textContent = (data.win_rate * 100).toFixed(1) + '%';
                document.getElementById('total-trades').textContent = data.total_trades;
                document.getElementById('record').textContent = data.wins + 'W / ' + data.losses + 'L';
                document.getElementById('pending').textContent = data.pending_count;

                // Prediction/learning stats
                document.getElementById('total-predictions').textContent = data.prediction_stats.total_predictions;
                document.getElementById('prediction-accuracy').textContent = data.prediction_stats.accuracy ? (data.prediction_stats.accuracy * 100).toFixed(1) + '%' : '--';
                document.getElementById('pending-predictions').textContent = data.prediction_stats.pending;
                document.getElementById('traded-ratio').textContent = data.prediction_stats.traded + ' / ' + (data.prediction_stats.total_predictions - data.prediction_stats.traded);
                document.getElementById('live-samples').textContent = data.live_samples;
                document.getElementById('live-accuracy').textContent = (data.live_accuracy * 100).toFixed(1) + '%';

                // Update trades table
                const tbody = document.getElementById('trades-body');
                if (data.recent_trades.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="9">No trades yet</td></tr>';
                } else {
                    tbody.innerHTML = data.recent_trades.map(t => `
                        <tr>
                            <td>${new Date(t.timestamp).toLocaleTimeString()}</td>
                            <td>${t.market_ticker}</td>
                            <td><span class="badge ${t.side.toLowerCase()}">${t.side}</span></td>
                            <td>$${t.strike_price?.toLocaleString() || '--'}</td>
                            <td>${(t.entry_price * 100).toFixed(1)}¢</td>
                            <td>${(t.model_prob * 100).toFixed(1)}%</td>
                            <td>${(t.edge * 100).toFixed(1)}%</td>
                            <td><span class="badge ${(t.outcome || 'pending').toLowerCase()}">${t.outcome || 'PENDING'}</span></td>
                            <td class="${t.pnl >= 0 ? 'positive' : 'negative'}">${t.pnl ? '$' + t.pnl.toFixed(2) : '--'}</td>
                        </tr>
                    `).join('');
                }
            } catch (e) {
                console.error('Fetch error:', e);
            }
        }

        fetchData();
        setInterval(fetchData, 10000);

        // Real-time log streaming
        const logsDiv = document.getElementById('logs');
        const logContainer = document.getElementById('log-container');
        let logLines = [];

        const evtSource = new EventSource('/stream/logs');
        evtSource.onmessage = function(event) {
            if (event.data === '[heartbeat]') return;

            logLines.push(event.data);
            if (logLines.length > 100) logLines.shift();

            // Color code logs
            const colored = logLines.map(line => {
                if (line.includes('ERROR') || line.includes('LOSS')) {
                    return `<span style="color: #f85149">${line}</span>`;
                } else if (line.includes('WIN') || line.includes('TRADE') || line.includes('filled')) {
                    return `<span style="color: #3fb950">${line}</span>`;
                } else if (line.includes('SCAN')) {
                    return `<span style="color: #8b949e">${line}</span>`;
                } else if (line.includes('LEARN') || line.includes('ML')) {
                    return `<span style="color: #a371f7">${line}</span>`;
                }
                return line;
            });

            logsDiv.innerHTML = colored.join('<br>');
            logContainer.scrollTop = logContainer.scrollHeight;
        };

        evtSource.onerror = function() {
            logsDiv.innerHTML += '<br><span style="color: #f85149">Log stream disconnected. Refresh page.</span>';
        };
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/stats')
def api_stats():
    """Return all stats as JSON."""
    try:
        btc_price = fetch_current_price()
    except:
        btc_price = 0

    # Get trade summary
    summary = get_paper_trade_summary(DB_PATH)

    # Get pending trades
    pending = get_pending_trades(DB_PATH)

    # Get recent trades
    recent_trades = []
    try:
        with sqlite3.connect(str(DB_PATH)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT * FROM paper_trades
                ORDER BY timestamp DESC
                LIMIT 20
            """).fetchall()
            recent_trades = [dict(row) for row in rows]
    except Exception as e:
        print(f"Error fetching trades: {e}")

    # Get prediction stats (all predictions, not just trades)
    try:
        prediction_stats = get_prediction_stats(DB_PATH)
    except:
        prediction_stats = {"total_predictions": 0, "pending": 0, "accuracy": None, "traded": 0}

    # Get live sample stats
    live_samples = 0
    live_accuracy = 0
    try:
        with sqlite3.connect(str(DB_PATH)) as conn:
            result = conn.execute("SELECT COUNT(*) FROM live_samples").fetchone()
            live_samples = result[0] if result else 0

            result = conn.execute("""
                SELECT AVG(was_correct) FROM live_samples
                WHERE was_correct IS NOT NULL
            """).fetchone()
            live_accuracy = result[0] if result and result[0] else 0
    except:
        pass

    return jsonify({
        "btc_price": btc_price,
        "cumulative_pnl": summary.get("total_pnl", 0),
        "total_trades": summary.get("total_trades", 0),
        "wins": summary.get("wins", 0),
        "losses": summary.get("losses", 0),
        "win_rate": summary.get("win_rate", 0) or 0,
        "pending_count": len(pending),
        "pending_trades": pending,
        "recent_trades": recent_trades,
        "prediction_stats": prediction_stats,
        "live_samples": live_samples,
        "live_accuracy": live_accuracy,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


@app.route('/api/trades')
def api_trades():
    """Return all trades as JSON."""
    try:
        with sqlite3.connect(str(DB_PATH)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT * FROM paper_trades
                ORDER BY timestamp DESC
            """).fetchall()
            return jsonify([dict(row) for row in rows])
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/live_samples')
def api_live_samples():
    """Return live learning samples."""
    try:
        with sqlite3.connect(str(DB_PATH)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT timestamp, outcome, prediction, model_prob, was_correct, pnl
                FROM live_samples
                ORDER BY timestamp DESC
                LIMIT 50
            """).fetchall()
            return jsonify([dict(row) for row in rows])
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/logs')
def api_logs():
    """Return recent logs from database."""
    logs = get_recent_logs(50, DB_PATH)
    return jsonify(logs)


@app.route('/stream/logs')
def stream_logs():
    """Server-sent events stream for real-time logs."""
    import time

    def generate():
        last_count = 0

        while True:
            logs = get_recent_logs(50, DB_PATH)
            current_count = len(logs)

            # Send new logs
            if current_count > last_count or last_count == 0:
                for log in logs[-(current_count - last_count) if last_count > 0 else 50:]:
                    ts = log['timestamp'][11:19]  # Extract time
                    level = log['level']
                    msg = log['message']
                    yield f"data: [{ts}] {msg}\n\n"
                last_count = current_count

            time.sleep(2)  # Poll every 2 seconds

    return Response(generate(), mimetype='text/event-stream')


if __name__ == "__main__":
    print("Starting BTC Scanner Dashboard...")
    print("Open http://localhost:5050 in your browser")
    app.run(host="0.0.0.0", port=5050, debug=False, threaded=True)
