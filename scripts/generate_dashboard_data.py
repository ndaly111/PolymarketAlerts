#!/usr/bin/env python3
"""
Generate JSON data for the trading dashboard from props and weather databases.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROPS_DB = ROOT / "props_trades.db"
WEATHER_DB = ROOT / "weather_trades.db"
SPORTS_DB = ROOT / "sports_trades.db"
WEATHER_ACCURACY_DB = ROOT / "weather" / "data" / "weather_forecast_accuracy.db"
OUTPUT_DIR = ROOT / "docs" / "data"


def get_props_stats(db_path: Path) -> dict:
    """Get props trading statistics."""
    if not db_path.exists():
        return {
            "total_trades": 0, "trades": [], "by_sport": {}, "by_stat_type": {},
            "wins": 0, "losses": 0, "pending": 0, "roi": 0, "by_ev_bucket": {}
        }

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Get all trades (excluding dry_run)
    cur.execute("""
        SELECT
            trade_date, ticker, player_name, stat_type, line, side,
            quantity, limit_price_cents, fair_prob, edge, books_count,
            order_id, status, fill_price_cents,
            COALESCE(settled, 0) as settled,
            won, payout_cents, settled_at
        FROM trades
        WHERE status IN ('filled', 'FILLED')
        ORDER BY trade_date DESC, created_at DESC
    """)
    trades = [dict(row) for row in cur.fetchall()]

    # Get daily stats
    cur.execute("""
        SELECT date, trades_attempted, trades_filled
        FROM daily_stats
        ORDER BY date DESC
        LIMIT 30
    """)
    daily_stats = [dict(row) for row in cur.fetchall()]

    # Aggregate by stat type (excluding dry_run)
    cur.execute("""
        SELECT stat_type, COUNT(*) as count, AVG(edge) as avg_edge
        FROM trades
        WHERE status IN ('filled', 'FILLED')
        GROUP BY stat_type
    """)
    by_stat_type = {row["stat_type"]: {"count": row["count"], "avg_edge": row["avg_edge"]}
                    for row in cur.fetchall()}

    # Aggregate by EV bucket (similar to weather)
    cur.execute("""
        SELECT
            CASE
                WHEN edge < 0.05 THEN '0-5%'
                WHEN edge < 0.10 THEN '5-10%'
                WHEN edge < 0.15 THEN '10-15%'
                WHEN edge < 0.20 THEN '15-20%'
                ELSE '20%+'
            END as ev_bucket,
            COUNT(*) as count,
            SUM(CASE WHEN won=1 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN COALESCE(settled, 0)=1 AND won=0 THEN 1 ELSE 0 END) as losses,
            SUM(CASE WHEN COALESCE(settled, 0)=0 THEN 1 ELSE 0 END) as pending,
            SUM(COALESCE(fill_price_cents, limit_price_cents)) as total_cost,
            SUM(COALESCE(payout_cents, 0)) as total_payout,
            SUM(CASE WHEN COALESCE(settled, 0)=1 THEN COALESCE(fill_price_cents, limit_price_cents) ELSE 0 END) as settled_cost
        FROM trades
        WHERE status IN ('filled', 'FILLED')
        GROUP BY ev_bucket
    """)
    by_ev_bucket = {}
    for row in cur.fetchall():
        bucket = row["ev_bucket"]
        total_cost = row["total_cost"] or 0
        total_payout = row["total_payout"] or 0
        settled_cost = row["settled_cost"] or 0
        # Only calculate ROI on settled trades to avoid showing -100% for pending
        roi = ((total_payout - settled_cost) / settled_cost * 100) if settled_cost > 0 else 0
        by_ev_bucket[bucket] = {
            "count": row["count"],
            "wins": row["wins"],
            "losses": row["losses"],
            "pending": row["pending"],
            "total_cost_cents": total_cost,
            "total_payout_cents": total_payout,
            "roi": roi,
        }

    conn.close()

    # Calculate totals (trades already filtered to filled only)
    total_trades = len(trades)
    settled_trades = [t for t in trades if t.get("settled")]
    wins = sum(1 for t in settled_trades if t.get("won"))
    losses = len(settled_trades) - wins
    pending = total_trades - len(settled_trades)
    total_cost = sum(t["fill_price_cents"] or t["limit_price_cents"] for t in trades)
    settled_cost = sum(t["fill_price_cents"] or t["limit_price_cents"] for t in settled_trades)
    total_payout = sum(t.get("payout_cents") or 0 for t in settled_trades)
    # Only calculate ROI on settled trades
    roi = ((total_payout - settled_cost) / settled_cost * 100) if settled_cost > 0 else 0
    avg_edge = sum(t["edge"] for t in trades) / len(trades) if trades else 0

    return {
        "total_trades": total_trades,
        "filled_trades": total_trades,  # All trades in list are filled (dry_run excluded)
        "total_cost_cents": total_cost,
        "settled_cost_cents": settled_cost,  # Cost of settled trades only (for P/L calc)
        "pending_cost_cents": total_cost - settled_cost,  # Cost of pending trades
        "avg_edge": avg_edge,
        "wins": wins,
        "losses": losses,
        "pending": pending,
        "total_payout_cents": total_payout,
        "roi": roi,
        "trades": trades[:50],  # Last 50 trades
        "daily_stats": daily_stats,
        "by_stat_type": by_stat_type,
        "by_ev_bucket": by_ev_bucket,
    }


def get_weather_stats(db_path: Path) -> dict:
    """Get weather trading statistics."""
    if not db_path.exists():
        return {"total_trades": 0, "trades": [], "by_city": {}, "by_ev_bucket": {}}

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Get actual trades (placed/filled only)
    cur.execute("""
        SELECT
            trade_date, city_key, market_ticker, event_display, side,
            quantity, limit_price_cents, fair_q, ev, forecast_high_f,
            order_id, status, fill_price_cents, settled, won, payout_cents
        FROM weather_trades
        WHERE status IN ('filled', 'FILLED', 'PLACED')
        ORDER BY trade_date DESC, created_at DESC
    """)
    trades = [dict(row) for row in cur.fetchall()]

    # Get daily stats
    cur.execute("""
        SELECT date, trades_attempted, trades_filled
        FROM weather_daily_stats
        ORDER BY date DESC
        LIMIT 30
    """)
    daily_stats = [dict(row) for row in cur.fetchall()]

    # Aggregate by city
    cur.execute("""
        SELECT city_key, COUNT(*) as count, AVG(ev) as avg_ev,
               SUM(CASE WHEN won=1 THEN 1 ELSE 0 END) as wins,
               SUM(CASE WHEN settled=1 AND won=0 THEN 1 ELSE 0 END) as losses
        FROM weather_trades
        WHERE status IN ('filled', 'FILLED', 'PLACED')
        GROUP BY city_key
    """)
    by_city = {row["city_key"]: {
        "count": row["count"],
        "avg_ev": row["avg_ev"],
        "wins": row["wins"],
        "losses": row["losses"]
    } for row in cur.fetchall()}

    # Aggregate by EV bucket
    cur.execute("""
        SELECT
            CASE
                WHEN ev < 0.05 THEN '0-5%'
                WHEN ev < 0.10 THEN '5-10%'
                WHEN ev < 0.15 THEN '10-15%'
                WHEN ev < 0.20 THEN '15-20%'
                ELSE '20%+'
            END as ev_bucket,
            COUNT(*) as count,
            SUM(CASE WHEN won=1 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN settled=1 AND won=0 THEN 1 ELSE 0 END) as losses,
            SUM(CASE WHEN settled=0 THEN 1 ELSE 0 END) as pending,
            SUM(COALESCE(fill_price_cents, limit_price_cents)) as total_cost,
            SUM(COALESCE(payout_cents, 0)) as total_payout,
            SUM(CASE WHEN settled=1 THEN COALESCE(fill_price_cents, limit_price_cents) ELSE 0 END) as settled_cost
        FROM weather_trades
        WHERE status IN ('filled', 'FILLED', 'PLACED')
        GROUP BY ev_bucket
    """)
    by_ev_bucket = {}
    for row in cur.fetchall():
        bucket = row["ev_bucket"]
        total_cost = row["total_cost"] or 0
        total_payout = row["total_payout"] or 0
        settled_cost = row["settled_cost"] or 0
        # Only calculate ROI on settled trades to avoid showing -100% for pending
        roi = ((total_payout - settled_cost) / settled_cost * 100) if settled_cost > 0 else 0
        by_ev_bucket[bucket] = {
            "count": row["count"],
            "wins": row["wins"],
            "losses": row["losses"],
            "pending": row["pending"],
            "total_cost_cents": total_cost,
            "total_payout_cents": total_payout,
            "roi": roi,
        }

    conn.close()

    # Calculate totals
    total_trades = len(trades)
    settled_trades = [t for t in trades if t["settled"]]
    wins = sum(1 for t in settled_trades if t["won"])
    losses = len(settled_trades) - wins
    total_cost = sum(t["fill_price_cents"] or t["limit_price_cents"] for t in trades)
    settled_cost = sum(t["fill_price_cents"] or t["limit_price_cents"] for t in settled_trades)
    total_payout = sum(t["payout_cents"] or 0 for t in settled_trades)
    # Only calculate ROI on settled trades to avoid showing -100% for pending
    roi = ((total_payout - settled_cost) / settled_cost * 100) if settled_cost > 0 else 0
    avg_ev = sum(t["ev"] for t in trades) / len(trades) if trades else 0

    return {
        "total_trades": total_trades,
        "settled_trades": len(settled_trades),
        "wins": wins,
        "losses": losses,
        "pending": total_trades - len(settled_trades),
        "total_cost_cents": total_cost,
        "settled_cost_cents": settled_cost,  # Cost of settled trades only (for P/L calc)
        "pending_cost_cents": total_cost - settled_cost,  # Cost of pending trades
        "total_payout_cents": total_payout,
        "roi": roi,
        "avg_ev": avg_ev,
        "trades": trades[:50],  # Last 50 trades
        "daily_stats": daily_stats,
        "by_city": by_city,
        "by_ev_bucket": by_ev_bucket,
    }


def get_sports_stats(db_path: Path) -> dict:
    """Get sports (moneylines/spreads/totals) trading statistics."""
    if not db_path.exists():
        return {
            "total_trades": 0, "trades": [], "by_line_type": {}, "by_ev_bucket": {},
            "wins": 0, "losses": 0, "pending": 0, "roi": 0, "avg_edge": 0,
            "total_cost_cents": 0, "total_payout_cents": 0, "filled_trades": 0, "daily_stats": []
        }

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Get all trades (excluding dry_run)
    cur.execute("""
        SELECT
            trade_date, ticker, event_title, line_type, line_label, side,
            quantity, limit_price_cents, fair_prob, edge, books_count,
            order_id, status, fill_price_cents,
            COALESCE(settled, 0) as settled,
            won, payout_cents, settled_at
        FROM trades
        WHERE status IN ('filled', 'FILLED')
        ORDER BY trade_date DESC, created_at DESC
    """)
    trades = [dict(row) for row in cur.fetchall()]

    # Get daily stats
    cur.execute("""
        SELECT date, trades_attempted, trades_filled
        FROM daily_stats
        ORDER BY date DESC
        LIMIT 30
    """)
    daily_stats = [dict(row) for row in cur.fetchall()]

    # Aggregate by line type
    cur.execute("""
        SELECT line_type, COUNT(*) as count, AVG(edge) as avg_edge
        FROM trades
        WHERE status IN ('filled', 'FILLED')
        GROUP BY line_type
    """)
    by_line_type = {row["line_type"]: {"count": row["count"], "avg_edge": row["avg_edge"]}
                    for row in cur.fetchall()}

    # Aggregate by sport (extracted from ticker)
    cur.execute("""
        SELECT
            CASE
                WHEN ticker LIKE '%NBA%' THEN 'NBA'
                WHEN ticker LIKE '%NFL%' THEN 'NFL'
                WHEN ticker LIKE '%NHL%' THEN 'NHL'
                WHEN ticker LIKE 'KXMLB%' THEN 'MLB'
                WHEN ticker LIKE '%NCAAB%' OR ticker LIKE '%CBB%' THEN 'NCAAB'
                WHEN ticker LIKE '%NCAAF%' OR ticker LIKE '%CFB%' THEN 'NCAAF'
                WHEN ticker LIKE 'KXEPL%' THEN 'EPL'
                WHEN ticker LIKE 'KXMLS%' THEN 'MLS'
                WHEN ticker LIKE 'KXLALIGA%' THEN 'La Liga'
                WHEN ticker LIKE 'KXBUNDESLIGA%' THEN 'Bundesliga'
                WHEN ticker LIKE 'KXLIGAPORTUGAL%' THEN 'Liga Portugal'
                WHEN ticker LIKE 'KXLIGAMX%' THEN 'Liga MX'
                ELSE 'Other'
            END as sport,
            COUNT(*) as count,
            AVG(edge) as avg_edge,
            SUM(CASE WHEN won=1 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN COALESCE(settled, 0)=1 AND won=0 THEN 1 ELSE 0 END) as losses,
            SUM(CASE WHEN COALESCE(settled, 0)=0 THEN 1 ELSE 0 END) as pending
        FROM trades
        WHERE status IN ('filled', 'FILLED')
        GROUP BY sport
    """)
    by_sport = {row["sport"]: {
        "count": row["count"],
        "avg_edge": row["avg_edge"],
        "wins": row["wins"],
        "losses": row["losses"],
        "pending": row["pending"]
    } for row in cur.fetchall()}

    # Aggregate by EV bucket
    cur.execute("""
        SELECT
            CASE
                WHEN edge < 0.05 THEN '0-5%'
                WHEN edge < 0.10 THEN '5-10%'
                WHEN edge < 0.15 THEN '10-15%'
                WHEN edge < 0.20 THEN '15-20%'
                ELSE '20%+'
            END as ev_bucket,
            COUNT(*) as count,
            SUM(CASE WHEN won=1 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN COALESCE(settled, 0)=1 AND won=0 THEN 1 ELSE 0 END) as losses,
            SUM(CASE WHEN COALESCE(settled, 0)=0 THEN 1 ELSE 0 END) as pending,
            SUM(COALESCE(fill_price_cents, limit_price_cents)) as total_cost,
            SUM(COALESCE(payout_cents, 0)) as total_payout,
            SUM(CASE WHEN COALESCE(settled, 0)=1 THEN COALESCE(fill_price_cents, limit_price_cents) ELSE 0 END) as settled_cost
        FROM trades
        WHERE status IN ('filled', 'FILLED')
        GROUP BY ev_bucket
    """)
    by_ev_bucket = {}
    for row in cur.fetchall():
        bucket = row["ev_bucket"]
        total_cost = row["total_cost"] or 0
        total_payout = row["total_payout"] or 0
        settled_cost = row["settled_cost"] or 0
        roi = ((total_payout - settled_cost) / settled_cost * 100) if settled_cost > 0 else 0
        by_ev_bucket[bucket] = {
            "count": row["count"],
            "wins": row["wins"],
            "losses": row["losses"],
            "pending": row["pending"],
            "total_cost_cents": total_cost,
            "total_payout_cents": total_payout,
            "roi": roi,
        }

    conn.close()

    # Calculate totals
    total_trades = len(trades)
    settled_trades = [t for t in trades if t.get("settled")]
    wins = sum(1 for t in settled_trades if t.get("won"))
    losses = len(settled_trades) - wins
    pending = total_trades - len(settled_trades)
    total_cost = sum(t["fill_price_cents"] or t["limit_price_cents"] for t in trades)
    settled_cost = sum(t["fill_price_cents"] or t["limit_price_cents"] for t in settled_trades)
    total_payout = sum(t.get("payout_cents") or 0 for t in settled_trades)
    roi = ((total_payout - settled_cost) / settled_cost * 100) if settled_cost > 0 else 0
    avg_edge = sum(t["edge"] for t in trades) / len(trades) if trades else 0

    return {
        "total_trades": total_trades,
        "filled_trades": total_trades,
        "total_cost_cents": total_cost,
        "settled_cost_cents": settled_cost,  # Cost of settled trades only (for P/L calc)
        "pending_cost_cents": total_cost - settled_cost,  # Cost of pending trades
        "avg_edge": avg_edge,
        "wins": wins,
        "losses": losses,
        "pending": pending,
        "total_payout_cents": total_payout,
        "roi": roi,
        "trades": trades[:50],
        "daily_stats": daily_stats,
        "by_line_type": by_line_type,
        "by_sport": by_sport,
        "by_ev_bucket": by_ev_bucket,
    }


def get_model_accuracy(db_path: Path) -> dict:
    """Get weather model accuracy comparison data."""
    if not db_path.exists():
        return {"models": [], "by_city": {}, "recent_forecasts": []}

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Get accuracy by model source
    cur.execute("""
        SELECT
            fs.source,
            COUNT(DISTINCT fs.target_date_local) as days,
            COUNT(*) as total_forecasts,
            ROUND(AVG(fs.forecast_high_f - oc.tmax_f), 2) as bias,
            ROUND(AVG(ABS(fs.forecast_high_f - oc.tmax_f)), 2) as mae
        FROM forecast_snapshots fs
        INNER JOIN observed_cli oc
            ON fs.city_key = oc.city_key
            AND fs.target_date_local = oc.date_local
        WHERE oc.tmax_f IS NOT NULL
        GROUP BY fs.source
        HAVING days >= 3
        ORDER BY mae ASC
    """)
    models = []
    for row in cur.fetchall():
        models.append({
            "source": row["source"],
            "days": row["days"],
            "total_forecasts": row["total_forecasts"],
            "bias": row["bias"],
            "mae": row["mae"],
        })

    # Get accuracy by model and city
    cur.execute("""
        SELECT
            fs.source,
            fs.city_key,
            COUNT(*) as count,
            ROUND(AVG(fs.forecast_high_f - oc.tmax_f), 2) as bias,
            ROUND(AVG(ABS(fs.forecast_high_f - oc.tmax_f)), 2) as mae
        FROM forecast_snapshots fs
        INNER JOIN observed_cli oc
            ON fs.city_key = oc.city_key
            AND fs.target_date_local = oc.date_local
        WHERE oc.tmax_f IS NOT NULL
        GROUP BY fs.source, fs.city_key
        ORDER BY fs.source, mae ASC
    """)
    by_city = {}
    for row in cur.fetchall():
        source = row["source"]
        if source not in by_city:
            by_city[source] = {}
        by_city[source][row["city_key"]] = {
            "count": row["count"],
            "bias": row["bias"],
            "mae": row["mae"],
        }

    # Get recent forecasts vs observed (last 7 days)
    cur.execute("""
        SELECT
            fs.target_date_local as date,
            fs.city_key,
            fs.source,
            fs.forecast_high_f as forecast,
            oc.tmax_f as observed,
            (fs.forecast_high_f - oc.tmax_f) as error
        FROM forecast_snapshots fs
        INNER JOIN observed_cli oc
            ON fs.city_key = oc.city_key
            AND fs.target_date_local = oc.date_local
        WHERE oc.tmax_f IS NOT NULL
        ORDER BY fs.target_date_local DESC, fs.city_key, fs.source
        LIMIT 200
    """)
    recent_forecasts = [dict(row) for row in cur.fetchall()]

    conn.close()

    return {
        "models": models,
        "by_city": by_city,
        "recent_forecasts": recent_forecasts,
    }


def update_dashboard(quiet: bool = False) -> None:
    """
    Regenerate all dashboard JSON files.

    Call this from trading workflows to keep dashboard data fresh.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Generate props data
    props_stats = get_props_stats(PROPS_DB)

    # Generate weather data
    weather_stats = get_weather_stats(WEATHER_DB)

    # Generate sports data
    sports_stats = get_sports_stats(SPORTS_DB)

    # Generate model accuracy data
    model_accuracy = get_model_accuracy(WEATHER_ACCURACY_DB)

    # Combined summary
    summary = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "props": {
            "total_trades": props_stats["total_trades"],
            "avg_edge": props_stats["avg_edge"],
            "record": f"{props_stats['wins']}W-{props_stats['losses']}L",
            "roi": props_stats["roi"],
        },
        "weather": {
            "total_trades": weather_stats["total_trades"],
            "record": f"{weather_stats['wins']}W-{weather_stats['losses']}L",
            "roi": weather_stats["roi"],
            "avg_ev": weather_stats["avg_ev"],
        },
        "sports": {
            "total_trades": sports_stats["total_trades"],
            "avg_edge": sports_stats["avg_edge"],
            "record": f"{sports_stats['wins']}W-{sports_stats['losses']}L",
            "roi": sports_stats["roi"],
        },
    }

    # Write files
    (OUTPUT_DIR / "props.json").write_text(json.dumps(props_stats, indent=2))
    (OUTPUT_DIR / "weather.json").write_text(json.dumps(weather_stats, indent=2))
    (OUTPUT_DIR / "sports.json").write_text(json.dumps(sports_stats, indent=2))
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    (OUTPUT_DIR / "model_accuracy.json").write_text(json.dumps(model_accuracy, indent=2))

    if not quiet:
        print(f"Dashboard data updated at {OUTPUT_DIR}")
        print(f"  Props: {props_stats['total_trades']} trades, {props_stats['wins']}W-{props_stats['losses']}L")
        print(f"  Weather: {weather_stats['total_trades']} trades, {weather_stats['wins']}W-{weather_stats['losses']}L")
        print(f"  Sports: {sports_stats['total_trades']} trades, {sports_stats['wins']}W-{sports_stats['losses']}L")
        if model_accuracy["models"]:
            best = model_accuracy["models"][0]
            print(f"  Model Accuracy: {len(model_accuracy['models'])} models tracked, best={best['source']} (MAE: {best['mae']}°F)")


def main():
    update_dashboard(quiet=False)


if __name__ == "__main__":
    main()
