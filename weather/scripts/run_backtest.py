#!/usr/bin/env python3
"""Run backtests for weather trading strategies.

This CLI provides access to the backtesting framework for evaluating:
1. Forecast calibration (Brier score, calibration error)
2. Trading strategy performance (PnL, win rate, Sharpe ratio)
3. Expected vs realized EV correlation

Example usage:
    # Run backtest for NYC over 2024
    python run_backtest.py --city NHIGH --start 2024-01-01 --end 2024-12-31

    # Run with custom EV threshold
    python run_backtest.py --city NHIGH --start 2024-06-01 --end 2024-12-31 --min-ev 0.05

    # Run for multiple cities
    python run_backtest.py --start 2024-01-01 --end 2024-12-31

    # Use MOS archive data (more historical coverage)
    python run_backtest.py --city NHIGH --start 2023-01-01 --end 2025-12-31 --forecast-source mos_gfs_18z_archive
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from weather.backtesting.engine import BacktestConfig, BacktestEngine, BacktestResults
from weather.backtesting.data_loader import AsOfDataLoader
from weather.backtesting.metrics import (
    brier_score,
    calibration_error,
    calibration_curve,
    compute_probability_bins_accuracy,
    sharpe_ratio,
)
from weather.lib import db as db_lib
from weather.lib.fair import normalize_pmf, shift_pmf, summarize_pmf
from weather.lib.kalshi_weather import parse_event_spec_from_title, prob_event
from weather.lib.fees import FeeSchedule, ev_yes, ev_no

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "weather" / "config" / "cities.yml"
DEFAULT_DB = Path(os.getenv("WEATHER_DB_PATH", str(ROOT / "weather" / "data" / "weather_forecast_accuracy.db")))
OUT_BASE = ROOT / "weather" / "outputs" / "backtests"


def load_city_keys(config_path: Path) -> List[str]:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return [str(c["key"]).strip() for c in (data.get("cities") or [])]


def load_city_tz(config_path: Path, city_key: str) -> str:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    for c in data.get("cities") or []:
        if str(c["key"]).strip() == city_key:
            return str(c.get("tz", "America/New_York"))
    return "America/New_York"


def run_simple_backtest(
    db_path: Path,
    city_key: str,
    timezone: str,
    start_date: str,
    end_date: str,
    forecast_source: str,
    error_model_source: str,
    error_model_hour: int,
    min_ev: float,
    min_q: float,
    fee_cents: int,
    trading_hour: int,
    calibration_only: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run a simple PMF-only backtest without the full engine.

    This backtest uses only the error PMF model to generate probabilities
    and evaluates calibration and trading performance.

    Args:
        calibration_only: If True, run calibration analysis without requiring Kalshi market data.
                         This tests whether our probability distributions are well-calibrated.
    """
    loader = AsOfDataLoader(db_path)
    fees = FeeSchedule(open_fee_cents=fee_cents)
    fee_open = fees.open_fee_dollars()

    # Get dates with both forecast and observation data
    dates = loader.get_date_range_with_data(city_key, start_date, end_date)

    if verbose:
        print(f"Found {len(dates)} dates with forecast/observation data")

    # Collect results
    predictions = []  # (predicted_prob, actual_outcome, ev, pnl, date, market)
    forecast_errors = []
    daily_pnls = []
    calibration_data = []  # For calibration-only mode: (predicted_exceedance, actual_outcome)
    skipped = {"no_forecast": 0, "no_error_model": 0, "no_markets": 0}

    for date in dates:
        month = int(date.split("-")[1])

        # Get latest forecast for this date
        forecast = db_lib.fetch_latest_forecast_snapshot(
            db_path,
            city_key=city_key,
            target_date_local=date,
            source=forecast_source,
        )

        if not forecast:
            skipped["no_forecast"] += 1
            continue

        forecast_high = int(forecast["forecast_high_f"])

        # Get error model
        error_model = db_lib.fetch_error_model(
            db_path,
            city_key=city_key,
            month=month,
            snapshot_hour_local=error_model_hour,
            source=error_model_source,
        )

        if not error_model:
            skipped["no_error_model"] += 1
            continue

        # Build PMF
        pmf_error = error_model["pmf"]
        pmf_high = normalize_pmf(shift_pmf(pmf_error, forecast_high))

        # Get observed high
        observed_high = loader.get_observed_high(city_key, date)
        if observed_high is not None:
            forecast_errors.append(observed_high - forecast_high)

            # For calibration analysis: test several temperature thresholds
            if calibration_only:
                # Test exceedance probabilities at various thresholds
                for threshold in range(forecast_high - 10, forecast_high + 11, 2):
                    prob_exceed = sum(p for t, p in pmf_high.items() if t >= threshold)
                    actual_exceed = 1 if observed_high >= threshold else 0
                    calibration_data.append({
                        "date": date,
                        "threshold": threshold,
                        "predicted_prob": prob_exceed,
                        "actual_outcome": actual_exceed,
                    })

        if calibration_only:
            continue  # Skip Kalshi market analysis

        # Get Kalshi markets
        snap_time = db_lib.fetch_latest_kalshi_weather_snapshot_time(
            db_path,
            city_key=city_key,
            target_date_local=date,
        )

        if not snap_time:
            skipped["no_markets"] += 1
            continue

        markets = list(db_lib.fetch_kalshi_weather_markets_at_snapshot(
            db_path,
            snapshot_time_utc=snap_time,
            city_key=city_key,
            target_date_local=date,
        ))

        if not markets:
            skipped["no_markets"] += 1
            continue

        # Evaluate each market
        day_pnl = 0.0

        for market in markets:
            raw = market.get("raw") if isinstance(market.get("raw"), dict) else {}
            title = None
            for k in ("title", "market_title", "marketTitle", "name"):
                v = raw.get(k)
                if isinstance(v, str) and v.strip():
                    title = v.strip()
                    break

            if not title:
                continue

            spec = parse_event_spec_from_title(title)
            if not spec:
                continue

            # Model probability
            q = float(prob_event(pmf_high, spec))

            # Skip extreme probabilities
            if q < min_q or q > (1.0 - min_q):
                continue

            # Get prices
            yes_ask = market.get("yes_ask")
            no_ask = market.get("no_ask")

            if yes_ask is None or no_ask is None:
                continue

            p_yes = yes_ask / 100.0
            p_no = no_ask / 100.0

            # Calculate EV for both sides
            ev_y = ev_yes(q, p_yes, fee_open)
            ev_n = ev_no(q, p_no, fee_open)

            # Determine actual outcome (if observed_high is available)
            if observed_high is not None:
                event_happened = prob_event({observed_high: 1.0}, spec) > 0.5
                actual_outcome = 1 if event_happened else 0
            else:
                actual_outcome = None

            # Best side
            if ev_y > ev_n and ev_y >= min_ev:
                best_side = "YES"
                best_ev = ev_y
                best_price = p_yes
                predicted_prob = q
            elif ev_n >= min_ev:
                best_side = "NO"
                best_ev = ev_n
                best_price = p_no
                predicted_prob = 1 - q
            else:
                continue

            # Calculate PnL if outcome known
            if actual_outcome is not None:
                bet_won = (best_side == "YES" and actual_outcome == 1) or \
                          (best_side == "NO" and actual_outcome == 0)
                if bet_won:
                    pnl = (1.0 - best_price) - fee_open
                else:
                    pnl = -best_price - fee_open

                predictions.append({
                    "date": date,
                    "market": market.get("market_ticker", ""),
                    "side": best_side,
                    "predicted_prob": predicted_prob,
                    "actual_outcome": 1 if bet_won else 0,
                    "ev": best_ev,
                    "pnl": pnl,
                    "price": best_price,
                })
                day_pnl += pnl

        daily_pnls.append(day_pnl)

    # Compute metrics
    if calibration_only:
        # Calibration-only analysis using temperature threshold exceedance
        if not calibration_data:
            return {
                "city_key": city_key,
                "start_date": start_date,
                "end_date": end_date,
                "forecast_source": forecast_source,
                "error_model_source": error_model_source,
                "mode": "calibration_only",
                "n_dates": len(dates),
                "n_predictions": 0,
                "skipped": skipped,
                "error": "no_calibration_data",
            }

        cal_probs = [d["predicted_prob"] for d in calibration_data]
        cal_outcomes = [d["actual_outcome"] for d in calibration_data]

        bs = brier_score(cal_probs, cal_outcomes)
        ce = calibration_error(cal_probs, cal_outcomes)
        cal_centers, cal_actuals, cal_counts = calibration_curve(cal_probs, cal_outcomes)

        # Forecast accuracy
        if forecast_errors:
            mae = sum(abs(e) for e in forecast_errors) / len(forecast_errors)
            bias = sum(forecast_errors) / len(forecast_errors)
        else:
            mae = float("nan")
            bias = float("nan")

        return {
            "city_key": city_key,
            "start_date": start_date,
            "end_date": end_date,
            "forecast_source": forecast_source,
            "error_model_source": error_model_source,
            "mode": "calibration_only",
            "min_ev": min_ev,
            "min_q": min_q,
            "fee_cents": fee_cents,
            "n_dates": len(dates),
            "n_predictions": len(calibration_data),
            "skipped": skipped,
            "forecast_metrics": {
                "n_errors": len(forecast_errors),
                "mae": round(mae, 2) if mae == mae else None,
                "bias": round(bias, 2) if bias == bias else None,
            },
            "calibration_metrics": {
                "brier_score": round(bs, 4) if bs == bs else None,
                "calibration_error": round(ce, 4) if ce == ce else None,
                "calibration_curve": {
                    "centers": [round(c, 3) for c in cal_centers],
                    "actuals": [round(a, 3) for a in cal_actuals],
                    "counts": cal_counts,
                },
            },
        }

    if not predictions:
        return {
            "city_key": city_key,
            "start_date": start_date,
            "end_date": end_date,
            "forecast_source": forecast_source,
            "error_model_source": error_model_source,
            "n_dates": len(dates),
            "n_predictions": 0,
            "skipped": skipped,
            "error": "no_predictions",
        }

    pred_probs = [p["predicted_prob"] for p in predictions]
    outcomes = [p["actual_outcome"] for p in predictions]
    evs = [p["ev"] for p in predictions]
    pnls = [p["pnl"] for p in predictions]

    # Calibration metrics
    bs = brier_score(pred_probs, outcomes)
    ce = calibration_error(pred_probs, outcomes)
    cal_centers, cal_actuals, cal_counts = calibration_curve(pred_probs, outcomes)

    # Trading metrics
    total_pnl = sum(pnls)
    total_ev = sum(evs)
    win_rate = sum(1 for p in pnls if p > 0) / len(pnls)
    avg_pnl = total_pnl / len(pnls)
    avg_ev = total_ev / len(pnls)

    # EV vs PnL correlation
    ev_pnl_ratio = total_pnl / total_ev if abs(total_ev) > 0.001 else float("nan")

    # Sharpe ratio (using daily PnL)
    sr = sharpe_ratio([p for p in daily_pnls if p != 0]) if any(p != 0 for p in daily_pnls) else float("nan")

    # Forecast accuracy
    if forecast_errors:
        mae = sum(abs(e) for e in forecast_errors) / len(forecast_errors)
        bias = sum(forecast_errors) / len(forecast_errors)
    else:
        mae = float("nan")
        bias = float("nan")

    return {
        "city_key": city_key,
        "start_date": start_date,
        "end_date": end_date,
        "forecast_source": forecast_source,
        "error_model_source": error_model_source,
        "min_ev": min_ev,
        "min_q": min_q,
        "fee_cents": fee_cents,
        "n_dates": len(dates),
        "n_predictions": len(predictions),
        "skipped": skipped,
        "forecast_metrics": {
            "n_errors": len(forecast_errors),
            "mae": round(mae, 2) if mae == mae else None,
            "bias": round(bias, 2) if bias == bias else None,
        },
        "calibration_metrics": {
            "brier_score": round(bs, 4) if bs == bs else None,
            "calibration_error": round(ce, 4) if ce == ce else None,
            "calibration_curve": {
                "centers": [round(c, 3) for c in cal_centers],
                "actuals": [round(a, 3) for a in cal_actuals],
                "counts": cal_counts,
            },
        },
        "trading_metrics": {
            "total_pnl": round(total_pnl, 4),
            "total_ev": round(total_ev, 4),
            "avg_pnl": round(avg_pnl, 4),
            "avg_ev": round(avg_ev, 4),
            "win_rate": round(win_rate, 4),
            "ev_pnl_ratio": round(ev_pnl_ratio, 4) if ev_pnl_ratio == ev_pnl_ratio else None,
            "sharpe_ratio": round(sr, 4) if sr == sr else None,
            "n_trades": len(predictions),
        },
    }


def format_results(results: Dict[str, Any], format_type: str = "text") -> str:
    """Format backtest results for display."""

    if format_type == "json":
        return json.dumps(results, indent=2)

    # Text format
    mode = results.get("mode", "full")
    lines = [
        f"=== Backtest Results: {results['city_key']} ({mode}) ===",
        f"Period: {results['start_date']} to {results['end_date']}",
        f"Forecast source: {results.get('forecast_source', 'unknown')}",
        f"Error model source: {results.get('error_model_source', 'unknown')}",
        f"N dates: {results.get('n_dates', 0)}, N predictions: {results.get('n_predictions', 0)}",
        "",
    ]

    if "error" in results:
        lines.append(f"Error: {results['error']}")
        return "\n".join(lines)

    # Forecast metrics
    fm = results.get("forecast_metrics", {})
    lines.extend([
        "--- Forecast Accuracy ---",
        f"  MAE: {fm.get('mae', 'N/A')}°F",
        f"  Bias: {fm.get('bias', 'N/A')}°F",
        f"  N: {fm.get('n_errors', 0)}",
        "",
    ])

    # Calibration metrics
    cm = results.get("calibration_metrics", {})
    lines.extend([
        "--- Calibration ---",
        f"  Brier Score: {cm.get('brier_score', 'N/A')}",
        f"  Calibration Error: {cm.get('calibration_error', 'N/A')}",
        "",
    ])

    # Calibration curve
    cc = cm.get("calibration_curve", {})
    if cc.get("centers"):
        lines.append("  Calibration Curve:")
        lines.append("  Predicted | Actual | Count")
        lines.append("  ----------|--------|------")
        for c, a, n in zip(cc["centers"], cc["actuals"], cc["counts"]):
            diff = a - c
            marker = "*" if abs(diff) < 0.05 else ("+" if diff > 0 else "-")
            lines.append(f"    {c:>5.1%}  | {a:>5.1%} | {n:>5} {marker}")
        lines.append("")

    # Trading metrics
    tm = results.get("trading_metrics", {})
    lines.extend([
        "--- Trading Performance ---",
        f"  Total PnL: ${tm.get('total_pnl', 0):.2f}",
        f"  Total EV: ${tm.get('total_ev', 0):.2f}",
        f"  EV/PnL Ratio: {tm.get('ev_pnl_ratio', 'N/A')}",
        f"  Win Rate: {tm.get('win_rate', 0):.1%}",
        f"  Sharpe Ratio: {tm.get('sharpe_ratio', 'N/A')}",
        f"  N Trades: {tm.get('n_trades', 0)}",
        "",
    ])

    # Skipped
    skipped = results.get("skipped", {})
    if any(v > 0 for v in skipped.values()):
        lines.extend([
            "--- Skipped ---",
            f"  No forecast: {skipped.get('no_forecast', 0)}",
            f"  No error model: {skipped.get('no_error_model', 0)}",
            f"  No markets: {skipped.get('no_markets', 0)}",
        ])

    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description="Run weather trading backtests")
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--db", default=str(DEFAULT_DB))
    p.add_argument("--city", default="", help="City key (default: all cities)")
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    p.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    p.add_argument("--forecast-source", default=os.getenv("WEATHER_FORECAST_SOURCE", "mos_gfs_18z_archive"))
    p.add_argument("--error-model-source", default=os.getenv("WEATHER_ERROR_MODEL_SOURCE", "mos_gfs_18z_archive"))
    p.add_argument("--error-model-hour", type=int, default=int(os.getenv("WEATHER_ERROR_MODEL_HOUR", "12")))
    p.add_argument("--min-ev", type=float, default=float(os.getenv("WEATHER_MIN_EV", "0.02")))
    p.add_argument("--min-q", type=float, default=float(os.getenv("WEATHER_MIN_Q", "0.05")))
    p.add_argument("--fee-cents", type=int, default=int(os.getenv("WEATHER_BUY_FEE_CENTS", "2")))
    p.add_argument("--trading-hour", type=int, default=6, help="Local trading hour (default: 6am)")
    p.add_argument(
        "--calibration-only",
        action="store_true",
        help="Run calibration analysis only (no Kalshi market data required). "
             "Tests if probability distributions are well-calibrated.",
    )
    p.add_argument("--format", choices=["text", "json"], default="text")
    p.add_argument("--output", default="", help="Output file (default: stdout)")
    p.add_argument("--save", action="store_true", help="Save to outputs/backtests/")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    db_path = Path(args.db)
    config_path = Path(args.config)

    # Get city list
    if args.city:
        city_keys = [args.city]
    else:
        city_keys = load_city_keys(config_path)

    all_results = []

    for city_key in city_keys:
        tz = load_city_tz(config_path, city_key)

        if args.verbose:
            print(f"\nRunning backtest for {city_key}...")

        results = run_simple_backtest(
            db_path=db_path,
            city_key=city_key,
            timezone=tz,
            start_date=args.start,
            end_date=args.end,
            forecast_source=args.forecast_source,
            error_model_source=args.error_model_source,
            error_model_hour=args.error_model_hour,
            min_ev=args.min_ev,
            min_q=args.min_q,
            fee_cents=args.fee_cents,
            trading_hour=args.trading_hour,
            calibration_only=args.calibration_only,
            verbose=args.verbose,
        )

        all_results.append(results)

        if args.format == "text" and not args.output:
            print(format_results(results, "text"))
            print()

    # Output
    if args.format == "json":
        output = json.dumps({
            "backtest_params": {
                "start_date": args.start,
                "end_date": args.end,
                "forecast_source": args.forecast_source,
                "error_model_source": args.error_model_source,
                "min_ev": args.min_ev,
                "min_q": args.min_q,
                "fee_cents": args.fee_cents,
            },
            "results": all_results,
        }, indent=2)

        if args.output:
            Path(args.output).write_text(output, encoding="utf-8")
            print(f"Wrote {args.output}")
        else:
            print(output)

    # Save
    if args.save:
        OUT_BASE.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(ZoneInfo("UTC")).strftime("%Y%m%dT%H%M%SZ")
        out_path = OUT_BASE / f"backtest_{args.start}_{args.end}_{timestamp}.json"
        out_data = {
            "backtest_params": {
                "start_date": args.start,
                "end_date": args.end,
                "forecast_source": args.forecast_source,
                "error_model_source": args.error_model_source,
                "min_ev": args.min_ev,
                "min_q": args.min_q,
                "fee_cents": args.fee_cents,
            },
            "results": all_results,
        }
        out_path.write_text(json.dumps(out_data, indent=2), encoding="utf-8")
        print(f"[saved] {out_path}")

    # Summary
    if len(all_results) > 1:
        total_pnl = sum(r.get("trading_metrics", {}).get("total_pnl", 0) for r in all_results)
        total_trades = sum(r.get("trading_metrics", {}).get("n_trades", 0) for r in all_results)
        avg_calibration = sum(
            r.get("calibration_metrics", {}).get("calibration_error", 0) or 0
            for r in all_results
            if r.get("calibration_metrics", {}).get("calibration_error") is not None
        ) / max(1, sum(1 for r in all_results if r.get("calibration_metrics", {}).get("calibration_error") is not None))

        print("\n=== Summary Across All Cities ===")
        print(f"Total PnL: ${total_pnl:.2f}")
        print(f"Total Trades: {total_trades}")
        print(f"Avg Calibration Error: {avg_calibration:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
