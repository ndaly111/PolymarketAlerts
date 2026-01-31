"""
Reporting and visualization for backtest results.

Generates performance reports including:
- Calibration curves
- PnL charts
- EV bucket analysis
- Summary statistics
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from weather.backtesting.engine import BacktestResults, DailyBacktestResult
from weather.backtesting.metrics import (
    TradeOutcome,
    calibration_curve,
    calibration_error,
    brier_score,
    log_loss,
    compute_probability_bins_accuracy,
    expected_vs_realized_ev,
    compute_drawdown,
    sharpe_ratio,
)


@dataclass
class BacktestReport:
    """Formatted report from backtest results."""
    results: BacktestResults
    generated_at: str = ""

    def __post_init__(self):
        self.generated_at = datetime.utcnow().isoformat()

    def generate_text_report(self) -> str:
        """Generate human-readable text report."""
        lines = []
        config = self.results.config
        summary = self.results.summary

        lines.append("=" * 70)
        lines.append("WEATHER TRADING BACKTEST REPORT")
        lines.append("=" * 70)
        lines.append("")

        # Configuration
        lines.append("CONFIGURATION")
        lines.append("-" * 40)
        lines.append(f"City:          {config.city_key}")
        lines.append(f"Date Range:    {config.start_date} to {config.end_date}")
        lines.append(f"Trading Time:  {config.trading_hour}:{config.trading_minute:02d} {config.timezone}")
        lines.append(f"Min EV:        {config.min_ev_threshold:.0%}")
        lines.append(f"Forecast:      {config.forecast_source}")
        lines.append(f"NBM:           {'Yes' if config.use_nbm else 'No'}")
        lines.append(f"ML Model:      {'Yes' if config.use_ml_model else 'No'}")
        lines.append("")

        # Data Coverage
        lines.append("DATA COVERAGE")
        lines.append("-" * 40)
        lines.append(f"Days in range:     {summary.get('n_days', 0)}")
        lines.append(f"Days with forecast:{summary.get('n_forecasts', 0)}")
        lines.append(f"Days with markets: {summary.get('n_markets', 0)}")
        lines.append("")

        # Forecast Accuracy
        lines.append("FORECAST ACCURACY")
        lines.append("-" * 40)
        mae = summary.get('forecast_mae')
        bias = summary.get('forecast_bias')
        lines.append(f"Mean Absolute Error: {mae:.2f}F" if mae else "Mean Absolute Error: N/A")
        lines.append(f"Bias:                {bias:+.2f}F" if bias else "Bias: N/A")
        lines.append("")

        # Trading Performance
        lines.append("TRADING PERFORMANCE")
        lines.append("-" * 40)
        n_trades = summary.get('n_trades', 0)
        lines.append(f"Total Trades:        {n_trades}")

        if n_trades > 0:
            total_pnl = summary.get('total_realized_pnl', 0)
            avg_pnl = summary.get('avg_realized_pnl', 0)
            win_rate = summary.get('win_rate', 0)
            max_dd = summary.get('max_drawdown', 0)

            lines.append(f"Total PnL:           ${total_pnl:.2f}")
            lines.append(f"Avg PnL/Trade:       ${avg_pnl:.4f}")
            lines.append(f"Win Rate:            {win_rate:.1%}")
            lines.append(f"Max Drawdown:        ${max_dd:.2f}")

            total_ev = summary.get('total_predicted_ev', 0)
            ev_ratio = summary.get('ev_to_pnl_ratio')
            lines.append("")
            lines.append(f"Predicted EV Total:  ${total_ev:.2f}")
            if ev_ratio and not (isinstance(ev_ratio, float) and ev_ratio != ev_ratio):
                lines.append(f"EV to PnL Ratio:     {ev_ratio:.2f}x")
        lines.append("")

        # EV Bucket Analysis
        lines.append("EV BUCKET ANALYSIS")
        lines.append("-" * 40)
        for bucket in ["0-5%", "5-10%", "10-15%", "15-20%", "20%+"]:
            n = summary.get(f'bucket_{bucket}_n', 0)
            if n > 0:
                wr = summary.get(f'bucket_{bucket}_win_rate', 0)
                pnl = summary.get(f'bucket_{bucket}_pnl', 0)
                lines.append(f"  {bucket:8s}: {n:3d} trades, {wr:.1%} win rate, ${pnl:.2f} PnL")
        lines.append("")

        # Footer
        lines.append("=" * 70)
        lines.append(f"Report generated: {self.generated_at}")
        lines.append("")

        return "\n".join(lines)

    def generate_json_report(self) -> Dict[str, Any]:
        """Generate machine-readable JSON report."""
        config = self.results.config

        # Collect all trades
        all_trades = []
        for daily in self.results.daily_results:
            for trade in daily.trades:
                all_trades.append({
                    "date": daily.date,
                    "market_ticker": trade.market_ticker,
                    "side": trade.side,
                    "predicted_prob": trade.predicted_prob,
                    "market_price": trade.market_price,
                    "predicted_ev": trade.predicted_ev,
                    "actual_outcome": trade.actual_outcome,
                    "pnl": trade.pnl,
                })

        # Daily PnL series
        daily_pnl = []
        for daily in self.results.daily_results:
            day_pnl = sum(t.pnl for t in daily.trades)
            if daily.trades:
                daily_pnl.append({
                    "date": daily.date,
                    "pnl": day_pnl,
                    "n_trades": len(daily.trades),
                })

        return {
            "metadata": {
                "generated_at": self.generated_at,
                "city_key": config.city_key,
                "start_date": config.start_date,
                "end_date": config.end_date,
            },
            "config": {
                "trading_hour": config.trading_hour,
                "trading_minute": config.trading_minute,
                "timezone": config.timezone,
                "min_ev_threshold": config.min_ev_threshold,
                "forecast_source": config.forecast_source,
                "use_nbm": config.use_nbm,
                "use_ml_model": config.use_ml_model,
            },
            "summary": self.results.summary,
            "trades": all_trades,
            "daily_pnl": daily_pnl,
        }

    def save_report(self, output_dir: Path, prefix: str = "backtest") -> Dict[str, Path]:
        """
        Save report in multiple formats.

        Args:
            output_dir: Directory to save reports
            prefix: Filename prefix

        Returns:
            Dict mapping format -> file path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        config = self.results.config
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        base_name = f"{prefix}_{config.city_key}_{timestamp}"

        saved = {}

        # Text report
        text_path = output_dir / f"{base_name}.txt"
        text_path.write_text(self.generate_text_report())
        saved["text"] = text_path

        # JSON report
        json_path = output_dir / f"{base_name}.json"
        json_path.write_text(json.dumps(self.generate_json_report(), indent=2))
        saved["json"] = json_path

        print(f"Saved reports to {output_dir}/")
        for fmt, path in saved.items():
            print(f"  {fmt}: {path.name}")

        return saved

    def print_summary(self) -> None:
        """Print summary to console."""
        print(self.generate_text_report())


def compare_reports(
    reports: List[BacktestReport],
    output_path: Optional[Path] = None,
) -> str:
    """
    Compare multiple backtest reports side by side.

    Useful for comparing different strategies or configurations.

    Args:
        reports: List of backtest reports
        output_path: Optional path to save comparison

    Returns:
        Comparison text
    """
    if not reports:
        return "No reports to compare"

    lines = []
    lines.append("=" * 80)
    lines.append("BACKTEST COMPARISON")
    lines.append("=" * 80)
    lines.append("")

    # Header row
    headers = ["Metric"] + [r.results.config.city_key for r in reports]
    col_width = 15
    lines.append(" | ".join(h.ljust(col_width) for h in headers))
    lines.append("-" * (col_width * len(headers) + 3 * (len(headers) - 1)))

    # Metrics to compare
    metrics = [
        ("Days", "n_days"),
        ("Trades", "n_trades"),
        ("Win Rate", "win_rate"),
        ("Total PnL", "total_realized_pnl"),
        ("Avg PnL", "avg_realized_pnl"),
        ("Max DD", "max_drawdown"),
        ("EV Ratio", "ev_to_pnl_ratio"),
    ]

    for label, key in metrics:
        row = [label]
        for report in reports:
            val = report.results.summary.get(key)
            if val is None or (isinstance(val, float) and val != val):
                row.append("N/A")
            elif key == "win_rate":
                row.append(f"{val:.1%}")
            elif key in ("total_realized_pnl", "avg_realized_pnl", "max_drawdown"):
                row.append(f"${val:.2f}")
            elif key == "ev_to_pnl_ratio":
                row.append(f"{val:.2f}x")
            else:
                row.append(str(val))

        lines.append(" | ".join(str(v).ljust(col_width) for v in row))

    lines.append("")
    lines.append("=" * 80)

    text = "\n".join(lines)

    if output_path:
        Path(output_path).write_text(text)
        print(f"Saved comparison to {output_path}")

    return text
