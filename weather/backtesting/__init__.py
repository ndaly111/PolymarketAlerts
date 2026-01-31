"""
Backtesting framework for weather trading strategies.

This module provides tools for historical validation of probability models
and trading strategies with strict temporal semantics to prevent leakage.

Key components:
- engine.py: Main backtest loop with as-of data loading
- data_loader.py: Point-in-time data retrieval
- metrics.py: Probability calibration and trading performance metrics
- report.py: Performance visualization and reporting

Usage:
    from weather.backtesting import BacktestEngine, BacktestConfig

    config = BacktestConfig(
        city_key="NHIGH",
        start_date="2025-01-01",
        end_date="2025-12-31",
    )
    engine = BacktestEngine(config)
    results = engine.run()
    report = engine.generate_report()
"""

from weather.backtesting.engine import BacktestEngine, BacktestConfig
from weather.backtesting.data_loader import AsOfDataLoader
from weather.backtesting.metrics import (
    brier_score,
    log_loss,
    calibration_curve,
    expected_vs_realized_ev,
)
from weather.backtesting.report import BacktestReport

__all__ = [
    "BacktestEngine",
    "BacktestConfig",
    "AsOfDataLoader",
    "brier_score",
    "log_loss",
    "calibration_curve",
    "expected_vs_realized_ev",
    "BacktestReport",
]
