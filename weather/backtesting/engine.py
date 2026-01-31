"""
Main backtest engine for weather trading strategies.

The engine simulates trading decisions day-by-day, using only data
that was available at each decision point. This ensures backtest
results reflect real-world performance.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from weather.backtesting.data_loader import AsOfDataLoader, trading_time_to_utc
from weather.backtesting.metrics import (
    TradeOutcome,
    brier_score,
    log_loss,
    calibration_error,
    expected_vs_realized_ev,
    compute_drawdown,
)
from weather.lib.probability_ladder import (
    ProbabilityLadder,
    from_percentiles,
    from_pmf,
    compute_ev_for_contract,
)
from weather.lib.fair import (
    build_ladder_from_forecast_and_error,
    build_ladder_from_nbm,
    build_blended_ladder,
)


@dataclass
class BacktestConfig:
    """Configuration for a backtest run."""
    city_key: str
    start_date: str  # YYYY-MM-DD
    end_date: str  # YYYY-MM-DD
    timezone: str = "America/New_York"

    # Trading window times (local)
    trading_hour: int = 5
    trading_minute: int = 10

    # Strategy parameters
    min_ev_threshold: float = 0.10  # 10% minimum EV
    min_prob: float = 0.05  # 5% minimum probability
    max_prob: float = 0.95  # 95% maximum probability
    max_price_cents: int = 85  # Maximum ask price

    # Data sources
    forecast_source: str = "open_meteo"
    use_nbm: bool = True
    use_ml_model: bool = False

    # Position sizing
    use_kelly: bool = False
    kelly_fraction: float = 0.25
    contracts_per_trade: int = 1

    # Fees
    fee_cents: int = 2


@dataclass
class DailyBacktestResult:
    """Result for a single day's backtest."""
    date: str
    forecast_available: bool
    market_available: bool
    forecast_high: Optional[int] = None
    observed_high: Optional[int] = None
    forecast_error: Optional[int] = None

    # Probability model outputs
    ladder_p10: Optional[int] = None
    ladder_p50: Optional[int] = None
    ladder_p90: Optional[int] = None

    # Trading opportunities found
    opportunities: List[Dict[str, Any]] = field(default_factory=list)

    # Trades executed
    trades: List[TradeOutcome] = field(default_factory=list)

    # Diagnostics
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BacktestResults:
    """Complete results from a backtest run."""
    config: BacktestConfig
    daily_results: List[DailyBacktestResult]
    summary: Dict[str, Any] = field(default_factory=dict)


class BacktestEngine:
    """
    Main engine for running backtests.

    Simulates trading decisions day-by-day with strict as-of data semantics.
    """

    def __init__(
        self,
        config: BacktestConfig,
        db_path: Path,
        model_dir: Optional[Path] = None,
    ):
        self.config = config
        self.db_path = Path(db_path)
        self.model_dir = model_dir
        self.data_loader = AsOfDataLoader(db_path)
        self.ml_model = None

        # Load ML model if configured
        if config.use_ml_model and model_dir:
            self._load_ml_model()

    def _load_ml_model(self):
        """Load ML model for the configured city."""
        try:
            from weather.lib.ml_models import get_model
            self.ml_model = get_model(self.config.city_key, self.model_dir)
            print(f"Loaded ML model for {self.config.city_key}")
        except Exception as e:
            print(f"Warning: Could not load ML model: {e}")

    def run(self) -> BacktestResults:
        """
        Run the backtest over the configured date range.

        Returns:
            BacktestResults with daily outcomes and summary statistics
        """
        print(f"Running backtest for {self.config.city_key}")
        print(f"  Date range: {self.config.start_date} to {self.config.end_date}")
        print(f"  Trading time: {self.config.trading_hour}:{self.config.trading_minute:02d} {self.config.timezone}")

        # Get dates with data
        dates = self.data_loader.get_date_range_with_data(
            self.config.city_key,
            self.config.start_date,
            self.config.end_date,
        )

        print(f"  Found {len(dates)} dates with forecast/observation data")

        daily_results = []
        all_probability_predictions = []
        all_outcomes = []
        all_trades = []

        for date in dates:
            result = self._backtest_single_day(date)
            daily_results.append(result)

            # Collect probability predictions for calibration analysis
            if result.ladder_p50 is not None and result.observed_high is not None:
                # Create a simple binary prediction: P(T >= observed)
                # This allows us to check if our distributions are calibrated
                pass

            # Collect trades
            all_trades.extend(result.trades)

        # Compute summary statistics
        summary = self._compute_summary(daily_results, all_trades)

        return BacktestResults(
            config=self.config,
            daily_results=daily_results,
            summary=summary,
        )

    def _backtest_single_day(self, date: str) -> DailyBacktestResult:
        """
        Backtest a single trading day.

        Simulates the complete trading workflow:
        1. Get forecast available at trading time
        2. Build probability ladder
        3. Get market prices
        4. Find trading opportunities
        5. Simulate trades
        6. Compare to actual outcome
        """
        result = DailyBacktestResult(date=date, forecast_available=False, market_available=False)

        # Convert trading time to UTC
        as_of_utc = trading_time_to_utc(
            date,
            self.config.trading_hour,
            self.config.trading_minute,
            self.config.timezone,
        )

        # Get forecast at trading time
        forecast = self.data_loader.get_forecast_at_time(
            self.config.city_key,
            date,
            as_of_utc,
            self.config.forecast_source,
        )

        if not forecast:
            result.diagnostics["skip_reason"] = "no_forecast"
            return result

        result.forecast_available = True
        result.forecast_high = forecast["forecast_high_f"]

        # Get observed high (settlement truth)
        result.observed_high = self.data_loader.get_observed_high(
            self.config.city_key,
            date,
        )

        if result.observed_high is not None:
            result.forecast_error = result.observed_high - result.forecast_high

        # Build probability ladder
        ladder = self._build_probability_ladder(date, as_of_utc, forecast)

        if ladder:
            result.ladder_p10 = ladder.p10
            result.ladder_p50 = ladder.p50
            result.ladder_p90 = ladder.p90
            result.diagnostics["ladder_source"] = ladder.source

        # Get market prices at trading time
        markets = self.data_loader.get_kalshi_market_at_time(
            self.config.city_key,
            date,
            as_of_utc,
        )

        if markets:
            result.market_available = True

        # Find trading opportunities
        if ladder and markets:
            opportunities = self._find_opportunities(ladder, markets)
            result.opportunities = opportunities

            # Simulate trades
            for opp in opportunities:
                trade = self._simulate_trade(opp, result.observed_high)
                if trade:
                    result.trades.append(trade)

        return result

    def _build_probability_ladder(
        self,
        date: str,
        as_of_utc: str,
        forecast: Dict[str, Any],
    ) -> Optional[ProbabilityLadder]:
        """Build probability ladder from available data."""
        forecast_high = forecast["forecast_high_f"]
        target_dt = datetime.strptime(date, "%Y-%m-%d")
        month = target_dt.month
        snapshot_hour = forecast.get("snapshot_hour_local", 6)

        ladders_to_blend = []

        # Try NBM-based ladder
        if self.config.use_nbm:
            nbm = self.data_loader.get_nbm_forecast_at_time(
                self.config.city_key,
                date,
                as_of_utc,
            )
            if nbm and nbm.get("p50") is not None:
                nbm_ladder = build_ladder_from_nbm(
                    p10=nbm["p10"],
                    p25=nbm["p25"],
                    p50=nbm["p50"],
                    p75=nbm["p75"],
                    p90=nbm["p90"],
                    source="nbm",
                )
                ladders_to_blend.append((nbm_ladder, 0.6))

        # Try error model-based ladder
        error_model = self.data_loader.get_error_model_at_time(
            self.config.city_key,
            month,
            snapshot_hour,
            self.config.forecast_source,
            as_of_utc,
        )
        if error_model and error_model.get("pmf"):
            error_ladder = build_ladder_from_forecast_and_error(
                forecast_high,
                error_model["pmf"],
                source="error_model",
            )
            ladders_to_blend.append((error_ladder, 0.4))

        # Blend or use single source
        if len(ladders_to_blend) >= 2:
            return build_blended_ladder(
                ladders_to_blend[0][0],
                ladders_to_blend[1][0],
                nbm_weight=ladders_to_blend[0][1],
                error_weight=ladders_to_blend[1][1],
            )
        elif ladders_to_blend:
            return ladders_to_blend[0][0]
        else:
            # Fallback: estimate from point forecast
            from weather.lib.nbm import estimate_percentiles_from_point_forecast
            p10, p25, p50, p75, p90 = estimate_percentiles_from_point_forecast(
                forecast_high, month, 18
            )
            return from_percentiles(p10, p25, p50, p75, p90, source="estimated")

    def _find_opportunities(
        self,
        ladder: ProbabilityLadder,
        markets: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Find trading opportunities from market prices."""
        opportunities = []
        fee = self.config.fee_cents / 100.0

        for market in markets:
            if market.get("status") != "active":
                continue

            # Parse market to determine contract type
            ticker = market.get("market_ticker", "")
            yes_ask = market.get("yes_ask")
            no_ask = market.get("no_ask")

            if yes_ask is None or no_ask is None:
                continue

            yes_price = yes_ask / 100.0
            no_price = no_ask / 100.0

            # Extract threshold from ticker (simplified)
            # Real implementation would parse properly
            contract_info = self._parse_contract_ticker(ticker)
            if not contract_info:
                continue

            contract_type = contract_info["type"]
            threshold_a = contract_info.get("a")
            threshold_b = contract_info.get("b")

            # Compute model probability
            if contract_type == "between":
                model_prob = ladder.prob_between(threshold_a, threshold_b)
            elif contract_type == "ge":
                model_prob = ladder.prob_above(threshold_a)
            elif contract_type == "le":
                model_prob = 1.0 - ladder.prob_above(threshold_a + 1)
            else:
                continue

            # Check probability bounds
            if not (self.config.min_prob <= model_prob <= self.config.max_prob):
                continue

            # Compute EV for YES side
            yes_ev = model_prob * (1 - yes_price) - (1 - model_prob) * yes_price - fee

            # Compute EV for NO side
            no_ev = (1 - model_prob) * (1 - no_price) - model_prob * no_price - fee

            # Check price limits
            best_side = None
            best_ev = 0.0
            best_price = 0.0

            if yes_ev >= self.config.min_ev_threshold and yes_ask <= self.config.max_price_cents:
                best_side = "YES"
                best_ev = yes_ev
                best_price = yes_price

            if no_ev > best_ev and no_ask <= self.config.max_price_cents:
                best_side = "NO"
                best_ev = no_ev
                best_price = no_price

            if best_side and best_ev >= self.config.min_ev_threshold:
                opportunities.append({
                    "market_ticker": ticker,
                    "contract_type": contract_type,
                    "threshold_a": threshold_a,
                    "threshold_b": threshold_b,
                    "model_prob": model_prob,
                    "side": best_side,
                    "price": best_price,
                    "ev": best_ev,
                    "yes_ask": yes_ask,
                    "no_ask": no_ask,
                })

        return opportunities

    def _parse_contract_ticker(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Parse Kalshi ticker to extract contract parameters.

        Example tickers:
        - KXHIGHNY-26JAN30-T70 -> ge (>= 70)
        - KXHIGHNY-26JAN30-B70.5 -> between (70-71)
        """
        try:
            parts = ticker.split("-")
            if len(parts) < 3:
                return None

            strike_part = parts[-1]

            if strike_part.startswith("B"):
                # Between contract
                threshold = float(strike_part[1:])
                a = int(threshold)
                b = a + 1 if threshold == int(threshold) else int(threshold) + 1
                return {"type": "between", "a": a, "b": b}
            elif strike_part.startswith("T"):
                # Threshold contract (>= threshold)
                threshold = int(float(strike_part[1:]))
                return {"type": "ge", "a": threshold}
            else:
                return None
        except Exception:
            return None

    def _simulate_trade(
        self,
        opportunity: Dict[str, Any],
        observed_high: Optional[int],
    ) -> Optional[TradeOutcome]:
        """Simulate trade outcome."""
        if observed_high is None:
            return None

        contract_type = opportunity["contract_type"]
        threshold_a = opportunity["threshold_a"]
        threshold_b = opportunity.get("threshold_b")
        side = opportunity["side"]
        price = opportunity["price"]

        # Determine if contract settled YES
        if contract_type == "between":
            settled_yes = threshold_a <= observed_high <= threshold_b
        elif contract_type == "ge":
            settled_yes = observed_high >= threshold_a
        elif contract_type == "le":
            settled_yes = observed_high <= threshold_a
        else:
            return None

        actual_outcome = 1 if settled_yes else 0

        # Compute PnL
        if side == "YES":
            if settled_yes:
                pnl = 1.0 - price - (self.config.fee_cents / 100.0)
            else:
                pnl = -price - (self.config.fee_cents / 100.0)
        else:  # NO side
            if not settled_yes:
                pnl = 1.0 - price - (self.config.fee_cents / 100.0)
            else:
                pnl = -price - (self.config.fee_cents / 100.0)

        return TradeOutcome(
            date=opportunity.get("date", ""),
            market_ticker=opportunity["market_ticker"],
            side=side,
            predicted_prob=opportunity["model_prob"] if side == "YES" else 1 - opportunity["model_prob"],
            market_price=price,
            predicted_ev=opportunity["ev"],
            actual_outcome=actual_outcome if side == "YES" else (1 - actual_outcome),
            pnl=pnl,
        )

    def _compute_summary(
        self,
        daily_results: List[DailyBacktestResult],
        all_trades: List[TradeOutcome],
    ) -> Dict[str, Any]:
        """Compute summary statistics from backtest results."""
        # Basic counts
        n_days = len(daily_results)
        n_forecasts = sum(1 for r in daily_results if r.forecast_available)
        n_markets = sum(1 for r in daily_results if r.market_available)
        n_trades = len(all_trades)

        # Forecast accuracy
        errors = [r.forecast_error for r in daily_results if r.forecast_error is not None]
        mae = sum(abs(e) for e in errors) / len(errors) if errors else float("nan")
        bias = sum(errors) / len(errors) if errors else float("nan")

        # Trading metrics
        trade_metrics = expected_vs_realized_ev(all_trades)

        # Cumulative PnL for drawdown
        cumulative_pnl = []
        running_pnl = 0.0
        for t in all_trades:
            running_pnl += t.pnl
            cumulative_pnl.append(running_pnl)

        max_dd, _, _ = compute_drawdown(cumulative_pnl) if cumulative_pnl else (0.0, 0, 0)

        return {
            "n_days": n_days,
            "n_forecasts": n_forecasts,
            "n_markets": n_markets,
            "n_trades": n_trades,
            "forecast_mae": round(mae, 2) if not isinstance(mae, float) or not mae != mae else None,
            "forecast_bias": round(bias, 2) if not isinstance(bias, float) or not bias != bias else None,
            "max_drawdown": round(max_dd, 4),
            **trade_metrics,
        }
