"""
Shared probability pipeline for building temperature probability ladders.

Extracted from BacktestEngine._build_probability_ladder() so that both the
backtester and the live auto-trader share identical probability logic.

Data source priority (matched forecast+error pairs):
1. MOS GFS point forecast + MOS GFS error model (~90 samples/city/month)
2. Fallback: estimated percentiles from point forecast (lowest quality)

Optional enhancements (when data is available):
- NBM probabilistic percentiles (blended at 0.6 weight)
- Intraday adjustments via truncate_below() + shrink_dispersion()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from weather.lib.probability_ladder import (
    ProbabilityLadder,
    blend_ladders,
    from_percentiles,
    truncate_below,
    shrink_dispersion,
)
from weather.lib.fair import (
    build_ladder_from_forecast_and_error,
    build_ladder_from_nbm,
)
from weather.lib import db as db_lib

log = logging.getLogger(__name__)


@dataclass
class LadderResult:
    """Result of building a probability ladder."""
    ladder: ProbabilityLadder
    forecast_high_f: int
    source: str
    sources_used: List[str]
    forecast_source: str


def build_trading_ladder(
    db_path: Path,
    *,
    city_key: str,
    target_date: str,
    forecast_source: str = "mos_gfs_12z",
    error_model_source: str = "mos_gfs_18z_archive",
    error_model_hour: int = 12,
    use_nbm: bool = True,
    intraday_floor: Optional[int] = None,
    intraday_progress: Optional[float] = None,
) -> Optional[LadderResult]:
    """
    Build a probability ladder for a city/date using the best available data.

    This is the single entry point for probability estimation, used by both
    the live auto-trader and the backtesting engine.

    Args:
        db_path: Path to the weather forecast accuracy database
        city_key: City identifier (e.g., "NHIGH")
        target_date: Target date string "YYYY-MM-DD"
        forecast_source: Which forecast to use as point estimate
        error_model_source: Which error model to use for spread estimation
        error_model_hour: Snapshot hour for error model lookup
        use_nbm: Whether to try NBM percentiles for blending
        intraday_floor: If set, truncate distribution below this temp
        intraday_progress: If set (0-1), shrink dispersion by this amount

    Returns:
        LadderResult with the probability ladder, or None if no data available
    """
    target_dt = datetime.strptime(target_date, "%Y-%m-%d")
    month = target_dt.month
    sources_used: List[str] = []

    # --- Get point forecast ---
    forecast = db_lib.fetch_latest_forecast_snapshot(
        db_path,
        city_key=city_key,
        target_date_local=target_date,
        source=forecast_source,
    )

    # If primary source unavailable, try MOS GFS 12z, then open_meteo
    if not forecast or forecast.get("forecast_high_f") is None:
        for fallback_src in ["mos_gfs_12z", "open_meteo", "nws_hourly"]:
            if fallback_src == forecast_source:
                continue
            forecast = db_lib.fetch_latest_forecast_snapshot(
                db_path,
                city_key=city_key,
                target_date_local=target_date,
                source=fallback_src,
            )
            if forecast and forecast.get("forecast_high_f") is not None:
                log.info(
                    "Primary forecast %s unavailable for %s on %s, "
                    "falling back to %s",
                    forecast_source, city_key, target_date, fallback_src,
                )
                break

    if not forecast or forecast.get("forecast_high_f") is None:
        log.warning("No forecast available for %s on %s", city_key, target_date)
        return None

    forecast_high = forecast["forecast_high_f"]
    snapshot_hour = forecast.get("snapshot_hour_local", 6)
    actual_forecast_source = forecast.get("source", forecast_source)

    # --- Build ladders from available sources ---
    ladders_to_blend: List[Tuple[ProbabilityLadder, float]] = []

    # 1. Try NBM percentiles (highest quality if available)
    if use_nbm:
        nbm = db_lib.fetch_latest_nbm_forecast(
            db_path,
            city_key=city_key,
            target_date_local=target_date,
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
            sources_used.append("nbm")

    # 2. Try error model (matched source — MOS GFS error model for MOS GFS forecast)
    error_model = db_lib.fetch_error_model(
        db_path,
        city_key=city_key,
        month=month,
        snapshot_hour_local=error_model_hour,
        source=error_model_source,
    )
    if error_model and error_model.get("pmf"):
        # Weight depends on whether NBM is also available
        error_weight = 0.4 if sources_used else 1.0
        error_ladder = build_ladder_from_forecast_and_error(
            forecast_high,
            error_model["pmf"],
            source="error_model",
        )
        ladders_to_blend.append((error_ladder, error_weight))
        sources_used.append(f"error_model({error_model_source},n={error_model.get('n_samples', '?')})")

    # --- Blend or use single source ---
    if len(ladders_to_blend) >= 2:
        ladder = blend_ladders(
            ladders_to_blend,
            source="+".join(l.source for l, _ in ladders_to_blend),
        )
    elif ladders_to_blend:
        ladder = ladders_to_blend[0][0]
    else:
        # Fallback: estimate from point forecast (lowest quality)
        log.warning(
            "No error model for %s on %s — falling back to estimated "
            "percentiles from point forecast (lowest quality)",
            city_key, target_date,
        )
        from weather.lib.nbm import estimate_percentiles_from_point_forecast
        p10, p25, p50, p75, p90 = estimate_percentiles_from_point_forecast(
            forecast_high, month, snapshot_hour,
        )
        ladder = from_percentiles(p10, p25, p50, p75, p90, source="estimated")
        sources_used.append("estimated_percentiles")

    # --- Apply intraday adjustments ---
    if intraday_floor is not None and intraday_floor > 0:
        ladder = truncate_below(ladder, intraday_floor)
        sources_used.append(f"truncated(floor={intraday_floor})")

    if intraday_progress is not None and intraday_progress > 0:
        ladder = shrink_dispersion(ladder, intraday_progress)
        sources_used.append(f"shrunk(progress={intraday_progress:.2f})")

    return LadderResult(
        ladder=ladder,
        forecast_high_f=forecast_high,
        source=ladder.source,
        sources_used=sources_used,
        forecast_source=actual_forecast_source,
    )
