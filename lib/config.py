"""
Centralized trading configuration.

Provides TradingConfig dataclass with all environment variables
and factory methods for props/sports pipelines.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional


def _env_int(name: str, default: int) -> int:
    """Parse environment variable as int with default."""
    value = os.getenv(name)
    if value is None or not str(value).strip():
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    """Parse environment variable as float with default."""
    value = os.getenv(name)
    if value is None or not str(value).strip():
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    """Parse environment variable as bool with default."""
    value = os.getenv(name)
    if value is None:
        return default
    return str(value).strip().lower() in ("1", "true", "yes")


@dataclass(frozen=True)
class TradingConfig:
    """
    Centralized configuration for trading pipelines.

    All trading parameters in one place with sensible defaults.
    Use factory methods for pipeline-specific configs.
    """

    # Edge thresholds
    min_edge: float = 0.05  # 5% minimum edge to trade
    min_initial_edge: float = 0.05  # Initial scan threshold
    min_final_edge: float = 0.05  # Final verification threshold
    max_edge_suspicious: float = 0.30  # Flag edges > 30% as suspicious

    # Price guardrails (in cents)
    min_price_cents: int = 5  # Block < 5 cent trades
    max_price_cents: int = 95  # Block > 95 cent trades
    max_kalshi_ask_cents: int = 67  # Better than -200
    min_kalshi_ask_cents: int = 33  # Better than +200

    # Books requirements
    min_books: int = 2  # Minimum books for consensus

    # Fee configuration (in cents)
    fee_cents: int = 2  # Kalshi 2 cent fee per contract

    # Trading limits
    max_trades_per_day: int = 10
    contracts_per_trade: int = 1
    order_timeout_seconds: int = 30

    # Kelly criterion
    kelly_fraction: float = 0.5  # Half Kelly

    # Outlier detection
    outlier_z_threshold: float = 2.0  # Z-score threshold for outliers

    # Mode flags
    dry_run: bool = False
    debug_mode: bool = False

    # Discord
    discord_webhook: str = ""
    discord_mention: str = ""

    # Database
    db_path: str = ""

    @classmethod
    def for_props(cls) -> TradingConfig:
        """Factory for props pipeline configuration."""
        return cls(
            min_edge=_env_float("MIN_EDGE", 0.05),
            min_initial_edge=_env_float("MIN_INITIAL_EV", 0.05),
            min_final_edge=_env_float("MIN_FINAL_EV", 0.05),
            max_edge_suspicious=_env_float("MAX_EDGE_SUSPICIOUS", 0.30),
            min_price_cents=_env_int("MIN_PRICE_CENTS", 5),
            max_price_cents=_env_int("MAX_PRICE_CENTS", 95),
            max_kalshi_ask_cents=_env_int("MAX_KALSHI_ASK", 67),
            min_kalshi_ask_cents=_env_int("MIN_KALSHI_ASK", 33),
            min_books=_env_int("MIN_BOOKS", 2),
            fee_cents=_env_int("FEE_CENTS", 2),
            max_trades_per_day=_env_int("MAX_TRADES_PER_DAY", 10),
            contracts_per_trade=_env_int("CONTRACTS_PER_TRADE", 1),
            order_timeout_seconds=_env_int("ORDER_TIMEOUT", 30),
            kelly_fraction=_env_float("KELLY_FRACTION", 0.5),
            outlier_z_threshold=_env_float("OUTLIER_Z_THRESHOLD", 2.0),
            dry_run=_env_bool("DRY_RUN", False),
            debug_mode=_env_bool("DEBUG_MODE", False),
            discord_webhook=os.getenv("DISCORD_PROPS_WEBHOOK", os.getenv("DISCORD_WEBHOOK_URL", "")),
            discord_mention=os.getenv("DISCORD_MENTION", ""),
            db_path=os.getenv("PROPS_DB_PATH", "props_trades.db"),
        )

    @classmethod
    def for_sports(cls) -> TradingConfig:
        """Factory for sports pipeline configuration."""
        return cls(
            min_edge=_env_float("SPORTS_MIN_EDGE", 0.05),
            min_initial_edge=_env_float("SPORTS_MIN_EDGE", 0.05),
            min_final_edge=_env_float("SPORTS_MIN_EDGE", 0.05),
            max_edge_suspicious=_env_float("SPORTS_MAX_EDGE_SUSPICIOUS", 0.30),
            min_price_cents=_env_int("SPORTS_MIN_PRICE_CENTS", 5),
            max_price_cents=_env_int("SPORTS_MAX_PRICE_CENTS", 95),
            max_kalshi_ask_cents=_env_int("SPORTS_MAX_ASK", 67),
            min_kalshi_ask_cents=_env_int("SPORTS_MIN_ASK", 33),
            min_books=_env_int("SPORTS_MIN_BOOKS", 2),
            fee_cents=_env_int("SPORTS_FEE_CENTS", 2),
            max_trades_per_day=_env_int("SPORTS_MAX_TRADES_PER_DAY", 10),
            contracts_per_trade=_env_int("SPORTS_CONTRACTS_PER_TRADE", 1),
            order_timeout_seconds=_env_int("SPORTS_ORDER_TIMEOUT", 30),
            kelly_fraction=_env_float("SPORTS_KELLY_FRACTION", 0.5),
            outlier_z_threshold=_env_float("SPORTS_OUTLIER_Z_THRESHOLD", 2.0),
            dry_run=_env_bool("SPORTS_DRY_RUN", False),
            debug_mode=_env_bool("DEBUG_MODE", False),
            discord_webhook=os.getenv("DISCORD_SPORTS_WEBHOOK", os.getenv("DISCORD_WEBHOOK_URL", "")),
            discord_mention=os.getenv("DISCORD_MENTION", ""),
            db_path=os.getenv("SPORTS_DB_PATH", "sports_trades.db"),
        )

    def fee_dollars(self) -> float:
        """Return fee in dollars."""
        return float(self.fee_cents) / 100.0
