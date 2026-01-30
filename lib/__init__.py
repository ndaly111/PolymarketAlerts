"""
Shared trading pipeline utilities.

Modules:
- config: Centralized TradingConfig dataclass
- edge: Fee-adjusted edge calculation
- validation: Price guardrails, edge sanity checks
- outlier: Sportsbook consensus outlier detection
- database: Unified DB schema and operations
- normalization: Player/team name normalization
- discord: Discord notification utilities
- http: Session creation with retry logic
"""

from lib.config import TradingConfig
from lib.edge import calculate_edge_with_fee, kelly_fraction
from lib.validation import validate_price, validate_edge, validate_ticker_matchup, TradeValidation
from lib.outlier import detect_outliers, consensus_fair_prob
from lib.normalization import normalize_player_name, normalize_team_name, name_similarity

__all__ = [
    "TradingConfig",
    "calculate_edge_with_fee",
    "kelly_fraction",
    "validate_price",
    "validate_edge",
    "validate_ticker_matchup",
    "TradeValidation",
    "detect_outliers",
    "consensus_fair_prob",
    "normalize_player_name",
    "normalize_team_name",
    "name_similarity",
]
