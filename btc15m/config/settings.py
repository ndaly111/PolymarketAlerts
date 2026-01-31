"""Configuration for BTC 15-min trading module."""

import os

# Binance API (no auth needed for public endpoints)
BINANCE_BASE_URL = "https://api.binance.com"

# Chainlink BTC/USD aggregator on Polygon
CHAINLINK_BTC_USD_AGGREGATOR = "0xc907E116054Ad103354f2D350FD2514433D57F6f"
POLYGON_RPC_URLS = [
    "https://polygon-rpc.com",
    "https://rpc-mainnet.matic.quiknode.pro",
    "https://polygon-mainnet.g.alchemy.com/v2/demo",
]

# Kalshi BTC market series
KALSHI_BTC_SERIES_TICKER = "KXBTC"  # BTC price markets

# Data collection settings
COLLECT_INTERVAL_SEC = 15  # How often to collect data
CANDLE_HISTORY_1M = 60     # Number of 1-minute candles to fetch
CANDLE_HISTORY_5M = 24     # Number of 5-minute candles to fetch

# Indicator settings
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Paper trading
PAPER_TRADE_BANKROLL = 1000.0  # Starting paper balance
MIN_EDGE_THRESHOLD = 0.05     # Minimum edge to place bet (5%)

# Database
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "btc15m.db")
