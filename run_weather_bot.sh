#!/bin/bash
set -euo pipefail
cd /home/ubuntu/PolymarketAlerts
source .env
export PYTHONPATH=/home/ubuntu/PolymarketAlerts

# Pull latest code
git pull origin main --quiet

# Run pipeline
python -m weather.scripts.collect_forecast_snapshot \
  --forecast-source "$FORECAST_SOURCE" \
  --use-latest \
  --db weather/data/weather_forecast_accuracy.db

python -m weather.scripts.collect_intraday_observations \
  --db weather/data/weather_forecast_accuracy.db

python -m weather.scripts.compute_fair_prices \
  --forecast-source "$FORECAST_SOURCE" \
  --use-latest \
  --db weather/data/weather_forecast_accuracy.db

python -m weather.scripts.collect_kalshi_weather_markets \
  --series-tickers "$WEATHER_KALSHI_SERIES_TICKERS" \
  --require-city-match \
  --out-json weather/outputs/kalshi_collect.json

python -m weather.scripts.compute_edges \
  --forecast-source "$FORECAST_SOURCE" \
  --date "$(date +%Y-%m-%d)" \
  --fee-cents "$WEATHER_BUY_FEE_CENTS" \
  --min-ev "$WEATHER_MIN_EV" \
  --min-q "$WEATHER_MIN_Q" \
  --require-ask \
  --db weather/data/weather_forecast_accuracy.db

python weather_auto_trade.py
