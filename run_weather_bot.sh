#!/bin/bash
set -euo pipefail
cd /home/ubuntu/PolymarketAlerts
set -a; source .env; set +a
export PYTHONPATH=/home/ubuntu/PolymarketAlerts
PYTHON=/home/ubuntu/venv/bin/python

# Pull latest code (restore bot-generated / binary files first to avoid merge conflicts)
# The DB and model .pkl files are updated locally by the pipeline but also committed
# by GitHub Actions, so we must restore them before pull to avoid conflicts.
git restore docs/data/summary.json docs/data/props.json docs/data/weather.json 2>/dev/null || true
git restore weather/data/weather_forecast_accuracy.db 2>/dev/null || true
git restore weather/models/*.pkl 2>/dev/null || true
git pull origin main --quiet

# Run pipeline
$PYTHON -m weather.scripts.collect_forecast_snapshot \
  --db weather/data/weather_forecast_accuracy.db

$PYTHON -m weather.scripts.collect_intraday_observations \
  --db weather/data/weather_forecast_accuracy.db

# Collect all model forecasts throughout the day (HRRR, ECMWF, GFS, NWS)
# Runs on every cron tick so we build an hourly time series of model consensus/spread
$PYTHON -m weather.scripts.collect_multi_model_forecasts \
  --db weather/data/weather_forecast_accuracy.db

# Collect upper-air features (z500, t850, dewpoint) from Open-Meteo pressure levels
$PYTHON -m weather.scripts.collect_upper_air \
  --db weather/data/weather_forecast_accuracy.db

# Collect climate teleconnection indices (NAO, AO, PNA, MEI, PDO, MJO) from NOAA
# These change monthly so we run on every tick but only update when stale
$PYTHON -m weather.scripts.collect_climate_indices \
  --db weather/data/weather_forecast_accuracy.db

$PYTHON -m weather.scripts.compute_fair_prices \
  --forecast-source "$FORECAST_SOURCE" \
  --use-latest \
  --db weather/data/weather_forecast_accuracy.db

$PYTHON -m weather.scripts.collect_kalshi_weather_markets \
  --series-tickers "$WEATHER_KALSHI_SERIES_TICKERS" \
  --require-city-match \
  --out-json weather/outputs/kalshi_collect.json

$PYTHON -m weather.scripts.compute_edges \
  --forecast-source "$FORECAST_SOURCE" \
  --date "$(date +%Y-%m-%d)" \
  --fee-cents "$WEATHER_BUY_FEE_CENTS" \
  --min-ev "$WEATHER_MIN_EV" \
  --min-q "$WEATHER_MIN_Q" \
  --require-ask \
  --db weather/data/weather_forecast_accuracy.db

# Low temperature fair prices and edges (paper trading only - no live orders)
$PYTHON -m weather.scripts.compute_fair_prices \
  --forecast-source "$FORECAST_SOURCE" \
  --use-latest \
  --metric low \
  --db weather/data/weather_forecast_accuracy.db || true

$PYTHON -m weather.scripts.compute_edges \
  --forecast-source "$FORECAST_SOURCE" \
  --date "$(date +%Y-%m-%d)" \
  --fee-cents "$WEATHER_BUY_FEE_CENTS" \
  --min-ev "$WEATHER_MIN_EV" \
  --min-q "$WEATHER_MIN_Q" \
  --metric low \
  --db weather/data/weather_forecast_accuracy.db || true

$PYTHON weather_auto_trade.py
