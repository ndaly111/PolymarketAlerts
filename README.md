# Polymarket OPEN/CLOSE Discord Alerts

Runs every 15 minutes via GitHub Actions and posts Discord alerts when the monitored user:
- OPENS a position (BUY when prior size was 0)
- CLOSES a position (SELL that brings size to 0)

## Workflows overview

- Polymarket OPEN/CLOSE Alerts: runs every 15 minutes.
- Polymarket Sports Value Scan: daily canonical scanner.
- (LEGACY) Moneyline Compare: manual-only reference workflow.

## Setup

1) In GitHub repo: Settings → Secrets and variables → Actions → New repository secret
   - For the open/close watcher: `DISCORD_WEBHOOK_URL`
   - For the sports value scan: `DISCORD_SPORTS_ALERT` (mapped to `DISCORD_WEBHOOK_URL` in the workflow)

2) Go to Actions tab and enable workflows if prompted.

3) Run the workflow once manually (Actions → workflow → Run workflow).
   - First run seeds state and sends no alerts.
   - Next scheduled runs will alert on new OPEN/CLOSE events.

## iPhone Push

Install the Discord iOS app and enable notifications for the channel/server.

## Polymarket Sports Value Scan

The `polymarket_sports_value.py` script pulls active Polymarket sports moneyline markets, compares them to sportsbook moneylines, and posts a single Discord message with the top value gaps. It also writes snapshots under `reports/` for auditing.

**Canonical workflow**
- Use the `Polymarket Sports Value Scan` workflow for the daily scanner.
- The legacy `Moneyline Compare` workflow exists for manual/reference use only and should not be scheduled.

**Inputs**
- Uses Polymarket Gamma API for markets and The Odds API (H2H / moneyline) for multi-book sportsbook pricing.
- Team names in output are shown exactly as Polymarket provides them (no aliasing/uniforming).
- Matching is lightweight (token overlap + time window). If a game can’t be matched, it still appears in `reports/unmatched_polymarket.csv` with teams + start time + URL.
- Filters: `sportsbook_ml < +200` and `polymarket_ml > -300` (plus a minimum edge threshold).

**Workflow**
The `Polymarket Sports Value Scan` GitHub Action runs daily and can be triggered manually. It reads the Discord webhook from the `DISCORD_SPORTS_ALERT` secret and uploads the `reports/` directory as an artifact.

### Required GitHub Secrets
- `THE_ODDS_API`: The Odds API key (legacy `ODDS_API_KEY` is also supported)
- `DISCORD_SPORTS_ALERT`: Discord webhook for alerts (legacy `DISCORD_WEBHOOK_URL` supported)

### Optional GitHub Variables (defaults exist)
- `MIN_EDGE` (float, default 0.03)
- `TOP_N` (int, default 20): number of edges to keep in the CSV
- `DISCORD_TOP_N` (int, default 8): number of lines to post to Discord
- `TIME_WINDOW_HOURS` (int, default 36)
- `MIN_BOOKS` (int, default 3): minimum sportsbooks required to compute consensus
- `ODDS_REGIONS` (string, default `us`)
- `POLY_FALLBACK` (0/1, default 1): retry Polymarket fetch with looser filters if empty
- `REQUIRE_OUTSIDE_RANGE` (0/1, default 1): require Polymarket moneyline to be outside sportsbook range
- `GAME_BETS_TAG_ID` (int, default 100639): Polymarket tag filter (can be overridden)
