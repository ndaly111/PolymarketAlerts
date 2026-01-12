# Polymarket OPEN/CLOSE Discord Alerts

Runs every 15 minutes via GitHub Actions and posts Discord alerts when the monitored user:
- OPENS a position (BUY when prior size was 0)
- CLOSES a position (SELL that brings size to 0)

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

**Inputs**
- Provide sportsbook odds either via `SPORTSBOOK_ODDS_URL` (recommended) or a local `data/sportsbook_odds.json` file.
- A sample format lives at `data/sportsbook_odds.sample.json`.
- Team names in output are shown exactly as Polymarket provides them (no aliasing/uniforming).
- Matching is lightweight (token overlap + time window). If a game can’t be matched, it still appears in `reports/unmatched_polymarket.csv` with teams + start time + URL.
- Filters: `sportsbook_ml < +200` and `polymarket_ml > -300` (plus a minimum edge threshold).

**Workflow**
The `Polymarket Sports Value Scan` GitHub Action runs daily and can be triggered manually. It reads the Discord webhook from the `DISCORD_SPORTS_ALERT` secret and uploads the `reports/` directory as an artifact.
