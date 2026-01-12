# Polymarket OPEN/CLOSE Discord Alerts

Runs every 15 minutes via GitHub Actions and posts Discord alerts when the monitored user:
- OPENS a position (BUY when prior size was 0)
- CLOSES a position (SELL that brings size to 0)

## Setup

1) In GitHub repo: Settings → Secrets and variables → Actions → New repository secret
   - Name: DISCORD_WEBHOOK_URL
   - Value: (paste your Discord webhook URL)

2) Go to Actions tab and enable workflows if prompted.

3) Run the workflow once manually (Actions → workflow → Run workflow).
   - First run seeds state and sends no alerts.
   - Next scheduled runs will alert on new OPEN/CLOSE events.

## iPhone Push

Install the Discord iOS app and enable notifications for the channel/server.
