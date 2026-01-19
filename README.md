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

---

## Kalshi lines scanner (moneyline, spread, total)

This repo now includes a **separate** workflow and script to scan Kalshi sports *lines* (no player props yet):

- **Moneyline (H2H)**
- **Spreads**
- **Totals / Over-Under**

It compares Kalshi prices vs sportsbook consensus (The Odds API), calculates a no-vig fair probability, and reports the biggest gaps.

### Important: Kalshi opening fee

Kalshi charges a taker fee based on price and rounded up to the nearest cent. By default `KALSHI_BUY_FEE_CENTS=0` to use the schedule; set a positive value to force a fixed per-contract fee override.

Example: if a contract is listed at **72¢**, the scanner adds the schedule fee (or your override) to compute the all-in price before evaluating value.

### Matching Kalshi games to Odds API events

The scanner matches Kalshi markets to The Odds API by (1) normalized team-name similarity and (2) game start time proximity.
If matching is too strict/loose for a league, you can set:
- `KALSHI_EVENT_TIME_TOLERANCE_HOURS` (int, default 36): max allowed start-time difference when selecting the best Odds API event.
- `KALSHI_MIN_TEAM_MATCH_SCORE` (float, default 1.30): minimum combined team similarity score.
- `KALSHI_MIN_TEAM_NAME_SCORE` (float, default 0.65): minimum per-team similarity score.

### Volume filter toggle (manual runs)

The workflow has a `volume_filter` input:
- `true` (default): enforce `min_volume`
- `false`: ignores `min_volume` by setting `KALSHI_MIN_VOLUME=0` and forces `KALSHI_VOLUME_MISSING_POLICY=include`

### Run in GitHub Actions

Workflow: `.github/workflows/kalshi_sports_value.yml`

Secrets used:
- `THE_ODDS_API` (or `ODDS_API_KEY`)
- `DISCORD_WEBHOOK_URL` (or `DISCORD_SPORTS_ALERT`)
- `KALSHI_KEY_ID` (Kalshi API key id)
- `KALSHI_PRIVATE_KEY` (Kalshi RSA private key PEM)

Optional:
- `KALSHI_BASE` (repo variable recommended): Kalshi trade API base URL. If you set only the host, the script will append `/trade-api/v2`.
- `KALSHI_SPORT_KEYS` (repo variable): default Odds API sport keys for scheduled runs.

Manual inputs let you change sport keys, edge thresholds, lookahead window, fee cents, and (optionally) the Kalshi series tickers.
