#!/usr/bin/env python3
"""Master data collection orchestrator for weather trading.

This script runs all data collection jobs with proper logging and error handling.
Designed to be scheduled via cron, launchd, or run manually.

Collection profiles:
- hourly: Forecasts + Kalshi markets (run every hour 4am-8pm local)
- frequent: Just Kalshi markets (run every 15-30 min during trading hours)
- daily: CLI observations + error model updates (run once after market close)

Example usage:
    # Run all hourly collections
    python run_collection.py --profile hourly

    # Run specific collectors
    python run_collection.py --collect forecasts,kalshi

    # Run for specific city
    python run_collection.py --profile hourly --city NHIGH

    # Dry run (show what would be collected)
    python run_collection.py --profile hourly --dry-run

Cron examples:
    # Every hour from 4am-8pm ET
    0 4-20 * * * cd /path/to/repo && python weather/scripts/run_collection.py --profile hourly

    # Every 30 min during trading hours
    */30 5-18 * * * cd /path/to/repo && python weather/scripts/run_collection.py --collect kalshi

    # Daily at 11pm for observations
    0 23 * * * cd /path/to/repo && python weather/scripts/run_collection.py --profile daily
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import yaml

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "weather" / "scripts"
LOG_DIR = ROOT / "weather" / "outputs" / "logs"
DEFAULT_CONFIG = ROOT / "weather" / "config" / "cities.yml"

# Collection profiles
PROFILES = {
    "hourly": {
        "description": "Hourly collection: multi-model forecasts, NBM, Kalshi markets",
        "collectors": ["forecasts", "nbm", "kalshi"],
    },
    "frequent": {
        "description": "Frequent collection: Kalshi markets only (for intraday trading)",
        "collectors": ["kalshi"],
    },
    "daily": {
        "description": "Daily collection: CLI observations, error models (run after market close)",
        "collectors": ["observations", "error_models"],
    },
    "full": {
        "description": "Full collection: all data collectors",
        "collectors": ["forecasts", "nbm", "kalshi", "observations", "intraday"],
    },
    # Complete trading workflows
    "morning_scan": {
        "description": "Morning trading scan: collect data, compute edges, post to Discord",
        "collectors": ["forecasts", "nbm", "kalshi", "fair_prices", "edges", "ml_edges", "ev_report", "discord_edges"],
    },
    "quick_scan": {
        "description": "Quick scan: Kalshi markets + compute edges (no forecast refresh)",
        "collectors": ["kalshi", "edges"],
    },
    "report_only": {
        "description": "Generate reports without collecting new data",
        "collectors": ["fair_prices", "edges", "ev_report"],
    },
    "discord_only": {
        "description": "Post existing results to Discord",
        "collectors": ["discord_edges", "discord_forecast"],
    },
}

# Collector definitions
COLLECTORS = {
    "forecasts": {
        "script": "collect_multi_model_forecasts.py",
        "description": "Multi-model forecasts (HRRR, ECMWF, GFS, NWS)",
        "requires_auth": False,
    },
    "nbm": {
        "script": "collect_nbm_forecast.py",
        "description": "NBM probabilistic forecasts (percentiles)",
        "requires_auth": False,
        "timeout": 60,  # Shorter timeout since it often falls back anyway
    },
    "kalshi": {
        "script": "collect_kalshi_weather_markets.py",
        "description": "Kalshi weather market orderbook snapshots",
        "requires_auth": True,
        "auth_vars": ["KALSHI_PRIVATE_KEY", "KALSHI_KEY_ID"],
    },
    "observations": {
        "script": "collect_cli_observed.py",
        "description": "CLI observed temperatures (settlement data)",
        "requires_auth": False,
    },
    "intraday": {
        "script": "collect_intraday_observations.py",
        "description": "Intraday temperature observations",
        "requires_auth": False,
    },
    "error_models": {
        "script": "build_error_models.py",
        "description": "Rebuild error PMF models from historical data",
        "requires_auth": False,
        "extra_args": ["--forecast-source", "mos_gfs_18z_archive", "--snapshot-hour", "12"],
    },
    # Computation and reporting
    "fair_prices": {
        "script": "compute_fair_prices.py",
        "description": "Compute fair prices from forecasts + error models",
        "requires_auth": False,
        "extra_args": ["--use-latest", "--error-model-source", "mos_gfs_18z_archive", "--error-model-hour", "12"],
    },
    "edges": {
        "script": "compute_edges.py",
        "description": "Compute trading edges (EV) from fair prices vs Kalshi",
        "requires_auth": False,
    },
    "ml_edges": {
        "script": "compute_ml_edges.py",
        "description": "Compute ML-based trading edges for all cities",
        "requires_auth": False,
        "extra_args": ["--all", "--save"],
    },
    "ev_report": {
        "script": "generate_ev_report.py",
        "description": "Generate EV report for current opportunities",
        "requires_auth": False,
        "extra_args": ["--save"],
    },
    # Discord notifications
    "discord_edges": {
        "script": "post_weather_edges_discord.py",
        "description": "Post trading edges to Discord",
        "requires_auth": True,
        "auth_vars": ["DISCORD_WEBHOOK_URL"],
    },
    "discord_forecast": {
        "script": "post_forecast_distribution_discord.py",
        "description": "Post forecast distribution to Discord",
        "requires_auth": True,
        "auth_vars": ["DISCORD_WEBHOOK_URL"],
    },
}


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Set up logging to both console and file."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(ZoneInfo("UTC")).strftime("%Y%m%dT%H%M%SZ")
    log_file = LOG_DIR / f"collection_{timestamp}.log"

    level = logging.DEBUG if verbose else logging.INFO

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
    console_handler.setFormatter(console_formatter)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    file_handler.setFormatter(file_formatter)

    logger = logging.getLogger("collection")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def check_auth(collector: str) -> bool:
    """Check if required auth environment variables are set."""
    config = COLLECTORS.get(collector, {})
    if not config.get("requires_auth"):
        return True

    auth_vars = config.get("auth_vars", [])
    for var in auth_vars:
        if os.getenv(var):
            return True

    return False


def run_collector(
    collector: str,
    city: Optional[str] = None,
    target_date: Optional[str] = None,
    dry_run: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Run a single collector script."""
    config = COLLECTORS.get(collector)
    if not config:
        return {"collector": collector, "status": "error", "error": "Unknown collector"}

    script_path = SCRIPTS_DIR / config["script"]
    if not script_path.exists():
        return {"collector": collector, "status": "error", "error": f"Script not found: {script_path}"}

    # Check auth
    if config.get("requires_auth") and not check_auth(collector):
        auth_vars = config.get("auth_vars", [])
        return {
            "collector": collector,
            "status": "skipped",
            "reason": f"Missing auth: {', '.join(auth_vars)}",
        }

    # Build command
    cmd = [sys.executable, str(script_path)]

    # Add city filter if supported and requested
    if city:
        if collector in ["nbm", "kalshi"]:
            cmd.extend(["--city", city])
        # forecasts script doesn't support --city yet

    # Add target date if requested
    if target_date:
        if collector in ["forecasts", "nbm"]:
            cmd.extend(["--target-date", target_date])

    # Add extra args from config
    if config.get("extra_args"):
        cmd.extend(config["extra_args"])

    if dry_run:
        return {
            "collector": collector,
            "status": "dry_run",
            "command": " ".join(cmd),
        }

    # Run the script
    timeout_seconds = config.get("timeout", 300)  # Default 5 minutes
    start_time = datetime.now()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(ROOT)},
        )

        elapsed = (datetime.now() - start_time).total_seconds()

        if result.returncode == 0:
            if logger:
                logger.info(f"[{collector}] Success ({elapsed:.1f}s)")
            return {
                "collector": collector,
                "status": "success",
                "elapsed_seconds": elapsed,
                "stdout": result.stdout[-1000:] if len(result.stdout) > 1000 else result.stdout,
            }
        else:
            if logger:
                logger.warning(f"[{collector}] Failed (exit code {result.returncode})")
            return {
                "collector": collector,
                "status": "failed",
                "exit_code": result.returncode,
                "elapsed_seconds": elapsed,
                "stderr": result.stderr[-500:] if len(result.stderr) > 500 else result.stderr,
            }

    except subprocess.TimeoutExpired:
        if logger:
            logger.error(f"[{collector}] Timeout after {timeout_seconds}s")
        return {"collector": collector, "status": "timeout", "timeout_seconds": timeout_seconds}
    except Exception as e:
        if logger:
            logger.error(f"[{collector}] Error: {e}")
        return {"collector": collector, "status": "error", "error": str(e)}


def run_profile(
    profile: str,
    city: Optional[str] = None,
    target_date: Optional[str] = None,
    dry_run: bool = False,
    logger: Optional[logging.Logger] = None,
) -> List[Dict[str, Any]]:
    """Run all collectors in a profile."""
    profile_config = PROFILES.get(profile)
    if not profile_config:
        return [{"profile": profile, "status": "error", "error": "Unknown profile"}]

    collectors = profile_config["collectors"]
    results = []

    for collector in collectors:
        if logger:
            logger.info(f"Running {collector}...")
        result = run_collector(
            collector,
            city=city,
            target_date=target_date,
            dry_run=dry_run,
            logger=logger,
        )
        results.append(result)

    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run weather data collection jobs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Profiles:
  hourly   - Multi-model forecasts, NBM, Kalshi markets
  frequent - Kalshi markets only (for intraday)
  daily    - CLI observations, error models
  full     - All collectors

Collectors:
  forecasts    - HRRR, ECMWF, GFS, NWS forecasts
  nbm          - NBM probabilistic forecasts
  kalshi       - Kalshi market orderbooks
  observations - CLI settlement data
  intraday     - Intraday temperature observations
  error_models - Rebuild error PMF models
        """,
    )
    parser.add_argument(
        "--profile",
        choices=list(PROFILES.keys()),
        help="Run all collectors in a profile",
    )
    parser.add_argument(
        "--collect",
        help="Comma-separated list of collectors to run",
    )
    parser.add_argument(
        "--city",
        help="Filter to specific city key",
    )
    parser.add_argument(
        "--target-date",
        help="Target date YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be run without executing",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available profiles and collectors",
    )
    args = parser.parse_args()

    # List mode
    if args.list:
        print("=== Profiles ===")
        for name, config in PROFILES.items():
            print(f"  {name}: {config['description']}")
            print(f"    Collectors: {', '.join(config['collectors'])}")
            print()

        print("=== Collectors ===")
        for name, config in COLLECTORS.items():
            auth = " [requires auth]" if config.get("requires_auth") else ""
            print(f"  {name}: {config['description']}{auth}")
        return 0

    # Set up logging
    logger = setup_logging(args.verbose)
    logger.info(f"Starting data collection at {datetime.now(ZoneInfo('UTC')).isoformat()}")

    # Determine what to run
    results = []

    if args.profile:
        logger.info(f"Running profile: {args.profile}")
        results = run_profile(
            args.profile,
            city=args.city,
            target_date=args.target_date,
            dry_run=args.dry_run,
            logger=logger,
        )
    elif args.collect:
        collectors = [c.strip() for c in args.collect.split(",")]
        for collector in collectors:
            logger.info(f"Running collector: {collector}")
            result = run_collector(
                collector,
                city=args.city,
                target_date=args.target_date,
                dry_run=args.dry_run,
                logger=logger,
            )
            results.append(result)
    else:
        parser.print_help()
        return 1

    # Summary
    success = sum(1 for r in results if r.get("status") == "success")
    failed = sum(1 for r in results if r.get("status") in ("failed", "error", "timeout"))
    skipped = sum(1 for r in results if r.get("status") == "skipped")

    logger.info(f"Collection complete: {success} success, {failed} failed, {skipped} skipped")

    # Print results summary
    print("\n=== Results ===")
    for r in results:
        status = r.get("status", "unknown")
        collector = r.get("collector", "unknown")

        if status == "success":
            elapsed = r.get("elapsed_seconds", 0)
            print(f"  ✓ {collector} ({elapsed:.1f}s)")
        elif status == "skipped":
            reason = r.get("reason", "unknown")
            print(f"  ○ {collector} - skipped: {reason}")
        elif status == "dry_run":
            cmd = r.get("command", "")
            print(f"  ~ {collector}: {cmd}")
        else:
            error = r.get("error") or r.get("stderr", "")[:100]
            print(f"  ✗ {collector} - {status}: {error}")

    return 1 if failed > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
