#!/usr/bin/env python3
"""
Schedule automatic ML model retraining.

This script sets up periodic retraining of the ML models to incorporate
new forecast/observation data.

Usage:
    # Install the cron job (weekly retraining on Sundays at 3am)
    python weather/scripts/schedule_ml_retrain.py --install

    # Remove the cron job
    python weather/scripts/schedule_ml_retrain.py --uninstall

    # Run retraining manually now
    python weather/scripts/schedule_ml_retrain.py --run-now

    # Show current cron status
    python weather/scripts/schedule_ml_retrain.py --status
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "weather" / "scripts"
LOGS_DIR = ROOT / "weather" / "outputs" / "logs"

# Cron schedule: Sunday at 3:00 AM
CRON_SCHEDULE = "0 3 * * 0"
CRON_MARKER = "# WEATHER_ML_RETRAIN"


def get_python_path() -> str:
    """Get the path to the Python interpreter."""
    return sys.executable


def get_cron_command() -> str:
    """Build the cron command for retraining."""
    python = get_python_path()
    script = SCRIPTS_DIR / "train_ml_models.py"
    log_file = LOGS_DIR / "ml_retrain.log"

    # Ensure logs directory exists
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    return (
        f'{CRON_SCHEDULE} cd "{ROOT}" && '
        f'{python} "{script}" --all >> "{log_file}" 2>&1 {CRON_MARKER}'
    )


def get_current_crontab() -> str:
    """Get the current crontab contents."""
    try:
        result = subprocess.run(
            ["crontab", "-l"],
            capture_output=True,
            text=True,
        )
        return result.stdout if result.returncode == 0 else ""
    except Exception:
        return ""


def install_cron() -> bool:
    """Install the retraining cron job."""
    current = get_current_crontab()

    # Check if already installed
    if CRON_MARKER in current:
        print("Cron job already installed.")
        return True

    # Add our cron job
    new_cron = current.rstrip() + "\n" + get_cron_command() + "\n"

    try:
        process = subprocess.Popen(
            ["crontab", "-"],
            stdin=subprocess.PIPE,
            text=True,
        )
        process.communicate(input=new_cron)

        if process.returncode == 0:
            print("✓ Cron job installed successfully!")
            print(f"  Schedule: {CRON_SCHEDULE} (Sundays at 3:00 AM)")
            print(f"  Log file: {LOGS_DIR / 'ml_retrain.log'}")
            return True
        else:
            print("✗ Failed to install cron job")
            return False
    except Exception as e:
        print(f"✗ Error installing cron: {e}")
        return False


def uninstall_cron() -> bool:
    """Remove the retraining cron job."""
    current = get_current_crontab()

    if CRON_MARKER not in current:
        print("Cron job not found.")
        return True

    # Remove our cron job
    lines = current.split("\n")
    new_lines = [line for line in lines if CRON_MARKER not in line]
    new_cron = "\n".join(new_lines)

    try:
        process = subprocess.Popen(
            ["crontab", "-"],
            stdin=subprocess.PIPE,
            text=True,
        )
        process.communicate(input=new_cron)

        if process.returncode == 0:
            print("✓ Cron job removed successfully!")
            return True
        else:
            print("✗ Failed to remove cron job")
            return False
    except Exception as e:
        print(f"✗ Error removing cron: {e}")
        return False


def show_status():
    """Show current cron job status."""
    current = get_current_crontab()

    print("ML Retraining Scheduler Status")
    print("=" * 40)

    if CRON_MARKER in current:
        print("Status: INSTALLED ✓")
        print(f"Schedule: {CRON_SCHEDULE} (Sundays at 3:00 AM)")
        print(f"Log file: {LOGS_DIR / 'ml_retrain.log'}")

        # Check if log file exists and show last run
        log_file = LOGS_DIR / "ml_retrain.log"
        if log_file.exists():
            stat = log_file.stat()
            mtime = datetime.fromtimestamp(stat.st_mtime)
            print(f"Last log update: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print("Status: NOT INSTALLED")
        print("\nTo install: python weather/scripts/schedule_ml_retrain.py --install")


def run_now() -> int:
    """Run retraining immediately."""
    print("Running ML model retraining now...")
    print("=" * 40)

    script = SCRIPTS_DIR / "train_ml_models.py"
    result = subprocess.run(
        [get_python_path(), str(script), "--all"],
        cwd=ROOT,
    )

    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Schedule automatic ML model retraining."
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="Install the weekly retraining cron job",
    )
    parser.add_argument(
        "--uninstall",
        action="store_true",
        help="Remove the retraining cron job",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current scheduler status",
    )
    parser.add_argument(
        "--run-now",
        action="store_true",
        help="Run retraining immediately",
    )
    args = parser.parse_args()

    if args.install:
        return 0 if install_cron() else 1
    elif args.uninstall:
        return 0 if uninstall_cron() else 1
    elif args.run_now:
        return run_now()
    elif args.status:
        show_status()
        return 0
    else:
        show_status()
        print("\nUse --help for options.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
