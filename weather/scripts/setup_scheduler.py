#!/usr/bin/env python3
"""Set up scheduled data collection for weather trading.

This script creates launchd plist files (macOS) or cron entries (Linux)
for automated data collection.

Usage:
    # Show what would be installed
    python setup_scheduler.py --dry-run

    # Install the schedules
    python setup_scheduler.py --install

    # Remove schedules
    python setup_scheduler.py --uninstall

    # Check status
    python setup_scheduler.py --status
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "weather" / "scripts"
PLIST_DIR = Path.home() / "Library" / "LaunchAgents"
PLIST_PREFIX = "com.weather-trading"

# Define the schedules
SCHEDULES = [
    {
        "name": "morning_scan",
        "description": "Morning trading scan (6 AM ET)",
        "script": "run_collection.py",
        "args": ["--profile", "morning_scan"],
        "hour": 6,
        "minute": 0,
    },
    {
        "name": "hourly_forecasts",
        "description": "Hourly forecast updates (every hour 7 AM - 8 PM ET)",
        "script": "run_collection.py",
        "args": ["--profile", "hourly"],
        "hour": "*",  # Every hour
        "minute": 15,
        "start_hour": 7,
        "end_hour": 20,
    },
    {
        "name": "frequent_kalshi",
        "description": "Frequent Kalshi updates (every 30 min during trading)",
        "script": "run_collection.py",
        "args": ["--collect", "kalshi,ml_edges"],
        "hour": "*",
        "minute": "0,30",
        "start_hour": 8,
        "end_hour": 18,
    },
]


def create_plist(schedule: dict, python_path: str) -> str:
    """Create a launchd plist for a schedule."""
    name = schedule["name"]
    script = SCRIPTS_DIR / schedule["script"]
    args = schedule.get("args", [])

    label = f"{PLIST_PREFIX}.{name}"

    # Build the command
    cmd_args = [python_path, str(script)] + args

    # Build calendar interval
    hour = schedule.get("hour", "*")
    minute = schedule.get("minute", 0)

    if hour == "*":
        # Hourly - use StartInterval
        interval_xml = f"""
    <key>StartCalendarInterval</key>
    <array>"""
        start_h = schedule.get("start_hour", 0)
        end_h = schedule.get("end_hour", 23)
        mins = str(minute).split(",")
        for h in range(start_h, end_h + 1):
            for m in mins:
                interval_xml += f"""
        <dict>
            <key>Hour</key>
            <integer>{h}</integer>
            <key>Minute</key>
            <integer>{int(m)}</integer>
        </dict>"""
        interval_xml += """
    </array>"""
    else:
        interval_xml = f"""
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>{hour}</integer>
        <key>Minute</key>
        <integer>{minute}</integer>
    </dict>"""

    plist = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{label}</string>
    <key>ProgramArguments</key>
    <array>
        {"".join(f'<string>{arg}</string>' for arg in cmd_args)}
    </array>
    <key>WorkingDirectory</key>
    <string>{ROOT}</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PYTHONPATH</key>
        <string>{ROOT}</string>
    </dict>
    {interval_xml}
    <key>StandardOutPath</key>
    <string>{ROOT}/weather/outputs/logs/{name}.out.log</string>
    <key>StandardErrorPath</key>
    <string>{ROOT}/weather/outputs/logs/{name}.err.log</string>
    <key>RunAtLoad</key>
    <false/>
</dict>
</plist>
"""
    return plist


def install_schedules(dry_run: bool = False):
    """Install launchd schedules."""
    python_path = sys.executable

    print("Setting up weather trading schedules...")
    print(f"  Python: {python_path}")
    print(f"  Scripts: {SCRIPTS_DIR}")
    print()

    # Ensure plist directory exists
    if not dry_run:
        PLIST_DIR.mkdir(parents=True, exist_ok=True)
        (ROOT / "weather" / "outputs" / "logs").mkdir(parents=True, exist_ok=True)

    for schedule in SCHEDULES:
        name = schedule["name"]
        label = f"{PLIST_PREFIX}.{name}"
        plist_path = PLIST_DIR / f"{label}.plist"

        print(f"[{name}] {schedule['description']}")

        plist_content = create_plist(schedule, python_path)

        if dry_run:
            print(f"  Would write: {plist_path}")
            print(f"  Would run: launchctl load {plist_path}")
        else:
            # Unload if already loaded
            subprocess.run(
                ["launchctl", "unload", str(plist_path)],
                capture_output=True,
            )

            # Write plist
            plist_path.write_text(plist_content)
            print(f"  Wrote: {plist_path}")

            # Load
            result = subprocess.run(
                ["launchctl", "load", str(plist_path)],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                print(f"  Loaded successfully")
            else:
                print(f"  Load error: {result.stderr}")

        print()

    if not dry_run:
        print("Schedules installed. Use --status to check.")


def uninstall_schedules():
    """Remove launchd schedules."""
    print("Removing weather trading schedules...")

    for schedule in SCHEDULES:
        name = schedule["name"]
        label = f"{PLIST_PREFIX}.{name}"
        plist_path = PLIST_DIR / f"{label}.plist"

        print(f"[{name}]")

        if plist_path.exists():
            # Unload
            subprocess.run(
                ["launchctl", "unload", str(plist_path)],
                capture_output=True,
            )
            # Remove file
            plist_path.unlink()
            print(f"  Removed: {plist_path}")
        else:
            print(f"  Not installed")

        print()

    print("Schedules removed.")


def show_status():
    """Show status of installed schedules."""
    print("Weather Trading Schedule Status")
    print("=" * 50)

    for schedule in SCHEDULES:
        name = schedule["name"]
        label = f"{PLIST_PREFIX}.{name}"
        plist_path = PLIST_DIR / f"{label}.plist"

        print(f"\n[{name}] {schedule['description']}")

        if not plist_path.exists():
            print("  Status: NOT INSTALLED")
            continue

        # Check if loaded
        result = subprocess.run(
            ["launchctl", "list", label],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            if len(lines) >= 2:
                parts = lines[1].split("\t")
                pid = parts[0] if parts[0] != "-" else "not running"
                status = parts[1] if len(parts) > 1 else "?"
                print(f"  Status: LOADED (PID: {pid}, Exit: {status})")
            else:
                print(f"  Status: LOADED")
        else:
            print(f"  Status: NOT LOADED (plist exists)")

        # Check logs
        out_log = ROOT / "weather" / "outputs" / "logs" / f"{name}.out.log"
        err_log = ROOT / "weather" / "outputs" / "logs" / f"{name}.err.log"

        if out_log.exists():
            size = out_log.stat().st_size
            print(f"  Output log: {size} bytes")
        if err_log.exists():
            size = err_log.stat().st_size
            if size > 0:
                print(f"  Error log: {size} bytes (!)")


def main():
    parser = argparse.ArgumentParser(
        description="Set up scheduled data collection for weather trading"
    )
    parser.add_argument("--install", action="store_true", help="Install schedules")
    parser.add_argument("--uninstall", action="store_true", help="Remove schedules")
    parser.add_argument("--status", action="store_true", help="Show status")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    args = parser.parse_args()

    if args.uninstall:
        uninstall_schedules()
    elif args.install or args.dry_run:
        install_schedules(dry_run=args.dry_run)
    elif args.status:
        show_status()
    else:
        parser.print_help()
        print("\n\nQuick start:")
        print("  python setup_scheduler.py --dry-run   # Preview")
        print("  python setup_scheduler.py --install   # Install")
        print("  python setup_scheduler.py --status    # Check status")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
