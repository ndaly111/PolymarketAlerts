#!/usr/bin/env python3
"""Backward-compatible entrypoint for collect_cli_observed."""

from weather.scripts.collect_cli_observed import main


if __name__ == "__main__":
    raise SystemExit(main())
