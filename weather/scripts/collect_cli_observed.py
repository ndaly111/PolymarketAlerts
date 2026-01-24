#!/usr/bin/env python3
"""
Collect observed highs from NWS CLI report text.

This collector:
  - Uses city-local "yesterday" as the target date
  - Fetches NWS CLI via forecast.weather.gov product viewer (product.php)
  - Walks versions to find the report whose CLIMATE SUMMARY FOR date matches target date
  - Upserts observed_cli rows into SQLite
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Tuple
from zoneinfo import ZoneInfo

import requests
import yaml

from weather.lib import db as db_lib
from weather.lib.cli_parse import detect_preliminary, parse_report_date_local, parse_tmax_f


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "weather" / "config" / "cities.yml"
DEFAULT_DB = Path(os.getenv("WEATHER_DB_PATH", str(ROOT / "weather" / "data" / "weather_forecast_accuracy.db")))


def build_cli_url(*, issuedby: str, site: str, product: str, version: int) -> str:
    return (
        "https://forecast.weather.gov/product.php"
        f"?issuedby={issuedby}&product={product}&site={site}&format=txt&glossary=0&version={version}"
    )


@dataclass(frozen=True)
class City:
    key: str
    label: str
    tz: str
    cli_issuedby: str
    cli_site: str
    cli_product: str


def _now_utc() -> datetime:
    return datetime.now(tz=ZoneInfo("UTC"))


def _yesterday_local(tz_name: str) -> date:
    now_local = _now_utc().astimezone(ZoneInfo(tz_name))
    return now_local.date() - timedelta(days=1)


def load_cities(config_path: Path) -> List[City]:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cities: List[City] = []
    for row in data.get("cities", []):
        cli = row.get("cli") or {}
        cities.append(
            City(
                key=str(row["key"]).strip(),
                label=str(row.get("label", row["key"])).strip(),
                tz=str(row.get("tz", "America/New_York")).strip(),
                cli_issuedby=str(cli.get("issuedby", "")).strip(),
                cli_site=str(cli.get("site", "")).strip(),
                cli_product=str(cli.get("product", "CLI")).strip(),
            )
        )
    if not cities:
        raise ValueError(f"No cities found in {config_path}")
    for c in cities:
        if not (c.cli_issuedby and c.cli_site and c.cli_product):
            raise ValueError(f"Missing cli fields for city {c.key} in {config_path}")
    return cities


def fetch_text(session: requests.Session, url: str) -> str:
    resp = session.get(url, timeout=25)
    resp.raise_for_status()
    text = resp.text
    if "<pre" in text.lower():
        match = re.search(r"<pre[^>]*>(.*?)</pre>", text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            body = match.group(1)
            body = body.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
            return body.replace("\r\n", "\n").strip()
    return text.replace("\r\n", "\n").strip()


def find_matching_version(
    session: requests.Session,
    *,
    issuedby: str,
    site: str,
    product: str,
    target_date_local: str,
    max_versions: int = 25,
) -> Tuple[int, str, str]:
    candidates = list(range(1, min(5, max_versions) + 1))
    if max_versions > 5:
        candidates.extend(range(max_versions, 5, -1))

    last_err = None
    for version in candidates:
        url = build_cli_url(issuedby=issuedby, site=site, product=product, version=version)
        try:
            text = fetch_text(session, url)
            report_date = parse_report_date_local(text)
            if report_date == target_date_local:
                return version, url, text
        except Exception as exc:
            last_err = exc
        sleep_seconds = float(os.getenv("WEATHER_NWS_SLEEP_SECONDS", "0.15"))
        if sleep_seconds > 0:
            import time

            time.sleep(sleep_seconds)

    raise RuntimeError(
        f"Could not find CLI report for date={target_date_local} issuedby={issuedby} site={site} "
        f"in versions 1..{max_versions}. Last error: {last_err}"
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--db", default=str(DEFAULT_DB))
    p.add_argument("--max-versions", type=int, default=25)
    args = p.parse_args()

    db_path = Path(args.db)
    db_lib.ensure_schema(db_path)

    cities = load_cities(Path(args.config))
    session = requests.Session()
    fetched_at_utc = _now_utc().isoformat()

    wrote = 0
    for city in cities:
        target_date_local = _yesterday_local(city.tz).isoformat()
        qc_flags: List[str] = []
        try:
            version_used, url, text = find_matching_version(
                session,
                issuedby=city.cli_issuedby,
                site=city.cli_site,
                product=city.cli_product,
                target_date_local=target_date_local,
                max_versions=int(args.max_versions),
            )

            report_date_local = parse_report_date_local(text)
            if report_date_local != target_date_local:
                qc_flags.append("REPORT_DATE_MISMATCH")
            tmax_f = parse_tmax_f(text)
            is_prelim = detect_preliminary(text)

            db_lib.upsert_observed_cli(
                db_path,
                city_key=city.key,
                date_local=target_date_local,
                tmax_f=tmax_f,
                fetched_at_utc=fetched_at_utc,
                source_url=url,
                version_used=int(version_used),
                report_date_local=report_date_local,
                is_preliminary=is_prelim,
                qc_flags=qc_flags,
                raw_text=text,
            )
            wrote += 1
            print(f"[ok] {city.key} {city.label}: {target_date_local} tmax={tmax_f}F")
        except Exception as exc:
            print(f"[err] {city.key} {city.label}: {exc}", file=sys.stderr)

    if wrote == 0:
        print("[done] No observed CLI rows written.")
        return 0
    print(f"[done] wrote {wrote}/{len(cities)} observed CLI rows into {db_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
