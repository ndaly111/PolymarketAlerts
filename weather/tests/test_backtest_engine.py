from pathlib import Path

from weather.lib import db as db_lib
from weather.scripts.backtest_generate_fair_prices import (
    City,
    build_error_pmf_as_of,
    generate_fair_price_artifact,
)


def seed_snapshot(db_path: Path, *, city_key: str, date_local: str, forecast_high_f: int) -> None:
    db_lib.upsert_forecast_snapshot(
        db_path,
        city_key=city_key,
        target_date_local=date_local,
        snapshot_time_utc="2024-01-01T13:00:00+00:00",
        snapshot_hour_local=5,
        snapshot_tz="America/New_York",
        forecast_high_f=forecast_high_f,
        source="unit_test",
        points_url="https://example.com/points",
        forecast_url="https://example.com/forecast",
        qc_flags=[],
        raw={},
    )


def seed_observed(db_path: Path, *, city_key: str, date_local: str, tmax_f: int) -> None:
    db_lib.upsert_observed_cli(
        db_path,
        city_key=city_key,
        date_local=date_local,
        tmax_f=tmax_f,
        fetched_at_utc="2024-01-02T01:00:00+00:00",
        source_url="https://example.com/cli",
        version_used=1,
        report_date_local=date_local,
        is_preliminary=False,
        qc_flags=[],
        raw_text="",
    )


def test_no_lookahead(tmp_path: Path) -> None:
    db_path = tmp_path / "weather.db"
    seed_snapshot(db_path, city_key="LAX", date_local="2024-01-01", forecast_high_f=70)
    seed_snapshot(db_path, city_key="LAX", date_local="2024-01-02", forecast_high_f=70)
    seed_observed(db_path, city_key="LAX", date_local="2024-01-01", tmax_f=70)
    seed_observed(db_path, city_key="LAX", date_local="2024-01-02", tmax_f=80)

    result = build_error_pmf_as_of(
        db_path,
        city_key="LAX",
        month=1,
        snapshot_hour_local=5,
        cutoff_date_local="2024-01-02",
        min_samples=1,
        laplace_alpha=0.0,
    )

    assert result is not None
    n_samples, pmf = result
    assert n_samples == 1
    assert pmf == {0: 1.0}


def test_pmf_normalizes(tmp_path: Path) -> None:
    db_path = tmp_path / "weather.db"
    seed_snapshot(db_path, city_key="CHI", date_local="2024-01-01", forecast_high_f=50)
    seed_snapshot(db_path, city_key="CHI", date_local="2024-01-02", forecast_high_f=50)
    seed_snapshot(db_path, city_key="CHI", date_local="2024-01-03", forecast_high_f=50)
    seed_observed(db_path, city_key="CHI", date_local="2024-01-01", tmax_f=49)
    seed_observed(db_path, city_key="CHI", date_local="2024-01-02", tmax_f=51)

    result = build_error_pmf_as_of(
        db_path,
        city_key="CHI",
        month=1,
        snapshot_hour_local=5,
        cutoff_date_local="2024-01-03",
        min_samples=1,
        laplace_alpha=0.0,
    )

    assert result is not None
    _, pmf = result
    assert abs(sum(pmf.values()) - 1.0) < 1e-9


def test_sample_gating_skips_artifact(tmp_path: Path) -> None:
    db_path = tmp_path / "weather.db"
    seed_snapshot(db_path, city_key="NYC", date_local="2024-01-01", forecast_high_f=60)
    seed_snapshot(db_path, city_key="NYC", date_local="2024-01-02", forecast_high_f=60)
    seed_observed(db_path, city_key="NYC", date_local="2024-01-01", tmax_f=62)

    city = City(key="NYC", label="New York", tz="America/New_York")
    out_base = tmp_path / "outputs"

    wrote = generate_fair_price_artifact(
        db_path,
        city=city,
        target_date_local="2024-01-02",
        snapshot_hour_local=5,
        min_samples=2,
        laplace_alpha=0.0,
        overwrite=False,
        out_base=out_base,
    )

    assert not wrote
    assert not (out_base / "2024-01-02" / "NYC.json").exists()


def test_deterministic_replay(tmp_path: Path) -> None:
    db_path = tmp_path / "weather.db"
    seed_snapshot(db_path, city_key="MIA", date_local="2024-01-01", forecast_high_f=80)
    seed_snapshot(db_path, city_key="MIA", date_local="2024-01-02", forecast_high_f=80)
    seed_snapshot(db_path, city_key="MIA", date_local="2024-01-03", forecast_high_f=80)
    seed_observed(db_path, city_key="MIA", date_local="2024-01-01", tmax_f=79)
    seed_observed(db_path, city_key="MIA", date_local="2024-01-02", tmax_f=81)

    city = City(key="MIA", label="Miami", tz="America/New_York")
    out_base = tmp_path / "outputs"

    wrote_first = generate_fair_price_artifact(
        db_path,
        city=city,
        target_date_local="2024-01-03",
        snapshot_hour_local=5,
        min_samples=1,
        laplace_alpha=0.0,
        overwrite=False,
        out_base=out_base,
    )
    first_content = (out_base / "2024-01-03" / "MIA.json").read_text(encoding="utf-8")
    wrote_second = generate_fair_price_artifact(
        db_path,
        city=city,
        target_date_local="2024-01-03",
        snapshot_hour_local=5,
        min_samples=1,
        laplace_alpha=0.0,
        overwrite=False,
        out_base=out_base,
    )

    assert wrote_first
    assert not wrote_second

    second_content = (out_base / "2024-01-03" / "MIA.json").read_text(encoding="utf-8")
    assert first_content == second_content
