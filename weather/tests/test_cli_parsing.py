from pathlib import Path

from weather.lib.cli_parse import parse_report_date_local, parse_tmax_f


def test_cli_parses_report_date_and_tmax():
    txt = (Path(__file__).parent / "fixtures" / "sample_cli.txt").read_text(encoding="utf-8")
    d = parse_report_date_local(txt)
    assert d == "2026-01-21"
    tmax = parse_tmax_f(txt)
    assert tmax == 71
