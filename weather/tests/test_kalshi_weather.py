from __future__ import annotations

from weather.lib.kalshi_weather import parse_event_spec_from_title, prob_event


def test_parse_between() -> None:
    spec = parse_event_spec_from_title("Will the high temperature be between 74° and 75°?")
    assert spec is not None
    assert spec.kind == "between"
    assert spec.a == 74 and spec.b == 75


def test_parse_ge() -> None:
    spec = parse_event_spec_from_title("Will the high temperature be 80° or higher?")
    assert spec is not None
    assert spec.kind == "ge"
    assert spec.a == 80


def test_parse_lt() -> None:
    spec = parse_event_spec_from_title("Will the high temperature be below 20°?")
    assert spec is not None
    assert spec.kind == "lt"
    assert spec.a == 20


def test_prob_event_between() -> None:
    pmf = {70: 0.1, 71: 0.2, 72: 0.3, 73: 0.4}
    spec = parse_event_spec_from_title("between 71 and 72")
    assert spec is not None
    q = prob_event(pmf, spec)
    assert abs(q - 0.5) < 1e-9
