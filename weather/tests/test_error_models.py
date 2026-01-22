from weather.scripts.build_error_models import build_pmf_from_counts


def test_build_pmf_sums_to_one():
    pmf = build_pmf_from_counts({-1: 2, 0: 6, 1: 2})
    s = sum(pmf.values())
    assert abs(s - 1.0) < 1e-9


def test_build_pmf_laplace_smoothing():
    pmf = build_pmf_from_counts({0: 1, 1: 1}, laplace_alpha=0.5)
    s = sum(pmf.values())
    assert abs(s - 1.0) < 1e-9
