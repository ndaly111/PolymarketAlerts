from weather.lib.fair import normalize_pmf, shift_pmf, summarize_pmf


def test_shift_and_normalize():
    pmf = {-1: 0.2, 0: 0.6, 1: 0.2}
    shifted = shift_pmf(pmf, 70)
    assert 69 in shifted and 70 in shifted and 71 in shifted
    n = normalize_pmf(shifted)
    assert abs(sum(n.values()) - 1.0) < 1e-12


def test_summarize_pmf_quantiles():
    pmf = {70: 0.1, 71: 0.8, 72: 0.1}
    s = summarize_pmf(pmf)
    assert s.p50 == 71
