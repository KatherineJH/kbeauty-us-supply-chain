import pandas as pd
import pytest

from kbeauty.structure import bootstrap_median, hhi, median_gap


def test_hhi_bounds():
    assert hhi([1]) == pytest.approx(10_000)
    assert hhi([1, 1, 1, 1]) == pytest.approx(2_500)
    with pytest.raises(ValueError):
        hhi([0, 0])


def test_bootstrap_median_is_deterministic_with_a_seed():
    x = list(range(1, 50))
    assert bootstrap_median(x, seed=3) == bootstrap_median(x, seed=3)
    r = bootstrap_median(x, seed=3)
    assert r["ci_low"] <= r["median"] <= r["ci_high"]


def test_median_gap_sign():
    t = pd.DataFrame({"region": ["east"] * 20 + ["west"] * 20,
                      "departure_to_filing_days": [28] * 20 + [13] * 20})
    assert median_gap(t, "east", "west")["gap_days"] == pytest.approx(15)
