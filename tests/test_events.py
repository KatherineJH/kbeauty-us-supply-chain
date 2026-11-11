import numpy as np
import pandas as pd
import pytest

from kbeauty.events import level_shift, placebo_rank, yoy_log_change


def monthly(values, start="2024-01-01"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="MS"))


def test_yoy_log_change_removes_a_fixed_seasonal_pattern():
    season = np.tile([1.0, 0.7, 1.1, 1.2, 1.3, 1.0, 1.0, 0.9, 1.5, 1.2, 1.0, 1.1], 3)
    s = monthly(100 * season * 1.05 ** (np.arange(36) / 12))  # 5% a year growth on top of seasonality
    y = yoy_log_change(s).dropna()
    assert np.allclose(y, np.log(1.05))


def test_level_shift_recovers_a_known_shift():
    rng = np.random.default_rng(1)
    y = monthly(np.r_[np.full(10, 0.10), np.full(12, 0.40)] + rng.normal(0, 0.01, 22), start="2025-01-01")
    r = level_shift(y, "2025-11-01")
    assert r["shift"] == pytest.approx(0.30, abs=0.02)
    assert r["ci_low"] > 0 and r["n_pre"] == 10 and r["n_post"] == 12


def test_level_shift_requires_observations_on_both_sides():
    y = monthly(np.ones(10), start="2025-01-01")
    with pytest.raises(ValueError):
        level_shift(y, "2025-10-01")


def test_placebo_ranks_the_true_break_first():
    rng = np.random.default_rng(2)
    y = monthly(np.r_[np.zeros(10), np.ones(10)] + rng.normal(0, 0.05, 20), start="2025-01-01")
    r = placebo_rank(y, "2025-11-01")
    assert r["largest_break"] == pd.Timestamp("2025-11-01")
    assert r["placebo_p"] == pytest.approx(1 / r["n_candidates"])
