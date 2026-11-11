import numpy as np
import pandas as pd
import pytest

from kbeauty.forecast import MODELS, backtest, score, seasonal_naive, seasonal_naive_growth


def monthly(values):
    return pd.Series(values, index=pd.date_range("2024-01-01", periods=len(values), freq="MS"), dtype=float)


@pytest.fixture
def series():
    rng = np.random.default_rng(0)
    season = np.tile([1.0, 0.7, 1.1, 1.2, 1.3, 1.0, 1.0, 0.9, 1.5, 1.2, 1.0, 1.1], 3)[:32]
    return monthly(150 * season * np.exp(rng.normal(0, 0.05, 32)))


def test_backtest_never_reads_the_future(series):
    """Changing values after an origin must not change any forecast made at that origin."""
    origin = 25
    bt = backtest(series, MODELS, first_origin=origin)
    tampered = series.copy()
    tampered.iloc[origin + 1:] *= 10
    bt2 = backtest(tampered, MODELS, first_origin=origin)
    first = lambda d: d[d.origin == series.index[origin]].set_index(["model", "horizon"]).forecast
    pd.testing.assert_series_equal(first(bt), first(bt2))


def test_seasonal_naive_returns_same_month_last_year(series):
    history = series.iloc[:24]
    assert np.allclose(seasonal_naive(history, 3), series.iloc[12:15])


def test_seasonal_naive_growth_scales_by_trailing_yoy(series):
    history = series.iloc[:24]
    growth = history.iloc[12:].sum() / history.iloc[:12].sum()
    assert np.allclose(seasonal_naive_growth(history, 2), series.iloc[12:14] * growth)


def test_backtest_only_scores_targets_inside_the_data(series):
    bt = backtest(series, {"sn": seasonal_naive}, first_origin=28, horizons=(1, 2, 3))
    assert bt.target.max() == series.index[-1]
    assert set(bt.groupby("horizon").size()) == {3, 2, 1}


def test_score_computes_wape_and_relative_wape():
    bt = pd.DataFrame({"model": ["a", "a", "ref", "ref"], "horizon": [1, 1, 1, 1],
                       "actual": [100.0, 100.0, 100.0, 100.0], "forecast": [90.0, 110.0, 80.0, 120.0]})
    sc = score(bt, reference="ref").set_index("model")
    assert sc.loc["a", "wape"] == pytest.approx(0.10)
    assert sc.loc["ref", "wape"] == pytest.approx(0.20)
    assert sc.loc["a", "relative_wape"] == pytest.approx(0.5)
