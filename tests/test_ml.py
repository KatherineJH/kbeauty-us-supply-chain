import numpy as np
import pandas as pd
import pytest

from kbeauty import ml
from kbeauty.hierarchy import summing_matrix

MODELS = ["ridge", "lightgbm"]  # one linear and one tree model keep the tests fast


def data(n=30, seed=0):
    rng = np.random.default_rng(seed)
    _, cols = summing_matrix()
    idx = pd.date_range("2024-01-01", periods=n, freq="MS")
    season = np.sin(2 * np.pi * idx.month.to_numpy() / 12)[:, None]
    bottom = pd.DataFrame(np.exp(10 + 0.2 * season + rng.normal(0, 0.1, (n, len(cols)))), index=idx, columns=cols)
    fx = pd.Series(1400 + rng.normal(0, 20, n), index=idx)
    return bottom, fx


def run(bottom, fx, first_origin=26, **kw):
    return ml.backtest(bottom, fx, first_origin, "2025-09-01", horizons=(1, 3), models=MODELS,
                       importance_repeats=0, n_jobs=1, **kw)


def test_forecasts_ignore_everything_after_the_origin():
    bottom, fx = data()
    origin = bottom.index[26]
    later = bottom.copy()
    later.loc[later.index > origin] *= 3.0  # future months change; the origin's forecasts must not
    a, b = run(bottom, fx), run(later, fx)
    key = ["horizon", "model", "method", "node"]
    fa = a.forecasts[a.forecasts.origin == origin].sort_values(key).forecast.to_numpy()
    fb = b.forecasts[b.forecasts.origin == origin].sort_values(key).forecast.to_numpy()
    assert np.allclose(fa, fb, rtol=0, atol=0)


def test_selected_model_is_the_best_inner_score_and_reuses_its_forecast():
    bottom, fx = data()
    res = run(bottom, fx)
    sel = res.selection
    for (o, h), g in sel.groupby(["origin", "horizon"]):
        per_model = g[g.model != "ml_selected"]
        chosen = g[g.model == "ml_selected"].iloc[0]
        assert chosen.inner_wape == pytest.approx(per_model.inner_wape.min())
        f = res.forecasts.query("origin == @o and horizon == @h and method == 'base'")
        own = f[f.model == f"ml_{chosen.chosen_model}"].sort_values("node").forecast.to_numpy()
        assert np.allclose(f[f.model == "ml_selected"].sort_values("node").forecast.to_numpy(), own)


def test_bottom_up_rows_are_coherent():
    bottom, fx = data()
    f = run(bottom, fx).forecasts.query("method == 'bottom_up'")
    _, cols = summing_matrix()
    for _, g in f.groupby(["origin", "horizon", "model"]):
        g = g.set_index("node").forecast
        assert g["total"] == pytest.approx(g[cols].sum())
        assert g["all|vessel"] + g["all|air"] == pytest.approx(g["total"])


def test_diff_target_recovers_a_pure_drift_exactly():
    """log value = a + 0.01 t per series: a linear model on the log change forecasts it without error."""
    _, cols = summing_matrix()
    idx = pd.date_range("2024-01-01", periods=30, freq="MS")
    bottom = pd.DataFrame(np.exp(10 + np.arange(len(cols))[None, :] * 0.1 + 0.01 * np.arange(30)[:, None]),
                          index=idx, columns=cols)
    cfg = ml.Config("ridge", (("alpha", 1e-8),), "diff", "short")
    d = ml.feature_frame(bottom, pd.Series(1400.0, index=idx), 1, "2025-09-01")
    pred = ml._fit_predict(cfg, d[d.index <= idx[25]], d[d.index == idx[26]])
    assert np.allclose(np.exp(pred), bottom.loc[idx[26]].to_numpy(), rtol=1e-6)


def test_parallel_and_serial_runs_are_identical():
    bottom, fx = data(seed=3)
    a = run(bottom, fx).forecasts
    b = ml.backtest(bottom, fx, 26, "2025-09-01", horizons=(1, 3), models=MODELS, importance_repeats=0,
                    n_jobs=2).forecasts
    pd.testing.assert_frame_equal(a, b)


def test_forecast_from_last_matches_the_backtest_at_that_origin():
    """Freezing at the last month uses exactly the rule the backtest used at every origin."""
    bottom, fx = data(seed=5)
    o = 26  # last origin with a 3-month-ahead target in 30 months
    bt = run(bottom, fx, first_origin=o).forecasts
    bt = bt[(bt.origin == bottom.index[o]) & (bt.method == "base")]
    fr = ml.forecast_from_last(bottom.iloc[: o + 1], fx, "2025-09-01", horizons=(1, 3), models=MODELS, n_jobs=1)
    merged = bt.assign(target_month=bt.target.dt.strftime("%Y-%m")).merge(
        fr, on=["target_month", "horizon", "model", "node"], suffixes=("_bt", "_fr"))
    assert len(merged) == len(fr) == 2 * 3 * 8
    assert np.allclose(merged.forecast_bt, merged.forecast_fr, rtol=1e-12)
