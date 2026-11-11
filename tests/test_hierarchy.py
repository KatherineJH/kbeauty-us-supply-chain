import numpy as np
import pandas as pd
import pytest

from kbeauty.hierarchy import (all_nodes, conformal_coverage, diebold_mariano, nowcast_backtest, reconcile,
                               summing_matrix)


def bottom_frame(n=30, seed=0):
    rng = np.random.default_rng(seed)
    S, cols = summing_matrix()
    idx = pd.date_range("2024-01-01", periods=n, freq="MS")
    return pd.DataFrame(np.exp(rng.normal(10, 0.3, (n, len(cols)))), index=idx, columns=cols)


def test_summing_matrix_shape_and_totals():
    S, cols = summing_matrix()
    assert S.shape == (15, 8)
    nodes = all_nodes(bottom_frame())
    assert np.allclose(nodes["total"], nodes[cols].sum(axis=1))
    assert np.allclose(nodes["all|vessel"] + nodes["all|air"], nodes["total"])


@pytest.mark.parametrize("method", ["bottom_up", "ols"])
def test_reconciled_forecasts_are_coherent(method):
    S, cols = summing_matrix()
    rng = np.random.default_rng(1)
    base = pd.Series(rng.uniform(1, 10, len(S)), index=S.index)  # incoherent on purpose
    r = reconcile(base, method)
    assert r["total"] == pytest.approx(r[cols].sum())
    for g in ("skincare", "makeup", "hair", "other"):
        assert r[f"{g}|all"] == pytest.approx(r[f"{g}|vessel"] + r[f"{g}|air"])


def test_ols_leaves_coherent_forecasts_unchanged():
    nodes = all_nodes(bottom_frame())
    base = nodes.iloc[-1]
    assert np.allclose(reconcile(base, "ols"), base)


def test_conformal_coverage_is_close_to_nominal_for_iid_errors():
    rng = np.random.default_rng(2)
    rows = []
    for o in range(40):
        for node in range(15):
            a = 100.0
            rows.append({"origin": o, "horizon": 1, "node": node, "actual": a, "forecast": a * np.exp(rng.normal(0, 0.1))})
    cov = conformal_coverage(pd.DataFrame(rows)).set_index("level")["coverage"]
    assert cov[0.8] == pytest.approx(0.8, abs=0.05) and cov[0.95] == pytest.approx(0.95, abs=0.04)


def test_conformal_counts_non_positive_forecasts_as_misses():
    rows = [{"origin": o, "horizon": 1, "node": n, "actual": 100.0, "forecast": 100.0}
            for o in range(10) for n in range(5)]
    rows[0]["forecast"] = -5.0  # e.g. an OLS-reconciled small node
    cov = conformal_coverage(pd.DataFrame(rows), levels=(0.8,)).iloc[0]
    assert cov["n_invalid"] == 1
    assert cov["coverage"] == pytest.approx(49 / 50)
    assert np.isfinite(cov["mean_width_log"])


def test_nowcast_is_exact_when_the_us_kr_ratio_is_constant():
    idx = pd.date_range("2024-01-01", periods=20, freq="MS")
    kr = pd.Series(np.linspace(100, 200, 20), index=idx)
    us = 0.8 * kr
    bt = nowcast_backtest(us, kr, first_origin=12)
    now = bt[bt.model == "nowcast_kr_ratio"]
    assert np.allclose(now.forecast, now.actual)


def test_diebold_mariano_sign():
    rng = np.random.default_rng(3)
    small, large = rng.normal(0, 1, 60), rng.normal(0, 3, 60)
    r = diebold_mariano(small, large)
    assert r["dm_stat"] < 0 and r["p_value"] < 0.01


def test_diebold_mariano_falls_back_when_long_run_variance_is_negative():
    e1 = np.array([1.0, -3.0, 1.0, -3.0, 1.0, -3.0])  # strongly negatively autocorrelated loss differential
    e2 = np.zeros(6)
    r = diebold_mariano(e1, e2, h=2)
    assert r["lag0_variance"] and np.isfinite(r["p_value"])
