import numpy as np
import pandas as pd
import pytest

from kbeauty.comparison import did, event_path, mde, placebo, synthetic_control


def panel(effect=0.0, n_months=20, countries=("Korea", "A", "B", "C", "D"), seed=0, noise=0.01):
    """YoY panel with month and country effects; Korea gets `effect` from 2025-09."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n_months, freq="MS")
    month_fx = rng.normal(0, 0.1, n_months)[:, None]
    country_fx = rng.normal(0, 0.1, len(countries))[None, :]
    y = month_fx + country_fx + rng.normal(0, noise, (n_months, len(countries)))
    y = pd.DataFrame(y, index=idx, columns=list(countries))
    y.loc[idx >= "2025-09-01", "Korea"] += effect
    return y


def test_did_recovers_a_known_effect():
    assert did(panel(effect=0.30), "Korea", "2025-09-01") == pytest.approx(0.30, abs=0.02)
    assert did(panel(effect=0.0), "Korea", "2025-09-01") == pytest.approx(0.0, abs=0.02)


def test_did_with_fx_control_absorbs_fx_driven_movement():
    y = panel(effect=0.0)
    rng = np.random.default_rng(7)
    # FX varies month to month for every country (so it is not collinear with Korea*post), and Korea's
    # currency also shifts after the break. Imports respond to FX in every country with the same elasticity.
    fx = pd.DataFrame(rng.normal(0, 0.05, y.shape), index=y.index, columns=y.columns)
    fx.loc[fx.index >= "2025-09-01", "Korea"] -= 0.2
    y = y + 0.5 * fx
    assert abs(did(y, "Korea", "2025-09-01")) > 0.08
    assert did(y, "Korea", "2025-09-01", fx) == pytest.approx(0.0, abs=0.02)


def test_event_path_is_zero_on_average_before_the_break():
    p = event_path(panel(effect=0.3), "Korea", "2025-09-01")
    assert p[p.index < "2025-09-01"].mean() == pytest.approx(0.0, abs=1e-12)
    assert p[p.index >= "2025-09-01"].mean() == pytest.approx(0.3, abs=0.03)


def test_placebo_ranks_a_strong_real_effect_first():
    r = placebo(panel(effect=0.5), "Korea", "2025-09-01")
    assert r["p_space"] == pytest.approx(1 / r["n_space"])
    assert r["p_space_time"] == pytest.approx(1 / r["n_space_time"])


def test_synthetic_control_recovers_convex_weights():
    rng = np.random.default_rng(3)
    idx = pd.date_range("2024-01-01", periods=32, freq="MS")
    donors = pd.DataFrame(np.exp(rng.normal(0, 0.2, (32, 3)).cumsum(axis=0)), index=idx, columns=["A", "B", "C"])
    korea = np.exp(0.6 * np.log(donors["A"]) + 0.4 * np.log(donors["B"]))
    sc = synthetic_control(donors.assign(Korea=korea), "Korea", "2025-09-01")
    assert sc["weights"]["A"] == pytest.approx(0.6, abs=0.02)
    assert sc["weights"]["B"] == pytest.approx(0.4, abs=0.02)
    assert sc["rmspe_pre"] < 1e-3


def test_pretrend_passes_when_korea_moves_with_controls_and_fails_on_a_divergent_trend():
    from kbeauty.comparison import pretrend
    parallel = pretrend(panel(effect=0.3, noise=0.05), "Korea", "2025-09-01")
    assert parallel["leads_p"] > 0.05 and parallel["drift_p"] > 0.05
    y = panel(effect=0.0, noise=0.01)
    y["Korea"] += np.linspace(0, 0.5, len(y))  # Korea drifts away from the controls before the break
    diverging = pretrend(y, "Korea", "2025-09-01")
    assert diverging["drift_p"] < 0.01 and diverging["drift_per_month"] > 0


def test_mde_is_2_8_standard_deviations():
    betas = np.array([-0.1, 0.0, 0.1, 0.2, -0.2])
    assert mde(betas) == pytest.approx(2.8016 * betas.std(ddof=1), rel=1e-3)
