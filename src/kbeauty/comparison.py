"""Was the post-August-2025 change specific to Korea? Cross-country comparison (pre-registration Q1, A1/A4).

All estimates use year-over-year log changes, y_ct = log x_ct - log x_c,t-12, which remove each country's own
seasonality. The treated unit is Korea; controls are the pre-registered peer countries.

- `did`: pooled difference-in-differences on the YoY panel, y_ct = a_c + g_t + b * Korea*post + d * dlog FX_ct.
- `event_path`: Korea minus the control mean each month, net of its pre-period average.
- `placebo`: re-estimate b for every (country, break month) pair, treating each control as if it were Korea and
  each admissible month as if it were the break. With one treated country, cluster-robust standard errors are
  unreliable, so inference is by this randomisation distribution: p = share of placebo |b| >= real |b|.
- `synthetic_control`: a convex combination of controls that tracks Korea's demeaned log level before the break.
- `mde`: smallest effect detectable with 80% power at 5% two-sided, from the spread of the placebo estimates.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize
from statsmodels.stats.multitest import multipletests

TREATED = "Korea"
LAND_BORDER = ("Canada", "Mexico")  # ~1% of value by sea or air: excluded for sea/air metrics (pre-registered)

# metric -> (panel column, use land-border peers?)
METRICS = {
    "value_usd": ("value_usd", True),
    "measured_weight_kg": ("measured_weight_kg", False),
    "air_weight_kg": ("air_weight_kg", False),
    "air_usd_per_kg": ("air_usd_per_kg", False),
    "vessel_usd_per_kg": ("vessel_usd_per_kg", False),
}
METRIC_KO = {"value_usd": "수입 금액", "measured_weight_kg": "수입 중량(해상+항공)", "air_weight_kg": "항공 중량",
             "air_usd_per_kg": "항공 kg당 단가", "vessel_usd_per_kg": "해상 kg당 단가"}


def wide(panel: pd.DataFrame, column: str, countries) -> pd.DataFrame:
    w = panel.pivot(index="month", columns="country", values=column).sort_index()
    return w[list(countries)].astype(float)


def yoy(levels: pd.DataFrame) -> pd.DataFrame:
    return np.log(levels).diff(12).dropna(how="all")


def countries_for(metric: str, panel: pd.DataFrame) -> list[str]:
    allowed = METRICS[metric][1]
    peers = sorted(c for c in panel["country"].unique() if c != TREATED and (allowed or c not in LAND_BORDER))
    return [TREATED, *peers]


def did(y: pd.DataFrame, treated: str, break_month, fx: pd.DataFrame | None = None) -> float:
    """Pooled DiD coefficient on treated*post. y and fx: month x country YoY log changes."""
    long = y.stack().rename("y").reset_index()
    long.columns = ["month", "country", "y"]
    X = pd.get_dummies(long[["month", "country"]].astype(str), drop_first=True, dtype=float)
    X["treat_post"] = ((long["country"] == treated) & (long["month"] >= pd.Timestamp(break_month))).astype(float)
    if fx is not None:
        X["dlog_fx"] = fx.stack().reindex(pd.MultiIndex.from_frame(long[["month", "country"]])).to_numpy()
    fit = sm.OLS(long["y"].to_numpy(), sm.add_constant(X).to_numpy()).fit()
    return float(fit.params[list(X.columns).index("treat_post") + 1])


def event_path(y: pd.DataFrame, treated: str, break_month) -> pd.Series:
    controls = [c for c in y.columns if c != treated]
    gap = y[treated] - y[controls].mean(axis=1)
    return gap - gap[gap.index < pd.Timestamp(break_month)].mean()


def pretrend(y: pd.DataFrame, treated: str, break_month) -> dict:
    """Parallel-trends diagnostics on the pre-break months only.

    - leads F-test: in a two-way fixed-effects model on pre-break data, Korea x month dummies (one month omitted)
      are jointly zero if Korea moved with the controls before the policy.
    - drift: slope of (Korea - control mean) on a time index before the break.
    Few pre-break months make both tests weak; a non-rejection is weak evidence, not proof of parallel trends.
    """
    pre = y[y.index < pd.Timestamp(break_month)]
    long = pre.stack().rename("y").reset_index()
    long.columns = ["month", "country", "y"]
    X = pd.get_dummies(long[["month", "country"]].astype(str), drop_first=True, dtype=float)
    months = sorted(pre.index)[1:]  # first pre month is the reference
    leads = []
    for m in months:
        name = f"lead_{m:%Y%m}"
        X[name] = ((long["country"] == treated) & (long["month"] == m)).astype(float)
        leads.append(name)
    X = sm.add_constant(X)
    fit = sm.OLS(long["y"].to_numpy(), X.to_numpy()).fit()
    idx = [list(X.columns).index(n) for n in leads]
    R = np.zeros((len(idx), X.shape[1]))
    R[np.arange(len(idx)), idx] = 1
    ftest = fit.f_test(R)
    gap = pre[treated] - pre.drop(columns=treated).mean(axis=1)
    t = np.arange(len(gap), dtype=float)
    drift = sm.OLS(gap.to_numpy(), sm.add_constant(t)).fit(cov_type="HAC", cov_kwds={"maxlags": 2})
    return {"n_pre_months": len(pre), "leads_F": float(ftest.fvalue), "leads_p": float(ftest.pvalue),
            "drift_per_month": float(drift.params[1]), "drift_p": float(drift.pvalues[1])}


def placebo(y: pd.DataFrame, treated: str, break_month, fx=None, min_side: int = 4) -> dict:
    """Space x time randomisation inference. Placebo units never use the real treated unit as a control."""
    months = y.index[min_side:len(y) - min_side + 1]
    real = did(y, treated, break_month, fx)
    rows = []
    donors = [c for c in y.columns if c != treated]
    for unit in [treated, *donors]:
        pool = y if unit == treated else y[donors]
        pfx = fx if (fx is None or unit == treated) else fx[donors]
        for m in months:
            rows.append({"unit": unit, "break_month": m, "beta": did(pool, unit, m, pfx)})
    dist = pd.DataFrame(rows)
    at_break = dist[dist.break_month == pd.Timestamp(break_month)]
    korea = dist[dist.unit == treated]
    tol = 1e-12
    return {
        "beta": real,
        "p_space_time": float((dist.beta.abs() >= abs(real) - tol).mean()),
        "p_space": float((at_break.beta.abs() >= abs(real) - tol).mean()),
        "p_time": float((korea.beta.abs() >= abs(real) - tol).mean()),
        "n_space_time": len(dist), "n_space": len(at_break), "n_time": len(korea),
        "distribution": dist,
    }


def mde(placebo_betas, alpha: float = 0.05, power: float = 0.8) -> float:
    from scipy.stats import norm
    return float((norm.ppf(1 - alpha / 2) + norm.ppf(power)) * np.std(placebo_betas, ddof=1))


def synthetic_control(levels: pd.DataFrame, treated: str, break_month) -> dict:
    """Demeaned synthetic control on log levels: weights w >= 0, sum w = 1, fit on pre-break months."""
    logs = np.log(levels)
    pre = logs.index < pd.Timestamp(break_month)
    centred = logs - logs[pre].mean()
    donors = [c for c in logs.columns if c != treated]
    Y1, Y0 = centred.loc[pre, treated].to_numpy(), centred.loc[pre, donors].to_numpy()
    k = len(donors)
    res = minimize(lambda w: ((Y1 - Y0 @ w) ** 2).sum(), np.full(k, 1 / k), method="SLSQP",
                   bounds=[(0, 1)] * k, constraints=({"type": "eq", "fun": lambda w: w.sum() - 1},),
                   options={"ftol": 1e-12, "maxiter": 1000})
    w = pd.Series(res.x, index=donors).clip(lower=0)
    w = w / w.sum()
    synth = centred[donors] @ w
    gap = centred[treated] - synth
    rmspe_pre = float(np.sqrt((gap[pre] ** 2).mean()))
    rmspe_post = float(np.sqrt((gap[~pre] ** 2).mean()))
    return {"weights": w, "gap": gap, "synthetic": synth, "actual": centred[treated],
            "rmspe_pre": rmspe_pre, "rmspe_post": rmspe_post, "ratio": rmspe_post / rmspe_pre,
            "post_mean_gap": float(gap[~pre].mean())}


def synthetic_placebo(levels: pd.DataFrame, treated: str, break_month) -> dict:
    """In-space placebo for the synthetic control: rank of Korea's post/pre RMSPE ratio among all units."""
    real = synthetic_control(levels, treated, break_month)
    donors = [c for c in levels.columns if c != treated]
    ratios = {treated: real["ratio"]}
    for d in donors:
        ratios[d] = synthetic_control(levels[donors], d, break_month)["ratio"]
    r = pd.Series(ratios)
    return {**real, "placebo_ratios": r, "p_rank": float((r >= real["ratio"] - 1e-12).mean())}


def run(panel: pd.DataFrame, break_month: str) -> dict[str, pd.DataFrame]:
    """All comparison outputs for the five pre-registered metrics."""
    panel = panel.assign(month=pd.to_datetime(panel["month"]))
    did_rows, paths, bands, weights, synth_rows, synth_paths, shifts = [], [], [], [], [], [], []
    for metric, (column, _) in METRICS.items():
        cs = countries_for(metric, panel)
        levels = wide(panel, column, cs)
        y = yoy(levels)
        # Raw post-minus-pre change of mean YoY growth, country by country (descriptive, no controls).
        post = y.index >= pd.Timestamp(break_month)
        change = y[post].mean() - y[~post].mean()
        shifts.append(pd.DataFrame({"metric": metric, "label": METRIC_KO[metric], "country": change.index,
                                    "change": change.values, "is_korea": change.index == TREATED}))
        fx = yoy(wide(panel, "lcu_per_usd", cs)).reindex(y.index)
        plc = placebo(y, TREATED, break_month, fx)
        dist = plc.pop("distribution")
        unadjusted = did(y, TREATED, break_month)
        pt = pretrend(y, TREATED, break_month)
        did_rows.append({"metric": metric, "label": METRIC_KO[metric], "controls": ", ".join(cs[1:]),
                         "pretrend_leads_p": pt["leads_p"], "pretrend_drift_p": pt["drift_p"],
                         "parallel_trends_ok": pt["leads_p"] >= 0.05 and pt["drift_p"] >= 0.05,
                         "beta_fx_adjusted": plc["beta"], "beta_unadjusted": unadjusted,
                         "pct_effect": float(np.expm1(plc["beta"])),
                         "p_space_time": plc["p_space_time"], "p_space": plc["p_space"], "p_time": plc["p_time"],
                         "n_space_time": plc["n_space_time"], "n_space": plc["n_space"], "n_time": plc["n_time"],
                         "mde_80pct_power": mde(dist[dist.unit != TREATED].beta)})
        path = event_path(y, TREATED, break_month)
        paths.append(pd.DataFrame({"metric": metric, "month": path.index, "korea_minus_controls": path.values}))
        for d in cs[1:]:
            dp = event_path(y[cs[1:]], d, break_month)
            bands.append(pd.DataFrame({"metric": metric, "country": d, "month": dp.index, "gap": dp.values}))
        sc = synthetic_placebo(levels, TREATED, break_month)
        weights.append(pd.DataFrame({"metric": metric, "donor": sc["weights"].index, "weight": sc["weights"].values}))
        synth_rows.append({"metric": metric, "label": METRIC_KO[metric], "rmspe_pre": sc["rmspe_pre"],
                           "rmspe_post": sc["rmspe_post"], "ratio": sc["ratio"], "post_mean_gap": sc["post_mean_gap"],
                           "p_rank": sc["p_rank"], "n_units": len(sc["placebo_ratios"])})
        synth_paths.append(pd.DataFrame({"metric": metric, "month": sc["gap"].index, "korea": sc["actual"].values,
                                         "synthetic": sc["synthetic"].values, "gap": sc["gap"].values}))
    did_tbl = pd.DataFrame(did_rows)
    for method, col in (("holm", "p_holm"), ("fdr_bh", "q_bh")):
        did_tbl[col] = multipletests(did_tbl["p_space_time"], method=method)[1]
    did_tbl["detectable"] = did_tbl["beta_fx_adjusted"].abs() >= did_tbl["mde_80pct_power"]
    return {"did": did_tbl, "shifts": pd.concat(shifts, ignore_index=True),
            "event_path": pd.concat(paths, ignore_index=True),
            "event_band": pd.concat(bands, ignore_index=True), "synth_weights": pd.concat(weights, ignore_index=True),
            "synth": pd.DataFrame(synth_rows), "synth_path": pd.concat(synth_paths, ignore_index=True)}
