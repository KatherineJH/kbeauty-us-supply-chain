"""Hierarchical monthly forecasts, prediction intervals and nowcasting (plan section 5.7, F1-F4).
Machine-learning forecasts of the same bottom series are in `ml`.
Scope: these are forecasts of public US import statistics (value recorded by Census), an external market
signal. They are not demand forecasts from a company's own sales, inventory or order data.

Hierarchy (sea + air import value, US side): 8 bottom series = 4 product groups x 2 modes, aggregated to
4 product groups, 2 modes and the total (15 series). Base forecasts for every node are made independently,
then made coherent (children sum to parents) by
- bottom-up: keep the 8 bottom forecasts and sum them,
- OLS reconciliation: project the 15 base forecasts onto the coherent subspace, y~ = S (S'S)^-1 S' y^.

Intervals are split-conformal on log(actual / forecast): for each origin, the band is calibrated on the other
origins of the same horizon (leave-one-origin-out), so a band is never calibrated on its own error.
"""

from typing import Callable

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX

from . import forecast

GROUPS = ["skincare", "makeup", "hair", "other"]
MODES = ["vessel", "air"]
GROUP_OF = {"skincare": "skincare", "makeup": "makeup", "hair": "hair",
            "other": "other", "nail": "other", "fragrance": "other"}


def bottom_series(fct: pd.DataFrame) -> pd.DataFrame:
    """month x 8 bottom series ('skincare|vessel', ...) of sea/air import value from fct_imports_hs10_mode_monthly."""
    d = fct[fct["mode"].isin(MODES)].assign(month=lambda x: pd.to_datetime(x["month"]),
                                             group=lambda x: x["category"].map(GROUP_OF))
    w = d.pivot_table(index="month", columns=["group", "mode"], values="value_usd", aggfunc="sum").sort_index()
    w.columns = [f"{g}|{m}" for g, m in w.columns]
    return w[[f"{g}|{m}" for g in GROUPS for m in MODES]].astype(float)


def summing_matrix() -> tuple[pd.DataFrame, list[str]]:
    bottom = [f"{g}|{m}" for g in GROUPS for m in MODES]
    rows = {"total": [1.0] * len(bottom)}
    for m in MODES:
        rows[f"all|{m}"] = [1.0 if b.endswith(f"|{m}") else 0.0 for b in bottom]
    for g in GROUPS:
        rows[f"{g}|all"] = [1.0 if b.startswith(f"{g}|") else 0.0 for b in bottom]
    for b in bottom:
        rows[b] = [1.0 if x == b else 0.0 for x in bottom]
    S = pd.DataFrame(rows, index=bottom).T
    return S, bottom


def all_nodes(bottom: pd.DataFrame) -> pd.DataFrame:
    S, cols = summing_matrix()
    return pd.DataFrame(bottom[cols].to_numpy() @ S.to_numpy().T, index=bottom.index, columns=S.index)


def reconcile(base: pd.Series, method: str) -> pd.Series:
    """base: one forecast per node (index = S.index). Returns coherent forecasts for all nodes."""
    S, cols = summing_matrix()
    Sm = S.to_numpy()
    if method == "bottom_up":
        b = base.loc[cols].to_numpy()
    elif method == "ols":
        b = np.linalg.solve(Sm.T @ Sm, Sm.T @ base.loc[S.index].to_numpy())
    else:
        raise ValueError(method)
    return pd.Series(Sm @ b, index=S.index)


Model = Callable[[pd.Series, int], np.ndarray]


def backtest(bottom: pd.DataFrame, models: dict[str, Model], first_origin: int, horizons=(1, 2, 3)) -> pd.DataFrame:
    """Rolling-origin backtest on every node, with base, bottom-up and OLS-reconciled forecasts."""
    nodes = all_nodes(bottom)
    hmax = max(horizons)
    rows = []
    for origin in range(first_origin, len(nodes) - 1):
        steps = min(hmax, len(nodes) - 1 - origin)
        hist = nodes.iloc[: origin + 1]
        for name, model in models.items():
            base = {node: np.asarray(model(hist[node].copy(), steps), dtype=float) for node in nodes.columns}
            for h in horizons:
                if h > steps:
                    continue
                b = pd.Series({node: base[node][h - 1] for node in nodes.columns})
                versions = {"base": b, "bottom_up": reconcile(b, "bottom_up"), "ols": reconcile(b, "ols")}
                for method, fc in versions.items():
                    for node in nodes.columns:
                        rows.append({"origin": nodes.index[origin], "target": nodes.index[origin + h], "horizon": h,
                                     "model": name, "method": method, "node": node,
                                     "actual": float(nodes.iloc[origin + h][node]), "forecast": float(fc[node])})
    return pd.DataFrame(rows)


def wape(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    g = df.assign(abs_err=(df.forecast - df.actual).abs()).groupby(by)
    return (g.abs_err.sum() / g.actual.sum()).rename("wape").reset_index()


def conformal_coverage(df: pd.DataFrame, levels=(0.8, 0.95)) -> pd.DataFrame:
    """Leave-one-origin-out split-conformal bands on log(actual/forecast), pooled over nodes, per horizon.
    Returns empirical coverage and average band width (in log points) per level and horizon."""
    # OLS reconciliation can push a small node below zero. A non-positive forecast has no log ratio: it is
    # left out of calibration and counted as a miss (outside the band), and reported in n_invalid.
    valid = df.forecast > 0
    d = df.assign(lr=np.where(valid, np.log(df.actual / df.forecast.where(valid)), np.nan))
    rows = []
    for h, dh in d.groupby("horizon"):
        for lvl in levels:
            inside, widths = [], []
            for o, test in dh.groupby("origin"):
                calib = dh[(dh.origin != o)].lr.dropna()
                lo, hi = np.quantile(calib, (1 - lvl) / 2), np.quantile(calib, 1 - (1 - lvl) / 2)
                inside.extend((test.lr.notna() & (test.lr >= lo) & (test.lr <= hi)).tolist())
                widths.append(hi - lo)
            rows.append({"horizon": h, "level": lvl, "coverage": float(np.mean(inside)), "n": len(inside),
                         "n_invalid": int(dh.lr.isna().sum()), "mean_width_log": float(np.mean(widths))})
    return pd.DataFrame(rows)


def diebold_mariano(e1, e2, h: int = 1) -> dict:
    """DM test on absolute-error loss with the Harvey-Leybourne-Newbold small-sample correction.
    Negative statistic = model 1 has smaller loss."""
    d = np.abs(np.asarray(e1)) - np.abs(np.asarray(e2))
    n = len(d)
    dbar = d.mean()
    gamma = [np.sum((d[k:] - dbar) * (d[: n - k] - dbar)) / n for k in range(h)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    # With few origins the autocovariance sum can make the long-run variance negative; then fall back to the
    # lag-0 variance (a common convention) and say so in the result.
    fallback = bool(var <= 0)
    if fallback:
        var = gamma[0] / n
    stat = dbar / np.sqrt(var) * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    from scipy.stats import t
    return {"dm_stat": float(stat), "p_value": float(2 * t.sf(abs(stat), df=n - 1)), "n": n, "lag0_variance": fallback}


def make_sarimax(policy_from: str, fx: pd.Series) -> Model:
    """SARIMAX(1,1,0) on log value with a post-policy dummy and log FX (future FX held at its last value)."""
    def model(history: pd.Series, h: int) -> np.ndarray:
        idx = history.index
        future = pd.date_range(idx[-1] + pd.offsets.MonthBegin(), periods=h, freq="MS")
        def exog(ix, last_fx=None):
            f = np.log(fx.reindex(ix)) if last_fx is None else pd.Series(last_fx, index=ix)
            return pd.DataFrame({"post": (ix >= pd.Timestamp(policy_from)).astype(float), "log_fx": f.to_numpy()}, index=ix)
        ex = exog(idx)
        fit = SARIMAX(np.log(history.to_numpy(float)), exog=ex.to_numpy(), order=(1, 1, 0)).fit(disp=False)
        return np.exp(fit.forecast(h, exog=exog(future, np.log(fx.reindex(idx)).iloc[-1]).to_numpy()))
    return model


def nowcast_backtest(us: pd.Series, kr: pd.Series, first_origin: int, window: int = 3) -> pd.DataFrame:
    """US month t+1 from Korean month t+1 (published about three weeks earlier) times the average US/KR ratio of
    the last `window` months up to t. Compared with naive (US month t) and seasonal naive."""
    rows = []
    idx = us.index
    for origin in range(first_origin, len(idx) - 1):
        o, target = idx[origin], idx[origin + 1]
        ratio = (us.iloc[origin - window + 1: origin + 1] / kr.reindex(idx[origin - window + 1: origin + 1])).mean()
        preds = {"nowcast_kr_ratio": float(kr[target] * ratio), "naive": float(us.iloc[origin]),
                 "seasonal_naive": float(us.iloc[origin - 11])}
        for name, p in preds.items():
            rows.append({"origin": o, "target": target, "horizon": 1, "model": name, "actual": float(us[target]), "forecast": p})
    return pd.DataFrame(rows)


LEVEL_OF = {"total": "total", "all|vessel": "mode", "all|air": "mode",
            **{f"{g}|all": "group" for g in GROUPS}, **{f"{g}|{m}": "bottom" for g in GROUPS for m in MODES}}


def level_wape(df: pd.DataFrame) -> pd.DataFrame:
    """WAPE per model, reconciliation method, horizon and hierarchy level (bottom = 8 series pooled)."""
    return wape(df.assign(level=df["node"].map(LEVEL_OF)), ["model", "method", "horizon", "level"])


def standard_models(policy_from: str, krw: pd.Series) -> dict[str, Model]:
    return {"naive": forecast.naive, "seasonal_naive": forecast.seasonal_naive,
            "seasonal_naive_growth": forecast.seasonal_naive_growth, "ets_damped": forecast.ets_damped,
            "ets_seasonal": forecast.ets_seasonal, "sarimax_fx_policy": make_sarimax(policy_from, krw)}


def cusum_break(log_series: pd.Series) -> dict:
    """OLS-residual CUSUM test for parameter stability of YoY log growth (constant-mean model)."""
    from statsmodels.stats.diagnostic import breaks_cusumolsresid
    y = log_series.diff(12).dropna()
    resid = sm.OLS(y.to_numpy(), np.ones(len(y))).fit().resid
    stat, p, _ = breaks_cusumolsresid(resid, ddof=1)
    return {"cusum_stat": float(stat), "p_value": float(p), "n": len(y)}
