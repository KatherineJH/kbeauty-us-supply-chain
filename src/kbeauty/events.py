"""Did year-over-year growth shift after the August 2025 policy changes?

Seasonality is removed by working on YoY log changes: y_t = log(x_t) - log(x_{t-12}).
The shift is the difference in mean y between post and pre months, estimated by OLS on a post dummy with
Newey-West (HAC) standard errors because monthly growth rates are autocorrelated.

A single before/after comparison cannot separate a policy effect from anything else that changed at the same
time, so every estimate is reported with a placebo check: the same statistic is computed for every other
candidate break month, and the real break is ranked among them.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm


def yoy_log_change(s: pd.Series) -> pd.Series:
    """s: monthly series indexed by month (contiguous). Returns log YoY change, NaN for the first 12 months."""
    return np.log(s).diff(12)


def level_shift(y: pd.Series, break_month: str, hac_lags: int = 3) -> dict:
    """OLS of y on a post-break dummy. y must be indexed by month and contain no NaN."""
    y = y.dropna()
    post = (y.index >= pd.Timestamp(break_month)).astype(float)
    if post.sum() < 3 or (1 - post).sum() < 3:
        raise ValueError("need at least 3 observations on each side of the break")
    X = sm.add_constant(pd.Series(post, index=y.index, name="post"))
    fit = sm.OLS(y.values, X.values).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    ci = fit.conf_int()[1]
    return {"break_month": pd.Timestamp(break_month), "pre_mean": float(fit.params[0]),
            "shift": float(fit.params[1]), "ci_low": float(ci[0]), "ci_high": float(ci[1]),
            "p_value": float(fit.pvalues[1]), "n_pre": int((1 - post).sum()), "n_post": int(post.sum())}


def placebo_rank(y: pd.Series, break_month: str, min_side: int = 4) -> dict:
    """Rank |shift| at the real break among all candidate breaks with >= min_side months on each side.
    The placebo p-value is the share of candidates whose |shift| is at least as large as the real one."""
    y = y.dropna()
    candidates = y.index[min_side:len(y) - min_side + 1]
    shifts = pd.Series({m: level_shift(y, m)["shift"] for m in candidates})
    real = level_shift(y, break_month)["shift"]
    p = float((shifts.abs() >= abs(real) - 1e-12).mean())
    return {"real_shift": real, "n_candidates": len(shifts), "placebo_p": p,
            "largest_break": shifts.abs().idxmax(), "shifts": shifts}


def policy_shift_table(summary: pd.DataFrame, metrics: dict[str, str], break_month: str) -> pd.DataFrame:
    """summary: mart_monthly_summary. metrics: {column: label}. One row per metric."""
    s = summary.set_index("month").sort_index()
    rows = []
    for col, label in metrics.items():
        y = yoy_log_change(s[col])
        est = level_shift(y, break_month)
        plc = placebo_rank(y, break_month)
        rows.append({"metric": col, "label": label, **est,
                     "shift_pct_points": np.expm1(est["pre_mean"] + est["shift"]) - np.expm1(est["pre_mean"]),
                     "placebo_p": plc["placebo_p"], "n_candidates": plc["n_candidates"],
                     "largest_break": plc["largest_break"]})
    return pd.DataFrame(rows)
