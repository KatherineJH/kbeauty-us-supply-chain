"""Pre-registered out-of-sample forecasts (docs/preregistration.md section 5).

Primary forecast = equal-weight mean of `naive` and `ets_seasonal`, the best models at horizon 1 and 3 in the
v1 backtest. The 80% interval comes from the empirical distribution of actual/forecast ratios of the primary
forecast in the same rolling-origin backtest (origins 2025-12 .. 2026-07), per series and horizon.
"""

import numpy as np
import pandas as pd

from . import forecast, warehouse
from .paths import BACKTEST_FIRST_ORIGIN

HORIZON = 3
FIRST_ORIGIN = pd.Timestamp(BACKTEST_FIRST_ORIGIN)
CATEGORY_GROUP = {"skincare": "skincare", "makeup": "makeup", "hair": "hair",
                  "other": "other", "nail": "other", "fragrance": "other"}  # pre-registered: 4 groups


def primary(history: pd.Series, h: int) -> np.ndarray:
    return (forecast.naive(history, h) + forecast.ets_seasonal(history, h)) / 2


MODELS = {"primary": primary, "naive": forecast.naive, "ets_seasonal": forecast.ets_seasonal}


def series_to_forecast() -> dict[str, pd.Series]:
    s = warehouse.load("mart_monthly_summary")
    s["month"] = pd.to_datetime(s["month"])
    s = s.set_index("month").sort_index()
    out = {"total|value_usd": s["value_usd"].astype(float), "total|weight_kg": s["measured_weight_kg"].astype(float)}
    f = warehouse.load("fct_imports_hs10_mode_monthly")
    f["month"] = pd.to_datetime(f["month"])
    f = f[f["mode"].isin(["vessel", "air"])].assign(group=lambda d: d["category"].map(CATEGORY_GROUP))
    g = f.groupby(["group", "mode", "month"])["value_usd"].sum()
    for (group, mode), ser in g.groupby(level=[0, 1]):
        out[f"{group}_{mode}|value_usd"] = ser.droplevel([0, 1]).sort_index().astype(float)
    return out


def build() -> pd.DataFrame:
    rows = []
    for name, ser in series_to_forecast().items():
        first = int(np.flatnonzero(ser.index == FIRST_ORIGIN)[0])
        bt = forecast.backtest(ser, {"primary": primary}, first_origin=first, horizons=(1, 2, 3))
        ratio = (bt.actual / bt.forecast).groupby(bt.horizon)
        # The backtest errors can be one-sided (the series grew faster than the forecasts), so the raw 10-90%
        # ratio band may exclude 1.0. The band is widened to always contain the point forecast, and the median
        # ratio is stored separately to show the bias (pre-registration change log, 2026-11-02).
        q10, q90 = ratio.quantile(0.10).clip(upper=1.0), ratio.quantile(0.90).clip(lower=1.0)
        med = ratio.median()
        preds = {m: np.asarray(fn(ser.copy(), HORIZON)) for m, fn in MODELS.items()}
        for h in range(1, HORIZON + 1):
            target = ser.index[-1] + pd.DateOffset(months=h)
            p = float(preds["primary"][h - 1])
            rows.append({"series": name, "target_month": target.strftime("%Y-%m"), "horizon": h,
                         "forecast_primary": round(p), "pi80_low": round(p * q10[h]), "pi80_high": round(p * q90[h]),
                         "forecast_naive": round(float(preds["naive"][h - 1])),
                         "forecast_ets_seasonal": round(float(preds["ets_seasonal"][h - 1])),
                         "backtest_median_actual_to_forecast": round(float(med[h]), 3),
                         "backtest_errors_used": int(ratio.size()[h]), "data_through": ser.index[-1].strftime("%Y-%m")})
    return pd.DataFrame(rows)
