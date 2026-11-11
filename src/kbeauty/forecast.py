"""Rolling-origin backtest for monthly import forecasts.
Scope: these are forecasts of public US import statistics (value recorded by Census), an external market
signal. They are not demand forecasts from a company's own sales, inventory or order data.

Every model is a function f(history, horizon) -> forecasts that only sees history up to the forecast origin.
`backtest` slices the series itself, so no model can read the future; tests/test_forecast.py checks this by
changing future values and asserting the forecasts do not move.

With 32 months of data the honest question is not "which model wins" but "does anything beat the simple
baselines a buyer would use anyway", so every model is compared with seasonal naive (same month last year).
"""

from typing import Callable

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

Model = Callable[[pd.Series, int], np.ndarray]


def naive(history: pd.Series, h: int) -> np.ndarray:
    return np.repeat(history.iloc[-1], h)


def seasonal_naive(history: pd.Series, h: int) -> np.ndarray:
    return history.iloc[-12:].to_numpy()[:h] if h <= 12 else np.resize(history.iloc[-12:].to_numpy(), h)


def seasonal_naive_growth(history: pd.Series, h: int) -> np.ndarray:
    """Same month last year scaled by the trailing-12-month YoY growth (needs 24 months of history)."""
    growth = history.iloc[-12:].sum() / history.iloc[-24:-12].sum()
    return seasonal_naive(history, h) * growth


def ets_damped(history: pd.Series, h: int) -> np.ndarray:
    """Damped additive-trend exponential smoothing on log values, no seasonal term (too few full years)."""
    fit = ExponentialSmoothing(np.log(history.to_numpy(float)), trend="add", damped_trend=True,
                               initialization_method="estimated").fit()
    return np.exp(fit.forecast(h))


def ets_seasonal(history: pd.Series, h: int) -> np.ndarray:
    """Log-additive Holt-Winters with a 12-month season. Needs >= 24 months; unstable with so few cycles."""
    fit = ExponentialSmoothing(np.log(history.to_numpy(float)), trend="add", damped_trend=True,
                               seasonal="add", seasonal_periods=12, initialization_method="estimated").fit()
    return np.exp(fit.forecast(h))


MODELS: dict[str, Model] = {
    "naive": naive,
    "seasonal_naive": seasonal_naive,
    "seasonal_naive_growth": seasonal_naive_growth,
    "ets_damped": ets_damped,
    "ets_seasonal": ets_seasonal,
}


def backtest(series: pd.Series, models: dict[str, Model], first_origin: int, horizons=(1, 2, 3)) -> pd.DataFrame:
    """series: monthly values indexed by month. first_origin: index position of the first origin (last month
    of training data). Returns one row per origin x horizon x model."""
    series = series.sort_index()
    hmax = max(horizons)
    rows = []
    for origin in range(first_origin, len(series) - 1):
        history = series.iloc[: origin + 1]
        steps = min(hmax, len(series) - 1 - origin)
        for name, model in models.items():
            preds = np.asarray(model(history.copy(), steps), dtype=float)
            for h in horizons:
                if h > steps:
                    continue
                target = series.index[origin + h]
                rows.append({"origin": series.index[origin], "target": target, "horizon": h, "model": name,
                             "actual": float(series.iloc[origin + h]), "forecast": float(preds[h - 1])})
    return pd.DataFrame(rows)


def score(bt: pd.DataFrame, reference: str = "seasonal_naive") -> pd.DataFrame:
    """WAPE and MAPE per model x horizon, plus WAPE relative to the reference model (< 1 means better)."""
    d = bt.assign(abs_err=(bt.forecast - bt.actual).abs(), ape=(bt.forecast - bt.actual).abs() / bt.actual)
    g = d.groupby(["model", "horizon"]).agg(n=("actual", "size"), wape=("abs_err", "sum"),
                                            actual_sum=("actual", "sum"), mape=("ape", "mean"))
    g["wape"] = g["wape"] / g.pop("actual_sum")
    ref = g.xs(reference, level="model")["wape"]
    g["relative_wape"] = g["wape"] / g.index.get_level_values("horizon").map(ref).to_numpy()
    return g.reset_index()
