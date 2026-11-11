"""Machine-learning forecasts for the 8 bottom series, with model selection that never looks at the test month.
Scope: these are forecasts of public US import statistics (value recorded by Census), an external market
signal. They are not demand forecasts from a company's own sales, inventory or order data.

Nine regressors from four families compete under one protocol:
- linear: ridge, elastic net
- instance-based: k-nearest neighbours
- tree ensembles: random forest, extra trees, gradient boosting (scikit-learn), LightGBM, XGBoost
- neural: a one-hidden-layer MLP

Every model is a pooled (global) regressor over the 8 series with a direct strategy per horizon h. Two design
choices are tuned together with each model's own hyperparameters, because neither is obvious on 8 x 32 months:
- target: the log level, or the log change from the last observed month (trees cannot extrapolate a level),
- features: 'short' (last 3 months, calendar month, series, policy flag, log KRW/USD at the origin), or 'full'
  (short + the same month a year earlier). 'full' loses the first 12 months of training rows.

Nested rolling origin. At each test origin o, every configuration is scored on the last `inner_folds` months
that are already observed at o (validation target v, trained only on rows whose target is at or before v - h),
so the score uses nothing after o. Per model the best configuration is refit on all rows up to o and forecasts
o + h ('ml_<model>'); the best configuration over all models gives 'ml_selected', the honest out-of-sample
result of "compare several models and keep the best".
"""

import warnings
from dataclasses import dataclass, field
from itertools import product

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .hierarchy import MODES, GROUPS, all_nodes

SERIES = [f"{g}|{m}" for g in GROUPS for m in MODES]
NUMERIC = {"short": ["lag0", "lag1", "lag2", "post", "log_fx"],
           "full": ["lag0", "lag1", "lag2", "lag_season", "post", "log_fx"]}
CATEGORICAL = ["series", "month_of_year"]
SEED = 0

# Model-specific hyperparameter grids, kept small: each fold trains on 56-200 rows.
GRIDS = {
    "ridge": [{"alpha": a} for a in (0.1, 1.0, 10.0)],
    "elastic_net": [{"alpha": a, "l1_ratio": 0.5} for a in (0.001, 0.01)],
    "knn": [{"n_neighbors": k} for k in (5, 10)],
    "random_forest": [{"min_samples_leaf": m} for m in (3, 8)],
    "extra_trees": [{"min_samples_leaf": m} for m in (3, 8)],
    "gradient_boosting": [{"max_depth": d} for d in (2, 3)],
    "lightgbm": [{"num_leaves": n} for n in (4, 8)],
    "xgboost": [{"max_depth": d} for d in (2, 3)],
    "mlp": [{"alpha": a} for a in (0.01, 1.0)],
}
FAMILY = {"ridge": "linear", "elastic_net": "linear", "knn": "instance", "random_forest": "tree",
          "extra_trees": "tree", "gradient_boosting": "tree", "lightgbm": "tree", "xgboost": "tree", "mlp": "neural"}
# Trees split on integer codes; distance- and gradient-based models need scaled numbers and one-hot categories.
SCALED = {"ridge", "elastic_net", "knn", "mlp"}


@dataclass(frozen=True)
class Config:
    model: str
    params: tuple
    target: str      # 'level' or 'diff'
    features: str    # 'short' or 'full'

    @property
    def label(self) -> str:
        p = ",".join(f"{k}={v}" for k, v in self.params)
        return f"{self.model}[{p};{self.target};{self.features}]"


def configs(models=None) -> list[Config]:
    out = []
    for m in (models or GRIDS):
        for params, target, feats in product(GRIDS[m], ("level", "diff"), ("short", "full")):
            out.append(Config(m, tuple(sorted(params.items())), target, feats))
    return out


def _estimator(cfg: Config):
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import ElasticNet, Ridge
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    p = dict(cfg.params)
    reg = {
        "ridge": lambda: Ridge(**p),
        "elastic_net": lambda: ElasticNet(**p, max_iter=20000),
        "knn": lambda: KNeighborsRegressor(**p, weights="distance"),
        "random_forest": lambda: RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=1, **p),
        "extra_trees": lambda: ExtraTreesRegressor(n_estimators=100, random_state=SEED, n_jobs=1, **p),
        "gradient_boosting": lambda: GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                                               random_state=SEED, **p),
        "lightgbm": lambda: __import__("lightgbm").LGBMRegressor(
            n_estimators=200, learning_rate=0.05, min_child_samples=5, random_state=SEED, n_jobs=1,
            deterministic=True, force_row_wise=True, verbose=-1, **p),
        "xgboost": lambda: __import__("xgboost").XGBRegressor(
            n_estimators=200, learning_rate=0.05, random_state=SEED, n_jobs=1, tree_method="exact", **p),
        "mlp": lambda: MLPRegressor(hidden_layer_sizes=(16,), solver="lbfgs", max_iter=2000,
                                    random_state=SEED, **p),
    }[cfg.model]()
    if cfg.model not in SCALED:
        return reg
    pre = ColumnTransformer([("num", StandardScaler(), NUMERIC[cfg.features]),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)])
    return make_pipeline(pre, reg)


def feature_frame(bottom: pd.DataFrame, fx: pd.Series, h: int, policy_from: str, future: int = 0) -> pd.DataFrame:
    """One row per (series, target month): features known at origin = target - h. Rows lacking short-feature
    lags are dropped; 'lag_season' is NaN for the first 12 months and those rows are dropped only for 'full'.
    `future` > 0 appends that many months after the data, with y = NaN, as rows to forecast."""
    idx = bottom.index.append(pd.date_range(bottom.index[-1] + pd.offsets.MonthBegin(), periods=future, freq="MS"))
    logv = np.log(bottom[SERIES]).reindex(idx)
    log_fx = np.log(fx.reindex(idx)).shift(h)
    frames = []
    for i, s in enumerate(SERIES):
        x = pd.DataFrame({"y": logv[s]})
        for k in range(3):
            x[f"lag{k}"] = logv[s].shift(h + k)
        x["lag_season"] = logv[s].shift(12) if h <= 12 else np.nan
        x["post"] = (x.index >= pd.Timestamp(policy_from)).astype(float)
        x["log_fx"] = log_fx
        x["series"] = i
        x["month_of_year"] = x.index.month
        x["node"] = s
        frames.append(x)
    out = pd.concat(frames).dropna(subset=["lag0", "lag1", "lag2", "log_fx"])
    return out[out["y"].notna() | (out.index > bottom.index[-1])]


def _rows(data: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    return data.dropna(subset=["lag_season"]) if cfg.features == "full" else data


def _fit(cfg: Config, train: pd.DataFrame):
    from sklearn.exceptions import ConvergenceWarning
    y = train["y"] - train["lag0"] if cfg.target == "diff" else train["y"]
    with warnings.catch_warnings():
        # lbfgs may stop at max_iter on 56-200 rows; the fit is still scored like any other configuration.
        warnings.simplefilter("ignore", ConvergenceWarning)
        return _estimator(cfg).fit(train[NUMERIC[cfg.features] + CATEGORICAL], y)


def _predict(cfg: Config, est, test: pd.DataFrame) -> np.ndarray:
    """Forecasts of the log level for the test rows."""
    pred = est.predict(test[NUMERIC[cfg.features] + CATEGORICAL])
    return pred + test["lag0"].to_numpy() if cfg.target == "diff" else pred


def _fit_predict(cfg: Config, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    return _predict(cfg, _fit(cfg, train), test)


def _wape_log(pred_log, y_log) -> float:
    a = np.exp(np.asarray(y_log, float))
    return float(np.abs(np.exp(np.asarray(pred_log, float)) - a).sum() / a.sum())


@dataclass
class Result:
    forecasts: pd.DataFrame          # origin, target, horizon, model, method, node, actual, forecast
    selection: pd.DataFrame          # per origin, horizon and model: chosen configuration and its inner WAPE
    importance: pd.DataFrame = field(default_factory=pd.DataFrame)


def _folds(cfg: Config, data: pd.DataFrame, targets: list, h: int) -> list:
    """For each month v: fit on rows whose target is at or before v - h (all known at origin v - h), forecast v.
    Single-threaded BLAS/OpenMP keeps every rerun bit-identical."""
    d = _rows(data, cfg)
    out = []
    with threadpool_limits(1):
        for v in targets:
            train, test = d[d.index <= v - pd.offsets.MonthBegin(h)], d[d.index == v]
            out.append(None if train.empty or test.empty else _fit_predict(cfg, train, test))
    return out


def _run_folds(cands: list, data: pd.DataFrame, targets: list, h: int, n_jobs: int) -> list:
    """Parallel over configurations. On Windows a worker pool can fail to start when standard handles are
    redirected (seen under pytest capture); the serial path gives identical results, only slower."""
    from joblib import Parallel, delayed
    from joblib.externals.loky.process_executor import BrokenProcessPool

    if n_jobs != 1:
        try:
            return Parallel(n_jobs=n_jobs, max_nbytes=None)(delayed(_folds)(cfg, data, targets, h) for cfg in cands)
        except (BrokenProcessPool, OSError):
            pass
    return [_folds(cfg, data, targets, h) for cfg in cands]


def backtest(bottom: pd.DataFrame, fx: pd.Series, first_origin: int, policy_from: str,
             horizons=(1, 2, 3), inner_folds: int = 3, models=None, importance_repeats: int = 10,
             n_jobs: int = -1) -> Result:
    """A fold (configuration, horizon h, forecast month v) depends only on v, never on the origin it is used
    for: as an inner fold of origins v .. v + inner_folds - 1, and as the outer forecast of origin v - h.
    Each fold is therefore fitted once (in parallel; every fit is single-threaded and seeded)."""
    months = bottom.index
    cands = configs(models)
    rows, sel, kept = [], [], {h: [] for h in horizons}
    for h in horizons:
        data = feature_frame(bottom, fx, h, policy_from)
        targets = list(months[first_origin - inner_folds + 1:])
        out = _run_folds(cands, data, targets, h, n_jobs)
        pred = {(cfg, v): p_ for cfg, res in zip(cands, out) for v, p_ in zip(targets, res)}
        y = {v: data[data.index == v].sort_values("series") for v in targets}
        for o in range(first_origin, len(months) - h):
            origin, target = months[o], months[o + h]
            # Inner score: the last `inner_folds` months already observed at the origin.
            inner = months[o - inner_folds + 1: o + 1]
            scores = {cfg: _wape_log(np.concatenate([pred[cfg, v] for v in inner]),
                                     np.concatenate([y[v]["y"].to_numpy() for v in inner]))
                      for cfg in cands if all(pred[cfg, v] is not None for v in inner)}
            best = {}
            for cfg, sc in scores.items():  # dict order follows `cands`, so ties keep the earlier configuration
                if cfg.model not in best or sc < scores[best[cfg.model]]:
                    best[cfg.model] = cfg
            overall = min(best.values(), key=lambda c: scores[c])
            test = y[target]
            for name, cfg in [*best.items(), ("selected", overall)]:
                for node, p_, a in zip(test["node"], pred[cfg, target], test["y"]):
                    rows.append({"origin": origin, "target": target, "horizon": h, "model": f"ml_{name}",
                                 "method": "base", "node": node, "actual": float(np.exp(a)),
                                 "forecast": float(np.exp(p_))})
                sel.append({"origin": origin, "horizon": h, "model": f"ml_{name}", "family": FAMILY[cfg.model],
                            "chosen": cfg.label, "chosen_model": cfg.model, "target": cfg.target,
                            "features": cfg.features, "inner_wape": scores[cfg]})
            if importance_repeats:
                with threadpool_limits(1):
                        kept[h].append((overall, _fit(overall, _rows(data, overall).loc[lambda d: d.index <= origin]),
                                    _rows(data, overall).loc[lambda d: d.index == target]))
    fc = pd.DataFrame(rows)
    return Result(_with_bottom_up(fc, bottom), pd.DataFrame(sel),
                  _permutation_importance(kept, importance_repeats) if importance_repeats else pd.DataFrame())


def forecast_from_last(bottom: pd.DataFrame, fx: pd.Series, policy_from: str, horizons=(1, 2, 3),
                       inner_folds: int = 3, models=None, n_jobs: int = -1) -> pd.DataFrame:
    """Forecasts after the last observed month with the same selection rule as `backtest`: configurations are
    scored on the last `inner_folds` observed months, each model keeps its best one ('ml_<model>'), and the best
    over all models is 'ml_selected'. Used to freeze out-of-sample forecasts before the data are published."""
    months = bottom.index
    cands = configs(models)
    rows = []
    for h in horizons:
        data = feature_frame(bottom, fx, h, policy_from, future=h)
        inner = list(months[len(months) - inner_folds:])
        out = _run_folds(cands, data, inner, h, n_jobs)
        y = np.concatenate([data[data.index == v].sort_values("series")["y"].to_numpy() for v in inner])
        scores = {cfg: _wape_log(np.concatenate(res), y) for cfg, res in zip(cands, out)
                  if all(r is not None for r in res)}
        best = {}
        for cfg, sc in scores.items():
            if cfg.model not in best or sc < scores[best[cfg.model]]:
                best[cfg.model] = cfg
        overall = min(best.values(), key=lambda c: scores[c])
        target = months[-1] + pd.offsets.MonthBegin(h)
        for name, cfg in [*best.items(), ("selected", overall)]:
            d = _rows(data, cfg)
            train, test = d[d.index <= months[-1]], d[d.index == target].sort_values("series")
            with threadpool_limits(1):
                pred = _fit_predict(cfg, train, test)
            for node, p_ in zip(test["node"], pred):
                rows.append({"target_month": target.strftime("%Y-%m"), "horizon": h, "model": f"ml_{name}",
                             "node": node, "forecast": float(np.exp(p_)), "chosen": cfg.label,
                             "inner_wape": scores[cfg], "data_through": months[-1].strftime("%Y-%m")})
    return pd.DataFrame(rows)


def _with_bottom_up(fc: pd.DataFrame, bottom: pd.DataFrame) -> pd.DataFrame:
    """Adds the 15-node bottom-up forecasts (sum of the 8 bottom forecasts) next to the base rows."""
    nodes = all_nodes(bottom)
    out = [fc]
    for (o, t, h, m), g in fc.groupby(["origin", "target", "horizon", "model"], sort=False):
        b = g.set_index("node").forecast.reindex(SERIES)
        agg = all_nodes(pd.DataFrame([b.to_numpy()], columns=SERIES)).iloc[0]
        out.append(pd.DataFrame({"origin": o, "target": t, "horizon": h, "model": m, "method": "bottom_up",
                                 "node": agg.index, "actual": nodes.loc[t, agg.index].to_numpy(float),
                                 "forecast": agg.to_numpy(float)}))
    return pd.concat(out, ignore_index=True)


def _permutation_importance(kept: dict, repeats: int) -> pd.DataFrame:
    """Out-of-sample permutation importance of the selected model, per horizon: each feature is shuffled across
    the pooled test rows of all origins (each row still predicted by its own origin's fitted model), and the rise in
    WAPE is averaged over `repeats` shuffles. Features a configuration did not use get no row."""
    rng = np.random.default_rng(SEED)
    out = []
    with threadpool_limits(1):
        out = _importance_rows(kept, repeats, rng)
    return pd.DataFrame(out)


def _importance_rows(kept: dict, repeats: int, rng) -> list:
    out = []
    for h, items in kept.items():
        if not items:
            continue
        test = pd.concat([t.assign(_k=k) for k, (_, _, t) in enumerate(items)])
        def predict(frame):
            return np.concatenate([_predict(cfg, est, frame[frame._k == k])
                                   for k, (cfg, est, _) in enumerate(items)])
        base_pred = predict(test)
        base = _wape_log(base_pred, test["y"])
        used = sorted(set().union(*[NUMERIC[c.features] for c, _, _ in items]) | set(CATEGORICAL),
                      key=(NUMERIC["full"] + CATEGORICAL).index)
        for f in used:
            rises = []
            for _ in range(repeats):
                shuffled = test.copy()
                shuffled[f] = rng.permutation(shuffled[f].to_numpy())
                rises.append(_wape_log(predict(shuffled), test["y"]) - base)
            out.append({"horizon": h, "feature": f, "wape_rise_mean": float(np.mean(rises)),
                        "wape_rise_sd": float(np.std(rises, ddof=1)), "base_wape": base, "repeats": repeats})
    return out
