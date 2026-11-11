"""Run-to-run determinism diagnostic: one full pipeline pass with every stage hashed (diagnostic branch only).

    uv run python scripts/diag_determinism.py --out diag/run.json [--build] [--label baseline-r1-A]

It runs the production code path (`report.generate_all`) once, with hooks that record a SHA-256 of the exact
bits produced at each stage, in the order the pipeline produces them:

  00_probe       kernel probes on fixed inputs (numpy exp/log/sum/dot/matmul, LAPACK solve, scipy L-BFGS-B,
                 a small sklearn MLP): separates "the CPU/library gives different bits" from pipeline logic
  10_input       every warehouse table read (row-order hash and order-free hash)
  20_series      series handed to the forecasting code (total, 8 bottom series, KRW/USD)
  30_features    ML feature frames per horizon (column order, row order, values)
  40_stat_fit    every statistical model call (ETS, SARIMAX, naive...) per series and origin; for ETS also the
                 brute-force start values, L-BFGS-B result and fitted parameters
  45_stat_table  forecast.backtest output (total series)
  50_ml_fold     every ML fold (configuration x horizon x forecast month): training-data hash and prediction
                 hash; for the MLP also n_iter_, final loss, L-BFGS-B status and message
  60_ml_select   ML selection (chosen configuration per origin/horizon/model) and ML forecasts
  70_reconcile   hierarchy.backtest output per model/origin/horizon/method (base, bottom-up, OLS)
  80_synth       synthetic-control SLSQP calls (start, solution, iterations)
  90_table       bytes of every CSV in reports/tables

It changes nothing in src/: the hooks wrap module attributes at run time and call the original functions.
"""

import argparse
import contextlib
import hashlib
import io
import json
import os
import platform
import subprocess
import sys
import time
import warnings
from importlib import metadata
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from kbeauty import comparison, forecast, hierarchy, ml, report, warehouse  # noqa: E402
from kbeauty.paths import TABLES_DIR  # noqa: E402

REC: list[dict] = []
THREAD_ENV = ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
              "NUMEXPR_NUM_THREADS", "OPENBLAS_CORETYPE", "OPENBLAS_VERBOSE", "NPY_DISABLE_CPU_FEATURES",
              "NPY_ENABLE_CPU_FEATURES", "OMP_PROC_BIND", "OMP_WAIT_POLICY", "PYTHONHASHSEED", "LOKY_MAX_CPU_COUNT"]
PACKAGES = ["numpy", "scipy", "scikit-learn", "statsmodels", "pandas", "lightgbm", "xgboost", "joblib",
            "threadpoolctl", "duckdb", "dbt-core", "dbt-duckdb", "scipy-openblas64", "scipy-openblas32"]


# ---------------------------------------------------------------- hashing

def h_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def h_arr(a) -> str:
    a = np.asarray(a)
    if a.dtype == object or a.dtype.kind in "USM":
        payload = "\x1f".join(map(repr, a.ravel().tolist())).encode()
    else:
        payload = np.ascontiguousarray(a).tobytes()
    return h_bytes(f"{a.dtype}|{a.shape}|".encode() + payload)


def h_df(df: pd.DataFrame, ordered: bool = True) -> str:
    d = df.reset_index()
    if not ordered:
        d = d.sort_values(list(d.columns), kind="mergesort").reset_index(drop=True)
    parts = [repr(list(map(str, d.columns))).encode(), repr([str(t) for t in d.dtypes]).encode()]
    parts += [h_arr(d[c].to_numpy()).encode() for c in d.columns]
    return h_bytes(b"|".join(parts))


def floats(a) -> list:
    return [float(x) for x in np.asarray(a, float).ravel()]


def rec(stage: str, key: str, h: str, **detail) -> None:
    REC.append({"stage": stage, "key": key, "hash": h, **({"detail": detail} if detail else {})})


# ---------------------------------------------------------------- environment

def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout
    except Exception as e:  # noqa: BLE001
        return f"<{type(e).__name__}: {e}>"


def environment() -> dict:
    from threadpoolctl import threadpool_info
    from numpy._core import _multiarray_umath as mu

    cpuinfo = Path("/proc/cpuinfo").read_text() if Path("/proc/cpuinfo").exists() else ""
    first = cpuinfo.split("\n\n")[0] if cpuinfo else ""
    field = {k.strip(): v.strip() for k, _, v in (ln.partition(":") for ln in first.splitlines())}
    flags = field.get("flags", "").split()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        np.show_runtime()
    try:
        np_config = np.show_config(mode="dicts")
    except Exception as e:  # noqa: BLE001
        np_config = repr(e)
    versions = {}
    for p in PACKAGES:
        try:
            versions[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            versions[p] = None
    return {
        "platform": platform.platform(), "python": sys.version, "machine": platform.machine(),
        "cpu_model": field.get("model name") or platform.processor(),
        "cpu_vendor": field.get("vendor_id"), "cpu_family": field.get("cpu family"), "cpu_model_id": field.get("model"),
        "cpu_stepping": field.get("stepping"), "microcode": field.get("microcode"),
        "cpu_flags_hash": h_bytes(" ".join(sorted(flags)).encode()) if flags else None,
        "avx512": sorted(f for f in flags if f.startswith("avx512")),
        "fma": "fma" in flags, "avx2": "avx2" in flags,
        "os_cpu_count": os.cpu_count(),
        "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "lscpu": _run(["lscpu"]),
        "thread_env": {k: os.environ.get(k) for k in THREAD_ENV},
        "versions": versions,
        "numpy_cpu_baseline": list(mu.__cpu_baseline__), "numpy_cpu_dispatch": list(mu.__cpu_dispatch__),
        "numpy_cpu_features_on": sorted(k for k, v in mu.__cpu_features__.items() if v),
        "numpy_show_runtime": buf.getvalue(), "numpy_show_config": np_config,
        "threadpool_main": threadpool_info(),
        "github": {k: os.environ.get(k) for k in ("GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT", "GITHUB_JOB", "GITHUB_SHA",
                                                  "RUNNER_NAME", "ImageVersion", "ImageOS")},
    }


# ---------------------------------------------------------------- kernel probes

def probes() -> None:
    import scipy.linalg
    from scipy.optimize import minimize, rosen, rosen_der
    from threadpoolctl import threadpool_limits

    rng = np.random.default_rng(0)
    x = rng.uniform(0.01, 50.0, 100_003)
    A, B = rng.standard_normal((200, 200)), rng.standard_normal((200, 200))
    b = rng.standard_normal(200)
    for limit in ("default", "1"):
        ctx = threadpool_limits(1) if limit == "1" else contextlib.nullcontext()
        with ctx:
            out = {
                "np.log": np.log(x), "np.exp": np.exp(x / 10), "np.log1p": np.log1p(x), "np.expm1": np.expm1(x / 10),
                "np.power": np.power(x, 0.37), "np.sqrt": np.sqrt(x), "np.sin": np.sin(x), "np.tanh": np.tanh(x / 25),
                "np.sum": np.sum(x), "np.dot": np.dot(x, x), "matmul": A @ B, "matvec": A @ b,
                "linalg.solve": np.linalg.solve(A, b), "linalg.lstsq": np.linalg.lstsq(A[:, :50], b, rcond=None)[0],
                "scipy.lu_solve": scipy.linalg.lu_solve(scipy.linalg.lu_factor(A), b),
            }
            for k, v in out.items():
                rec("00_probe", f"{k}|threads={limit}", h_arr(v))
            r = minimize(rosen, np.linspace(-1.2, 1.0, 10), jac=rosen_der, method="L-BFGS-B")
            rec("00_probe", f"lbfgsb.rosen|threads={limit}", h_arr(r.x), nit=int(r.nit), fun=float(r.fun),
                message=str(r.message))
            from sklearn.neural_network import MLPRegressor
            X = np.random.default_rng(1).standard_normal((150, 12))
            y = np.sin(X[:, 0]) + 0.1 * X[:, 1]
            m = MLPRegressor(hidden_layer_sizes=(16,), solver="lbfgs", max_iter=2000, random_state=0, alpha=0.01)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.fit(X, y)
            rec("00_probe", f"sklearn.mlp_lbfgs|threads={limit}", h_arr(m.predict(X)), n_iter=int(m.n_iter_),
                loss=float(m.loss_))


# ---------------------------------------------------------------- hooks

def hook_warehouse():
    orig = warehouse.load
    count: dict[str, int] = {}

    def load(table, *a, **k):
        df = orig(table, *a, **k)
        count[table] = count.get(table, 0) + 1
        rec("10_input", f"{table}#{count[table]}", h_df(df), unordered=h_df(df, ordered=False), rows=len(df),
            columns=list(map(str, df.columns)))
        return df
    return mock.patch.object(warehouse, "load", load)


def _ets_detail(history: pd.Series, h: int, seasonal: bool):
    """Same call as forecast.ets_damped / ets_seasonal, with the optimiser's start and result captured."""
    import statsmodels.tsa.holtwinters.model as hw
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    calls = []
    orig_min = hw.minimize

    def spy(fun, x0, *a, **k):
        res = orig_min(fun, x0, *a, **k)
        calls.append({"x0": floats(x0), "x": floats(res.x), "fun": float(res.fun), "nit": int(getattr(res, "nit", -1)),
                      "nfev": int(getattr(res, "nfev", -1)), "status": int(getattr(res, "status", -1)),
                      "message": str(getattr(res, "message", "")), "method": k.get("method")})
        return res
    kw = dict(trend="add", damped_trend=True, initialization_method="estimated")
    if seasonal:
        kw.update(seasonal="add", seasonal_periods=12)
    with mock.patch.object(hw, "minimize", spy):
        fit = ExponentialSmoothing(np.log(history.to_numpy(float)), **kw).fit()
    params = {k: (floats(v) if np.ndim(v) else float(v)) for k, v in fit.params.items()
              if isinstance(v, (float, int, np.floating, np.ndarray)) and k not in ("lamda", "remove_bias", "use_boxcox")}
    return np.exp(fit.forecast(h)), {"opt": calls, "params": params, "sse": float(fit.sse),
                                     "params_hash": h_arr(np.concatenate([np.ravel(v) for v in params.values()]))}


def _wrap_model(name: str, fn):
    def model(history: pd.Series, h: int):
        out = np.asarray(fn(history, h), dtype=float)
        key = f"{name}|{history.name}|{history.index[-1]:%Y-%m}|steps{h}|hist{h_arr(history.to_numpy(float))[:8]}"
        detail = {}
        if name in ("ets_damped", "ets_seasonal"):
            rep, detail = _ets_detail(history, h, seasonal=name == "ets_seasonal")
            detail["replica_matches"] = bool(np.array_equal(rep, out))
        detail["forecast"] = floats(out)
        rec("40_stat_fit", key, h_arr(out), **detail)
        return out
    model.__name__ = name
    return model


def hook_stat_models():
    patches = [mock.patch.object(forecast, n, _wrap_model(n, getattr(forecast, n)))
               for n in ("naive", "seasonal_naive", "seasonal_naive_growth", "ets_damped", "ets_seasonal")]
    patches.append(mock.patch.dict(forecast.MODELS, {n: _wrap_model(n, f) for n, f in forecast.MODELS.items()}))
    orig_sm = hierarchy.standard_models

    def standard_models(policy_from, krw):
        # forecast.<model> is already wrapped here; only the SARIMAX closure needs wrapping.
        models = orig_sm(policy_from, krw)
        models["sarimax_fx_policy"] = _wrap_model("sarimax_fx_policy", models["sarimax_fx_policy"])
        return models
    patches.append(mock.patch.object(hierarchy, "standard_models", standard_models))
    return patches


def hook_backtests():
    orig_fbt, orig_hbt, orig_mbt, orig_ff = forecast.backtest, hierarchy.backtest, ml.backtest, ml.feature_frame

    def fbt(series, models, first_origin, horizons=(1, 2, 3)):
        rec("20_series", f"forecast.backtest:series:{series.name}", h_arr(series.to_numpy(float)),
            index=h_arr(series.index.to_numpy()))
        bt = orig_fbt(series, models, first_origin, horizons)
        for (o, m), g in bt.groupby(["origin", "model"], sort=False):
            rec("45_stat_table", f"{m}|{o:%Y-%m}", h_arr(g[["horizon", "forecast"]].to_numpy(float)))
        return bt

    def hbt(bottom, models, first_origin, horizons=(1, 2, 3)):
        rec("20_series", "hierarchy.backtest:bottom", h_df(bottom), columns=list(bottom.columns))
        rec("20_series", "hierarchy.all_nodes", h_df(hierarchy.all_nodes(bottom)))
        out = orig_hbt(bottom, models, first_origin, horizons)
        for (m, o, hz, meth), g in out.groupby(["model", "origin", "horizon", "method"], sort=False):
            rec("70_reconcile", f"{m}|{o:%Y-%m}|h{hz}|{meth}", h_arr(g["forecast"].to_numpy(float)))
        return out

    def mbt(bottom, fx, first_origin, policy_from, *a, **k):
        rec("20_series", "ml.backtest:bottom", h_df(bottom))
        rec("20_series", "ml.backtest:fx", h_arr(fx.to_numpy(float)), index=h_arr(fx.index.to_numpy()))
        r = orig_mbt(bottom, fx, first_origin, policy_from, *a, **k)
        for _, s in r.selection.iterrows():
            rec("60_ml_select", f"select|{s.origin:%Y-%m}|h{s.horizon}|{s.model}", h_bytes(s.chosen.encode()),
                chosen=s.chosen, inner_wape=float(s.inner_wape))
        for (o, hz, m, meth), g in r.forecasts.groupby(["origin", "horizon", "model", "method"], sort=False):
            rec("60_ml_select", f"forecast|{o:%Y-%m}|h{hz}|{m}|{meth}", h_arr(g["forecast"].to_numpy(float)))
        if not r.importance.empty:
            rec("60_ml_select", "importance", h_df(r.importance))
        return r

    def ff(bottom, fx, h, policy_from, future=0):
        out = orig_ff(bottom, fx, h, policy_from, future)
        rec("30_features", f"h{h}|future{future}", h_df(out), unordered=h_df(out, ordered=False),
            columns=list(map(str, out.columns)), rows=len(out))
        return out

    return [mock.patch.object(forecast, "backtest", fbt), mock.patch.object(hierarchy, "backtest", hbt),
            mock.patch.object(ml, "backtest", mbt), mock.patch.object(ml, "feature_frame", ff)]


def diag_folds(cfg, data, targets, h):
    """Mirror of ml._folds (same rows, same fit/predict calls, same thread limit) that also returns a record per
    fold. Runs inside the joblib workers, like production."""
    import scipy.optimize
    from threadpoolctl import threadpool_info, threadpool_limits

    d = ml._rows(data, cfg)
    preds, recs = [], []
    with threadpool_limits(1):
        if not os.environ.get("_KB_DIAG_WORKER_SEEN"):  # once per worker process
            os.environ["_KB_DIAG_WORKER_SEEN"] = "1"
            recs.append({"stage": "51_worker", "key": f"pid{os.getpid()}", "hash": "",
                         "detail": {"threadpool_in_limit": threadpool_info(),
                                    "env": {k: os.environ.get(k) for k in THREAD_ENV}}})
        for v in targets:
            train, test = d[d.index <= v - pd.offsets.MonthBegin(h)], d[d.index == v]
            if train.empty or test.empty:
                preds.append(None)
                continue
            opt = []
            orig_min = scipy.optimize.minimize

            def spy(*a, **k):
                res = orig_min(*a, **k)
                opt.append(res)
                return res
            with mock.patch.object(scipy.optimize, "minimize", spy):
                est = ml._fit(cfg, train)
            p = ml._predict(cfg, est, test)
            preds.append(p)
            cols = ml.NUMERIC[cfg.features] + ml.CATEGORICAL
            detail = {"train": h_df(train[cols + ["y"]]), "test": h_df(test[cols + ["lag0"]]), "n_train": len(train)}
            if cfg.model == "mlp":
                mlp = est[-1]
                r = opt[-1] if opt else None
                detail.update(n_iter=int(mlp.n_iter_), loss=float(mlp.loss_),
                              coefs=h_arr(np.concatenate([c.ravel() for c in mlp.coefs_ + mlp.intercepts_])),
                              status=int(r.status) if r is not None else None,
                              message=str(r.message) if r is not None else None,
                              nfev=int(r.nfev) if r is not None else None,
                              hit_max_iter=bool(mlp.n_iter_ >= mlp.max_iter),
                              converged=bool(r is not None and r.status == 0))
            recs.append({"stage": "50_ml_fold", "key": f"h{h}|{cfg.label}|{v:%Y-%m}", "hash": h_arr(p),
                         "detail": detail})
    return preds, recs


def hook_ml_folds(serial: bool = False):
    def run_folds(cands, data, targets, h, n_jobs):
        from joblib import Parallel, delayed
        if serial:  # local smoke tests on Windows, where a loky pool can stall with redirected handles
            out = [diag_folds(c, data, targets, h) for c in cands]
        else:
            out = Parallel(n_jobs=n_jobs, max_nbytes=None)(delayed(diag_folds)(c, data, targets, h) for c in cands)
        for _, recs in out:
            REC.extend(recs)
        return [p for p, _ in out]
    return mock.patch.object(ml, "_run_folds", run_folds)


def hook_synth():
    orig = comparison.minimize
    n = [0]

    def spy(fun, x0, *a, **k):
        res = orig(fun, x0, *a, **k)
        n[0] += 1
        rec("80_synth", f"slsqp#{n[0]}", h_arr(res.x), x0=h_arr(x0), nit=int(res.nit), fun=float(res.fun),
            message=str(res.message))
        return res
    return mock.patch.object(comparison, "minimize", spy)


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--build", action="store_true", help="run dbt build first (fresh warehouse for this pass)")
    ap.add_argument("--no-probes", action="store_true")
    ap.add_argument("--serial", action="store_true", help="fit ML folds in this process (local smoke tests only)")
    args = ap.parse_args()

    t0 = time.time()
    env = environment()
    if args.build:
        s = warehouse.build()
        if s["tests_passed"] != s["tests_total"]:
            sys.exit(f"dbt data tests not passed: {s['not_passed']}")
    if not args.no_probes:
        probes()
    patches = [hook_warehouse(), *hook_stat_models(), *hook_backtests(), hook_ml_folds(args.serial), hook_synth()]
    with contextlib.ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        report.generate_all()
    for f in sorted(TABLES_DIR.glob("*.csv")):
        rec("90_table", f.name, h_bytes(f.read_bytes()))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"label": args.label, "seconds": round(time.time() - t0, 1), "env": env,
                               "records": REC}, indent=1, default=str), encoding="utf-8")
    tables = [r for r in REC if r["stage"] == "90_table"]
    print(f"diag  {len(REC)} records, {len(tables)} tables, fingerprint "
          f"{h_bytes(''.join(r['hash'] for r in tables).encode())}, cpu {env['cpu_model']!r}, "
          f"{time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
