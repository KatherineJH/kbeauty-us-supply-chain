"""Cross-platform checks for the result tables whose values depend on the floating-point libraries.

Two references, two kinds of check:
- reports/platform/linux/ (canonical): the output of the pinned Linux CI run (src/kbeauty/numeric_env.py). In
  that environment every table is byte-identical on any runner CPU, so scripts/compare_tables.py requires all
  31 tables to match it byte for byte. The official numbers quoted in the deck and README come from it.
- reports/tables/ (author environment, Windows): the tables the notebooks are executed against. On another
  platform the platform-sensitive tables below differ, so they are checked by the rules in this module.

Why some tables are platform-sensitive: every fit is single-threaded and seeded, so a rerun on the same machine
gives identical numbers. Across CPU classes and operating systems the last bits of numpy's exp/log and of the
BLAS kernels differ; the ETS start-value search and L-BFGS-B, SARIMAX, ridge and the MLP's lbfgs solver amplify
them, which can change which ML configuration the validation step selects at some origins.

Rules, applied to the freshly generated tables against reports/tables/:
1. schema       same columns in the same order
2. keys         the same set of key tuples (no row added or lost)
3. finite       every numeric value is finite
4. ranges       values lie in a plausible range for their meaning (WAPE, p-value, coverage, ...)
5. statistical  rows of non-ML models agree within STAT_RTOL (coverage within one observation)
6. recommend.   the operating recommendations derived from the tables are the same on every platform
ML values themselves are not compared across platforms; rules 1-4 and 6 cover them, and the byte-identical
canonical comparison covers every value in the canonical environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
TABLES = REPO / "reports" / "tables"

STAT_RTOL = 5e-3          # observed Windows-Linux gap for non-ML models: at most 1.1e-3 relative


@dataclass
class Spec:
    keys: list[str]
    ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    ml_row: str | None = None      # column whose value marks an ML row ("ml_" prefix); None = all rows ML
    all_ml: bool = False


SENSITIVE: dict[str, Spec] = {
    "forecast_backtest_detail.csv": Spec(["origin", "target", "horizon", "model"],
                                         {"actual": (0, np.inf), "forecast": (0, np.inf)}, ml_row="model"),
    "forecast_conformal.csv": Spec(["model", "method", "horizon", "level"],
                                   {"coverage": (0, 1), "level": (0, 1), "mean_width_log": (0, 10),
                                    "n": (1, np.inf), "n_invalid": (0, np.inf)}, ml_row="model"),
    "forecast_dm_tests.csv": Spec(["comparison"], {"p_value": (0, 1), "n": (2, np.inf)}, ml_row="comparison"),
    "forecast_hierarchy_wape.csv": Spec(["model", "method", "horizon", "level"], {"wape": (0, 2)}, ml_row="model"),
    "forecast_ml_importance.csv": Spec(["horizon", "feature"],
                                       {"wape_rise_mean": (-1, 2), "wape_rise_sd": (0, 2), "base_wape": (0, 2),
                                        "repeats": (1, np.inf)}, all_ml=True),
    "forecast_ml_selection.csv": Spec(["origin", "horizon", "model"], {"inner_wape": (0, 2)}, all_ml=True),
    "forecast_scores.csv": Spec(["model", "horizon"],
                                {"wape": (0, 2), "mape": (0, 2), "relative_wape": (0, 5), "n": (1, np.inf)},
                                ml_row="model"),
    # claims.csv: only claim C10 quotes ML numbers; every other cell must match exactly.
    "claims.csv": Spec(["claim_id"]),
}
ML_CLAIMS = {"C10"}


def _is_ml(series: pd.Series) -> pd.Series:
    return series.astype(str).str.contains(r"(?:^|\W)ml_", regex=True)


def _num(df: pd.DataFrame, keys: list[str]) -> list[str]:
    return [c for c in df.columns if c not in keys and pd.api.types.is_numeric_dtype(df[c])]


def check_table(name: str, new: pd.DataFrame, ref: pd.DataFrame) -> list[str]:
    spec, out = SENSITIVE[name], []
    if list(new.columns) != list(ref.columns):                                              # 1
        return [f"{name}: columns {list(new.columns)} != {list(ref.columns)}"]
    kn, kr = set(map(tuple, new[spec.keys].astype(str).values)), set(map(tuple, ref[spec.keys].astype(str).values))
    if kn != kr or len(new) != len(kn):                                                      # 2
        return [f"{name}: keys differ (+{len(kn - kr)} / -{len(kr - kn)}, duplicates {len(new) - len(kn)})"]
    num = _num(new, spec.keys)
    vals = new[num].to_numpy(float)
    if not np.isfinite(vals).all():                                                          # 3
        out.append(f"{name}: non-finite values in {[c for c in num if not np.isfinite(new[c]).all()]}")
    for col, (lo, hi) in spec.ranges.items():                                                # 4
        bad = ~new[col].between(lo, hi)
        if bad.any():
            out.append(f"{name}: {col} outside [{lo}, {hi}] in {int(bad.sum())} rows")
    if name == "claims.csv":                                                                 # 5 (text)
        m = new.merge(ref, on="claim_id", suffixes=("", "_ref"))
        for col in [c for c in new.columns if c != "claim_id"]:
            diff = m[col].astype(str) != m[f"{col}_ref"].astype(str)
            if col == "value":
                diff &= ~m.claim_id.isin(ML_CLAIMS)
            if diff.any():
                out.append(f"{name}: {col} differs for {list(m.claim_id[diff])}")
        return out
    if spec.all_ml:
        return out
    m = new.merge(ref, on=spec.keys, suffixes=("", "_ref"))                                  # 5 (numbers)
    stat = m[~_is_ml(m[spec.ml_row])]
    for col in num:
        a, b = stat[col].to_numpy(float), stat[f"{col}_ref"].to_numpy(float)
        if col == "coverage":   # a discrete share: allow one observation to fall on the other side
            ok = np.abs(a - b) <= 1 / stat["n"].to_numpy(float) + 1e-12
        elif col == "n_invalid":
            ok = np.abs(a - b) <= 1
        else:
            ok = np.isclose(a, b, rtol=STAT_RTOL, atol=1e-9)
        if not ok.all():
            i = int(np.flatnonzero(~ok)[0])
            out.append(f"{name}: non-ML {col} outside tolerance in {int((~ok).sum())} rows "
                       f"(e.g. {dict(stat[spec.keys].iloc[i])}: {b[i]!r} -> {a[i]!r})")
    return out


def recommendations(tables: dict[str, pd.DataFrame]) -> dict[str, object]:
    """The operating rules the deck and model card state, recomputed from one set of tables."""
    h = tables["forecast_hierarchy_wape.csv"]
    dm = tables["forecast_dm_tests.csv"].set_index("comparison").p_value
    tot = h[(h.level == "total") & ~_is_ml(h.model)]
    best = {k: tuple(tot[tot.horizon == k].sort_values("wape").iloc[0][["model", "method"]]) for k in (1, 3)}
    return {
        "1m_rule": "naive" if best[1][0] == "naive" else f"{best[1]}",
        "3m_rule": "ets_seasonal bottom_up" if best[3] == ("ets_seasonal", "bottom_up") else f"{best[3]}",
        "3m_beats_seasonal_naive": bool(dm["ets_seasonal(BU) vs seasonal_naive (h3)"] < 0.05),
        "ml_reliable_gain_1m": bool(dm["ml_selected(BU) vs naive (h1)"] < 0.05),
        "ml_reliable_gain_3m": bool(dm["ml_selected(BU) vs ets_seasonal(BU) (h3)"] < 0.05),
    }


EXPECTED = {"1m_rule": "naive", "3m_rule": "ets_seasonal bottom_up", "3m_beats_seasonal_naive": True,
            "ml_reliable_gain_1m": False, "ml_reliable_gain_3m": False}


def compare_frames(new: pd.DataFrame, old: pd.DataFrame, rtol: float, atol: float) -> str | None:
    """Same columns and rows; numeric columns within rtol/atol, every other column exactly."""
    if list(new.columns) != list(old.columns):
        return f"columns differ: {list(old.columns)} -> {list(new.columns)}"
    if len(new) != len(old):
        return f"rows differ: {len(old)} -> {len(new)}"
    for col in new.columns:
        a, b = new[col], old[col]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            av, bv = a.to_numpy(float), b.to_numpy(float)
            bad = ~np.isclose(av, bv, rtol=rtol, atol=atol, equal_nan=True)
            if bad.any():
                i = int(np.flatnonzero(bad)[0])
                return f"{col}: {bad.sum()} values outside tolerance (row {i}: {bv[i]!r} -> {av[i]!r})"
        elif not a.astype(str).equals(b.astype(str)):
            i = int(np.flatnonzero(a.astype(str).to_numpy() != b.astype(str).to_numpy())[0])
            return f"{col}: text differs (row {i}: {b.iloc[i]!r} -> {a.iloc[i]!r})"
    return None


def check_all(new: dict[str, pd.DataFrame], ref: dict[str, pd.DataFrame]) -> tuple[list[str], list[str]]:
    """new: freshly generated tables; ref: reports/tables at the reference revision (author environment)."""
    problems, notes = [], []
    for name in SENSITIVE:
        problems += check_table(name, new[name], ref[name])
    rec = recommendations(new)                                                               # 6
    for k, v in EXPECTED.items():
        if rec[k] != v:
            problems.append(f"recommendation changed: {k} = {rec[k]!r} (expected {v!r})")
    notes.append("recommend. " + ", ".join(f"{k}={v}" for k, v in rec.items()))
    return problems, notes
