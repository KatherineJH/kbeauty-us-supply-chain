"""Market structure: concentration of entry ports and lead-time gaps between coasts."""

import numpy as np
import pandas as pd


def hhi(values) -> float:
    """Herfindahl-Hirschman index on a 0-10,000 scale from non-negative values (shares are computed here)."""
    v = np.asarray(values, dtype=float)
    if v.sum() <= 0:
        raise ValueError("values must have a positive sum")
    s = v / v.sum()
    return float((s ** 2).sum() * 10_000)


def bootstrap_median(x, n_boot: int = 5000, seed: int = 0, ci: float = 0.95) -> dict:
    """Median with a percentile bootstrap interval."""
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(x, size=(n_boot, len(x)), replace=True), axis=1)
    a = (1 - ci) / 2
    return {"n": len(x), "median": float(np.median(x)),
            "ci_low": float(np.quantile(meds, a)), "ci_high": float(np.quantile(meds, 1 - a))}


def transit_by_region(transit: pd.DataFrame, regions=("west_coast", "gulf_coast", "east_coast"),
                      seed: int = 0) -> pd.DataFrame:
    """transit: mart_vessel_transit. Median Busan -> US filing days with bootstrap CI and percentiles."""
    rows = []
    for r in regions:
        d = transit.loc[transit["region"] == r, "departure_to_filing_days"]
        rows.append({"region": r, **bootstrap_median(d, seed=seed),
                     "p10": float(d.quantile(0.1)), "p90": float(d.quantile(0.9))})
    return pd.DataFrame(rows)


def median_gap(transit: pd.DataFrame, a: str, b: str, n_boot: int = 5000, seed: int = 0) -> dict:
    """Difference in median days (region a minus region b) with a bootstrap interval."""
    xa = transit.loc[transit["region"] == a, "departure_to_filing_days"].to_numpy(float)
    xb = transit.loc[transit["region"] == b, "departure_to_filing_days"].to_numpy(float)
    rng = np.random.default_rng(seed)
    diffs = (np.median(rng.choice(xa, (n_boot, len(xa))), axis=1)
             - np.median(rng.choice(xb, (n_boot, len(xb))), axis=1))
    return {"gap_days": float(np.median(xa) - np.median(xb)),
            "ci_low": float(np.quantile(diffs, 0.025)), "ci_high": float(np.quantile(diffs, 0.975))}
