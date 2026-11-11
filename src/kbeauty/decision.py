"""Decisions for a K-beauty brand that stocks US warehouses (plan section 5.8, Q5, D1, D2).

Lead time here is the ocean leg observed in public data: Busan departure to US entrance filing (CF-1400).
Terminal dwell, customs and inland delivery are not observed and enter only as explicit scenario parameters.

Safety stock in days of average demand (normal approximation, demand and lead time independent):
    SS / d = z * sqrt(L * CV_d^2 + sigma_L^2)
with L the mean lead time (days), sigma_L its standard deviation (days) and CV_d the coefficient of variation
of daily demand. Total inventory a route ties up = pipeline stock (L days) + safety stock (SS/d days).
"""

import numpy as np
import pandas as pd
from scipy.stats import norm


def period_shift(transit: pd.DataFrame, region: str, split: str, n_boot: int = 5000, seed: int = 0) -> dict:
    """Pre-registered Q5: change in median Busan -> US filing days after `split`, with a bootstrap 95% CI."""
    d = transit[transit["region"] == region]
    month = pd.to_datetime(d["filing_month"])
    pre = d.loc[month < pd.Timestamp(split), "departure_to_filing_days"].to_numpy(float)
    post = d.loc[month >= pd.Timestamp(split), "departure_to_filing_days"].to_numpy(float)
    rng = np.random.default_rng(seed)
    diffs = (np.median(rng.choice(post, (n_boot, len(post))), axis=1)
             - np.median(rng.choice(pre, (n_boot, len(pre))), axis=1))
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return {"region": region, "n_pre": len(pre), "n_post": len(post), "median_pre": float(np.median(pre)),
            "median_post": float(np.median(post)), "diff": float(np.median(post) - np.median(pre)),
            "ci_low": float(lo), "ci_high": float(hi), "sd_pre": float(pre.std(ddof=1)), "sd_post": float(post.std(ddof=1)),
            "changed": not (lo <= 0 <= hi)}


def lead_time_stats(transit: pd.DataFrame, region: str) -> dict:
    x = transit.loc[transit["region"] == region, "departure_to_filing_days"].astype(float)
    return {"region": region, "n": len(x), "mean": float(x.mean()), "sd": float(x.std(ddof=1)),
            "p10": float(x.quantile(0.1)), "p50": float(x.median()), "p90": float(x.quantile(0.9))}


def safety_stock_days(lead_mean: float, lead_sd: float, cv_demand: float, service_level: float) -> float:
    z = norm.ppf(service_level)
    return float(z * np.sqrt(lead_mean * cv_demand ** 2 + lead_sd ** 2))


def route_inventory(lead_mean: float, lead_sd: float, cv_demand: float, service_level: float) -> dict:
    ss = safety_stock_days(lead_mean, lead_sd, cv_demand, service_level)
    return {"pipeline_days": lead_mean, "safety_days": ss, "total_days": lead_mean + ss}


def route_comparison(east: dict, west: dict, inland_days, inland_sd: float, cv_grid, service_levels) -> pd.DataFrame:
    """East-coast direct (observed ocean leg) vs West-coast unloading + inland leg (scenario), in inventory days."""
    rows = []
    for cv in cv_grid:
        for sl in service_levels:
            e = route_inventory(east["mean"], east["sd"], cv, sl)
            for inland in inland_days:
                w = route_inventory(west["mean"] + inland, np.sqrt(west["sd"] ** 2 + inland_sd ** 2), cv, sl)
                rows.append({"cv_demand": cv, "service_level": sl, "inland_days": inland,
                             "east_total_days": e["total_days"], "west_total_days": w["total_days"],
                             "east_safety_days": e["safety_days"], "west_safety_days": w["safety_days"],
                             "west_minus_east_days": w["total_days"] - e["total_days"]})
    return pd.DataFrame(rows)


def break_even_inland_days(east: dict, west: dict, inland_sd: float, cv: float, service_level: float) -> float:
    """Inland days at which West routing ties up as much inventory as East direct (bisection on a monotone gap)."""
    def gap(inland):
        w = route_inventory(west["mean"] + inland, np.sqrt(west["sd"] ** 2 + inland_sd ** 2), cv, service_level)
        e = route_inventory(east["mean"], east["sd"], cv, service_level)
        return w["total_days"] - e["total_days"]
    lo, hi = 0.0, 60.0
    if gap(lo) > 0:
        return 0.0
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if gap(mid) < 0 else (lo, mid)
    return float((lo + hi) / 2)


def port_disruption(port_region: pd.DataFrame, concentration: pd.DataFrame, window: tuple[str, str],
                    closure_days=(7, 14, 30)) -> pd.DataFrame:
    """Sea-import value exposed if LA/Long Beach stopped for N days, using the window's average daily vessel value."""
    pr = port_region.assign(month=pd.to_datetime(port_region["month"]))
    conc = concentration.assign(month=pd.to_datetime(concentration["month"]))
    w = (pr["month"] >= pd.Timestamp(window[0])) & (pr["month"] <= pd.Timestamp(window[1]))
    vessel_total = pr.loc[w, "vessel_value_usd"].sum()
    days = (pd.Timestamp(window[1]) + pd.offsets.MonthEnd(0) - pd.Timestamp(window[0])).days + 1
    cw = (conc["month"] >= pd.Timestamp(window[0])) & (conc["month"] <= pd.Timestamp(window[1]))
    la_lb_share = float((conc.loc[cw, "la_lb_share"]).mean())
    per_day = vessel_total / days * la_lb_share
    return pd.DataFrame([{"closure_days": n, "la_lb_share": la_lb_share, "vessel_value_per_day_usd": vessel_total / days,
                          "exposed_value_usd": per_day * n} for n in closure_days])


def route_sensitivity(east: dict, west: dict, inland_days, inland_sds, cv_grid, service_levels) -> pd.DataFrame:
    """Preferred route over the full scenario grid (inland days x inland sd x demand CV x service level).
    'west' when West unloading + inland ties up fewer inventory days than East direct, else 'east'.
    Every axis is an assumption; only the two ocean legs are observed."""
    frames = []
    for sd in inland_sds:
        g = route_comparison(east, west, inland_days, sd, cv_grid, service_levels)
        frames.append(g.assign(inland_sd_days=sd,
                               break_even_inland_days=[break_even_inland_days(east, west, sd, cv, sl)
                                                       for cv, sl in zip(g.cv_demand, g.service_level)]))
    out = pd.concat(frames, ignore_index=True)
    out["preferred_route"] = np.where(out.west_minus_east_days < 0, "west", "east")
    return out[["inland_sd_days", "cv_demand", "service_level", "inland_days", "east_total_days", "west_total_days",
                "west_minus_east_days", "preferred_route", "break_even_inland_days"]]
