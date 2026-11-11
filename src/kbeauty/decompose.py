"""Split a change in average unit value (USD/kg) into price, mix and interaction effects.

The average unit value of a mode is a weight-share-weighted mean of item unit values:
    UV = sum_i s_i * uv_i,  s_i = kg_i / sum kg,  uv_i = usd_i / kg_i
Between a base period (0) and a comparison period (1):
    dUV = sum (s1 - s0) * uv0      mix effect   (shift toward cheaper or dearer items)
        + sum s0 * (uv1 - uv0)     price effect (items themselves got cheaper or dearer)
        + sum (s1 - s0)(uv1 - uv0) interaction
The three parts add up exactly to dUV.
"""

import numpy as np
import pandas as pd


def unit_value_decomposition(base: pd.DataFrame, comp: pd.DataFrame, item: str = "hs10",
                             value: str = "value_usd", weight: str = "weight_kg") -> dict:
    """base/comp: one row per item with total value and weight for the period. Items with no weight are dropped."""
    def prep(df):
        g = df.groupby(item)[[value, weight]].sum()
        g = g[g[weight] > 0]
        return pd.DataFrame({"s": g[weight] / g[weight].sum(), "uv": g[value] / g[weight]})

    b, c = prep(base), prep(comp)
    items = b.index.union(c.index)
    b = b.reindex(items).fillna({"s": 0.0})
    c = c.reindex(items).fillna({"s": 0.0})
    # An item present in only one period has no price change; its unit value is carried across
    # so that its whole contribution is counted as mix.
    b["uv"] = b["uv"].fillna(c["uv"])
    c["uv"] = c["uv"].fillna(b["uv"])

    uv0 = float((b.s * b.uv).sum())
    uv1 = float((c.s * c.uv).sum())
    mix = float(((c.s - b.s) * b.uv).sum())
    price = float((b.s * (c.uv - b.uv)).sum())
    interaction = float(((c.s - b.s) * (c.uv - b.uv)).sum())
    return {"uv_base": uv0, "uv_comp": uv1, "change": uv1 - uv0,
            "mix": mix, "price": price, "interaction": interaction}


def decompose_by_mode(fct: pd.DataFrame, pre: tuple[str, str], post: tuple[str, str]) -> pd.DataFrame:
    """fct: rows of fct_imports_hs10_mode_monthly. Returns one row per mode (vessel, air)."""
    rows = []
    for mode in ("vessel", "air"):
        d = fct[fct["mode"] == mode]
        base = d[d["month"].between(*pre)]
        comp = d[d["month"].between(*post)]
        rows.append({"mode": mode, **unit_value_decomposition(base, comp)})
    out = pd.DataFrame(rows)
    out["change_pct"] = out["change"] / out["uv_base"]
    for k in ("mix", "price", "interaction"):
        out[f"{k}_share_of_change"] = out[k] / out["change"]
    return out


def window_decomposition(df: pd.DataFrame, item: str, value: str, weight: str,
                         pre: tuple[str, str], post: tuple[str, str], month: str = "month") -> dict:
    """Unit-value decomposition between two windows for any item code (HS10, HSK10, HS6) and value/weight pair."""
    d = df.assign(**{month: pd.to_datetime(df[month])})
    base = d[d[month].between(*pre)].rename(columns={value: "value_usd", weight: "weight_kg"})
    comp = d[d[month].between(*post)].rename(columns={value: "value_usd", weight: "weight_kg"})
    r = unit_value_decomposition(base, comp, item=item)
    r["change_pct"] = r["change"] / r["uv_base"]
    for k in ("mix", "price", "interaction"):
        r[f"{k}_share_of_change"] = r[k] / r["change"]
    return r


def fx_decomposition(usd: pd.Series, kg: pd.Series, lcu_per_usd: pd.Series,
                     pre: tuple[str, str], post: tuple[str, str]) -> dict:
    """Split the change in USD per kg into local-currency per kg and the exchange rate (pre-registration Q3).

    With window totals, USD/kg = (sum LCU / sum kg) / R, where R = sum LCU / sum USD is the value-weighted
    average exchange rate over the window. Hence the identity
        dlog(USD/kg) = dlog(LCU/kg) - dlog(R)
    holds exactly. A fall in USD/kg caused only by a weaker local currency shows up as dlog(R) > 0 with
    dlog(LCU/kg) = 0.
    """
    idx = pd.to_datetime(usd.index)
    usd, kg, fx = (pd.Series(s.to_numpy(float), index=idx) for s in (usd, kg, lcu_per_usd))
    def window(a, b):
        w = (idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(b))
        u, k, lcu = usd[w].sum(), kg[w].sum(), (usd[w] * fx[w]).sum()
        return {"usd_per_kg": u / k, "lcu_per_kg": lcu / k, "rate": lcu / u}
    p0, p1 = window(*pre), window(*post)
    out = {f"{k}_pre": v for k, v in p0.items()} | {f"{k}_post": v for k, v in p1.items()}
    out["dlog_usd_per_kg"] = float(np.log(p1["usd_per_kg"] / p0["usd_per_kg"]))
    out["dlog_lcu_per_kg"] = float(np.log(p1["lcu_per_kg"] / p0["lcu_per_kg"]))
    out["dlog_rate"] = float(np.log(p1["rate"] / p0["rate"]))
    out["fx_share_of_change"] = -out["dlog_rate"] / out["dlog_usd_per_kg"] if out["dlog_usd_per_kg"] else np.nan
    return out
