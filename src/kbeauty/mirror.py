"""Mirror statistics: Korean exports to the US (Korea Customs Service) vs US imports from Korea (Census).

Pre-registration Q2. De minimis parcels (<= $800) were not recorded in Census import statistics until the
suspension on 2025-08-29, while Korean export declarations can include them. If the post-August change in US
air imports is partly parcels entering US statistics, the ratio US imports / Korean exports should rise after
the policy. If the ratio does not move, the change is more consistent with a real shift in how goods are moved.

Both sides exclude international freight and insurance (Korea: FOB; US: customs value), but they still differ in
timing (sea transit spans month ends), valuation details and coverage, so the level of the ratio is not expected
to be 1. The test is about its change.
"""

import gzip
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import events
from .paths import RAW_DIR


def monthly(mirror: pd.DataFrame) -> pd.DataFrame:
    """Months where both sides are published: totals and the US/KR ratios (value; weight uses US sea+air)."""
    m = mirror[mirror["us_published"]].groupby("month")[
        ["kr_export_usd", "kr_export_kg", "us_import_usd", "us_vessel_usd", "us_air_usd", "us_vessel_kg", "us_air_kg"]
    ].sum()
    m.index = pd.to_datetime(m.index)
    m["us_sea_air_kg"] = m["us_vessel_kg"] + m["us_air_kg"]
    m["ratio_value"] = m["us_import_usd"] / m["kr_export_usd"]
    m["ratio_weight"] = m["us_sea_air_kg"] / m["kr_export_kg"]
    m["log_ratio_value"] = np.log(m["ratio_value"])
    m["log_ratio_weight"] = np.log(m["ratio_weight"])
    return m.sort_index()


def lag_correlation(us: pd.Series, kr: pd.Series, lags=(0, 1, 2)) -> pd.DataFrame:
    """Correlation of month-on-month log changes, US month t against Korea month t - lag.
    Sea freight shipped late in a month reaches the US the next month, so lag 1 may fit better than lag 0."""
    du, dk = np.log(us).diff(), np.log(kr).diff()
    return pd.DataFrame([{"lag_months": k, "corr": du.corr(dk.shift(k)), "n": int((du.notna() & dk.shift(k).notna()).sum())}
                         for k in lags])


def shift_test(log_ratio: pd.Series, break_month) -> dict:
    """Pre-registered test: level shift of the log ratio after the break (OLS + HAC), with a time placebo."""
    est = events.level_shift(log_ratio, break_month)
    plc = events.placebo_rank(log_ratio, break_month)
    return {**est, "pct_change_in_ratio": float(np.expm1(est["shift"])),
            "placebo_p": plc["placebo_p"], "n_candidates": plc["n_candidates"], "largest_break": plc["largest_break"]}


def hs6_gap_change(mirror: pd.DataFrame, pre: tuple[str, str], post: tuple[str, str]) -> pd.DataFrame:
    """Which HS6 lines drive the change in the US-minus-Korea gap between two 12-month windows."""
    d = mirror[mirror["us_published"]].assign(month=lambda x: pd.to_datetime(x["month"]))
    def window(a, b):
        w = d[d["month"].between(a, b)].groupby("hs6")[["kr_export_usd", "us_import_usd"]].sum()
        return w.assign(ratio=w.us_import_usd / w.kr_export_usd)
    p0, p1 = window(*pre), window(*post)
    out = pd.DataFrame({"kr_pre": p0.kr_export_usd, "us_pre": p0.us_import_usd, "ratio_pre": p0.ratio,
                        "kr_post": p1.kr_export_usd, "us_post": p1.us_import_usd, "ratio_post": p1.ratio})
    out["gap_change_usd"] = (out.us_post - out.kr_post) - (out.us_pre - out.kr_pre)
    return out.sort_values("kr_post", ascending=False)


def raw_exports(prefixes: tuple[str, ...], raw_dir: Path = RAW_DIR / "kcs") -> pd.DataFrame:
    """Korean exports to the US for any HS prefix, read directly from the archived API responses.
    Used to compare product scopes (e.g. HS 33 only vs HS 33 + soap 3401) without new API calls."""
    rows = []
    for path in sorted(raw_dir.glob("US_*.xml.gz")):
        ym = path.stem.split("_")[1].split(".")[0]
        text = gzip.decompress(path.read_bytes()).decode("utf-8", "replace")
        for item in re.findall(r"<item>(.*?)</item>", text, flags=re.S):
            f = dict(re.findall(r"<(\w+)>([^<]*)</\1>", item))
            hs = f.get("hsCd", "")
            if hs.startswith(prefixes) and f.get("year") == f"{ym[:4]}.{ym[4:]}":
                rows.append({"month": pd.Timestamp(f"{ym[:4]}-{ym[4:]}-01"), "hsk10": hs,
                             "export_usd": int(f["expDlr"]), "export_kg": int(f["expWgt"])})
    return pd.DataFrame(rows)
