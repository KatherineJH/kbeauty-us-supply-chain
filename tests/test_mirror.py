import gzip

import numpy as np
import pandas as pd
import pytest

from kbeauty.mirror import hs6_gap_change, lag_correlation, monthly, raw_exports, shift_test


def mirror_frame():
    months = pd.date_range("2025-01-01", periods=4, freq="MS")
    rows = []
    for i, m in enumerate(months):
        rows.append({"month": m, "hs6": "330499", "kr_export_usd": 100.0, "kr_export_kg": 10.0,
                     "us_import_usd": 80.0 + 10 * i, "us_vessel_usd": 60.0, "us_air_usd": 20.0 + 10 * i,
                     "us_vessel_kg": 6.0, "us_air_kg": 2.0, "us_published": True})
    rows.append({**rows[-1], "month": months[-1] + pd.offsets.MonthBegin(), "us_published": False})
    return pd.DataFrame(rows)


def test_monthly_uses_only_months_published_on_both_sides():
    m = monthly(mirror_frame())
    assert len(m) == 4
    assert m["ratio_value"].tolist() == pytest.approx([0.8, 0.9, 1.0, 1.1])
    assert m["ratio_weight"].iloc[0] == pytest.approx(0.8)  # (6 + 2) / 10


def test_lag_correlation_finds_a_one_month_delay():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=30, freq="MS")
    kr = pd.Series(np.exp(rng.normal(0, 0.2, 30).cumsum()), index=idx)
    us = kr.shift(1).bfill()  # US sees Korean shipments one month later
    t = lag_correlation(us, kr).set_index("lag_months")["corr"]
    assert t.idxmax() == 1 and t[1] == pytest.approx(1.0)


def test_shift_test_detects_a_rise_in_the_ratio():
    idx = pd.date_range("2025-01-01", periods=20, freq="MS")
    rng = np.random.default_rng(1)
    y = pd.Series(np.r_[np.full(8, -0.3), np.full(12, 0.0)] + rng.normal(0, 0.02, 20), index=idx)
    r = shift_test(y, "2025-09-01")
    assert r["shift"] == pytest.approx(0.3, abs=0.03) and r["p_value"] < 0.01
    assert r["largest_break"] == pd.Timestamp("2025-09-01")


def test_hs6_gap_change_sign():
    d = mirror_frame()
    out = hs6_gap_change(d, ("2025-01-01", "2025-02-01"), ("2025-03-01", "2025-04-01"))
    assert out.loc["330499", "gap_change_usd"] == pytest.approx((100 + 110 - 200) - (80 + 90 - 200))


def test_raw_exports_reads_archived_xml(tmp_path):
    xml = ("<response><body><items>"
           "<item><expDlr>500</expDlr><expWgt>5</expWgt><hsCd>3401110000</hsCd><year>2025.01</year></item>"
           "<item><expDlr>900</expDlr><expWgt>9</expWgt><hsCd>3304991000</hsCd><year>2025.01</year></item>"
           "<item><expDlr>1</expDlr><expWgt>1</expWgt><hsCd>-</hsCd><year>total</year></item>"
           "</items></body></response>").encode()
    (tmp_path / "US_202501.xml.gz").write_bytes(gzip.compress(xml))
    out = raw_exports(("3401",), raw_dir=tmp_path)
    assert out["export_usd"].tolist() == [500] and out["hsk10"].tolist() == ["3401110000"]
