import numpy as np
import pandas as pd
import pytest

from kbeauty.decompose import fx_decomposition, unit_value_decomposition, window_decomposition


def test_fx_decomposition_identity_and_pure_currency_case():
    idx = pd.date_range("2025-01-01", periods=4, freq="MS")
    # Exporter keeps 30,000 won per kg; the won weakens from 1,200 to 1,500 per dollar in the second window.
    rate = pd.Series([1200.0, 1200.0, 1500.0, 1500.0], index=idx)
    kg = pd.Series([10.0, 20.0, 10.0, 30.0], index=idx)
    usd = kg * 30000 / rate
    r = fx_decomposition(usd, kg, rate, ("2025-01-01", "2025-02-01"), ("2025-03-01", "2025-04-01"))
    assert r["dlog_usd_per_kg"] == pytest.approx(r["dlog_lcu_per_kg"] - r["dlog_rate"])
    assert r["dlog_lcu_per_kg"] == pytest.approx(0.0, abs=1e-12)
    assert r["dlog_usd_per_kg"] == pytest.approx(np.log(1200 / 1500))
    assert r["fx_share_of_change"] == pytest.approx(1.0)


def test_window_decomposition_matches_direct_call():
    df = pd.DataFrame({"month": ["2025-01-01", "2025-01-01", "2025-03-01", "2025-03-01"],
                       "code": ["A", "B", "A", "B"], "usd": [100, 300, 300, 300], "kg": [10, 10, 30, 10]})
    r = window_decomposition(df, "code", "usd", "kg", ("2025-01-01", "2025-01-01"), ("2025-03-01", "2025-03-01"))
    assert r["price"] == pytest.approx(0) and r["mix"] == pytest.approx(r["change"])


def frame(rows):
    return pd.DataFrame(rows, columns=["hs10", "value_usd", "weight_kg"])


def test_parts_add_up_to_total_change():
    base = frame([("A", 100, 10), ("B", 300, 10)])
    comp = frame([("A", 160, 20), ("B", 250, 5)])
    r = unit_value_decomposition(base, comp)
    assert r["mix"] + r["price"] + r["interaction"] == pytest.approx(r["change"])


def test_pure_mix_change_has_no_price_effect():
    # Item prices unchanged (A=10, B=30 USD/kg); only the weight shares move toward the cheap item.
    base = frame([("A", 100, 10), ("B", 300, 10)])
    comp = frame([("A", 300, 30), ("B", 300, 10)])
    r = unit_value_decomposition(base, comp)
    assert r["price"] == pytest.approx(0)
    assert r["interaction"] == pytest.approx(0)
    assert r["mix"] == pytest.approx(r["change"]) and r["change"] < 0


def test_pure_price_change_has_no_mix_effect():
    base = frame([("A", 100, 10), ("B", 300, 10)])
    comp = frame([("A", 50, 10), ("B", 150, 10)])
    r = unit_value_decomposition(base, comp)
    assert r["mix"] == pytest.approx(0)
    assert r["price"] == pytest.approx(r["change"]) == pytest.approx(-10)


def test_item_entering_only_in_comparison_counts_as_mix():
    base = frame([("A", 100, 10)])
    comp = frame([("A", 100, 10), ("C", 50, 10)])
    r = unit_value_decomposition(base, comp)
    assert r["price"] == pytest.approx(0)
    assert r["mix"] == pytest.approx(r["change"])
