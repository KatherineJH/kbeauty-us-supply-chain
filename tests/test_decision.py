import numpy as np
import pandas as pd
import pytest

from kbeauty.decision import break_even_inland_days, period_shift, route_comparison, safety_stock_days


def test_safety_stock_formula():
    # z(0.95) = 1.645; L = 25, CV = 0.4, sigma_L = 3 -> 1.645 * sqrt(25 * 0.16 + 9) = 1.645 * sqrt(13)
    assert safety_stock_days(25, 3, 0.4, 0.95) == pytest.approx(1.6449 * np.sqrt(13), rel=1e-3)
    assert safety_stock_days(25, 0, 0.0, 0.95) == pytest.approx(0)


def test_break_even_is_where_routes_tie():
    east, west = {"mean": 28.0, "sd": 2.0}, {"mean": 13.0, "sd": 1.5}
    be = break_even_inland_days(east, west, inland_sd=2.0, cv=0.5, service_level=0.95)
    t = route_comparison(east, west, [be], 2.0, [0.5], [0.95]).iloc[0]
    assert t.west_minus_east_days == pytest.approx(0, abs=1e-6)
    assert 0 < be < 15  # cheaper ocean leg is partly eaten by the extra inland variability


def test_route_gap_increases_with_inland_days():
    east, west = {"mean": 28.0, "sd": 2.0}, {"mean": 13.0, "sd": 1.5}
    t = route_comparison(east, west, [5, 10, 15], 2.0, [0.5], [0.95])
    assert t.west_minus_east_days.is_monotonic_increasing


def test_period_shift_detects_a_known_increase():
    rng = np.random.default_rng(0)
    months = pd.date_range("2024-06-01", periods=24, freq="MS")
    rows = [{"region": "west_coast", "filing_month": m,
             "departure_to_filing_days": 13 + (3 if m >= pd.Timestamp("2025-09-01") else 0) + rng.integers(-1, 2)}
            for m in months for _ in range(10)]
    r = period_shift(pd.DataFrame(rows), "west_coast", "2025-09-01")
    assert r["diff"] == pytest.approx(3) and r["changed"]
