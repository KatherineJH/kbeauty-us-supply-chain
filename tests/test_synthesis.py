import numpy as np
import pandas as pd
import pytest

from kbeauty import decision, synthesis
from kbeauty.paths import TABLES_DIR

EAST = {"mean": 27.5, "sd": 2.2}
WEST = {"mean": 13.1, "sd": 1.7}
LEVELS = {"observed", "estimated", "scenario", "hypothesis", "not_measured"}


def test_preferred_route_switches_at_the_break_even_inland_days():
    g = decision.route_sensitivity(EAST, WEST, range(5, 21), (1.0, 4.0), (0.3, 0.9), (0.9, 0.99))
    for _, row in g.iterrows():
        expected = "west" if row.inland_days < row.break_even_inland_days else "east"
        assert row.preferred_route == expected
    # more variable inland legs make West routing worse, so the break-even comes earlier
    be = g.groupby("inland_sd_days").break_even_inland_days.max()
    assert be[4.0] < be[1.0]


@pytest.fixture(scope="module")
def results():
    names = ["window_totals", "unit_value_decomposition", "comparison_did", "mirror_shift", "granularity_decomposition",
             "fx_decomposition", "decision_q5_shift", "forecast_hierarchy_wape", "forecast_conformal",
             "forecast_nowcast", "decision_port_disruption", "decision_route_sensitivity", "forecast_ml_selection"]
    return {n: pd.read_csv(TABLES_DIR / f"{n}.csv", encoding="utf-8-sig") for n in names}


def test_claims_carry_valid_levels_and_numbers_from_their_tables(results):
    c = synthesis.claims(results)
    assert c.claim_id.is_unique and set(c.evidence_level) <= LEVELS
    assert not c.causal_claim_allowed.any()  # no conclusion in this project is causal
    wt = results["window_totals"].set_index("metric").change_pct
    assert f"{wt['value_usd'] * 100:+.1f}%" in c.set_index("claim_id").loc["C01", "value"]
    for tables in c.source_table:
        for name in tables.split(", "):
            assert (TABLES_DIR / f"{name}.csv").exists(), name


def test_recommendations_cite_existing_claims_and_inherit_their_levels(results):
    c = synthesis.claims(results)
    rec = synthesis.recommendations(results, c)
    level = c.set_index("claim_id").evidence_level
    for _, row in rec.iterrows():
        ids = [x.strip() for x in row.evidence.split(",")]
        assert set(ids) <= set(level.index)
        assert row.evidence_level == ", ".join(sorted({level[i] for i in ids}))
        assert row.limitation and row.uncertainty


def test_every_scenario_assumption_has_a_range_or_a_stated_effect():
    stats = {"east_coast": {**EAST, "n": 1, "p10": 25.0, "p90": 30.0}, "west_coast": {**WEST, "n": 1, "p10": 11.0, "p90": 15.0}}
    sens = decision.route_sensitivity(EAST, WEST, range(6, 19), (1.0, 2.0, 4.0), (0.3, 0.5, 0.9), (0.9, 0.95, 0.99))
    port = pd.DataFrame({"closure_days": [7, 14, 30], "exposed_value_usd": [1e7, 2e7, 4e7]})
    base = {"inland_sd_days": 2.0, "cv_demand": 0.5, "service_level": 0.95, "closure_days": 14}
    ranges = {"inland_sd_days": (1.0, 2.0, 4.0), "cv_demand": (0.3, 0.5, 0.9), "service_level": (0.9, 0.95, 0.99),
              "closure_days": (7, 14, 30)}
    a = synthesis.assumptions(stats, base, ranges, sens, port)
    assert set(a.evidence_level) <= LEVELS
    scen = a[a.evidence_level == "scenario"]
    assert (scen.sensitivity_low.notna() | scen.effect_on_result.str.len().gt(0)).all()
    ranged = scen.dropna(subset=["base_value"])
    assert ((ranged.sensitivity_low <= ranged.base_value) & (ranged.base_value <= ranged.sensitivity_high)).all()


def test_claims_accept_in_memory_types(results):
    """generate_all passes DataFrames whose date columns are Timestamps, not the strings read back from CSV."""
    as_built = {k: v.assign(**{c: pd.to_datetime(v[c]) for c in v.columns
                               if c in ("largest_break", "break_month", "origin")}) for k, v in results.items()}
    a, b = synthesis.claims(results), synthesis.claims(as_built)
    pd.testing.assert_frame_equal(a, b)
