import pandas as pd

from kbeauty import documentation
from kbeauty.paths import TABLES_DIR

PLATFORM_LINUX = TABLES_DIR.parent / "platform" / "linux"


def tables(*names):
    return {n: pd.read_csv(TABLES_DIR / f"{n}.csv", encoding="utf-8-sig") for n in names}


def test_model_card_numbers_come_from_the_canonical_reference():
    """The card quotes the committed pinned-Linux reference, even when run_all passes internal results that are not
    saved tables (forecast_backtest); it never falls back to a stub or to this environment's tables."""
    r = tables("forecast_hierarchy_wape", "forecast_dm_tests", "forecast_ml_importance", "forecast_ml_selection",
               "forecast_conformal")
    r["forecast_backtest"] = pd.DataFrame()
    card = documentation.model_card(r)
    assert "공식 기준(`reports/platform/linux/`)이 없어" not in card
    lw = pd.read_csv(PLATFORM_LINUX / "forecast_hierarchy_wape.csv", encoding="utf-8-sig")
    for model, method, h in (("naive", "base", 1), ("ets_seasonal", "bottom_up", 3), ("ml_selected", "bottom_up", 1)):
        w = lw[(lw.model == model) & (lw.method == method) & (lw.horizon == h) & (lw.level == "total")].wape.iloc[0]
        assert f"| {w:.3f}" in card or f"{w:.3f}," in card or f" {w:.3f} " in card, (model, h)
    assert "공개 수입통계의 예측" in card and "보류" in card and "Linux 고정 환경(공식)" in card
    assert card.count("_frozen_") >= 2  # both frozen files listed with their hash status


def test_lineage_edges_point_to_real_models():
    edges = documentation.lineage()
    marts = set(edges.child)
    assert len(marts) == 9 and all(c.startswith(("mart_", "dim_", "fct_")) for c in marts)
    assert set(documentation.seeds()) == {"country_currency", "hs10_category", "peer_countries", "policy_events", "state_region"}

