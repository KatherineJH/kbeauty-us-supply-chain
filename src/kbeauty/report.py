"""Produce every table (reports/tables) and figure (reports/figures) used in the notebook and case study."""

import numpy as np
import pandas as pd

from . import (comparison, decision, decompose, documentation, events, forecast, hierarchy, mirror, ml, structure,
               synthesis, viz, warehouse)
from .numeric_env import run_label
from .paths import BACKTEST_FIRST_ORIGIN, POLICY_MONTH, POST_WINDOW, PRE_WINDOW, REPO, TABLES_DIR
from .viz.theme import COLORS

# Scenario parameters for the route comparison (not observed in public data; varied in notebook 06).
INLAND_SD = 2.0                      # standard deviation of the West-coast-to-East inland leg, days
CV_GRID = (0.3, 0.5, 0.9)            # coefficient of variation of daily demand
SERVICE_LEVELS = (0.90, 0.95, 0.99)  # cycle service level
INLAND_SD_GRID = (1.0, 2.0, 4.0)     # sensitivity range of the inland-leg standard deviation, days
INLAND_DAYS_GRID = tuple(range(6, 19))  # inland transit days, West port to East market
CLOSURE_DAYS = (7, 14, 30)           # LA/Long Beach closure scenarios, days
BASE = {"inland_sd_days": INLAND_SD, "cv_demand": 0.5, "service_level": 0.95, "closure_days": 14}

# Colour names used by the notebook builders, mapped to the design tokens (src/kbeauty/viz/theme.py).
INK, ACCENT, ACCENT2, WARN, MUTED = COLORS["ink"], COLORS["primary"], COLORS["faint"], COLORS["primary_dark"], COLORS["muted"]
ML_KO = viz.forecasting.ML_KO

SHIFT_METRICS = {
    "value_usd": "수입 금액",
    "measured_weight_kg": "수입 중량(해상+항공)",
    "vessel_weight_kg": "해상 중량",
    "air_weight_kg": "항공 중량",
    "vessel_usd_per_kg": "해상 kg당 단가",
    "air_usd_per_kg": "항공 kg당 단가",
}
REGION_KO = {"west_coast": "서부", "gulf_coast": "걸프", "east_coast": "동부"}
MODEL_KO = {"naive": "직전 값", "seasonal_naive": "작년 같은 달", "seasonal_naive_growth": "작년 같은 달×성장률",
            "ets_damped": "ETS(감쇠 추세)", "ets_seasonal": "ETS(계절)"}


def _table(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Write a result table. Floats are rounded to 12 decimals so that re-running produces byte-identical files
    (summation order can otherwise change the 17th significant digit)."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    floats = out.select_dtypes("float").columns
    out[floats] = out[floats].round(12)
    out.to_csv(TABLES_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")
    return df


def load_inputs() -> dict[str, pd.DataFrame]:
    out = {}
    for t in ("mart_monthly_summary", "fct_imports_hs10_mode_monthly", "mart_category_monthly",
              "mart_port_concentration_monthly", "mart_port_region_monthly", "mart_vessel_transit"):
        df = warehouse.load(t)
        for c in ("month", "filing_month"):
            if c in df:
                df[c] = pd.to_datetime(df[c])
        out[t] = df
    return out


def window_totals(summary: pd.DataFrame) -> pd.DataFrame:
    """12-month totals before and after the policy month and their change."""
    cols = ["value_usd", "vessel_value_usd", "air_value_usd", "vessel_weight_kg", "air_weight_kg", "measured_weight_kg"]
    pre = summary[summary.month.between(*PRE_WINDOW)][cols].sum()
    post = summary[summary.month.between(*POST_WINDOW)][cols].sum()
    t = pd.DataFrame({"metric": cols, "pre_12m": pre.values, "post_12m": post.values})
    t["change_pct"] = t.post_12m / t.pre_12m - 1
    extra = pd.DataFrame([
        {"metric": "vessel_usd_per_kg", "pre_12m": pre.vessel_value_usd / pre.vessel_weight_kg,
         "post_12m": post.vessel_value_usd / post.vessel_weight_kg},
        {"metric": "air_usd_per_kg", "pre_12m": pre.air_value_usd / pre.air_weight_kg,
         "post_12m": post.air_value_usd / post.air_weight_kg},
        {"metric": "air_share_weight", "pre_12m": pre.air_weight_kg / pre.measured_weight_kg,
         "post_12m": post.air_weight_kg / post.measured_weight_kg},
    ])
    extra["change_pct"] = extra.post_12m / extra.pre_12m - 1
    return pd.concat([t, extra], ignore_index=True)


def category_growth(cat: pd.DataFrame) -> pd.DataFrame:
    pre = cat[cat.month.between(*PRE_WINDOW)].groupby(["category", "category_ko"]).value_usd.sum()
    post = cat[cat.month.between(*POST_WINDOW)].groupby(["category", "category_ko"]).value_usd.sum()
    t = pd.DataFrame({"pre_12m": pre, "post_12m": post}).reset_index()
    t["change_pct"] = t.post_12m / t.pre_12m - 1
    t["post_share"] = t.post_12m / t.post_12m.sum()
    return t.sort_values("post_12m", ascending=False)


DM_PAIRS = [  # (name, model 1, method 1, model 2, method 2, horizon); negative statistic = model 1 better
    ("naive vs seasonal_naive (h1)", "naive", "base", "seasonal_naive", "base", 1),
    ("ets_seasonal(BU) vs seasonal_naive (h3)", "ets_seasonal", "bottom_up", "seasonal_naive", "base", 3),
    ("naive vs ets_seasonal(BU) (h1)", "naive", "base", "ets_seasonal", "bottom_up", 1),
    ("ml_selected(BU) vs naive (h1)", "ml_selected", "bottom_up", "naive", "base", 1),
    ("ml_selected(BU) vs ets_seasonal(BU) (h3)", "ml_selected", "bottom_up", "ets_seasonal", "bottom_up", 3),
    ("[post hoc] ml_lightgbm(BU) vs naive (h1)", "ml_lightgbm", "bottom_up", "naive", "base", 1),
    ("[post hoc] ml_xgboost(BU) vs naive (h1)", "ml_xgboost", "bottom_up", "naive", "base", 1),
]


def dm_tests(bt: pd.DataFrame) -> pd.DataFrame:
    """Diebold-Mariano tests on the total node (same pairs as notebook 05, section 6)."""
    def errors(model, method, h):
        d = bt[(bt.model == model) & (bt.method == method) & (bt.horizon == h) & (bt.node == "total")].sort_values("origin")
        return (d.forecast - d.actual).to_numpy()
    return pd.DataFrame([{"comparison": name, "horizon": h, **hierarchy.diebold_mariano(errors(m1, r1, h), errors(m2, r2, h), h)}
                         for name, m1, r1, m2, r2, h in DM_PAIRS])


def generate_all() -> dict[str, pd.DataFrame]:
    d = load_inputs()
    s = d["mart_monthly_summary"].sort_values("month")
    results = {
        "window_totals": _table(window_totals(s), "window_totals"),
        "unit_value_decomposition": _table(decompose.decompose_by_mode(d["fct_imports_hs10_mode_monthly"],
                                                                       PRE_WINDOW, POST_WINDOW),
                                           "unit_value_decomposition"),
        "policy_shifts": _table(events.policy_shift_table(s, SHIFT_METRICS, POLICY_MONTH), "policy_shifts"),
        "category_growth": _table(category_growth(d["mart_category_monthly"]), "category_growth"),
        "transit_by_region": _table(structure.transit_by_region(d["mart_vessel_transit"]), "transit_by_region"),
    }
    gap = structure.median_gap(d["mart_vessel_transit"], "east_coast", "west_coast")
    results["transit_gap"] = _table(pd.DataFrame([{"comparison": "east_coast - west_coast", **gap}]), "transit_gap")
    conc = d["mart_port_concentration_monthly"].sort_values("month")
    conc_tbl = pd.DataFrame([{"window": name, "hhi_vessel_mean": c.hhi_vessel.mean(), "la_lb_share_mean": c.la_lb_share.mean(),
                              "west_coast_share_mean": c.west_coast_share.mean(), "east_coast_share_mean": c.east_coast_share.mean()}
                             for name, c in (("pre_12m", conc[conc.month.between(*PRE_WINDOW)]),
                                             ("post_12m", conc[conc.month.between(*POST_WINDOW)]))])
    results["port_concentration"] = _table(conc_tbl, "port_concentration")

    series = s.set_index("month")["value_usd"]
    # First origin needs 24 months of history (seasonal ETS and YoY growth use two full years).
    first_origin = int(np.flatnonzero(series.index == pd.Timestamp(BACKTEST_FIRST_ORIGIN))[0])
    bt = forecast.backtest(series, forecast.MODELS, first_origin=first_origin)
    results["forecast_backtest"] = _table(bt, "forecast_backtest_detail")
    results["forecast_scores"] = _table(forecast.score(bt), "forecast_scores")

    cmp = comparison.run(warehouse.load("mart_country_panel_monthly"), POLICY_MONTH)
    for key in ("did", "shifts", "synth", "synth_weights"):
        results[f"comparison_{key}"] = _table(cmp[key], f"comparison_{key}")

    mir = mirror.monthly(warehouse.load("mart_mirror_monthly"))
    results["mirror_monthly"] = _table(mir.reset_index(), "mirror_monthly")
    results["mirror_shift"] = _table(pd.DataFrame([
        {"metric": col, **{k: v for k, v in mirror.shift_test(mir[col], POLICY_MONTH).items()}}
        for col in ("log_ratio_value", "log_ratio_weight")]), "mirror_shift")

    # Q3: exchange-rate share of the unit-value change, US side by mode and Korean export side.
    krw = warehouse.load("stg_fed_board__fx_monthly").query("currency == 'KRW'")
    krw = krw.assign(month=pd.to_datetime(krw["month"])).set_index("month")["lcu_per_usd"].reindex(s.month)
    ss = s.set_index("month")
    fx_rows = [{"series": "US imports, vessel", **decompose.fx_decomposition(ss.vessel_value_usd, ss.vessel_weight_kg, krw, PRE_WINDOW, POST_WINDOW)},
               {"series": "US imports, air", **decompose.fx_decomposition(ss.air_value_usd, ss.air_weight_kg, krw, PRE_WINDOW, POST_WINDOW)},
               {"series": "US imports, sea+air", **decompose.fx_decomposition(ss.vessel_value_usd + ss.air_value_usd,
                                                                               ss.measured_weight_kg, krw, PRE_WINDOW, POST_WINDOW)},
               {"series": "Korean exports (FOB)", **decompose.fx_decomposition(mir.kr_export_usd, mir.kr_export_kg,
                                                                               krw.reindex(mir.index), PRE_WINDOW, POST_WINDOW)}]
    results["fx_decomposition"] = _table(pd.DataFrame(fx_rows), "fx_decomposition")

    # Q4: mix vs within-item decomposition at different code granularities, Korean and US sides.
    kr = warehouse.load("stg_kcs__kr_exports_us")
    us = d["fct_imports_hs10_mode_monthly"].query("mode != 'other'").groupby(["month", "hs10"], as_index=False)[["value_usd", "weight_kg"]].sum()
    us["hs6"] = us["hs10"].str[:6]
    gran = [("Korea, HSK10", kr, "hsk10", "export_usd", "export_kg"),
            ("Korea, HS6", kr, "hs6", "export_usd", "export_kg"),
            ("US, HS10 (sea+air)", us, "hs10", "value_usd", "weight_kg"),
            ("US, HS6 (sea+air)", us, "hs6", "value_usd", "weight_kg")]
    results["granularity_decomposition"] = _table(pd.DataFrame([
        {"series": name, **decompose.window_decomposition(df, item, v, w, PRE_WINDOW, POST_WINDOW)}
        for name, df, item, v, w in gran]), "granularity_decomposition")

    # F1-F4: hierarchical backtest, machine-learning comparison, intervals, nowcast, structural break.
    bottom = hierarchy.bottom_series(d["fct_imports_hs10_mode_monthly"])
    h_origin = int(np.flatnonzero(bottom.index == pd.Timestamp(BACKTEST_FIRST_ORIGIN))[0])
    hbt = hierarchy.backtest(bottom, hierarchy.standard_models(POLICY_MONTH, krw), h_origin)
    mlr = ml.backtest(bottom, krw, h_origin, POLICY_MONTH)
    results["forecast_hierarchy_wape"] = _table(hierarchy.level_wape(pd.concat([hbt, mlr.forecasts], ignore_index=True)),
                                                "forecast_hierarchy_wape")
    results["forecast_dm_tests"] = _table(dm_tests(pd.concat([hbt, mlr.forecasts], ignore_index=True)), "forecast_dm_tests")
    results["forecast_ml_selection"] = _table(mlr.selection.assign(origin=lambda x: x.origin.dt.strftime("%Y-%m-%d")),
                                              "forecast_ml_selection")
    results["forecast_ml_importance"] = _table(mlr.importance, "forecast_ml_importance")
    results["forecast_conformal"] = _table(pd.concat([
        hierarchy.conformal_coverage(hbt.query("model == @m and method == @r")).assign(model=m, method=r)
        for m, r in (("naive", "base"), ("ets_seasonal", "ols"))], ignore_index=True), "forecast_conformal")
    us_total = (ss.vessel_value_usd + ss.air_value_usd)
    kr_total = mir.kr_export_usd.reindex(us_total.index)
    nbt = hierarchy.nowcast_backtest(us_total, kr_total, int(np.flatnonzero(us_total.index == pd.Timestamp(BACKTEST_FIRST_ORIGIN))[0]))
    results["forecast_nowcast"] = _table(hierarchy.wape(nbt, ["model"]), "forecast_nowcast")
    results["forecast_cusum"] = _table(pd.DataFrame([{"series": "log sea+air value", **hierarchy.cusum_break(np.log(us_total))}]),
                                       "forecast_cusum")

    # Q5, D1, D2: lead-time change, route inventory trade-off (scenario parameters below), port exposure.
    transit = d["mart_vessel_transit"]
    regions = ("west_coast", "gulf_coast", "east_coast")
    results["decision_q5_shift"] = _table(pd.DataFrame([decision.period_shift(transit, r, POLICY_MONTH) for r in regions]),
                                          "decision_q5_shift")
    stats = {r: decision.lead_time_stats(transit, r) for r in regions}
    results["decision_breakeven"] = _table(pd.DataFrame([
        {"cv_demand": cv, "service_level": sl, "inland_sd_days": INLAND_SD,
         "break_even_inland_days": decision.break_even_inland_days(stats["east_coast"], stats["west_coast"], INLAND_SD, cv, sl)}
        for cv in CV_GRID for sl in SERVICE_LEVELS]), "decision_breakeven")
    results["decision_port_disruption"] = _table(decision.port_disruption(
        d["mart_port_region_monthly"], d["mart_port_concentration_monthly"], POST_WINDOW, CLOSURE_DAYS),
        "decision_port_disruption")
    results["decision_route_sensitivity"] = _table(decision.route_sensitivity(
        stats["east_coast"], stats["west_coast"], INLAND_DAYS_GRID, INLAND_SD_GRID, CV_GRID, SERVICE_LEVELS),
        "decision_route_sensitivity")

    # Final layer: assumptions, claims with evidence levels, operating recommendations.
    ranges = {"inland_sd_days": INLAND_SD_GRID, "cv_demand": CV_GRID, "service_level": SERVICE_LEVELS,
              "closure_days": CLOSURE_DAYS}
    results["assumptions"] = _table(synthesis.assumptions(stats, BASE, ranges, results["decision_route_sensitivity"],
                                                          results["decision_port_disruption"]), "assumptions")
    results["claims"] = _table(synthesis.claims(results), "claims")
    results["decision_recommendations"] = _table(synthesis.recommendations(results, results["claims"]),
                                                 "decision_recommendations")

    # Figures (src/kbeauty/viz): each reads the tables above, so a chart shows the numbers in reports/tables.
    wt = results["window_totals"]
    V = viz
    V.timeseries.value_vs_weight(s, wt)
    V.timeseries.unit_values(s, wt, PRE_WINDOW, POST_WINDOW)
    V.timeseries.decomposition(results["unit_value_decomposition"])
    V.timeseries.air_weight(s, wt, PRE_WINDOW, POST_WINDOW)
    V.timeseries.port_concentration(conc, results["port_concentration"])
    V.timeseries.transit_by_region(results["transit_by_region"])
    V.timeseries.mirror(mir, results["granularity_decomposition"], POLICY_MONTH)
    V.comparison.policy_shifts(results["policy_shifts"])
    V.comparison.category_growth(results["category_growth"])
    V.comparison.country_shifts(cmp["shifts"], cmp["did"])
    V.comparison.event_study(cmp["event_path"], cmp["event_band"], POLICY_MONTH, comparison.METRIC_KO)
    V.comparison.synthetic(cmp["synth_path"], cmp["synth"], POLICY_MONTH, comparison.METRIC_KO)
    V.forecasting.backtest(results["forecast_scores"])
    V.forecasting.hierarchy(results["forecast_hierarchy_wape"])
    V.forecasting.ml_comparison(results["forecast_hierarchy_wape"], run_label())
    sens = results["decision_route_sensitivity"]
    V.decision.route_breakeven(sens[(sens.cv_demand == BASE["cv_demand"]) & (sens.service_level == BASE["service_level"])],
                               BASE["cv_demand"], BASE["service_level"])
    V.decision.route_sensitivity(sens, BASE["service_level"])
    V.lineage.lineage(documentation.lineage(), results, documentation.seeds())

    # Reference documents generated from the warehouse, dbt manifest and the tables above.
    documentation.data_dictionary().to_csv(REPO / "reports" / "data_dictionary.csv", index=False, encoding="utf-8-sig")
    (REPO / "reports" / "model_card.md").write_bytes(documentation.model_card(results).encode("utf-8"))
    return results


