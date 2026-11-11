"""Forecast evaluation: the baseline, the operating rule and the rest, kept in three visual tiers."""

import numpy as np
import pandas as pd

from . import annotations as an
from .export import save
from .theme import COLORS, TEXT

SRC_BT = "U.S. Census Bureau 수입 통계로 계산한 rolling-origin 백테스트(기준 시점 2025.12~2026.07)"
MODEL_KO = {"naive": "직전 값", "seasonal_naive": "작년 같은 달", "seasonal_naive_growth": "작년 같은 달 × 성장률",
            "ets_damped": "ETS(감쇠 추세)", "ets_seasonal": "ETS(계절)", "sarimax_fx_policy": "SARIMAX(환율·정책)"}
ML_KO = {"ml_lightgbm": "LightGBM", "ml_xgboost": "XGBoost", "ml_gradient_boosting": "Gradient Boosting",
         "ml_random_forest": "Random Forest", "ml_extra_trees": "Extra Trees", "ml_ridge": "Ridge",
         "ml_elastic_net": "Elastic Net", "ml_knn": "kNN", "ml_mlp": "MLP", "ml_selected": "ML 검증 선택"}


def backtest(sc: pd.DataFrame):
    """Total-series WAPE by horizon. Operating rules in primary, the 'last year' baseline dark, the rest faint."""
    fig, ax = an.figure("wide")
    tiers = {"naive": (COLORS["primary"], 2.8, 700), "ets_seasonal": (COLORS["primary"], 2.8, 700),
             "seasonal_naive": (COLORS["ink"], 2.0, 600)}
    items = []
    for m, g in sc.groupby("model"):
        g = g.sort_values("horizon")
        color, lw, wgt = tiers.get(m, (COLORS["faint"], 1.4, 400))
        ls = (0, (4, 2)) if m == "seasonal_naive" else "-"
        ax.plot(g.horizon, g.wape, color=color, lw=lw, ls=ls, marker="o", ms=5 if m in tiers else 3.5, zorder=3 if m in tiers else 2)
        items.append((3, g.wape.iloc[-1], MODEL_KO[m], COLORS["primary_dark"] if color == COLORS["primary"] else
                      COLORS["ink"] if m in tiers else COLORS["muted"], wgt))
    ax.set_xticks([1, 2, 3], ["1개월 앞", "2개월 앞", "3개월 앞"])
    ax.set_xlim(0.85, 3.75)
    ax.set_ylabel("WAPE (낮을수록 좋음)")
    an.end_labels(ax, items, min_gap_pt=TEXT["label"] * 1.25)
    best1 = sc[sc.horizon == 1].sort_values("wape").iloc[0]
    best3 = sc[sc.horizon == 3].sort_values("wape").iloc[0]
    an.header(fig, f"가까운 달은 {MODEL_KO[best1.model]}, 먼 달은 {MODEL_KO[best3.model]}의 오차가 가장 작았다",
              f"전체 수입 금액(모든 운송수단, 고정 예측의 기준 백테스트). 1개월 앞 {best1.wape:.3f}, 3개월 앞 {best3.wape:.3f}. 점선 = '작년 같은 달'", "estimated")
    an.footer(fig, SRC_BT, "공개 수입통계의 예측이며 기업 수요예측이 아님")
    return save(fig, "07_forecast_backtest")


def _wape(t, model, method, h):
    return float(t[(t.model == model) & (t.method == method) & (t.horizon == h)].wape.iloc[0])


def hierarchy(lw: pd.DataFrame):
    t = lw[lw.level == "total"]
    rows = [("seasonal_naive", "base", "작년 같은 달"), ("sarimax_fx_policy", "base", "SARIMAX(환율·정책)"),
            ("naive", "base", "직전 값"), ("ets_seasonal", "bottom_up", "계절 ETS, 품목·운송수단별 합")]
    rule = {1: "naive", 3: "ets_seasonal"}
    fig, axes = an.figure("wide", 1, 2, sharey=True)
    for ax, h in zip(axes, (1, 3)):
        vals = [_wape(t, m, r, h) for m, r, _ in rows]
        colors = [COLORS["primary"] if m == rule[h] else COLORS["ink"] if m == "seasonal_naive" else COLORS["faint"]
                  for m, _, _ in rows]
        ax.barh(range(len(rows)), vals, color=colors, height=0.6)
        for i, v in enumerate(vals):
            an.note(ax, v, i, f" {v:.3f}", ha="left", size=TEXT["label"],
                    weight=700 if rows[i][0] == rule[h] else 400,
                    color=COLORS["primary_dark"] if rows[i][0] == rule[h] else COLORS["ink"])
        ax.set_title(f"{h}개월 앞")
        ax.set_xlim(0, max(vals) * 1.25)
        an.clean_axis(ax, "x")
        an.hide_y(ax)
    axes[0].set_yticks(range(len(rows)), [r[2] for r in rows], fontsize=TEXT["label"])
    axes[0].invert_yaxis()
    for ax in axes:
        ax.set_xlabel("합계 WAPE (낮을수록 좋음)")
    an.header(fig, "1개월 앞은 직전 값, 3개월 앞은 품목·운송수단별 계절 ETS 합이 가장 정확했다",
              "15개 노드 계층 예측의 합계 노드 오차. 노란 막대 = 해당 기간의 운영 규칙, 검은 막대 = '작년 같은 달'", "estimated")
    an.footer(fig, SRC_BT, "3개월 앞 우위 DM p 0.001(n 6), 1개월 앞 순위는 미확정(p 0.17)")
    return save(fig, "13_forecast_hierarchy")


def ml_comparison(lw: pd.DataFrame, env: str):
    t = lw[lw.level == "total"]
    ml_models = [m for m in t.model.unique() if m.startswith("ml_") and m != "ml_selected"]
    fig, axes = an.figure("wide", 1, 2)
    for ax, h in zip(axes, (1, 3)):
        best_ml = min(ml_models, key=lambda m: _wape(t, m, "bottom_up", h))
        rule = ("naive", "base") if h == 1 else ("ets_seasonal", "bottom_up")   # the operating rule per horizon
        rows = [("naive", "base", "직전 값(기준선)", COLORS["ink"] if rule[0] == "naive" else COLORS["muted"]),
                ("ets_seasonal", "bottom_up", "계절 ETS 합", COLORS["ink"] if rule[0] == "ets_seasonal" else COLORS["muted"]),
                ("ml_selected", "bottom_up", "ML 검증 선택", COLORS["primary"]),
                (best_ml, "bottom_up", f"{ML_KO[best_ml]} (사후 최선)", COLORS["faint"])]
        rest = sorted((m for m in ml_models if m != best_ml), key=lambda m: _wape(t, m, "bottom_up", h))
        y = 0
        for m, r, label, color in rows:
            v = _wape(t, m, r, h)
            ax.barh(y, v, color=color, height=0.62, hatch="///" if "사후" in label else None,
                    edgecolor=COLORS["muted"] if "사후" in label else color, linewidth=0)
            an.note(ax, v, y, f" {v:.3f}", ha="left", size=TEXT["label"], weight=700 if m == "ml_selected" else 400, box=True,
                    color=COLORS["primary_dark"] if m == "ml_selected" else COLORS["ink"])
            is_rule = (m, r) == rule
            ax.text(-0.004, y, (label.split("(")[0] + " · 운영 규칙") if is_rule else label, ha="right", va="center", fontsize=TEXT["label"],
                    color=COLORS["ink"], transform=ax.get_yaxis_transform(),
                    fontweight=600 if is_rule or m == "ml_selected" else 400)
            y += 1
        shown = max(_wape(t, m, r, h) for m, r, _, _ in rows)
        limit = shown * 1.55       # an outlier (e.g. the MLP) would otherwise squash every bar
        far = []
        for m in rest:
            v = _wape(t, m, "bottom_up", h)
            if v > limit:
                far.append((m, v))
                ax.plot([limit * 0.985], [y], ">", ms=7, color=COLORS["muted"])
            else:
                ax.plot([v], [y], "|", ms=12, mew=2, color=COLORS["faint"])
        near = [_wape(t, m, "bottom_up", h) for m in rest if _wape(t, m, "bottom_up", h) <= limit]
        if far:   # the off-axis models get their value next to the arrow; the count goes left of the ticks
            an.note(ax, limit * 0.005, y, f"나머지 ML {len(rest)}종", ha="left", color=COLORS["muted"])
            an.note(ax, limit * 0.70, y - 0.45, " · ".join(f"{ML_KO[m]} {v:.3f}" for m, v in far) + " →",
                    ha="left", color=COLORS["muted"])
        else:
            an.note(ax, max(near), y, f"  나머지 ML {len(rest)}종", ha="left", color=COLORS["muted"])
        ax.axvline(_wape(t, *rule, h), color=COLORS["ink"], lw=0.8, ls=(0, (3, 2)), zorder=0.5)
        ax.set_yticks([])
        ax.set_ylim(y + 0.6, -0.6)
        ax.set_xlim(0, limit)
        ax.set_title(f"{h}개월 앞 합계 WAPE")
        an.clean_axis(ax, "x")
        an.hide_y(ax)
    sel1, nv1 = _wape(t, "ml_selected", "bottom_up", 1), _wape(t, "naive", "base", 1)
    an.header(fig, "검증으로 고른 ML은 운영 규칙보다 나은 것을 입증하지 못했다",
              f"{env} · 1개월 앞 {sel1:.3f} vs {nv1:.3f} · 빗금 = 사후 최선(판정 제외)", "estimated")
    an.footer(fig, SRC_BT, "점선 = 운영 규칙 · ML vs 운영 규칙 DM p > 0.1 · 규칙 유지, ML 보류")
    return save(fig, "15_ml_comparison")
