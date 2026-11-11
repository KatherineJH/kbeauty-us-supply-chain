"""Estimated effects: policy shifts, category growth, the nine-country comparison, event study, synthetic control."""

import numpy as np
import pandas as pd

from . import annotations as an
from .export import save
from .formats import month_axis, pct, percent_axis
from .theme import COLORS, TEXT

SRC_CENSUS = "U.S. Census Bureau 수입 통계(한국산)"
SRC_PEERS = "U.S. Census Bureau 수입 통계(한국 + 비교 8개국) · Federal Reserve Board 환율(G.5)"
COUNTRY_KO = {"Korea": "한국", "Canada": "캐나다", "China": "중국", "France": "프랑스", "Germany": "독일",
              "Italy": "이탈리아", "Japan": "일본", "Mexico": "멕시코", "United Kingdom": "영국"}
CMP_ORDER = ["value_usd", "measured_weight_kg", "air_weight_kg", "air_usd_per_kg", "vessel_usd_per_kg"]


def policy_shifts(shifts: pd.DataFrame):
    d = shifts.reset_index(drop=True)
    fig, ax = an.figure("wide")
    y = np.arange(len(d))
    for i, r in d.iterrows():
        excl = not (r.ci_low <= 0 <= r.ci_high)
        ax.hlines(i, r.ci_low, r.ci_high, color=COLORS["ink"] if excl else COLORS["muted"], lw=2.2)
        ax.plot(r["shift"], i, "o", ms=8, color=COLORS["primary"] if excl else COLORS["paper"],
                markeredgecolor=COLORS["ink"] if excl else COLORS["muted"], markeredgewidth=1.4)
        an.note(ax, 1.02, i, f"{r.placebo_p:.2f}", coords=("axes fraction", "data"), ha="left",
                color=COLORS["ink"], size=TEXT["label"])
    an.note(ax, 1.02, -0.9, "시점 위약 p", coords=("axes fraction", "data"), ha="left", color=COLORS["muted"])
    ax.axvline(0, color=COLORS["ink"], lw=0.8)
    ax.set_yticks(y, d.label, fontsize=TEXT["label"])
    ax.set_ylim(len(d) - 0.4, -1.3)
    ax.set_xlabel("전년 대비 로그 성장률의 변화(정책 후 − 전)와 95% HAC 신뢰구간")
    an.clean_axis(ax, "x")
    an.hide_y(ax)
    an.header(fig, "항공 중량과 항공 단가가 가장 크게 움직였지만, 시점 위약 검정을 통과한 지표는 없다",
              "채운 점 = 95% 구간이 0을 포함하지 않음. 오른쪽 열 = 다른 달을 정책 시점으로 두었을 때의 순위 p", "estimated")
    an.footer(fig, SRC_CENSUS, "Newey-West HAC, 정책 전 8 · 후 12개월")
    return save(fig, "05_policy_shifts")


def category_growth(cat: pd.DataFrame):
    d = cat.assign(contrib=cat.post_12m - cat.pre_12m).sort_values("contrib")
    total = (cat.post_12m - cat.pre_12m).sum()
    fig, ax = an.figure("wide")
    y = np.arange(len(d))
    colors = [COLORS["primary"] if v == d.contrib.max() else COLORS["ink"] if v > 0 else COLORS["muted"] for v in d.contrib]
    ax.barh(y, d.contrib / 1e6, color=colors, height=0.62)
    for i, r in enumerate(d.itertuples()):
        x = r.contrib / 1e6
        an.note(ax, max(x, 0), i, f"  {x:+,.0f}백만 달러 · 성장 {pct(r.change_pct, 0)} · 비중 {r.post_share * 100:.0f}%",
                ha="left", size=TEXT["note"] + 0.5, color=COLORS["ink"])
    ax.axvline(0, color=COLORS["ink"], lw=0.8)
    ax.set_yticks(y, d.category_ko, fontsize=TEXT["label"])
    ax.set_xlim(min(d.contrib.min() / 1e6 * 1.6, -30), d.contrib.max() / 1e6 * 1.9)
    ax.set_xlabel("12개월 수입 금액 증감 (백만 달러)")
    an.clean_axis(ax, "x")
    an.hide_y(ax)
    top = d.iloc[-1]
    an.header(fig, f"늘어난 수입 금액의 {top.contrib / total * 100:.0f}%를 {top.category_ko}가 만들었고 메이크업은 줄었다",
              f"정책 전후 12개월 품목별 수입 금액 증감(전체 {total / 1e6:+,.0f}백만 달러)", "observed")
    an.footer(fig, SRC_CENSUS)
    return save(fig, "08_category_growth")


def country_shifts(shifts: pd.DataFrame, did_tbl: pd.DataFrame):
    p = did_tbl.set_index("metric")
    fig, ax = an.figure("wide")
    k_prev = None
    for i, m in enumerate(CMP_ORDER):
        d = shifts[shifts.metric == m]
        others = d[~d.is_korea]
        ax.hlines(i, others.change.min(), others.change.max(), color=COLORS["grid"], lw=5, zorder=1)
        ax.scatter(others.change, np.full(len(others), i), color=COLORS["muted"], s=34, zorder=2, edgecolor=COLORS["paper"])
        k = d[d.is_korea].change.iloc[0]
        for r in {others.change.idxmin(), others.change.idxmax()}:
            # A peer name sits above its row; Korea's label sits below its own row, i.e. just above this one.
            if any(kk is not None and abs(others.loc[r, "change"] - kk) < 0.12 for kk in (k, k_prev)):
                continue
            an.note(ax, others.loc[r, "change"], i, COUNTRY_KO.get(others.loc[r, "country"], others.loc[r, "country"]),
                    ha="center", va="bottom", xytext=(0, 7), color=COLORS["muted"])
        ax.scatter([k], [i], color=COLORS["primary"], s=110, zorder=3, edgecolor=COLORS["ink"], linewidth=1.0)
        an.note(ax, k, i, "한국", ha="center", va="top", xytext=(0, -9), color=COLORS["primary_dark"], weight=700)
        k_prev = k
        an.note(ax, 1.02, i, f"{p.loc[m, 'p_holm']:.2f}", coords=("axes fraction", "data"), ha="left", size=TEXT["label"])
    an.note(ax, 1.02, -0.85, "Holm p", coords=("axes fraction", "data"), ha="left", color=COLORS["muted"])
    ax.axvline(0, color=COLORS["ink"], lw=0.8)
    ax.set_yticks(range(len(CMP_ORDER)), [p.loc[m, "label"] for m in CMP_ORDER], fontsize=TEXT["label"])
    ax.set_ylim(len(CMP_ORDER) - 0.4, -1.2)
    ax.set_xlabel("정책 후 − 전, 전년 대비 로그 성장률 평균의 변화")
    an.clean_axis(ax, "x")
    an.hide_y(ax)
    an.header(fig, "한국의 단가 하락은 비교국보다 컸지만, 위약 순위로는 우연과 구분되지 않았다",
              "점 = 국가, 회색 띠 = 비교국 범위. 오른쪽 열 = 시간 × 국가 위약 순위 p의 Holm 보정값(기준 0.05)", "estimated")
    an.footer(fig, SRC_PEERS, "비교국은 2024년 HS 3304 수입액 상위 8개국(사전 등록)")
    return save(fig, "09_country_shifts")


def event_study(path: pd.DataFrame, band: pd.DataFrame, policy_month: str, metric_ko: dict):
    fig, axes = an.figure("wide", 1, 2, sharex=True)
    for ax, m in zip(axes, ["air_weight_kg", "air_usd_per_kg"]):
        b = band[band.metric == m].pivot(index="month", columns="country", values="gap")
        lo, hi = b.min(axis=1), b.max(axis=1)
        ax.fill_between(b.index, lo, hi, color=COLORS["faint"], alpha=0.55, lw=0)
        k = path[path.metric == m].set_index("month").korea_minus_controls
        an.policy_marker(ax, label=False)
        ax.axhline(0, color=COLORS["ink"], lw=0.7)
        ax.plot(k.index, k, color=COLORS["primary"], lw=2.6)
        ax.set_title(metric_ko[m])
        month_axis(ax, (1, 7))
        an.note(ax, k.index[-1], k.iloc[-1], "한국", ha="left", xytext=(5, 0), color=COLORS["primary_dark"], weight=700)
        an.note(ax, b.index[0], lo.min(), "비교국을 한국처럼 둔 위약 범위", va="bottom", xytext=(0, 2), color=COLORS["muted"])
    axes[0].set_ylabel("비교국 대비 YoY 격차 (정책 전 평균 = 0)")
    an.header(fig, "한국의 항공 단가 격차는 위약 범위의 아래쪽 끝에 붙었고, 항공 중량은 범위 안이었다",
              "선 = 한국 − 비교국 평균, 회색 띠 = 각 비교국을 처리군으로 둔 같은 계산의 최소~최대(통계적 신뢰구간 아님)", "estimated")
    an.footer(fig, SRC_PEERS)
    return save(fig, "10_event_study")


def synthetic(path: pd.DataFrame, synth: pd.DataFrame, policy_month: str, metric_ko: dict):
    s = synth.set_index("metric")
    fig, axes = an.figure("wide", 2, 2, sharex=True, height_ratios=[1.6, 1])
    for j, m in enumerate(["air_weight_kg", "air_usd_per_kg"]):
        d = path[path.metric == m]
        ax = axes[0, j]
        an.policy_marker(ax, label=False)
        ax.plot(d.month, d.synthetic, color=COLORS["ink"], lw=1.6, ls=(0, (4, 2)))
        ax.plot(d.month, d.korea, color=COLORS["primary"], lw=2.4)
        ax.set_title(metric_ko[m])
        an.note(ax, d.month.iloc[-1], d.korea.iloc[-1], "한국", xytext=(5, 0), color=COLORS["primary_dark"], weight=700)
        an.note(ax, d.month.iloc[-1], d.synthetic.iloc[-1], "합성 한국", xytext=(5, 0), color=COLORS["ink"])
        an.note(ax, 0.02, 0.94, f"RMSPE 비 {s.loc[m, 'ratio']:.1f} · 순위 p {s.loc[m, 'p_rank']:.2f} (n={int(s.loc[m, 'n_units'])})",
                coords="axes fraction", color=COLORS["muted"], va="top", box=True)
        ax = axes[1, j]
        an.policy_marker(ax, label=False)
        ax.axhline(0, color=COLORS["ink"], lw=0.7)
        ax.fill_between(d.month, 0, d.gap, color=COLORS["primary"], alpha=0.35, lw=0)
        ax.plot(d.month, d.gap, color=COLORS["primary_dark"], lw=1.2)
        month_axis(ax, (1, 7))
    axes[0, 0].set_ylabel("로그 수준 (정책 전 평균 = 0)")
    axes[1, 0].set_ylabel("격차")
    n = int(s.loc["air_usd_per_kg", "n_units"])
    an.header(fig, f"한국의 정책 후 격차는 {n}개국 중 가장 컸지만, 나라 수가 적어 순위 p는 1/{n}이 하한이다",
              "위 = 한국과 비교국 가중 합(합성 한국), 아래 = 격차. 순위 p = 같은 방법을 각국에 적용한 RMSPE 비(정책 후 ÷ 전)의 순위", "estimated")
    an.footer(fig, SRC_PEERS)
    return save(fig, "11_synthetic_control")
