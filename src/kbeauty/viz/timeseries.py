"""Observed series: value vs weight, unit values, decomposition, air weight, ports, lead time, the mirror chart."""

import numpy as np
import pandas as pd

from . import annotations as an
from .export import save
from .formats import month_axis, pct
from .theme import COLORS, KR, TEXT, US

SRC_CENSUS = "U.S. Census Bureau 수입 통계(한국산 HS 3303·3304·3305·3307)"
SRC_PORT = "U.S. Census Bureau 하역항별 수입(HS6)"
SRC_VESSEL = "CBP CF-1400 입항 신고 × 해양수산부 PORT-MIS 출항 기록"
SRC_MIRROR = "관세청 품목별 국가별 수출실적 · U.S. Census Bureau 수입 통계(HS6 16개)"


def value_vs_weight(s: pd.DataFrame, wt: pd.DataFrame):
    chg = wt.set_index("metric").change_pct
    base = s[s.month.between("2024-01-01", "2024-12-01")]
    value = s.value_usd / base.value_usd.mean() * 100
    weight = s.measured_weight_kg / base.measured_weight_kg.mean() * 100
    fig, ax = an.figure("wide")
    an.policy_marker(ax, shade_to=s.month.max())
    ax.plot(s.month, value, color=COLORS["ink"], lw=2)
    ax.plot(s.month, weight, color=COLORS["primary"], lw=2.6)
    ax.axhline(100, color=COLORS["muted"], lw=0.8, ls=":")
    month_axis(ax, span=(s.month.min(), s.month.max()))
    ax.set_xlim(s.month.min(), s.month.max() + pd.Timedelta(days=150))
    last = s.month.max()
    an.end_labels(ax, [(last, weight.iloc[-1], f"해상+항공 중량  {pct(chg['measured_weight_kg'])}", COLORS["primary_dark"], 700),
                       (last, value.iloc[-1], f"수입 금액  {pct(chg['value_usd'])}", COLORS["ink"], 600)])
    an.header(fig, "정책 이후 중량이 금액보다 훨씬 빨리 늘었다",
              "월별 지수(2024년 평균 = 100). 라벨의 %는 정책 전후 12개월 합계의 변화", "observed")
    an.footer(fig, SRC_CENSUS)
    return save(fig, "01_value_vs_weight")


def unit_values(s: pd.DataFrame, wt: pd.DataFrame, pre: tuple, post: tuple):
    w = wt.set_index("metric")
    fig, ax = an.figure("wide")
    an.policy_marker(ax, shade_to=s.month.max())
    for col, color, lw in (("vessel_usd_per_kg", COLORS["muted"], 1.4), ("air_usd_per_kg", COLORS["primary"], 1.6)):
        ax.plot(s.month, s[col], color=color, lw=lw, alpha=0.9)
        for (a, b), key in ((pre, "pre_12m"), (post, "post_12m")):
            ax.hlines(w.loc[col, key], pd.Timestamp(a), pd.Timestamp(b) + pd.offsets.MonthEnd(0),
                      color=COLORS["primary_dark"] if col.startswith("air") else COLORS["ink"], lw=2.6)
    month_axis(ax, span=(s.month.min(), s.month.max()))
    ax.set_ylabel("USD / kg")
    ax.set_xlim(s.month.min(), s.month.max() + pd.Timedelta(days=260))
    last = s.month.max() + pd.Timedelta(days=40)  # labels start after the post-window average line
    an.end_labels(ax, [
        (last, w.loc["air_usd_per_kg", "post_12m"],
         f"항공  {w.loc['air_usd_per_kg', 'pre_12m']:.1f} → {w.loc['air_usd_per_kg', 'post_12m']:.1f}  ({pct(w.loc['air_usd_per_kg', 'change_pct'])})",
         COLORS["primary_dark"], 700),
        (last, w.loc["vessel_usd_per_kg", "post_12m"],
         f"해상  {w.loc['vessel_usd_per_kg', 'pre_12m']:.1f} → {w.loc['vessel_usd_per_kg', 'post_12m']:.1f}  ({pct(w.loc['vessel_usd_per_kg', 'change_pct'])})",
         COLORS["ink"], 600)])
    an.header(fig, f"미국 기록 kg당 단가는 항공에서 {abs(w.loc['air_usd_per_kg', 'change_pct']) * 100:.0f}% 떨어졌다",
              "가는 선 = 월별 값, 굵은 선 = 정책 전·후 12개월 평균(금액 합 ÷ 중량 합)", "observed")
    an.footer(fig, SRC_CENSUS)
    return save(fig, "02_unit_values")


def decomposition(dec: pd.DataFrame):
    fig, axes = an.figure("wide", 1, 2, sharey=True)
    colors = {"mix": COLORS["muted"], "price": COLORS["primary"], "interaction": COLORS["faint"]}
    labels = ["정책 전", "품목 구성", "품목 내 단가", "상호작용", "정책 후"]
    for ax, (_, r) in zip(axes, dec.iterrows()):
        steps = [("mix", r["mix"]), ("price", r["price"]), ("interaction", r["interaction"])]
        ax.bar(0, r["uv_base"], color=COLORS["ink"], width=0.62)
        level = r["uv_base"]
        for i, (k, v) in enumerate(steps, start=1):
            ax.bar(i, v, bottom=level, color=colors[k], width=0.62)
            ax.plot([i - 1 + 0.31, i - 0.31], [level, level], color=COLORS["muted"], lw=0.8)
            an.note(ax, i, level + v, "0.0" if abs(v) < 0.05 else f"{v:+.1f}", ha="center",
                    va="top" if v < 0 else "bottom", xytext=(0, -3 if v < 0 else 3), size=TEXT["label"],
                    weight=700 if k == "price" else 500, color=COLORS["primary_dark"] if k == "price" else COLORS["ink"])
            level += v
        ax.plot([3 + 0.31, 4 - 0.31], [level, level], color=COLORS["muted"], lw=0.8)
        ax.bar(4, r["uv_comp"], color=COLORS["ink"], width=0.62)
        for x, v in ((0, r["uv_base"]), (4, r["uv_comp"])):
            an.note(ax, x, v, f"{v:.1f}", ha="center", va="bottom", xytext=(0, 3), size=TEXT["label"], weight=600)
        ax.set_xticks(range(5), labels, fontsize=TEXT["tick"] - 0.5)
        mode = "해상" if r["mode"] == "vessel" else "항공"
        share = r["price_share_of_change"]
        ax.set_title(f"{mode}  {pct(r['change_pct'])}  ·  품목 내 단가 몫 {share * 100:.0f}%")
        an.clean_axis(ax)
    axes[0].set_ylabel("USD / kg")
    axes[0].set_ylim(0, dec.uv_base.max() * 1.14)
    an.header(fig, "단가 하락의 대부분은 품목 구성이 아니라 같은 품목 안에서 일어났다",
              "정책 전 → 후 12개월 kg당 단가 변화를 품목 구성·품목 내 단가·상호작용으로 나눈 항등식 분해(HS10)", "observed")
    an.footer(fig, SRC_CENSUS)
    return save(fig, "03_unit_value_decomposition")


def air_weight(s: pd.DataFrame, wt: pd.DataFrame, pre: tuple, post: tuple):
    w = wt.set_index("metric").loc["air_weight_kg"]
    fig, ax = an.figure("wide")
    an.policy_marker(ax, shade_to=s.month.max() + pd.Timedelta(days=15))
    ax.bar(s.month, s.air_weight_kg / 1e6, width=22, color=COLORS["faint"])
    roll = (s.air_weight_kg / 1e6).rolling(3).mean()
    ax.plot(s.month, roll, color=COLORS["ink"], lw=1.6)
    for (a, b), key, color in ((pre, "pre_12m", COLORS["ink"]), (post, "post_12m", COLORS["primary_dark"])):
        ax.hlines(w[key] / 12 / 1e6, pd.Timestamp(a) - pd.Timedelta(days=12), pd.Timestamp(b) + pd.Timedelta(days=12),
                  color=color, lw=2.8)
        an.note(ax, pd.Timestamp(a), w[key] / 12 / 1e6, f"월평균 {w[key] / 12 / 1e6:.2f}백만 kg", color=color,
                ha="left", va="bottom", xytext=(-6, 5), size=TEXT["label"], weight=700, box=True)
    month_axis(ax, span=(s.month.min(), s.month.max()))
    ax.set_ylabel("항공 중량 (백만 kg)")
    an.note(ax, s.month.iloc[3], roll.iloc[3], "3개월 이동평균", color=COLORS["ink"], va="top", xytext=(0, -8), box=True)
    an.header(fig, f"항공 중량은 정책 후 12개월 동안 {pct(w['change_pct'], 0)} 늘었다",
              "막대 = 월별 항공 중량, 굵은 선 = 정책 전·후 12개월 월평균", "observed")
    an.footer(fig, SRC_CENSUS)
    return save(fig, "04_air_weight")


def port_concentration(conc: pd.DataFrame, port_tbl: pd.DataFrame):
    post = port_tbl.set_index("window").loc["post_12m"]
    fig, ax = an.figure("wide")
    an.policy_marker(ax, shade_to=conc.month.max())
    ax.plot(conc.month, conc.west_coast_share, color=COLORS["west"], lw=1.6)
    ax.plot(conc.month, conc.la_lb_share, color=COLORS["primary"], lw=2.6)
    ax.plot(conc.month, conc.east_coast_share, color=COLORS["east"], lw=1.6)
    ax.set_ylim(0, 1)
    from .formats import percent_axis
    percent_axis(ax, "y")
    month_axis(ax, span=(conc.month.min(), conc.month.max()))
    ax.set_xlim(conc.month.min(), conc.month.max() + pd.Timedelta(days=170))
    last = conc.month.max()
    an.end_labels(ax, [(last, conc.west_coast_share.iloc[-1], "서부 전체", COLORS["west"], 600),
                       (last, conc.la_lb_share.iloc[-1], f"LA·롱비치  {post.la_lb_share_mean * 100:.0f}%", COLORS["primary_dark"], 700),
                       (last, conc.east_coast_share.iloc[-1], "동부 전체", COLORS["east"], 600)])
    an.header(fig, f"해상 수입의 {post.la_lb_share_mean * 100:.0f}%가 LA·롱비치 두 항만으로 들어온다",
              "해상 수입 금액의 하역항 비중(월별). 라벨의 %는 정책 후 12개월 평균", "observed")
    an.footer(fig, SRC_PORT)
    return save(fig, "06_port_concentration")


def transit_by_region(transit_tbl: pd.DataFrame):
    order = ["west_coast", "gulf_coast", "east_coast"]
    names = {"west_coast": "서부", "gulf_coast": "걸프", "east_coast": "동부"}
    colors = {"west_coast": COLORS["west"], "gulf_coast": COLORS["muted"], "east_coast": COLORS["east"]}
    t = transit_tbl.set_index("region").loc[order]
    fig, ax = an.figure((10.0, 3.9))
    for y, (region, r) in enumerate(t.iterrows()):
        ax.hlines(y, r.p10, r.p90, color=colors[region], lw=6, alpha=0.35)
        ax.plot(r["median"], y, "o", color=colors[region], ms=10)
        an.note(ax, r["median"], y, f"{r['median']:.0f}일", ha="center", va="bottom", xytext=(0, 9),
                size=TEXT["label"], weight=700, color=colors[region])
        an.note(ax, r.p90, y, f"p10~p90  {r.p10:.0f}~{r.p90:.0f}일 · 관측 {int(r.n)}건", ha="left",
                xytext=(10, 0), color=COLORS["muted"])
    ax.set_yticks(range(3), [names[r] for r in order], fontsize=TEXT["label"])
    ax.set_ylim(2.55, -0.75)
    ax.set_xlim(0, t.p90.max() + 12)
    ax.set_xlabel("부산 출항 → 미국 입항 신고 (일)")
    an.clean_axis(ax, "x")
    an.hide_y(ax)
    an.header(fig, f"동부 직송은 서부보다 해상에서 {t.loc['east_coast', 'median'] - t.loc['west_coast', 'median']:.0f}일 더 걸린다",
              "권역별 해상 리드타임: 점 = 중앙값, 띠 = 10~90% 분위", "observed")
    an.footer(fig, SRC_VESSEL)
    return save(fig, "06_transit_by_region")


def mirror(mir: pd.DataFrame, gran: pd.DataFrame, policy_month: str):
    g = gran.set_index("series")
    kr, us = g.loc["Korea, HS6"], g.loc["US, HS6 (sea+air)"]
    fig, axes = an.figure("wide", 1, 2, width_ratios=[1.35, 1])
    ax = axes[0]
    base = mir.loc["2024-01-01":"2024-12-01"]
    an.policy_marker(ax, label=False, shade_to=mir.index.max())
    kr_v = mir.kr_export_usd / base.kr_export_usd.mean() * 100
    us_v = mir.us_import_usd / base.us_import_usd.mean() * 100
    ax.plot(mir.index, kr_v, color=KR, lw=2)
    ax.plot(mir.index, us_v, color=US, lw=2.6)
    ax.set_title("금액: 같은 흐름이 두 나라에 기록된다 (2024년 평균 = 100)")
    month_axis(ax, span=(mir.index.min(), mir.index.max()))
    ax.set_xlim(mir.index.min(), mir.index.max() + pd.Timedelta(days=170))
    an.end_labels(ax, [(mir.index.max(), kr_v.iloc[-1], "한국 수출 기록", KR, 600),
                       (mir.index.max(), us_v.iloc[-1], "미국 수입 기록", COLORS["primary_dark"], 700)])
    ax = axes[1]
    for r, color, name in ((kr, KR, "한국 기록"), (us, US, "미국 기록")):
        ax.plot([0, 1], [r.uv_base, r.uv_comp], color=color, lw=3 if color == US else 2.2, marker="o", ms=8)
        txt_color = COLORS["primary_dark"] if color == US else KR
        an.note(ax, 0, r.uv_base, f"{r.uv_base:.1f}", ha="right", xytext=(-9, 0), size=TEXT["label"], color=txt_color, weight=600)
        an.note(ax, 1, r.uv_comp, f"{r.uv_comp:.1f}  {name} {pct(r.change_pct)}", ha="left", xytext=(9, 0),
                size=TEXT["label"] + 1, color=txt_color, weight=700)
    ax.set_xticks([0, 1], ["정책 전 12개월", "정책 후 12개월"])
    ax.set_xlim(-0.35, 1.9)
    vals = [kr.uv_base, kr.uv_comp, us.uv_base, us.uv_comp]
    ax.set_ylim(min(vals) - 2.5, max(vals) + 2.5)
    ax.set_title("kg당 기록 금액(USD): 반대 방향으로 갈라졌다")
    an.clean_axis(ax)
    ax.spines["bottom"].set_visible(False)
    an.header(fig, "같은 흐름이 한국에서는 더 비싸게, 미국에서는 더 싸게 기록됐다",
              "kg당 금액 = 12개월 금액 합 ÷ 중량 합(HS6 16개). 한국은 FOB 수출 신고가, 미국은 수입 신고가(과세가격) 기준", "observed")
    an.footer(fig, SRC_MIRROR, "점선 = 2025.08 정책 시점. 격차의 원인(신고 가격 산정 변화)은 가설로 남김")
    return save(fig, "12_mirror_same_flow_different_price")
