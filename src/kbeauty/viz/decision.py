"""Scenario charts for the routing decision. Every axis except the two ocean legs is an assumption, so these
charts carry the SCENARIO badge and a dashed frame, and West/East keep their fixed hues."""

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from . import annotations as an
from .export import save
from .theme import COLORS, TEXT

SRC_ROUTE = "CF-1400 × PORT-MIS 관측 해상 리드타임 + 가정한 내륙 구간(assumptions.csv)"


def route_breakeven(curves: pd.DataFrame, cv: float, service_level: float):
    """curves: inland_sd_days, inland_days, west_minus_east_days, break_even_inland_days."""
    fig, ax = an.figure("wide")
    x0, x1 = curves.inland_days.min(), curves.inland_days.max()
    lim = float(np.abs(curves.west_minus_east_days).max()) * 1.15
    ax.axhspan(-lim, 0, color=COLORS["west"], alpha=0.08, lw=0)
    ax.axhspan(0, lim, color=COLORS["east"], alpha=0.08, lw=0)
    an.note(ax, x0, -lim * 0.92, "서부 하역이 재고를 덜 묶음", color=COLORS["west"], weight=600, size=TEXT["label"], va="bottom", xytext=(8, 0))
    an.note(ax, x0, lim * 0.92, "동부 직송이 재고를 덜 묶음", color=COLORS["east"], weight=600, size=TEXT["label"], va="top", xytext=(8, 0))
    ax.axhline(0, color=COLORS["ink"], lw=0.9)
    shades = {1.0: COLORS["faint"], 2.0: COLORS["ink"], 4.0: COLORS["muted"]}
    items = []
    for sd, g in curves.groupby("inland_sd_days"):
        g = g.sort_values("inland_days")
        ax.plot(g.inland_days, g.west_minus_east_days, color=shades.get(sd, COLORS["muted"]), lw=2.6 if sd == 2.0 else 1.8)
        be = g.break_even_inland_days.iloc[0]
        ax.plot([be], [0], "o", ms=7, color=COLORS["paper"], markeredgecolor=shades.get(sd, COLORS["muted"]), mew=2, zorder=4)
        above = sd == 2.0
        an.note(ax, be, 0, f"{be:.1f}일", ha="center", va="bottom" if above else "top",
                xytext=(0, 9 if above else -9), weight=700 if above else 600, size=TEXT["label"])
        items.append((x1, g.west_minus_east_days.iloc[-1], f"내륙 표준편차 {sd:.0f}일", shades.get(sd, COLORS["muted"]) if sd != 1.0 else COLORS["muted"], 600))
    ax.set_xlim(x0, x1 + 3.2)
    ax.set_ylim(-lim, lim)
    an.end_labels(ax, items)
    ax.set_xlabel("서부 항만 → 동부 시장 내륙 운송 일수 (가정)")
    ax.set_ylabel("서부 − 동부 재고 일수")
    an.clean_axis(ax, "y")
    an.scenario_hatch(ax)
    base_be = curves[curves.inland_sd_days == 2.0].break_even_inland_days.iloc[0]
    an.header(fig, f"내륙 운송이 약 {base_be:.0f}일 이하이면 서부 하역이 재고를 덜 묶는다",
              f"재고 일수 = 운송 중 재고 + 안전재고. 수요 변동계수 {cv}, 서비스 수준 {service_level:.0%} 가정. 점 = 손익분기", "scenario")
    an.footer(fig, SRC_ROUTE, "운임·터미널 체류·통관 비용은 포함하지 않음")
    return save(fig, "14_route_breakeven")


def route_sensitivity(sens: pd.DataFrame, service_level: float):
    d = sens[sens.service_level == service_level]
    sds = sorted(d.inland_sd_days.unique())
    lim = float(np.abs(d.west_minus_east_days).max())
    cmap = LinearSegmentedColormap.from_list("route", [COLORS["west"], "#EEF0EC", COLORS["east"]])
    norm = TwoSlopeNorm(0, -lim, lim)
    fig, axes = an.figure("three_panel", 1, len(sds), sharey=True, subtitle_lines=2)
    for ax, sd in zip(axes, sds):
        g = d[d.inland_sd_days == sd].pivot(index="cv_demand", columns="inland_days", values="west_minus_east_days")
        x, ycv = g.columns.to_numpy(float), g.index.to_numpy(float)
        xe = np.r_[x - 0.5, x[-1] + 0.5]
        ye = np.arange(len(ycv) + 1) - 0.5
        mesh = ax.pcolormesh(xe, ye, g.to_numpy(), cmap=cmap, norm=norm, shading="flat", alpha=0.85)
        be = d[d.inland_sd_days == sd].groupby("cv_demand").break_even_inland_days.first().reindex(ycv)
        # Boundary: vertical segment at each row's break-even, joined between rows.
        bx = np.repeat(be.to_numpy(), 2)
        by = np.ravel([[i - 0.5, i + 0.5] for i in range(len(ycv))])
        ax.plot(bx, by, color=COLORS["ink"], lw=2.4)
        for i, v in enumerate(be):
            an.note(ax, v, i, f"{v:.1f}", ha="right", xytext=(-5, 0), weight=700, size=TEXT["note"] + 0.5)  # west side: clear of the "E" label
        an.note(ax, x[0] - 0.3, len(ycv) - 0.72, "W 서부 유리", color=COLORS["paper"], weight=700, size=TEXT["label"])
        an.note(ax, x[-1] + 0.3, len(ycv) - 0.72, "E 동부 유리", color="#7A3A20", weight=700, size=TEXT["label"], ha="right")
        ax.set_yticks(range(len(ycv)), [f"{v:.1f}" for v in ycv])
        ax.set_xticks(x[::2])
        ax.set_xlabel("내륙 운송 일수 (가정)")
        ax.set_title(f"내륙 표준편차 {sd:.0f}일 · 손익분기 {be.min():.1f}~{be.max():.1f}일", fontsize=TEXT["label"])
        ax.grid(False)
        an.scenario_hatch(ax)
    axes[0].set_ylabel("수요 변동계수 (가정)", labelpad=8)
    cb = fig.colorbar(mesh, ax=list(axes), orientation="horizontal", location="bottom", shrink=0.35, aspect=30, pad=0.02)
    cb.set_label("서부 − 동부 재고 일수 (음수 = 서부 유리)", fontsize=TEXT["note"])
    cb.ax.tick_params(labelsize=TEXT["note"])
    cb.outline.set_visible(False)
    an.header(fig, "경계는 수요 변동보다 내륙 운송 일수의 흔들림에 따라 움직인다",
              f"검은 선 = 서부 하역과 동부 직송이 같은 재고 일수를 묶는 내륙 일수(숫자). 서비스 수준 {service_level:.0%}.\n"
              "모든 축이 가정값이며 관측값은 두 해상 구간의 리드타임뿐이다", "scenario")
    an.footer(fig, SRC_ROUTE, "운임·터미널 체류·통관 비용은 포함하지 않음")
    return save(fig, "16_route_sensitivity")
