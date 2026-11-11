"""Figure scaffolding and annotations: header, evidence badge, source line, policy marker, direct labels.

Layout: every figure is created by `figure()`, which reserves a header band (title, subtitle, evidence badge)
and a footer band (source, unit, note) outside the constrained-layout area. Titles therefore never collide with
panel titles, and the chart area keeps the same proportions across figures of the same size.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .theme import CHART_SIZE, COLORS, POLICY_DATE, POLICY_LABEL, TEXT

# Evidence levels shown as a badge in the top-right corner (same vocabulary as reports/tables/claims.csv).
EVIDENCE = {
    "observed": ("OBSERVED · 관측", {"facecolor": COLORS["panel"], "edgecolor": COLORS["ink"], "linestyle": "-"},
                 COLORS["ink"]),
    "estimated": ("ESTIMATED · 추정", {"facecolor": COLORS["ink"], "edgecolor": COLORS["ink"], "linestyle": "-"},
                  COLORS["paper"]),
    "inferred": ("INFERRED · 추론", {"facecolor": COLORS["panel"], "edgecolor": COLORS["ink"], "linestyle": ":"},
                 COLORS["ink"]),
    "scenario": ("SCENARIO · 시나리오", {"facecolor": COLORS["panel"], "edgecolor": COLORS["east"], "linestyle": "--"},
                 COLORS["east"]),
}

HEADER_IN = {0: 0.62, 1: 0.88, 2: 1.12}   # header band height by number of subtitle lines
FOOTER_IN = 0.42


def figure(size="wide", nrows=1, ncols=1, subtitle_lines=1, **kw):
    """Create a figure whose axes stay clear of the header and footer bands. Returns (fig, axes)."""
    w, h = CHART_SIZE[size] if isinstance(size, str) else size
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h), layout="constrained", **kw)
    top = 1 - HEADER_IN[subtitle_lines] / h
    bottom = FOOTER_IN / h
    fig.get_layout_engine().set(rect=(0.0, bottom, 1.0, top - bottom), w_pad=0.06, h_pad=0.06)
    fig._kb_header = (HEADER_IN[subtitle_lines], h)
    return fig, axes


def header(fig, title: str, subtitle: str | None = None, evidence: str | None = None) -> None:
    """Conclusion title, one- or two-line subtitle and the evidence badge."""
    w, h = fig.get_size_inches()
    y = 1 - 0.16 / h
    fig.text(0.012, y, title, ha="left", va="top", fontsize=TEXT["title"], fontweight=700, color=COLORS["ink"])
    if subtitle:
        fig.text(0.012, y - 0.36 / h, subtitle, ha="left", va="top", fontsize=TEXT["subtitle"], fontweight=400,
                 color=COLORS["muted"], linespacing=1.35)
    if evidence:
        label, box, color = EVIDENCE[evidence]
        fig.text(0.988, y - 0.02 / h, label, ha="right", va="top", fontsize=TEXT["badge"], fontweight=600,
                 color=color, bbox={"boxstyle": "round,pad=0.38,rounding_size=0.25", "linewidth": 0.9, **box})


def footer(fig, source: str, note: str | None = None) -> None:
    w, h = fig.get_size_inches()
    text = f"출처: {source}" + (f"   ·   {note}" if note else "")
    fig.text(0.012, 0.12 / h, text, ha="left", va="bottom", fontsize=TEXT["note"], color=COLORS["muted"])


def policy_marker(ax, label: bool = True, shade_to=None) -> None:
    """The same mark on every time axis: a thin dashed line on 2025-08-07 and a faint post-policy band."""
    t = pd.Timestamp(POLICY_DATE)
    end = pd.Timestamp(shade_to) if shade_to is not None else None
    if end is not None:
        ax.axvspan(t, end, color=COLORS["grid"], alpha=0.35, lw=0, zorder=0)
    ax.axvline(t, color=COLORS["primary_dark"], lw=0.9, ls=(0, (3, 2)), zorder=1)
    if label:
        ax.annotate(POLICY_LABEL, (t, 1.0), xycoords=("data", "axes fraction"), xytext=(4, -2),
                    textcoords="offset points", ha="left", va="top", fontsize=TEXT["note"],
                    color=COLORS["primary_dark"], fontweight=500)


def end_labels(ax, items, min_gap_pt: float | None = None, dx_pt: float = 6) -> None:
    """Direct labels right of the last point of each line, nudged apart vertically so they never overlap.
    items: list of (x, y, text, color, weight)."""
    if not items:
        return
    fig = ax.figure
    fig.canvas.draw()
    gap = min_gap_pt or TEXT["label"] * 1.35
    to_disp = ax.transData.transform
    pts = sorted(((to_disp((pd.Timestamp(x).to_datetime64() if isinstance(x, (str, pd.Timestamp)) else x, y))[1],
                   x, y, t, c, wgt) for x, y, t, c, wgt in items), key=lambda r: r[0])
    dpi_scale = fig.dpi / 72
    ys = [p[0] for p in pts]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + gap * dpi_scale)
    for (y_disp, x, y, text, color, wgt), y_new in zip(pts, ys):
        ax.annotate(text, (x, y), xytext=(dx_pt, (y_new - y_disp) / dpi_scale), textcoords="offset points",
                    ha="left", va="center", fontsize=TEXT["label"], color=color, fontweight=wgt,
                    annotation_clip=False)


def note(ax, x, y, text, color=None, ha="left", va="center", xytext=(0, 0), weight=400, size=None, coords="data",
         box=False):
    """A text label at a data point. box=True puts a paper-coloured backing behind it for busy areas."""
    bbox = {"boxstyle": "round,pad=0.25", "facecolor": COLORS["paper"], "edgecolor": "none", "alpha": 0.92} if box else None
    ax.annotate(text, (x, y), xycoords=coords, xytext=xytext, textcoords="offset points", ha=ha, va=va,
                fontsize=size or TEXT["note"], color=color or COLORS["ink"], fontweight=weight,
                annotation_clip=False, bbox=bbox, zorder=5)


def clean_axis(ax, grid_axis: str = "y") -> None:
    ax.grid(False)
    if grid_axis:
        ax.grid(True, axis=grid_axis)
    ax.tick_params(length=0 if grid_axis == "x" else 3)


def hide_y(ax) -> None:
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def scenario_hatch(ax) -> None:
    """Scenario panels get a dashed frame so assumptions never look like observed data."""
    for side in ("left", "bottom", "top", "right"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linestyle((0, (4, 3)))
        ax.spines[side].set_edgecolor(COLORS["east"])
        ax.spines[side].set_linewidth(0.9)


def arrays(*cols):
    return [np.asarray(c, dtype=float) for c in cols]
