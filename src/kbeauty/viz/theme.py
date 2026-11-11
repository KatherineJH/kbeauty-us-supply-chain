"""Design tokens and the matplotlib theme shared by every chart (report figures and notebook charts).

Fonts: the bundled Pretendard OTFs (assets/fonts/pretendard, OFL 1.1) are registered by file path, so a chart
renders the same on Windows and on a Linux CI runner whatever is installed. Malgun Gothic stays only as the last
fallback in the family list; `export.save` refuses to write a chart with a character Pretendard does not cover,
so a silent fallback cannot slip into an output.
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from ..paths import REPO

FONT_DIR = REPO / "assets" / "fonts" / "pretendard"
FONT_FILES = {400: "Pretendard-Regular.otf", 500: "Pretendard-Medium.otf",
              600: "Pretendard-SemiBold.otf", 700: "Pretendard-Bold.otf"}
# findfont logs a weight lookup for the last-resort family on every call; coverage is enforced in export.save.
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
FAMILY = "Pretendard"
LAST_RESORT = "Malgun Gothic"

# Colour tokens. One meaning per colour across the project.
COLORS = {
    "ink": "#171717",        # text, axes, reference series
    "paper": "#F5F1E8",      # figure background
    "panel": "#FBF9F4",      # card / badge background
    "muted": "#8B8B85",      # comparison series, peers, secondary text
    "faint": "#C9C4B8",      # placebo bands, background series
    "grid": "#DED9CF",
    "primary": "#D9A21B",    # the series the chart is about (Korea, the chosen rule, the policy moment)
    "primary_dark": "#8A6100",  # primary used as text or thin lines (contrast on paper ≥ 4.5:1)
    "west": "#315F78",       # West-coast unloading / Korean record in the mirror charts
    "east": "#C96F4A",       # East-coast direct / US record in the mirror charts
    "positive": "#2F766D",
    "negative": "#B64A42",
}
# Mirror charts: the US record is the focus series (primary), the Korean record is the reference (ink).
# West/East hues are reserved for the routing decision and are not reused here.
KR, US = COLORS["ink"], COLORS["primary"]

# Canvas sizes in inches at 100 dpi; PNG is written at 2x.
CHART_SIZE = {
    "full_slide": (12.0, 5.8),   # one chart filling a 16:9 slide body
    "wide": (10.0, 4.4),         # report / case-study width
    "three_panel": (11.5, 4.6),
    "two_column": (5.6, 4.4),
    "notebook": (9.0, 4.2),
}
TEXT = {"title": 15, "subtitle": 11, "label": 10.5, "tick": 10, "note": 8.5, "badge": 8}

POLICY_DATE = "2025-08-07"           # 15% reciprocal tariff; de minimis suspended on 2025-08-29
POLICY_LABEL = "2025.08 관세 15%·소액면세 중단"


def register_fonts() -> None:
    """Register the four bundled weights by path. Raises if a file is missing or resolves elsewhere."""
    for weight, name in FONT_FILES.items():
        path = FONT_DIR / name
        if not path.exists():
            raise FileNotFoundError(f"bundled font missing: {path} (see assets/fonts/pretendard/SOURCE.md)")
        font_manager.fontManager.addfont(str(path))
    for weight, name in FONT_FILES.items():
        found = Path(font_manager.findfont(font_manager.FontProperties(family=FAMILY, weight=weight),
                                           fallback_to_default=False))
        if found.resolve() != (FONT_DIR / name).resolve():
            raise RuntimeError(f"{FAMILY} weight {weight} resolved to {found}, not the bundled file")


def apply() -> None:
    register_fonts()
    c = COLORS
    plt.rcParams.update({
        "font.family": [FAMILY, LAST_RESORT],
        "font.size": TEXT["label"], "font.weight": 400,
        "axes.unicode_minus": False,
        "figure.facecolor": c["paper"], "axes.facecolor": c["paper"], "savefig.facecolor": c["paper"],
        "figure.dpi": 100, "savefig.dpi": 200,
        "axes.edgecolor": c["ink"], "axes.linewidth": 0.8, "axes.labelcolor": c["ink"],
        "axes.labelsize": TEXT["label"], "axes.titlesize": TEXT["subtitle"], "axes.titleweight": 600,
        "axes.titlelocation": "left", "axes.titlepad": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": c["grid"], "grid.linewidth": 0.7,
        "axes.axisbelow": True, "axes.prop_cycle": plt.cycler(color=[c["primary"], c["muted"], c["west"], c["east"]]),
        "xtick.color": c["ink"], "ytick.color": c["ink"], "xtick.labelsize": TEXT["tick"], "ytick.labelsize": TEXT["tick"],
        "xtick.major.size": 3, "ytick.major.size": 0, "ytick.major.pad": 6,
        "lines.linewidth": 2.0, "lines.solid_capstyle": "round",
        "date.converter": "concise",  # date ticks that never collide (notebook charts; report figures set their own)
        "legend.frameon": False, "legend.fontsize": TEXT["note"],
        "svg.fonttype": "none",   # SVG keeps editable text (font-family Pretendard, weights 400-700) for Figma/PPT
        "pdf.fonttype": 42,       # PDF embeds the OTF outlines, one subset per weight
        "svg.hashsalt": "kbeauty",  # stable element ids, so re-running yields the same SVG
    })



BASE_TEXT = dict(TEXT)


def scale_text(factor: float = 1.0, note_factor: float | None = None) -> None:
    """Scale every text size used by the charts (TEXT is shared by reference) and re-apply the theme.
    1.0 restores the report sizes. Used for slide versions of the charts, where a chart is shown smaller.
    `note_factor` scales the small text (notes, sources, badges) separately so long source lines still fit."""
    for k, v in BASE_TEXT.items():
        f = note_factor if note_factor is not None and k in ("note", "badge") else factor
        TEXT[k] = round(v * f, 1)
    apply()


apply()
