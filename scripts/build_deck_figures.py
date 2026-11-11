"""Slide versions of the charts used in the presentation deck: same data and canvas, text scaled up so axis
labels stay legible when a chart fills part of a 16:9 slide. Reads the tables written by scripts/run_all.py.

    uv run python scripts/build_deck_figures.py        # writes reports/figures/deck/*.png and *.svg
"""

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from kbeauty import documentation, report, viz  # noqa: E402
from kbeauty.paths import FIGURES_DIR, POST_WINDOW, PRE_WINDOW, TABLES_DIR  # noqa: E402
from kbeauty.viz import export, theme  # noqa: E402

SCALE = 1.45        # axis labels, ticks and in-chart labels
NOTE_SCALE = 1.2    # sources, footnotes and badges: long lines that must stay inside the canvas
# Dense charts get smaller factors (text, notes) so labels keep clear of each other and of box edges.
SCALE_FOR = {"route_sensitivity": (1.2, NOTE_SCALE), "lineage": (1.15, 1.05), "ml_comparison": (SCALE, 1.1)}
OUT = FIGURES_DIR / "deck"


def main() -> None:
    t = {p.stem: pd.read_csv(p, encoding="utf-8-sig") for p in TABLES_DIR.glob("*.csv")}
    # Forecast numbers in the deck are the official ones: the pinned Linux reference, not the author environment.
    canonical = FIGURES_DIR.parent / "platform" / "linux" / "forecast_hierarchy_wape.csv"
    if not canonical.exists():
        sys.exit(f"missing canonical reference {canonical}")
    lw_official = pd.read_csv(canonical, encoding="utf-8-sig")
    d = report.load_inputs()
    s = d["mart_monthly_summary"].sort_values("month")
    conc = d["mart_port_concentration_monthly"].sort_values("month")
    sens = t["decision_route_sensitivity"]
    charts = {
        "value_vs_weight": lambda: viz.timeseries.value_vs_weight(s, t["window_totals"]),
        "decomposition": lambda: viz.timeseries.decomposition(t["unit_value_decomposition"]),
        "port_concentration": lambda: viz.timeseries.port_concentration(conc, t["port_concentration"]),
        "policy_shifts": lambda: viz.comparison.policy_shifts(t["policy_shifts"]),
        "country_shifts": lambda: viz.comparison.country_shifts(t["comparison_shifts"], t["comparison_did"]),
        "ml_comparison": lambda: viz.forecasting.ml_comparison(lw_official, "Linux 고정 환경(공식)"),
        "route_sensitivity": lambda: viz.decision.route_sensitivity(sens, report.BASE["service_level"]),
        "lineage": lambda: viz.lineage.lineage(documentation.lineage(), {k: None for k in t}, documentation.seeds()),
    }
    export.OUTPUT_DIR = OUT
    try:
        for name, draw in charts.items():
            theme.scale_text(*SCALE_FOR.get(name, (SCALE, NOTE_SCALE)))
            draw()
    finally:
        export.OUTPUT_DIR = None
        theme.scale_text(1.0)
    print(f"wrote {len(list(OUT.glob('*.png')))} slide charts to {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
