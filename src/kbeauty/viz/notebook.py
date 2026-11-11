"""Helpers for charts drawn inside the notebooks, so method and diagnostic charts share the report style.

Conclusion charts in the notebooks are the report figures themselves (`show`), not re-drawn copies.
"""

import matplotlib.pyplot as plt

from ..paths import FIGURES_DIR
from . import annotations as an
from . import theme


def setup() -> None:
    """Call after `%matplotlib inline`: the inline backend resets a few rcParams, so the theme is applied again,
    and charts are embedded at 2x resolution."""
    from IPython import get_ipython
    ip = get_ipython()
    if ip is not None:
        ip.run_line_magic("config", "InlineBackend.figure_formats = ['retina']")
    theme.apply()


def figure(size="notebook", nrows=1, ncols=1, title=None, subtitle=None, evidence=None, source=None, **kw):
    lines = 0 if not subtitle else 1 + subtitle.count("\n")
    fig, axes = an.figure(size, nrows, ncols, subtitle_lines=min(lines, 2), **kw)
    if title:
        an.header(fig, title, subtitle, evidence)
    if source:
        an.footer(fig, source)
    return fig, axes


def show(name: str):
    """Display a report figure (reports/figures/<name>.png)."""
    from IPython.display import Image, display
    display(Image(filename=str(FIGURES_DIR / f"{name}.png")))


def done(fig) -> None:
    plt.show()
