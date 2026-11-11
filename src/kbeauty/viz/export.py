"""Write a figure as PNG (2x, notebooks/README/GitHub) and SVG (editable text, Figma/PPT), after checking fonts.

The font check walks every visible text object and refuses to save if any character is missing from the
bundled Pretendard, so a glyph can never fall back silently to another font.
"""

from functools import lru_cache

import matplotlib.pyplot as plt
from matplotlib.ft2font import FT2Font
from matplotlib.text import Text

from ..paths import FIGURES_DIR
from .theme import FONT_DIR, FONT_FILES


@lru_cache(maxsize=1)
def covered_chars() -> frozenset:
    return frozenset(chr(c) for c in FT2Font(str(FONT_DIR / FONT_FILES[400])).get_charmap())


def missing_glyphs(fig) -> dict[str, str]:
    """{text: missing characters} for every visible text in the figure."""
    cover = covered_chars()
    out = {}
    for t in fig.findobj(Text):
        s = t.get_text()
        if not s or not t.get_visible():
            continue
        miss = "".join(sorted({ch for ch in s if ch not in cover and not ch.isspace()}))
        if miss:
            out[s] = miss
    return out


OUTPUT_DIR = None  # overrides FIGURES_DIR when set (slide versions of the charts)


def save(fig, stem: str, formats=("png", "svg"), directory=None) -> list:
    miss = missing_glyphs(fig)
    if miss:
        plt.close(fig)
        raise ValueError(f"{stem}: characters not in Pretendard (would fall back): {miss}")
    directory = directory or OUTPUT_DIR or FIGURES_DIR
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in formats:
        path = directory / f"{stem}.{ext}"
        meta = {"Software": None} if ext == "png" else {"Date": None, "Creator": None}
        fig.savefig(path, metadata=meta)
        paths.append(path)
    plt.close(fig)
    return paths
