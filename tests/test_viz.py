import hashlib
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
from matplotlib import font_manager

from kbeauty.paths import FIGURES_DIR
from kbeauty.viz import annotations as an
from kbeauty.viz import export, theme


def test_bundled_fonts_match_their_recorded_hashes():
    sums = (theme.FONT_DIR / "SHA256SUMS").read_text(encoding="utf-8").split("\n")
    recorded = dict(reversed(line.split("  ", 1)) for line in sums if line)
    assert set(theme.FONT_FILES.values()) <= set(recorded)
    for name, digest in recorded.items():
        assert hashlib.sha256((theme.FONT_DIR / name).read_bytes()).hexdigest() == digest, name


@pytest.mark.parametrize("weight", [400, 500, 600, 700])
def test_each_weight_resolves_to_the_bundled_file(weight):
    found = Path(font_manager.findfont(font_manager.FontProperties(family=theme.FAMILY, weight=weight),
                                       fallback_to_default=False))
    assert found.resolve() == (theme.FONT_DIR / theme.FONT_FILES[weight]).resolve()
    assert plt.rcParams["font.family"][0] == theme.FAMILY and plt.rcParams["font.family"][-1] == theme.LAST_RESORT


def test_save_refuses_a_character_missing_from_pretendard(tmp_path):
    fig, ax = an.figure("notebook")
    ax.set_title("정상 텍스트")
    assert not export.missing_glyphs(fig)
    ax.text(0.5, 0.5, "\U0001F600")  # an emoji: not in Pretendard, would fall back silently
    with pytest.raises(ValueError, match="would fall back"):
        export.save(fig, "probe", directory=tmp_path)


def test_svg_keeps_editable_text_with_all_weights(tmp_path):
    fig, ax = an.figure("notebook")
    for i, w in enumerate((400, 500, 600, 700)):
        ax.text(0.1, 0.2 * i, f"가중치 {w}", fontweight=w)
    export.save(fig, "weights", formats=("svg",), directory=tmp_path)
    svg = (tmp_path / "weights.svg").read_text(encoding="utf-8")
    assert "Pretendard" in svg
    assert {"500", "600", "700"} <= set(re.findall(r"font-weight[:=]\s*\"?(\d{3})", svg))


def _luminance(hex_color):
    rgb = [int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


@pytest.mark.parametrize("token", ["ink", "primary_dark", "west", "muted"])
def test_text_colours_have_enough_contrast_on_paper(token):
    a, b = _luminance(theme.COLORS[token]), _luminance(theme.COLORS["paper"])
    ratio = (max(a, b) + 0.05) / (min(a, b) + 0.05)
    assert ratio >= (3.0 if token == "muted" else 4.5), (token, round(ratio, 2))  # muted is for secondary notes


def test_every_report_figure_has_png_and_svg():
    pngs = {p.stem for p in FIGURES_DIR.glob("*.png")}
    svgs = {p.stem for p in FIGURES_DIR.glob("*.svg")}
    assert len(pngs) >= 17 and pngs == svgs
