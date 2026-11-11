"""Print the case study (portfolio/index.html) to portfolio/case_study.pdf with headless Edge/Chromium.

    uv run --with playwright python scripts/build_case_study_pdf.py

The HTML loads the report figures by relative path, so re-run this after the figures change.
"""

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "portfolio" / "index.html"
OUT = REPO / "portfolio" / "case_study.pdf"


def main(channel: str = "msedge") -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(channel=channel, args=["--allow-file-access-from-files"])
        page = browser.new_page()
        page.goto(SRC.as_uri())
        page.wait_for_load_state("networkidle")
        page.evaluate("document.fonts.ready")
        page.pdf(path=str(OUT), format="A4", prefer_css_page_size=True, print_background=True)
        browser.close()
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main(*sys.argv[1:])
