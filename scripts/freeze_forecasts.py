"""Write the pre-registered 2026-09..11 forecasts once and record their SHA-256. Never overwrites.

    uv run python scripts/freeze_forecasts.py

Scoring happens later in a separate file (reports/frozen/score_*.csv) when Census publishes the months.
"""

import hashlib
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from kbeauty import freeze  # noqa: E402

OUT_DIR = REPO / "reports" / "frozen"


def main() -> None:
    df = freeze.build()
    first, last = df.target_month.min(), df.target_month.max()
    out = OUT_DIR / f"forecast_{first}_to_{last}_frozen_{date.today().isoformat()}.csv"
    if out.exists() or list(OUT_DIR.glob(f"forecast_{first}_to_{last}_frozen_*.csv")):
        sys.exit(f"a frozen forecast for {first}..{last} already exists; frozen files are never rewritten")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = df.to_csv(index=False, lineterminator="\n").encode("utf-8")
    out.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    out.with_suffix(".sha256").write_bytes(f"{digest}  {out.name}\n".encode("utf-8"))
    print(f"frozen {out.relative_to(REPO)} ({len(df)} rows) sha256={digest[:16]}...")


if __name__ == "__main__":
    main()
