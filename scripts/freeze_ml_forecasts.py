"""Freeze the machine-learning forecasts for the months after the data, once, with their SHA-256. Never overwrites.

    uv run python scripts/freeze_ml_forecasts.py

The backtest (notebook 05, section 5) found that each boosting model beat the naive forecast, but that finding
was read off the test months. Freezing every ML model's forecast before Census publishes the target months turns
it into a genuine out-of-sample test (pre-registration change log, 2026-11-02). Rows: 8 bottom series (group x
mode, sea + air import value) x horizons 1-3 x 10 ML forecasts (9 models + the validation-selected one).
Scored with the primary frozen forecast by scripts/score_frozen_forecasts.py.
"""

import hashlib
import sys
from datetime import date
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from kbeauty import hierarchy, ml, warehouse  # noqa: E402
from kbeauty.paths import POLICY_MONTH  # noqa: E402

OUT_DIR = REPO / "reports" / "frozen"


def main() -> None:
    bottom = hierarchy.bottom_series(warehouse.load("fct_imports_hs10_mode_monthly"))
    krw = warehouse.load("stg_fed_board__fx_monthly").query("currency == 'KRW'")
    krw = krw.assign(month=pd.to_datetime(krw["month"])).set_index("month")["lcu_per_usd"]
    df = ml.forecast_from_last(bottom, krw, POLICY_MONTH)
    # Same series names as the primary frozen file ('skincare_vessel|value_usd'), so both score on one key.
    df.insert(0, "series", df.pop("node").str.replace("|", "_") + "|value_usd")
    df["forecast"] = df["forecast"].round().astype("int64")
    df["inner_wape"] = df["inner_wape"].round(6)
    first, last = df.target_month.min(), df.target_month.max()
    stem = f"ml_forecast_{first}_to_{last}_frozen_"
    if list(OUT_DIR.glob(f"{stem}*.csv")):
        sys.exit(f"an ML frozen forecast for {first}..{last} already exists; frozen files are never rewritten")
    out = OUT_DIR / f"{stem}{date.today().isoformat()}.csv"
    data = df.to_csv(index=False, lineterminator="\n").encode("utf-8")
    out.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    out.with_suffix(".sha256").write_bytes(f"{digest}  {out.name}\n".encode("utf-8"))
    print(f"frozen {out.relative_to(REPO)} ({len(df)} rows) sha256={digest[:16]}...")


if __name__ == "__main__":
    main()
