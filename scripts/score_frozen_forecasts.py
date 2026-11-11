"""Score the frozen 2026-09..11 forecasts once Census publishes the target months (pre-registration section 5).

    uv run python pipeline/fetch_census_hs10.py --end 2026-09     # after Census publishes September (needs CENSUS_API_KEY)
    uv run python scripts/run_all.py --write-sums && uv run python scripts/run_all.py
    uv run python scripts/score_frozen_forecasts.py

The frozen file is verified against its .sha256 before scoring and is never modified. Scores are written to
reports/frozen/score_<frozen file stem>.csv, one row per series and target month that has been published.

The frozen ML file (scripts/freeze_ml_forecasts.py) is scored on the same 8 bottom series, per model and target
month and pooled: WAPE over the 8 series and, per month, the error of their sum (sea + air total), next to
the primary frozen forecast.
"""

import hashlib
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from kbeauty import freeze  # noqa: E402

FROZEN_DIR = REPO / "reports" / "frozen"


def verified(path: Path) -> pd.DataFrame:
    expected = path.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        sys.exit(f"{path.name} does not match its .sha256 - frozen file was modified")
    return pd.read_csv(path)


def score_ml(primary: pd.DataFrame, actual: pd.DataFrame) -> None:
    for path in sorted(FROZEN_DIR.glob("ml_forecast_*_frozen_*.csv")):
        fc = verified(path)
        prim = primary[primary.series.isin(fc.series.unique())][["series", "target_month", "forecast_primary"]]
        prim = prim.rename(columns={"forecast_primary": "forecast"}).assign(model="frozen_primary")
        both = pd.concat([fc[["series", "target_month", "model", "forecast"]], prim], ignore_index=True)
        scored = both.merge(actual[["series", "target_month", "actual"]], on=["series", "target_month"], how="inner")
        if scored.empty:
            print(f"{path.name}: target months {sorted(fc.target_month.unique())} not published yet - nothing to score")
            continue
        scored = scored.assign(abs_err=(scored.forecast - scored.actual).abs())
        # Per target month, then pooled over the published months ('all'): the pre-registered decision uses
        # the pooled bottom-8 WAPE (pre-registration change log, 2026-11-02).
        pooled = scored.assign(target_month="all")
        g = pd.concat([scored, pooled]).groupby(["model", "target_month"])
        out = pd.DataFrame({"wape_bottom8": g.abs_err.sum() / g.actual.sum(),
                            "n_series_months": g.size()}).reset_index()
        monthly = scored.groupby(["model", "target_month"])[["forecast", "actual"]].sum()
        out = out.merge(((monthly.forecast - monthly.actual).abs() / monthly.actual).rename("abs_pct_error_total")
                        .reset_index(), on=["model", "target_month"], how="left")
        dest = FROZEN_DIR / f"score_{path.stem}.csv"
        out.to_csv(dest, index=False, lineterminator="\n")
        print(f"{dest.relative_to(REPO)}: months {sorted(out.target_month.unique())}")


def main() -> None:
    frozen_files = sorted(FROZEN_DIR.glob("forecast_*_frozen_*.csv"))
    if not frozen_files:
        sys.exit("no frozen forecast file")
    actual = pd.concat({name: s for name, s in freeze.series_to_forecast().items()}, names=["series", "month"])
    actual = actual.rename("actual").reset_index().assign(target_month=lambda d: d.month.dt.strftime("%Y-%m"))
    for path in frozen_files:
        fc = verified(path)
        scored = fc.merge(actual[["series", "target_month", "actual"]], on=["series", "target_month"], how="inner")
        if scored.empty:
            print(f"{path.name}: target months {sorted(fc.target_month.unique())} not published yet - nothing to score")
            continue
        scored["abs_pct_error"] = (scored.forecast_primary - scored.actual).abs() / scored.actual
        scored["inside_pi80"] = scored.actual.between(scored.pi80_low, scored.pi80_high)
        out = FROZEN_DIR / f"score_{path.stem}.csv"
        scored.to_csv(out, index=False, lineterminator="\n")
        print(f"{out.relative_to(REPO)}: {len(scored)} rows, months {sorted(scored.target_month.unique())}, "
              f"MAPE {scored.abs_pct_error.mean():.3f}, PI80 coverage {scored.inside_pi80.mean():.2f}")
    score_ml(verified(frozen_files[0]), actual)


if __name__ == "__main__":
    main()
