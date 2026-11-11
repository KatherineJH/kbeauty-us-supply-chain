"""CI check (draft): run the forecasting and ML stages twice in fresh processes in the same job, and fail if any
recorded bit differs.

    uv run python scripts/check_rerun_determinism.py               # two fresh passes, then compare
    uv run python scripts/check_rerun_determinism.py --compare A.json B.json   # compare two existing passes

Each pass is `scripts/diag_determinism.py --no-probes`: the production `report.generate_all` with every stage
hashed (statistical fits, ML folds, ML selection, reconciliation, tables). The warehouse must already be built
(run after scripts/run_all.py). Both passes reuse that warehouse, so a difference here comes from the
forecasting code on this machine, not from the inputs.

This catches nondeterminism within one machine. It cannot catch a difference between two runner machines
(different CPUs); that needs the kernel pinning and the CPU record described in
docs/experiments/linux_determinism.md.
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from diag_compare import describe, first_diffs, load  # noqa: E402

CHECKED = ("10_input", "20_series", "30_features", "40_stat_fit", "45_stat_table", "50_ml_fold", "60_ml_select",
           "70_reconcile", "80_synth", "90_table")


def compare(a: Path, b: Path) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        for name, src in (("A", a), ("B", b)):
            (Path(tmp) / f"{name}.json").write_bytes(src.read_bytes())
        ra, rb = sorted(load(Path(tmp)), key=lambda r: r["file"])
    _, per = first_diffs(ra, rb)
    bad = {s: per[s] for s in CHECKED if s in per}
    keys_a = {k for k in ra["recs"] if k[0] in CHECKED}
    if not bad and keys_a == {k for k in rb["recs"] if k[0] in CHECKED}:
        n = len(keys_a)
        print(f"rerun-determinism  OK: {n} stage records identical across two passes (pipeline fp {ra['pipe_fp']})")
        return 0
    print("rerun-determinism  FAIL: the two passes in this job differ. First differing record per stage:")
    for s in CHECKED:
        if s in bad:
            print("\n".join(describe(bad[s])))
    return 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", nargs=2, type=Path)
    args = ap.parse_args()
    if args.compare:
        sys.exit(compare(*args.compare))
    with tempfile.TemporaryDirectory() as tmp:
        outs = []
        for name in ("A", "B"):
            out = Path(tmp) / f"{name}.json"
            subprocess.run([sys.executable, str(REPO / "scripts" / "diag_determinism.py"), "--no-probes",
                            "--out", str(out), "--label", f"rerun-{name}"], check=True)
            outs.append(out)
        sys.exit(compare(*outs))


if __name__ == "__main__":
    main()
