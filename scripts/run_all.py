"""Run the whole workflow from the committed raw data, with no network access.

    uv run python scripts/run_all.py

1. verify   SHA-256 of every raw and extracted file against data/SHA256SUMS
2. extract  rebuild data/extracted/*.csv from archived Census, KCS and Federal Reserve Board responses (offline)
            and check the rebuilt files are byte-identical to the committed ones
3. build    dbt build: seeds, staging, marts and data tests into warehouse/kbeauty.duckdb; prints passed/total
            data tests read from dbt's run_results.json (e.g. "49/49 data tests passed")
4. report   tables (reports/tables) and figures (reports/figures)

Use --write-sums once to (re)create data/SHA256SUMS after intentionally changing the raw data.
"""

import argparse
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from kbeauty import report, warehouse  # noqa: E402
from kbeauty.paths import (CENSUS_DATA_END, DATA_START, EXTRACTED_DIR, KCS_DATA_END, RAW_DIR,  # noqa: E402
                           SHA256SUMS)

TRACKED = sorted([*RAW_DIR.rglob("*.json"), *RAW_DIR.rglob("*.csv"), *RAW_DIR.rglob("*.gz"), *RAW_DIR.rglob("*.zip"),
                  *EXTRACTED_DIR.glob("*.csv")])

# (collector, extra arguments) — every collector re-runs from archived raw responses only.
EXTRACTORS = [
    ("fetch_census_hs10.py", ["--start", DATA_START, "--end", CENSUS_DATA_END]),
    ("fetch_census_port.py", ["--start", DATA_START, "--end", CENSUS_DATA_END]),
    ("fetch_census_peers.py", ["--start", DATA_START, "--end", CENSUS_DATA_END]),
    ("fetch_kcs_exports.py", ["--start", DATA_START, "--end", KCS_DATA_END]),
    ("fetch_fed_board_fx.py", []),
    ("build_hsk_names.py", []),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_sums() -> None:
    lines = [f"{sha256(p)}  {p.relative_to(REPO).as_posix()}" for p in TRACKED]
    # LF line endings so the file also works with `sha256sum -c data/SHA256SUMS`.
    SHA256SUMS.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    print(f"wrote {SHA256SUMS.relative_to(REPO)} ({len(lines)} files)")


def verify() -> None:
    expected = dict(reversed(line.split("  ", 1)) for line in SHA256SUMS.read_text(encoding="utf-8").splitlines() if line)
    bad = [rel for rel, digest in expected.items() if not (REPO / rel).exists() or sha256(REPO / rel) != digest]
    if bad:
        sys.exit(f"hash mismatch: {bad}")
    print(f"verify   {len(expected)} files match data/SHA256SUMS")


def extract() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        for script, extra in EXTRACTORS:
            subprocess.run([sys.executable, str(REPO / "pipeline" / script), "--offline", *extra, "--out", tmp],
                           check=True, capture_output=True)
        for rebuilt in sorted(Path(tmp).glob("*.csv")):
            committed = EXTRACTED_DIR / rebuilt.name
            if sha256(rebuilt) != sha256(committed):
                sys.exit(f"extract  {rebuilt.name} differs from the committed file")
    print("extract  rebuilt CSVs are byte-identical to data/extracted")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-sums", action="store_true")
    args = parser.parse_args()
    if args.write_sums:
        write_sums()
        return
    verify()
    extract()
    s = warehouse.build()
    print(f"build    {s['tests_passed']}/{s['tests_total']} data tests passed "
          f"({s['models']} models, {s['seeds']} seeds)")
    if s["tests_passed"] != s["tests_total"]:  # dbt exits 0 on 'warn' tests; a warning still fails the run
        sys.exit(f"build    data tests not passed: {s['not_passed']}")
    results = report.generate_all()
    print(f"report   {len(results)} tables, figures in reports/figures")


if __name__ == "__main__":
    main()
