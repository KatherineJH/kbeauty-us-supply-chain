"""Record the numeric environment, check it is pinned, and write the canonical Linux reference.

    uv run python scripts/platform_reference.py verify               # print the record; on Linux fail unless pinned
    uv run python scripts/platform_reference.py write --out DIR      # copy reports/tables/*.csv to DIR + MANIFEST.json

`write` runs only in the bootstrap workflow (.github/workflows/canonical-bootstrap.yml), only in the pinned Linux
environment, and only produces an artifact. A person reviews it and commits it to reports/platform/linux/ in a
separate commit; the verification CI on a new runner must then reproduce all 31 tables byte for byte.

The canonical reference (reports/platform/linux/) is the output of the pinned Linux CI run: all 31 result
tables plus MANIFEST.json, which records the generating commit and workflow run, the time, the runner (CPU
model, AVX-512 exposure, image), package, BLAS and numpy SIMD versions, the pinned environment variables and
the SHA-256 of every table. CI regenerates the tables in the same pinned environment and requires every file
to be byte-identical to it (scripts/compare_tables.py). See src/kbeauty/numeric_env.py for why the pins exist.
"""

import argparse
import hashlib
import json
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from kbeauty.numeric_env import pin_problems, record  # noqa: E402
from kbeauty.paths import TABLES_DIR  # noqa: E402


def summary(rec: dict) -> str:
    cpu = rec["runner"]["cpu"]
    blas = ", ".join(f"{b['prefix']} {b['version']} {b['architecture']} x{b['num_threads']}" for b in rec["blas"]
                     if b.get("internal_api") == "openblas")
    return (f"env      cpu {cpu['model']} (AVX-512 exposed: {'yes' if cpu['avx512_exposed'] else 'no'}), "
            f"numpy SIMD {rec['numpy_simd']['enabled']}, BLAS {blas}")


def verify() -> None:
    rec = record(REPO)
    print(summary(rec))
    problems = pin_problems(rec)
    if platform.system() != "Linux":
        print("env      not Linux: pins are only required in the canonical Linux environment"
              + (f" (unpinned: {len(problems)})" if problems else ""))
        return
    if problems:
        print("env      NOT the pinned canonical environment:")
        print("\n".join(f"  {p}" for p in problems))
        sys.exit(1)
    print("env      pinned canonical environment (threads 1, OpenBLAS Haswell, numpy without AVX-512)")


def write(out: Path) -> None:
    rec = record(REPO)
    if platform.system() != "Linux" or pin_problems(rec):
        sys.exit("refusing to write a canonical reference outside the pinned Linux environment: "
                 f"{platform.system()} {pin_problems(rec)}")
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for src in sorted(TABLES_DIR.glob("*.csv")):
        shutil.copyfile(src, out / src.name)
        data = src.read_bytes()
        files[src.name] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    manifest = {"generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), **rec,
                "tables": files}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(summary(rec))
    print(f"wrote    {len(files)} tables and MANIFEST.json to {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("verify")
    w = sub.add_parser("write")
    w.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    verify() if args.cmd == "verify" else write(args.out)


if __name__ == "__main__":
    main()
