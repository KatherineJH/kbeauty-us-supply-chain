"""Short experiment: do dbt/DuckDB threads cause the last-bit differences between dbt builds of some marts?

    uv run python scripts/diag_dbt_threads.py run --arm current --out diag/current-r1.json   # two passes (A, B)
    uv run python scripts/diag_dbt_threads.py summarize DIR --md summary.md

Each pass builds the warehouse from scratch (dbt build), hashes every model table order-free with exact float
bits, then runs report.generate_all and hashes the 31 result tables. The arm is set by the workflow, which
edits warehouse/profiles.yml before the run:
  current   profile as committed (dbt threads 4, DuckDB's own thread pool at its default)
  dbt1      dbt threads 1 (one model at a time)
  duckdb1   DuckDB setting threads 1 (no parallel aggregation inside a query), dbt threads as committed
Run in the pinned numeric environment (src/kbeauty/numeric_env.py), so the forecasting stages do not add noise.
"""

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from diag_determinism import h_df  # noqa: E402

from kbeauty import report, warehouse  # noqa: E402
from kbeauty.numeric_env import record  # noqa: E402
from kbeauty.paths import DUCKDB_PATH, TABLES_DIR  # noqa: E402


def mart_hashes() -> dict[str, str]:
    import duckdb
    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    try:
        names = [r[0] for r in con.execute(
            "select table_name from information_schema.tables where table_schema = 'main' order by 1").fetchall()]
        return {n: h_df(con.execute(f'select * from "{n}"').df(), ordered=False) for n in names}
    finally:
        con.close()


def one_pass() -> dict:
    DUCKDB_PATH.unlink(missing_ok=True)  # a fresh database for every build
    t0 = time.perf_counter()
    s = warehouse.build()
    t1 = time.perf_counter()
    marts = mart_hashes()
    t2 = time.perf_counter()
    report.generate_all()
    t3 = time.perf_counter()
    tables = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in sorted(TABLES_DIR.glob("*.csv"))}
    return {"tests": f"{s['tests_passed']}/{s['tests_total']}", "dbt_seconds": round(t1 - t0, 2),
            "report_seconds": round(t3 - t2, 2), "marts": marts, "tables": tables}


def run(arm: str, out: Path) -> None:
    rec = record(REPO)
    profile = (REPO / "warehouse" / "profiles.yml").read_text(encoding="utf-8")
    passes = {name: one_pass() for name in ("A", "B")}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"arm": arm, "profile": profile, "cpu": rec["runner"]["cpu"]["model"],
                               "avx512": bool(rec["runner"]["cpu"]["avx512_exposed"]), "passes": passes},
                              indent=1), encoding="utf-8")
    for k, p in passes.items():
        print(f"{arm} {k}: dbt {p['dbt_seconds']}s, report {p['report_seconds']}s, tests {p['tests']}")


def fp(d: dict) -> str:
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]


def summarize(folder: Path, md: Path | None) -> None:
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(folder.rglob("*.json"))]
    by_arm = defaultdict(list)
    for r in runs:
        by_arm[r["arm"]].append(r)
    lines = ["| arm | jobs | passes | distinct mart fingerprints | A/B mart differences (jobs) | marts that ever "
             "differ | distinct table fingerprints | dbt s (median) | report s (median) |", "|---|---|---|---|---|---|---|---|---|"]
    detail = []
    for arm in ("current", "dbt1", "duckdb1"):
        rs = by_arm.get(arm, [])
        if not rs:
            continue
        passes = [p for r in rs for p in r["passes"].values()]
        mart_fps = Counter(fp(p["marts"]) for p in passes)
        table_fps = Counter(fp(p["tables"]) for p in passes)
        ab = sum(r["passes"]["A"]["marts"] != r["passes"]["B"]["marts"] for r in rs)
        names = sorted({n for p in passes for n in p["marts"]})
        varying = [n for n in names if len({p["marts"].get(n) for p in passes}) > 1]
        med = lambda xs: sorted(xs)[len(xs) // 2]  # noqa: E731
        lines.append(f"| {arm} | {len(rs)} | {len(passes)} | {len(mart_fps)} | {ab} of {len(rs)} | "
                     f"{', '.join(varying) or 'none'} | {len(table_fps)} | {med([p['dbt_seconds'] for p in passes])} | "
                     f"{med([p['report_seconds'] for p in passes])} |")
        for r in rs:
            detail.append(f"| {arm} | {r['cpu']} | {'yes' if r['avx512'] else 'no'} | "
                          + " | ".join(f"{fp(r['passes'][k]['marts'])} / {fp(r['passes'][k]['tables'])} / "
                                       f"{r['passes'][k]['dbt_seconds']}s" for k in ("A", "B")) + " |")
    text = "\n".join(lines + ["", "| arm | CPU | AVX-512 | pass A (marts / tables / dbt) | pass B |",
                              "|---|---|---|---|---|", *detail]) + "\n"
    print(text)
    if md:
        md.write_text(text, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--arm", required=True)
    r.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("summarize")
    s.add_argument("folder", type=Path)
    s.add_argument("--md", type=Path)
    a = ap.parse_args()
    run(a.arm, a.out) if a.cmd == "run" else summarize(a.folder, a.md)


if __name__ == "__main__":
    main()
