"""Build the DuckDB warehouse with dbt and read its marts."""

import json
import os
import subprocess
import sys
from pathlib import Path

import duckdb
import pandas as pd

from .paths import DUCKDB_PATH, REPO, WAREHOUSE_DIR


def build(db_path: Path = DUCKDB_PATH) -> dict:
    """Run `dbt build` (seeds, models, data tests) from the repository root. Raises on any failure.
    Returns node counts from target/run_results.json: data tests total and passed (status 'pass'; 'warn' and
    'fail' are not passed), models and seeds built, and the ids of tests that did not pass."""
    env = {**os.environ, "KBEAUTY_DUCKDB": str(db_path)}
    dbt = Path(sys.executable).with_name("dbt.exe" if os.name == "nt" else "dbt")
    cmd = [str(dbt), "build", "--project-dir", str(WAREHOUSE_DIR), "--profiles-dir", str(WAREHOUSE_DIR)]
    result = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"dbt build failed:\n{result.stdout[-4000:]}\n{result.stderr[-2000:]}")
    return run_summary(WAREHOUSE_DIR / "target" / "run_results.json")


def run_summary(run_results: Path) -> dict:
    results = json.loads(run_results.read_text(encoding="utf-8"))["results"]
    tests = [r for r in results if r["unique_id"].startswith("test.")]
    return {"tests_total": len(tests), "tests_passed": sum(r["status"] == "pass" for r in tests),
            "not_passed": [r["unique_id"] for r in tests if r["status"] != "pass"],
            "models": sum(r["unique_id"].startswith("model.") for r in results),
            "seeds": sum(r["unique_id"].startswith("seed.") for r in results)}


def load(table: str, db_path: Path = DUCKDB_PATH) -> pd.DataFrame:
    """Read one mart as a DataFrame. Date columns come back as pandas Timestamps."""
    if not table.replace("_", "").isalnum():
        raise ValueError(f"invalid table name: {table}")
    with duckdb.connect(str(db_path), read_only=True) as con:
        return con.sql(f"select * from {table}").df()
