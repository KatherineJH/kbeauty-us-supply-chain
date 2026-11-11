"""Integration: the committed raw data rebuild the same extracted files and a warehouse that passes dbt tests."""

import hashlib
import subprocess
import sys

import duckdb
import pytest

from kbeauty import warehouse
from kbeauty.paths import CENSUS_DATA_END, EXTRACTED_DIR, REPO, SHA256SUMS

pytestmark = pytest.mark.integration


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_raw_and_extracted_files_match_recorded_hashes():
    for line in SHA256SUMS.read_text(encoding="utf-8").splitlines():
        digest, rel = line.split("  ", 1)
        assert sha(REPO / rel) == digest, rel


def test_offline_extraction_is_byte_identical(tmp_path):
    for script in ("fetch_census_hs10.py", "fetch_census_port.py"):
        subprocess.run([sys.executable, str(REPO / "pipeline" / script), "--offline", "--end", CENSUS_DATA_END,
                        "--out", str(tmp_path)], check=True, capture_output=True)
    subprocess.run([sys.executable, str(REPO / "pipeline" / "fetch_fed_board_fx.py"), "--offline",
                    "--out", str(tmp_path)], check=True, capture_output=True)
    for name in ("hs10_monthly.csv", "hs6_port_monthly.csv", "hs10_district_monthly.csv", "fx_monthly.csv"):
        assert sha(tmp_path / name) == sha(EXTRACTED_DIR / name), name

    # Federal Reserve Board exchange rates: the pound series publishes UNIT_MULT="0.00999999978" but its values are
    # U.S. dollars per pound (2024-01: 1.2701, as in the Board's G.5 table). The multiplier must not be applied,
    # and no month after the configured end may enter the extract.
    sys.path.insert(0, str(REPO / "pipeline"))
    import fetch_fed_board_fx as fx
    from kbeauty.paths import FX_DATA_END
    parsed = fx.parse((fx.RAW / fx.ZIP_NAME).read_bytes())
    uk = parsed["RXI$US_N.M.UK"]
    assert uk["attrs"]["UNIT_MULT"] == "0.00999999978" and uk["obs"]["2024-01"] == "1.2701"
    assert fx.lcu_per_usd("RXI$US_N.M.UK", "1.2701") == round(1 / 1.2701, 6)  # pounds per dollar, no multiplier
    months = sorted({line.split(",")[0] for line in (tmp_path / "fx_monthly.csv").read_text().splitlines()[1:]})
    assert months[-1] == FX_DATA_END and max(uk["obs"]) > FX_DATA_END  # later months exist but are not extracted


@pytest.fixture(scope="module")
def built_db(tmp_path_factory):
    db = tmp_path_factory.mktemp("wh") / "test.duckdb"
    summary = warehouse.build(db)  # raises if any dbt model or data test fails
    return db, summary


def test_dbt_summary_counts_every_data_test_as_passed(built_db):
    _, s = built_db
    assert s["tests_total"] >= 49 and s["tests_passed"] == s["tests_total"] and not s["not_passed"]
    assert (s["models"], s["seeds"]) == (16, 5)


def test_run_summary_does_not_count_warnings_as_passed(tmp_path):
    f = tmp_path / "run_results.json"
    f.write_text('{"results": [{"unique_id": "test.a", "status": "pass"}, {"unique_id": "test.b", "status": "warn"},'
                 ' {"unique_id": "model.m", "status": "success"}]}', encoding="utf-8")
    s = warehouse.run_summary(f)
    assert (s["tests_passed"], s["tests_total"], s["not_passed"], s["models"]) == (1, 2, ["test.b"], 1)


def test_warehouse_covers_32_contiguous_months(built_db):
    with duckdb.connect(str(built_db[0]), read_only=True) as con:
        n, first, last = con.execute("select count(*), min(month), max(month) from mart_monthly_summary").fetchone()
    assert (n, str(first), str(last)) == (32, "2024-01-01", "2026-08-01")


def test_long_fact_table_preserves_national_total(built_db):
    with duckdb.connect(str(built_db[0]), read_only=True) as con:
        fact = con.execute("select sum(value_usd) from fct_imports_hs10_mode_monthly").fetchone()[0]
        stg = con.execute("select sum(value_usd) from stg_census__hs10_monthly").fetchone()[0]
    assert fact == stg


def test_data_dictionary_describes_every_warehouse_column(built_db):
    from kbeauty import documentation
    dd = documentation.data_dictionary(db_path=built_db[0])
    assert len(dd) >= 171 and dd.table.nunique() == 21
    missing = dd[(dd.description == "") | (dd.unit == "")][["table", "column"]]
    assert missing.empty, f"add these to warehouse/column_glossary.yml: {missing.values.tolist()}"
    assert (dd[dd.layer != "seed"].table_description != "").all()
