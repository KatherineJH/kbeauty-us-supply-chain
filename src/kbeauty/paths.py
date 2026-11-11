from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RAW_DIR = REPO / "data" / "raw"
EXTRACTED_DIR = REPO / "data" / "extracted"
SHA256SUMS = REPO / "data" / "SHA256SUMS"
WAREHOUSE_DIR = REPO / "warehouse"
DUCKDB_PATH = WAREHOUSE_DIR / "kbeauty.duckdb"
FIGURES_DIR = REPO / "reports" / "figures"
TABLES_DIR = REPO / "reports" / "tables"

# Fixed analysis dates (docs/preregistration.md). A monthly data refresh must not change these.
# Seasonally matched 12-month windows on either side of the August 2025 policy changes
# (15% reciprocal tariff on 2025-08-07, de minimis suspension on 2025-08-29).
PRE_WINDOW = ("2024-09-01", "2025-08-01")
POST_WINDOW = ("2025-09-01", "2026-08-01")
POLICY_MONTH = "2025-09-01"  # first full month after both changes
BACKTEST_FIRST_ORIGIN = "2025-12-01"  # first rolling origin with 24+ months of training (also used by the frozen forecasts)

# Data coverage: the only dates to move when new months are collected (then run_all.py --write-sums).
DATA_START = "2024-01"
CENSUS_DATA_END = "2026-08"  # last month in the archived US Census responses
KCS_DATA_END = "2026-09"     # Korean customs publishes about three weeks before Census
FX_DATA_START = "2023-01"    # exchange rates start a year earlier (12-month windows before DATA_START)
FX_DATA_END = "2026-09"      # Federal Reserve Board G.5 monthly averages: months later than this are not extracted
