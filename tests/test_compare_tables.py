import importlib.util

import pandas as pd

from kbeauty.paths import REPO

spec = importlib.util.spec_from_file_location("compare_tables", REPO / "scripts" / "compare_tables.py")
compare_tables = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare_tables)


def frame(value, label="naive"):
    return pd.DataFrame({"model": [label, "ets"], "wape": [value, 0.2]})


def test_last_digit_differences_are_within_tolerance():
    assert compare_tables.compare_frames(frame(0.111957996711), frame(0.111957996712), 1e-6, 1e-9) is None


def test_real_numeric_change_is_reported():
    assert "wape" in compare_tables.compare_frames(frame(0.12), frame(0.11), 1e-6, 1e-9)


def test_text_must_match_exactly():
    assert "model" in compare_tables.compare_frames(frame(0.11, "xgboost"), frame(0.11, "lightgbm"), 1e-6, 1e-9)
