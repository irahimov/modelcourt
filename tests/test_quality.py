import numpy as np
import pandas as pd
import pytest

from modelcourt.quality import (
    FINDING_COLUMNS,
    classify,
    directional_auc,
    duplicate_rows,
    findings_to_frame,
    missing_rates,
    single_feature_leakage,
    unexpected_categories,
)


def test_classify_bands_and_boundaries():
    assert classify(0.5, 1, 5, "medium") == ("pass", "none")
    assert classify(1, 1, 5, "medium") == ("conditional", "low")   # boundary belongs to conditional
    assert classify(3, 1, 5, "medium") == ("conditional", "low")
    assert classify(5, 1, 5, "medium") == ("conditional", "low")
    assert classify(7, 1, 5, "medium") == ("fail", "medium")


def test_missing_rate_known_value():
    df = pd.DataFrame({"a": [1, None, 3, 4, 5, 6, 7, 8, 9, 10], "b": range(10)})
    rows = {r["metric"]: r for r in missing_rates(df, "test")}
    assert rows["missing_pct:a"]["value"] == 10.0
    assert rows["missing_pct:a"]["decision"] == "fail"
    assert rows["missing_pct:b"]["decision"] == "pass"


def test_duplicates_counted_once_per_repeat():
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    (row,) = duplicate_rows(df, "test")
    assert row["value"] == 1
    assert row["decision"] == "conditional"


def test_directional_auc_ignores_sign():
    y = np.array([0, 0, 1, 1, 0, 1])
    x = np.array([1, 2, 5, 6, 3, 7])
    assert directional_auc(y, x) == pytest.approx(directional_auc(y, -x))


def test_planted_leakage_is_caught():
    """Positive control: a column built from the target must be flagged, noise must pass."""
    rng = np.random.default_rng(0)
    y = pd.Series(rng.integers(0, 2, 200))
    df = pd.DataFrame({
        "noise": rng.normal(size=200),
        "leak": y + rng.normal(scale=0.01, size=200),
    })
    rows = {r["metric"]: r for r in single_feature_leakage(df, y, "test", ["noise", "leak"], [])}
    assert rows["single_feature_auc_dir:leak"]["decision"] == "fail"
    assert rows["single_feature_auc_dir:leak"]["severity"] == "critical"
    assert rows["single_feature_auc_dir:noise"]["decision"] == "pass"


def test_categorical_level_scored_one_vs_rest():
    y = pd.Series([0, 0, 0, 1, 1, 1])
    df = pd.DataFrame({"c": ["a", "a", "a", "b", "b", "b"]})
    rows = {r["metric"]: r for r in single_feature_leakage(df, y, "test", [], ["c"])}
    assert rows["single_feature_auc_dir:c=b"]["value"] == 1.0
    assert rows["single_feature_auc_dir:c=b"]["decision"] == "fail"


def test_unexpected_categories():
    df = pd.DataFrame({"housing": ["own", "rent", "own", "castle"]})
    (row,) = unexpected_categories(df, {"housing": {"own", "rent", "free"}}, "test")
    assert row["value"] == 1
    assert "castle" in row["note"]
    assert row["decision"] == "conditional"


def test_findings_frame_has_required_columns():
    df = pd.DataFrame({"a": [1, 2]})
    out = findings_to_frame(missing_rates(df, "test"))
    assert list(out.columns) == FINDING_COLUMNS
