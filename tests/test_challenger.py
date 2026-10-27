import numpy as np
import pytest

from modelcourt.challenger import challenger_findings, classify_difference, paired_auc_difference


def test_identical_scores_give_zero_difference():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    s = rng.normal(size=300) + y
    diff, lo, hi = paired_auc_difference(y, s, s, n_boot=200)
    assert diff == pytest.approx(0.0)
    assert lo <= 0.0 <= hi


def test_better_challenger_gives_positive_difference():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 2000)
    strong = rng.normal(size=2000) + 2.0 * y
    weak = rng.normal(size=2000) + 0.2 * y
    diff, _, _ = paired_auc_difference(y, weak, strong, n_boot=200)
    assert diff > 0.2


def test_classification_bands():
    assert classify_difference(-0.10) == ("pass", "none")
    assert classify_difference(0.01) == ("pass", "none")
    assert classify_difference(0.03) == ("conditional", "low")
    assert classify_difference(0.08) == ("fail", "high")


def test_findings_shape():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 300)
    s = rng.normal(size=300) + y
    (row,) = challenger_findings(y, s, s, "t")
    assert row["test_id"] == "M6.challenger_auc_diff"
    assert row["decision"] == "pass"
