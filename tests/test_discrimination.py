import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from modelcourt.discrimination import (
    auc,
    bootstrap_auc_ci,
    cap_curve,
    discrimination_findings,
    gini,
    ks,
)


def test_auc_matches_sklearn():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    s = rng.normal(size=300) + y
    assert auc(y, s) == pytest.approx(roc_auc_score(y, s))


def test_gini_identity():
    assert gini(0.8) == pytest.approx(0.6)
    assert gini(0.5) == pytest.approx(0.0)


def test_ks_matches_manual_cdf_difference():
    """Positive control: KS computed by hand over every threshold."""
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 200)
    s = rng.normal(size=200) + 0.8 * y
    thresholds = np.unique(s)
    diffs = [
        abs(np.mean(s[y == 1] <= t) - np.mean(s[y == 0] <= t))
        for t in thresholds
    ]
    assert ks(y, s) == pytest.approx(max(diffs))


def test_perfect_separation_gives_max_scores():
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert auc(y, s) == pytest.approx(1.0)
    assert ks(y, s) == pytest.approx(1.0)


def test_random_score_gives_auc_near_half():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 2000)
    s = rng.random(2000)
    assert 0.45 < auc(y, s) < 0.55


def test_cap_curve_ends_at_one_and_starts_above_diagonal():
    y = np.array([0, 0, 1, 0, 1, 1, 0, 0, 1, 0])
    s = np.array([0.1, 0.2, 0.9, 0.3, 0.8, 0.7, 0.05, 0.15, 0.6, 0.25])
    cap = cap_curve(y, s)
    assert cap["population_share"].iloc[-1] == pytest.approx(1.0)
    assert cap["bad_captured_share"].iloc[-1] == pytest.approx(1.0)
    # Top 10% (1 row, the riskiest) captures a defaulted row here: above the diagonal.
    assert cap["bad_captured_share"].iloc[0] > cap["population_share"].iloc[0]


def test_bootstrap_interval_contains_point_estimate():
    rng = np.random.default_rng(4)
    y = rng.integers(0, 2, 300)
    s = rng.normal(size=300) + y
    lo, hi, n_used = bootstrap_auc_ci(y, s, n_boot=500)
    point = auc(y, s)
    assert lo <= point <= hi
    assert n_used > 400


def test_inverted_score_is_flagged_not_flipped():
    """AUC < 0.5 must be reported as inverted, and the AUC must stay as computed."""
    rng = np.random.default_rng(5)
    y = rng.integers(0, 2, 300)
    s = -(rng.normal(size=300) + y)  # higher score = safer: inverted
    rows = discrimination_findings(y, s, "test")
    auc_row = next(r for r in rows if r["test_id"] == "M2.auc")
    assert auc_row["value"] < 0.5
    assert "inverted" in auc_row["note"]
    assert auc_row["decision"] == "fail"


def test_findings_have_expected_ids():
    rng = np.random.default_rng(6)
    y = rng.integers(0, 2, 300)
    s = rng.normal(size=300) + y
    ids = [r["test_id"] for r in discrimination_findings(y, s, "test")]
    assert ids == ["M2.auc", "M2.gini", "M2.ks"]
