import numpy as np
import pandas as pd
import pytest
from scipy.stats import binomtest
from sklearn.metrics import brier_score_loss

from modelcourt.calibration import (
    brier,
    brier_skill,
    bucket_table,
    calibration_findings,
    overall_binomial,
)


def _sample(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.05, 0.6, n)
    y = rng.binomial(1, p)
    return y, p


def test_brier_matches_sklearn():
    y, p = _sample(500)
    assert brier(y, p) == pytest.approx(brier_score_loss(y, p))


def test_brier_skill_zero_for_constant_base_rate():
    y = np.array([0, 0, 1, 0, 1, 0, 0, 0, 0, 1])
    base = 0.3
    assert brier_skill(y, np.full(10, base), base) == pytest.approx(
        1 - brier(y, np.full(10, base)) / np.mean((y - base) ** 2)
    )
    assert brier_skill(y, np.full(10, base), base) == pytest.approx(0.0, abs=1e-9)


def test_well_calibrated_sample_passes():
    """Positive control: a calibrated model must not FAIL.

    A 'conditional' bucket result is possible by chance. With 5 buckets and no
    multiplicity correction, P(at least one bucket p<0.05) is about 0.23 under the null,
    which is why the threshold allows one conditional bucket.
    """
    y, p = _sample(4000, seed=1)
    rows = {r["test_id"]: r for r in calibration_findings(y, p, base_rate=y.mean(), dataset="t")}
    assert rows["M3.overall"]["decision"] == "pass"
    assert rows["M3.buckets"]["decision"] != "fail"


def test_bucket_fail_rate_on_calibrated_models_is_low():
    """Across 30 calibrated samples, the bucket test should rarely reach 'fail' (>=2 buckets).

    Under the null, P(>=2 of 5 buckets with p<0.05) is about 0.02. The bound 0.15 leaves room for noise.
    """
    fails = 0
    for seed in range(100, 130):
        y, p = _sample(4000, seed=seed)
        r = {x["test_id"]: x for x in calibration_findings(y, p, base_rate=y.mean(), dataset="t")}
        if r["M3.buckets"]["decision"] == "fail":
            fails += 1
    assert fails / 30 <= 0.15


def test_overconfident_model_is_caught():
    """Negative control: true PD is p, but the model reports 2p (capped). Must not pass."""
    y, p = _sample(4000, seed=2)
    reported = np.minimum(2 * p, 0.99)
    rows = {r["test_id"]: r for r in calibration_findings(y, reported, base_rate=y.mean(), dataset="t")}
    assert rows["M3.overall"]["decision"] == "fail"
    assert rows["M3.overall"]["severity"] == "high"
    assert rows["M3.brier_skill"]["decision"] != "pass"


def test_bucket_sizes_are_equal_and_sum_to_n():
    y, p = _sample(500, seed=3)
    t = bucket_table(y, p, n_bins=5)
    assert t["n"].sum() == 500
    assert t["n"].max() - t["n"].min() <= 1


def test_tied_scores_do_not_break_buckets():
    y = np.array([0, 1] * 50)
    p = np.repeat([0.2, 0.4], 50)
    t = bucket_table(y, p, n_bins=5)
    assert len(t) == 5
    assert t["n"].sum() == 100


def test_overall_binomial_matches_scipy():
    y = np.array([1] * 30 + [0] * 70)
    p = np.full(100, 0.2)
    res = overall_binomial(y, p)
    assert res["p_value"] == pytest.approx(binomtest(30, 100, 0.2).pvalue)
    assert res["defaults"] == 30
