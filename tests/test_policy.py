import numpy as np
import pytest

from modelcourt.policy import (
    bootstrap_incremental_ci,
    breakeven_pd,
    policy_summary,
    realized_profit,
    stop_rules,
)


def test_breakeven_formula():
    assert breakeven_pd(5) == pytest.approx(1 / 6)
    assert breakeven_pd(3) == pytest.approx(0.25)


def test_realized_profit_manual_example():
    # approve all: good +1, bad -5, good +1  -> -3
    assert realized_profit([0, 1, 0], [True, True, True], 5) == pytest.approx(-3.0)
    # reject the bad one: +1 +1 -> +2
    assert realized_profit([0, 1, 0], [True, False, True], 5) == pytest.approx(2.0)


def test_policy_approves_only_below_cutoff():
    y = [0, 1, 0, 1]
    p = [0.10, 0.20, 0.05, 0.50]
    s = policy_summary(y, p, loss_multiple=5)  # cutoff = 1/6 = 0.1667
    assert s["approval_rate"] == pytest.approx(0.5)
    assert s["bad_rate_among_approved"] == pytest.approx(0.0)


def test_perfect_scores_beat_random_scores():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.3, 3000)
    perfect = np.where(y == 1, 0.6, 0.1)
    random_scores = rng.uniform(0.05, 0.6, 3000)
    inc_perfect = policy_summary(y, perfect, 5)["incremental_per_applicant"]
    inc_random = policy_summary(y, random_scores, 5)["incremental_per_applicant"]
    assert inc_perfect > inc_random


def test_bootstrap_interval_contains_point_estimate():
    rng = np.random.default_rng(1)
    y = rng.binomial(1, 0.3, 300)
    p = np.clip(0.3 + 0.3 * (y - 0.3) + rng.normal(0, 0.1, 300), 0.01, 0.99)
    point = policy_summary(y, p, 5)["incremental_per_applicant"]
    lo, hi = bootstrap_incremental_ci(y, p, 5, n_boot=300)
    assert lo <= point <= hi


def _clean():
    return {"critical_fail": False, "score_psi": 0.03, "brier_skill": 0.25,
            "buckets_fail": False, "auc": 0.80}


def test_stop_rule_critical_fail_stops():
    m = _clean()
    m["critical_fail"] = True
    action, reasons = stop_rules(m)
    assert action == "stop"
    assert any(r.startswith("R1") for r in reasons)


def test_stop_rule_psi_band_is_review():
    m = _clean()
    m["score_psi"] = 0.15
    action, reasons = stop_rules(m)
    assert action == "review"
    assert any(r.startswith("R3") for r in reasons)


def test_stop_rule_clean_continues():
    assert stop_rules(_clean()) == ("continue", [])


def test_stop_rule_low_auc_stops():
    m = _clean()
    m["auc"] = 0.60
    action, reasons = stop_rules(m)
    assert action == "stop"
    assert any(r.startswith("R5") for r in reasons)
