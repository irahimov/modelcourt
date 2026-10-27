"""M6: challenger comparison.

Compares the baseline score with a challenger score on the same holdout rows.
The statistic is the paired difference of AUC, with a paired bootstrap interval:
the same resampled rows are used for both models, so the difference is not inflated
by sample noise that the two models share.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def paired_auc_difference(y, s_base, s_chal, n_boot: int = 1000, seed: int = 42):
    """Return (difference, ci_low, ci_high). Difference = AUC(challenger) - AUC(baseline)."""
    y = np.asarray(y)
    s_base = np.asarray(s_base, dtype=float)
    s_chal = np.asarray(s_chal, dtype=float)
    diff = roc_auc_score(y, s_chal) - roc_auc_score(y, s_base)

    rng = np.random.default_rng(seed)
    n = len(y)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        stats.append(roc_auc_score(yb, s_chal[idx]) - roc_auc_score(yb, s_base[idx]))
    lo, hi = np.quantile(stats, [0.025, 0.975])
    return float(diff), float(lo), float(hi)


def classify_difference(diff: float):
    """Negative or small difference: pass. Large positive difference: the baseline is weaker."""
    if diff < 0.02:
        return "pass", "none"
    if diff <= 0.05:
        return "conditional", "low"
    return "fail", "high"


def challenger_findings(y, s_base, s_chal, dataset: str) -> list[dict]:
    diff, lo, hi = paired_auc_difference(y, s_base, s_chal)
    dec, sev = classify_difference(diff)
    return [{
        "test_id": "M6.challenger_auc_diff",
        "module": "M6",
        "dataset": dataset,
        "metric": "auc_challenger_minus_baseline",
        "value": round(diff, 4),
        "threshold": "<0.02 pass, 0.02-0.05 conditional, >0.05 fail",
        "decision": dec,
        "severity": sev,
        "note": f"paired bootstrap 95% CI [{lo:.3f}, {hi:.3f}]",
    }]
