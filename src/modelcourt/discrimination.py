"""M2: discrimination (ranking power) of a default score.

Measures: AUC, Gini = 2*AUC - 1, KS, CAP curve, bootstrap confidence interval for AUC.
All functions expect y in {0, 1} (1 = default) and a score where higher = riskier.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score


def auc(y, score) -> float:
    return float(roc_auc_score(y, score))


def gini(auc_value: float) -> float:
    return 2.0 * auc_value - 1.0


def ks(y, score) -> float:
    """KS = max |CDF_bad(s) - CDF_good(s)|, computed as the two-sample KS statistic."""
    y = np.asarray(y)
    score = np.asarray(score)
    return float(ks_2samp(score[y == 1], score[y == 0]).statistic)


def cap_curve(y, score) -> pd.DataFrame:
    """Cumulative accuracy profile.

    Sort by score from riskiest to safest. For each share of the population,
    report the share of all defaults captured so far. A random model is the diagonal.
    """
    d = pd.DataFrame({"y": np.asarray(y), "score": np.asarray(score)})
    d = d.sort_values("score", ascending=False).reset_index(drop=True)
    n = len(d)
    total_bad = d["y"].sum()
    return pd.DataFrame({
        "population_share": np.arange(1, n + 1) / n,
        "bad_captured_share": d["y"].cumsum().to_numpy() / total_bad,
    })


def bootstrap_auc_ci(y, score, n_boot: int = 2000, seed: int = 42, alpha: float = 0.05):
    """Percentile bootstrap interval for AUC.

    Resamples rows with replacement. Resamples that contain only one class are skipped,
    because AUC is undefined for them.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    score = np.asarray(score)
    n = len(y)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        stats.append(roc_auc_score(yb, score[idx]))
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi), len(stats)


def classify_higher_is_better(value, pass_at, fail_below):
    """Higher values are better. pass: value >= pass_at; fail: value < fail_below; else conditional."""
    if value >= pass_at:
        return "pass", "none"
    if value < fail_below:
        return "fail", "high"
    return "conditional", "low"


def discrimination_findings(y, score, dataset: str) -> list[dict]:
    """Findings for AUC, Gini and KS, using the thresholds in docs/thresholds.md (M2)."""
    a = auc(y, score)
    g = gini(a)
    k = ks(y, score)
    lo, hi, n_used = bootstrap_auc_ci(y, score)

    note_inv = " inverted: AUC < 0.5, check score direction" if a < 0.5 else ""

    rows = []

    dec, sev = classify_higher_is_better(a, 0.70, 0.65)
    rows.append(_row("M2.auc", dataset, "auc_holdout", round(a, 3),
                     ">=0.70 pass, 0.65-0.70 conditional, <0.65 fail", dec, sev,
                     note=f"bootstrap 95% CI [{lo:.3f}, {hi:.3f}], {n_used} valid resamples." + note_inv))

    dec, sev = classify_higher_is_better(g, 0.40, 0.30)
    rows.append(_row("M2.gini", dataset, "gini_holdout", round(g, 3),
                     ">=0.40 pass, 0.30-0.40 conditional, <0.30 fail", dec, sev,
                     note="Gini = 2*AUC-1; not an independent test of AUC."))

    dec, sev = classify_higher_is_better(k, 0.25, 0.20)
    rows.append(_row("M2.ks", dataset, "ks_holdout", round(k, 3),
                     ">=0.25 pass, 0.20-0.25 conditional, <0.20 fail", dec, sev))

    return rows


def _row(test_id, dataset, metric, value, threshold, decision, severity, note=""):
    """Same columns as quality.findings_to_frame. Module is set to M2 here."""
    return {
        "test_id": test_id, "module": "M2", "dataset": dataset, "metric": metric,
        "value": value, "threshold": threshold, "decision": decision,
        "severity": severity, "note": note,
    }
