"""M3: calibration of a probability-of-default score.

AUC (M2) measures ranking. Calibration measures level: when the model says 30%,
do about 30% of those customers default? A model can rank perfectly and still be
systematically too high or too low.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import brier_score_loss


def brier(y, p) -> float:
    return float(brier_score_loss(np.asarray(y), np.asarray(p)))


def brier_skill(y, p, base_rate: float) -> float:
    """1 - Brier / Brier_null. Null is a constant prediction equal to base_rate (training rate)."""
    y = np.asarray(y, dtype=float)
    null = float(np.mean((y - base_rate) ** 2))
    return 1.0 - brier(y, p) / null


def bucket_table(y, p, n_bins: int = 5) -> pd.DataFrame:
    """Equal-count buckets by predicted PD, with a two-sided binomial test per bucket.

    Ranks are used before qcut so that tied scores cannot merge buckets or create empty ones.
    """
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p, dtype=float)})
    d["bucket"] = pd.qcut(d["p"].rank(method="first"), n_bins, labels=False) + 1
    rows = []
    for b, g in d.groupby("bucket"):
        n = len(g)
        k = int(g["y"].sum())
        expected = float(g["p"].mean())
        rows.append({
            "bucket": int(b),
            "n": n,
            "defaults": k,
            "mean_predicted_pd": round(expected, 4),
            "observed_rate": round(k / n, 4),
            "binomial_p": float(binomtest(k, n, expected).pvalue),
        })
    return pd.DataFrame(rows)


def overall_binomial(y, p) -> dict:
    """Observed defaults vs the sum of predicted PDs, one binomial test on the whole sample."""
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    n = len(y)
    k = int(y.sum())
    expected = float(p.mean())
    return {
        "n": n,
        "defaults": k,
        "mean_predicted_pd": round(expected, 4),
        "observed_rate": round(k / n, 4),
        "p_value": float(binomtest(k, n, expected).pvalue),
    }


def classify_higher_is_better(value, pass_at, fail_below):
    if value >= pass_at:
        return "pass", "none"
    if value < fail_below:
        return "fail", "high"
    return "conditional", "low"


def calibration_findings(y, p, base_rate: float, dataset: str) -> list[dict]:
    rows = []

    skill = brier_skill(y, p, base_rate)
    dec, sev = classify_higher_is_better(skill, 0.05, 0.00)
    rows.append(_row("M3.brier_skill", dataset, "brier_skill_holdout", round(skill, 4),
                     ">=0.05 pass, 0.00-0.05 conditional, <0.00 fail", dec, sev,
                     note=f"Brier={brier(y, p):.4f}, training base rate={base_rate:.3f}"))

    buckets = bucket_table(y, p)
    n_low = int((buckets["binomial_p"] < 0.05).sum())
    if n_low == 0:
        dec, sev = "pass", "none"
    elif n_low == 1:
        dec, sev = "conditional", "low"
    else:
        dec, sev = "fail", "high"
    rows.append(_row("M3.buckets", dataset, "buckets_p_below_005", n_low,
                     "0 pass, 1 conditional, >=2 fail", dec, sev,
                     note="5 equal-count buckets; no multiplicity correction"))

    ov = overall_binomial(y, p)
    dec = "pass" if ov["p_value"] >= 0.05 else "fail"
    sev = "none" if dec == "pass" else "high"
    rows.append(_row("M3.overall", dataset, "overall_binomial_p", round(ov["p_value"], 4),
                     "p>=0.05 pass, p<0.05 fail", dec, sev,
                     note=f"observed {ov['observed_rate']} vs mean PD {ov['mean_predicted_pd']}"))
    return rows


def _row(test_id, dataset, metric, value, threshold, decision, severity, note=""):
    return {
        "test_id": test_id, "module": "M3", "dataset": dataset, "metric": metric,
        "value": value, "threshold": threshold, "decision": decision,
        "severity": severity, "note": note,
    }
