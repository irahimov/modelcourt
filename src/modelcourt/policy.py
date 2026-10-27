"""M9: decision policy on top of the validated score.

Turns a default-probability score into an approve / reject rule with an explicit loss
assumption, then evaluates the rule on the holdout using realized outcomes.

Economics per applicant: approve a good customer +1, approve a bad customer -L, reject 0.
Break-even: approve when (1 - p) - L * p > 0, i.e. p < 1 / (1 + L).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def breakeven_pd(loss_multiple: float) -> float:
    return 1.0 / (1.0 + loss_multiple)


def realized_profit(y, approve, loss_multiple: float) -> float:
    y = np.asarray(y)
    a = np.asarray(approve, dtype=bool)
    per = np.where(y == 1, -loss_multiple, 1.0)
    return float(per[a].sum())


def policy_summary(y, p, loss_multiple: float, cutoff: float | None = None) -> dict:
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    n = len(y)
    c = breakeven_pd(loss_multiple) if cutoff is None else cutoff
    approve = p < c
    n_app = int(approve.sum())
    profit = realized_profit(y, approve, loss_multiple)
    profit_all = realized_profit(y, np.ones(n, dtype=bool), loss_multiple)
    return {
        "loss_multiple": loss_multiple,
        "cutoff": round(float(c), 4),
        "approval_rate": round(n_app / n, 4),
        "bad_rate_among_approved": round(float(y[approve].mean()), 4) if n_app else float("nan"),
        "profit_per_applicant": round(profit / n, 4),
        "approve_all_profit_per_applicant": round(profit_all / n, 4),
        "incremental_per_applicant": round((profit - profit_all) / n, 4),
    }


def profit_curve(y, p, loss_multiple: float, cutoffs=None) -> pd.DataFrame:
    grid = np.round(np.arange(0.05, 0.96, 0.05), 2) if cutoffs is None else cutoffs
    return pd.DataFrame([policy_summary(y, p, loss_multiple, cutoff=float(c)) for c in grid])


def bootstrap_incremental_ci(y, p, loss_multiple: float, cutoff: float | None = None,
                             n_boot: int = 2000, seed: int = 42, alpha: float = 0.05):
    """Paired bootstrap: the same resampled rows score the policy and approve-all."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    c = breakeven_pd(loss_multiple) if cutoff is None else cutoff
    n = len(y)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        pb = p[idx]
        pol = realized_profit(yb, pb < c, loss_multiple)
        allp = realized_profit(yb, np.ones(n, dtype=bool), loss_multiple)
        stats.append((pol - allp) / n)
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def stop_rules(metrics: dict) -> tuple[str, list[str]]:
    """metrics: critical_fail (bool), score_psi (float), brier_skill (float),
    buckets_fail (bool), auc (float). Returns (action, reasons)."""
    stops, reviews = [], []
    if metrics["critical_fail"]:
        stops.append("R1: a critical validation test failed")
    if metrics["score_psi"] > 0.25:
        stops.append("R2: score PSI > 0.25 (population shift)")
    elif metrics["score_psi"] >= 0.10:
        reviews.append("R3: score PSI in 0.10-0.25 (review shift)")
    if metrics["buckets_fail"] or metrics["brier_skill"] < 0:
        stops.append("R4: calibration failure (bucket fail or negative Brier skill)")
    if metrics["auc"] < 0.65:
        stops.append("R5: holdout AUC below 0.65")
    if stops:
        return "stop", stops + reviews
    if reviews:
        return "review", reviews
    return "continue", []
