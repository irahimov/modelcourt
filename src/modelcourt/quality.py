"""M1: data quality checks.

Each check returns a list of findings (plain dicts). No check modifies the data.
Thresholds come from docs/thresholds.md and are written there before results.
"""
from __future__ import annotations

import pandas as pd
from sklearn.metrics import roc_auc_score

FINDING_COLUMNS = [
    "test_id", "module", "dataset", "metric", "value",
    "threshold", "decision", "severity", "note",
]


def classify(value, pass_below, fail_above, severity_if_fail):
    """Return (decision, severity) using the three bands from docs/thresholds.md.

    pass:        value < pass_below
    conditional: pass_below <= value <= fail_above
    fail:        value > fail_above
    """
    if value < pass_below:
        return "pass", "none"
    if value > fail_above:
        return "fail", severity_if_fail
    return "conditional", "low"


def finding(test_id, dataset, metric, value, threshold, decision, severity, note=""):
    return {
        "test_id": test_id,
        "module": "M1",
        "dataset": dataset,
        "metric": metric,
        "value": value,
        "threshold": threshold,
        "decision": decision,
        "severity": severity,
        "note": note,
    }


def findings_to_frame(rows):
    return pd.DataFrame(rows, columns=FINDING_COLUMNS)


def missing_rates(df, dataset):
    """Missing percentage per column. Threshold: <1% pass, 1-5% conditional, >5% fail."""
    rows = []
    for col in df.columns:
        pct = 100 * df[col].isna().mean()
        decision, severity = classify(pct, pass_below=1.0, fail_above=5.0, severity_if_fail="medium")
        rows.append(finding(
            "M1.missing", dataset, f"missing_pct:{col}", round(pct, 2),
            "<1 pass, 1-5 conditional, >5 fail", decision, severity,
        ))
    return rows


def duplicate_rows(df, dataset):
    """Count of exact duplicate rows (second and later occurrences). Threshold: 0 pass, 1-10 conditional, >10 fail."""
    n = int(df.duplicated().sum())
    decision, severity = classify(n, pass_below=1, fail_above=10, severity_if_fail="low")
    return [finding(
        "M1.duplicates", dataset, "duplicate_rows", n,
        "0 pass, 1-10 conditional, >10 fail", decision, severity,
    )]


def directional_auc(y, x):
    """AUC corrected for direction: max(AUC, 1 - AUC).

    A feature that is strongly predictive in the opposite direction still gets a high score.
    """
    auc = roc_auc_score(y, x)
    return max(auc, 1.0 - auc)


def _leak_row(name, x, y, dataset):
    mask = x.notna()
    auc_dir = directional_auc(y[mask], x[mask])
    gini = 2 * auc_dir - 1
    decision, severity = classify(auc_dir, pass_below=0.85, fail_above=0.95, severity_if_fail="critical")
    return finding(
        "M1.leakage", dataset, f"single_feature_auc_dir:{name}", round(auc_dir, 3),
        "<0.85 pass, 0.85-0.95 conditional, >0.95 fail", decision, severity,
        note=f"gini={gini:.3f}",
    )


def single_feature_leakage(df, target, dataset, num_cols, cat_cols):
    """Single-feature leakage heuristic.

    Numeric columns are scored directly. Categorical columns are scored one level at a time
    (level vs rest), since AUC needs a numeric score. Constant indicators are skipped.
    """
    y = pd.Series(target).astype(int)
    rows = [_leak_row(c, df[c], y, dataset) for c in num_cols]
    for c in cat_cols:
        for lvl in df[c].dropna().unique():
            indicator = (df[c] == lvl).astype(int)
            if indicator.nunique() < 2:
                continue
            rows.append(_leak_row(f"{c}={lvl}", indicator, y, dataset))
    return rows


def unexpected_categories(df, expected, dataset):
    """Count of observed values outside the expected set, per column.

    expected: dict {column: set_of_allowed_values}. Threshold: 0 pass, 1-2 conditional, >2 fail.
    Missing values are not counted here; the missing-rate check covers them.
    """
    rows = []
    for col, allowed in expected.items():
        observed = df[col].dropna().unique()
        bad = [v for v in observed if v not in allowed]
        decision, severity = classify(len(bad), pass_below=1, fail_above=2, severity_if_fail="medium")
        rows.append(finding(
            "M1.categories", dataset, f"unexpected_values:{col}", len(bad),
            "0 pass, 1-2 conditional, >2 fail", decision, severity,
            note=", ".join(map(str, bad))[:200],
        ))
    return rows
