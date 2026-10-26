"""M4: stability of the score and of features between a development and a new sample.

PSI = sum_i (a_i - e_i) * ln(a_i / e_i), where e_i is the development share and a_i the new share
in bin i. Numeric bins are set by development quantiles, computed once and applied to the new
sample. Zero shares are replaced by EPS so the logarithm is finite.

Every PSI is computed twice: in pandas and in pure SQL (DuckDB). The two must agree, and the
agreement is itself a finding (M4.consistency).
"""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

EPS = 1e-4
DEV_NAME = "dev_t"
NEW_NAME = "new_t"


def classify_psi(value: float):
    if value < 0.10:
        return "pass", "none"
    if value > 0.25:
        return "fail", "high"
    return "conditional", "low"


def numeric_edges(dev: pd.Series, n_bins: int = 10) -> np.ndarray:
    """Inner cut points from development quantiles. Uses the development sample only."""
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    return np.unique(np.quantile(dev.dropna().to_numpy(), qs))


def _psi_from_shares(a: np.ndarray, e: np.ndarray) -> float:
    a = np.where(a == 0, EPS, a)
    e = np.where(e == 0, EPS, e)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_numeric(dev: pd.Series, new: pd.Series, edges: np.ndarray) -> float:
    k = len(edges) + 1
    bd = np.searchsorted(edges, dev.dropna().to_numpy(), side="right")
    bn = np.searchsorted(edges, new.dropna().to_numpy(), side="right")
    e = np.bincount(bd, minlength=k) / len(bd)
    a = np.bincount(bn, minlength=k) / len(bn)
    return _psi_from_shares(a, e)


def psi_categorical(dev: pd.Series, new: pd.Series) -> float:
    cats = sorted(set(dev.dropna().unique()) | set(new.dropna().unique()), key=str)
    e = dev.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    a = new.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    return _psi_from_shares(a, e)


def _sql_bin_numeric(col: str, edges: np.ndarray) -> str:
    """Bin index = number of inner edges <= x. Same rule as np.searchsorted(side='right')."""
    if len(edges) == 0:
        return "0"
    return " + ".join(f'CAST("{col}" >= {float(x)!r} AS INTEGER)' for x in edges)


def _psi_sql(con: duckdb.DuckDBPyConnection, dev_sql: str, new_sql: str) -> float:
    q = f"""
    WITH
    e_s AS (
        SELECT b, COUNT(*) * 1.0 / SUM(COUNT(*)) OVER () AS share
        FROM ({dev_sql}) GROUP BY b
    ),
    a_s AS (
        SELECT b, COUNT(*) * 1.0 / SUM(COUNT(*)) OVER () AS share
        FROM ({new_sql}) GROUP BY b
    ),
    j AS (
        SELECT COALESCE(a_s.share, 0) AS a, COALESCE(e_s.share, 0) AS e
        FROM a_s FULL OUTER JOIN e_s ON a_s.b = e_s.b
    ),
    k AS (
        SELECT CASE WHEN a = 0 THEN {EPS} ELSE a END AS a2,
               CASE WHEN e = 0 THEN {EPS} ELSE e END AS e2
        FROM j
    )
    SELECT SUM((a2 - e2) * LN(a2 / e2)) FROM k
    """
    return float(con.execute(q).fetchone()[0])


def psi_numeric_sql(con, col: str, edges: np.ndarray) -> float:
    b = _sql_bin_numeric(col, edges)
    dev_sql = f'SELECT {b} AS b FROM {DEV_NAME} WHERE "{col}" IS NOT NULL'
    new_sql = f'SELECT {b} AS b FROM {NEW_NAME} WHERE "{col}" IS NOT NULL'
    return _psi_sql(con, dev_sql, new_sql)


def psi_categorical_sql(con, col: str) -> float:
    dev_sql = f'SELECT CAST("{col}" AS VARCHAR) AS b FROM {DEV_NAME} WHERE "{col}" IS NOT NULL'
    new_sql = f'SELECT CAST("{col}" AS VARCHAR) AS b FROM {NEW_NAME} WHERE "{col}" IS NOT NULL'
    return _psi_sql(con, dev_sql, new_sql)


def _row(test_id, dataset, metric, value, threshold, decision, severity, note=""):
    return {
        "test_id": test_id, "module": "M4", "dataset": dataset, "metric": metric,
        "value": value, "threshold": threshold, "decision": decision,
        "severity": severity, "note": note,
    }


def stability_findings(score_psi, csi, consistency_diff, control_psi, dataset):
    rows = []

    dec, sev = classify_psi(score_psi)
    rows.append(_row("M4.score_psi", dataset, "psi_score", round(score_psi, 4),
                     "<0.10 pass, 0.10-0.25 conditional, >0.25 fail", dec, sev))

    not_pass = [k for k, v in csi.items() if v >= 0.10]
    n = len(not_pass)
    if n == 0:
        dec, sev = "pass", "none"
    elif n <= 2:
        dec, sev = "conditional", "low"
    else:
        dec, sev = "fail", "medium"
    if csi:
        worst = max(csi, key=csi.get)
        note = f"worst feature: {worst} (CSI={csi[worst]:.4f})"
    else:
        note = "no features"
    rows.append(_row("M4.csi", dataset, "features_csi_not_pass", n,
                     "0 pass, 1-2 conditional, >2 fail", dec, sev, note=note))

    ok = consistency_diff < 1e-9
    rows.append(_row("M4.consistency", dataset, "max_abs_diff_pandas_vs_sql",
                     float(f"{consistency_diff:.3e}"), "<1e-9 pass, >=1e-9 fail",
                     "pass" if ok else "fail", "none" if ok else "critical"))

    detected = control_psi > 0.25
    rows.append(_row("M4.control", dataset, "psi_control_credit_amount_x1.5",
                     round(control_psi, 4),
                     "detected (PSI>0.25) = pass, not detected = fail",
                     "pass" if detected else "fail", "none" if detected else "critical",
                     note="positive control: the method must see a known shift"))
    return rows
