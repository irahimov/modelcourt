"""M8: combine findings from M1 to M6 and apply the project decision rule.

Decision rule (docs/thresholds.md):
  any critical fail                         -> rejected
  any high fail, or two or more conditional -> conditional
  otherwise                                 -> pass
"""
from __future__ import annotations

import pandas as pd

ALLOWED = {"pass", "conditional", "fail"}


def decide(findings: pd.DataFrame) -> str:
    rows = findings[findings["decision"].isin(ALLOWED)]
    crit_fail = ((rows["decision"] == "fail") & (rows["severity"] == "critical")).any()
    high_fail = ((rows["decision"] == "fail") & (rows["severity"] == "high")).any()
    n_cond = int((rows["decision"] == "conditional").sum())
    if crit_fail:
        return "rejected"
    if high_fail or n_cond >= 2:
        return "conditional"
    return "pass"


def combine(paths) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in paths]
    return pd.concat(frames, ignore_index=True)


def not_applicable_row(test_id: str, dataset: str, reason: str) -> dict:
    return {
        "test_id": test_id, "module": test_id.split(".")[0], "dataset": dataset,
        "metric": "not_applicable", "value": None, "threshold": "-",
        "decision": "n/a", "severity": "none", "note": reason,
    }


def to_html(findings: pd.DataFrame, overall: str, title: str) -> str:
    counts = findings["decision"].value_counts().to_dict()
    table = findings.to_html(index=False, border=0)
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>body{{font-family:sans-serif;max-width:1100px;margin:2em auto;padding:0 1em}}
table{{border-collapse:collapse;font-size:13px}}th,td{{border-bottom:1px solid #ddd;padding:6px;text-align:left}}
.big{{font-size:20px;font-weight:600}}</style></head><body>
<h1>{title}</h1>
<p class="big">Overall decision: {overall}</p>
<p>Counts: {counts}. Tests marked n/a are not performed and are not counted.</p>
<p>Thresholds were written before results (docs/thresholds.md). Limitations are in README.md
and docs/decisions.md. The data is public (German Credit, 1000 rows, no dates), so this report
does not validate any bank portfolio.</p>
{table}
</body></html>"""
