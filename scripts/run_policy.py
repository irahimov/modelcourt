"""M9: decision policy on the holdout, for three loss multiples, plus the stop-rule check.

Reads reports/scores.csv (holdout) and reports/validation_summary.csv (written by run_m8.py).
Writes:
  reports/policy_curve.csv    cutoff sweep for each L
  reports/policy_summary.csv  break-even result per L, with paired bootstrap CI
  reports/stop_rules.csv      action (continue / review / stop) and the reasons
"""
from pathlib import Path

import pandas as pd

from modelcourt.policy import bootstrap_incremental_ci, policy_summary, profit_curve, stop_rules

ROOT = Path(__file__).resolve().parents[1]
SCORES = ROOT / "reports" / "scores.csv"
SUMMARY = ROOT / "reports" / "validation_summary.csv"
CURVE_OUT = ROOT / "reports" / "policy_curve.csv"
SUMMARY_OUT = ROOT / "reports" / "policy_summary.csv"
STOP_OUT = ROOT / "reports" / "stop_rules.csv"
LOSS_GRID = [3.0, 5.0, 8.0]


def stop_metrics(summary: pd.DataFrame) -> dict:
    def val(test_id):
        return float(summary.loc[summary["test_id"] == test_id, "value"].iloc[0])
    return {
        "critical_fail": bool(((summary["decision"] == "fail") & (summary["severity"] == "critical")).any()),
        "score_psi": val("M4.score_psi"),
        "brier_skill": val("M3.brier_skill"),
        "buckets_fail": bool((summary.loc[summary["test_id"] == "M3.buckets", "decision"] == "fail").any()),
        "auc": val("M2.auc"),
    }


def main() -> None:
    s = pd.read_csv(SCORES)
    h = s[s["split"] == "holdout"]
    y = h["y"].to_numpy()
    p = h["score"].to_numpy()

    curves, rows = [], []
    for L in LOSS_GRID:
        curves.append(profit_curve(y, p, L))
        base = policy_summary(y, p, L)
        lo, hi = bootstrap_incremental_ci(y, p, L)
        rows.append({**base, "incremental_ci_low": round(lo, 4), "incremental_ci_high": round(hi, 4)})

    pd.concat(curves, ignore_index=True).to_csv(CURVE_OUT, index=False)
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(SUMMARY_OUT, index=False)

    action, reasons = stop_rules(stop_metrics(pd.read_csv(SUMMARY)))
    pd.DataFrame([{"action": action, "reasons": " | ".join(reasons) or "none"}]).to_csv(STOP_OUT, index=False)

    print(summary_df.to_string(index=False))
    print(f"\nStop-rule action: {action}")
    for r in reasons:
        print(" -", r)


if __name__ == "__main__":
    main()
