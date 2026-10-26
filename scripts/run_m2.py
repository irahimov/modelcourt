"""Run M2 on the holdout sample only and write reports/m2_findings.csv and CAP data."""
from pathlib import Path

import pandas as pd

from modelcourt.discrimination import cap_curve, discrimination_findings

ROOT = Path(__file__).resolve().parents[1]
SCORES = ROOT / "reports" / "scores.csv"
FINDINGS_OUT = ROOT / "reports" / "m2_findings.csv"
CAP_OUT = ROOT / "reports" / "m2_cap_curve.csv"
DATASET = "german_credit_openml"


def main() -> None:
    s = pd.read_csv(SCORES)
    hold = s[s["split"] == "holdout"]
    print(f"holdout rows: {len(hold)}, defaults: {int(hold['y'].sum())}")

    rows = discrimination_findings(hold["y"], hold["score"], DATASET)
    out = pd.DataFrame(rows)
    out.to_csv(FINDINGS_OUT, index=False)

    cap = cap_curve(hold["y"], hold["score"])
    cap.to_csv(CAP_OUT, index=False)

    print(out[["test_id", "value", "decision", "severity", "note"]].to_string(index=False))


if __name__ == "__main__":
    main()
