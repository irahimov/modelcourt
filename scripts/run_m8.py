"""M8: combine module findings, apply the decision rule and write the HTML report."""
from pathlib import Path

import pandas as pd

from modelcourt.report import combine, decide, not_applicable_row, to_html

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DATASET = "german_credit_openml"

FINDING_FILES = [
    REPORTS / "m1_findings.csv",
    REPORTS / "m2_findings.csv",
    REPORTS / "m3_findings.csv",
    REPORTS / "m4_findings.csv",
    REPORTS / "m6_findings.csv",
]


def main() -> None:
    df = combine(FINDING_FILES)
    df = pd.concat([df, pd.DataFrame([not_applicable_row(
        "M5.backtest", DATASET,
        "no date column in German Credit; time-based backtest not possible")])], ignore_index=True)

    overall = decide(df)
    df.to_csv(REPORTS / "validation_summary.csv", index=False)
    (REPORTS / "validation_report.html").write_text(
        to_html(df, overall, "ModelCourt validation report: German Credit baseline"), encoding="utf-8")

    print(df[["test_id", "value", "decision", "severity"]].to_string(index=False))
    print(f"\nOverall decision: {overall}")
    print(f"Wrote {REPORTS / 'validation_report.html'}")


if __name__ == "__main__":
    main()
