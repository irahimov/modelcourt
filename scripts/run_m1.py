"""Run M1 data quality checks on German Credit and write reports/m1_findings.csv."""
from pathlib import Path

import pandas as pd

from modelcourt.quality import (
    duplicate_rows,
    findings_to_frame,
    missing_rates,
    single_feature_leakage,
    unexpected_categories,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "german_credit_openml.csv"
OUT = ROOT / "reports" / "m1_findings.csv"
DATASET = "german_credit_openml"

# Allowed values, taken from the dataset description (OpenML credit-g).
EXPECTED = {
    "checking_status": {"<0", "0<=X<200", ">=200", "no checking"},
    "credit_history": {
        "no credits/all paid", "all paid", "existing paid",
        "delayed previously", "critical/other existing credit",
    },
    "savings_status": {"<100", "100<=X<500", "500<=X<1000", ">=1000", "no known savings"},
}


def main() -> None:
    df = pd.read_csv(DATA)
    y = (df["class"] == "bad").astype(int)
    features = df.drop(columns=["class"])
    num_cols = features.select_dtypes(include="number").columns.tolist()
    cat_cols = features.select_dtypes(exclude="number").columns.tolist()

    rows = []
    rows += missing_rates(df, DATASET)
    rows += duplicate_rows(df, DATASET)
    rows += unexpected_categories(features, EXPECTED, DATASET)
    rows += single_feature_leakage(features, y, DATASET, num_cols, cat_cols)

    out = findings_to_frame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    print(out["decision"].value_counts())
    print("\nNon-pass findings:")
    print(out.loc[out["decision"] != "pass", ["test_id", "metric", "value", "decision", "severity"]])


if __name__ == "__main__":
    main()
