"""Run M3 on the holdout sample. Base rate for Brier skill comes from the training sample."""
from pathlib import Path

import pandas as pd

from modelcourt.calibration import bucket_table, calibration_findings

ROOT = Path(__file__).resolve().parents[1]
SCORES = ROOT / "reports" / "scores.csv"
FINDINGS_OUT = ROOT / "reports" / "m3_findings.csv"
BUCKETS_OUT = ROOT / "reports" / "m3_buckets.csv"
DATASET = "german_credit_openml"


def main() -> None:
    s = pd.read_csv(SCORES)
    train = s[s["split"] == "train"]
    hold = s[s["split"] == "holdout"]
    base_rate = float(train["y"].mean())
    print(f"holdout rows: {len(hold)}, training base rate: {base_rate:.3f}")

    rows = calibration_findings(hold["y"], hold["score"], base_rate, DATASET)
    out = pd.DataFrame(rows)
    out.to_csv(FINDINGS_OUT, index=False)

    buckets = bucket_table(hold["y"], hold["score"])
    buckets.to_csv(BUCKETS_OUT, index=False)

    print(out[["test_id", "value", "decision", "severity", "note"]].to_string(index=False))
    print("\nBuckets:")
    print(buckets.to_string(index=False))


if __name__ == "__main__":
    main()
