"""M6: train a LightGBM challenger on the training rows and compare on the holdout.

Writes reports/m6_findings.csv and reports/m6_challenger_scores.csv.
The baseline score is read from reports/scores.csv, so both models are judged on the same rows.
"""
from pathlib import Path

import lightgbm as lgb
import pandas as pd

from modelcourt.challenger import challenger_findings

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "german_credit_openml.csv"
SCORES = ROOT / "reports" / "scores.csv"
FINDINGS_OUT = ROOT / "reports" / "m6_findings.csv"
CHAL_OUT = ROOT / "reports" / "m6_challenger_scores.csv"
DATASET = "german_credit_openml"
SEED = 42


def main() -> None:
    raw = pd.read_csv(RAW)
    y = (raw["class"] == "bad").astype(int)
    X = raw.drop(columns=["class"])
    X = pd.get_dummies(X, drop_first=False).astype(float)

    s = pd.read_csv(SCORES)
    tr_idx = s.loc[s["split"] == "train", "row_id"].to_numpy()
    ho = s[s["split"] == "holdout"].reset_index(drop=True)

    model = lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.05, num_leaves=8,
        min_child_samples=20, random_state=SEED, verbose=-1,
    )
    model.fit(X.loc[tr_idx], y.loc[tr_idx])
    chal = model.predict_proba(X.loc[ho["row_id"].to_numpy()])[:, 1]

    pd.DataFrame({"row_id": ho["row_id"], "y": ho["y"], "score_baseline": ho["score"],
                  "score_challenger": chal}).to_csv(CHAL_OUT, index=False)

    rows = challenger_findings(ho["y"], ho["score"], chal, DATASET)
    pd.DataFrame(rows).to_csv(FINDINGS_OUT, index=False)
    for r in rows:
        print(r["test_id"], r["value"], r["decision"], r["severity"], r["note"])


if __name__ == "__main__":
    main()
