"""Seed sensitivity: repeat the 70/30 split, the fit and the M3 and M9 checks for many seeds.

Does not overwrite reports/scores.csv. Seed 42 is the control and must reproduce the
baseline holdout AUC (0.801).
Writes reports/seed_sensitivity.csv.
"""
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from modelcourt.calibration import bucket_table, brier_skill, overall_binomial
from modelcourt.policy import policy_summary

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "german_credit_openml.csv"
OUT = ROOT / "reports" / "seed_sensitivity.csv"
SEEDS = [42] + list(range(1, 30))
LOSS_GRID = [3.0, 5.0, 8.0]


def build_model(num_cols, cat_cols, seed):
    prep = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])
    return Pipeline([("prep", prep),
                     ("clf", LogisticRegression(max_iter=1000, random_state=seed))])


def run_seed(df, seed):
    y = (df["class"] == "bad").astype(int)
    X = df.drop(columns=["class"])
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()

    tr, ho = train_test_split(X.index, test_size=0.30, stratify=y, random_state=seed)
    model = build_model(num_cols, cat_cols, seed).fit(X.loc[tr], y.loc[tr])
    p = model.predict_proba(X.loc[ho])[:, 1]
    yh = y.loc[ho].to_numpy()
    base_rate = float(y.loc[tr].mean())

    row = {
        "seed": seed,
        "auc_holdout": roc_auc_score(yh, p),
        "brier_skill": brier_skill(yh, p, base_rate),
        "buckets_p_below_005": int((bucket_table(yh, p)["binomial_p"] < 0.05).sum()),
        "overall_p": overall_binomial(yh, p)["p_value"],
    }
    for L in LOSS_GRID:
        row[f"incr_L{int(L)}"] = policy_summary(yh, p, L)["incremental_per_applicant"]
    return row


def main() -> None:
    df = pd.read_csv(DATA)
    out = pd.DataFrame([run_seed(df, s) for s in SEEDS])
    out.to_csv(OUT, index=False)

    control = float(out.loc[out["seed"] == 42, "auc_holdout"].iloc[0])
    print(f"control seed 42 AUC: {control:.3f} (baseline: 0.801)")
    print(out[["auc_holdout", "brier_skill", "incr_L3", "incr_L5", "incr_L8"]]
          .describe().loc[["mean", "min", "max"]].round(3).to_string())
    print(f"\nAUC min across seeds: {out['auc_holdout'].min():.3f}")
    print(f"share of seeds with incr_L5 > 0: {(out['incr_L5'] > 0).mean():.0%}")
    print(f"share of seeds with 2+ buckets p<0.05: {(out['buckets_p_below_005'] >= 2).mean():.0%}")


if __name__ == "__main__":
    main()
