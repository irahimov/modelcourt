"""Train the validated baseline: logistic regression on German Credit.

Writes:
  models/baseline_logreg.joblib   the fitted pipeline (not committed, see .gitignore)
  reports/scores.csv              score for every row, with a 'split' column (train / holdout)

M2 to M4 read reports/scores.csv. Only the holdout rows are used for the M2 decision.
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "german_credit_openml.csv"
MODEL_OUT = ROOT / "models" / "baseline_logreg.joblib"
SCORES_OUT = ROOT / "reports" / "scores.csv"

SEED = 42
HOLDOUT_SIZE = 0.30


def main() -> None:
    df = pd.read_csv(DATA)
    y = (df["class"] == "bad").astype(int)
    X = df.drop(columns=["class"])

    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()

    # Stratified split keeps the 30% default rate in both parts.
    idx_train, idx_hold = train_test_split(
        X.index, test_size=HOLDOUT_SIZE, stratify=y, random_state=SEED
    )

    # Preprocessing and model live in one pipeline, so the scaler and encoder
    # are fitted on the training part only. Holdout rows never influence them.
    preprocess = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])
    model = Pipeline([
        ("prep", preprocess),
        ("clf", LogisticRegression(max_iter=1000, random_state=SEED)),
    ])
    model.fit(X.loc[idx_train], y.loc[idx_train])

    # Scores: probability of default (class 1).
    score = pd.Series(model.predict_proba(X)[:, 1], index=X.index)
    split = pd.Series("train", index=X.index)
    split.loc[idx_hold] = "holdout"

    scores = pd.DataFrame({
        "row_id": X.index,
        "split": split.values,
        "y": y.values,
        "score": score.values,
    })

    # Development-sample AUC is reported only for comparison (in-sample is optimistic).
    auc_train = roc_auc_score(y.loc[idx_train], score.loc[idx_train])
    auc_hold = roc_auc_score(y.loc[idx_hold], score.loc[idx_hold])
    print(f"train rows={len(idx_train)}  holdout rows={len(idx_hold)}")
    print(f"AUC train   (comparison only): {auc_train:.3f}")
    print(f"AUC holdout (M2 decision basis): {auc_hold:.3f}")

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    SCORES_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_OUT)
    scores.to_csv(SCORES_OUT, index=False)
    print(f"saved model:  {MODEL_OUT}")
    print(f"saved scores: {SCORES_OUT}")


if __name__ == "__main__":
    main()
