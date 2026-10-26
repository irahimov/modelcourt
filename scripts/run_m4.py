"""Run M4 (stability): training sample as development, holdout as the new sample.

Writes:
  reports/m4_psi_table.csv   per-feature PSI in pandas and SQL, with the difference
  reports/m4_findings.csv    the four M4 findings
"""
from pathlib import Path

import duckdb
import pandas as pd

from modelcourt.stability import (
    DEV_NAME,
    NEW_NAME,
    numeric_edges,
    psi_categorical,
    psi_categorical_sql,
    psi_numeric,
    psi_numeric_sql,
    stability_findings,
)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "german_credit_openml.csv"
SCORES = ROOT / "reports" / "scores.csv"
TABLE_OUT = ROOT / "reports" / "m4_psi_table.csv"
FINDINGS_OUT = ROOT / "reports" / "m4_findings.csv"
DATASET = "german_credit_openml"


def main() -> None:
    raw = pd.read_csv(RAW)
    X = raw.drop(columns=["class"])
    s = pd.read_csv(SCORES)
    tr = s[s["split"] == "train"].reset_index(drop=True)
    ho = s[s["split"] == "holdout"].reset_index(drop=True)

    dev = X.loc[tr["row_id"].to_numpy()].reset_index(drop=True)
    new = X.loc[ho["row_id"].to_numpy()].reset_index(drop=True)
    dev["score"] = tr["score"].to_numpy()
    new["score"] = ho["score"].to_numpy()

    num_cols = dev.select_dtypes(include="number").columns.tolist()
    cat_cols = dev.select_dtypes(exclude="number").columns.tolist()

    con = duckdb.connect()
    con.register(DEV_NAME, dev)
    con.register(NEW_NAME, new)

    rows = []
    for c in num_cols:
        edges = numeric_edges(dev[c])
        p_pd = psi_numeric(dev[c], new[c], edges)
        p_sql = psi_numeric_sql(con, c, edges)
        rows.append({"feature": c, "kind": "numeric", "psi_pandas": p_pd,
                     "psi_sql": p_sql, "abs_diff": abs(p_pd - p_sql)})
    for c in cat_cols:
        p_pd = psi_categorical(dev[c], new[c])
        p_sql = psi_categorical_sql(con, c)
        rows.append({"feature": c, "kind": "categorical", "psi_pandas": p_pd,
                     "psi_sql": p_sql, "abs_diff": abs(p_pd - p_sql)})

    table = pd.DataFrame(rows).sort_values("psi_pandas", ascending=False)
    table.to_csv(TABLE_OUT, index=False)

    score_psi = float(table.loc[table["feature"] == "score", "psi_pandas"].iloc[0])
    feat = table[table["feature"] != "score"]
    csi = dict(zip(feat["feature"], feat["psi_pandas"]))
    consistency = float(table["abs_diff"].max())

    # Positive control: a known 1.5x shift on credit_amount in the holdout sample.
    shifted = new.copy()
    shifted["credit_amount"] = shifted["credit_amount"] * 1.5
    control = psi_numeric(dev["credit_amount"], shifted["credit_amount"],
                          numeric_edges(dev["credit_amount"]))

    out = pd.DataFrame(stability_findings(score_psi, csi, consistency, control, DATASET))
    out.to_csv(FINDINGS_OUT, index=False)

    print(table.head(8).to_string(index=False))
    print(f"\nmax |pandas - SQL| over all features: {consistency:.3e}")
    print(out[["test_id", "value", "decision", "severity", "note"]].to_string(index=False))


if __name__ == "__main__":
    main()
