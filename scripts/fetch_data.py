"""Download German Credit (credit-g) from OpenML and save the raw CSV.

Output is not committed (data/raw/ is in .gitignore).
"""
from pathlib import Path

from sklearn.datasets import fetch_openml

OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "german_credit_openml.csv"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        print(f"Already exists, not overwriting: {OUT}")
        return

    df = fetch_openml(name="credit-g", version=1, as_frame=True, parser="auto").frame
    df.to_csv(OUT, index=False)
    print(f"Saved {df.shape} to {OUT}")
    print(df["class"].value_counts())


if __name__ == "__main__":
    main()
