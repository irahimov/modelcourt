# ModelCourt

Independent validation of credit and customer models: data quality, discrimination,
calibration, stability (PSI/CSI), backtesting, challenger comparison, a small IFRS 9
ECL example, and an automated validation report.

**Status:** skeleton only. Modules are being implemented one by one (M1 to M8).

## Problem

A validator does not rebuild a model. The validator checks it independently and
records findings with a severity level. This project does that for two models:

1. The churn model from ChurnLens (P1), trained on the IBM Telco Customer Churn dataset.
2. A credit default model, trained on Give Me Some Credit (Kaggle), with German Credit
   (UCI / OpenML) used as a small second case for calibration tests.

## Modules

| ID | Module | Content |
|----|--------|---------|
| M1 | `quality.py` | Missing rate, duplicates, anomalies, category values, target leakage checks, development vs new population drift |
| M2 | `discrimination.py` | AUC, Gini = 2·AUC − 1, KS, CAP curve |
| M3 | `calibration.py` | Brier score, calibration curve, bucket-level expected vs observed default rate, binomial test |
| M4 | `stability.py` | PSI (pandas and pure SQL, results must match), CSI at feature level |
| M5 | `backtesting.py` | Out-of-time test when dates exist, otherwise a labelled pseudo out-of-time split |
| M6 | `challenger.py` | WoE logistic scorecard (optbinning) vs gradient boosting |
| M7 | `ecl.py` | Optional. Small synthetic portfolio, Stage 1/2/3, ECL = PD × LGD × EAD |
| M8 | `report.py` | Test-by-test result, threshold, decision (pass / conditional / fail), findings |

## Quick start

Requires Python 3.12 (see `docs/environment.md`).

```bash
uv venv --python 3.12 ~/.fin-env
source ~/.fin-env/bin/activate
uv pip install -r requirements.txt
uv pip install -e ".[dev]"
pytest
```

One-command demo (to be added after M1 to M3):

```bash
bash scripts/run_all.sh
```

## Limitations

- Public datasets only. No real bank data is used.
- Bank Marketing and Telco are not bank customer data. Churn and propensity results
  do not transfer directly to a bank portfolio.
- The "out-of-time" split is pseudo out-of-time unless the dataset has real dates.
  This is stated in the reports where it applies.
- PSI thresholds 0.1 and 0.25 are industry practice, not regulatory requirements.

## What this project is not

- It is not a production monitoring system.
- It is not a regulatory-compliant IFRS 9 implementation. M7 is an illustration only.
- It does not rebuild the source models to improve them.

## Layout

```
src/modelcourt/   library code (one module per validation area)
tests/            pytest unit tests
sql/              SQL versions of checks (M1, M4)
data/raw/         downloaded datasets (not committed)
data/processed/   intermediate data (not committed)
reports/          result files (CSV, HTML/DOCX)
docs/             methodology and model documentation
notebooks/        exploration only
scripts/          one-command entry points
```

## Roadmap

| Phase | Content | Status |
|-------|---------|--------|
| 0 | Environment, skeleton, CI-free local tests | done |
| 1 | Data description, thresholds, decision log | in progress |
| 2 | M1 data quality, M2 discrimination | planned |
| 3 | M3 calibration, M4 PSI (pandas and SQL) | planned |
| 4 | M5 pseudo out-of-time, M6 challenger | planned |
| 5 | M8 report, `run_all.sh`, final README numbers | planned |

M7 (IFRS 9 ECL illustration) is out of scope for the first version. See `docs/future_work.md`.
