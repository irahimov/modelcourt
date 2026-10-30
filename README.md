# ModelCourt

Independent validation of credit and customer models: data quality, discrimination,
calibration, stability (PSI/CSI), backtesting, challenger comparison, a small IFRS 9
ECL example, and an automated validation report.

**Status:** first version complete (M1 to M4, M6, M8). M5 is not applicable (no dates). Open items: seed sensitivity, M7, bank-data version.

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

One-command run (all modules):

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
| 1 | Data description, thresholds, decision log | done |
| 2 | M1 data quality, M2 discrimination | done |
| 3 | M3 calibration, M4 PSI (pandas and SQL) | done |
| 4 | M5 pseudo out-of-time, M6 challenger | done |
| 5 | M8 report, `run_all.sh`, final README numbers | done |

M7 (IFRS 9 ECL illustration) is out of scope for the first version. See `docs/future_work.md`.

## Results (first version)

All numbers are from `reports/validation_summary.csv`. Thresholds were written before
results (`docs/thresholds.md`). Overall decision on the main split: **pass**. The seed-sensitivity check moves the bucket-level calibration test (M3) to **conditional**; see the seed section below.

Model: logistic regression baseline, trained on a stratified 70% split of German Credit
(OpenML `credit-g`, 1000 rows, 30% default). All tests below use the 300-row holdout unless stated.

| Module | Test | Value | Decision |
|--------|------|-------|----------|
| M1 | Missing rate, duplicates, unexpected categories | 0%, 0, 0 | pass (86 checks, all pass) |
| M1 | Single-feature directional AUC (max) | 0.672 (`checking_status = no checking`) | pass (threshold 0.85) |
| M2 | AUC | 0.801, 95% bootstrap CI [0.743, 0.855] | pass |
| M2 | Gini | 0.602 | pass |
| M2 | KS | 0.494 | pass |
| M3 | Brier skill | 0.251 | pass (single split); mean 0.21 across 30 seeds |
| M3 | Buckets with binomial p < 0.05 | 0 on the main split (lowest p = 0.089); 2 or more in 2 of 30 seeds | pass on main split, **conditional across seeds** |
| M3 | Overall binomial p | 0.95 | pass |
| M4 | Score PSI (train vs holdout) | 0.029 | pass |
| M4 | Max CSI (`age`) | 0.064 | pass |
| M4 | pandas vs SQL max difference | 6.9e-18 | pass |
| M4 | Positive control (credit_amount x1.5), PSI | 0.387 (detected) | pass |
| M5 | Backtest | not applicable: no date column | n/a |
| M6 | AUC difference, LightGBM minus baseline | -0.021, 95% CI [-0.055, 0.014] | pass |

### How to read these results

- **M4 PSI and CSI are a null result.** Train and holdout come from the same random split of
  one population. Low PSI shows the split is balanced, not that the model is stable over time.
- **M4 control shows the method can see a shift of that size.** It does not show it would see
  a smaller shift.
- **M3 pass means no error was detected in 300 rows.** It does not show the model is calibrated
  in the population. Bucket 1 under-predicts risk (0.040 predicted, 0.083 observed, p = 0.089),
  recorded for monitoring.
- **M6 interval is wide.** The challenger is neither clearly better nor clearly worse on this
  sample. The upper bound (0.014) is below the conditional band (0.02), so the test cannot fail
  in either direction.
- **M5 is not performed.** Time-based backtesting needs a dataset with dates.

### Limitations

- Public data (German Credit, 1990s German consumer loans). Not a bank portfolio.
- One random 70/30 split, one seed (42). Sensitivity to the split is not yet measured.
- Thresholds are working values chosen by the author, not regulatory requirements.
- Single-feature leakage heuristic. Leakage that appears only in a combination of features
  is not detected.
- No time stamps, so no out-of-time validation.

## Decision policy (M9, illustrative)

Policy: approve an applicant when the predicted PD is below the break-even cutoff 1/(1+L).
L is the loss multiple of a bad approval relative to a good one. It is an assumption of this
project, not a bank parameter. Results on the 300-row holdout, compared with approving everyone:

| L | Cutoff | Approval rate | Bad rate among approved | Incremental per applicant | 95% CI |
|---|--------|---------------|-------------------------|---------------------------|--------|
| 3 | 0.250 | 52.7% | 10.8% | +0.50 | [0.34, 0.67] |
| 5 | 0.167 | 40.7% | 9.0% | +0.99 | [0.71, 1.26] |
| 8 | 0.111 | 28.0% | 10.7% | +1.71 | [1.29, 2.15] |

Reading:
- The interval excludes zero for every L on this holdout, so the policy beats approve-all here.
- Approve-all is not a bank policy. The comparison is a reference point, not a claim of
  improvement over current practice, which this project does not have.
- Cutoffs are not tuned on the holdout. The sweep in `reports/policy_curve.csv` is for sensitivity only.
- The interval reflects holdout sampling only. It does not cover the choice of L, the split, or the seed.

Stop rules (`reports/stop_rules.csv`) are evaluated against the validation summary. Current action: continue.
See `docs/case_playbook.md` for how these results would be used in a case discussion.

## Scope

What this project does: validates a probability-of-default score on a public credit dataset with
data-quality checks (M1), discrimination (M2), calibration (M3), stability (M4), a challenger comparison
(M6), a decision policy with stop rules (M9), a seed-sensitivity check, and a summary report (M8).

What it does not do: time-based backtesting (no dates), IFRS 9 / ECL (no LGD or EAD), fairness testing,
reject inference, production monitoring, or explainability. Each is listed with the data or policy
it would need in `docs/future_work.md`.

The thresholds, the loss multiples and the stop rules are working assumptions written before results.
They are not regulatory values. Public data is used throughout; nothing here validates a bank portfolio.

## Final status (first version)

**Done:** data-quality checks (M1), discrimination (M2), calibration (M3), stability (M4),
challenger comparison (M6), decision policy with stop rules (M9), seed sensitivity (30 seeds),
summary report (M8). 54 tests. Reproducible from a clean clone with `scripts/run_all.sh`.

**Not done, and why:**
- M5 (time-based backtest): the data has no dates.
- M7 (IFRS 9 / ECL): no LGD or EAD in the data.
- Fairness by sex and age: not tested. It needs a protected-attribute policy and group-level
  thresholds that this project cannot set. Out-of-fold scores over all 1000 rows would be the
  basis for a future test.
- Reject inference: no data on rejected applicants.
- Monitoring, explainability, model inventory and governance: out of scope for a prototype.

**Not claimed:** that the model works on a bank portfolio, is stable over time, or is fair.
The thresholds, loss multiples and stop rules are working assumptions written before results.
