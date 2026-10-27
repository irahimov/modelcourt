# Future work

Items deliberately left out of the first version, with the reason for each.

## M7: IFRS 9 ECL illustration
Small synthetic portfolio, Stage 1/2/3 rules (30 days past due as backstop),
ECL = PD × LGD × EAD, 12-month vs lifetime. Left out because the time budget for
the first version is used by M1 to M4 and the report. A conceptual answer is in
`docs/decisions.md`.

## P1 churn model as a second validation object
The ChurnLens model could be validated with the same modules. Left out of the
first version to keep the credit-risk path focused. The findings format already
supports a second `dataset` value, so no structural change is needed.

## Real time-stamped data for backtesting
Current datasets have no dates, so M5 uses a pseudo out-of-time split. A dataset
with real dates would allow a rolling-window backtest.

## Bank-specific extensions
Next Best Product and CLV need multi-product customer data, which is not public.

## Deliberately not in the first version

Each item is listed with what it would need. None is claimed as implemented.

### Fairness and disparate impact
Needs: group-level approval rate and default rate by sex and age band, with a proper protected-attribute
policy. German Credit's personal_status combines sex and marital status, so a clean sex variable must be
derived first. Holdout has only 90 defaults, so group estimates would be unstable; out-of-fold scores over
all 1000 rows would be the honest basis. Not done because thresholds for disparate impact need a policy
decision that this project cannot make.

### Reject inference (selection bias)
Needs: data on applicants who were rejected and their outcomes, or a documented reject-inference method
(reweighting, parcelling). German Credit contains only accepted-style labelled loans. M9 therefore evaluates
on fully labelled rows, which hides the selection problem rather than solving it.

### Out-of-time validation (M5)
Needs: a dataset with application or observation dates. Not possible with the public datasets used here.

### ECL and IFRS 9 (M7)
Needs: LGD and EAD per loan, staging rules and macroeconomic scenarios. Only PD is available here.

### Ongoing monitoring
Needs: a scheduled run on new data, stored history of M1 to M4 results, and alert rules. The project is a
one-time snapshot. run_all.sh is reproducible but not scheduled.

### Explainability
Needs: per-applicant reason codes (for example SHAP for the challenger, or coefficient-based reasons for the
logistic baseline). Not included because it adds a dependency and a second interpretation layer that needs
its own validation.

### Origin of thresholds
All thresholds in docs/thresholds.md are working values chosen by the author. A bank derives them from its
risk appetite, regulatory requirements and historical experience. This project makes no claim about which
values are right, only that they were fixed before results were computed.

### Multi-product and CLV (P1 scope)
Needs: multi-product customer data, which is not public. Churn and CLV for P1 are outside this project.
