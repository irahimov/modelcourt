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
