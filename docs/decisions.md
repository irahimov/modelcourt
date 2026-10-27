# Decisions

istiqamətə görə AUC qərarı və “no checking” kateqoriyasının boşluq sayılmaması

## Log

- (empty, first entry will be added with M1)

## 2026-10 M1: data quality

**Done:** implemented five checks in `quality.py` (missing rate, duplicates, unexpected categories, single-feature leakage with directional AUC, shared `classify` function). Ran on German Credit via `scripts/run_m1.py`.

**Why directional AUC:** a feature can predict default strongly in the opposite direction (AUC 0.3 is as informative as 0.7). Using max(AUC, 1-AUC) avoids a false pass. Decided before the M1 run was scored; threshold in `docs/thresholds.md` is written on this measure.

**Why "no checking" is not missing:** 394 rows have `checking_status = no checking`. This is a meaningful category (no current account), not a gap. It is counted as a valid category and is the strongest single feature (directional AUC 0.672), not a leakage signal at this level.

**Alternatives:** leakage on pairs of features (not done; limitation of the single-feature heuristic); expected categories taken from the dataset description, not from the data itself, so the check cannot confirm its own reference.

**Known limitation:** the single-feature heuristic misses leakage that only appears in a combination of features.

**Clarification (documentation only, no threshold changed):** `docs/thresholds.md` now states the
leakage threshold on directional AUC, max(AUC, 1-AUC). The measure was already used in the EDA
notebook (cell 15) before the M1 run was scored. The text was made precise afterwards; the
numbers 0.85 and 0.95 are unchanged.

## 2026-10 M2: validation sample and model

**Decided before any M2 result was computed.**

- The validated object is a logistic regression baseline trained by this project on German Credit.
  The validator does not rebuild a better model; it tests the one it is given.
- Stratified 70/30 split, random_state=42. Model trained on 70%.
- M2 decision thresholds apply to the 30% holdout. Development-sample AUC is reported only for
  comparison, because in-sample AUC is optimistic.
- Limitation: a single random split. Results depend on the split; the M4 stability and the
  seed sensitivity check (planned) address this later.

## 2026-10 M3: multiplicity in bucket tests

**Found by a positive-control test, not by a result on German Credit.**
The first version of the test expected a calibrated model to pass the bucket check every time.
With 5 buckets and no multiplicity correction, one bucket reaches p<0.05 by chance in about 23%
of calibrated samples. The threshold already allowed one conditional bucket; the test expectation
was wrong, not the threshold.

**Change:** the test now asserts (a) a calibrated model never FAILS on the bucket check, and
(b) across 30 calibrated samples the FAIL rate is at most 0.15 (theory: about 0.02).
No threshold was changed.

**Limitation kept:** no Bonferroni or Holm correction. A "conditional" bucket is a review item,
not a model failure. Adding a correction would make the bucket test stricter; it is listed for
discussion, not applied.

## 2026-10 M3: result on holdout (read after thresholds were fixed)

- Brier skill 0.25 (pass, threshold 0.05). Brier 0.157, base rate 0.300.
- Overall binomial p = 0.95: mean predicted PD 0.3026 vs observed 0.300 (pass).
- Buckets: 0 with p<0.05 (pass). Bucket 1 predicts 0.040, observes 0.083, p=0.089: the model
  under-predicts risk in its lowest bucket. Not significant at 0.05, recorded for monitoring.
- Reading: no calibration failure detected in a 300-row holdout. This is not evidence of
  calibration in the population; the sample is too small to exclude moderate errors.

## 2026-10 M4: stability

- Bin edges from the training sample only, computed once. Holdout does not change them.
- Every PSI computed in pandas and in DuckDB SQL. Max difference 6.9e-18 (consistency pass).
- Score PSI 0.029 and CSI max 0.064 (age): both pass. These compare two random parts of the
  same population, so they are a null result. They do not show stability over time.
- Positive control: credit_amount x1.5 on holdout gives PSI 0.387, detected. This shows the
  method can see a shift of that size. It does not show it would see a smaller one.
- Open: seed sensitivity of the 70/30 split is not yet measured. Planned as a sensitivity
  check, not as a pass/fail module.
