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

**Clarification (documentation only, no threshold changed):** `docs/thresholds.md` now states the
leakage threshold on directional AUC, max(AUC, 1-AUC). The measure was already used in the EDA
notebook (cell 15) before the M1 run was scored. The text was made precise afterwards; the
numbers 0.85 and 0.95 are unchanged.
