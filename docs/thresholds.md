# Thresholds

Written before any M1 result is computed. Changes after results are shown are
recorded in `docs/decisions.md` with the reason; they are not made silently.

## M1: data quality

| Test | Pass | Conditional | Fail | Severity if fail |
|------|------|-------------|------|------------------|
| Missing rate per column | < 1% | 1% to 5% | > 5% | medium |
| Duplicate rows | 0 | 1 to 10 | > 10 | low |
| Single-feature directional AUC, max(AUC, 1-AUC), vs target (leakage heuristic) | < 0.85 | 0.85 to 0.95 | > 0.95 | critical |
| Category values outside expected set | 0 | 1 to 2 | > 2 | medium |
| PSI, development vs new sample (per feature) | < 0.10 | 0.10 to 0.25 | > 0.25 | high |

Notes:
- The 0.85 and 0.95 leakage cut-offs are a working heuristic chosen by the author,
  not an industry standard. A feature above 0.95 is flagged for manual review first;
  it is not removed automatically.
- PSI 0.10 and 0.25 are industry practice, not regulatory requirements.
- Decision rule (from the project plan): any critical fail means "rejected";
  any high fail or several conditional results means "conditional"; otherwise "pass".

## M2 to M4: to be written before those modules are computed.
