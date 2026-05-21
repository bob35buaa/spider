# E031 P0 9case Conservative Summary

Cases (9): box023_p1, box023_p2, box025_p1, box025_p2, bucket001_p2, bucket005_s2_p1, bucket005_s2_p2, bucket007_p1, bucket007_p2

| Method | Role | N | Missing | Obj Pos cm ↓ | Contact 5cm/proxy ↑ | Deep Pen % ↓ | Falls ↓ | Strict ↑ |
|---|---|---:|---|---:|---:|---:|---:|---:|
| omniretarget_kinematic | baseline | 9/9 | - | 0.00 | - | - | 0 | 0/9 |
| spider_E081_full_rerun | baseline | 9/9 | - | 21.27 | 42.17 | 16.25 | 1 | 4/9 |
| spider_E018b | best-positive | 9/9 | - | 5.14 | 63.61 | 35.83 | 1 | 1/9 |
| spider_best_conservative_E018b_E022_E025 | best-positive | 9/9 | - | 4.97 | 65.81 | 30.52 | 0 | 1/9 |
| spider_E028 | diagnostic | 5/9 | box023_p1, box023_p2, box025_p1, bucket007_p2 | 4.95 | 40.18 | 36.35 | 1 | 0/5 |
| spider_E029 | diagnostic | 2/9 | box023_p1, box023_p2, box025_p1, bucket005_s2_p1, bucket005_s2_p2, bucket007_p1, bucket007_p2 | 4.43 | 98.09 | 34.71 | 0 | 1/2 |
| spider_E030 | diagnostic | 6/9 | bucket001_p2, bucket005_s2_p2, bucket007_p1 | 5.17 | 32.89 | 17.32 | 2 | 0/6 |

## Data Caveats

| Case | Quality | P0 | Success-denominator discard | Rationale |
|---|---|---:|---:|---|
| box023_p1 | usable_with_caveat | True | False | usable_with_caveat: failed_evidence_classes=1; discard_from_p0=False; discard_from_success_denominator=False |
| box023_p2 | usable_with_caveat | True | False | usable_with_caveat: failed_evidence_classes=1; discard_from_p0=False; discard_from_success_denominator=False |
| box025_p1 | retarget_questionable | True | False | retarget_questionable: failed_evidence_classes=1; discard_from_p0=False; discard_from_success_denominator=False |
| box025_p2 | usable_algorithmic_failure | True | False | usable_algorithmic_failure: failed_evidence_classes=0; discard_from_p0=False; discard_from_success_denominator=False |
| bucket001_p2 | usable_with_caveat | True | False | usable_with_caveat: failed_evidence_classes=1; discard_from_p0=False; discard_from_success_denominator=False |
| bucket005_s2_p1 | usable_algorithmic_failure | True | False | usable_algorithmic_failure: failed_evidence_classes=0; discard_from_p0=False; discard_from_success_denominator=False |
| bucket005_s2_p2 | usable_algorithmic_failure | True | False | usable_algorithmic_failure: failed_evidence_classes=0; discard_from_p0=False; discard_from_success_denominator=False |
| bucket007_p1 | usable_algorithmic_failure | True | False | usable_algorithmic_failure: failed_evidence_classes=0; discard_from_p0=False; discard_from_success_denominator=False |
| bucket007_p2 | retarget_questionable | True | False | retarget_questionable: failed_evidence_classes=1; discard_from_p0=False; discard_from_success_denominator=False |

Notes:
- E028-E030 rows are shown as diagnostic baselines only; they are not eligible for `best-positive` selection.
- `desk021_p1` remains in P1 caveats but is the only success-denominator discard from E027.
- If the conservative best strict count remains unchanged, E031 is a ledger result, not a new optimization gain.
