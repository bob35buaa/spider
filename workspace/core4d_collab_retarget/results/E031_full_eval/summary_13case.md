# E031 P1 13case Conservative Summary

Cases (13): box021_p1, box021_p2, box023_p1, box023_p2, box025_p1, box025_p2, bucket001_p1, bucket001_p2, bucket005_s2_p1, bucket005_s2_p2, bucket007_p1, bucket007_p2, desk021_p1

| Method | Role | N | Missing | Obj Pos cm ↓ | Contact 5cm/proxy ↑ | Deep Pen % ↓ | Falls ↓ | Strict ↑ |
|---|---|---:|---|---:|---:|---:|---:|---:|
| omniretarget_kinematic | baseline | 12/13 | desk021_p1 | 0.00 | - | - | 0 | 0/12 |
| spider_E081_full_rerun | baseline | 13/13 | - | 27.10 | 36.23 | 12.39 | 4 | 4/13 |
| spider_E018b | best-positive | 13/13 | - | 5.45 | 54.35 | 30.04 | 4 | 1/13 |
| spider_best_conservative_E018b_E022_E025 | best-positive | 13/13 | - | 5.33 | 55.87 | 26.37 | 3 | 1/13 |
| spider_E028 | diagnostic | 5/13 | box021_p1, box021_p2, box023_p1, box023_p2, box025_p1, bucket001_p1, bucket007_p2, desk021_p1 | 4.95 | 40.18 | 36.35 | 1 | 0/5 |
| spider_E029 | diagnostic | 3/13 | box021_p1, box021_p2, box023_p1, box023_p2, box025_p1, bucket005_s2_p1, bucket005_s2_p2, bucket007_p1, bucket007_p2, desk021_p1 | 4.06 | 65.39 | 23.14 | 0 | 1/3 |
| spider_E030 | diagnostic | 6/13 | box021_p1, box021_p2, bucket001_p1, bucket001_p2, bucket005_s2_p2, bucket007_p1, desk021_p1 | 5.17 | 32.89 | 17.32 | 2 | 0/6 |

## Data Caveats

| Case | Quality | P0 | Success-denominator discard | Rationale |
|---|---|---:|---:|---|
| box021_p1 | usable_algorithmic_failure | False | False | usable_algorithmic_failure: failed_evidence_classes=0; discard_from_p0=True; discard_from_success_denominator=False |
| box021_p2 | usable_algorithmic_failure | False | False | usable_algorithmic_failure: failed_evidence_classes=0; discard_from_p0=True; discard_from_success_denominator=False |
| box023_p1 | usable_with_caveat | True | False | usable_with_caveat: failed_evidence_classes=1; discard_from_p0=False; discard_from_success_denominator=False |
| box023_p2 | usable_with_caveat | True | False | usable_with_caveat: failed_evidence_classes=1; discard_from_p0=False; discard_from_success_denominator=False |
| box025_p1 | retarget_questionable | True | False | retarget_questionable: failed_evidence_classes=1; discard_from_p0=False; discard_from_success_denominator=False |
| box025_p2 | usable_algorithmic_failure | True | False | usable_algorithmic_failure: failed_evidence_classes=0; discard_from_p0=False; discard_from_success_denominator=False |
| bucket001_p1 | usable_with_caveat | False | False | usable_with_caveat: failed_evidence_classes=1; discard_from_p0=True; discard_from_success_denominator=False |
| bucket001_p2 | usable_with_caveat | True | False | usable_with_caveat: failed_evidence_classes=1; discard_from_p0=False; discard_from_success_denominator=False |
| bucket005_s2_p1 | usable_algorithmic_failure | True | False | usable_algorithmic_failure: failed_evidence_classes=0; discard_from_p0=False; discard_from_success_denominator=False |
| bucket005_s2_p2 | usable_algorithmic_failure | True | False | usable_algorithmic_failure: failed_evidence_classes=0; discard_from_p0=False; discard_from_success_denominator=False |
| bucket007_p1 | usable_algorithmic_failure | True | False | usable_algorithmic_failure: failed_evidence_classes=0; discard_from_p0=False; discard_from_success_denominator=False |
| bucket007_p2 | retarget_questionable | True | False | retarget_questionable: failed_evidence_classes=1; discard_from_p0=False; discard_from_success_denominator=False |
| desk021_p1 | discard_from_success_denominator | False | True | discard_from_success_denominator: failed_evidence_classes=3; discard_from_p0=True; discard_from_success_denominator=True |

Notes:
- E028-E030 rows are shown as diagnostic baselines only; they are not eligible for `best-positive` selection.
- `desk021_p1` remains in P1 caveats but is the only success-denominator discard from E027.
- If the conservative best strict count remains unchanged, E031 is a ledger result, not a new optimization gain.
