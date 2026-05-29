# E095 Worklike Candidate Mining

## Summary

- Candidate rows: `32`
- Box004 priority Stage2b rows: `3`
- Score is geometry/raw-contact only; source-scene readiness and object key are not numeric score terms.
- Rows are sorted by execution tier first, then score; `rank` is therefore queue rank, not pure score rank.

Tier counts:

| tier | count |
|---|---:|
| `tier2_target_posture_gate_review` | 15 |
| `tier4_large_reach_dynamics_holdout` | 7 |
| `tier3_missing_raw_contact_long_edge_review` | 6 |
| `tier1_worklike_priority` | 3 |
| `tier0_known_work` | 1 |

## First Batch

| rank | target | score | raw score | source scene | note |
|---:|---|---:|---:|---|---|
| 2 | `e091_box004_20231003_2_083_p1` | 87.998 | 100.0 | `box004_person1` | box004/box023-scale geometry selected for first-batch execution |
| 3 | `e091_box004_20231003_2_082_p1` | 79.582 | 91.932 | `box004_person1` | box004/box023-scale geometry selected for first-batch execution |
| 4 | `e091_box004_20231003_2_082_p2` | 70.283 | 80.682 | `box004_person2` | box004/box023-scale geometry selected for first-batch execution |

## Interpretation

- `tier1_worklike_priority` is the only new batch to run immediately.
- `tier2_target_posture_gate_review` is held for target/posture-gated review before CEM.
- `tier4_large_reach_dynamics_holdout` remains in the bank for traceability but is not a first-batch data source after large-box E092/E094 failures.
