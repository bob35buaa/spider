# E095 Worklike Candidate Mining

## Summary

- Candidate rows: `32`
- Box004 priority Stage2b rows: `3`
- Score is geometry/raw-contact only; source-scene readiness and object-history route are not numeric score terms.
- Rows are sorted by execution tier first, then score; `rank` is therefore queue rank, not pure score rank.

Tier counts:

| tier | count |
|---|---:|
| `tier2_box021_review_after_target_gate` | 15 |
| `tier4_box026_deprioritized` | 7 |
| `tier3_box022_needs_raw_contact` | 6 |
| `tier1_box004_priority` | 3 |
| `tier0_known_work` | 1 |

## First Batch

| rank | target | score | raw score | source scene | note |
|---:|---|---:|---:|---|---|
| 2 | `e091_box004_20231003_2_083_p1` | 87.998 | 100.0 | `box004_person1` | closest to E092/E094 WORK pattern |
| 3 | `e091_box004_20231003_2_082_p1` | 79.582 | 91.932 | `box004_person1` | closest to E092/E094 WORK pattern |
| 4 | `e091_box004_20231003_2_082_p2` | 70.283 | 80.682 | `box004_person2` | closest to E092/E094 WORK pattern |

## Interpretation

- `tier1_box004_priority` is the only batch to run immediately.
- `tier2_box021_review_after_target_gate` is held for a later target/posture-gated route because D003/Box021 has repeated CEM failures.
- `tier4_box026_deprioritized` remains in the bank for traceability but is not a first-batch data source after E092/E094 failures.
