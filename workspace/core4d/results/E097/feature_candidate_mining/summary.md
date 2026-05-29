# E097 Feature-based Candidate Mining

## Summary

- Candidate/audit rows: `38`
- Already verified excluded rows: `14`
- Unverified candidate rows: `8`
- Enabled next-batch rows: `6`
- Score uses raw contact and geometry/reach proxy only; source scene readiness and object key are metadata, not score terms.
- Pipeline `# enabled=1` means run preprocess + inside/support target gate first, not direct RL.

Route counts:

| route | count |
|---|---:|
| `excluded_verified` | 14 |
| `candidate_target_posture_gate` | 8 |
| `raw_contact_preflight_disabled` | 6 |
| `hold_raw_contact_failed` | 6 |
| `large_reach_holdout` | 4 |

## Enabled Next Batch

| target | route | score | raw | both | L/R | longest | reach | note |
|---|---|---:|---:|---:|---|---:|---|---|
| `e091_box021_20231018_028_p1` | `candidate_target_posture_gate` | 118.99 | 100.0 | 1.0 | 1.0/1.0 | 1.0 | `medium_box_target_posture_gate` | raw contact strong; run preprocess plus inside/support target gate before CEM |
| `e091_box021_20231018_028_p2` | `candidate_target_posture_gate` | 118.99 | 100.0 | 1.0 | 1.0/1.0 | 1.0 | `medium_box_target_posture_gate` | raw contact strong; run preprocess plus inside/support target gate before CEM |
| `e091_box021_20231020_020_p2` | `candidate_target_posture_gate` | 118.99 | 100.0 | 1.0 | 1.0/1.0 | 1.0 | `medium_box_target_posture_gate` | raw contact strong; run preprocess plus inside/support target gate before CEM |
| `e091_box021_20231011_035_p1` | `candidate_target_posture_gate` | 113.846 | 95.933 | 0.9867 | 0.9867/1.0 | 0.7067 | `medium_box_target_posture_gate` | raw contact strong; run preprocess plus inside/support target gate before CEM |
| `e091_box021_20231018_030_p2` | `candidate_target_posture_gate` | 113.636 | 95.417 | 0.9722 | 1.0/0.9722 | 0.7778 | `medium_box_target_posture_gate` | raw contact strong; run preprocess plus inside/support target gate before CEM |
| `e091_box021_20231020_019_p2` | `candidate_target_posture_gate` | 112.426 | 94.079 | 0.9474 | 1.0/0.9474 | 0.8158 | `medium_box_target_posture_gate` | raw contact strong; run preprocess plus inside/support target gate before CEM |

## Top Unverified Candidates

| rank | target | route | score | raw | reach | scene | note |
|---:|---|---|---:|---:|---|---|---|
| 1 | `e091_box021_20231018_028_p1` | `candidate_target_posture_gate` | 118.99 | 100.0 | `medium_box_target_posture_gate` | `box021_person1` | raw contact strong; run preprocess plus inside/support target gate before CEM |
| 2 | `e091_box021_20231018_028_p2` | `candidate_target_posture_gate` | 118.99 | 100.0 | `medium_box_target_posture_gate` | `box021_person2` | raw contact strong; run preprocess plus inside/support target gate before CEM |
| 3 | `e091_box021_20231020_020_p2` | `candidate_target_posture_gate` | 118.99 | 100.0 | `medium_box_target_posture_gate` | `box021_person2` | raw contact strong; run preprocess plus inside/support target gate before CEM |
| 4 | `e091_box021_20231011_035_p1` | `candidate_target_posture_gate` | 113.846 | 95.933 | `medium_box_target_posture_gate` | `box021_person1` | raw contact strong; run preprocess plus inside/support target gate before CEM |
| 5 | `e091_box021_20231018_030_p2` | `candidate_target_posture_gate` | 113.636 | 95.417 | `medium_box_target_posture_gate` | `box021_person2` | raw contact strong; run preprocess plus inside/support target gate before CEM |
| 6 | `e091_box021_20231020_019_p2` | `candidate_target_posture_gate` | 112.426 | 94.079 | `medium_box_target_posture_gate` | `box021_person2` | raw contact strong; run preprocess plus inside/support target gate before CEM |
| 7 | `e091_box021_20231018_030_p1` | `candidate_target_posture_gate` | 108.294 | 90.417 | `medium_box_target_posture_gate` | `box021_person1` | raw contact strong; run preprocess plus inside/support target gate before CEM |
| 8 | `e091_box021_20231018_029_p1` | `candidate_target_posture_gate` | 108.177 | 90.625 | `medium_box_target_posture_gate` | `box021_person1` | raw contact strong; run preprocess plus inside/support target gate before CEM |

## Holdouts / Preflight

| target | route | raw | reach | note |
|---|---|---:|---|---|
| `e091_box022_20231023_125_p1` | `raw_contact_preflight_disabled` | 0.0 | `long_edge_raw_contact_preflight` | raw contact not run; run D002/raw-contact visual first |
| `e091_box022_20231023_125_p2` | `raw_contact_preflight_disabled` | 0.0 | `long_edge_raw_contact_preflight` | raw contact not run; run D002/raw-contact visual first |
| `e091_box022_20231023_126_p1` | `raw_contact_preflight_disabled` | 0.0 | `long_edge_raw_contact_preflight` | raw contact not run; run D002/raw-contact visual first |
| `e091_box022_20231023_126_p2` | `raw_contact_preflight_disabled` | 0.0 | `long_edge_raw_contact_preflight` | raw contact not run; run D002/raw-contact visual first |
| `e091_box022_20231023_127_p1` | `raw_contact_preflight_disabled` | 0.0 | `long_edge_raw_contact_preflight` | raw contact not run; run D002/raw-contact visual first |
| `e091_box022_20231023_127_p2` | `raw_contact_preflight_disabled` | 0.0 | `long_edge_raw_contact_preflight` | raw contact not run; run D002/raw-contact visual first |
| `e091_box026_20231020_134_p1` | `large_reach_holdout` | 100.0 | `large_reach_holdout` | large reach risk after Box026 failures; needs separate repair route |
| `e091_box026_20231020_134_p2` | `large_reach_holdout` | 100.0 | `large_reach_holdout` | large reach risk after Box026 failures; needs separate repair route |
| `e091_box026_20231020_135_p1` | `large_reach_holdout` | 85.0 | `large_reach_holdout` | large reach risk after Box026 failures; needs separate repair route |
| `e091_box026_20231018_039_p1` | `large_reach_holdout` | 66.867 | `large_reach_holdout` | large reach risk after Box026 failures; needs separate repair route |

## Excluded Verified

| target | status | stage | note |
|---|---|---|---|
| `e091_box004_20231003_2_082_p1` | `verified_positive` | `full_cem_work` | E096/E096b full CEM WORK. |
| `e091_box004_20231003_2_082_p2` | `verified_reject` | `omniretarget_infeasible` | E095/E096 no-fingertip and fingertip retry both CVXPY infeasible. |
| `e091_box004_20231003_2_083_p1` | `verified_positive` | `full_cem_work` | E096/E096b full CEM WORK. |
| `e091_box004_20231003_2_083_p2` | `verified_positive` | `full_cem_work` | E092/E094 known WORK control; E091 D005b PASS. |
| `e091_box021_20231011_034_p1` | `verified_review` | `preprocess_or_smoke_review` | Older data_construction/D003 route already used this case. |
| `e091_box021_20231011_034_p2` | `verified_reject` | `omniretarget_infeasible` | Older data_construction route reported CVXPY infeasible. |
| `e091_box021_20231011_035_p2` | `verified_reject` | `full_cem_fail` | D003/E082-E083 Box021 representative full CEM fail. |
| `e091_box021_20231018_029_p2` | `verified_reject` | `full_cem_fail` | D003/E082-E088 Box021 representative full CEM fail. |
| `e091_box021_20231018_031_p2` | `verified_review` | `topface_smoke_review` | E089/E090 topface/smoke route already used this case. |
| `e091_box021_20231020_019_p1` | `verified_reject` | `full_cem_fail` | D003/E082-E083 Box021 representative full CEM fail. |
| `e091_box021_20231020_020_p1` | `verified_review` | `topface_smoke_review` | E089/E090 topface/smoke route already used this case. |
| `e091_box026_20231018_039_p2` | `verified_reject` | `d005b_reject_and_full_cem_fail` | E091 support reject; E092/E094 full CEM posture fail. |
| `e091_box026_20231018_040_p2` | `verified_reject` | `omniretarget_infeasible` | E091 OmniRetarget CVXPY infeasible. |
| `e091_box026_20231020_135_p2` | `verified_reject` | `d005b_reject_and_full_cem_fail` | E091 right-inside reject; E092/E094 full CEM posture fail. |

## Decision

- New likely-work candidates are not yet RL-ready. They should run Stage2b/OmniRetarget, then D005b/E096-style inside/support target gate, then full CEM posture gate.
- The current best new batch is box021, but only as `candidate_target_posture_gate`; this reflects both its strong raw contact and its known posture/target risk.
- Box026 remains a large-reach holdout despite raw contact; Box022 needs raw-contact preflight before preprocess.
