# E097 Feature-based Candidate Mining

## Summary

- Candidate/audit rows: `38`
- Already verified excluded rows: `22`
- Unverified candidate rows: `0`
- Enabled next-batch rows: `0`
- Score uses raw contact and geometry/reach proxy only; source scene readiness and object key are metadata, not score terms.
- Pipeline `# enabled=1` means run preprocess + inside/support target gate first, not direct RL.

Route counts:

| route | count |
|---|---:|
| `excluded_verified` | 22 |
| `raw_contact_preflight_disabled` | 6 |
| `hold_raw_contact_failed` | 6 |
| `large_reach_holdout` | 4 |

## Enabled Next Batch

| target | route | score | raw | both | L/R | longest | reach | note |
|---|---|---:|---:|---:|---|---:|---|---|

## Top Unverified Candidates

| rank | target | route | score | raw | reach | scene | note |
|---:|---|---|---:|---:|---|---|---|

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
| `e091_box021_20231011_035_p1` | `verified_review` | `d004_visual_pass` | Legacy D004 visual QC already passed/reviewed this sequence; do not rediscover as new data. |
| `e091_box021_20231011_035_p2` | `verified_reject` | `full_cem_fail` | D003/E082-E083 Box021 representative full CEM fail. |
| `e091_box021_20231018_028_p1` | `verified_reject` | `d003_omniretarget_infeasible` | Legacy D003 OmniRetarget log reports CVXPY infeasible; E097 visual review has raw-contact only. |
| `e091_box021_20231018_028_p2` | `verified_reject` | `d004_visual_reject_fall_prone` | Legacy D004 visual QC rejected this retargeted sequence as fall/prone. |
| `e091_box021_20231018_029_p1` | `verified_review` | `d004_visual_review` | Legacy D004 visual QC already marked this sequence for review. |
| `e091_box021_20231018_029_p2` | `verified_reject` | `full_cem_fail` | D003/E082-E088 Box021 representative full CEM fail. |
| `e091_box021_20231018_030_p1` | `verified_review` | `d004_visual_review` | Legacy D004 visual QC already marked this sequence for review. |
| `e091_box021_20231018_030_p2` | `verified_review` | `d004_visual_pass_check_shortcut` | Legacy D004 visual QC already passed/reviewed this sequence with shortcut concern. |
| `e091_box021_20231018_031_p2` | `verified_review` | `topface_smoke_review` | E089/E090 topface/smoke route already used this case. |
| `e091_box021_20231020_019_p1` | `verified_reject` | `full_cem_fail` | D003/E082-E083 Box021 representative full CEM fail. |
| `e091_box021_20231020_019_p2` | `verified_review` | `d004_visual_review` | Legacy D004 visual QC already marked this sequence for review. |
| `e091_box021_20231020_020_p1` | `verified_review` | `topface_smoke_review` | E089/E090 topface/smoke route already used this case. |
| `e091_box021_20231020_020_p2` | `verified_review` | `d004_visual_pass` | Legacy D004 visual QC already passed/reviewed this sequence; do not rediscover as new data. |
| `e091_box026_20231018_039_p2` | `verified_reject` | `d005b_reject_and_full_cem_fail` | E091 support reject; E092/E094 full CEM posture fail. |
| `e091_box026_20231018_040_p2` | `verified_reject` | `omniretarget_infeasible` | E091 OmniRetarget CVXPY infeasible. |
| `e091_box026_20231020_135_p2` | `verified_reject` | `d005b_reject_and_full_cem_fail` | E091 right-inside reject; E092/E094 full CEM posture fail. |

## Decision

- After adding legacy D003/D004 visual-QC outcomes, this mined pool has no clean unverified next-batch candidates.
- The prior six Box021 rows are useful for visual review/diagnosis, but should not be rediscovered as new data.
- The remaining actionable expansion is to run raw-contact preflight for Box022 or broaden the inventory beyond the current medium-box old D001/D002 pool; Box026 remains a large-reach holdout.
