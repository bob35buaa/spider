# E148 E143 24-case rubber hand collision results

Date: 2026-06-09 CST

Plan: `workspace/core4d/plan/156_E148_e143_24case_rubber_hand_collision_plan.md`

## Scope

E148 extends the E147 `rubber_hull` hand-collision variant to the E143 `raw_mask_ref_fk` 24-case workset. The comparison uses three methods on the same cases:

- `OmniRetarget`
- `sphere Spider` = E143 `raw_mask_ref_fk` sphere-hand Spider/CEM baseline
- `rubber hand Spider` = E147/E148 `rubber_hull` Spider/CEM

The run does not change reward, CEM optimizer, target route, E143 sphere outputs, or OmniRetarget source inputs. The only CEM-side change is replacing the hand collision geometry with the rubber-hand mesh hull sidecar scene.

## Reuse And Execution

E148 reuses 8 E147 overlap rows and only launches the 16 missing rubber rows.

Reused from E147:

- `box023_person2`
- `bucket004_20231003_1_012_p1`
- `d003_box021_20231011_035_p1`
- `d003_box021_20231011_035_p2`
- `e091_box004_20231003_2_082_p1`
- `e091_box004_20231003_2_083_p1`
- `e091_box026_20231020_134_p1`
- `e091_box026_20231020_134_p2`

Remote full CEM ran on `spider-remote` in tmux session `E148_full_020443`.

Split:

- GPU0: `bucket004_20231002_022_p1`, `e091_box004_20231003_2_083_p2`, `e091_box026_20231018_039_p2`, `e091_box026_20231020_133_p2`, `e091_box026_20231020_135_p2`, `e091_box026_20231020_139_p2`, `e091_box026_20231020_141_p2`, `e091_box026_20231023_139_p1`
- GPU1: `d003_box021_20231018_029_p2`, `e091_box026_20231018_039_p1`, `e091_box026_20231020_133_p1`, `e091_box026_20231020_135_p1`, `e091_box026_20231020_138_p2`, `e091_box026_20231020_141_p1`, `e091_box026_20231023_137_p1`, `e091_box026_20231023_139_p2`

Commands:

- manifest: `.venv/bin/python workspace/core4d/scripts/E148/build_rubber_hand_collision_manifest.py`
- launch: `bash workspace/core4d/scripts/run_E148_remote.sh full`
- pull: `bash workspace/core4d/scripts/pull_E148_remote_results.sh full`
- eval: `bash workspace/core4d/scripts/eval/eval_E148_e143_rubber_hand_collision.sh full`
- xlsx recalc: `python /home/ubuntu/.codex/skills/xlsx/scripts/recalc.py workspace/core4d/results/E148/e143_24case_rubber_hand_collision/comparison/E148_e143_omni_sphere_rubber_hand_comparison.xlsx 60`

The tmux session ended naturally. No remote process was killed.

## Artifacts

| Artifact | Path | Count |
|---|---:|---:|
| manifest | `workspace/core4d/scripts/E148/variants.tsv` | 24 rows |
| reused rubber rows | E147 rubber paths referenced by E148 manifest | 8 |
| new root NPZ | `workspace/core4d/results/E148/e143_24case_rubber_hand_collision/cem/full/*.npz` | 16 |
| new MP4 | `workspace/core4d/results/E148/e143_24case_rubber_hand_collision/cem/full/*_full.mp4` | 16 |
| new outdir trajectory | `workspace/core4d/results/E148/e143_24case_rubber_hand_collision/cem/full/*_outdir_full/trajectory_mjwp_act.npz` | 16 |
| run logs | `logs/E148/cem/full/*.log` | 16 |
| keyframe dirs | `workspace/core4d/results/E148/e143_24case_rubber_hand_collision/cem/full/keyframes/` | 16 dirs |
| method metrics | `workspace/core4d/results/E148/e143_24case_rubber_hand_collision/eval/full/e148_method_metrics.tsv` | 72 rows |
| case comparison | `workspace/core4d/results/E148/e143_24case_rubber_hand_collision/eval/full/e148_case_comparison.tsv` | 24 rows |
| eval summary | `workspace/core4d/results/E148/e143_24case_rubber_hand_collision/eval/full/e148_summary.md` | 1 |
| comparison XLSX | `workspace/core4d/results/E148/e143_24case_rubber_hand_collision/comparison/E148_e143_omni_sphere_rubber_hand_comparison.xlsx` | 4 sheets |

XLSX sheets after LibreOffice recalculation:

- `24case平均`: 3 method rows, `case_count=24`
- `逐case对比`: 24 case rows
- `filtered平均`: 3 method rows, `case_count=23`
- `filtered逐case`: 23 case rows

Formula recalculation reported `total_errors=0`, `total_formulas=1476`.

## Quantitative Result

### 24-case average

| method | cases | fall | 5cm | 10cm | hand pen | deep2 | leg pen | object floor | pelvis min |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `OmniRetarget` | 24 | 0 | 0.6463 | 0.6643 | 0.5470 | 0.2940 | 0.1137 | 0.3518 | 0.6944 |
| `sphere Spider` | 24 | 0 | 0.5585 | 0.6122 | 0.3324 | 0.0138 | 0.0895 | 0.5047 | 0.6778 |
| `rubber hand Spider` | 24 | 0 | 0.5675 | 0.6131 | 0.2988 | 0.0115 | 0.0949 | 0.5008 | 0.6800 |

Diffs:

| metric | OmniRetarget-rubber | rubber-sphere |
|---|---:|---:|
| 5cm | +0.0788 | +0.0090 |
| 10cm | +0.0512 | +0.0010 |
| hand penetration | +0.2482 | -0.0336 |
| leg penetration | +0.0188 | +0.0053 |

### Filtered average

Filtered excludes only `bucket004_20231003_1_012_p1`, because visual inspection from E147/E143 showed late bucket drift and the contact metric is dominated by that failed trajectory.

| method | cases | fall | 5cm | 10cm | hand pen | deep2 | leg pen | object floor | pelvis min |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `OmniRetarget` | 23 | 0 | 0.6417 | 0.6598 | 0.5440 | 0.2929 | 0.1124 | 0.3667 | 0.6941 |
| `sphere Spider` | 23 | 0 | 0.5512 | 0.6064 | 0.3309 | 0.0137 | 0.0920 | 0.5103 | 0.6770 |
| `rubber hand Spider` | 23 | 0 | 0.5727 | 0.6179 | 0.3010 | 0.0120 | 0.0972 | 0.5051 | 0.6788 |

Diffs:

| metric | OmniRetarget-rubber | rubber-sphere |
|---|---:|---:|
| 5cm | +0.0690 | +0.0216 |
| 10cm | +0.0420 | +0.0114 |
| hand penetration | +0.2430 | -0.0299 |
| leg penetration | +0.0151 | +0.0052 |

## Visual Observations

Representative new E148 keyframes checked:

- `E148_bucket004_20231002_022_p1_rubber_hull/f120.jpg`: sim keeps the bucket close to both hands and upright, with visible hand-object proximity.
- `E148_box021_029_p2_rubber_hull/f120.jpg`: sim roughly follows the bent-over box contact pose, but the body posture remains low and legs are close to the object.
- `E148_box004_083_p2_rubber_hull/f120.jpg`: hand remains near the box side, but the pose includes lower-body proximity consistent with the small leg-penetration increase.
- `E148_box026_139_p1_rubber_hull/f120.jpg`: sim keeps one hand near the box while the carried object is displaced relative to reference, consistent with rubber not solving all work-quality failures.

These observations match the aggregate metrics: rubber hand improves near-contact and reduces hand penetration versus sphere Spider, but it does not close the gap to OmniRetarget contact and slightly worsens leg penetration.

## Claims

| Claim | Result | Evidence |
|---|---|---|
| Compute reuse | PASS | Manifest has 24 rows: 8 `reuse_e147`, 16 `to_run`; remote splits are 8/8; only E148 `to_run` rows were launched. |
| Comparison completeness | PASS | Evaluator produced `method_metric_rows=72`, `case_rows=24`, `missing_rubber_rows=0`; workbook has 24 case rows. |
| Metric clarity | PASS | Workbook contains both 24-case and filtered 23-case summaries; filtered excludes only `bucket004_20231003_1_012_p1`. |
| No algorithm change | PASS | E148 only changes `hand_collision_variant_id=rubber_hull` sidecar scenes/overrides; no reward/optimizer/Omni/E143 sphere baseline changes were made for this experiment. |

## Decision

E148 strengthens the E147 conclusion on the larger E143 workset.

Compared with sphere Spider, rubber hand Spider:

- improves 5cm near-contact by +0.9 pp over all 24 cases and +2.2 pp on the filtered 23 cases;
- improves 10cm near-contact by +0.1 pp over all 24 cases and +1.1 pp filtered;
- reduces hand-object penetration by 3.4 pp over all 24 cases and 3.0 pp filtered;
- slightly increases leg penetration by about 0.5 pp.

Compared with OmniRetarget, rubber hand Spider still has lower 5cm/10cm contact and far lower penetration, so this is a geometry tradeoff rather than a complete contact-quality win. Keep `rubber_hull` as a useful collision/evaluation variant, but do not replace the default sphere Spider baseline with rubber hand Spider yet.

## Validation

- Manifest audit: 24 rows = 8 reuse + 16 to-run; splits are reuse 8 / remote-gpu0 8 / remote-gpu1 8.
- Artifact audit: 16/16 new E148 root NPZ, 16/16 MP4, 16/16 outdir trajectory, 16 logs; 8/8 reused E147 root NPZ/MP4/outdir trajectory exist through manifest paths.
- Full eval: `method_rows=72`, `case_rows=24`, `missing_rubber=0`.
- XLSX recalculation: `total_errors=0`, `total_formulas=1476`.
- Workbook readback: `24case平均` 3 rows, `逐case对比` 24 rows, `filtered平均` 3 rows, `filtered逐case` 23 rows.
