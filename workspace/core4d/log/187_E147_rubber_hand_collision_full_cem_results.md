# E147 rubber hand collision full CEM results

Date: 2026-06-09 CST

Plan: `workspace/core4d/plan/155_E147_rubber_hand_collision_cem_plan.md`

## Scope

E147 adds `hand_collision_variant_id` as the third v3 downstream/CEM axis and tests `rubber_hull` against historical `sphere5cm` CEM outputs. The A/B holds retarget route, target route, reward, algorithm, and case fixed; only the robot hand collision scene changes.

Rubber implementation:

- `hand_collision_variant_id=rubber_hull`
- sidecar scene: `scene_act_E147_rubber_hull.xml`
- `lh/rh` replaced with mesh geoms backed by `left_rubber_hand/right_rubber_hand`
- `maxhullvert=64`
- source `scene_act.xml` not overwritten

## Execution

Remote full CEM ran on `spider-remote` in tmux session `E147_full_rubber_231132`.

Split:

- GPU0: `e091_box004_083_p1`, `d003_box021_035_p1`, `box023_person2`, `box026_134_p1`, `bucket004_012_p1`
- GPU1: `e091_box004_082_p1`, `d003_box021_035_p2`, `box023_person1`, `box026_134_p2`, `bucket004_021_p1`

Commands:

- launch: `bash workspace/core4d/scripts/run_E147_remote.sh full`
- pull: `bash workspace/core4d/scripts/pull_E147_remote_results.sh full`
- eval: `bash workspace/core4d/scripts/eval/eval_E147_rubber_hand_collision.sh full`

No remote processes were killed. The tmux session ended naturally after all rows completed.

## Artifacts

| Artifact | Path | Count |
|---|---:|---:|
| root NPZ | `workspace/core4d/results/E147/rubber_hand_collision/cem/full/*.npz` | 10 |
| MP4 | `workspace/core4d/results/E147/rubber_hand_collision/cem/full/*_full.mp4` | 10 |
| outdir trajectory | `workspace/core4d/results/E147/rubber_hand_collision/cem/full/*_outdir_full/trajectory_mjwp_act.npz` | 10 |
| run logs | `logs/E147/cem/full/*_full.log` | 10 |
| keyframes | `workspace/core4d/results/E147/rubber_hand_collision/cem/full/keyframes/*/*.jpg` | 100 |
| eval summary | `workspace/core4d/results/E147/rubber_hand_collision/eval/full/e147_eval_summary.md` | 1 |
| Omni/sphere/rubber comparison XLSX | `workspace/core4d/results/E147/rubber_hand_collision/comparison/E147_omni_sphere_rubber_hand_comparison.xlsx` | 2 sheets |
| Omni/sphere/rubber case TSV | `workspace/core4d/results/E147/rubber_hand_collision/comparison/e147_omni_sphere_rubber_case_comparison.tsv` | 10 rows |
| S6 evidence | `workspace/core4d/results/E147/s6_downstream/evidence/downstream_evidence_manifest.tsv` | 10 rows |
| E147 registry | `workspace/core4d/results/E147/registries/case_state_registry.tsv` | 10 rows |

## Quantitative Result

Full evaluator wrote 20 metric rows: 10 historical `sphere5cm` rows and 10 new `rubber_hull` rows.

| method | rows | fall | hand5 | hand8 | hand pen | hand deep2 | physics contact | pelvis min | obj err | leg pen | body pen |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sphere5cm | 10 | 0 | 0.5827 | 0.6052 | 0.4456 | 0.0526 | 0.4027 | 0.6649 | 0.0081 | 0.0787 | 0.0000 |
| rubber_hull | 10 | 1 | 0.5776 | 0.6051 | 0.2877 | 0.0007 | 0.2780 | 0.6125 | 0.0078 | 0.0935 | 0.0051 |

Primary A/B status counts:

- `rubber_primary_pass`: 2
- `rubber_contact_stable_but_penetration_not_improved`: 6
- `rubber_primary_fail`: 2

S6 downstream CEM gate:

- `cem_status=pass`: 3
- `cem_status=fail`: 7
- no historical fail case became downstream CEM pass.

## Pair Notes

| case | old | delta hand5 | delta deep2 | delta hand pen | rubber pelvis | A/B status | S6 |
|---|---|---:|---:|---:|---:|---|---|
| `e091_box004_20231003_2_083_p1` | pass | +0.0098 | +0.0000 | -0.2549 | 0.6274 | contact stable, deep not improved | fail: lowerbody |
| `e091_box004_20231003_2_082_p1` | pass | +0.0183 | +0.0000 | -0.0183 | 0.6419 | contact stable, deep not improved | pass |
| `d003_box021_20231011_035_p1` | pass | +0.0155 | +0.0000 | -0.6047 | 0.6421 | contact stable, deep not improved | pass |
| `d003_box021_20231011_035_p2` | fail | +0.0075 | +0.0000 | -0.3534 | 0.5896 | contact stable, deep not improved | fail: lowerbody |
| `box023_person2` | pass | +0.0074 | -0.0147 | +0.0588 | 0.6693 | primary pass | fail: object floor |
| `box023_person1` | fail | +0.1103 | -0.0588 | -0.0588 | 0.1558 | primary fail | fail: pelvis fall |
| `e091_box026_20231020_134_p1` | pass | +0.0098 | +0.0000 | -0.0294 | 0.7093 | contact stable, deep not improved | fail: object floor |
| `e091_box026_20231020_134_p2` | fail | +0.0093 | +0.0000 | +0.0093 | 0.6869 | contact stable, deep not improved | fail: object floor |
| `bucket004_20231003_1_012_p1` | pass | -0.2800 | -0.2800 | -0.3200 | 0.7073 | primary fail | pass |
| `bucket004_20231002_021_p1` | fail | +0.0414 | -0.1655 | -0.0069 | 0.6957 | primary pass | fail: lowerbody |

## Visual Observations

Representative sheets:

- `workspace/core4d/results/E147/rubber_hand_collision/eval/full/visual_sheets/E147_d003_box021_20231011_035_p1_rubber_hull_sheet.jpg`
- `workspace/core4d/results/E147/rubber_hand_collision/eval/full/visual_sheets/E147_box023_person2_rubber_hull_sheet.jpg`
- `workspace/core4d/results/E147/rubber_hand_collision/eval/full/visual_sheets/E147_box023_person1_rubber_hull_sheet.jpg`
- `workspace/core4d/results/E147/rubber_hand_collision/eval/full/visual_sheets/E147_bucket004_20231003_1_012_p1_rubber_hull_sheet.jpg`
- `workspace/core4d/results/E147/rubber_hand_collision/eval/full/visual_sheets/E147_bucket004_20231002_021_p1_rubber_hull_sheet.jpg`

Observed:

- `d003_box021_035_p1`: stable carry-like motion; no obvious hand-through-box in sampled frames; matches downstream CEM pass.
- `box023_person2`: hand contact improves and deep penetration drops, but box visibly tilts/drags near the floor; matches `object_floor_contact` fail.
- `box023_person1`: mid/late frames show low body posture and collapse toward the object; matches `pelvis_fall`.
- `bucket004_20231003_1_012_p1`: motion is stable and downstream pass, but sampled contact is sparser than historical sphere metric; this explains the A/B primary fail despite work-gate pass.
- `bucket004_20231002_021_p1`: contact/deep metrics improve, but lower-body interference remains; not a rescued fail.

## Decision

E147 supports the geometry hypothesis only partially.

What improved:

- mean hand penetration dropped from 0.4456 to 0.2877;
- mean deep hand penetration dropped from 0.0526 to 0.0007;
- object tracking mean stayed essentially unchanged, 0.0081m to 0.0078m;
- rubber sidecar scene and `hand_collision_variant_id` plumbing are technically viable.

What did not improve enough:

- physics contact dropped from 0.4027 to 0.2780;
- mean hand-near-5cm stayed flat/slightly down, 0.5827 to 0.5776;
- rubber introduced 1 fall and some body penetration;
- no historical fail was rescued into downstream CEM pass.

Conclusion: keep `hand_collision_variant_id=rubber_hull` as a valid v3 CEM axis and useful penetration diagnostic, but do not make it the default replacement for `sphere5cm` yet. The next useful branch is not more rubber-only full CEM; it is a combined contact-quality/work-gate route that preserves the reduced deep penetration while recovering physics contact and reducing lower-body/object-floor failures.

## Validation

- `find workspace/core4d/scripts/data_construction_v3 workspace/core4d/scripts/E147 -name '*.py' -print0 | xargs -0 .venv/bin/python -m py_compile`
- `.venv/bin/python -m py_compile workspace/core4d/scripts/eval/eval_E147_rubber_hand_collision.py`
- `bash -n workspace/core4d/scripts/train/train_E147_rubber_hand_collision.sh workspace/core4d/scripts/run_E147_remote.sh workspace/core4d/scripts/pull_E147_remote_results.sh workspace/core4d/scripts/eval/eval_E147_rubber_hand_collision.sh`
- `git diff --check` on E147/data_construction/docs/progress/tracker paths
- `bash workspace/core4d/scripts/data_construction_v3/orchestration/run_release_checks.sh --no-smoke` -> 65 passed / 0 failed
- E147 artifact audit: 10/10 NPZ, 10/10 MP4, 10/10 outdir trajectories/configs, 10/10 logs, 100 keyframes
- Comparison XLSX formula recalculation: `total_errors=0`, `total_formulas=245`
