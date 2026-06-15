# E162 — E147+ RL-safe unified re-evaluation plan

Date: 2026-06-14

## 0. Context

E161 `surfaceBandReleaseDecay` looked good under clean8 aggregate metrics, but downstream RL found
`box023_p2` dropped from high-success to full failure. The root cause is an evaluation mismatch:

- RL primarily depends on in-mask robot hand contact being present.
- RL is much less sensitive to lower penetration when the contact gate is not active.
- E161 improved aggregate penetration/release metrics but reduced per-case contact on `box023_p2`.

Therefore the next step is not another CEM run. It is a pure re-evaluation pass over all post-E147
retargeting/CEM versions with a single baseline:

```text
baseline = E147 spider-rubberhand
```

Any method after E147 must be judged by per-case contact preservation relative to E147, not only by
aggregate clean8 mean.

Reference principle:

```text
For downstream RL, in-mask hand contact is a hard floor.
Penetration/release false contact are secondary objectives that can only improve a method after
the method preserves per-case contact.
```

## 1. Scope

### 1.1 Included experiments / methods

The audit covers every post-E147 method that produced reusable retargeting/CEM trajectories or was
used as a clean/RL candidate:

| Exp | Role | Include |
|---|---|---|
| E147 | unified baseline: `spider-rubberhand` | yes, baseline only |
| E148 | 24-case rubber-hand extension | yes |
| E149 | E143 clean rubber benchmark | yes |
| E150 | contact anchor `eef_offset` sweep | yes |
| E151 | Route-B hand surface contact reward | yes |
| E152 | hand-object physics gate | yes |
| E153 | gate threshold sweep | yes |
| E155 | release smooth transition variants | yes |
| E156 | clean8 Omni / rubberhand / gateA / decay benchmark | yes |
| E158 | gateA + surfaceBand-A | yes |
| E159 | gateA + surfaceBand-A2 | yes |
| E160 | surfaceBand-A2 + posture rerank | yes |
| E161 | surface release ablation | yes |

E154 is not a method version; it is the metric standard. E157 is an RL export/handoff and should be
audited through the source method rows, not as an independent retargeting method.

### 1.2 Case coverage rule

Use E147 `spider-rubberhand` as the canonical baseline for each case whenever an E147 baseline row
exists.

If a later method contains a case not present in E147:

- write `baseline_status=missing_e147_case`;
- compute absolute metrics and body tracking;
- do not mark it `rl_safe_pass`;
- report it separately as `needs_baseline_or_manual_review`.

For clean8 rows that were historically imported through E148/E149/E156, canonicalize aliases so that
the baseline method still appears as `E147_spider_rubberhand` in the new outputs.

## 2. New hard criterion

### 2.1 Per-case contact preservation gate

Primary RL gate metric:

```text
contact_inmask = hand_object_physics_contact_in_mask_frac
```

Hard regression rule:

```text
contact_inmask_delta_vs_E147 = method_contact_inmask - E147_contact_inmask

if contact_inmask_delta_vs_E147 < -0.05:
    case_status = contact_regression_fail
```

This failure is final for that case:

- it is not overridden by lower penetration;
- it is not overridden by better release false contact;
- it is not overridden by better object tracking or body tracking.

### 2.2 Method-level RL-safe gate

A method is `rl_safe_method_pass=true` only if all applicable cases pass:

```text
all cases with E147 baseline:
    contact_inmask_delta_vs_E147 >= -0.05
    success_tracked == true
    fall_flag == false
```

The method summary must list:

```text
failed_cases
contact_regression_failed_cases
tracking_failed_cases
baseline_missing_cases
```

### 2.3 Secondary metrics

These are reported but cannot compensate for contact regression:

- `hand_object_physics_penetration_3mm_frame_frac`
- `hand_object_physics_penetration_5mm_frame_frac`
- `hand_object_physics_contact_3mm_in_mask_frac`
- `hand_object_physics_contact_5mm_in_mask_frac`
- `hand_geom_penetration_2mm_frac`
- `hand_geom_penetration_5mm_frac`
- `hand_object_release_false_contact_3mm_frac`
- `hand_object_release_false_contact_5mm_frac`
- `leg_penetration_frac`
- `obj_err_mean_m`

## 3. Body tracking metrics aligned to SPIDER Table 4

The current E154+ tracking diagnostics are useful but incomplete for Table 4 alignment. They include:

```text
track_root_pos_err_mean_m
track_root_pos_err_terminal_m
track_root_quat_err_mean
track_root_quat_err_terminal
track_joint_err_mean_rad
track_joint_err_terminal_rad
track_pelvis_z_err_mean_m
track_pelvis_z_err_terminal_m
```

Add Table-4-style fields to the shared evaluator:

| Field | Unit | Definition |
|---|---:|---|
| `track_joint_err_deg_mean` | deg | mean absolute joint angle error over robot joints `qpos[7:36]` |
| `track_eef_pos_err_cm_mean` | cm | mean L/R wrist position error |
| `track_eef_ori_err_deg_mean` | deg | mean L/R wrist orientation geodesic error |
| `track_root_pos_err_cm_mean` | cm | pelvis/root position error |
| `track_root_ori_err_deg_mean` | deg | pelvis/root orientation error |
| `track_obj_pos_err_cm_mean` | cm | object position error when a fixed ref object pose is available |
| `track_obj_ori_err_deg_mean` | deg | object orientation error when a fixed ref object pose is available |

Ground truth is always the SPIDER input kinematic reference:

```text
<case>/0/trajectory_kinematic.npz
```

This is the OmniRetarget retargeting result used as SPIDER input. Do not compare against a run's own
drifted/internal reference.

Reuse existing historical logic from:

```text
workspace/core4d/scripts/eval/legacy/eval_e035_comprehensive.py
workspace/core4d/docs/eval_metrics.md
```

## 4. Implementation plan

### 4.1 Build a post-E147 audit manifest

Create:

```text
workspace/core4d/scripts/experiments/E162/build_post_e147_reeval_manifest.py
workspace/core4d/scripts/experiments/E162/variants.tsv
```

Manifest rows should normalize:

```text
source_exp_id
source_method_id
canonical_method_id
case_id
short_case_id
qpos_path
scene_xml
video_path
contact_mask_path
person_idx
baseline_method_id=E147_spider_rubberhand
baseline_qpos_path
baseline_source_exp_id
baseline_status
```

Input sources should be read from existing result TSVs and variants files, not hand-coded per row.

### 4.2 Extend shared metrics

Modify:

```text
workspace/core4d/scripts/eval/core/core_metrics.py
workspace/core4d/scripts/eval/core/METRICS_STANDARD.md
```

Add the Table 4 tracking fields while preserving backward compatibility. Existing evaluators should
continue to run if the new fields are unused.

### 4.3 Add a unified post-E147 evaluator

Create:

```text
workspace/core4d/scripts/eval/runners/eval_E162_post_e147_rl_safe_reeval.py
workspace/core4d/scripts/eval/wrappers/eval_E162_post_e147_rl_safe_reeval.sh
```

Outputs:

```text
workspace/core4d/results/E162/post_e147_rl_safe_reeval/eval/full/e162_method_metrics.tsv
workspace/core4d/results/E162/post_e147_rl_safe_reeval/eval/full/e162_delta_vs_E147.tsv
workspace/core4d/results/E162/post_e147_rl_safe_reeval/eval/full/e162_case_failures.tsv
workspace/core4d/results/E162/post_e147_rl_safe_reeval/eval/full/e162_method_summary.tsv
workspace/core4d/results/E162/post_e147_rl_safe_reeval/eval/full/E162_post_E147_RL_safe_reeval.xlsx
```

Required XLSX sheets:

```text
method_summary
per_case_metrics
delta_vs_E147
case_failures
tracking_table4
baseline_missing
```

### 4.4 Update logs after evaluation

After implementation and strict evaluation, write:

```text
workspace/core4d/log/205_E162_post_e147_rl_safe_reeval_results.md
```

The log must explicitly revise E161's previous conclusion if the new gate fails.

## 5. Expected findings to verify

The audit should specifically verify:

1. Whether E161 `surfaceBandReleaseDecay` fails `box023_p2` relative to E147.
2. Whether E160/E161 contact gains on average hide any other per-case regressions.
3. Whether E152/E153 gate variants preserve E147 contact on clean cases while reducing penetration.
4. Whether E155/E156 decay variants improve contact without creating new E147-relative regressions.
5. Which post-E147 method is the best RL-safe candidate under:

```text
primary: no per-case contact_inmask drop > 0.05 vs E147
secondary: lower penetration / release false / leg penetration
diagnostic: Table 4 body tracking
```

## 6. Claims

| Claim | Success criterion |
|---|---|
| C1: E147 can serve as a unified baseline | every audited overlapping case has an E147 baseline row or is listed as baseline missing |
| C2: contact regression is no longer hidden by aggregation | any case with raw in-mask contact drop >0.05 appears in `case_failures` and method summary |
| C3: E161 box023 regression is caught | `box023_p2/person2` releaseDecay is marked `contact_regression_fail` if the known drop persists |
| C4: Table 4 tracking is available | method/per-case TSV includes joint deg, EEF pos cm, EEF ori deg, root cm/deg fields |
| C5: no CEM rerun required | all outputs are generated from existing trajectories |

## 7. Verification commands

Static checks:

```bash
python3 -m py_compile workspace/core4d/scripts/experiments/E162/build_post_e147_reeval_manifest.py
python3 -m py_compile workspace/core4d/scripts/eval/runners/eval_E162_post_e147_rl_safe_reeval.py
bash -n workspace/core4d/scripts/eval/wrappers/eval_E162_post_e147_rl_safe_reeval.sh
```

Evaluation:

```bash
bash workspace/core4d/scripts/eval/wrappers/eval_E162_post_e147_rl_safe_reeval.sh full
```

Sanity checks:

```bash
python3 - <<'PY'
import pandas as pd
root = 'workspace/core4d/results/E162/post_e147_rl_safe_reeval/eval/full'
fail = pd.read_csv(f'{root}/e162_case_failures.tsv', sep='\t')
summary = pd.read_csv(f'{root}/e162_method_summary.tsv', sep='\t')
print(fail[['short_case_id','canonical_method_id','case_status','contact_inmask_delta_vs_E147']].head(20))
print(summary[['canonical_method_id','rl_safe_method_pass','failed_cases']].to_string(index=False))
PY
```

## 8. Stop conditions

Stop and report before modifying method conclusions if:

- E147 baseline rows cannot be mapped for more than two clean8 cases.
- The same case has multiple incompatible E147 baseline trajectories and no deterministic canonical choice.
- Table 4 tracking cannot be computed because scene/ref qpos layouts are incompatible.
