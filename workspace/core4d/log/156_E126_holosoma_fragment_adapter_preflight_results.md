# E126 Holosoma Fragment Adapter Preflight Results

Date: 2026-06-03

## Goal

E125 found no main-case RL-ready row, but it did identify `box021_035_p1` and `box021_035_p2` as same-sequence `FRAGMENT_HOLDOUT_ONLY` rows with clean hand-support geometry. E126 tests only the downstream adapter path: whether those two fragments can be converted into Holosoma-style motion NPZ files with object motion and partner hand fields.

This is not a CEM run, not Holosoma RL training, and not release evidence for `box021_029_p2`.

## Implementation

Added:

- `workspace/core4d/plan/135_E126_holosoma_fragment_adapter_preflight_plan.md`
- `workspace/core4d/scripts/E126/export_holosoma_fragment_adapter.py`
- `workspace/core4d/scripts/eval/eval_E126_holosoma_fragment_adapter.sh`

The adapter reads E125 `selected_preflight.tsv`, requires the two `box021_035` rows to keep `source_decision=FRAGMENT_HOLDOUT_ONLY`, strips the stored `fps` key before invoking Holosoma's MJ converter, converts each person to `_mj_w_obj.npz`, then builds paired files by adding the other person's left/right wrist pose as:

- `partner_hand_pos_w (T,2,3)`
- `partner_hand_quat_w (T,2,4)`

The paired fragments are head-cropped to the shared converted length and marked with `alignment_policy=min_frames_head_crop_fragment_pair_no_raw_window`.

## Outputs

```text
workspace/core4d/results/E126/holosoma_fragment_adapter_preflight/
  e126_adapter_manifest.tsv
  e126_adapter_summary.json
  e126_adapter_summary.md
  converter_inputs/
  exports/
  logs/convert.log
```

Summary:

- rows: 2
- pass rows: 2
- RL-ready rows: 0
- training launched: false
- failures: 0

Rows:

| case | partner | source decision | paired frames | output fps | partner L mean | partner R mean | adapter |
|---|---|---|---:|---:|---:|---:|---|
| `box021_035_p1` | `box021_035_p2` | `FRAGMENT_HOLDOUT_ONLY` | 214 | 50.0 | 0.368m | 0.357m | pass |
| `box021_035_p2` | `box021_035_p1` | `FRAGMENT_HOLDOUT_ONLY` | 214 | 50.0 | 0.388m | 0.376m | pass |

Paired exports:

- `workspace/core4d/results/E126/holosoma_fragment_adapter_preflight/exports/E126_box021_035_p1_with_partner_box021_035_p2_mj_w_obj_w_partner.npz`
- `workspace/core4d/results/E126/holosoma_fragment_adapter_preflight/exports/E126_box021_035_p2_with_partner_box021_035_p1_mj_w_obj_w_partner.npz`

Static NPZ validation confirmed each paired export has the required Holosoma inspection keys, finite values, and expected shapes:

- `joint_pos (214,36)`
- `joint_vel (214,35)`
- `body_pos_w (214,52,3)`
- `object_pos_w (214,3)`
- `partner_hand_pos_w (214,2,3)`
- `partner_hand_quat_w (214,2,4)`
- `fps [50]`

## Decision

E126 succeeds as an adapter preflight for holdout fragment inspection. It preserves `rl_train_allowed=false` and `FRAGMENT_HOLDOUT_ONLY`, so these files may be used to inspect downstream hand-support reward wiring but must not be counted as release, main-case, or RL-ready evidence.

The main case `box021_029_p2` remains blocked by the E125 main gate.

## Validation

Passed:

```text
.venv/bin/python -m py_compile workspace/core4d/scripts/E126/export_holosoma_fragment_adapter.py
bash -n workspace/core4d/scripts/eval/eval_E126_holosoma_fragment_adapter.sh
bash workspace/core4d/scripts/eval/eval_E126_holosoma_fragment_adapter.sh --force
bash workspace/core4d/scripts/eval/eval_E126_holosoma_fragment_adapter.sh
```

Additional paired NPZ key/shape/finite validation passed for both exports.
