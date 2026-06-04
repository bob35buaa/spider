# E125 RL Hand-Support Preflight Results

Date: 2026-06-03

## Goal

E124 ended with no release/RL-ready row. E125 therefore does not launch another CEM or RL run. It builds a strict downstream preflight manifest for a future hand-support policy objective and exports only converter inputs that are clearly labeled as blocked or fragment-only.

## Implementation

Added:

- `workspace/core4d/plan/134_E125_rl_hand_support_preflight_plan.md`
- `workspace/core4d/scripts/E125/build_rl_hand_support_preflight.py`
- `workspace/core4d/scripts/eval/eval_E125_rl_hand_support_preflight.sh`

The builder scans recent E120-E124 smoke metric tables, keeps only actual test rows (`method == ablation`), applies strict hand-support gates, selects one best row per workset case, and converts selected scene-act qpos `(T,2,42)` or `(T,42)` to source/freejoint `qpos43` using each row's `scene_act.xml`.

Subagent note: a read-only subagent review was attempted for the RL-entry audit, but spawning failed with the existing thread limit. The result is recorded as local audit evidence.

## Outputs

```text
workspace/core4d/results/E125/rl_hand_support_preflight/
  candidate_metrics.tsv
  selected_preflight.tsv
  rl_objective_manifest.tsv
  preflight_summary.json
  preflight_summary.md
  converter_inputs/
```

Summary:

- candidate rows: 52
- selected rows: 4
- RL-ready rows: 0
- training launched: false

Decision counts:

| decision | count |
|---|---:|
| `BLOCK_MAIN_GATE_FAIL` | 13 |
| `FRAGMENT_HOLDOUT_ONLY` | 12 |
| `NO_CONTACT` | 7 |
| `REVIEW_FRAGMENT` | 9 |
| `SHORTCUT_SUPPORT` | 11 |

Selected rows:

| case | experiment | variant | decision | physics | lower-body | non-hand | hand near-zero | pelvis | qpos43 |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| `box004_083_p2` | E120 | `E120_box004_083_p2_support_surface_direct` | `REVIEW_FRAGMENT` | 45.7% | 0.0% | 0.0% | 55.2% | 0.622m | `(105,43)` |
| `box021_029_p2` | E121 | `E121_box021_029_p2_terminal_hard_surface` | `BLOCK_MAIN_GATE_FAIL` | 32.0% | 16.0% | 21.3% | 66.7% | 0.182m | `(75,43)` |
| `box021_035_p1` | E121 | `E121_box021_035_p1_terminal_soft_surface` | `FRAGMENT_HOLDOUT_ONLY` | 71.3% | 0.0% | 0.0% | 81.4% | 0.650m | `(129,43)` |
| `box021_035_p2` | E121 | `E121_box021_035_p2_terminal_soft_surface` | `FRAGMENT_HOLDOUT_ONLY` | 78.2% | 0.0% | 0.0% | 79.7% | 0.587m | `(133,43)` |

All exported converter inputs contain finite `qpos` and `fps` arrays.

## Decision

E125 confirms that there are useful holdout hand-support fragments, but no main-case RL-ready row. `box021_035_p1/p2` can be used for downstream reward inspection or Holosoma adapter debugging, but must not be counted as release candidates. The main `box021_029_p2` remains blocked by posture and non-hand/lower-body support.

Next valid routes:

1. implement explicit stage-local carry constraints or a constrained pose/contact teacher for the main case;
2. move the holdout fragments into a downstream Holosoma hand-support reward inspection run, preserving the `FRAGMENT_HOLDOUT_ONLY` label.

## Validation

Passed:

```text
.venv/bin/python -m py_compile workspace/core4d/scripts/E125/build_rl_hand_support_preflight.py
bash -n workspace/core4d/scripts/eval/eval_E125_rl_hand_support_preflight.sh
bash workspace/core4d/scripts/eval/eval_E125_rl_hand_support_preflight.sh
```

Additional shape/finite check passed for all 4 exported `qpos43` files.
