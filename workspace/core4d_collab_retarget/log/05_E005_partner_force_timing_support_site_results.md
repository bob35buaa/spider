# E005 Results: corrected partner-force timing and support-site geometry

日期：2026-05-17

## Status

Setup and smoke passed. E005 follows E004 with a code-level correction: partner-force reference indexing now uses explicit per-task `partner_force_ref_dt` instead of hard-coded 30Hz. `box025` uses `0.03333333333333333` from task_info; `box023` uses `0.02` default. Full result tables will be filled after local/remote full runs.

## Planned Commands

```bash
bash workspace/core4d_collab_retarget/scripts/run_E005_preprocess.sh
bash workspace/core4d_collab_retarget/scripts/train/train_E005.sh smoke 0
bash workspace/core4d_collab_retarget/scripts/train/train_E005.sh local_wave 0
bash workspace/core4d_collab_retarget/scripts/run_E005_remote.sh
bash workspace/core4d_collab_retarget/scripts/pull_E005_remote_results.sh
.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E005.py --all
```

## Setup

Variants compare:

- corrected COM spring (`com_s20/s40`) against E004 bugged-timing COM result;
- off-COM support-site force on object-local `-Y/+Y` for box025 and `-X/+X` for box023, matching the ref contact normals from preprocess;
- guard variants on `box023_p2`;
- one hold-contact variant for robot participation.

All variants must keep `scene.xml` true-freejoint parity: `contact_guidance=false`, `scene_name=""`, `object_action_dims=0`, `object_actuator_ids=[]`, `kp_rot=0`.

## Smoke

Command:

```bash
bash workspace/core4d_collab_retarget/scripts/run_E005_preprocess.sh
bash workspace/core4d_collab_retarget/scripts/train/train_E005.sh smoke 0
bash workspace/core4d_collab_retarget/scripts/train/train_E005.sh eval --all
```

Smoke uses `max_sim_steps=4`; it only validates wiring and parity, not task success.

Aggregate:

```json
{
  "num_results": 9,
  "num_freejoint_parity_ok": 9,
  "num_main_results": 6,
  "num_guard_results": 3,
  "num_guard_stable_proxy": 3
}
```

## Result Paths

| Artifact | Path |
|----------|------|
| Results | `workspace/core4d_collab_retarget/results/E005/` |
| Logs | `logs/core4d_collab_retarget/E005/` |
| Overrides | `examples/config/override/core4d_collab_E005_*.yaml` |
| Variants | `workspace/core4d_collab_retarget/scripts/E005/variants.tsv` |

## Pending Result Tables

- Config parity / smoke
- Corrected COM timing metrics
- Support-site side ablation metrics
- Guard stability metrics
- Keyframe/video observations
- Claims C1-C5
- E006 decision
