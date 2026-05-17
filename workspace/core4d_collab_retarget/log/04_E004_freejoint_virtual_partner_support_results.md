# E004 Results: true-freejoint virtual partner support sweep

日期：2026-05-17

## Status

Setup and smoke passed. Full Wave A/B runs are next: local GPU runs the `local` queue, remote GPUs run `remote_gpu0` / `remote_gpu1`.

## Planned Commands

```bash
bash workspace/core4d_collab_retarget/scripts/run_E004_preprocess.sh
bash workspace/core4d_collab_retarget/scripts/train/train_E004.sh smoke 0
bash workspace/core4d_collab_retarget/scripts/train/train_E004.sh local_wave 0
bash workspace/core4d_collab_retarget/scripts/run_E004_remote.sh
bash workspace/core4d_collab_retarget/scripts/pull_E004_remote_results.sh
```

## Setup

Nine Wave A/B variants were generated. All full variants keep the required freejoint parity:

- `scene_name: ""`
- `contact_guidance: false`
- `object_pd_override: false`
- `object_action_dims: 0`
- `partner_force_spring_kp_rot: 0`

Wave C rotation probes remain commented out in `variants.tsv` and are not part of the default smoke/full queues.

## Smoke

Command:

```bash
bash workspace/core4d_collab_retarget/scripts/run_E004_preprocess.sh
bash workspace/core4d_collab_retarget/scripts/train/train_E004.sh smoke 0
.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E004.py --all
```

Smoke uses `max_sim_steps=4`; it only validates wiring and parity, not task success.

| Variant | T | nu | nq_obj | contact_guidance | scale | kp | kp_rot | hold | parity |
|---------|---|----|--------|------------------|-------|----|--------|------|--------|
| `E004_box025_p2_g05` | 4 | 29 | 7 | False | 0.5 | 0.0 | 0.0 | 0.0 | True |
| `E004_box025_p2_s10` | 4 | 29 | 7 | False | 0.5 | 10.0 | 0.0 | 0.0 | True |
| `E004_box025_p2_s20` | 4 | 29 | 7 | False | 0.5 | 20.0 | 0.0 | 0.0 | True |
| `E004_box025_p2_s40` | 4 | 29 | 7 | False | 0.5 | 40.0 | 0.0 | 0.0 | True |
| `E004_box023_p2_s10` | 4 | 29 | 7 | False | 0.5 | 10.0 | 0.0 | 0.0 | True |
| `E004_box023_p2_s20` | 4 | 29 | 7 | False | 0.5 | 20.0 | 0.0 | 0.0 | True |
| `E004_box025_p2_s20_hc` | 4 | 29 | 7 | False | 0.5 | 20.0 | 0.0 | 1.0 | True |
| `E004_box025_p2_s40_hc` | 4 | 29 | 7 | False | 0.5 | 40.0 | 0.0 | 1.0 | True |
| `E004_box023_p2_s10_hc` | 4 | 29 | 7 | False | 0.5 | 10.0 | 0.0 | 1.0 | True |

## Result Paths

| Artifact | Path |
|----------|------|
| Results | `workspace/core4d_collab_retarget/results/E004/` |
| Logs | `logs/core4d_collab_retarget/E004/` |
| Overrides | `examples/config/override/core4d_collab_E004_*.yaml` |
| Variants | `workspace/core4d_collab_retarget/scripts/E004/variants.tsv` |

## Pending Result Tables

To fill after evaluation:

- Preprocess / config parity
- Smoke
- Wave A full metrics
- Wave B full metrics
- Keyframe/video observations
- Claims C1-C5
- Next decision
