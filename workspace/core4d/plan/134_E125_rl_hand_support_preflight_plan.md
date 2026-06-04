# E125 RL Hand-Support Preflight Plan

Date: 2026-06-03

## Context

E120-E124 tested support decomposition, terminal carry gates, snap warmstarts, two-stage curriculum, and SBTO. The main `box021_029_p2` case still has no release/RL-ready row. E124 also showed that changing the optimizer horizon alone loses contact rather than recovering a stable carry.

The next useful step is not to launch another CEM full run. It is to make the downstream RL hand-support objective handoff concrete and auditable:

- identify the strongest recent hand-support fragments;
- convert scene-act CEM rollouts back to source/freejoint `qpos43` converter inputs;
- keep strict labels that prevent fragment-only rows from being called RL-ready;
- produce a manifest that the Holosoma side can inspect or consume in a separate RL run.

## Inputs

Recent smoke metrics and rollouts:

- `workspace/core4d/results/E120/cem/smoke/smoke_method_metrics.csv`
- `workspace/core4d/results/E121/cem/smoke/smoke_method_metrics.csv`
- `workspace/core4d/results/E122/cem/smoke/smoke_method_metrics.csv`
- `workspace/core4d/results/E123/cem/smoke/smoke_method_metrics.csv`
- `workspace/core4d/results/E124/cem/smoke/smoke_method_metrics.csv`

The selected trajectory inputs are the corresponding `trajectory_mjwp_act.npz` files and `scene_act.xml` paths recorded in those tables.

## Success Criteria

E125 succeeds if:

1. It writes a complete recent-candidate metric table with deterministic decision labels.
2. It selects one best row per current workset case and exports finite `qpos43` converter inputs for them.
3. It reports `rl_ready_rows=0` unless the main `box021_029_p2` strict hand-support gate is actually satisfied.
4. It explicitly separates `FRAGMENT_HOLDOUT_ONLY` rows from `RL_EXPORT_READY`.
5. It does not launch Holosoma training or any full CEM.

## Decision Rules

A row is `RL_EXPORT_READY` only when all are true:

- case is the main `box021_029_p2`;
- physics hand-object contact `>= 57%`;
- lower-body/object contact `<= 5%`;
- non-hand/object support `<= 5%`;
- hand near-zero support `>= 50%`;
- deep hand penetration `<= 3%`;
- pelvis minimum height `>= 0.55m`;
- object mean error `<= 2cm`;
- local work status is `PASS`.

Rows outside the main case can be `FRAGMENT_HOLDOUT_ONLY` when they satisfy the same geometric/support thresholds. These rows may be useful for downstream reward debugging, but they are not release candidates.

## Outputs

```text
workspace/core4d/results/E125/rl_hand_support_preflight/
  candidate_metrics.tsv
  selected_preflight.tsv
  rl_objective_manifest.tsv
  preflight_summary.json
  preflight_summary.md
  converter_inputs/*.npz
  converter_inputs/*.json
```

## Non-Goals

- Do not run full CEM.
- Do not launch Holosoma RL training.
- Do not change CEM reward code.
- Do not declare holdout fragments as RL-ready release rows.
