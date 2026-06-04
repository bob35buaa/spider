# E139 E138 Semantic Ref-Object-Contact Env Probe Plan

## Context

E138 converted four E137 trimmed qpos-style semantic contact candidates into Holosoma WBT motion files. CPU `MotionLoader` verified that each converted file contains `object_contact` with the expected `(frames, 2)` shape.

The remaining runtime gap is whether Holosoma env stepping exposes those semantic masks through `motion_command.ref_object_contact`. E133 already proved this plumbing for E131 geometry-proxy masks. E139 repeats the bounded no-debug env probe on E138 semantic masks.

## Claims

1. E139 partner-injected E138 semantic WBT files can start under the Holosoma Box021 handbox partner env.
2. `motion_command.motion.has_object_contact` is true for every E138 semantic file.
3. `motion_command.motion.has_partner` is true for every E139 partner-injected file.
4. `motion_command.ref_object_contact` returns nonzero semantic contact samples during bounded no-debug stepping.
5. E139 does not launch PPO, CEM, remote jobs, or checkpoint creation.

## Scope

Probe all four E138 converted semantic files:

- `box021_029_p1`
- `box021_029_p2`
- `box021_035_p1`
- `box021_035_p2`

Use the existing R135 Box021 handbox env alias:

`exp:g1-29dof-wbt-w-object-r135-box021-handbox-exp0601-v4-3`

Because E138 files do not contain `partner_hand_pos_w` or `partner_hand_quat_w`, E139 first writes separate E139 partner-injected copies. It does not modify E138 outputs in place. Partner hands are sampled from the paired E138 converted motion using nearest raw-frame alignment derived from E137/E138 provenance.

## Implementation

1. Add `workspace/core4d/scripts/E139/build_e138_partner_semantic_motions.py`.
2. Add `workspace/core4d/scripts/E139/holosoma_semantic_ref_object_contact_no_debug_probe.py`.
3. Add fixed eval entry `workspace/core4d/scripts/eval/eval_E139_e138_semantic_ref_object_contact_env_probe.sh`.
4. Build partner-injected semantic motions:
   - read E138 manifest;
   - pair p1/p2 rows by case ID;
   - compute raw-frame index per converted frame as `E137 trim_start + E138 object_contact_source_frame_index`;
   - sample partner wrists by nearest partner converted raw-frame index;
   - write `partner_hand_pos_w (T,2,3)` and `partner_hand_quat_w (T,2,4)`;
   - record partner provenance and nearest raw-frame diff.
5. Reuse the E133 no-debug pattern:
   - initialize Holosoma/IsaacSim;
   - disable debug drawing;
   - patch replay sleep for fast bounded stepping;
   - load one motion at a time;
   - collect full-motion contact stats and stepped `ref_object_contact` stats;
   - emit JSON summary markers for shell-side parsing.
6. Parse probe logs into:
   - `e139_ref_object_contact_manifest.tsv`;
   - `e139_ref_object_contact_summary.json`;
   - `e139_ref_object_contact_summary.md`.

## Success Criteria

- Four probe rows run with return code 0 and no timeout/error marker.
- Every row reports `has_object=true`.
- Every row reports `has_partner=true`.
- Every row reports `has_object_contact=true`.
- Every row reports full-motion `motion_contact_total > 0`.
- Every row reports stepped `ref_contact_total > 0`.
- Summary reports `semantic_ref_mask_runtime_ready_rows=4`.
- `rl_ready_rows=0`, `training_launched=false`, `cem_launched=false`, `remote_jobs_launched=false`.

## Non-Goals

- Do not launch PPO or create checkpoints.
- Do not run CEM or remote multi-GPU jobs.
- Do not mark E139 outputs as RL-ready.
- Do not modify E138 outputs in place.
- Do not use head-crop partner alignment without recording the policy.

## Fixed Command

```bash
bash workspace/core4d/scripts/eval/eval_E139_e138_semantic_ref_object_contact_env_probe.sh
```

## Expected Outputs

- `workspace/core4d/results/E139/e138_semantic_ref_object_contact_env_probe/e139_ref_object_contact_manifest.tsv`
- `workspace/core4d/results/E139/e138_semantic_ref_object_contact_env_probe/e139_ref_object_contact_summary.json`
- `workspace/core4d/results/E139/e138_semantic_ref_object_contact_env_probe/e139_ref_object_contact_summary.md`
- `workspace/core4d/results/E139/e138_semantic_ref_object_contact_env_probe/e139_partner_injection_manifest.tsv`
- `workspace/core4d/results/E139/e138_semantic_ref_object_contact_env_probe/partner_semantic_motions/`
- `workspace/core4d/results/E139/e138_semantic_ref_object_contact_env_probe/logs/`
