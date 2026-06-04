# E136 E135 Semantic Contact to Holosoma Bridge Audit Plan

Date: 2026-06-03

## Context

E135 closed the bounded Box021 v3 S1 raw-contact gap by producing pass rows and
raw-contact proxy NPZ files for:

- `20231011/035` with 182 raw frames;
- `20231018/029` with 134 raw frames.

Those artifacts are still raw-axis masks. E134 already established that legacy
semantic masks and E131 geometry proxies must not be treated as Holosoma
semantic `object_contact` without a safe time-axis bridge.

E136 audits whether the new E135 v3 raw-contact masks can bridge to existing
Holosoma-style exports without resizing, interpolation, or silent semantic
changes. It is an audit/preflight only: it does not write semantic
`object_contact` exports and does not launch CEM, PPO, Holosoma training,
checkpoints, or remote jobs.

## Claims

1. E135 raw-contact masks expose valid per-person/per-hand 3cm and 5cm arrays
   for the bounded Box021 cases.
2. E107 retargeted-untrimmed exports can be classified as direct raw-axis
   candidates only when their frame count equals the E135 raw frame count.
3. E107 trimmed exports can be classified as slice candidates only when
   `trim_window.json` proves `untrimmed_frames == raw_frames`,
   `trimmed_frames == export_frames`, and `trim_end - trim_start == export_frames`.
4. E126/E131 fragment exports remain blocked unless an explicit raw-window or
   retarget-window mapping exists; matching by shape alone is not sufficient.
5. E136 does not write Holosoma semantic `object_contact` exports and does not
   launch CEM/PPO/training/remote jobs.

## Inputs

E135 raw-contact evidence:

```text
workspace/core4d/results/E135/box021_v3_s1_raw_contact_remine/
  s1_raw_contact/raw_contact/per_sequence/20231011_035_box021/raw_contact_proxy.npz
  s1_raw_contact/raw_contact/per_sequence/20231018_029_box021/raw_contact_proxy.npz
  registries_combined_5cm_then_3cm/case_state_registry.tsv
```

Holosoma-side exports:

- E107 retargeted and trimmed `box021_20231011_035_p1/p2`.
- E107 retargeted and trimmed `box021_20231018_029_p1/p2`.
- E126 paired fragment exports for `box021_035_p1/p2`.
- E131 structural proxy exports for `box021_035_p1/p2`.

## Implementation

Add:

```text
workspace/core4d/scripts/E136/audit_e135_semantic_contact_holosoma_bridge.py
workspace/core4d/scripts/eval/eval_E136_e135_semantic_contact_holosoma_bridge.sh
```

The audit script will write:

```text
workspace/core4d/results/E136/e135_semantic_contact_holosoma_bridge/
  e136_e135_semantic_contact_bridge_manifest.tsv
  e136_e135_semantic_contact_bridge_summary.json
  e136_e135_semantic_contact_bridge_summary.md
```

## Success Criteria

- Fixed local eval entry runs end to end.
- Every row records source raw frames, export frames, threshold, person index,
  mapping type, mapping proof, and failure mode.
- E107 direct and trim-window candidates are counted separately.
- E126/E131 rows are explicitly blocked unless mapping evidence exists.
- `semantic_object_contact_exports_written=0`, `training_launched=false`,
  `cem_launched=false`, and `remote_jobs_launched=false`.

## Command

```bash
bash workspace/core4d/scripts/eval/eval_E136_e135_semantic_contact_holosoma_bridge.sh
```

## No-Go Rules

- Do not resize or interpolate E135 masks.
- Do not infer E126/E131 fragment alignment from matching frame counts.
- Do not treat E131 `actor_rubber_hand_box021_surface_proxy_5cm` as semantic
  contact evidence.
- Do not update the global v3 registry.
- Do not launch CEM, PPO, Holosoma training, checkpoints, or remote jobs.
