# E137 E107 Semantic Object-Contact Export Preflight Plan

Date: 2026-06-03

## Context

E136 proved that E135 v3 raw-contact masks can be safely mapped to E107
Holosoma-side qpos exports:

- retargeted-untrimmed exports map by direct raw-axis frame equality;
- trimmed exports map by explicit `trim_window.json` slice proof.

E136 also proved that E126/E131 fragment exports must remain blocked because
they lack raw-window mapping. E137 therefore writes semantic `object_contact`
candidate NPZs only for the E107 candidates, and records an artifact-level
contract check. It does not run CEM, PPO, Holosoma training, checkpoints, or
remote jobs.

## Claims

1. E107 direct and trimmed candidates from E136 can be copied into isolated E137
   output files with a semantic `object_contact (T,2)` bool array.
2. The written `object_contact` arrays exactly match the E136-proven frame
   count and mapping slice.
3. Export metadata records the E135 raw-contact source, threshold, mapping type,
   mapping proof, raw case ID, person index, and trim window when applicable.
4. E126/E131 exports are not written by E137.
5. E137 does not claim Holosoma `MotionLoader`/runtime readiness for qpos-style
   E107 files; that remains a later converter/runtime probe.
6. E137 does not launch CEM/PPO/training/remote jobs.

## Inputs

```text
workspace/core4d/results/E136/e135_semantic_contact_holosoma_bridge/e136_e135_semantic_contact_bridge_manifest.tsv
workspace/core4d/results/E135/box021_v3_s1_raw_contact_remine/s1_raw_contact/raw_contact/per_sequence/*/raw_contact_proxy.npz
workspace/core4d/results/E107/s6_downstream/rl_export/full_cem_omnirt/results/omnirt_v1/*/{retargeted,trimmed}/*.npz
```

## Implementation

Add:

```text
workspace/core4d/scripts/E137/export_e107_semantic_object_contact.py
workspace/core4d/scripts/eval/eval_E137_e107_semantic_object_contact_export.sh
```

The exporter will:

- read only E136 rows with `semantic_bridge_candidate=true`;
- write the default semantic label `3cm` as `object_contact`;
- also store `object_contact_3cm` and `object_contact_5cm` arrays where both
  threshold rows exist;
- preserve original qpos export keys while adding semantic contact metadata;
- write manifest/summary artifacts under
  `workspace/core4d/results/E137/e107_semantic_object_contact_export/`.

## Success Criteria

- Fixed local eval entry runs end to end.
- Eight E107 semantic export NPZ files are written: four retargeted-untrimmed
  and four trimmed.
- Every output contains `object_contact (T,2)` bool and matching
  `object_contact_3cm/object_contact_5cm` arrays.
- Manifest rows verify `source_frames == output_frames == object_contact_frames`.
- E126/E131 written rows are zero.
- `motionloader_runtime_probe_launched=false`, `training_launched=false`,
  `cem_launched=false`, and `remote_jobs_launched=false`.

## Command

```bash
bash workspace/core4d/scripts/eval/eval_E137_e107_semantic_object_contact_export.sh
```

## No-Go Rules

- Do not write into E107 source directories.
- Do not write E126/E131 semantic masks.
- Do not resize, interpolate, or pad masks.
- Do not claim runtime readiness for qpos-style files without a separate
  loader/runtime probe.
- Do not launch CEM, PPO, Holosoma training, checkpoints, or remote jobs.
