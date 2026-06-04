# E138 E137 Semantic Contact Converter Preflight Plan

## Context

E135 reminted semantic raw-contact masks for the bounded Box021 raw cases. E136 showed that E107 Box021 exports can be mapped to those masks. E137 wrote qpos-style E107 semantic `object_contact` candidates, but those files are not Holosoma runtime motions: they contain `qpos` plus contact metadata, while Holosoma WBT `MotionLoader` expects converted `joint_pos/body_*` files.

Read-only audit and local inspection agree on the bridge:

- `convert_data_format_mj.py` reads only `qpos` and drops extra keys.
- Holosoma `MotionLoader` reads `object_contact` only from converted WBT motion files, with exact shape `(joint_pos_frames, 2)`.
- E126 established the safer converter pattern: save qpos-only inputs without an `fps` key, then pass `--input-fps` explicitly.

## Claims

1. E137 trimmed semantic contact files can be converted into Holosoma WBT motion files without treating qpos-style files as runtime-ready.
2. The semantic `object_contact` channel can be post-injected after conversion with an explicit temporal mapping policy from 30 Hz input masks to 50 Hz converted frames.
3. CPU Holosoma `MotionLoader` can load the converted files and report `has_object_contact=True` when the injected shape matches `(frames, 2)`.

## Scope

Primary scope is the four E107 trimmed E137 rows:

- `box021_029_p1`
- `box021_029_p2`
- `box021_035_p1`
- `box021_035_p2`

Retargeted-untrimmed rows remain out of scope for this preflight to keep the first runtime-format bridge focused on trimmed training-style fragments.

## Implementation

1. Add `workspace/core4d/scripts/E138/convert_e137_semantic_contact_to_holosoma.py`.
2. Read `workspace/core4d/results/E137/e107_semantic_object_contact_export/e137_e107_semantic_object_contact_manifest.tsv`.
3. Filter `source_export_kind == E107_trimmed`.
4. For each row:
   - save a qpos-only converter input under E138 results;
   - run Holosoma `convert_data_format_mj.py` with `--input-fps 30`, `--output-fps 50`, `--has-dynamic-object`, `--object-name Box021`, and `--once`;
   - map the 30 Hz bool contact mask to the converter output frame grid using nearest-time source frame selection on the converter duration `(input_frames - 1) / input_fps`;
   - inject `object_contact`, `object_contact_3cm`, `object_contact_5cm`, and provenance keys into the converted WBT NPZ;
   - validate required converted keys, finite arrays, contact shape, dtype, and non-empty contact.
5. Add `workspace/core4d/scripts/eval/eval_E138_e137_semantic_contact_converter_preflight.sh`.
6. Run a CPU `MotionLoader` contract probe inside the Holosoma setup environment and record whether each converted file loads with `has_object_contact=True`.

## Success Criteria

- Four trimmed E137 rows are converted and post-injected.
- Every output has `joint_pos`, `joint_vel`, `body_pos_w`, `body_quat_w`, `object_pos_w`, `object_quat_w`, `object_lin_vel_w`, and `object_contact`.
- `object_contact.shape == (joint_pos.shape[0], 2)` for every row.
- Manifest records input frames, converted frames, fps, temporal mapping policy, source index min/max, and status.
- CPU `MotionLoader` probe passes for every converted output.

## No-Go / Non-Goals

- Do not launch CEM, PPO, Holosoma RL training, remote jobs, or checkpoint creation.
- Do not mark any E138 output as RL-ready.
- Do not load qpos-style E137 files directly in Holosoma runtime.
- Do not inject `object_contact` before conversion and assume it survives.
- Do not silently pad or resize a mismatched mask. A mismatch is a failed row.

## Fixed Command

```bash
bash workspace/core4d/scripts/eval/eval_E138_e137_semantic_contact_converter_preflight.sh
```

## Expected Outputs

- `workspace/core4d/results/E138/e137_semantic_contact_converter_preflight/converter_inputs/`
- `workspace/core4d/results/E138/e137_semantic_contact_converter_preflight/converted/`
- `workspace/core4d/results/E138/e137_semantic_contact_converter_preflight/e138_manifest.tsv`
- `workspace/core4d/results/E138/e137_semantic_contact_converter_preflight/e138_summary.json`
- `workspace/core4d/results/E138/e137_semantic_contact_converter_preflight/e138_summary.md`
