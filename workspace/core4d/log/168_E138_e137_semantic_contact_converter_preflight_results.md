# E138 E137 Semantic Contact Converter Preflight Results

## Summary

E138 converted the four E137 trimmed qpos-style semantic contact candidates into Holosoma WBT motion format and post-injected `object_contact` after conversion.

Result: pass.

- Converted rows: 4/4.
- Post-injection rows: 4/4.
- CPU Holosoma `MotionLoader` rows: 4/4 pass.
- RL-ready rows: 0.
- Training launched: false.
- CEM launched: false.
- Remote jobs launched: false.

## Result Paths

| artifact | path |
|---|---|
| manifest | `workspace/core4d/results/E138/e137_semantic_contact_converter_preflight/e138_manifest.tsv` |
| summary json | `workspace/core4d/results/E138/e137_semantic_contact_converter_preflight/e138_summary.json` |
| summary md | `workspace/core4d/results/E138/e137_semantic_contact_converter_preflight/e138_summary.md` |
| qpos-only converter inputs | `workspace/core4d/results/E138/e137_semantic_contact_converter_preflight/converter_inputs/` |
| converted raw WBT motions | `workspace/core4d/results/E138/e137_semantic_contact_converter_preflight/converted_raw/` |
| converted WBT motions with contact | `workspace/core4d/results/E138/e137_semantic_contact_converter_preflight/converted/` |
| converter logs | `workspace/core4d/results/E138/e137_semantic_contact_converter_preflight/logs/` |

## Row Results

| case | input frames @30Hz | converted frames @50Hz | object_contact shape | either active | MotionLoader |
|---|---:|---:|---|---:|---|
| `box021_029_p1` | 71 | 117 | `117x2` | 0.683761 | pass |
| `box021_029_p2` | 75 | 124 | `124x2` | 0.741935 | pass |
| `box021_035_p1` | 127 | 210 | `210x2` | 0.795238 | pass |
| `box021_035_p2` | 133 | 220 | `220x2` | 0.763636 | pass |

## Temporal Mapping

The Holosoma converter uses a 50Hz output grid over `duration = (input_frames - 1) / input_fps` and samples times with `arange(0, duration, output_dt)`. E138 maps each bool contact output frame to the nearest source frame on that same converter grid:

`converter_time_grid_nearest_source_frame`

The manifest records expected output frames, converted output frames, source index min/max, and a SHA256 over the source index vector. All rows had `time_grid_match=true`.

This intentionally does not pad to include the terminal source frame when the converter output grid stops before it.

## Claims Verification

| Claim | Result |
|---|---|
| C1: E137 trimmed semantic files can be converted into Holosoma WBT format | pass |
| C2: `object_contact` is post-injected after conversion with explicit temporal mapping | pass |
| C3: injected `object_contact.shape == (joint_pos_frames, 2)` | pass |
| C4: CPU Holosoma `MotionLoader` reports `has_object_contact=True` | pass |
| C5: qpos-style E137 files are not claimed runtime-ready directly | pass |
| C6: no CEM/PPO/training/remote launch | pass |

## Verification

- `.venv/bin/python -m py_compile workspace/core4d/scripts/E138/convert_e137_semantic_contact_to_holosoma.py`
- `bash -n workspace/core4d/scripts/eval/eval_E138_e137_semantic_contact_converter_preflight.sh`
- `bash workspace/core4d/scripts/eval/eval_E138_e137_semantic_contact_converter_preflight.sh`

## Notes

E138 is a runtime-format bridge, not an RL handoff. The converted files are structurally loadable by `MotionLoader` with semantic `object_contact`, but `rl_ready_rows=0` remains correct because no bounded Holosoma env `ref_object_contact` probe, PPO smoke, or source-row release gate was run.
