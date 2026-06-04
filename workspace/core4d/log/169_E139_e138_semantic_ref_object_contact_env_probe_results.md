# E139 E138 Semantic Ref-Object-Contact Env Probe Results

## Summary

E139 verified that E138 semantic `object_contact` masks can be exposed through Holosoma runtime `motion_command.ref_object_contact` under the existing R135 Box021 partner env.

Result: pass.

- Partner-injected semantic motions: 4/4 pass.
- Bounded no-debug env startup rows: 4/4 pass.
- Runtime `has_partner=true` rows: 4/4.
- Runtime `has_object_contact=true` rows: 4/4.
- Runtime `ref_object_contact` positive rows: 4/4.
- RL-ready rows: 0.
- Training launched: false.
- CEM launched: false.
- Remote jobs launched: false.

## Result Paths

| artifact | path |
|---|---|
| partner injection manifest | `workspace/core4d/results/E139/e138_semantic_ref_object_contact_env_probe/e139_partner_injection_manifest.tsv` |
| runtime manifest | `workspace/core4d/results/E139/e138_semantic_ref_object_contact_env_probe/e139_ref_object_contact_manifest.tsv` |
| summary json | `workspace/core4d/results/E139/e138_semantic_ref_object_contact_env_probe/e139_ref_object_contact_summary.json` |
| summary md | `workspace/core4d/results/E139/e138_semantic_ref_object_contact_env_probe/e139_ref_object_contact_summary.md` |
| partner-injected motions | `workspace/core4d/results/E139/e138_semantic_ref_object_contact_env_probe/partner_semantic_motions/` |
| runtime logs | `workspace/core4d/results/E139/e138_semantic_ref_object_contact_env_probe/logs/` |

## Partner Injection

E139 did not modify E138 outputs in place. It wrote separate partner-injected copies under `partner_semantic_motions/`.

Policy:

`nearest_partner_converted_frame_by_raw_contact_frame`

For each converted base frame, raw frame was reconstructed as:

`E137 object_contact_trim_start + E138 object_contact_source_frame_index`

Partner wrist poses were sampled from the paired converted E138 motion by nearest raw-frame index. The manifest records raw-frame ranges and nearest-frame diffs. Two rows had exact raw-frame alignment; reverse rows had bounded differences caused by non-identical p1/p2 trim windows:

| case | frames | partner frames | nearest diff mean | nearest diff max |
|---|---:|---:|---:|---:|
| `box021_029_p1` | 117 | 124 | 0.000000 | 0 |
| `box021_029_p2` | 124 | 117 | 0.120968 | 4 |
| `box021_035_p1` | 210 | 220 | 0.000000 | 0 |
| `box021_035_p2` | 220 | 210 | 0.150000 | 6 |

All partner fields passed shape and finite checks:

- `partner_hand_pos_w`: `(T, 2, 3)`
- `partner_hand_quat_w`: `(T, 2, 4)`

## Runtime Rows

| case | startup | has partner | has object contact | motion total | ref total | ref either | observed step max |
|---|---|---|---|---:|---:|---:|---:|
| `box021_029_p1` | pass | true | true | 154 | 154 | 80 | 204 |
| `box021_029_p2` | pass | true | true | 177 | 151 | 79 | 204 |
| `box021_035_p1` | pass | true | true | 329 | 153 | 79 | 204 |
| `box021_035_p2` | pass | true | true | 335 | 154 | 77 | 204 |

Holosoma prepended/appended default-pose transition segments, so runtime `motion_shape` is larger than the converted E138 clip frame count. The semantic mask remains visible during bounded stepping, and the sampled `ref_object_contact` totals are nonzero for all rows.

## Claims Verification

| Claim | Result |
|---|---|
| C1: partner-injected E138 semantic files start under R135 Box021 partner env | pass |
| C2: runtime `has_object_contact=true` for every row | pass |
| C3: runtime `has_partner=true` for every row | pass |
| C4: bounded stepping samples nonzero `ref_object_contact` | pass |
| C5: no CEM/PPO/training/remote launch | pass |

## Verification

- `.venv/bin/python -m py_compile workspace/core4d/scripts/E139/build_e138_partner_semantic_motions.py workspace/core4d/scripts/E139/holosoma_semantic_ref_object_contact_no_debug_probe.py`
- `bash -n workspace/core4d/scripts/eval/eval_E139_e138_semantic_ref_object_contact_env_probe.sh`
- `bash workspace/core4d/scripts/eval/eval_E139_e138_semantic_ref_object_contact_env_probe.sh`

## Notes

E139 proves runtime plumbing for semantic `object_contact` masks in a partner-enabled Holosoma env. It does not prove reward effectiveness, PPO readiness, source-row release readiness, or main-case strict success.
