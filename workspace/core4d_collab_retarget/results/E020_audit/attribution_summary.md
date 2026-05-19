# E020 Attribution Summary

## Root-Cause Counts

| root_cause | count |
|---|---:|
| `algo_contact` | 4 |
| `algo_stability` | 4 |
| `contact_mask` | 1 |
| `pass` | 1 |
| `raw_data` | 1 |
| `retarget_kinematic` | 2 |

## Per-Case Attribution

| Case | E018b diag | S1 | S2 | S3 | S4 | root_cause | next |
|---|---|---|---|---|---|---|---|
| `box021_p1` | `robot_fall_visual_fail` | False | True | False | False | `algo_stability` | `E022_stability_leg_collision` |
| `box021_p2` | `robot_fall_visual_fail` | False | True | False | False | `algo_stability` | `E022_stability_leg_collision` |
| `box023_p1` | `contact_preservation_gap` | False | True | False | False | `contact_mask` | `E021_per_eef_mask_repair` |
| `box023_p2` | `contact_preservation_gap` | False | True | False | False | `algo_contact` | `E023_robot_side_contact_closure` |
| `box025_p1` | `contact_preservation_gap` | False | False | False | False | `retarget_kinematic` | `E021_ref_geometry_repair` |
| `box025_p2` | `paper_generalization_pass` | False | True | False | True | `pass` | `none` |
| `bucket001_p1` | `robot_fall_visual_fail` | True | True | False | False | `algo_stability` | `E022_stability_leg_collision` |
| `bucket001_p2` | `robot_fall_visual_fail` | True | True | False | False | `algo_stability` | `E022_stability_leg_collision` |
| `bucket005_s2_p1` | `push_or_leg_shortcut` | False | True | False | False | `algo_contact` | `E023_robot_side_contact_closure` |
| `bucket005_s2_p2` | `artifact_failed` | False | True | False | False | `algo_contact` | `E023_robot_side_contact_closure` |
| `bucket007_p1` | `artifact_failed` | False | True | False | False | `algo_contact` | `E023_robot_side_contact_closure` |
| `bucket007_p2` | `contact_preservation_gap` | True | False | False | False | `retarget_kinematic` | `E021_ref_geometry_repair` |
| `desk021_p1` | `contact_preservation_gap` | False | True | False | False | `raw_data` | `E024_multi_agent_or_data_filter` |

## Actionable Next Experiments

- `E021_per_eef_mask_repair` / `E021_ref_geometry_repair`: repair current all-on `trajectory_kinematic.contact` and high-leg-interference kinematic refs before another 13-case sweep.
- `E022_stability_leg_collision`: add robot-side fall, pelvis, and leg/object collision gates to the CEM objective for `box021*` and `bucket001*` fall cases.
- `E023_robot_side_contact_closure`: target cases with good object tracking but bad contact/artifact metrics using per-hand contact shaping and collision penalties.
- `E024_multi_agent_or_data_filter`: separate cases where raw partner support dominates or single-G1 physics is under-specified (`desk021`, partner-heavy buckets).

## Protocol Notes

S1 recomputes raw SMPL-X hand/object contact centroids from CORE4D raw files and compares them to the E018b canonical support anchor after axis-wise mesh-frame scaling.
S2 uses the saved kinematic reference and E018b ref geometry metrics; S3 compares the processed SPIDER contact field with raw 3cm per-EEF masks; S4 overlays saved sim/ref rollout metrics.
