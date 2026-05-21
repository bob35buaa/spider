# E031 Visual / Metric Audit

E031 reuses existing videos/keyframes from E026-E030 and records why diagnostic variants are not positive candidates.

## Conservative Best Selection

| Case | Selected variant | Signal | Visual evidence path |
|---|---|---|---|
| box021_p1 | `E018b_box021_p1_canonical_t02` | contact=71.74%, deep=53.10%, fall=True, strict=False | `workspace/core4d_collab_retarget/results/E018b/online_video/E018b_box021_p1_canonical_t02.mp4` |
| box021_p2 | `E018b_box021_p2_canonical_t02` | contact=10.30%, deep=0.74%, fall=True, strict=False | `workspace/core4d_collab_retarget/results/E018b/online_video/E018b_box021_p2_canonical_t02.mp4` |
| box023_p1 | `E022_box023_p1_raw3_eval_axis` | contact=25.30%, deep=0.00%, fall=False, strict=False | `workspace/core4d_collab_retarget/results/E022/online_video/E022_box023_p1_raw3_eval_axis.mp4` |
| box023_p2 | `E018b_box023_p2_canonical_t02` | contact=28.57%, deep=3.33%, fall=False, strict=False | `workspace/core4d_collab_retarget/results/E018b/online_video/E018b_box023_p2_canonical_t02.mp4` |
| box025_p1 | `E018b_box025_p1_canonical_t02` | contact=66.07%, deep=5.06%, fall=False, strict=False | `workspace/core4d_collab_retarget/results/E018b/online_video/E018b_box025_p1_canonical_t02.mp4` |
| box025_p2 | `E018b_box025_p2_canonical_t02` | contact=86.93%, deep=0.00%, fall=False, strict=True | `workspace/core4d_collab_retarget/results/E018b/online_video/E018b_box025_p2_canonical_t02.mp4` |
| bucket001_p1 | `E024_bucket001_p1_root025_gain2_stab_t065` | contact=0.00%, deep=0.00%, fall=True, strict=False | `workspace/core4d_collab_retarget/results/E024/online_video/E024_bucket001_p1_root025_gain2_stab_t065.mp4` |
| bucket001_p2 | `E024_bucket001_p2_root025_gain2_stab_t065` | contact=77.53%, deep=59.60%, fall=False, strict=False | `workspace/core4d_collab_retarget/results/E024/online_video/E024_bucket001_p2_root025_gain2_stab_t065.mp4` |
| bucket005_s2_p1 | `E018b_bucket005_s2_p1_canonical_t02` | contact=97.59%, deep=88.15%, fall=False, strict=False | `workspace/core4d_collab_retarget/results/E018b/online_video/E018b_bucket005_s2_p1_canonical_t02.mp4` |
| bucket005_s2_p2 | `E025_bucket005_s2_p2_penalty_s4_hc1` | contact=96.42%, deep=64.53%, fall=False, strict=False | `workspace/core4d_collab_retarget/results/E025/online_video/E025_bucket005_s2_p2_penalty_s4_hc1.mp4` |
| bucket007_p1 | `E025_bucket007_p1_penalty_s4_hc1` | contact=84.13%, deep=35.57%, fall=False, strict=False | `workspace/core4d_collab_retarget/results/E025/online_video/E025_bucket007_p1_penalty_s4_hc1.mp4` |
| bucket007_p2 | `E018b_bucket007_p2_canonical_t02` | contact=29.75%, deep=18.42%, fall=False, strict=False | `workspace/core4d_collab_retarget/results/E018b/online_video/E018b_bucket007_p2_canonical_t02.mp4` |
| desk021_p1 | `E018b_desk021_p1_canonical_t02` | contact=52.03%, deep=14.29%, fall=False, strict=False | `workspace/core4d_collab_retarget/results/E018b/online_video/E018b_desk021_p1_canonical_t02.mp4` |

## Rejected Diagnostics

| Method | Case | Variant | Reject reason | Visual/keyframe root |
|---|---|---|---|---|
| spider_E028 | bucket007_p1 | `E028_bucket007_p1_barrier_quad_m02` | fall; low_or_collapsed_contact=1.11%; E028_hard_barrier_diagnostic_not_positive_pool | `workspace/core4d_collab_retarget/results/E028/keyframes/E028_bucket007_p1_barrier_quad_m02/` |
| spider_E028 | bucket005_s2_p2 | `E028_bucket005_s2_p2_barrier_quad_m02` | high_deep_pen=88.18%; max_pen_over_5cm=8.78cm; E028_hard_barrier_diagnostic_not_positive_pool | `workspace/core4d_collab_retarget/results/E028/keyframes/E028_bucket005_s2_p2_barrier_quad_m02/` |
| spider_E028 | bucket005_s2_p1 | `E028_bucket005_s2_p1_contact_gate_m02` | high_deep_pen=92.89%; max_pen_over_5cm=6.33cm; E028_hard_barrier_diagnostic_not_positive_pool | `workspace/core4d_collab_retarget/results/E028/keyframes/E028_bucket005_s2_p1_contact_gate_m02/` |
| spider_E028 | bucket001_p2 | `E028_bucket001_p2_contact_gate_m02` | low_or_collapsed_contact=2.81%; E028_hard_barrier_diagnostic_not_positive_pool | `workspace/core4d_collab_retarget/results/E028/keyframes/E028_bucket001_p2_contact_gate_m02/` |
| spider_E028 | bucket007_p1 | `E028_bucket007_p1_scorecap_m01` | high_deep_pen=67.79%; max_pen_over_5cm=10.58cm; E028_hard_barrier_diagnostic_not_positive_pool | `workspace/core4d_collab_retarget/results/E028/keyframes/E028_bucket007_p1_scorecap_m01/` |
| spider_E028 | box025_p2 | `E028_box025_p2_guard_barrier_m02` | low_or_collapsed_contact=0.00%; E028_hard_barrier_diagnostic_not_positive_pool | `workspace/core4d_collab_retarget/results/E028/keyframes/E028_box025_p2_guard_barrier_m02/` |
| spider_E029 | bucket001_p1 | `E029_bucket001_p1_upright_barrier_t055` | low_or_collapsed_contact=0.00%; stability_only_contact_remains_zero | `workspace/core4d_collab_retarget/results/E029/keyframes/E029_bucket001_p1_upright_barrier_t055/` |
| spider_E029 | bucket001_p1 | `E029_bucket001_p1_posture_gate_t055` | low_or_collapsed_contact=0.00%; stability_only_contact_remains_zero | `workspace/core4d_collab_retarget/results/E029/keyframes/E029_bucket001_p1_posture_gate_t055/` |
| spider_E029 | bucket001_p1 | `E029_bucket001_p1_scorecap_t045` | low_or_collapsed_contact=0.00%; stability_only_contact_remains_zero | `workspace/core4d_collab_retarget/results/E029/keyframes/E029_bucket001_p1_scorecap_t045/` |
| spider_E029 | bucket001_p1 | `E029_bucket001_p1_tilt_gate_t055` | low_or_collapsed_contact=0.00%; stability_only_contact_remains_zero | `workspace/core4d_collab_retarget/results/E029/keyframes/E029_bucket001_p1_tilt_gate_t055/` |
| spider_E029 | bucket001_p2 | `E029_bucket001_p2_guard_posture_gate` | high_deep_pen=63.64%; max_pen_over_5cm=9.00cm; E029_stability_diagnostic_not_positive_pool | `workspace/core4d_collab_retarget/results/E029/keyframes/E029_bucket001_p2_guard_posture_gate/` |
| spider_E029 | box025_p2 | `E029_box025_p2_guard_posture_gate` | non_regression_guard_only_E018b_strict_baseline_preferred | `workspace/core4d_collab_retarget/results/E029/keyframes/E029_box025_p2_guard_posture_gate/` |
| spider_E030 | box025_p1 | `E030_box025_p1_tinygeom_surface_gate` | fall; low_or_collapsed_contact=2.50%; E030_geometry_surface_negative_result_not_positive_pool | `workspace/core4d_collab_retarget/results/E030/keyframes/E030_box025_p1_tinygeom_surface_gate/` |
| spider_E030 | bucket007_p2 | `E030_bucket007_p2_tinygeom_surface_gate` | E030_geometry_surface_negative_result_not_positive_pool | `workspace/core4d_collab_retarget/results/E030/keyframes/E030_bucket007_p2_tinygeom_surface_gate/` |
| spider_E030 | box023_p1 | `E030_box023_p1_surface_hold_gate` | low_or_collapsed_contact=29.32%; E030_geometry_surface_negative_result_not_positive_pool | `workspace/core4d_collab_retarget/results/E030/keyframes/E030_box023_p1_surface_hold_gate/` |
| spider_E030 | box023_p2 | `E030_box023_p2_surface_hold_gate` | fall; low_or_collapsed_contact=8.33%; E030_geometry_surface_negative_result_not_positive_pool | `workspace/core4d_collab_retarget/results/E030/keyframes/E030_box023_p2_surface_hold_gate/` |
| spider_E030 | bucket005_s2_p1 | `E030_bucket005_s2_p1_leg_guard_surface` | high_deep_pen=94.31%; max_pen_over_5cm=5.69cm; E030_geometry_surface_negative_result_not_positive_pool | `workspace/core4d_collab_retarget/results/E030/keyframes/E030_bucket005_s2_p1_leg_guard_surface/` |
| spider_E030 | box025_p2 | `E030_box025_p2_guard_surface` | low_or_collapsed_contact=0.00%; E030_geometry_surface_negative_result_not_positive_pool | `workspace/core4d_collab_retarget/results/E030/keyframes/E030_box025_p2_guard_surface/` |

Guard rule: high-contact high-penetration rows and contact-collapse rows are diagnostic negatives, even when object tracking remains good.
