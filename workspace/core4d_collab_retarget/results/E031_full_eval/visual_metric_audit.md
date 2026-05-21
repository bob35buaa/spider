# E031 可视化 / 指标一致性审计

E031 复用 E026-E030 已归档的视频和关键帧，记录每个诊断 variant 为什么不能作为正向候选。

## 保守 Best 选择

| Case | 选中 variant | 指标信号 | 可视化证据路径 |
|---|---|---|---|
| box021_p1 | `E018b_box021_p1_canonical_t02` | 接触=71.74%, 深穿透=53.10%, 摔倒=是, strict=否 | `workspace/core4d_collab_retarget/results/E018b/online_video/E018b_box021_p1_canonical_t02.mp4` |
| box021_p2 | `E018b_box021_p2_canonical_t02` | 接触=10.30%, 深穿透=0.74%, 摔倒=是, strict=否 | `workspace/core4d_collab_retarget/results/E018b/online_video/E018b_box021_p2_canonical_t02.mp4` |
| box023_p1 | `E022_box023_p1_raw3_eval_axis` | 接触=25.30%, 深穿透=0.00%, 摔倒=否, strict=否 | `workspace/core4d_collab_retarget/results/E022/online_video/E022_box023_p1_raw3_eval_axis.mp4` |
| box023_p2 | `E018b_box023_p2_canonical_t02` | 接触=28.57%, 深穿透=3.33%, 摔倒=否, strict=否 | `workspace/core4d_collab_retarget/results/E018b/online_video/E018b_box023_p2_canonical_t02.mp4` |
| box025_p1 | `E018b_box025_p1_canonical_t02` | 接触=66.07%, 深穿透=5.06%, 摔倒=否, strict=否 | `workspace/core4d_collab_retarget/results/E018b/online_video/E018b_box025_p1_canonical_t02.mp4` |
| box025_p2 | `E018b_box025_p2_canonical_t02` | 接触=86.93%, 深穿透=0.00%, 摔倒=否, strict=是 | `workspace/core4d_collab_retarget/results/E018b/online_video/E018b_box025_p2_canonical_t02.mp4` |
| bucket001_p1 | `E024_bucket001_p1_root025_gain2_stab_t065` | 接触=0.00%, 深穿透=0.00%, 摔倒=是, strict=否 | `workspace/core4d_collab_retarget/results/E024/online_video/E024_bucket001_p1_root025_gain2_stab_t065.mp4` |
| bucket001_p2 | `E024_bucket001_p2_root025_gain2_stab_t065` | 接触=77.53%, 深穿透=59.60%, 摔倒=否, strict=否 | `workspace/core4d_collab_retarget/results/E024/online_video/E024_bucket001_p2_root025_gain2_stab_t065.mp4` |
| bucket005_s2_p1 | `E018b_bucket005_s2_p1_canonical_t02` | 接触=97.59%, 深穿透=88.15%, 摔倒=否, strict=否 | `workspace/core4d_collab_retarget/results/E018b/online_video/E018b_bucket005_s2_p1_canonical_t02.mp4` |
| bucket005_s2_p2 | `E025_bucket005_s2_p2_penalty_s4_hc1` | 接触=96.42%, 深穿透=64.53%, 摔倒=否, strict=否 | `workspace/core4d_collab_retarget/results/E025/online_video/E025_bucket005_s2_p2_penalty_s4_hc1.mp4` |
| bucket007_p1 | `E025_bucket007_p1_penalty_s4_hc1` | 接触=84.13%, 深穿透=35.57%, 摔倒=否, strict=否 | `workspace/core4d_collab_retarget/results/E025/online_video/E025_bucket007_p1_penalty_s4_hc1.mp4` |
| bucket007_p2 | `E018b_bucket007_p2_canonical_t02` | 接触=29.75%, 深穿透=18.42%, 摔倒=否, strict=否 | `workspace/core4d_collab_retarget/results/E018b/online_video/E018b_bucket007_p2_canonical_t02.mp4` |
| desk021_p1 | `E018b_desk021_p1_canonical_t02` | 接触=52.03%, 深穿透=14.29%, 摔倒=否, strict=否 | `workspace/core4d_collab_retarget/results/E018b/online_video/E018b_desk021_p1_canonical_t02.mp4` |

## 被拒绝的诊断候选

| 方法 | Case | Variant | 拒绝原因 | 可视化/关键帧目录 |
|---|---|---|---|---|
| spider_E028 | bucket007_p1 | `E028_bucket007_p1_barrier_quad_m02` | 摔倒; 接触过低/坍缩=1.11%; E028 hard barrier 仅作诊断，不进入正向池 | `workspace/core4d_collab_retarget/results/E028/keyframes/E028_bucket007_p1_barrier_quad_m02/` |
| spider_E028 | bucket005_s2_p2 | `E028_bucket005_s2_p2_barrier_quad_m02` | 深穿透过高=88.18%; 最大穿透超过 5cm=8.78cm; E028 hard barrier 仅作诊断，不进入正向池 | `workspace/core4d_collab_retarget/results/E028/keyframes/E028_bucket005_s2_p2_barrier_quad_m02/` |
| spider_E028 | bucket005_s2_p1 | `E028_bucket005_s2_p1_contact_gate_m02` | 深穿透过高=92.89%; 最大穿透超过 5cm=6.33cm; E028 hard barrier 仅作诊断，不进入正向池 | `workspace/core4d_collab_retarget/results/E028/keyframes/E028_bucket005_s2_p1_contact_gate_m02/` |
| spider_E028 | bucket001_p2 | `E028_bucket001_p2_contact_gate_m02` | 接触过低/坍缩=2.81%; E028 hard barrier 仅作诊断，不进入正向池 | `workspace/core4d_collab_retarget/results/E028/keyframes/E028_bucket001_p2_contact_gate_m02/` |
| spider_E028 | bucket007_p1 | `E028_bucket007_p1_scorecap_m01` | 深穿透过高=67.79%; 最大穿透超过 5cm=10.58cm; E028 hard barrier 仅作诊断，不进入正向池 | `workspace/core4d_collab_retarget/results/E028/keyframes/E028_bucket007_p1_scorecap_m01/` |
| spider_E028 | box025_p2 | `E028_box025_p2_guard_barrier_m02` | 接触过低/坍缩=0.00%; E028 hard barrier 仅作诊断，不进入正向池 | `workspace/core4d_collab_retarget/results/E028/keyframes/E028_box025_p2_guard_barrier_m02/` |
| spider_E029 | bucket001_p1 | `E029_bucket001_p1_upright_barrier_t055` | 接触过低/坍缩=0.00%; 只修稳定性，接触仍为 0 | `workspace/core4d_collab_retarget/results/E029/keyframes/E029_bucket001_p1_upright_barrier_t055/` |
| spider_E029 | bucket001_p1 | `E029_bucket001_p1_posture_gate_t055` | 接触过低/坍缩=0.00%; 只修稳定性，接触仍为 0 | `workspace/core4d_collab_retarget/results/E029/keyframes/E029_bucket001_p1_posture_gate_t055/` |
| spider_E029 | bucket001_p1 | `E029_bucket001_p1_scorecap_t045` | 接触过低/坍缩=0.00%; 只修稳定性，接触仍为 0 | `workspace/core4d_collab_retarget/results/E029/keyframes/E029_bucket001_p1_scorecap_t045/` |
| spider_E029 | bucket001_p1 | `E029_bucket001_p1_tilt_gate_t055` | 接触过低/坍缩=0.00%; 只修稳定性，接触仍为 0 | `workspace/core4d_collab_retarget/results/E029/keyframes/E029_bucket001_p1_tilt_gate_t055/` |
| spider_E029 | bucket001_p2 | `E029_bucket001_p2_guard_posture_gate` | 深穿透过高=63.64%; 最大穿透超过 5cm=9.00cm; E029 stability 仅作诊断，不进入正向池 | `workspace/core4d_collab_retarget/results/E029/keyframes/E029_bucket001_p2_guard_posture_gate/` |
| spider_E029 | box025_p2 | `E029_box025_p2_guard_posture_gate` | 仅是回归 guard，保留 E018b strict baseline | `workspace/core4d_collab_retarget/results/E029/keyframes/E029_box025_p2_guard_posture_gate/` |
| spider_E030 | box025_p1 | `E030_box025_p1_tinygeom_surface_gate` | 摔倒; 接触过低/坍缩=2.50%; E030 geometry/surface 负结果，不进入正向池 | `workspace/core4d_collab_retarget/results/E030/keyframes/E030_box025_p1_tinygeom_surface_gate/` |
| spider_E030 | bucket007_p2 | `E030_bucket007_p2_tinygeom_surface_gate` | E030 geometry/surface 负结果，不进入正向池 | `workspace/core4d_collab_retarget/results/E030/keyframes/E030_bucket007_p2_tinygeom_surface_gate/` |
| spider_E030 | box023_p1 | `E030_box023_p1_surface_hold_gate` | 接触过低/坍缩=29.32%; E030 geometry/surface 负结果，不进入正向池 | `workspace/core4d_collab_retarget/results/E030/keyframes/E030_box023_p1_surface_hold_gate/` |
| spider_E030 | box023_p2 | `E030_box023_p2_surface_hold_gate` | 摔倒; 接触过低/坍缩=8.33%; E030 geometry/surface 负结果，不进入正向池 | `workspace/core4d_collab_retarget/results/E030/keyframes/E030_box023_p2_surface_hold_gate/` |
| spider_E030 | bucket005_s2_p1 | `E030_bucket005_s2_p1_leg_guard_surface` | 深穿透过高=94.31%; 最大穿透超过 5cm=5.69cm; E030 geometry/surface 负结果，不进入正向池 | `workspace/core4d_collab_retarget/results/E030/keyframes/E030_bucket005_s2_p1_leg_guard_surface/` |
| spider_E030 | box025_p2 | `E030_box025_p2_guard_surface` | 接触过低/坍缩=0.00%; E030 geometry/surface 负结果，不进入正向池 | `workspace/core4d_collab_retarget/results/E030/keyframes/E030_box025_p2_guard_surface/` |

Guard 规则：高接触但高穿透、以及接触坍缩的行都只能作为诊断负例；即使 object tracking 仍然好，也不能算成功。
