# OmniRetarget 硬骨头 — 修复 retargeting 层的具体角度

日期：2026-05-30
作者：研究分析（未修改任何代码）
范围：假设我们**不**通过上游数据筛选绕过 OmniRetarget，那么对于 box021 / Box026 / 中等尺寸 box 类别，修复 OmniRetarget 使其为 SPIDER 输出 G1-feasible reference 的具体角度有哪些。

本次分析实际阅读的来源（全部为绝对路径）：

- `/mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/spider/workspace/exp_diagnostic/diagnostic_report.md`
- `/mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/spider/workspace/exp_diagnostic/findings/08_B_path_omniretarget_audit.md`
- `/mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/spider/workspace/core4d/log/111_E089_g1_feasibility_AB_results.md`
- `/mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/spider/workspace/core4d/log/112_E090_h2_first_retarget_and_spider_smoke_results.md`
- `/mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/spider/workspace/core4d/log/115_E093_g1_handbox_target_projection_results.md`
- `/mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/spider/workspace/core4d/log/116_E094_g1_handbox_target_projection_results.md`
- `/mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/spider/workspace/core4d/log/120_E097_feature_based_data_construction_v2_results.md`
- `/mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/spider/workspace/core4d/log/121_E097_visual_review_and_candidate_correction.md`
- `/mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/holosoma/workspace/pipeline/convert_core4d_to_omniretarget.py`
- `/mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/holosoma/workspace/pipeline/retarget_core4d_obj_interaction.py`
- `/mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/holosoma/workspace/pipeline/retarget_py_workflow.md`
- `/mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/holosoma/workspace/pipeline/README.md`
- `/mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/holosoma/src/holosoma_retargeting/holosoma_retargeting/src/interaction_mesh_retargeter.py`
  （init L46-194、retarget_motion L387-595、solve_single_iteration L597+、Phase-4 contact L483-520）
- `/mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/holosoma/src/holosoma_retargeting/holosoma_retargeting/src/utils.py`
  （preprocess_motion_data L219-264、augment_object_poses L301-340）
- `/mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/holosoma/src/holosoma_retargeting/holosoma_retargeting/config_types/data_type.py`
  （SMPLX_DEMO_JOINTS L151、JOINTS_MAPPINGS L185+、core4d_v2/g1 mapping L273-291）

---

## 第 1 节 — OmniRetarget pipeline 现状

```
CORE4D Y-up SMPL-X mocap（person1、person2、object 每帧 4x4 transform）
        │
        │  STAGE A: convert_core4d_to_omniretarget.py
        │            (workspace/pipeline/convert_core4d_to_omniretarget.py)
        ▼
{date}-{seq}-{person}-{Box}_with_obj.npz
   keys:  global_joint_positions (T,22,3) Z-up float32   ← rows 0-21 = SMPL-X body
          height (scalar)                                  ← 来自 betas T-pose
          object_poses (T,7) [qw,qx,qy,qz,x,y,z] Z-up    ← 来自 smooth_objposes.npy
          obj_name (str)
   可选 flags（D003 production 当前 OFF）：
       --replace_wrist_with_fingertip   用 5 指尖均值覆盖 rows 20/21
       --include_fingertip_centers      追加 rows 22/23（仅 core4d_v2 格式）
        │
        │  STAGE B: examples/robot_retarget.py + InteractionMeshRetargeter
        │            (src/holosoma_retargeting/holosoma_retargeting/src/interaction_mesh_retargeter.py)
        ▼
{task}_original.npz   keys: qpos (T,43) free_obj+G1, human_joints, fps=30, cost
   - smpl_scale = ROBOT_HEIGHT(1.32) / human_height       (utils.preprocess_motion_data L223,251)
   - 22 个 joint 全部 + object_poses[:, x,y] 按 smpl_scale 缩放（z 仅做 floor 重定位）
   - JOINTS_MAPPING (core4d_v2/g1) 把 L_Wrist→left_wrist_yaw_link，
     L_Fingertip_Center→left_rubber_hand_link（data_type.py L287-290）
   - 每帧 DiffIK SQP via cvxpy：object frame 下的 Laplacian 形变是 COST；
     地面/object 不穿模 + foot-stick + joint-limit + trust region 是 CONSTRAINTS。
     interaction_mesh_retargeter.iterate (L506-520)
   - object 轨迹被锁定到 q[-7:]（`q_locked_list[:, -7:] = object_poses_augmented`，
     interaction_mesh_retargeter.py L423）——只对 robot 求解。
   - 可选 `enable_contact_preservation`（Phase 4c，L483-520）用 wrist→object-center 距离，
     D003 production 中 OFF（E089 §3 / 08_B_path_omniretarget_audit §6）。
        │
        │  STAGE C: workspace/pipeline/trim_no_contact.py
        ▼
   裁剪起始无接触帧（IGL signed distance hand→mesh，contact 阈值 0.05 m，margin 0.5 s）
        │
        │  STAGE D (Spider side): workspace/core4d/data_preprocess/create_spider_scene_from_template.py
        ▼
   example_datasets/processed/core4d/unitree_g1/humanoid_object/{task}/scene.xml
        object collision-box half-extents = mesh AABB * smpl_scale
        │
        │  STAGE E: spider/process_datasets/core4d.py
        ▼
   {task}/0/trajectory_kinematic.npz   ← qpos_ref, contact_pos, contact_mask, eef_pos, …
        │
        ▼  SPIDER MJWP sampling-based MPC（消费方）
```

下文的关键事实：在 stage B 中驱动 G1 wrist 的**唯一**信号是
`global_joint_positions[i, 20:21, :]`（启用时还包括 rows 22/23 fingertip centers）。
在默认 D003 production 配置下，**没有**单独的 "contact target" 通道，也**没有** object-collision-aware
的 cost 项 —— Laplacian solver 把 wrist 当作另一个普通的 mesh 顶点去 track。（已通过 retarget_motion
L440-451 + iterate L506-520 + Phase-4 flag OFF 确认。）

**STAGE A 的语义致命缺陷（关键背景，详见 §3 Angle 8）**：CORE4D 原始 mocap 每只手有 5 个指尖；
SMPL-X 22-joint 标准只保留 L_Wrist/R_Wrist，5 指尖直接被丢；除非启用 core4d_v2 +
`--include_fingertip_centers`（rows 22/23 = 5 指尖均值）才补回 **1 个**均值点 —— 而 D003 production
该 flag 是 OFF；JOINTS_MAPPINGS（data_type.py:287-289）又把 L_Fingertip_Center 映射到
**无 DoF** 的 left_rubber_hand_link 球。结果：5 指尖 / palm / 弯指全部坍缩成"球贴一点"。
spider 侧 `process_datasets/core4d.py:128` 写 `contact_pos = palm site FK`，整条 pipeline
**没有任何位置真正代表"指尖在哪"**。E093 量化证据：`wrist + 5 cm offset → raw mean` ≥ 20 cm 全 case，
Box026 / Box025 达 49-64 cm（详见 `115_E093_…md` §"关键指标"）。

---

## 第 2 节 — OmniRetarget 注入每种失败的具体位置

| 失败模式（下游观测） | 注入 stage | 机制 | 一行证据 |
|---|---|---|---|
| **STAGE A wrist 抽样代替 contact（5 指尖丢失）** | STAGE A（数据通道） | SMPL-X 22-joint 只保留 wrist、5 指尖被丢；G1 rubber_hand 无 DoF；整条 pipeline 没有任何位置代表"指尖在哪"；spider 端 `contact_pos = palm site FK` 是 wrist 的 5 cm offset，不是真接触点 | E093 wrist+5 cm vs raw mean ≥ 20 cm 全 case；Box026/Box025 49-64 cm |
| Hand-in-box（R wrist 33% INSIDE） | STAGE B（wrist 无 obj-collision cost） | Laplacian 只最小化 mesh 形变；若 source wrist 落在 object frame 内部，solver 乐于把 FK 留在那里 | diagnostic_report §3.3 box021_18029_p2 R-wrist 33% INSIDE（**注**：v1 §3.3 报告的这一现象，至少一部分是 STAGE A wrist 内陷的几何假象，见 §3 Angle 8） |
| Wrist target 低于 pelvis（-25 cm 缺口） | STAGE A→B（mocap wrist 本来就在 pelvis 下方；smpl_scale 按比例缩小缺口，但不修正符号） | smpl_scale=0.73 把整个人按相同比例缩放；"person2 蹲在 box 下"仍然映射为"robot 蹲在 box 下" | diagnostic_report §3.4 hand z 0.47 m vs pelvis 0.58–0.72 m |
| 选错 contact face（side / -z 而非 +z top） | STAGE A（无 face 监督）+ STAGE B（无 face-side cost）；**对于 quat≠identity 的 object 还会被 world-up 假设进一步带偏** | wrist 是无表面法向的单点；选中的 face 取决于 raw mocap 手指刚好落在哪里；当 box quat 偏离 identity（如 box021 的 90° around +X），world-up 与 object local +z 解耦，原本"投到 world-up face"的启发式会指错方向 | E094 投影表：D003 box021 old support 9/77% → patched 77/85%；box021 18029_p2 obj quat 90° around +X，local +z 实际朝水平方向（v1 §3.2） |
| Pelvis 太低（坍缩到 <0.55 m） | STAGE A（mocap 是从下方蹲举）+ STAGE B（无 pelvis-height floor cost） | Laplacian 均匀地 track Pelvis→pelvis_contour_link；没有任何项阻止 pelvis 穿地板或低于站姿 | E090 §3 S1 full pelvis_min=0.134；E094 §6 C2 pelvis 0.440 但 object tracking 0.002 m |
| 身体部位比例不匹配（G1 手臂相对 SMPL-X 偏短，但 smpl_scale 仅按 HEIGHT 选） | STAGE B init | `smpl_scale = ROBOT_HEIGHT/human_height`（utils.py L223,251）—— 单一全局标量；G1 vs SMPL-X 的臂展比并**不等于**身高比 | diagnostic §3.4 + log 76/77 "box025 真问题是臂展物理硬限制" |
| object collision **未**用于 wrist target | STAGE B | `activate_obj_non_penetration` 仅作用于 robot **body** 与 object URDF 的碰撞，但 Laplacian wrist **cost** 仍拉向可能在 box 内部的 human source coord | retargeter init L52 activate_obj_non_penetration=True，但 iterate 用 `J_OC_dict` 做 cost 而非 wrist-vs-obj-SDF |
| contact target / contact mask 不一致（D003 anchor face review 在 9/13 case 上 TRUE） | 上游 D003 contact-mask 生成（Spider 侧，B 之后）—— 当 wrist coord 歧义时自动 face 选择不稳 | 若 wrist 落在两个面之间（corner / 内部），自动 face 选择会逐帧翻转 | diagnostic §5 H3，E028 9/13 anchor_face_review=true |
| Mass field 差异（29 kg vs 5 kg） | STAGE D（scene.xml 重新生成）—— 严格说不是 OmniRetarget | scene template 默认 mass 与 object 不一致 | E087 mass sweep；diagnostic §3.6 |
| object 轨迹不可行（如 box 中途翻转） | STAGE A（raw mocap object pose） | object pose 从 `smooth_objposes.npy` 原样写入；未对 robot 可达性做 plausibility check | E094 §6 C3 obj_max 0.041 m，RH floor 22%（sim 翻 box） |
| CVXPY infeasible（稀少的 hard fail） | STAGE B | hard 约束（joint limits + non-penetration + foot-stick + trust region）在某些帧互相冲突 | E090 §2 d003_box021_20231020_019_p1 nofing "CVXPY solve failed: infeasible"；E097 addendum 028_p1 d003_infeasible_cvxpy |

**不可归咎于 OmniRetarget 的失败**（列在此处避免混淆）：
- 当 wrist contact 不可能时，SPIDER reward 过度加权 body-on-box 接触（"躺在 box 上"的局部最优）。这是 SPIDER 侧 reward 问题，见 E094 §9。
- object actuator + 大 mass 初始 transient。属于 scene 侧，不属于 retargeter。

**对 v1 H1 的修正解读**：v1 §3.3 报告 box021 18029_p2 R wrist FK 在 33% 帧 INSIDE box。
新认识是：这一现象**至少一部分**是 wrist 几何内陷的假象 —— 人指尖贴 box +y 侧外侧，wrist 被反向
弯曲带到 box 几何中心方向，FK 落点看上去 INSIDE 但实际人手并未穿模。所以 H1 主张的方向对、粒度错：
真正的根本错位发生在 STAGE A（用 wrist 抽样代替 contact），IK INSIDE 是次生现象。

---

## 第 3 节 — 修复 OmniRetarget 的角度（本文真正的问题）

下文 cost 单位："engineer-day" = 在本机上专注的一天；"GPU h" = retarget 批量墙钟时间
（retargeter 本体是 CPU-bound CVXPY，但视觉评估需要 SPIDER smoke CEM，这部分要 GPU）。
每个角度都用 **touch=** 标注它会改动哪个 repo。

---

### Angle 1 — Object-SDF wrist cost / hard wrist non-penetration  (touch=OmniRetarget)

**做什么。** 在 `InteractionMeshRetargeter.iterate` 内部对每一帧、每个 wrist link
（`left_wrist_yaw_link`、`right_wrist_yaw_link`，外加 fingertip-center 映射到的 rubber-hand link）
计算与 object collision mesh 的 signed distance（mesh 已通过 `object_urdf_path` 加载），然后
(a) 在 cost 中加二次惩罚 `max(0, ε - sdf)^2`，或 (b) 在 SQP 中加线性化硬约束 `sdf + J·dq ≥ ε`。
所用 `mj_jacBody` 机制 solver 已经有。ε ≈ 2 cm。

**为什么。** 在 §2 row 1 的源头消除 "hand-in-box" 失败。今日 `activate_obj_non_penetration`
只在 body-vs-object self-collision pair 上起作用；wrist cost 拉向可能本身就在 box 内部的 human
source coord。

**Cost.** 2 engineer-days。Jacobian 路径已有；新增仅是每帧 SDF 查询（用 trimesh
`proximity.signed_distance` 或预计算离散 SDF；~10 ms/帧）。retarget 阶段无需 GPU，然后 1 GPU-h
做 E089 风格的 gate + smoke 验证。

**风险。** 硬约束可能让某些帧 CVXPY-infeasible（已经是 tail 风险，E090 S2 nofing）。
缓解：用 slack 变量 + `enable_constraint_relaxation`（已在代码中，L65-67）。

**快速证伪测试。** 在 3 个 box021 D003 case 上跑。目标：retarget 后 wrist-inside-box %
从 {0, 0, 33}% 在三个 case 上全部降为 0%，且不抬升 CVXPY infeasible 率。若在 box023_p2（guard）
上引入 ≥30% infeasible 帧，则否决。

---

### Angle 2 — 用 raw mocap 面标签驱动 wrist 投影（不假设 world-up）  (touch=OmniRetarget input，不动 retargeter)

**做什么。** E094 `adaptive_support` 启发式的泛化，作用于
`global_joint_positions[:, 20:22, :]`（如果存在还包括 rows 22/23），**在 retargeter 看到之前**。
检测：逐帧计算 box-local wrist 坐标；若帧有 raw-active contact 且 wrist 落在非"目标 face"上
或 box 内部，将其投影到（**由 Angle 8 面标签选出的 face** + ε margin），in-plane 坐标 clip
到 half-extents − 2 cm。spider 已实现 post-IK 版本 `workspace/exp_diagnostic/scripts/wrist_repair_top_face.py`；
通过重写 converted NPZ 提升为 pre-IK（08_B_path_omniretarget_audit B-1）。

**关键修正（相对 v1）**：原方案默认假设"world-up = top face"，已被实测证伪：

- box021 18029_p2 obj quat = 90° around +X（v1 §3.2），box half-extents (0.16, 0.21, **0.265**)。
  旋转后：**local +z → world −y（水平方向）；local +y → world +z（向上）**。
- 02_face_selection_audit.md §3 实测表说 box021 R wrist "主接触面 = local +z"（按 object local 投票）
  —— 对应世界里是某个**水平长侧面**，不是顶面。
- v1 §6 / E090 的 "topface-preIK"（把 contact 投到 local +z + 5 cm），相当于**把 contact 强行
  搬到 box 的水平长侧面**，反着推 IK 必然得到不可行姿态。
- E090 S1 full 失败模式（safety 全 0% 但 pelvis 0.134 m）的几何解释：IK 把 G1 推到"伸手到该侧面"，
  但 ref motion 是"蹲箱旁、按 local +z 方向、世界里其实是按水平面"，姿态对不上 → reward 必须跟
  ref motion → pelvis 塌下去。
- Box025 同理：Box025 在 0.31 m 高凳上，人是侧抓 +y 侧壁，world-up 投影把 contact 搬到 box 顶面，
  IK 当然不可行（E090 实测 `inside 48.1/51.9%`）。

结论：world-up = top 假设**只对 quat ≈ identity 的 object 成立**；任何旋转过的或侧抓 intent 的 object
都需要根据 raw mocap 的实际接触面决定投影方向 —— 也就是 Angle 8 的面标签驱动 Angle 2。

**Cost.** 1 engineer-day 把 post-IK→pre-IK 移植到 holosoma 侧（~150 LOC + pipeline glue，
audit 文档 §5 已规定）。1 GPU-h 重跑 13-case 批量 + smoke。audit 指明了投影规则、mesh-AABB
推导和 smpl_scale 处理（08_B_path §4 B-1）。

**风险。** 若仍以 world-up 为默认会在 Box025 失败。必须**用 Angle 8 的面标签替代 world-up 假设**，
而不是 conditional fallback。E094 adaptive rule（已在 raw target 上 / 离 raw 近时保留原 target）
作为额外保护层保留。

**快速证伪测试。** 重跑完整 audit 的 B-1（真 pre-IK，非 B-2 lite damped-LS fallback），目标
`box021_18029_p2` + `box023_p2` + `box025_p2`。通过 = （box021 wrist-inside → 0 AND box023
reward delta p90 ≤ 5 cm AND box025 面投票结果与侧抓 intent 一致，即**不投到 +z 顶面**）。
hsretargeting 环境装好后 30 min。

---

### Angle 2.5 — quat 偏离 identity 的 object 普查（新；防退化）  (touch=read-only inspection)

**做什么。** 一次性脚本：对所有 D003 候选 case 计算 object quat 与 identity 的夹角；标记 > 30°
的全部 case 为"world-up 投影潜在反退化对象"。这些 case 在 Angle 2 上线后必须用 Angle 8
的面标签驱动，禁用任何 world-up 默认。

**为什么。** box021 18029_p2 已经证明 world-up 假设在 quat = 90° around +X 时彻底反向；
要避免未来再有同类回归，必须先知道还有多少 case 落在这个分布上。

**Cost.** 0.5 engineer-day（脚本 + 表）。无 GPU。

**风险。** 无。

**快速证伪测试。** 输出 13 D003 box021 case 的 quat-vs-identity 角度分布表；若 ≥3 个 case 角度
> 30°，强制把它们纳入 Angle 2 + Angle 8 联合验证集。

---

### Angle 3 — 分肢体的 scale 标定（把 smpl_scale 拆成 limb group）  (touch=OmniRetarget, utils.py)

**做什么。** 把单一标量 `smpl_scale = ROBOT_HEIGHT/human_height`（utils.py L251）替换为
per-chain scale：`s_arm = G1_arm_span / SMPL_arm_span`、`s_leg = G1_inseam / SMPL_inseam`、
`s_torso = ...`，应用到对应 joint 子树。G1 chain 长度离线计算一次（常量表）；按 format 存到
`DATA_FORMAT_CONSTANTS`（data_type.py L344）。在 preprocess_motion_data 中用父 joint 作 anchor
逐链应用 scale（保持链连接）。

**为什么。** 修复 §2 row 5 "身体部位比例不匹配"。今日 1.81 m 的人 → G1 1.32 m 会把每条肢体
统一缩放 0.73，但 G1 的臂/身高比**小于** SMPL-X；手臂相对身体变得**过长**，IK 必须把 pelvis
拉低才能让手够到远处，正是 log 76/77 "box025 真问题是臂展物理硬限制" 这条对角线洞察的源头。

**Cost.** 2-3 engineer-days。工作量大部分在验证 chain-rescale 不破坏 Laplacian source mesh
拓扑（顶点数必须不变；只动位置）。~0.5 GPU-h 在 13 case 上重验证。

**风险。** 如果链被缩放但 object 侧 mesh 没缩放，会断开 Laplacian preservation 逻辑。缓解：在
`preprocess_motion_data` 里**在 foot-on-floor 重定位之后、Laplacian 之前**计算 scale；object_poses
缩放（utils.py L254-258）也需要审查，因为它目前用同一个 `scale` 参数。

**快速证伪测试。** 在 `box021_18029_p2` 上度量 per-limb scale 前/后的 wrist-below-pelvis 缺口。
目标：缺口从 0.26 m → < 0.05 m。然后跑 gate + smoke；pelvis_min 应从 ≤0.18 抬到 ≥0.40
（当前 full CEM 量级）。

---

### Angle 4 — Pelvis-height floor cost（G1-aware 站姿先验）  (touch=OmniRetarget)

**做什么。** 加一条单边软约束 `pelvis_world_z[i] ≥ z_floor(i)`，其中 `z_floor(i)` 例如
`max(0.55, 0.55 + 0.4 * object_off_floor)`；以 slack 惩罚形式加到 `solve_single_iteration` 的
CVXPY cost。pelvis_z 的 Jacobian 已为 Laplacian cost 计算。

**为什么。** 直接消除 §2 row 4 "pelvis 坍缩"模式。E094 §6 C2 已证明 SPIDER 下游无法修复它
（object tracking 0.002 m 但 pelvis 0.44 m）。retargeter 目前完全没有"G1 robot **必须**站直才能
抬 box"的概念。

**Cost.** 1 engineer-day。

**风险。** 如果 floor 设得太激进，会损害合法蹲举帧（box-pick-from-floor）的运动可行性。建议把
floor 从 frame-0 = mocap pelvis z 起按线性递增到 standing，与 object 离地高度同步。

**快速证伪测试。** 重跑 E090 S1（`box021_20231011_035_p2_btop`）+ 同一组 SPIDER smoke。
目标：retarget 后 pelvis_min_world_z ≥ 0.50，且 SPIDER full-CEM pelvis_min ≥ 0.55
（E090 §3 今日不达标的阈值）。

---

### Angle 5 — 可达性感知的 target relocation（capability map）  (touch=OmniRetarget input + scoring)

**做什么。** 为每个 (object pose, robot base placement) 对预计算双手可达 workspace 点集，要么
(a) MoveIt 风格离散 capability map，要么 (b) 解析球形可达（G1 arm-span ≈ 从肩 0.55 m）。
每帧若原始 human wrist target 不在 proposed robot base 的可达范围内，要么 (i) 把 base 向 object
滑近，要么 (ii) 把 wrist target 重定位到 support face 上最近的可达点。与 Angle 2 自然组合。

**为什么。** 修复 §2 row 2/5/6 背后的物理 reach 根因 —— 不依赖惩罚而是确保 IK 输入本就可达。

**Cost.** 4-5 engineer-days（capability map 构建 + 集成）。1 GPU-h 在
{box021_18029_p2, box023_p2, box025_p2, Box026_039_p2} 上验证。

**风险。** robot base placement 当前从 frame-0 的 human root + orientation 推断（workflow §6）。
若 base 为满足 reach 而移动，可能破坏后续帧的 foot-stick contact。缓解：只允许 episode 起始时
重定位 base。

**快速证伪测试。** 在 13 D003 box021 case 上度量 "raw wrist out-of-reach %" 前/后。目标：从当前
≥30%（§3.4 hand-below-pelvis 估计）降到 <5%。然后按 Angle 2 验证 gate + smoke。

---

### Angle 6 — 两阶段 retarget：kinematic IK → mini physics MPC 在 OmniRetarget 内部  (touch=OmniRetarget)

**做什么。** 在当前 Laplacian DiffIK 输出 `qpos_kinematic` 之后，在 OmniRetarget pipeline **内部**
跑一个小的 MuJoCo MPC pass（如 50-step warm-started CEM with 8 samples），cost =
（qpos_kinematic tracking + pelvis-floor + wrist-on-support + contact-time），约束为 hard non-penetration。
输出 refined `qpos` 作为新的 STAGE B 结果。

**为什么。** 当前 SPIDER 继承了物理可行性的全部负担；OmniRetarget 交给它的是 infeasible 的
kinematic pose。一个轻量物理感知 refinement 层先吸收掉 infeasibility，SPIDER reward stack 就不必
反向拉回合理 posture（即 E082-E088 的失败模式）。

**Cost.** 5-7 engineer-days（SPIDER MJWP CEM 栈的大部分需要部分移植或 shim 进 holosoma 环境，
或在 stage B 之后以子进程跑）。改动量最大、最 non-trivial。

**风险。** 串联两次物理 pass 可能互相对抗。把 refinement 限制在 per-frame ≤5 cm 根运动 / ≤10°
joint delta（相对 kinematic）。

**快速证伪测试。** 取 E090 S1 `box021_20231011_035_p2_btop` 输出，跑提议的 refinement，检查
最终 qpos 满足 pelvis_min ≥ 0.50、hand-in-box = 0；新 qpos 喂 SPIDER smoke 仍 track box 轨迹
（obj_err_mean ≤ 3 cm）。

---

### Angle 7 — Single-G1 vs dual-G1 vs partner-mocap retargeting 选择  (touch=OmniRetarget input selection)

**做什么。** 今日 D003 pipeline 选一个 G1 对 {person1, person2} 之一 retarget，partner 完全丢弃。
CORE4D 是双人协作数据集；box021 是经典的"两人一起搬低位地面 box"场景。三个子方案：
  (a) Dual-G1 retarget（两个 robot 同时在场景，绑到 person1+person2）；
  (b) 保持 single-G1 但把 partner 当 kinematic puppet 在 box 上回放（提供静态 reach 参考）；只需
      第二个人的 wrist 轨迹。
  (c) 自动路由：用 G1-feasibility gate（E089 §1）把每个 case 分类为"single-G1 feasible"或
      "dual-G1 only"，对 "dual-only" case 跳过 single-G1 retarget。

**为什么。** 一些 box021 D003 序列本质上是双人任务（diagnostic H2）。解决错误的任务是最贵的
失败模式。(b) 最便宜。

**Cost.** (a) ~5 engineer-days（多 robot Laplacian、collision pair）。(b) ~2 days（把 partner
作为装饰可视化；SPIDER reward 可忽略）。(c) ~1 day（仅 routing）。

**风险。** (a) 集成 cost 高、下游 RL 接口不明。(b)/(c) 只是分桶 —— 低风险。

**快速证伪测试。** 用现有 E089 gate 在 13 D003 box021 case 上跑 (c)，确认 routing 与经验
pass/fail 桶匹配。若 gate 预测"single-G1 feasible"的 ≥2 个 case 历史 PASS、且 ≥0 个历史 FAIL：
信号有效。

---

### Angle 8 — 手部 contact-face 监督（用 raw mocap 手按在哪个面）  (touch=OmniRetarget input + new label)

**做什么。** CORE4D mocap 每只手有 5 fingertip joint。从这些指尖逐帧计算手指压在哪个 object face
上（指尖投票哪个面最近）。把这个 face label 作为额外通道加进 converted NPZ
（`contact_face[T,2]`）。在 retargeter 中加一条 per-frame 软 cost，惩罚 G1-fingertip-center / wrist
偏离该面的外法向方向。

**更深层用法（关键升级，相对 v1）**：5 指尖投票应当**同时驱动三个出口**，而不是只做面选择：

1. **面选择**：每帧 5 票，只有 `signed_distance < threshold`（"实际接触上的"）那部分指尖投票，
   悬空的指尖不污染投票。允许 "corner / edge" 标签（≥2 个面票数接近）而非强制单面。
2. **reward target**：取"投票面 + 实际接触上的指尖里离该面最近那个的位置"，而**不是** wrist+5 cm。
   这直接修复 §2 row 1 "STAGE A wrist 抽样代替 contact" 的语义错位 —— 不再用 wrist 假装代表 contact。
3. **OmniRetarget IK 输入的 contact 监督**：Angle 2 的投影方向由该面标签决定（替代 world-up 假设），
   投影目标位置取上一项算出的 reward target。

**为什么。** 系统性消除 §2 row 1（wrist ≠ contact）+ row 3（wrong face）+ row 7（contact mask
mismatch）。E093 量化证据显示 `wrist + 5cm → raw mean` 偏差 ≥ 20 cm 全 case、Box026/Box025
49-64 cm —— 这不是 5 cm palm offset 能补的；唯一根治办法是在 STAGE A 输入端就用 5 指尖语义。
E094 evidence 显示 world-up 投影对 Box021/023 大致对但对 Box025 错（侧抓）；用 raw mocap face
标签可以捕捉人的真实 intent。

**Cost.** 2 engineer-days（3 小时从现有 converted data 算 label；其余是把 label 集成到 retargeter
cost + spider 侧 contact_pos 改造）。

**风险。** mocap 噪声会让 face label 逐帧抖。用 ±5 帧多数投票平滑。edge/corner contact（无单面）
用 fallback 规则（按 signed distance 取最近 face）。signed-distance 阈值需要按 object 类别调
（box 类 1-2 cm 合理）。

**快速证伪测试。** 在 {box025_p2（侧抓）、box021_18029_p2（顶按）、box023_p2（顶按）、
box022_127_p2} 上计算 label。通过 = top/side label 在 ≥90% 接触帧上与人工目检一致。

---

### Angle 9 — wrist-inside-object 惩罚（沿用现有结构）  (touch=OmniRetarget, lightweight)

**做什么。** Angle 1 的子集 —— 不做完整 SDF cost，仅检查 wrist joint 在 *human source* 位置
（应用 smpl_scale 后）是否在 object collision box 内部（object-local frame 的便宜 AABB test），
若是，**在构建 Laplacian cost 之前**把它投到最近面外 + ε。这是最小可行修复。

**为什么。** §2 row 1 最便宜的切口。Pre-IK 而非 in-solver。

**Cost.** 0.5 engineer-day。

**风险。** 只滑动 wrist（不改 elbow/shoulder 的 source coord）会推高该 limb chain 的 local
Laplacian 形变 cost；solver 可能通过别扭地扭臂来补偿。与 Angle 2 组合时可接受。

**快速证伪测试。** 同 Angle 1：3 个 D003 case 上 retarget 后 wrist-inside-box → 0%，且不抬升
CVXPY infeasible 率。半天即可确认。

---

### Angle 10 — IK target 的时序平滑  (touch=OmniRetarget input)

**做什么。** 对 `global_joint_positions[:, 20:22, :]` 在 retarget 前应用 Savitzky-Golay 或低通
（30 fps 下截止 4-6 Hz）。任何 per-frame 跳变 > 8 cm（对应 >2.4 m/s wrist 速度）做阈值裁剪。

**为什么。** 去尖刺 —— 减少单帧 CVXPY infeasible（§2 row 10）和"anchor face 抖动"（row 7）。
独立、可与所有其他 angle 叠加。

**Cost.** 0.25 engineer-day。

**风险。** 过平滑会抹掉合法的快速动作（grasp 过渡）。用 one-pole 滤波器，α 按 object 类别调。

**快速证伪测试。** 在 13 D003 case 上度量 CVXPY infeasible 帧占比前后变化。目标：下降 ≥30%。
无需 SPIDER smoke。

---

### Angle 11 — 学习型 retargeter（PHC / NN 蒸馏）  (touch=OmniRetarget alternative)

**做什么。** 训一个小 MLP / transformer：输入 SMPL-X joint window + object pose window，输出
G1 qpos，监督来自 (a) 当前通过 G1 feasibility gate 的 OmniRetarget 输出（E089），(b) 可得的
AMASS 风格 G1 motion。

**为什么。** 类别性替代 —— 用 amortised 模型替代 per-frame CVXPY，可隐式学到 G1 特定可达性。
**未来 1-2 周内拒绝**：训练数据太少（按 E097 §1，今日 ≤20 PASS case），模型只会记忆。等
Angle 2+3+4+8 把 PASS 池扩到 ≥200 后再考虑。

**Cost.** 10+ engineer-days + GPU 训练。

**风险。** 数据不足；对重 / 大 box 有分布偏移。

**快速证伪测试。** 当前不可执行。

---

### Angle 12 — Object-collision-aware Laplacian（intent 改写 / 不同 contact intent）  (touch=OmniRetarget + label)

**做什么。** 当选定的 contact intent（box021 的 top-hold）被检测为 single-G1 infeasible（通过
Angle 5 的可达性检查），自动以不同 intent（side-grip、双手 under-grip）重发该 case：重打 face
监督标签（Angle 8）并重跑。每 case 产 1-3 个 retarget 变体；下游 gate 选最好的。

**为什么。** 一些 case 在原 mocap intent 下没有 single-G1 解（box021 D003 蹲举），但换 intent
后有解（G1 站立、从侧面把手插进 box 下方）。今日 OmniRetarget 无条件锁定单一 intent。

**Cost.** 3-4 engineer-days（intent 枚举 + 变体生成 + gate 驱动选择）。

**风险。** retarget cost ×3。可能产出"通过 gate 但 RL 不可用"的变体（wrist 接到了侧面但 body
仍是原坍缩 pose）。

**快速证伪测试。** 在 box021_18029_p2 上生成 (top-hold, side-grip, under-grip) 三变体。通过 =
至少一个变体 gate-PASS 且视觉确认是"搬运"姿态而非"躺在 box 上"。

---

### Angle 13 — 课程式 / 渐进式约束激活  (touch=OmniRetarget)

**做什么。** 今日 retargeter 在 frame 0 即激活全部 hard 约束（non-penetration、foot-stick、
joint-limits，加上 Angle-1 wrist-non-pen 若启用）。加一个 schedule：前 10 帧 wrist-non-pen ε
从 0 → 2 cm 线性 ramp，pelvis-floor（Angle 4）同样。

**为什么。** 减少边界（接触阶段起点）的 CVXPY-infeasible，同时不放弃后续约束。Angle 1 + 4 + 6
的便宜辅助。

**Cost.** 0.5 engineer-day。

**风险。** 微小。

**快速证伪测试。** 仅检查 CVXPY infeasible 帧占比 ≤ baseline。

---

### 此处明确拒绝的类别

- **更多数据**（一般性）：无具体角度。E097 当前瓶颈是 feature-gated candidate 数量小，但加更多
  raw data **不会**消除 retargeter 的 wrong-face / hand-in-box / pelvis-low 模式。拒绝。
- **SPIDER 中 reward 权重重调**：超出范围（本文针对 OmniRetarget），且 E060-E088 已穷举证明死胡同
  （diagnostic §7）。
- **整体换成完全学习型 retargeter**：Angle 11，延后。

---

## 第 4 节 — 优先级路线图（未来 1-2 周）

Score = Impact (0-5) × Feasibility (0-5)。与 §3 "Cost" 列工程日数同序。

| Rank | Angle | Impact | Feasibility | Score | Touches |
|---|---|---:|---:|---:|---|
| 1 | A2 + A8 合并：pre-IK wrist-face 投影 + 5 指尖驱动的面标签 + reward target | 5 | 4 | 20 | OmniRetarget input + spider contact_pos |
| 2 | A1：retargeter 内部 object-SDF wrist cost / hard non-pen | 5 | 4 | 20 | OmniRetarget |
| 3 | A4：pelvis-height floor cost | 4 | 5 | 20 | OmniRetarget |
| 4 | A3：分肢体 scale 标定 | 5 | 3 | 15 | OmniRetarget |
| 5 | A7c：按 E089 gate 路由（dual-only case 跳过 single-G1） | 3 | 5 | 15 | OmniRetarget input |
| 6 | A9：最小 wrist-AABB 投影（A1 便宜子集） | 3 | 5 | 15 | OmniRetarget input |
| 7 | A2.5：quat 偏离 identity 的 object 普查 | 3 | 5 | 15 | read-only inspection |
| 8 | A6：pipeline 内物理 MPC refinement | 5 | 2 | 10 | OmniRetarget |
| 9 | A10：时序平滑 | 2 | 5 | 10 | OmniRetarget input |
| 10 | A13：课程式约束激活 | 2 | 5 | 10 | OmniRetarget |
| 11 | A5：capability-map reach 重定位 | 4 | 2 | 8 | OmniRetarget input |
| 12 | A12：intent 改写变体 | 3 | 2 | 6 | OmniRetarget input + label |
| 13 | A11：学习型 retargeter | 5 | 1 | 5 | OmniRetarget alternative（延后） |

### 接下来执行的 Top-3，含明确成功判据

**Top-1 — A2 + A8（pre-IK 用 5 指尖驱动的面标签 + 投影）。**
从 converted NPZ 中已存的 fingertip joint 构建 per-case 面标签（Angle 8 三个出口：面选择 / reward
target / OmniRetarget IK 输入的 contact 监督）；以此驱动**取代 world-up 假设**的面感知投影
（Angle 2）。在 13 D003 box021 case + box023_p2（guard）+ box025_p2（负向 guard）上重跑 STAGE B。
成功判据：

- 几何：≥10/13 D003 case 通过修正后的（face-aware、不假设 world-up）G1-feasibility gate
  （E089 §6 P1）；box023_p2 通过状态保持；box025_p2 不退化（face label 投票结果与 box025 侧抓
  intent 一致，**即不投到 +z 顶面**）。
- **Box021 IK FK INSIDE box 占比从 v1 §3.3 的 33% 降到 < 5%**（虚假 INSIDE 部分被消除）。
- SPIDER 24-step CEM 在 `box021_18029_p2` 上：head_pen ≤ 10%、upper_pen ≤ 10%、
  LH/RH_floor ≤ 5%（vs E082-E088 baseline 70-89%）。
- 视觉：f25 / f55 / f80 关键帧显示 G1 处于搬运姿态而非躺在 box 上。

**Top-2 — A1（object-SDF wrist non-penetration cost）。**
在 `solve_single_iteration` 加 SDF cost 项（Jacobian 已有）。渐进验证：先做软惩罚，再升级到带
slack 的硬约束。成功判据：

- retarget 后 wrist-inside-box = 0% 在全部 13 D003 case + box023_p2。
- per-case CVXPY infeasible 帧占比 ≤ 5%（今日 tail）。若更高，回退到 soft-only。
- 与 Top-1 叠加：SPIDER full CEM 在 `box021_20231011_035_p2_btop` 达到 pelvis_min ≥ 0.45
  （今日 E090 full = 0.134）。

**Top-3 — A4（retargeter 内部 pelvis-floor 软约束）。**
在同一 solver pass 加 `pelvis_world_z ≥ z_floor(t)` slack 惩罚。成功判据：

- retarget 后 pelvis_min_world_z ≥ 0.50 在 ≥10/13 D003 case 上（今日大多 0.58-0.72，关键是失败
  tail：必须拉起最差的 3 个）。
- 与 Top-1 + Top-2 叠加：SPIDER full CEM `box021_20231011_035_p2_btop`
  pelvis_min ≥ 0.55（E090 §3 失败阈值）。
- Box023_p2 guard 保持（full CEM `WORK`）。

若 Top-1+2+3 完成后 D003 box021 上 pelvis_min 仍 < 0.55，A3（分肢体 scale）自然成为 Top-4，
因为它直击 pelvis 坍缩的根本几何原因（手臂相对躯干过长 → IK 把 pelvis 拉低让手贴近 box）。

---

## 第 5 节 — 关于 OmniRetarget 仍不清楚的事

每条列出未知项 + 解决它所需的最小实验。

1. **IK loss 是否可达性感知？**
   *状态：* 否，只有 Laplacian 形变。Solver 无 closed-form 概念去判断"该点在当前 base 下 G1 工作空间外"。
   *确认：* 在 STAGE B 上跑一个合成帧，wrist source 放头顶上方 1.5 m；记录 residual
   `||FK(qpos)_wrist − source_wrist||`。若无 warning 地爆掉，确认。

2. **IK 用的是 G1 kinematic 模型（URDF）还是仅 JOINTS_MAPPING 名→link 表？**
   *状态：* 用 MuJoCo 加载 `g1_29dof_w_<obj>.xml`（interaction_mesh_retargeter L139-145），
   URDF kinematics 在场。mapping 只决定哪个 link 接哪个 source。无未知。

3. **box021 D003 IK 解的实际 per-frame residual 是多少？**
   *状态：* 作为 `cost` 字段存到 `{task}_original.npz`（interaction_mesh_retargeter L555-560），
   但它是所有帧的标量。没有 per-frame per-link residual 日志。
   *解决：* 加一次性日志（或事后逐帧 FK）—— ~30 LOC，只读 inspection 脚本即可。否则无法分辨
   "infeasible" case 是均匀坏还是集中在特定帧。

4. **`smpl_scale` 如何与 retargeter 中的 object collision mesh 交互？**
   *部分答案：* utils.py L254-258 把 object x/y translation 按 `smpl_scale` 缩放，但 z 仅做 floor-shift、
   不缩放。这是不一致的 —— 多帧下 object 轨迹相对人体（继而相对 robot）的几何会失真。需要在
   box021_18029_p2 上明确验证 scaled vs unscaled 帧之间 box 相对 pelvis 的运动是否匹配。

5. **Phase 4 `enable_contact_preservation`（L483-520）在 D003 上打开是否有用？**
   *状态：* production OFF（audit §6）。机制是 wrist-to-object-center 距离保持，**非**位置。是否
   帮助或损害未知。
   *解决：* 在 `box021_18029_p2` 上跑 STAGE B 打开该 flag，对比 gate 指标。~10 min。应在 A1/A2
   实现前做 —— 若 Phase 4 已部分解决 wrist-inside-box，A1/A2 更便宜。

6. **base placement（q_init）是否是 Box025 的绑定约束？**
   *状态：* `initialize_robot_pose`（robot_retarget.py L549-575 附近）从 human 的 frame-0 root +
   orientation 推断。对 Box025（object 在凳上）人的 frame-0 已经站在 60 cm 高 object 旁 —— robot
   起始姿态可达。但 hand world z（0.62 m）低于 box top（0.78 m），对侧抓需要 hand 在腰侧高度。
   需要确认 base placement 不是 bug。
   *解决：* 读取并 log q_init for box025_p2 vs box023_p2 vs box021_p2，看 object 的 x/y 偏移是否
   可比。

7. **(新) 还有多少 case 的 object quat 偏离 identity > 30°？**
   *状态：* 已知 box021 18029_p2 quat = 90° around +X 让 world-up 投影完全反向。其他 D003 case 的
   quat 分布未系统化普查。
   *解决：* Angle 2.5 普查脚本，0.5 engineer-day。若多于 3 个 case 偏离 > 30°，必须强制纳入
   Angle 2 + Angle 8 联合验证集，避免上线后退化。

第 1、3、4、5、7 项是会改变 Top-3 起点的项。第 5 和第 3 最便宜（各 <1 hour），应在写代码前回答。
第 1 项会在 Angle 5 构建时被隐式回答。第 7 项是 Angle 2 + 8 上线前的必做前置。

---

## 关于只读 inspection 的备注

本次分析未使用只读脚本 —— 相关代码与历史 log 已提供完整信息。Top-1/2/3 实施前应执行的两个
便宜 inspection：

- 30 行脚本：`box021_18029_p2` 的 per-frame per-link FK residual（解 §5 项 3）。
- 5 行脚本：用 `--retargeter.enable-contact-preservation` flag 重跑 STAGE B 并重 gate
  （解 §5 项 5）。
- 0.5 engineer-day 脚本：D003 全 case 的 object quat-vs-identity 角度普查（Angle 2.5，解 §5 项 7）。

三者就绪时放到 `workspace/exp_diagnostic_v2/scripts/`，目前先不动。
