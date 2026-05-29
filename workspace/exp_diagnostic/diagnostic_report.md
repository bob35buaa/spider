# Box021 失败实验全面诊断

日期：2026-05-28
对应任务：`workspace/exp_diagnostic.md`
执行人：诊断会话（spider 主仓库，本地 GPU）

## TL;DR

E028 / E030 / E082-E088 上 box021 D003 全部失败，box023_p2、box025_p2 在同一 pipeline 上通过/接近通过。
**根因是 box021 D003 case 的 kinematic retargeted 接触几何对 G1 单人不可行**：
- box021 物体本身贴地、半高 53cm；CORE4D 人体是双人协作下蹲合抱再起身的姿势；
- OmniRetarget 出来的 G1 双手 IK 目标位于 box 上沿/侧沿附近、世界 z ≈ 0.47–0.56m，**比 G1 pelvis (0.58–0.72m) 还低 5–25cm**，且 box 顶面在世界系只有 0.35m；
- 更糟的是 `d003_box021_20231018_029_p2` 的右手 IK FK 落点 **33% 帧完全位于 box 内部**（box023_p2 是 0%），其余两 case 的双手平均也只在 box 表面外 6–9cm，远比 box023 的 17–26cm 紧；
- 在这种几何下机器人必须深度前倾才能去摸到目标，CEM 找到的局部解是**抛弃手部接触、用上半身/头/胸/肩去贴 box 来满足 contact reward**，于是出现 E082-E088 视频反复看到的 head pen 80%+、upper pen 80%+、hand-floor 40%、倒伏 / 趴箱 / 翻箱。

E082-E088 的失败不是 reward 调参问题，也不是 mass、不是 hard gate 阈值，而是 **SPIDER 上游 (CORE4D → Holosoma → SPIDER) 提供的参考运动 + 接触目标本身就不是 G1 可执行的搬运**。

修复方向应当回到上游：(a) 重做 box021 hand contact target，把双手目标投到 G1 可达的高位/上沿；或者 (b) 承认 box021 在 G1 单人下不可行，改走 dual-G1 / Mocap partner 路径；或 (c) 修改 motion 让 pelvis 不要被强制蹲到 0.58m，给 robot 一个直立 carry 的姿态参考。继续在 reward/mass/hard gate 上调参已经被 E060-E088 证伪。

## 1. 实验背景

- 框架：SPIDER (sampling-based MPC + MuJoCo Warp) 把 CORE4D 人-人-物协作 mocap → G1 humanoid 动作，下游接 Holosoma RL。
- 失败数据：CORE4D D003 序列的 Box021，三条：
  - `d003_box021_20231018_029_p2` (E082-E088 主验证)
  - `d003_box021_20231011_035_p2`
  - `d003_box021_20231020_019_p1`
- 通过/接近通过的对照：
  - `box023_person2` (E082-E088 guard、pass)
  - `box025_person2` (E080 partial)
- 历史：E028 (13 case) → E030 (D6 locked + CEM) → E082-E088 (leg/upper collision、mass sweep、hard gate、absolute clearance)，0/16+ case 成功。

## 2. 调研范围

| 来源 | 内容 |
|---|---|
| `workspace/core4d/EXPERIMENT_TRACKER.md` | 88 个实验的全局历史 |
| `workspace/core4d/log/103_E082_*` ... `110_E088_*` | E082-E088 完整结果与失败模式 |
| `workspace/core4d_collab_retarget/log/29_E028_*` 与 `37_E030_*` | E028/E030 在 D003 13 case 与 3 case 上的失败 |
| `example_datasets/processed/core4d/unitree_g1/humanoid_object/{task}/0/trajectory_kinematic.npz` | 直接读 box021 / box023 / box025 的 retargeted qpos、contact mask、contact_pos |
| `example_datasets/.../{task}/scene.xml` | 物体几何、mass、quat、collision half-extents |
| `spider/simulators/mjwp.py` (L940-1080) + `examples/run_mjwp.py` (L920-1015) | contact_hdmi reward target 的实际构造路径 |
| `workspace/exp_diagnostic/findings/0{1,2,3,4}_*` | 本次新生成的对照表 + 可视化 |

注意：当前仓库不含 D003 box021 原始 holosoma source qpos，所以本诊断只能基于 SPIDER 端的 trajectory_kinematic.npz；这已经足够判定 SPIDER 输入侧的几何不可执行，但不能区分到底是 OmniRetarget 阶段错还是后续 D003 contact mask 生成阶段错（H3）。

## 3. 数据对照（核心证据）

### 3.1 SPIDER reward 用的 contact target 真实来源

读 `examples/run_mjwp.py:920-1015`，contact_hdmi 的 per-frame target 通过 `contact_target_per_frame` 进入 reward；其来源分两种：

- `contact_hdmi_target_source = ref_fk` (默认)：对 `qpos_ref[t]` 做 mj_forward，取 `xpos[hand_body]` + `xquat[hand_body] @ eef_offset (= [0.05,0,0])`，再投到 object local frame。等价于：**"G1 IK retarget 后的 wrist + 5cm 前向偏移点"**。
- `contact_hdmi_target_source = external` (E085+)：从 `*_contact_target_object_local.npz` 读 raw CORE4D 人手坐标投到 object local frame。

`contact_pos` 字段（存于 trajectory_kinematic.npz）是 SPIDER 上游写入的"参考接触点"，与 reward target 在概念上对齐但实际数值可能微差。本诊断同时检查两者，结果一致：见下表。

### 3.2 contact_pos (raw mocap reference) 在 object local frame 的几何

| Case | T | obj quat | L 主面 (频次) | L signed dist | R 主面 | R signed dist | L hand world z | R hand world z |
|---|---:|---|---|---:|---|---:|---:|---:|
| **box021 18029_p2** | 75 | 90° around +X | -x(52), -z(21) | +0.079 | +z(49), -x(15) | +0.063 | 0.470 | 0.497 |
| **box021 11035_p2** | 133 | 90° around +X | +x(102) | +0.082 | -x(107) | +0.096 | 0.523 | 0.521 |
| **box021 20019_p1** | 98 | yaw ~360° | +x(47), +y(27), -z(24) | +0.190 | -x(64) | +0.117 | 0.563 | 0.492 |
| **box023 p2 (OK)** | 136 | yaw 180° | **+z(70), +y(64)** | **+0.263** | +x(70), +y(65) | **+0.174** | 0.629 | 0.552 |
| **box025 p2 (partial)** | 124 | identity | +z(124) | +0.049 | +z(124) | +0.035 | 0.619 | 0.615 |

含义：`signed dist`>0 = 手在 box 外该面外侧多少米；`L hand world z` = 左手世界 z 坐标 (m)。

### 3.3 G1 IK 重定向的 wrist FK 落点（reward `ref_fk` 实际看到的目标）

读 `qpos_ref` 做 mj_forward + eef_offset = [0.05, 0, 0]：

| Case | L wrist world z mean/min/max | R wrist world z mean/min/max | L INSIDE box | R INSIDE box | L signed dist | R signed dist |
|---|---|---|---:|---:|---:|---:|
| **box021 18029_p2** | 0.498 / 0.243 / 0.742 | 0.522 / 0.269 / 0.741 | 0% | **33.3%** | -0.036 | +0.008 |
| **box021 11035_p2** | 0.549 / 0.294 / 0.710 | 0.548 / 0.278 / 0.705 | 0% | 0% | +0.071 | +0.007 |
| **box021 20019_p1** | 0.583 / 0.234 / 1.286 | 0.518 / 0.231 / 0.721 | 0% | 0% | +0.156 | +0.031 |
| **box023 p2 (OK)** | 0.645 / 0.225 / 1.160 | 0.578 / 0.262 / 0.735 | 0% | 0% | **+0.276** | +0.188 |
| **box025 p2** | 0.645 / 0.554 / 0.701 | 0.641 / 0.544 / 0.693 | 0% | 9.7% | +0.049 | +0.034 |

**关键观察**：`box021_18029_p2` 的右手 IK FK 落点 33% 帧位于 box 内部，左手 signed dist 平均 -0.036m（贴在 -x 面内部 3.6cm）。box023_p2 IK 落点 0% 在 box 内、signed dist 平均 +18 ~ +27cm 在外。这说明 OmniRetarget IK 把 G1 单人压缩到双人协作的 contact 目标时，box021_18029_p2 已经 **解出物理上嵌入物体的姿态**，CEM 必然要在"维持 IK 姿态 → 自/物体穿透"与"放弃 IK 姿态 → 失去 contact reward" 之间做权衡。

### 3.4 几何决定 G1 是否可达

世界坐标下的可达性：

| Case | obj 世界 z (init) | obj 世界 top z | half-extents | G1 pelvis z min | L hand z mean | hand 高出 box top |
|---|---:|---:|---|---:|---:|---:|
| box021 18029_p2 | 0.145 | ~0.35 (rot 90° + half_y=0.21) | 0.16/0.21/0.265 | 0.716 | 0.470 | +12cm |
| box021 11035_p2 | 0.139 | ~0.35 | 同上 | 0.583 | 0.523 | +17cm |
| box021 20019_p1 | 0.152 | ~0.36 | 同上 | 0.671 | 0.563 | +20cm |
| box023 p2 | 0.140 | ~0.32 (yaw 180°) | 0.153/0.157/0.177 | 0.679 | 0.629 | +31cm |
| box025 p2 | 0.310 | ~0.78 (identity) | 0.377/0.378/0.469 | 0.745 | 0.619 | -16cm（手在 box 上半部侧面） |

- box021 的手在世界系比 box 顶面高 12-20cm，**但比 pelvis 低 12-25cm**，意味着 G1 必须"弯腰送手到大致与 box 顶平齐再合拢"。
- box023 的手比 box 顶面高 30cm 左右，正好是双手在 box 上方稍内侧的"双手抱/扶箱顶" 姿态，G1 直立就能达到。
- box025 因为 box 本身被人放在 0.31m 高的支撑物上（mocap 中是凳子/桌子），即使 box 比 G1 还高，hand 仍能在 G1 直立时按住 box 中部侧壁。

### 3.5 可视化：object local frame 下的 wrist 与 contact_pos 散点（已生成）

| Case | 文件 |
|---|---|
| box021_18029_p2 (FAIL) | `findings/04_overlay_box021_18029_p2_FAIL.png` |
| box021_11035_p2 (FAIL) | `findings/04_overlay_box021_11035_p2_FAIL.png` |
| box021_20019_p1 (FAIL) | `findings/04_overlay_box021_20019_p1_FAIL.png` |
| box023_p2 (OK) | `findings/04_overlay_box023_p2_OK.png` |
| box025_p2 (partial) | `findings/04_overlay_box025_p2_partial.png` |

肉眼对照三视图（XY top / XZ side / YZ side），黑色矩形 = box collision boundary；红/蓝点 = G1 wrist IK；橙/绿叉 = 上游 contact_pos：

- box023_p2 的 4 类点都 **明显聚集在 box 上方的 +z 半空间**，左右点呈对称的"双手扶上沿"模式，距 box 表面 >15cm。
- box021_18029_p2 的左右手分别聚集在 **box 的 -x 与 -z 两个相邻面附近，紧贴或穿入 box 边缘**；左手点云在 (-x,-z) 角附近，离 box 表面 0–8cm，部分点完全落在 box 体内的 -x 侧。
- box021_11035_p2 与 box021_20019_p1 类似，双手主要在 ±x 侧面紧贴。
- box025_p2 双手都在 +z 面正上方约 5cm 处。

这与 §3.2/§3.3 的数值完全一致。

### 3.6 Box021 的特殊性是上游不是 reward

| 维度 | box021 (D003) | box023 p2 | box025 p2 |
|---|---|---|---|
| obj 世界起始 z | 0.14 m（贴地） | 0.14 | **0.31**（垫高）|
| obj 半高（local z） | 0.265 | 0.177 | 0.469 |
| obj quat 主分量 | ~90° around +X | yaw 180° | identity |
| Trajectory T | 75–133 帧 | 136 | 124 |
| 物体 mass | 29.6 kg → 10 kg (E087) | 5 kg | 5 kg |
| IK wrist 主面 | 侧面 / 底缘 | top 与 高侧 edge | top 面 100% |
| IK wrist 世界 z 均值 | 0.47-0.58 m | 0.55-0.65 m | 0.62-0.65 m |
| G1 pelvis z min | 0.58-0.72 m | 0.68 m | 0.75 m |
| IK wrist 在 box 内 % | **R 33%** (18029) / 0 / 0 | 0% / 0% | 0% / 10% |
| upper-body pen (E082-E088 sim) | 76-93% | 0% (guard) | (没跑) |

## 4. 已被排除的因素（按 E082-E088 实证）

| 候选根因 | 证据 | 状态 |
|---|---|---|
| 物体 mass 异常 (29 kg) | E087 5 kg/10 kg sweep 仍 fail (头/胸压箱 70-89%) | 部分有影响，非主因 |
| safety penalty 权重太小 | E087C 把 penalty 量级提到 -0.634，head pen 仍 89% | 排除 |
| object lift / floor reward 失效 | E088C 加绝对 clearance 后能离地但靠翻箱 | 口径修复有效但不解决根因 |
| Hard CEM gate 不存在 | E088 已实现 hard gate + fallback，依旧 0/3，valid frac ≤ 0.28 | 排除 |
| Leg-object collision pair 缺失 | E081/E082 加 16 个 pair 仍 fail | 排除 |
| Upper-body-object collision pair 缺失 | E083 加 7 个 pair，guard pass、main 仍 fail | 必要前提但不能解决根因 |
| Knot_dt / noise / actuator port (HDMI 对照) | E065-E067 完整 ablation 全 fail | 与本案不直接相关 |
| reward task-specific (palm normal、eef offset) | E060-E062 ablation case-divergent，最终走 X1 auto | 部分通过但 box021 D003 没显著 |

## 5. 主诊断与候选 hypothesis 排序

**H1 (主): box021 D003 kinematic retarget 出的 G1 双手目标位置在世界系既低于 pelvis、又紧贴或嵌入 box 体；CEM 在该几何下没有可行轨迹。**
- 直接证据：§3.3 中 `box021_18029_p2` 的 R wrist 33% 帧 INSIDE box；§3.4 hand z 远低于 pelvis；§3.5 可视化清晰显示三个 D003 case 双手都在 box 的 -x / +x / -z 紧贴或埋入。
- 解释 E082-E088 视频里"头/胸压箱、hand-floor、倒伏"：CEM 必须在 (a) 维持 contact reward 但穿透物体 (b) 避免穿透但失去 reward 两个选项里选；当前 reward 鼓励 (a)，而 (a) 的 collision-resolved 解就是上半身贴 box。
- 可证伪：若把 box021 IK target 重投到 box 顶面 +5cm（或合理的双手"扶上沿"），同一 reward stack 下 head pen 应 <10%、hand-floor <5%。

**H2 (二级): box021 D003 ref motion 物理上接近"双人下蹲合抱并起身搬一个 53cm 高 ground-level 53×42×32cm 重物"，G1 单人臂展+身高根本不足以模拟。**
- 证据：CORE4D 是协作数据，box021 D003 的 person2 在 mocap 中是"对侧蹲下接物"角色；G1 单人臂展 ~ 0.5m << 人臂展 0.8m + body width；E054/E056 历史上对 `box021_person1` (老版本) 给出过 "同面异常 = 双手按压顶面，不是搬运" 的分类，D003 case 不一定一样但任务难度类似。
- 这与 log 76/77 的 strategic correction "box025 真问题是臂展物理硬限制" 是同一类问题。
- 可证伪：H1 修好之后跑 full CEM，若 head/upper pen 已降但物体仍掉地/未抬起，说明 H2 binding。

**H3 (三级): D003 box021 转换链路在 person 选择 / time window / smpl_scale 上有误，导致 contact target 实际不对应"搬运" intent。**
- 证据弱：mass 数值不一致 (29.6 vs 5 kg) 暗示 box021 的 scene 由另一条转换链生成；早期 `box021_person1` 与 D003 person2 不同 case；E028 报告 9/13 D003 case `anchor_face_review=true` 说明 face 自动选择本身在大多数 D003 box021 case 上都不稳。
- 可证伪：用同一份原始 CORE4D 20231018-029-person2，重新走 spider 标准 (非 D003) 转换链跑出 contact target，与 D003 这条做差。

**H4 (低优): scene_act 物体 actuator + 大质量耦合带来 init 反作用力。**
- E087 mass 降到 5/10 kg 已尝试，效果有限。可作 secondary。

## 6. 诊断 plan（行动顺序）

### Phase A — 已完成（不需 GPU）

| Step | 行动 | 输出 |
|---|---|---|
| A0 | 调研 E082-E088 + E028-E030 全部 log，梳理失败模式 | §3-4 |
| A1 | 追踪 SPIDER `contact_hdmi_rew` 的 target 真实来源到 `examples/run_mjwp.py:920-1015` 与 `spider/simulators/mjwp.py:940-1080` | §3.1 |
| A2 | 读 trajectory_kinematic.npz，计算 contact_pos 在 object local frame 下的 face / signed dist | `findings/02_local_frame_compare.txt` |
| A3 | mj_forward(qpos_ref) 计算 G1 wrist IK 在 object local frame 的位置 + INSIDE 占比 | `findings/03_fk_wrist_local.txt` |
| A4 | 用 matplotlib 把 box 与双手 (IK + raw mocap) 散点叠加渲染三视图 | `findings/04_overlay_*.png` |

### Phase B — 待执行（需要小量 GPU 但不必 full CEM）

| Step | 行动 | 期望产出 | 判定 |
|---|---|---|---|
| B1 | 写最小脚本：对 box021 三 case 做 IK 修复——把 `contact_pos[t, hand]` 投到"box 顶面 +5cm" 或上沿（沿 outward normal 把 IK 失效帧推到 box 表面外 +5cm，并在世界 z 上限制 ≥ pelvis_z - 0.10m），并 dump 出 `repaired_contact_target_object_local.npz` 给 `contact_hdmi_target_source=external` 用 | `workspace/exp_diagnostic/scripts/repair_contact_target.py` + `findings/05_repaired_targets/*.npz` | 提供 H1 修复输入 |
| B2 | 用 `examples/run_mjwp.py` 跑 box021 18029_p2 的 4-step smoke 与 24-step (~5min) mini CEM，使用 B1 的 repaired target；不为对比 quality，只看 reward breakdown 与 head/upper penetration | `findings/06_smoke_repaired/*.log` 与 csv | 若 head pen 在 mini CEM 阶段就从 80%+ 降到 box023 量级 (<5%)，强烈支持 H1 |
| B3 | 同样跑 box021 18029_p2 的 baseline (E085 raw target) 作为对照 | `findings/06_smoke_raw/*.log` | 隔离"target 改动"是唯一变量 |

### Phase C — 看 B 结果决定

| 情况 | 行动 |
|---|---|
| B2 vs B3 head/upper pen 差异 ≥ 50pp | C1: 编写 `workspace/core4d/scripts/E089_contact_target_repair.py`，对 D003 13 case 全部重生成 contact target；用 E085 reward stack + E088C 的 absolute clearance 跑 full CEM；预期 anchor_face_review 数与 head/upper pen 双双下降 |
| B2 改善有限 (head pen 仍 >40%) | C2: H2 binding。承认 box021 D003 单 G1 不可行，从 candidate 中剔除，或启用 dual-G1 / Mocap partner 路径（参考 holosoma collab retarget 已有的 dual_humanoid_object 流程） |
| B1 修复时发现 IK feasibility 大量失败 (residual > 15cm) | C3: 回到 holosoma OmniRetarget，把 box021 的接触 intent 改成"上沿/顶面"，重做整条 D003 box021 pipeline。或换用 person1 而不是 person2，看是否 person 选反 |
| 任意结果 | C4: 保留 E088C 的 absolute clearance + leg/upper collision pair 作为下一轮 reward stack 默认；不要再调单个 weight |

## 7. 不建议继续的方向

- 任何 `robot_object_penalty_scale` / `contact_hdmi_gain` / `task_obj` 的小 sweep — E060-E067 + E082-E087 已证 dead-end。
- 任何 mass sweep — E087 已扫 5/10/29 kg。
- D6 locked support / COLA support body — E029-E030 已证 support scaffold 进入 CEM 会退化。
- 在当前 contact target 上继续调 hard-gate 阈值 — E088 valid sample 已稀薄到 fallback ≥ 60%，问题在约束本身不可行。
- 假设 box021 与 box023/box025 共用一套 reward 调参方案 — 当前评估表明 box021 D003 的几何与那两个本质不同。

## 8. 风险与未知

- 本诊断依赖 `trajectory_kinematic.npz` 中的 `contact_pos` 与 `qpos_ref`；如果 SPIDER 内部 `contact_hdmi_target_source=external` 路径用的 raw target 与 contact_pos 数值差异大，B1 修复应同时覆盖两条路径。
- 仓库无 D003 box021 holosoma 源数据本地副本，H3 验证（A4 / C3）需要远端配合。
- B2 smoke 只能给方向性证据，不能替代 full CEM 验证。
- 即使 H1 修复，box021 物体重 + 体积大 + 起始贴地，G1 单人是否真能搬，仍受 H2 约束。
- 当前已 commit 的 `c3e0169 exp(core4d): E083` ... `ee6783a Track E082-E088 ...` 都用旧 contact target，重做后旧数据不应用作 RL seed。

## 9. 产出清单

- `workspace/exp_diagnostic/scripts/compare_traj_data.py` — 轨迹基础对照
- `workspace/exp_diagnostic/scripts/deep_compare.py` — 详细 obj / pelvis / contact 对照
- `workspace/exp_diagnostic/scripts/object_local_contact.py` — contact_pos 在 object local frame 下的 face / signed dist 分析（signed dist sign 已修）
- `workspace/exp_diagnostic/scripts/fk_wrist_in_object_frame.py` — G1 wrist IK FK 在 object local frame
- `workspace/exp_diagnostic/scripts/render_wrist_overlay.py` — 三视图散点叠加
- `workspace/exp_diagnostic/findings/01_traj_compare.txt`
- `workspace/exp_diagnostic/findings/02_local_frame_compare.txt`
- `workspace/exp_diagnostic/findings/03_fk_wrist_local.txt`
- `workspace/exp_diagnostic/findings/04_overlay_*.png` (5 张)
- `workspace/exp_diagnostic/diagnostic_report.md` — 本文档

---

## ERRATA — 2026-05-30（在 exp_diagnostic_v2 + E098 修订后追加）

**§3.2 / §3.5 的"接触面 / 主面 / signed dist"具体数值受 B1 + B4 污染**：

- **B1**：当时使用的 `face_label` 来自 `workspace/core4d_collab_retarget/scripts/E017/audit_select_anchors.py:192`，只用 xy 二维 argmax，**完全屏蔽 ±z 面**。所以 §3.2 表中 "box021 18029_p2 L 主面 = -x、signed dist +0.079" 这类 "L/R 主面"应理解为"xy 投影下投票最多的水平面"，不是真 3D 主面。按全 3D argmax 重算，box021 D003 多数 hand-case 主面是 +z（见 `workspace/exp_diagnostic_v2/findings/02_face_selection_audit.md` §3 B1 表）。
- **B4**：§3.3 表中"L wrist world z 0.470 / R wrist world z 0.497"是从 `trajectory_kinematic.npz` 的 `contact_pos` 读出，但 `contact_pos` 实际是 **IK FK palm site** 而非 raw mocap。下游 §3.2 / §3.5 文字也踩了这个坑。
- **B6**：进一步，"wrist FK 33% 帧 INSIDE box" 这条 H1 关键证据本身**部分是 wrist 内陷的几何假象**——人指尖贴 +y 侧外、wrist 被反向弯曲带到 box 几何中心方向，FK 落点看上去 INSIDE 但实际人手没穿模。详 `workspace/exp_diagnostic_v2/findings/02_face_selection_audit.md` §4。

**H1 结论方向仍然正确**（box021 D003 几何对单 G1 不可行），**但具体描述的几何画面需以 v2 §3 B1 + §4 B6 为准**。E098 (Stage 0) 修复 B1+B2+B3+B4+B5；E099+ 起所有"接触面 / 主面"统计应使用 `workspace/core4d/scripts/E098/face_utils.py` 的全 3D helper，不应回到本文档 §3.2/§3.5 的 xy-only 表。
