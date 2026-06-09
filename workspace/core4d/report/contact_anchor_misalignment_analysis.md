# 接触锚点错位分析：为何"两边都用橡胶手"接触仍不涨

日期：2026-06-09
分支：exp/core4d-collab-retarget
git HEAD：d91a6c3edb40b0295a1dc779671195f62f620efc
性质：纯分析报告（不改代码、不跑 CEM / OmniRetarget）

---

## 结论先行

1. **想做的实验是冗余的**。"OmniRetarget 侧用橡胶手 geom + SPIDER 侧也用橡胶手"——OmniRetarget 跑 CORE4D 物体交互**从来就用橡胶手 mesh 做手碰撞 geom**（与 RL/SPIDER 同一个网格），且**根本没有 sphere-手的物体场景**。所以 E147-E149 表里 "OmniRetarget" 列就是橡胶手参考、"rubber Spider" 列就是橡胶手 CEM，"两边橡胶手"= E149 relaxed8 已有结果，再跑只复现 rubber 列。

2. **接触不涨的真因不是"谁还在用球"，而是接触奖励的锚点埋错了位置**。`contact_hdmi` 奖励跟踪的是 `wrist_yaw_link + [0.05,0,0]` 这个点——它**埋在橡胶手根部**（掌心在 0.08、指尖伸到 0.173）。CEM 把这个埋在手内的点往箱面拽，真实手面必然插进箱子（→ 穿透↑）；而奖励**全程不读橡胶手表面**（不碰任何 geom/mesh/SDF），所以接触保真度上不去（→ 接触↓）。这正是 E147-149 "穿透改善、接触不升反降" 的机制。

3. **参考侧和机器人侧其实早已对齐**——它们用同一个 `config.contact_hdmi_eef_offset` 字段、同一个 body、同一套 FK+offset 公式。错位不在两侧之间，而在**这个被双方共享的锚点本身不落在橡胶手接触面上**。因此"对齐"的正确含义是把锚点挪到手的接触面，而不是去同步两侧。

---

## §1 OmniRetarget 侧本就是橡胶手（字面实验冗余的证据）

### 1.1 加载链

OmniRetarget 跑 CORE4D 物体交互的模型加载路径（逐行核实）：

1. `workspace/core4d/data_preprocess/pipeline.sh:315-320` → 调 `robot_retarget.py --task-type object_interaction --task-config.object-name <object>`
2. `holosoma_retargeting/src/interaction_mesh_retargeter.py:138-146`：`robot_model_path = ROBOT_URDF_FILE = models/g1/g1_29dof.urdf`，物体分支做
   `robot_xml_path = robot_model_path.replace(".urdf", "_w_" + object_name + ".xml")`
3. **实际加载 `holosoma_retargeting/models/g1/g1_29dof_w_<object>.xml`**（如 box → `g1_29dof_w_box004.xml`）

该文件 `:263` 手部碰撞 geom：
```xml
<geom name="left_rubber_hand_link" type="mesh" mesh="left_rubber_hand_link" />
```
无 `contype/conaffinity`，默认 1/1 → 是**可碰撞的橡胶手 mesh**（同 body 里另一条带 `contype=0 conaffinity=0` 的才是纯视觉）。

### 1.2 三方橡胶手是同一几何

| 仓库 | 文件 | 校验 |
|---|---|---|
| 下游 RL (holosoma) | `meshes/left_rubber_hand.STL` | sha256 `cff2221a690fa69303f61fce68f2d155c1517b52efb6ca9262dd56e0bc6e70fe` |
| SPIDER | `assets/robots/unitree_g1/meshes/left_rubber_hand.STL` | sha256 同上（字节一致） |
| OmniRetarget | `models/g1/assets/left_rubber_hand.obj` | 同几何：22876 顶点 / 45748 面，bbox 逐轴吻合到 mm |

OmniRetarget MJCF 加载 `.obj`（`g1_29dof_w_box004.xml:32` `file="left_rubber_hand.obj"`），`.obj` 与 RL/SPIDER 的 `.STL` 是同一网格（仅格式不同）。

### 1.3 不存在 sphere-手的物体场景

`holosoma_retargeting/models/g1/` 下 28 个 `g1_29dof_w_<object>.xml`（box/bucket/board/chair/desk/stick/largebox 等）**全是橡胶手**；`g1_29dof_spherehand.xml` 只有 robot-only 版（给爬墙 demo），**没有任何 `spherehand_w_<object>.xml`**。`ensure_g1_object_xml`（`pipeline.sh:161-215`）的 seed（box004/bucket001/...）也全是橡胶手。→ 想做"球-Omni vs 橡胶-Omni"对照，得先造场景（本轮不做）。

### 1.4 OmniRetarget 的手 geom 只单向防穿、不主动贴

引 `tmp/OmniRetarget_gemo.md`（已调研结论）：手碰撞 geom 仅通过 non-penetration 硬约束 φ(q)≥0 参与（`interaction_mesh_retargeter.py:1236-1287, 729-737`），**只防手插进箱、不奖励贴近**；主动贴近项 `enable_contact_preservation` 默认关（`config_types/retargeter.py`），且即便开也是 wrist→物中心距离（B6 线），不涉及手 mesh 形状。→ OmniRetarget 参考的接触上限**不由换不换 mesh 决定**（它早是 mesh），换橡胶手在 OmniRetarget 层是空操作。

---

## §2 contact_hdmi 接触锚点机制：两边对齐到同一个 wrist+0.05 点

接触奖励 `contact_hdmi` 的锚点定义为 `wrist_yaw_link 原点 + R(wrist_quat)·eef_offset`，`eef_offset = [0.05,0,0]`（`config.py:195-197`，注释 `# wrist→palm`）。`hand_approach_body_names = ["left_wrist_yaw_link","right_wrist_yaw_link"]`（`config.py:168-170`）。

**参考侧**（`examples/run_mjwp.py:1016-1038`，`ref_fk` + `target_uses_eef_offset=True`，E143/E148 等 manifest 均设 True）：
```python
for ei, hid in enumerate(config.hand_approach_body_ids):
    hand_pos = mj_data_ref.xpos[hid]                 # wrist_yaw_link 原点
    if config.contact_hdmi_target_uses_eef_offset:
        ...
        contact_delta = hand_rot.apply(eef_offset_np)   # R·[0.05,0,0]
        hand_pos = hand_pos + contact_delta
    target_np[t, ei] = obj_mat.T @ (hand_pos - obj_pos) # 存为物体局部系单点
```

**机器人侧**（`spider/simulators/mjwp.py:1031-1044`）：
```python
eef_pos  = xpos_sim[:, bid]            # 同一个 wrist_yaw_link 原点
eef_quat = xquat_sim[:, bid]
contact_point = eef_pos + _lf_quat_apply(eef_quat, eef_offset...)  # 同一个 R·[0.05,0,0]
dist = (target_world - contact_point).norm(dim=-1)
pos_rew = torch.exp(-dist / config.contact_hdmi_sigma)
```

→ **参考与机器人用同一 body、同一 `eef_offset` 字段、同一 FK+offset 公式**，天然对齐。`contact_hdmi` 奖励是**纯点对点 exp 核距离**，全程不读任何 geom / mesh / SDF。

佐证（同一 wrist+offset 点被多处复用，全非 mesh 表面）：
- `hand_approach_rew`（`mjwp.py:923-946`）：wrist body xpos vs 箱半轴。
- leg-object gate（`mjwp.py:1396-1432`）、carry_corridor（`mjwp.py:1643-1676`）：同 wrist+offset 点。
- `_geom_box_sdf_min`（`mjwp.py:68-130`）：即便 `rubber_hull` 变体把 `lh/rh` 变成 mesh，这个解析 SDF 仍把 geom 退化成**包围球**（读 `geom_size[gid,0]` 当半径），不读真实凸壳面。
- 代码里**不存在** `hand_object_contact` / `hand_object_distance` 奖励项；唯一读 geom 的是 `hand_object_deep_penalty`（默认 scale=0，且也走包围球 SDF）。

---

## §3 几何错位：锚点埋在橡胶手根部

所有坐标在 `wrist_yaw_link` 局部系、沿手指向（x 轴），已实测：

| 物体 | wrist-x | 出处 |
|---|---:|---|
| **被跟踪锚点** eef_offset | **0.050** | `config.py:195-196` |
| 橡胶手 mesh 根部（visual geom pos） | 0.0415 | `scene.xml:362` |
| palm / contact_left_hand site | 0.080 | `scene.xml:364,370` |
| 默认球碰撞体（中心 0.10，r=0.05） | [0.05, 0.15] | `scene.xml:30-32` |
| **橡胶手 mesh** bbox（含手指） | **[0.0415, 0.173]** | 实测 STL bbox x∈[0,0.132] + 0.0415 |

ASCII 侧视（手沿 +x 伸向箱子）：

```
wrist                      锚点          掌心           指尖
原点                    eef_offset      palm-site      mesh末端
0.00 ────────────────────●0.05 ──┬───────●0.08 ────────────●0.173 ──▶ x(手指方向)
                          │       │
            橡胶手mesh ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒  [0.0415 .. 0.173]
                          │       │
                          │       └── 默认球碰撞体中心 0.10，覆盖[0.05,0.15]
                          │
                CEM 把这个●(0.05)拽到箱面 ──────────────────────┐
                                                                ▼
                                                          ┌──────────┐
   当 0.05 贴到箱面时，掌(0.08)和指尖(0.17)              │   箱子    │
   已经捅进箱内 3~12cm ──────────────────────────────────│▶▶▶▶▶▶    │
                                                          └──────────┘
```

锚点 0.05 落在橡胶手**根部**——掌心(0.08)、指尖(0.173)都在它前面。CEM 优化把这个埋在手内部的点拽到箱面，真实手的掌/指必然插进箱子。这是几何上不可避免的：奖励驱动一个内部点贴箱 ⇒ 外壳过箱。

---

## §4 与 E147-149 数据互证

E149 `relaxed8_valid_like`（8 case）三方对比（`results/E149/.../e149_method_summary.tsv`，git HEAD d91a6c3）：

| 方法 | 手物接触 | 5cm | 10cm | 手物穿透 | 深穿透2cm | 腿穿透 |
|---|---:|---:|---:|---:|---:|---:|
| OmniRetarget（橡胶参考回放） | 0.5957 | 0.6924 | 0.7140 | 0.5957 | 0.2860 | 0.0207 |
| sphere Spider（球CEM） | 0.4635 | 0.6449 | 0.6847 | 0.4635 | 0.0155 | 0.0564 |
| rubber Spider（橡胶CEM） | 0.3070 | 0.6592 | 0.6940 | 0.2777 | 0.0084 | 0.0953 |

rubber − sphere 差（relaxed8）：手物接触 **−0.1565**、手物穿透 **−0.1858**、5cm +0.0144、10cm +0.0093、腿穿透 +0.0389。
clean6_primary（6 case）更尖锐：手物接触 −0.1980、手物穿透 −0.2293、腿穿透 +0.0507。

**机制预测 vs 实测完全一致**：
- 换橡胶手 → 手物穿透↓（0.46→0.28）✓：埋点被 `deep_penalty` 往外掰 + mesh 真实体积让浅穿统计变化。
- 同时 手物接触↓（0.46→0.31）✓：奖励不读手面，且把埋点贴箱反而让手面整体姿态偏离贴合。
- 5cm/10cm 仅微升（+0.01 级）✓：near-band 对锚点位置不敏感，证明问题不在"远近"而在"锚点位置定义"。
- 腿穿透↑ ✓：手被往外掰后整体姿态代偿。

注：此处"手物接触"列在 E149 表里数值上等于"手物穿透"（同一 SDF≤0 口径，呼应 `[[project_contact_metric_pitfall]]` 的 contact==penetration 退化）；near-band 5cm/10cm 才是独立的接触贴合指标。

---

## §5 两条对齐路线（仅陈述，待拍板再立 E 号 + plan）

因为参考与机器人共享同一个 `config.contact_hdmi_eef_offset`，**改这一个字段两侧自动同步平移**，对称性天然保持。

### 路线 A：config-only 重锚定（改动最小）
- 把 `contact_hdmi_eef_offset` 从 `[0.05,0,0]` 挪到橡胶手实际接触区（掌心 site 0.08，或由 mesh 接触面几何反推一个 x 值）。
- A/B：8 个 benchmark case × {0.05, 0.08, ...} × 橡胶手 CEM；看 near-band 接触↑、穿透是否同时↓。
- **代价**：纯 config，无代码改动，一次扫参。**风险**：单点平移仍是"点对点"，无法表达手面随箱面朝向的贴合；偏移过大可能让奖励点跑到手外。**对称性**：参考/机器人同字段，自动一致。

### 路线 B：真·mesh 表面接触（改 reward 路径）
- 把 `contact_point`（`mjwp.py:1037`）从"wrist+常量"换成"橡胶手碰撞 geom → 箱最近表面点 / SDF"；参考侧 target（`run_mjwp.py:1016-1038`）做匹配定义；并修 `_geom_box_sdf_min`（`mjwp.py:68-130`）当前把 mesh 退化成包围球的近似，使其尊重凸壳。
- 前置依赖：`rubber_hull` 碰撞变体（E147 已建，`patch_hand_collision.py`）。
- **代价**：改 reward 核心路径 + SDF helper，需回归测试。**风险**：mesh-box 最近点每帧求解开销、可微性/CEM 采样稳定性。**对称性**：参考与机器人都要改成读各自手面，需保证口径一致。

两条路线可叠加（先 A 拿低成本增益，再视情况上 B）。**本轮不执行**——后续若推进，单独立 E 号 + plan + per-exp snapshot（experiment.md §7）。

---

## 范围约束（本报告已遵守）
- 未改任何代码（SPIDER / OmniRetarget / 管线），未跑 CEM / OmniRetarget，未改 scene/config。
- 本文为分析报告，非 E-实验。
- 所有代码论断可点回 `file:line`；几何数字来自 `scene.xml` + 实测 STL bbox；E149 数字来自 `e149_method_summary.tsv`。

## 关键引用
- `spider/config.py:186-217`（contact_hdmi 字段，eef_offset=[0.05,0,0]）、`:168-170`（wrist body 名）
- `spider/simulators/mjwp.py:1031-1044`（机器人侧 contact_point/dist）、`:68-130`（包围球 SDF）、`:923-946`/`:1396-1432`/`:1643-1676`（同点复用）
- `examples/run_mjwp.py:1016-1038`（参考侧 ref_fk target）
- `spider/assets/robots/unitree_g1/scene.xml:30-32`（球碰撞体默认）、`:356-371`（wrist_yaw geoms/sites）
- `holosoma_retargeting/models/g1/g1_29dof_w_box004.xml:263`（OmniRetarget 橡胶手碰撞 geom）、`interaction_mesh_retargeter.py:138-146`（加载链）、`workspace/core4d/data_preprocess/pipeline.sh:315-320`
- `workspace/core4d/results/E149/e143_clean_rubber_benchmark/e149_method_summary.tsv`、`e149_diff_summary.tsv`
- `tmp/OmniRetarget_gemo.md`（OmniRetarget 手 geom 只单向防穿）
