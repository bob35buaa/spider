# 02 — 接触面 (Face) 选择 与 接触语义 审计

范围：codebase 中所有「为每只手选一个 box 面 (±x/±y/±z) 作为 anchor face」以及「定义某个 3D 点作为 contact 几何代理」的位置，覆盖 SPIDER 核心 (`spider/`)、用户工作区 (`workspace/core4d_collab_retarget/`、`workspace/core4d/`)、以及 holosoma `data_construction_v2` pipeline。

read-only。一个辅助验证脚本：`workspace/exp_diagnostic_v2/scripts/check_face_selection.py`。

本次更新（2026-05-30）包含两个新发现，与最初版本相比有重要修订：
- **B1 影响范围下调**：grep 全仓库确认 `support_proxy_enabled: true` 只在 E006-E008 的 `core4d_collab_*` override 出现，E082-E097 阶段所有 box021/box023/box025/box004/Box026 的 config 都没启用 support weld。B1 在 E082-E097 阶段是「统计污染」而非「失败驱动因素」。
- **新增 B6（接触语义错位 / wrist ≠ contact）**：这是比 B1 更深一层、贯穿整条 OmniRetarget → SPIDER pipeline 的根本性 bug，单独成节（§4）。

---

## 1. 面选择 / 接触代理点 发生的位置

### SPIDER 核心（无 face 概念，但有 contact 代理点）

- `spider/preprocess/detect_contact.py:355-414` —— 记录 raw 指尖接触位置（5 fingertip per hand），**不挑面**。
- `spider/process_datasets/core4d.py:128-143` —— 写 `contact_pos[T,2,3]` 为 IK 后的 palm site FK 位置，`contact[T,2]` 为 distance < 0.15 m。**这是下游所有"raw contact" audit 真正读到的字段，但它不是 raw mocap，是 IK FK palm**（详 §3 B4）。
- `spider/preprocess/generate_xml.py` —— 加碰撞 box / 关节，**不用 face label**。
- `spider/simulators/mjwp.py:773-1086` —— reward 消费 per-frame 3D target 点 (`contact_pos_ref`, `contact_target_dynamic`)，**reward 层没有 face 概念**。
- `examples/run_mjwp.py:920-1009` —— `contact_target_per_frame` 从三个分支之一构造：`external`（预算好的 npz, 如 E085/E094 输出）、`ref_fk`（IK FK wrist 投到 object local）、static `contact_hdmi_target_left/right`。**三条都消费已是 3D 的点，运行时不挑面**。

### Legacy anchor pipeline（xy-only face 选择 —— 含 bug，B1/B2/B3 在这里）

- `workspace/core4d_collab_retarget/scripts/E017/audit_select_anchors.py:48` 定义 `FACE_ORDER = ["+x","-x","+y","-y"]` —— **±z 被显式排除**。
- `audit_select_anchors.py:192-202` —— `face_label()` 用 `np.argmax(|point[:2]|)`、`'xy'[axis]`。**z 被静默丢弃**。
- `audit_select_anchors.py:199-202, 210-222` —— `_face_axis_sign`、`_clip_anchor`、`_snap_to_face` 只在两个水平轴上操作；z 被强压在 ≤ `0.65·half[2]`。
- `workspace/core4d_collab_retarget/scripts/E018b/generate_e018b_assets.py:136-153` —— `_face_axis_sign` / `face_label` / `canonical_anchor` 全 xy-only；`canonical_anchor` 把 z 用固定 `z_frac * half[2]` 写死。
- `workspace/core4d_collab_retarget/scripts/E020_audit/audit_common.py:219-234` —— `_face_label` 和 `_face_counts` xy-only（只 4 个侧面）。
- `workspace/core4d_collab_retarget/scripts/E028/build_e028_manifest.py` / `scripts/E028/generate_e028_assets.py` —— 继承 `_face_axis_sign` 语义；构建 full CEM 用的 canonical 侧面 support proxy。
- `workspace/core4d_collab_retarget/scripts/E028b/build_e028b_manifest.py:76-80` —— `_face_label` 是全 3D（OK），但
- `…/E028b/build_e028b_manifest.py:164-174` —— `_project_to_face` 硬编码 `axis = 0 if face.endswith("x") else 1`。如果上游 `anchor_face` 是 `+z`/`-z`（修后的 `_face_label` 会产出），投影就把它当成 `+y`/`-y`，写到错的面上。
- `workspace/core4d_collab_retarget/scripts/E029/generate_e029_d6_assets.py:166` —— `side = row["anchor_face"] if row["anchor_face"] in common.SIDE_FACES else "+x"`，即任何 ±z anchor 静默退化为 `+x`。

### 现代 / E093+ 路径（全 3D —— 正确）

- `workspace/core4d/scripts/E093/audit_contact_geometry.py:257-376` —— 全 3D `argmax(|local|/half)`，`'xyz'[axis]`；E094/E095 的参考实现。
- `workspace/core4d/scripts/E094/build_handbox_target_projection.py:176-198` —— 同样全 3D，按"local-world-up 对齐度"区分 support face。
- `workspace/exp_diagnostic/scripts/object_local_contact.py:93-105`，`fk_wrist_in_object_frame.py:97-111`，`wrist_repair_top_face.py` —— 全 3D。
- `workspace/v3/data_construction/scripts/check_d005_handbox_surface_gate.py:245-264` —— 按"到边界的最小 gap"覆盖 6 面（OK）。
- `workspace/v3/data_construction/scripts/check_bucket_surface_semantics.py:145-169` —— 径向 vs 法向分类（OK，bucket-specific）。
- `workspace/core4d/scripts/E097/mine_feature_based_candidates.py` —— 纯 metadata mining，不挑面。

### OmniRetarget（无 face 概念，无指尖单独信号）

- `holosoma/src/holosoma_retargeting/holosoma_retargeting/examples/robot_retarget.py:355-358` —— 只有 `surface_weight_threshold`（climbing 风格按 z 偏置采样）。
- `holosoma/workspace/pipeline/retarget_core4d_obj_interaction.py` —— runner only。
- **JOINTS_MAPPINGS** (`data_type.py:287-289`) —— `L_Wrist → left_wrist_yaw_link`、`L_Fingertip_Center → left_rubber_hand_link`。L_Fingertip_Center 只在 `core4d_v2 + --include_fingertip_centers` 时存在，**D003 production 默认 OFF**（详 §4 B6）。

### 可视化渲染脚本

- `workspace/exp_diagnostic/scripts/render_wrist_overlay.py` —— 渲染 box 和 wrist 散点的 3 个视图（XY/XZ/YZ）。无 face label，直接用 3D 位置（忠实）。
- `workspace/exp_diagnostic_v2/findings/` —— 暂无自有渲染脚本。
- `holosoma/workspace/v3/data_construction_v2/visualizations/raw_contact/*.png`、`…/d005b/*_object_local_overlay.png` —— 由外部 pipeline 产出；**生成脚本在两个 repo 里都不存在**，只有 PNG 输出和 manifest，不可复核。

---

## 2. 算法摘要 (按位置)

| 位置 | 是否 per-frame | 考虑的面轴 | 平局规则 |
|---|---|---|---|
| `E017.face_label` / `_face_stats` | 轨迹聚合（FK palm site 点云在 object local）| **只 xy** | top count → FACE_ORDER 序 |
| `E018b.canonical_anchor` | 轨迹级 | **只 xy**；z 固定 `z_frac·half[2]` | – |
| `E020_audit._face_label` | per point | **只 xy** | – |
| `E028b._face_label` | per point | 全 3D | – |
| `E028b._project_to_face` | 轨迹级 | **只 xy**，z 静默被映射到 y | – |
| `E028b._face_stats` | 聚合 | 全 3D（top/side/z 分类）| 先 count，后字母序 |
| `E029.generate_e029_d6_assets` | 轨迹级 | 仅侧面 (xy)；fallback 到 `+x` | – |
| `E093.face_stats` | per frame | 全 3D | – |
| `E094.face_stats_tol` | per frame | 全 3D；按"对齐 world up"标 support | – |
| `D005 handbox_surface_gate.face_label` | per point | 全 6 面，按 min gap to bounds | edge flag 当 ≥2 轴贴边 |
| `process_datasets/core4d.py` | per frame | 无 —— 仅 FK 距离阈值 | – |

---

## 3. Bugs / 不一致（B1-B5，老 anchor pipeline）

### B1 — Anchor 选择器屏蔽顶/底面 (xy-only argmax)

- **位置**：`core4d_collab_retarget/scripts/E017/audit_select_anchors.py:192-196` (`face_label`)、`:48` (`FACE_ORDER`)，被 E018/E018b/E020/E028 继承。
- **证据**（5 个 box021 D003 case 的真 top face 是 ±z）：

  | case | hand | 全 3D top face | xy-only 选出的 face | z 面占比 |
  |---|---|---|---|---:|
  | 20231018_029_p2 | R | +z (49/75) | -x (64) | 65% |
  | 20231018_030_p1 | L | +z (64/88) | +y (84) | 73% |
  | 20231018_030_p1 | R | +z (59/88) | +x (65) | 67% |
  | 20231020_019_p2 | L | +z (60/102) | -x (66) | 59% |
  | 20231020_019_p2 | R | +z (60/102) | +x (61) | 59% |
  | 20231020_020_p2 | R | +z (66/87) | +x (68) | 76% |
  | box023_person2 | L | +z (70/136) | +y (73) | **51%（边界）** |
  | box023_person1 | R | +y (62) but -z 60 | +y (70) | 44% |

  （从 `trajectory_kinematic.npz` 的 `contact_pos` 投到 object local 重算。脚本：`workspace/exp_diagnostic_v2/scripts/check_face_selection.py`。）

  Cross-check：`E028/manifest.tsv` 早已报 030_p1 `anchor_top_face=+z` (z_face_frac=0.70)、031_p2 (0.56)，但仍把 `anchor_face` 选为 `+y`。

- **严重程度**（**已修正，比初稿降级**）：
  - **针对 E006-E028 weld-based 实验**：**wrong-target**。`support_weld_anchor` 焊在错的面，canonical anchor 落在人手抓握范围内，hand-to-anchor reward 把 IK 推向错的面。
  - **针对 E082-E097（无 support weld）**：影响**降级**为 (a) 让 `anchor_face_review=true` 这个 informational flag 失去诊断价值；(b) 让所有"接触面统计"（含 v1 诊断 §3.2/§3.5）给出错误的面解读。**但 INSIDE box %、wrist-vs-pelvis 高度差这些纯几何量不受影响**——它们无 face 概念。
  - 因此：**E082-E088 box021 D003 失败的主因不是 B1**。v1 H1 关于"几何不可行"的方向正确，但 v1 §3.2/§3.5 描述的"主面 = -x、左手在 -x 面外 8 cm"具体画面是被 B1 误读出来的；真实图像（按全 3D 投票）是"主面 = local +z，但局部 +z 在 box021 18029_p2 因 quat 90° X 旋转后对应世界水平方向"——详 §5。
  - **9/13 `anchor_face_review=true` 的根因依然是 B1**，但这个 flag 是 informational，case 仍照常进 full CEM（见 B5）。

### B2 — E028b refit 把 ±z anchor 静默投到 ±y
- **位置**：`core4d_collab_retarget/scripts/E028b/build_e028b_manifest.py:164-174`。
- **证据**：`axis = 0 if face.endswith("x") else 1`。上游全 3D `_face_label` 能产出 `face=+z`，`_project_to_face` 会固定 `point[1] = ±half[1]`，写出 `+y`/`-y` 锚点，但 manifest 仍记录 `anchor_face=+z`。
- **严重程度**：wrong-target（一旦触发）。**目前因为上游 B1 一直输 ±x/±y，所以未触发**。修 B1 之后立即变 wrong-target，必须同步修。

### B3 — E029 D6 asset writer 硬回退 ±z 为 `+x`
- **位置**：`core4d_collab_retarget/scripts/E029/generate_e029_d6_assets.py:166`。
- **证据**：`side = row["anchor_face"] if row["anchor_face"] in common.SIDE_FACES else "+x"`。`common.SIDE_FACES = ("+x","-x","+y","-y")`。
- **严重程度**：misleading / wrong-target（针对任何未来 z-face case）。与 B2 同样：B1 修后立即生效。

### B4 — `trajectory_kinematic.npz` 中 `contact_pos` 是 FK，不是 raw mocap
- **位置**：`spider/process_datasets/core4d.py:128-144`；下游所有 E0NN anchor audit 把它当 "raw human contact" 读。
- **证据**：shape `(T, 2, 3)` per case，但 `detect_contact.py` 本来会写 `(T, 10, 3)`（5 指尖 × 2 手）。`core4d.py` 写的是 `site_xpos[contact_site_ids]`，site 是 IK 后机器人的 `left_palm`/`right_palm`（在 IK 后人 pose 上），阈值 `<0.15 m`。这是 IK FK，不是 raw 人手。
- **严重程度**：misleading。E017/E018b/E028/E028b 里所有"raw contact face 统计"实际算的是 G1 wrist FK，不是人手。**v1 §3.2/§3.5 也踩了这个坑**——表中"L 主面 = -x、signed dist +0.079m"是 G1 IK wrist 在 box local 的投影，不是人指尖位置。这与 B6 紧密关联（详 §4）。

### B5 — `anchor_face_review=true` flag 仅 informational
- **位置**：`E028/manifest.tsv` 中标了 flag 的行被直接送入 full CEM。
- **证据**：`workspace/core4d_collab_retarget/log/29_E028_d003_box021_…_results.md:90,216` ("9 anchor_face_review=true, allowed to enter smoke/full but not clean")。
- **严重程度**：cosmetic + procedural。叠加 B1 让错面 case 在 E082-E088 时代继续消耗 CEM 预算（虽然 E082-E088 不用 weld，但 manifest 的过滤还是基于这个 flag 做决策）。

### 没发现 bug 的地方

- `spider/preprocess/detect_contact.py`（无 face 逻辑）。
- `spider/simulators/mjwp.py` reward 栈（消费 3D 点，对各轴对称）。
- `examples/run_mjwp.py` `contact_hdmi_*` target wiring（消费 3D 点）。
- `E093.audit_contact_geometry` 和 `E094.build_handbox_target_projection`（全 3D, world-up aware）。
- `data_construction_v2.check_d005_handbox_surface_gate`（全 6 面）。
- OmniRetarget 重定向器 (无 per-face 概念；仅 climbing 任务的 z-加权采样)。

---

## 4. 新增 B6 — 接触语义错位（wrist ≠ contact）

这是比 B1 更深一层的根本性 bug，**贯穿整条 OmniRetarget → SPIDER pipeline 的所有阶段**。B1/B2/B3 是局部算法错误（修一个函数就能止血），B6 是 **数据建模选择上的系统性损失**——pipeline 中**没有任何一个位置真正代表"指尖在哪里"**。

### 4.1 数据流与三层信息损失

```
人 (真实接触：5 指尖 + palm 分别压在 box 不同面 / 棱 / 角)
   ↓ (CORE4D 用 SMPL-X 标准 22-joint body model + 手部 model 捕获)
SMPL-X 22 joint                ← rows 20/21 仅保留 L_Wrist / R_Wrist
+ 5 指尖 (index3/middle3/pinky3/ring3/thumb3, 在 SMPL-X 手部 model 里)
   ↓ STAGE A: convert_core4d_to_omniretarget.py
global_joint_positions[:, 0:22, :]       ← body wrist
+ (可选) rows 22/23 = L/R_Fingertip_Center = 5 指尖均值 (单点)
     仅当 core4d_v2 格式 + --include_fingertip_centers flag 时存在
     **D003 production 默认 OFF**
     (引证：08_B_path_omniretarget_audit.md §6 + E090 log)
   ↓ JOINTS_MAPPINGS (data_type.py:287-289)
"L_Wrist"           → left_wrist_yaw_link        # 永远启用
"L_Fingertip_Center"→ left_rubber_hand_link      # 仅在 v2+flag 时启用
   ↓ STAGE B: InteractionMeshRetargeter (CVXPY DiffIK)
G1 qpos (29 关节, **手部没有任何自由度**)
   ↓ STAGE E: spider/process_datasets/core4d.py:128
contact_pos[T,2,3] = mj_data.site_xpos[contact_site_ids]
   site 挂在 G1 的 left_rubber_hand_link 某个 palm offset
```

**信息损失累加**：

1. **人→SMPL-X 22-joint**：5 指尖在 body model 转换中被丢；只剩 wrist。
2. **SMPL-X → OmniRetarget**：即便 `--include_fingertip_centers` 开了，**5 指尖也被坍缩成 1 个均值点**，捕捉不到「拇指与四指对夹」「四指弯过 +z 顶边勾到 +y 侧」这种用 5 个独立点才能描述的握姿。
3. **OmniRetarget → G1**：G1 标准 URDF 的 `left_rubber_hand_link` 是**一个无 DoF 的球**，没有手指关节。无论 mocap 多么精细，到 G1 都坍缩成"一个球贴 box 某点"。

**总效应**：整条 pipeline 中**没有任何位置真正代表"指尖在哪里"**。下游每一个把 `contact_pos` 解读为"人手接触位置"的算法（B1/B2 的 face_label、E093 的 raw mean target、E094 的 adaptive_support、SPIDER reward 的 `ref_fk wrist + 5 cm`）实际拿到的都是"假装 wrist + palm offset 就是 contact"的代理量。

### 4.2 量化证据

E093 (`workspace/core4d/log/115_E093_contact_target_geometry_audit_results.md` §"关键指标")：

| 量 | 数值 | 解读 |
|---|---|---|
| `wrist+5cm → raw mean` 距离 | **≥ 20 cm 全 case** | 5 cm palm offset 完全无法表达真实接触 |
| Box026 / Box025 同上 | **49-64 cm** | 大物体 / 侧抓 case 偏差更大 |
| Sphere-surface p90 gap | > 10 cm 全 case；Box026/Box025 45-61 cm | 即便换用 sphere proxy 也偏差 10+ cm |
| Handbox 是最佳代理 | 14/14 best | 但 p90 仍 16-59 cm |

E093 当时把它当作"target source 偏差"做统计；按 B6 的视角理解，**它的本质是"用 wrist 假装代表 contact"的系统性偏差**。E093 选 handbox 作为最佳代理是局部更好，但仍是在 wrist + 固定 offset 框架内打补丁，没有解决"接触语义在 wrist 这一层就已经丢失"的根因。

### 4.3 对 v1 H1 的修正解读

v1 §3.3 报 box021 18029_p2 R wrist FK **33% 帧 INSIDE box**。按 B6 视角理解：

- **真实物理**：人在搬 box021 时，5 指尖很可能正贴在 +y 侧外、palm 也贴在 +y 面，wrist 在腕关节那里**被手腕反向弯曲带到 box 几何中心方向**——anatomically wrist 比 palm 往内陷 8-10 cm。
- **FK 结果**：G1 IK 跟 wrist 走，加上 +5 cm palm offset 也补不回这个偏移；FK 落点就在 box 内部。
- **视觉上"INSIDE"**：但**实际人手并没穿模**——指尖在 +y 面外，palm 贴 +y 面，只有"如果 G1 有 wrist 但没手"这种几何近似下 FK 才落到 box 内。
- **CEM 看到的**：reward target 是这个 FK 点，所以 CEM 必须把 G1 wrist 推进 box 体内才能拿到 contact reward；这是物理不可行的，必然走 reward hacking（上半身贴 box、抛弃手部 contact）。

**所以 v1 H1 "几何不可达" 主张方向对、但粒度太粗**。真正的根本错位发生在 STAGE A（wrist 抽样代替 contact），IK FK INSIDE 是次生现象。从修复角度看：
- 在 SPIDER reward 端继续摇 wrist+offset 是错的（E085-E088 已证）；
- 在 SPIDER reward 端切到 `external` target + adaptive_support 投影是局部缓解（E094 box004 work，但 Box026 反退化）；
- **真正的修复在 STAGE A 输入端补上 fingertip 信息**（启用 `--include_fingertip_centers` 加上 5 指尖独立 voting），见 §6 推荐。

### 4.4 二阶坑：box021 quat 90° X 让 "world-up = top" 假设失效

这是 B6 的几何变形版，单独提出来因为它解释了 E090 的反退化模式。

**事实**：
- box021 18029_p2 `obj quat = 90° around +X` (v1 §3.2)；half-extents = (0.16, 0.21, **0.265**)，最长轴是 local +z。
- 旋转 90° around +X 后：**local +z → world −y（水平方向）；local +y → world +z（向上）；local +x → world +x**。
- §3 B1 实测表里 box021 R wrist "主接触面 = local +z" (按 object local frame 投票)。
- 翻译到世界：**接触在 box 的某个水平长侧面，不是顶面**。

**因果链**（解释 E090 失败）：

- v1 §6 / E090 实施的 "topface-preIK"：把 contact target 投到"box 顶面 + 5 cm"。
- 实现里"box 顶面"是按 **object local +z + half_z** 取——对 quat = identity 的对象等价于世界顶面，对 box021 quat 90° X 等价于**世界 -y 方向 + half_z**。
- 也就是说，E090 把 contact target **强行搬到了 box 的水平长侧面**——但人手在该 case 真实接触面也正好是 local +z (= world 水平面)，所以投影方向**没把人推得更离谱，但也完全没修对"world-up"** —— 该 case 真正的世界向上面是 local +y，contact 本来就不在那里。
- **E090 S1 full 失败模式（safety 全 0% 但 pelvis 0.134 m）的几何解释**：IK 试图把 G1 推到"伸手到该水平长侧面"，但 ref motion 是"蹲箱旁、按局部 +z 方向 = 按世界水平面"，姿态对不上（人是侧着推，机器人 IK 解出来是俯身够）；reward 必须跟 ref motion，pelvis 只能往下塌。
- **Box025 反退化**（E090 `inside 48.1/51.9%`）也是同理：Box025 在 0.31 m 高凳上，人侧抓 box 侧壁（+y 或 -y），world-up 投影把 contact 搬到 box 顶面 → IK 当然不可行。
- **结论**：world-up = top 假设只对 quat ≈ identity 的对象成立。box021 (quat 90° X) 和 Box025 (侧抓 intent) 都不成立。任何"投到 box 顶面"的启发式都需要根据 **raw mocap 的实际接触面**决定投影方向。

这正是 Angle 8（用 raw mocap 5 指尖投票打面标签）应该驱动 Angle 2（wrist 投影）的几何依据，见 `findings/03_omniretarget_hardbone_angles.md` §3 Top-1。

### 4.5 与 B4 的区别

B4 说 `contact_pos` **是 FK 不是 raw mocap**——但下游算法假定它是 raw mocap。修法是"rename 或同时存两路"。

B6 说**整个 pipeline 都没有真正代表 contact 的位置**——即使 raw mocap 都被 SMPL-X 22-joint 截断成 wrist。修法是改 STAGE A 数据 schema，让 5 指尖真正进入下游。

B4 是符号学错误（取错变量），B6 是建模选择错误（变量本身就不够）。

---

## 5. 可视化忠实度

| 渲染物 | 是否与代码选中的面一致？ | 备注 |
|---|---|---|
| `workspace/exp_diagnostic/findings/04_overlay_box021_*.png` (3 视图)| 是 | 在 object local 渲染 raw `contact_pos` 和 FK wrist；无面标签，无法 mismatch；box021 显示 wrists 在 box 内——对存的 npz 忠实，**但 npz 本身受 B4/B6 污染**。|
| `workspace/exp_diagnostic/findings/04_overlay_box023_p2_OK.png` 等 | 是 | 同上渲染器。 |
| `holosoma/.../data_construction_v2/visualizations/raw_contact/*.png` | **不可复核** | 生成脚本在两个 repo 都不存在。每张 PNG 是 per-hand object-local；如果标了面，标签遵循生成时的约定（未知）。无 embedded provenance。|
| `holosoma/.../d005b/*_object_local_overlay.png` | 可能是 | 对应 gate (`check_d005_handbox_surface_gate.py`) 用全 6 面，标签应一致。但 overlay 渲染脚本不在 `v3/data_construction/scripts/` 里，未直接验证。 |
| `holosoma/.../dashboard/raw_contact_*.png` | n/a | 仅聚合 count，无 per-face decision。 |
| E028 anchor 可视化 `core4d_collab_retarget/results/E028/anchor_visual/` | 部分 | 渲染 E018b/E028 选中的 support proxy 点（xy-only）。**渲染对代码忠实，但代码选错（B1）**，所以 PNG 跟代码一致 ↔ 跟真相不一致，肉眼复核看不出。|

新增风险（来自 B6）：**所有现存的"face 可视化"都画的是 IK FK palm site 的面分布，不是人指尖的面分布**。即便把 B1 修成全 3D，可视化展示出来的面也是"G1 假设无手时该贴哪个面"，而不是"人真正按在哪个面"。要让可视化恢复诊断价值，需要在 PNG 上同时叠加 raw 5 指尖位置（在 STAGE A enable fingertip_centers 之后才能拿到）。

---

## 6. 推荐修复（按优先级）

### 立即（< 1 工程日，止血）

1. **修 B1**（xy-only argmax）。把 `E017.audit_select_anchors.face_label` / `FACE_ORDER` 改全 3D 加 ±z；同样修 `E018b.face_label`、`_face_axis_sign`、`canonical_anchor`、`E020_audit._face_label`。**同步修 B2 (`_project_to_face`) 和 B3 (E029 fallback)，否则 B1 修后立即触发 B2/B3 的 wrong-target**。重跑 E017→E018b anchor refit 所有 box021/box023 D003 case。预期：9/13 `anchor_face_review=true` 翻转，support_proxy_weld 焊到 +z（针对将来仍可能用 weld 的 case）。

2. **写一行 deprecation warning 到 `core4d.py:128`**：标注 `contact_pos` 是 IK FK palm，不是 raw mocap；下游算法不应当作 raw 读取。同步在 v1 诊断报告 §3.2/§3.5 加一条 errata 引用 B4/B6。

3. **加一个 pelvis-collapse detector 到 CEM gate**：`pelvis_min_world_z < 0.40` 或 `pelvis_pitch > 60°` 直接 FAIL，否则 E094 C2 "趴箱"那类失败模式仍会被错判为 "WORK"。

### 短期（1-2 工程日，根治 B6）

4. **OmniRetarget 端 D003 production 永久启用 `--include_fingertip_centers` + `core4d_v2` 格式**（零开发成本的 flag 切换；先在 box021 D003 13 case 上验证不影响 box023/box004 守门）。

5. **face 决策端从 wrist 改成 5 指尖投票**：从 raw CORE4D `body_pose + hand_pose` 重算每只手 5 指尖在 object local 的位置；per-frame 多数票决定接触面；signed_distance < threshold 的指尖才投票（悬空指尖不污染）；允许 corner/edge label（≥2 个面票数接近）。

6. **contact reward target 从 wrist+5cm 改成 fingertip-vote-face + 投票面上最贴近接触的指尖位置**。E094 `adaptive_support` 是这条路线的 world-up 版近似，应推广到任意 face；新版本由 §5 fingertip 投票驱动方向，不再假设 world-up。

### 中期（结构整理）

7. **PNG 上加面标签**，并把 `data_construction_v2/raw_contact/` 与 `d005b/*_object_local_overlay.png` 的生成脚本找出来 commit 进 repo；任何"靠肉眼复核"的诊断证据链都必须脚本可复现。如可，同时叠加 raw 5 指尖散点，让 face 与 contact 的真实分布在同一张图上可见。

8. **统一 face 逻辑到一个 helper**。codebase 中至少 6 处不同的 `face_label` 定义，至少 3 种轴集约定（xy-only / 全 xyz / gap-to-bounds）。统一到 `spider/math.py`（或 `workspace/exp_diagnostic_v2/scripts/face.py`），所有 caller import 同一份。同时把"face 决策用 5 指尖投票"内嵌进这个 helper。

9. **tighten gating on `anchor_face_review=true`**（B5）：标了 flag 的 case 在 anchor refit 重跑之前 hard block 出 full CEM 队列。
