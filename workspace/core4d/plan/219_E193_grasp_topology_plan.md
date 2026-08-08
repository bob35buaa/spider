# E193 实验计划：抓取拓扑是否解释大箱手物穿透（机制 (c)）

_Core4D · Phase 56 · 计划态，**待用户批准后才创建脚本 / 跑 CEM** · 承接 [E191](../log/266_E191_object_support_offline_audit_results.md)，与 [E192](218_E192_gate_threshold_size_dependence_plan.md) 正交_

---

## 📋 Context

E191 把「大箱失败」留在三条与尺寸共线的机制上。E192 测 (a) 阈值，本实验测 **(c) 抓取拓扑**：

> **抓取拓扑** = 两只手在物体上形成的接触结构，具体指接触法向能否**互相对置**从而构成承重抓握。
> 小箱能跨着抓（两手落在相对面，法向对顶，靠夹持抗重力）；大箱只能贴在同一个大平面上
> （法向近乎平行、都指向面内），这种排布**只能推、不能提**，CEM 为了凑出法向力就把手往面里压。

E173 自己的头号结论就是这个现象的现象级描述（`log/233:113`）：

> 大箱（box024/box001 ~35%）显著退化，**主因 hand_penetration（手陷大平面）**

E189 又把归因从「体积」改口到拓扑（`log/265:240-245`）：

> 真正决定 PRG 依赖程度的更可能是**抓取姿态/接触拓扑等物体几何细节**，而非单纯尺寸

两次都是**推测，从未测量过**——因为仓库里根本没有任何接触法向/抓握质量的度量（见 §关键事实 3）。

### 关键 insight：box001 vs box024 是一对天然的**体积配平对照**

E191 的全量表提供了一个此前没人用过的对照。两个物体体积几乎相同、**参考轨迹自带的穿透也几乎相同**，
但物理仿真产出的穿透差 1.78 倍：

| | box001 (n=28) | box024 (n=9) | 差异 |
|---|---:|---:|---|
| 体积 | 0.256 m³ | 0.253 m³ | **−1%（配平）** |
| half extents | 0.251 / 0.313 / 0.406 | 0.255 / 0.253 / 0.491 | |
| **长宽比**（最长/次长半轴） | **1.30** | **1.93** | **1.5×** |
| `ref_hand_geom_penetration_frac`（参考自带） | 0.628 | 0.603 | **−4%（配平）** |
| `hand_object_physics_penetration_3mm_frame_frac` | **0.212** | **0.378** | **1.78×** |
| `hand_object_physics_contact_3mm_in_mask_frac`（干净接触） | 0.488 | 0.308 | 0.63× |
| `hand_object_con_dist_min_m` | −0.0107 | −0.0201 | 顶死 −0.020 硬地板 |
| `grip_far_arm_m` | 0.778 | 0.978 | |

**体积和参考穿透都配平了，物理穿透还是差 1.78 倍** → 差异是物理阶段产生的，不是数据自带的，
也不是「大小」本身。剩下的候选就是形状/抓握结构。这是 (c) 目前最强的间接证据，也是本实验的出发点。

补充反例（说明「顶死硬地板」不是充分条件）：box023（小箱）`con_dist_min` 同样是 −0.0200、
硬地板饱和率 0.0113，但穿透只有 0.148。**能陷多深 ≠ 会陷多久**，两者的区别正是拓扑要解释的东西。

### 关键事实（scout 结论，决定本计划怎么设计）

1. **接触法向在本仓库中从未被读取过。** `contact.frame` 只在三处出现，全是 CPU→Warp 的整体状态拷贝
   （`mjwp.py:4640`、`:4943`，`mjwp_eq.py:757,572`），**没有任何一处读它**。
   全仓库 grep `mj_contactForce` / `force_closure` / `grasp_wrench` / `contact_normal` / `opposed`：**零命中**。
   `core_metrics.py` 的接触循环只取 `geom1/geom2/dist`。**(c) 必须先把度量造出来。**

2. **抓握点可以纯配置重定位，不用重跑重定向。** `examples/run_mjwp.py:1024-1069` 的 `external` 分支
   读一个 `(T, 2, 3)` 的物体局部坐标 NPZ：

   ```yaml
   contact_hdmi_dynamic_target: true
   contact_hdmi_target_source: external
   contact_hdmi_target_path: <(T,2,3) npz>
   contact_hdmi_target_uses_eef_offset: false   # external 约定，offset 已烘进点里
   ```

   而 `ref_fk` 分支（当前 box024 用的）只是**照抄人手腕当时在哪**
   （`target_np[t,ei] = obj_mat.T @ (hand_pos - obj_pos)`，`:1099`）——
   **没有任何面感知、没有表面投影、没有对置约束**。这正是拓扑失控的来源。

   现成基础设施：`workspace/core4d/scripts/E098/face_utils.py`（`face_axis_sign` `:32`、`majority_face` `:95`、
   `snap_to_face` `:135`、`project_to_face` `:156`）；先例 `E100/build_fingertip_aware_target.py`
   已经在做「投票面 → snap 到 ±half[axis] → 面内 clip 到 ±(half−1cm)」。

3. **手部碰撞几何这条轴基本被穷尽了，且它不改拓扑。** 只有 `sphere5cm` / `rubber_hull` 两个变体
   （`data_construction_v3/lib/common.py:23-40`），**两者都是每只手一个单块**，没有对置指垫。
   E146→E149 A/B 过四次，结论是 tradeoff 更尖锐而非净胜（E149 clean6：rubber 穿透 0.481→0.252 但
   物理接触 0.481→0.283、腿穿透 0.047→0.098）。**本实验不再动这条轴。**

4. **CoACD 不是 box 的杠杆。** E181 `ASSET_REJECTED`（Gate B 0/18）、E182 全族未过接触门、
   E186 编译 22/22 通过但 R fidelity 卡住、Full 从未启动；且全是 bucket-only，assets 在本机也不存在。
   更关键：box024 的 `object_collision` 就是 mesh AABB 精确等值（E191 已验），**箱子本来就是箱子**，
   凹几何分解对它没有意义。

5. **双手门存在但不含对置概念。** `contact_hdmi_bimanual_*`（`mjwp.py:1657-1686`）与
   `surface_band_bimanual_*`（`:2199-2240`）都只做 `min(left,right)` 和「两手都在带内」，
   **两只手压在同一个大平面上完全满足这两个门**。这解释了为什么现有 gate 对拓扑完全瞎。

---

## 🎯 Claims

分两阶段，Stage 2 **由 Stage 1 结果 gate**。

### Stage 1（零算力，先做）

| # | Claim | 最低证据 |
|---|---|---|
| C1 | 接触法向对置度可被测量，且在既有 141 条 rollout 上有区分度 | 新列 `hand_contact_opposition_mean` 在 10 个物体上的组间极差 ≥ 0.5（满量程 [−1, +1]）|
| C2 | **对置度解释 box001↔box024 的穿透差**（体积与参考穿透已配平） | box024 对置度显著低于 box001，差值 ≥ **0.3**，且 Mann-Whitney p < 0.05（n=9 vs 28）|
| C3 | 对置度是比尺寸更好的穿透预测量 | 141 条 case 级：ρ(对置度, 穿透) 的绝对值 > ρ(最长半轴, 穿透)，且前者 p < 0.01 |
| C4 | 「同面按压」指纹可判定 | box024 的 `hand_face_antipodal_frac`（两手落在对置面的帧占比）显著低于 box004/box023 |

### Stage 2（CEM，**仅当 C2 成立才启动**）

| # | Claim | 最低证据 |
|---|---|---|
| C5 | 把抓握点重定位到对置面能降低穿透 | box024 `hand_object_physics_penetration_3mm_frame_frac` ≤ **0.20**（基线 0.3776），≥7/9 例改善 |
| C6 | 且**不是靠脱手**换来的 | `hand_object_physics_contact_3mm_in_mask_frac` ≥ **0.3079** 且 `contact_in_mask` ≥ 0.75 |
| C7 | 对置度确实被改动了（机制自洽） | 干预臂的 `hand_contact_opposition_mean` 必须显著上升；若穿透降了但对置度没升，归因错误，C5 不成立 |
| C8 | 不破坏参考跟踪 | `track_eef_pos_err_cm_mean` 增幅 ≤ 5 cm，`track_obj_pos_err_cm_mean` 不劣化 > 2 cm |

---

## ⚙️ Stage 1：造度量 + 在既有 rollout 上回答（零 GPU）

### 改动：`core_metrics.py` **纯追加**（照 E191 的 `E191_SUPPORT_FIELDS` 模式）

在既有接触循环（`core_metrics.py:1259-1281`，已经在遍历 `data.contact`）里增读 `con.frame`：

```python
# con.frame 是 3x3 接触坐标系，第一行是法向（由 geom1 指向 geom2）
normal = np.array(con.frame).reshape(3, 3)[0]
if other[0] == lh_gid:
    left_normals.append(normal * sign)   # sign 统一成"由手指向物体"
elif other[0] == rh_gid:
    right_normals.append(normal * sign)
```

新增列 `E193_TOPOLOGY_FIELDS`：

| 列 | 定义 | 语义 |
|---|---|---|
| `hand_contact_opposition_mean` / `_p10` | 两手都有接触的帧上 `−(n_L · n_R)` | **+1 = 完美对置（夹持），−1 = 同向（同面按压）** |
| `hand_contact_normal_sum_norm_mean` | `‖n_L + n_R‖` | 力闭合替代量，0 = 完美对置 |
| `hand_bimanual_contact_frac` | 两手同时有接触的帧占比 | 上面两列的分母 |
| `hand_face_left` / `hand_face_right` | 用 `E098/face_utils.face_label` 给每手接触点打面标签的众数 | 可读的拓扑标签 |
| `hand_face_antipodal_frac` | 两手落在**对置面**的帧占比 | C4 的直接判据 |
| `hand_face_same_frac` | 两手落在**同一面**的帧占比 | 「同面按压」指纹 |

**回归要求**：与 E191 同样标准——既有全部列在 E189 的 43 行上逐值 bit-identical，只多出新列。

### 分析

复用 E191 的 runner（加 `--fields e193` 或直接重跑），在同一批 **141 条 rollout / 10 个物体**上出：

1. 逐物体对置度表（含 box001 vs box024 的配平对照，C2）
2. case 级 Spearman：对置度 vs 穿透，对照 最长半轴 vs 穿透（C3）
3. 与 R018 §9.6③「只跟不抬」指纹交叉：`高 surface + 高穿透 + 低 3mm 硬接触` 的 case 是否正是低对置度的 case

### 产物

`workspace/core4d/results/E193/audit/{e193_topology_audit.tsv, E193_topology_report.md}`

---

## ⚙️ Stage 2：对置面抓握点重定位（**gated on C2**）

### 干预：纯配置，不重跑重定向、不改 scene

为每个 case 生成 `(T,2,3)` 物体局部目标 NPZ，把两手 snap 到**一对对置面**：

```
对每帧：
  1. 取 ref_fk 的两手物体局部位置 (E100 同款起点)
  2. 选面对：在 SIDE_FACES 里选一对对置面 (±axis)，使两手到该对面的总投影距离最小
     —— 即"离哪对面最近就用哪对"，不强行改变人的搬运语义
  3. 左手 snap 到 −axis 面、右手 snap 到 +axis 面（按当前 local 坐标符号分配）
  4. 面内两轴 clip 到 ±(half − 1cm)，避免落在棱上
  5. 若某帧两手本来就在对置面上（已对置），保持原值不动
```

| 臂 | 目标来源 | 需要跑 |
|---|---|---|
| **B0** 基线 | `ref_fk`（现状） | 否，E173/E172 已落盘 |
| **B1** 对置面 snap | `external` + 新 NPZ | box024×9 + box001×9(抽样) + box004×6 = **24 例** |

box001 抽样 9 例（与 box024 等 n，便于配对）：它是体积配平对照，**如果 B1 对 box024 有效而对 box001
无效/无所谓**，就说明干预确实作用在「长宽比 1.93 导致的同面按压」上，而不是泛泛地改善抓握。
box004 6 例做阴性对照（小箱本来就能跨抓，预期几乎不动）。

**冻结不变量**：PRG 开启；`rubber_hull`；CEM `seed=0 / 1024 × 32`；所有 gate/reward 阈值保持 E167A 原值
（**不与 E192 的改动叠加**——两个实验必须能独立归因）。

### 预算

24 条 Full CEM。先 canary（`64 × 4`）6 例：box024 `026_p1 / 027_p2 / 031_p1` + box001 ×1 + box004 ×2。

---

## 📊 必报指标

12 门全部 + E191 的 22 个 support 列 + E193 新增拓扑列。逐物体分别报，**禁止合并跨物体结论**。
Stage 2 用配对统计（McNemar 门级 + bootstrap CI 连续量，种子 0、10000 次）。

Stage 2 基线（B0，已落盘）：

| 指标 | box004 (n=6) | box001 (n=28) | box024 (n=9) |
|---|---:|---:|---:|
| 12 门通过 | 2/6 | 5/28 | 2/9 |
| `hand_object_physics_penetration_3mm_frame_frac` | 0.1469 | 0.2120 | **0.3776** |
| `hand_object_physics_contact_3mm_in_mask_frac` | 0.4507 | 0.4877 | 0.3079 |
| `hand_object_physics_contact_in_mask_frac` | 0.6944 | 0.7886 | 0.8188 |
| `hand_object_con_dist_min_m` | −0.0141 | −0.0107 | **−0.0201** |
| `ref_hand_geom_penetration_frac` | 0.5080 | 0.6284 | 0.6025 |
| `grip_far_arm_m` | 0.4310 | 0.7778 | 0.9782 |

---

## 🛡️ 成功标准与 stop-loss

| 结果 | 判定 |
|---|---|
| C1–C4 过 且 C5–C8 过 | `TOPOLOGY_IS_CAUSAL` — (c) 成立，对置面目标进入生产候选 |
| C1–C4 过 但 C5 不过 | `TOPOLOGY_DIAGNOSTIC_ONLY` — 拓扑能**预测**失败但重定位**修不好**（可能受手部单块碰撞体限制），下一步转向新 `patch_mode` 的对置指垫 |
| C2 不过 | **Stage 2 不启动**。(c) 被证伪，box001↔box024 的差异另有来源，剩余归因回到 (a)/(b) |
| C5 过但 C7 不过 | 归因错误，结果不可用 |

**Stop-loss**：
- Stage 1 若 `hand_bimanual_contact_frac` 在多数 case 上 < 0.2（两手很少同时接触），对置度无统计基础，
  **如实记录并终止**，改为先解决「两手同时接触」这个更基础的问题。
- Stage 2 canary 若 ≥3 例 `track_eef_pos_err_cm_mean` 暴涨（> 15 cm），说明 snap 破坏了手臂可达性，
  改为「只在参考已接近对置面的帧上 snap」的保守变体，或直接停。

---

## 🔧 拟新增文件（**批准后才创建**）

### Stage 1

| 文件 | 用途 |
|---|---|
| `workspace/core4d/scripts/eval/core/core_metrics.py` | **纯追加** `E193_TOPOLOGY_FIELDS` + 接触循环内读 `con.frame` + 面标签 |
| `workspace/core4d/scripts/eval/runners/eval_E193_grasp_topology_audit.py` | 复用 E191 的 source 列表重打分 141 条 |
| `workspace/core4d/scripts/eval/reports/gen_E193_topology_report.py` | C1–C4 判定 + box001↔box024 配平对照 |
| `workspace/core4d/scripts/eval/wrappers/eval_E193_grasp_topology_audit.sh` | shell 入口 |

### Stage 2（gated）

| 文件 | 用途 |
|---|---|
| `workspace/core4d/scripts/experiments/E193/e193_common.py` | 冻结 contract：面选择规则、case 集、CEM 预算、`E193_METHOD_ID` |
| `workspace/core4d/scripts/experiments/E193/build_opposed_face_targets.py` | 生成 24 个 `(T,2,3)` NPZ（复用 `E098/face_utils.py`，照 `E100/build_fingertip_aware_target.py`）|
| `workspace/core4d/scripts/experiments/E193/build_opposed_target_manifest.py` | 生成 override + manifest |
| `workspace/core4d/scripts/experiments/E193/audit_opposed_targets.py` | 审计：`target_source=external`、路径存在、shape `(T,2,3)`、且非目标字段与 B0 逐字段相同 |
| `workspace/core4d/scripts/launch/active/run_E193_local_8gpu.sh` | canary / full |
| `workspace/core4d/scripts/eval/runners/eval_E193_opposed_target_arms.py` | B0 vs B1 配对评测 |
| `workspace/core4d/results/E193/scene_snapshot/` | **Stage 2 训练前必做**（Stage 1 不跑仿真，豁免）|

## 🚀 执行入口

```bash
# Stage 1（零 GPU）
bash workspace/core4d/scripts/eval/wrappers/eval_E193_grasp_topology_audit.sh

# Stage 2（仅当 C2 成立）
bash workspace/core4d/scripts/convert/snapshot_scenes.sh E193 <24 cases>
.venv/bin/python workspace/core4d/scripts/experiments/E193/build_opposed_face_targets.py --apply
.venv/bin/python workspace/core4d/scripts/experiments/E193/build_opposed_target_manifest.py --apply --snapshot
.venv/bin/python workspace/core4d/scripts/experiments/E193/audit_opposed_targets.py --require-all
MODE=canary bash workspace/core4d/scripts/launch/active/run_E193_local_8gpu.sh
MODE=full   bash workspace/core4d/scripts/launch/active/run_E193_local_8gpu.sh
bash workspace/core4d/scripts/eval/wrappers/eval_E193_opposed_target_arms.sh full
```

**结果路径**：`workspace/core4d/results/E193/{audit,s6_downstream/{cem,eval,render}}/`

## 👁️ 可视化（强制）

- **Stage 1**：对置度最高与最低各 3 例抽帧，肉眼确认「对置度低」确实长得像两手压同一个面。
  这一步是新度量的**语义校验**，不做就不能信任 C2。
- **Stage 2**：24/24 self MP4 + B0/B1 paired 对照；`/video-frames` 抽首次接触 / 抬起峰值 / 搬运中段 / 放下。
  重点看 B1 是否把「手陷大平面」换成了真正的夹持，还是换成了「够不着 / 手臂拧麻花」。
- box024 `026_p1`（穿透 0.546，最差）与 `027_p2`（0.137，最好）两端都必须看。

## 🚫 Non-goals

- 不测 (a) 阈值（E192）、不测 (b) 伺服/partner。**不与 E192 的改动叠加。**
- 不新增手部碰撞体变体（E146–E149 已证该轴无净胜；对置指垫留作 `TOPOLOGY_DIAGNOSTIC_ONLY` 后的下一步）。
- 不做 CoACD / 物体凹几何分解（box 的 AABB 本来就精确）。
- 不改 reward scale、不改 gate 阈值、不动 PRG。
- 不下 RL 结论（12 门 MJ-replay 口径）。

## ✅ 批准前 checklist

- [ ] 用户确认 Stage 1 / Stage 2 的 gating（C2 不过就不跑 CEM）
- [ ] 用户确认对置面选择规则（「离哪对面最近就用哪对」vs 强制指定某对面）
- [ ] 用户确认 Stage 2 的 24 例组成（box024 9 + box001 抽样 9 + box004 6）
