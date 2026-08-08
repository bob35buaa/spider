# E192 实验计划：收紧 CEM 手门硬地板——阈值是否是大箱失败的原因（机制 (a)，精简案）

_Core4D · Phase 55 · 计划态，**待用户批准后才创建脚本 / 跑 CEM** · 承接 [E191](../log/266_E191_object_support_offline_audit_results.md)_

---

## 📋 Context

E191 离线审计（零算力，141 条 rollout 重打分）确认了 box024 的远端下沉现象，但把「大箱失败」的归因留在三条**与物体尺寸完美共线**的机制上，现有实验无法区分：

| | 机制 | E191 给出的证据强度 |
|---|---|---|
| **(a)** | 固定米制 CEM gate / reward 阈值 | **本实验的对象** |
| (b) | 物体伺服过软 + partner 未建模 | H1b 成立（下垂量级 = m·g/kp，观测/预测 0.93）但 H1 证伪（z 只占误差 0.21–0.41）|
| (c) | 大箱无几何闭合，只能压平面 | 残差项，见 E193 |

E191 §5.5 的 provenance 审计给出 (a) 的事实基础：

- **156/156** 个 per-case config 继承同一个 `core4d_E167_box004_082_p1_E167A`（box004 case 级 override）
- 追踪的 **27 个米制/尺度敏感参数：27 个完全相同、0 个随物体变化**，跨越 2.0–120.4 kg 与 0.196–0.491 m
- `plan/189` §147 书面要求过 per-object base 但从未创建，且要求本身写明「reward 字段级 diff 必须为空」
- `log/233:131` 明确把「尺寸自适应 hand-collision margin」划出范围，至今无实验做过

### 关键 insight：(a) 有**两个方向相反**的子机制，不能用单一「按尺寸缩放」来测

这是本计划与「A4 臂按 `max_half_extent/0.224` 线性缩放」初稿的核心分歧。逐门看，干预方向不同：

| 子机制 | 含义 | 该往哪边调 | E191 证据 |
|---|---|---|---|
| **(a1) gate 饥饿** | 门太紧 → CEM 没有合法候选 → fallback 到劣解 | **放松** | 弱（见下）|
| **(a2) gate 被吃满** | 门太松 → CEM 把解顶到门限上 | **收紧** | 强 |

**(a2) 证据强**：box024 的 `hand_object_con_dist_min_m = −0.0201`，正好压在
`cem_hand_gate_hard_floor_m = −0.020` 上；硬地板饱和帧占比 **0.0136 vs box004 0.0046（约 3×）**。
即「手能陷多深，就陷多深」。

**⚠️ (a1) 证据在 PRG 侧被削弱**：E191 报告里 box024 posture fallback 0.0888 vs box004 0.0367（约 2.4×）
是 **no-PRG 侧**的数字。本计划核对了 **PRG 侧**（E172/E173，即生产默认）：

| | `cem_gate_valid_frac` | `cem_posture_gate_fallback_used` |
|---|---:|---:|
| box024 (E173 PRG) | 0.7954 | **0.1427** |
| box004 (E172 PRG) | 0.7790 | **0.1530** |

**PRG 侧 box004 的 fallback 反而更高**，两者基本持平。所以 (a1) 在生产配置下**没有物体间差异**。
**本轮据此不测 (a1)**（精简案，见 §Claims 里删 C5 的理由）——它唯一的立足点只剩「解释大箱 scene reject」
（`log/233:54`，起始腿-物重叠 −0.016/−0.011/−0.009 全撞固定的 −0.005 地板），不足以占 2/3 算力。
相关诊断列照报，不设判据。

### 判别设计：单点新值 + box004 阴性对照（**不是调参**）

不做阈值 sweep，也不按尺寸缩放，而是**对两个物体施加同一组绝对新值**，判别力全部来自两者的**差**：

```
若 box024 显著改善 且 box004 不退化   → 阈值确实是对大箱设错的 → (a) 成立
若 两者同向同幅改善                    → 阈值只是全局次优，与尺寸无关 → (a) 作为"尺寸机制"被证伪
若 两者都不动                          → (a) 被证伪，剩余归因转向 (b)/(c)
```

**注意第二行**：数字变好、但结论是「假设错了」。这是本实验与调参的根本区别——
交付物是**「(a) 是否成立」的判决**，不是「最优阈值是多少」。C3（Δ 差 ≥ 0.10）和
C6（饱和率必须同步下降）就是为此设的否决条件；调参不会给自己设这种条件。

box004（最长半轴 0.224 m、手门饱和率 0.0046、`con_dist_min` −0.0141 离硬地板还有 6mm）是天然阴性对照。

---

## 🎯 Claims

| # | Claim | 最低证据（预注册，事后不得调整） |
|---|---|---|
| C1 | 手门硬地板是 box024 穿透的**主动约束**，收紧它能降低穿透 | A2 下 box024 `hand_object_physics_penetration_3mm_frame_frac` ≤ **0.20**（基线 0.3776），≥7/9 例改善 |
| C2 | 降穿透**不是靠把手拿开**换来的 | 同时 box024 `hand_object_physics_contact_3mm_in_mask_frac` ≥ **0.3079**（基线，不得下降）且 `hand_object_physics_contact_in_mask_frac` ≥ 0.75（基线 0.8188，允许小幅回落） |
| C3 | 效应是**尺寸特异**的，不是全局调参 | Δ(box024 穿透) − Δ(box004 穿透) ≥ **0.10**（绝对值） |
| C4 | 不产生跨门回退 | box004 六例 12 门任一门不得由 PASS→FAIL；box024 `leg_penetration_frac` 不得升高 > 0.05 |
| ~~C5~~ | ~~(a1) 放松门能否解 gate 饥饿~~ | **已删除**（见下）。(a1) 降级为**观测项**：照报 `cem_posture_gate_fallback_used` / `cem_gate_valid_frac` / `cem_leg_gate_*` 诊断，但不设判据、不单独开臂 |
| C6 | 机制读数与结果一致 | `hand_gate_floor_saturation_frac` 在 A2 下必须下降；若穿透降了但饱和率没降，则归因错误，C1 不成立 |

> **为什么删 C5**：本计划 Context 已核出 (a1) 在**生产（PRG）侧没有物体间差异**
> —— box004 posture fallback `0.1530` **反而高于** box024 `0.1427`。先验太弱，不值得占 2/3 的算力。
> 若 A2 跑完后 (a) 被证实成立，再单开一轮补 (a1)；若 (a) 被证伪，(a1) 也没有再测的必要。
> 保留观测列，是为了万一 A2 意外把 gate 饥饿也改善了，能看得见。

---

## ⚙️ 实验设计：单臂 + 阴性对照（精简案）

**冻结不变量**：PRG **开启**（生产默认，E189 已判「不能据 12-gate 非劣撤 PRG」）；
E167A_zOnlyBody profile；`rubber_hull` 手部碰撞体；CEM `seed=0 / 1024 samples / 32 opt steps`（与 E172/E173/E189 同）。

| 臂 | `cem_hand_gate` 收紧 (a2) | 需要跑吗 |
|---|:---:|---|
| **A0** 基线 | ✗ | **否** — E172(box004×6) + E173(box024×9) 已落盘 |
| **A2** 收紧手门 | ✓ | **15 例** |

一臂两物体：box024×9 是待检验对象，box004×6 是阴性对照。判别力全部来自
**Δ(box024) − Δ(box004)**（C3），不来自 box024 自己变好没有。

### 参数取值（纯 Hydra override，零核心代码改动）

**A2 —— 只动手门这两个数**：
```yaml
cem_hand_gate_hard_floor_m: -0.010        # 基线 -0.020
cem_hand_gate_max_violation_pct: 0.05     # 基线 0.10
# cem_hand_gate_min_sdf_m 保持 -0.010 不动（避免一次动三个参数）
```

> **不动的东西**（避免混淆）：
> `cem_leg_gate_*` / `cem_posture_gate_*`（(a1)，本轮不测，见删 C5 的理由）、
> `surface_band_*`（3mm 表面带是局部接触判据，E191 明确排除）、
> `object_lift_sigma`（与机制 (b) 伺服下垂耦合，留给 (b) 的实验）、
> `leg_object_penalty_*`（PRG 的 R 分量，保持冻结）、任何 reward scale。

### Case 集合（15 例，与 E191/E189 完全一致）

- **box024 × 9**：`20231011_{026_p1, 026_p2, 027_p1, 027_p2, 028_p1, 028_p2, 030_p1, 031_p1, 031_p2}`
- **box004 × 6**：`20231003_2_{082_p1, 082_p2, 083_p1, 083_p2, 086_p1, 086_p2}`

box001/box023 本轮不跑（box001 与 box024 体积几乎相同但穿透只有一半，是 E193 抓取拓扑的关键对照，
留在那里做；box023 是小箱，box004 已经承担阴性对照角色）。

### 预算

**15 条 Full CEM**（1 臂 × 15 例），约为 E189（43 条）的 1/3。8×L20Y 本地，按 `assigned_gpu` 分片。
先跑 **canary**（`64 samples × 4 opt_steps`）**3 例**：box024 `026_p1`（基线穿透 0.546，最差）、
box024 `027_p2`（0.137，最好）、box004 `082_p1`（对照），确认 override 生效且不崩，再放 Full。

---

## 📊 基线（A0，已落盘，只读）

| 指标 | box004 (E172, n=6) | box024 (E173, n=9) |
|---|---:|---:|
| 12 门通过 | 2/6 | 2/9 |
| `hand_object_physics_penetration_3mm_frame_frac` | 0.1469 | **0.3776** |
| `hand_object_physics_contact_3mm_in_mask_frac` | 0.4507 | 0.3079 |
| `hand_object_physics_contact_in_mask_frac` | 0.6944 | 0.8188 |
| `hand_gate_floor_saturation_frac` (E191 新列) | 0.0046 | **0.0136** |
| `hand_object_con_dist_min_m` | −0.0141 | **−0.0201** |
| `leg_penetration_frac` | 0.0565 | 0.0386 |
| `track_obj_pos_err_cm_mean` | 12.0105 | 13.6421 |
| `track_obj_ori_err_deg_mean` | 11.5344 | 6.2508 |
| `obj_side_z_asym_cm` (E191 新列) | 1.1588 | **7.3322** |
| `cem_gate_valid_frac` | 0.7790 | 0.7954 |
| `cem_posture_gate_fallback_used` | 0.1530 | 0.1427 |

### 必报指标

12 门全部 + 上表全部 + E191 的 22 个 support 列。逐物体分别报，**禁止合并成单一跨物体结论**（沿用 E189 约定）。
配对统计：按物体做 McNemar（门级）+ bootstrap CI（连续量），种子 0、10000 次。

---

## 🛡️ 成功标准与 stop-loss

### 判决

| 结果 | 判定 |
|---|---|
| C1+C2+C3+C4+C6 全过 | `THRESHOLD_IS_CAUSAL` — (a) 成立且尺寸特异，进入 E193 前先把新阈值固化 |
| C1+C2 过但 C3 不过 | `GLOBALLY_SUBOPTIMAL` — 阈值该改，但**不是**大箱专属机制，(a) 作为尺寸机制被证伪 |
| C1 不过 | `THRESHOLD_NOT_CAUSAL` — 剩余归因压到 (b)/(c)，E193 优先级上调 |
| C1 过但 C6 不过 | 归因错误，结果不可用，需查是不是别的东西变了 |

### Stop-loss

- canary 3 例里若 ≥2 例 CEM 崩溃或 override 未生效（审计脚本检出），**停止**，不进 Full。
- 若 canary 上 box004 出现新的 12 门 FAIL，说明收紧手门有全局代价：
  **A2 降级为 box024-only（9 例）**，并在 log 里如实记录 box004 的代价——
  注意此时 C3 将失去阴性对照，判决只能降级为 `INCONCLUSIVE`，不得据此宣称 (a) 成立。

---

## 🔧 拟新增文件（**批准后才创建**）

| 文件 | 用途 |
|---|---|
| `workspace/core4d/scripts/experiments/E192/e192_common.py` | 冻结 contract：A2 参数、15 例 case 集、CEM 预算、`E192_METHOD_ID` |
| `workspace/core4d/scripts/experiments/E192/build_hand_gate_manifest.py` | 生成 15 条 override + manifest（`--apply --snapshot`）|
| `workspace/core4d/scripts/experiments/E192/audit_gate_overrides.py` | **正向审计**：确认 `cem_hand_gate_hard_floor_m/-max_violation_pct` 确为新值，且**其余全部字段与 A0 逐字段相同**（尤其 `cem_leg_gate_*`/`cem_posture_gate_*` 必须没动）|
| `workspace/core4d/scripts/launch/active/run_E192_local_8gpu.sh` | canary / full 启动（照 E189，`unset MUJOCO_GL`）|
| `workspace/core4d/scripts/eval/runners/eval_E192_hand_gate_arm.py` | A0 vs A2 × 2 物体配对评测 |
| `workspace/core4d/scripts/eval/wrappers/eval_E192_hand_gate_arm.sh` | shell 入口 |
| `workspace/core4d/scripts/eval/reports/gen_E192_arm_comparison.py` | 报表 + C1–C4/C6 逐条判定 |
| `workspace/core4d/results/E192/scene_snapshot/` | **训练前必做**：`snapshot_scenes.sh E192 <15 cases>` |

> 优先用 `gen_experiment.py --exp-id E192` 生成 train/launch/eval wrapper 骨架，
> manifest builder 的实验特有逻辑手写。

## 🚀 执行入口

```bash
# 0. scene 快照（第一步，强制）
bash workspace/core4d/scripts/convert/snapshot_scenes.sh E192 <15 cases>

# 1. 构建 + 审计
.venv/bin/python workspace/core4d/scripts/experiments/E192/build_hand_gate_manifest.py --apply --snapshot
.venv/bin/python workspace/core4d/scripts/experiments/E192/audit_gate_overrides.py --require-all

# 2. canary → full
MODE=canary bash workspace/core4d/scripts/launch/active/run_E192_local_8gpu.sh
MODE=full   bash workspace/core4d/scripts/launch/active/run_E192_local_8gpu.sh

# 3. 评测 + 报表 + 渲染
bash workspace/core4d/scripts/eval/wrappers/eval_E192_hand_gate_arm.sh full
```

**结果路径**：`workspace/core4d/results/E192/s6_downstream/{cem,eval,render}/`

## 👁️ 可视化（强制）

- 15/15 self MP4 + 与 A0 的 paired 对照视频。
- 用 `/video-frames` 对**每个发生 migration 的 case** 抽帧（首次接触 / 抬起峰值 / 搬运中段 / 放下），
  重点看 A2 是否把「手陷进大平面」换成了「贴着面滑」或「干脆不接触」——
  后两者即使数字变好也算 C2 失败。
- box024 `026_p1`（基线穿透 0.546，最差）与 `027_p2`（0.137，最好）两端都必须看。

## 🚫 Non-goals

- **不测 (a1) gate 饥饿** —— 不动 `cem_leg_gate_*` / `cem_posture_gate_*`（先验太弱，见删 C5 的理由）。
- 不测机制 (b)（伺服/重力补偿）—— 不动 `init_pos/rot_actuator_gain`、不加 `gravcomp`、不开 `partner_force_scale`。见 E194。
- 不测机制 (c)（抓取拓扑）—— 抓握目标保持 `ref_fk`，不用 `contact_hdmi_target_source: external`。见 E193。
- **E192 / E193 / E194 三个实验互不叠加**：各自只动一条轴、冻结另两条的相关参数，保证可独立归因。
- 不做 PRG on/off 消融（E189 已判决）。
- **不做阈值 sweep** —— A2 是单点预设值，不搜参数空间。本实验的交付物是「(a) 是否成立」的判决，
  不是「最优阈值是多少」。若判决为 `THRESHOLD_IS_CAUSAL`，再单开一轮做标定。
- 不动 reward scale、不动 `surface_band_*`、不跑 box001/box023/bucket/desk。
- 不下 RL 结论：本实验是 12 门 MJ-replay 口径，**不等于** RL/Holo 结论（沿用 E189 §10.2 的教训）。

## ✅ 批准前 checklist

- [ ] 用户确认 A2 的两个数：`cem_hand_gate_hard_floor_m −0.020 → −0.010`、`max_violation_pct 0.10 → 0.05`
- [ ] 用户确认 15 例 case 集与「不跑 box001」的取舍
- [ ] 用户确认 15 条 Full CEM 的算力预算
