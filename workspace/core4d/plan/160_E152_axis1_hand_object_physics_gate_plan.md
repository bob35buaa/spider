# E152 — 轴1：手-物体物理穿透硬约束（CEM safety gate 纳入手 geom）

## 0. 背景：E151 暴露的真因 + 一个被纠正的事实

E151（log `191`）做了 reward 轴（B2/B1），结论"接触↑但穿透同步↑"。复查时纠正了两个错误认知：

1. **橡胶手不是"哑几何"**：rubber 场景里有 explicit `<pair name="left_hand_object" geom1="lh" geom2="object_collision" condim="4">`（rh 同）。MuJoCo 的 explicit pair **绕过 contype/conaffinity 过滤强制生成接触**，mjwarp `put_model` 完整搬运（`npair=49`）。→ **手-箱物理碰撞一直开着**（b2_tip 执行轨迹 48/75 帧有真实接触）。我此前说"contype=0/0 非物理碰撞"是只看了 contype、漏了 pair，错误。

2. **穿透是真深穿透，不是软接触嵌入**：b2_tip 物理接触 `con.dist` 平均 **−11.5mm、最深 −32.5mm、77% 接触 >5mm**。原因：pair `solref=[0.008,1]` 接触很软 + CEM 是开环采样无力控 → CEM 主动利用软接触把手怼进箱 1~3cm 换取高接触 reward 分。

**关键判断**：穿透是**物理/约束轴（轴1）**的职责，E151 日志"下一步加 reward penalty"是回 reward 轴（轴2）打补丁，方向错。E152 正面做轴1。

## 1. 轴1 杠杆 A：CEM safety gate 纳入手 geom（本实验）

现状（E151 config 实证）：
- `cem_safety_gate_enabled=True`、`mode=elite_filter`、`min_sdf_m=-0.005`、`max_violation_pct=0.0`、`fallback=least_violation`。
- `cem_safety_gate_geom_names=['head/torso/pelvis/shoulder×2/elbow×2_collision']`——**只含躯干/手臂，不含 `lh`/`rh`**。
- gate 机制（`mjwp.py:1544-1551` + `sampling.py:201-318`）：对 gate geom 集合算 `geom_box_sdf_min`，任一帧 SDF < `min_sdf_m` 即标记该样本 violation；CEM elite_filter 阶段**丢弃 violation 样本**（valid 不足时 fallback 到 least_violation）。这是采样层硬约束，比 reward penalty 强。
- `geom_box_sdf_min` = E151 已加 mesh 分支的 `_geom_box_sdf_min` → **手 mesh 顶点凸壳精确 SDF 现成可用**（E151 改动顺带解锁）。

### 1.1 关键设计约束：手需要**独立 gate**，不能并入现有 body gate

现有 gate 用**单一共享 `min_sdf_m`**，对 gate geom 集合取 **min-SDF**。但：
- 躯干/手臂：**绝不能碰箱**，SDF 须保持正（`min_sdf_m=-0.005` 容忍 5mm）。
- 手：**必须贴箱**，SDF 合法地 ≈0 甚至轻微接触。

若把手并入现有 gate 集合 → min-SDF 被手拉到 ≈0 → 要么 gate 对躯干失效（阈值被迫放宽），要么 gate 拒绝所有手接触样本（阈值严）。**两者不可调和 → 必须给手一个独立阈值的 gate。**

### 1.2 实现：新增独立 hand-object gate（隔离、可逆、纯增量）

镜像现有 `cem_safety_gate` 模式，加一组手专用字段（`config.py` + `mjwp.py` + `sampling.py` 聚合）：
- config 新增（默认关，不影响存量实验）：
  - `cem_hand_gate_enabled: bool = False`
  - `cem_hand_gate_geom_names: list[str] = []`（实验设 `['lh','rh']`）
  - `cem_hand_gate_geom_ids: list[int]`（运行时从 names 解析，仅 enabled 时）
  - `cem_hand_gate_min_sdf_m: float`（**手允许的最大穿透深度**，待扫：−0.005 / −0.010 / −0.015；比躯干宽松，允许轻接触但禁深穿）
  - `cem_hand_gate_max_violation_pct: float`（允许少量帧超限，初值如 0.05~0.10——手接触相位短暂深一点可容忍；与躯干的 0.0 区分）
  - 复用现有 `min_valid_frac` / `fallback` 逻辑。
- `mjwp.py`：在 `cem_safety_gate` 块旁加并列块，对 `cem_hand_gate_geom_ids` 算 `geom_box_sdf_min`，按 `cem_hand_gate_min_sdf_m` 算 violation_depth/violation，**合并进** `info["cem_gate_violation"]`（与现有 body gate 取 max，统一进 elite_filter）。
- `sampling.py`：无需改（已对 `cem_gate_violation` 做 valid_mask；新 gate 只是增大 violation 集合）。
- **纯增量**：`cem_hand_gate_enabled=False`（默认）时零行为变化；存量 E147-E151 不受影响（回归测试钉死）。

### 1.3 为什么这是"硬约束"而非又一个 penalty
elite_filter 在**采样选择阶段直接剔除**手深穿透的候选动作（不是给 reward 扣分让优化器权衡）。CEM 无法再用"穿透换接触分"，因为穿透样本根本进不了 elite set。这是轴1（约束）与轴2（reward 吸引）的本质区别。

## 2. 实验矩阵（用户拍板：1-A 单独 + 1-A 叠加 B1）

| 配置 | reward | 轴1 hand gate | 说明 |
|---|---|---|---|
| `baseline` | ref_fk contact_hdmi（=E148 rubber off05） | 关 | 复用，不重跑 |
| `gateA` | ref_fk contact_hdmi（与 baseline 同 reward） | **开**（lh/rh, min_sdf 扫） | **轴1 单独**：只加物理穿透硬约束，reward 不变 |
| `b1` | contact_hdmi + hand_support(mesh) | 关 | 复用 E151 b1_mesh（不重跑） |
| `gateA_b1` | contact_hdmi + hand_support(mesh) | **开** | **叠加**：reward 吸引(B1) + 物理阻挡(轴1) |

- 新跑：`gateA` + `gateA_b1`，各 3 case。`baseline`/`b1` 复用 E148/E151。
- gate 阈值扫：先用单 case smoke 定 `cem_hand_gate_min_sdf_m`（−0.005/−0.010）+ `max_violation_pct`（0.05），避免 gate 太严导致 fallback 退化（valid 样本不足 → least_violation 兜底等于没 gate）。smoke 必查 `cem_gate_valid_frac`/`cem_gate_fallback_used`。

## 3. Benchmark：沿用 E151 的 3 case

| case | task | 物体 | baseline | b1 复用 |
|---|---|---|---|---|
| box021_029_p2 | d003_box021_20231018_029_p2_e107_clean | box021 | E148 rubber | E151 b1_mesh |
| box004_083_p2 | e091_box004_20231003_2_083_p2_e092_dyn | box004 | E148 rubber | E151 b1_mesh |
| box023_person2 | box023_person2_legobj | box023 | E147 rubber | E151 b1_mesh |

## 4. 评测与成功判据（事前定义，experiment.md §5）

复用 E151 evaluator（含新增 hand-floor 指标）+ 物理接触深度。
- **主判据（轴1 的核心命题）**：`gateA` 相对 baseline，**`hand_geom_penetration_frac` 显著↓（≥−0.10）且 `hand_geom_near_5cm` 不掉（≥baseline−0.02）**，不摔、obj_err 不恶化。即"挡住穿透但不杀接触"。
- **叠加判据**：`gateA_b1` 相对 b1（E151），**穿透↓ 且 near-5cm 接触保持 B1 的增益**——验证 reward(吸引)+gate(阻挡) 协同 = 接触↑且穿透不↑（E151 单 reward 做不到的）。
- **新增物理穿透深度指标**：从执行轨迹算 hand-object `con.dist` 分布（mean/最深/`frac<−5mm`），直接量化物理穿透改善（比 SDF<0 frac 更贴物理）。建议加进 evaluator。
- gate 健康度：`cem_gate_valid_frac` 不长期塌到 `min_valid_frac`（否则 gate 退化为 fallback）。
- 报 mean+std+worst（3 case），禁 cherry-pick；A/B 同 clip 关键帧并排（看手是否停在箱面而非插入）。

## 5. 风险与对策

- **R1 gate 太严 → fallback 退化**：min_sdf 太浅 + violation_pct=0 → 几乎所有接触样本 violation → fallback least_violation 兜底 = gate 名存实亡。对策：手 gate 用独立较宽阈值 + 非零 violation_pct；smoke 监控 valid_frac/fallback_used；扫阈值。
- **R2 gate 杀接触**：硬剔除接触样本可能让 CEM 不敢贴箱 → near-5cm 掉。对策：min_sdf 允许轻接触（如 −5mm 表示"可碰但不深于 5mm"），不是要求 SDF>0；主判据已含"接触不掉"。
- **R3 改 SPIDER 核心（gate 路径）**：改动隔离在新 `cem_hand_gate_*` 字段，默认关；回归测试 E147-E151 数值不变。
- **R4 box021 摔倒残留**：E151 b2_sup 摔是 target 偏移 33cm 所致；本实验 gateA 用 ref_fk（不用 external target），不继承该问题；gateA_b1 用 b1 的 ref_fk，亦不涉及。

## 6. 数据管线（不破坏原有）

- 沿用 E151 rubber 场景 + scene_snapshot fallback（远端 task dir 缺失时本地评测，experiment.md §7）。
- gate 走 override overlay（新增字段），不改基链；scene 不动（pair 已存在，无需 patch）。
- 复用 baseline(E148)/b1(E151) 轨迹，不重跑。
- per-exp snapshot 进 `results/E152/scene_snapshot/` + manifest.txt（git HEAD+sha256）。

## 7. 验证（端到端）

1. **回归**：`cem_hand_gate_enabled=False` 时，E151 某 case 重算 reward/gate 数值不变（钉死纯增量）。
2. **单元**：构造手深穿箱已知姿态，新 hand gate 的 violation 判定与手 mesh SDF 一致。
3. **smoke**：1 case gateA，确认 gate 字段吃进、`cem_gate_valid_frac` 合理（不塌）、CEM 收敛、出轨迹；定 min_sdf/violation_pct。
4. **全量**：3 case ×{gateA, gateA_b1}，eval 三/四方对比（baseline/gateA/b1/gateA_b1）。
5. **结论**：轴1 物理硬约束能否单独压住穿透（gateA）；与 B1 reward 叠加能否实现"接触↑且穿透不↑"（gateA_b1）——回答两轴协同是否是正解。

## 8. 范围约束

- 只动轴1（CEM hand gate）+ 复用轴2 的 B1；不碰 target 投影（B2）、不调 solref（轴1 杠杆 B，留作后续单独 E 号）、不动跟踪/稳定/object actuator/partner。
- 所有改动 git 追踪 + 回归；scene/manifest per-exp snapshot。

## 9. 记录

- `log/<N>_E152_*.md`：四方对比双指标表（接触 + SDF穿透 + 物理穿透深度）+ gate 健康度 + 视频 + 结论。
- `EXPERIMENT_TRACKER.md` 加 E152 行。
- memory `project_contact_anchor_misalignment` / 新建 axis 维度 memory：记录"橡胶手物理碰撞一直开（explicit pair）、穿透是软接触+开环采样所致、轴1=safety gate 纳入手"。
