# core4d_collab_retarget 方向评估与下一步规划

日期：2026-05-18（v4：E081 改为软参考 target；E014/E015 各增加一轮内部 sweep 再判断是否升级；E016 触发条件收紧）
评估范围：E001-E012（E012 仅 plan）
参考资料：本工作区 EXPERIMENT_TRACKER / progress / E001 & E011 log / E012 plan / HYPOTHESIS_BACKLOG / paper_notes/02 COLA；外部 holosoma `v2/log/10_v3.3_v4.x_results.md`、`v2/log/12_v6.0_architecture_results.md`；与用户 2026-05-18 讨论修订

## 一、术语：用大白话先把两个关键词钉死

- **外部协作力**（原文 external coupling）= 不是机器人手提供的、用来帮物体到达参考位置的辅助力。E004-E012 全在做这件事的不同形态：COM 弹簧、support 点 wrench、mocap pad、dual-point spring。物理含义都是"一个看不见的虚拟伙伴在帮物体走"。
- **作弊上限**（原文 oracle ceiling）= 假设有人能完美牵着物体走，机器人最好能跟到多接近 E081 的精度。如果这个开金手指的上限都到不了 `obj 0.143m`，说明我们追的就是个虚假目标。

## 二、当前实验背景与进度还原

**任务定位**：在 `workspace/core4d` 的 `E081` baseline 之上，构建面向 sim2real 的 **单机器人 + 虚拟 partner** 协作搬运 retarget。E081 已是 `scene_act` actuator-guided object（物体被 6 个 PD actuator 拉动 → object position 有一条"无延迟通道"，CEM 在 robot 控制上做优化），不是真物理搬运。**E081 视频看着自然是合理的**：robot 全程都在走、姿态正常；actuator 只是减轻了 robot 实时跟踪物体的压力，不是 robot 在装样子。本工作区从 E001 起，所有实验在 **真 freejoint object + leg-object collision pair** 的更严苛设定下推进。

**两个验收基线**：

- main `box025_p2_legobj`：obj `0.143/0.271m`、hand `89.0%`、leg intf `7.5%`、floor `59.5%`
- guard `box023_p2_legobj`：obj `0.164/0.317m`、floor `34.7%`

**E001-E011 已完成的诊断链**（12 个实验，11 个有 full 结果）：

| 阶段 | 实验 | 关键结论 |
|------|------|----------|
| 审计 | E001-E002 | 确认 E081 依赖 actuator guidance；真 freejoint 下 main/guard 都失败 (obj `0.7-0.8m`) |
| 物理可行性 | E003 | 被动 mass/friction 不是瓶颈；1kg+高摩擦 main 仍 `0.4m`、guard 仍 `0.8m+`（**H001 被证伪**） |
| 虚拟力探索 | E004-E005 | gravity / COM / support-site 力变体；ref_dt timing bug 修正；main 上限约 `0.66m`，无 transport |
| COLA proxy 简化版工程接入 | E006 | support-body proxy（**实际是 spring force，无 dynamic body**）工程接入成功，但产生显著旋转捷径（rot `20-48°`，xy 只走 `21-50%`） |
| 时基/限速修复 | E007-E008 | 修正 `sim_dt` 索引和 `vmax` clamp，最好 `ypos_k20_vmax2` obj `0.363/0.684m`、xy `0.724`、rot `13.6°` |
| robot-side 闭环尝试 | E009 | hold-contact reward 增强 → 没有改善 E008 best，反而把 rot 推到 `38°` |
| 结构性 contact pad | E010 | mocap pad 让 hand 接触升到 `83-91%`，但 object 几乎不动（xy `0.17-0.33`）；几何/pose 来源问题，不是范式问题 |
| 诊断性外力 | E011 | COM-level spring k100 达到 E011 best：obj `0.340/0.673m`、xy `0.905`、rot `3.5°`，**首次轻微 beat E008** 但仍远未到 E081 |
| Planned | E012 | dual-point partner pose closure（仍是 spring force 范式） |

**E011 k100 的细分对比详见** `docs/02_E011_k100_vs_E081_full_metric_comparison.md`。核心要点：E011 k100 在 8 项里有 4 项 ≥ E081（leg-obj、floor、bottom lift、xy ratio、yaw、hand 几乎平手、partner force 在合理范围），**残差几乎 100% 集中在 xy 位置滞后**（rot 1.5°、bottom 差 1mm、xy ratio 0.905 完成 91%，但 obj_mean 0.34m）。

## 三、对当前大方向的判断

### 3.1 已经被实验链清楚证明的事实

1. **真 freejoint + 单机器人 + spring 范式的外部协作力**，在 SPIDER CEM 下达不到 E081 级精度。E011 已扫完 COM 单点 / off-COM 单点 / mocap pad / hold-contact reward / 竖直 gravity 补偿；**最好 obj_mean = 0.34m，是 E081 的 2.4 倍**。
2. **强 spring 力与 hand contact 之间存在权衡**：`g1`（xy 0.971）和 `pad10_vmax2`（hand 90.8%）是两个极端。CEM + 当前 reward 不会自发构造"手部跟随被外力牵引的物体"的双端协调，只在两个局部解之间摆动。
3. **残差形态是 xy 位置滞后**（详见 §3.4 物理诊断），不是飘走/翻倒。

### 3.2 重新审视 E001-E011 与 COLA 的关系

之前的判断（"E006/E011 工程接入成功但算法未改善 → COLA 思路在 SPIDER 失败"）**不准确**。正确解读是：**E006/E011 实际上从未真正实现 COLA**，只复现了 COLA 思路的一个简化退化版本。

COLA 的核心结构有两个独立特征：

- **特征 A**：support body 自身有质量、惯性、freejoint 状态
- **特征 B**：partner-object 之间是 **6-DoF joint 约束**（带 stiffness / damping / 限位 / 摩擦），不是"力"

两者作用分离：

| 特征 | 作用 | 我们是否实现过 |
|------|------|----------------|
| A（dynamic body） | 让 partner PD 控制器有稳定 plant 追 velocity/height/yaw command；提供 partner-object interaction force 这个 metric；产生 partner 端 compliance | **从未实现** |
| B（位置约束 coupling） | 把 partner 端位置直接传递给物体，**消除 spring lag**；物理上等价于"半刚性连接" | **从未实现**；E006/E011 都是 spring force，E010 是 contact pad（位置约束的简化版但几何参数错了） |

**关键洞察**（与用户讨论得出）：消除 xy lag 的根本机制是 **B（位置约束）**，不是 A（dynamic body）。

- 如果只有 A 没 B（dynamic body + spring 连接）→ 物体仍受力反馈控制 → 还有 spring lag
- 如果只有 B 没 A（kinematic body + 位置约束）→ 物体被位置直接驱动 → 无 spring lag
- 完整 A+B = COLA：B 解决 lag，A 给 RL training 提供 compliance/effort metric

**因此**：

- **COLA 思路本身没死**；E001-E011 失败是因为 **被错误简化成了 spring force 范式**（特征 A、B 都未实现）
- E014 应当 **以对齐 COLA 为主线**，先实现 B，再加 A
- 如果完整 COLA 仍失败，再退到 holosoma 风格的 **kinematic + contact**（这是更弱版的 B，用 contact 而非 joint 实现位置约束）

### 3.3 E012 的预期与诊断价值

E012 (dual-point spring) 仍属于 spring force 范式，与 E011 单点 COM 的 **net force 相同**（对称双点 → ΣF_i = F_net 一致），唯一新增的是 torque。

- xy lag 预测：**不会改善**（spring time constant τ=√(m/k) 不变 → 滞后量不变）
- rot 预测：会更紧（torque 闭环），但 E011 rot 已经只有 3.5°（ref 2°），改善空间有限
- obj_mean 预测：**0.30-0.35m**，最好可能压到 0.25m，**仍不过 E081 0.20m gate**

**E012 仍值得跑完**：作为 spring 范式的最后诊断（穷尽姿态约束），并定量验证"xy lag 是结构性"的判断。但**对它的成功概率不应抱期待**；E012 失败不是结果上意外，是预期。

### 3.4 大方向的核心矛盾：xy lag 是 spring 范式的结构性问题

物理分析：

```
弹簧-质量系统 time constant: τ = √(m/k)
box025 m=5kg, k=100 → τ ≈ 0.22s
ref 平均速度 0.4 m/s → 稳态滞后 ≈ v×τ = 0.09m
加速段峰值滞后 0.4-0.7m → 与 E011 实测 obj_max 0.67m 完全吻合
```

要把 lag 压到过 E081 gate 的 5cm 以内，需要 k ≥ 320 N/m。继续推 kp 会触发两个崩溃：

1. **Robot 手脱开**（E011 `g1` 实测，hand 86%→75%）：物体被外力拉太硬，robot 追不上
2. **Spring 振荡 + 接触不稳定**：高 kp + hard contact 在 MJWarp 数值上可能爆解

→ **xy lag 在 spring 范式内是结构性的，加 kp 解决不了**。要消除 lag，只有两条路：

| 范式 | 实例 | 控制信号 | xy lag | 状态 |
|------|------|----------|--------|------|
| 力反馈 (spring) | E004-E012 | 物体位置误差 → 力 | **必有** (τ=√(m/k)) | ❌ 0.34m 上限 |
| 主动前馈 (Dynamic PD) | holosoma v6.0 R036 | ref 速度/加速度 → PD 力 | 理论无 | holosoma 实测失败（手飘走、wrist 扭曲） |
| 位置约束 (joint constraint / contact) | **E014 提案 / COLA** | ref 位置 → 几何约束 → 反力 | 几何决定（接近 0） | **未测过** |

### 3.5 流程/方法层面的小问题

- 大量实验最终回到 "目标 ≈ 0.34m，差 E081 2 倍"，但 **从未 ablate "如果 robot-side 完全冻结只用外力，object error 能多低"** — 这能定量给出 robot-side 能贡献多少的上界
- 验收一直严格用 E081 数字 gate，但 E081 本身依赖 actuator guidance；**真 freejoint 下的"作弊上限"** 没有建立过 — 如果连那个上限都到不了 `0.143m`，gate 本身就需要修正

## 四、回退方案的参考：holosoma v4.x / v6.0 经验

下游 RL 框架 holosoma 已经把 "physical partner hands" 路线踩过一遍。**这些经验现在定位为 E016 回退方案的几何参数和失败模式参考**，不是主线参考（主线是 COLA B+A）。

### 4.1 v4.x（Kinematic Hands + contact）— 对 E016 回退方案的几何参数

| 经验 | 实测 | 对 E016 回退的含义 |
|------|------|---------------------|
| 必须 relocalization 策略 | R023 absolute positioning → hands "飞走"；R023-fix 跟 robot drift 后贴近 box | E016 用方案 B：**纯 ref-driven 不 relocalize**（SPIDER 单 case retarget 中 robot drift 较小，让 hand 跟 ref 才能把 sim box 推到 ref） |
| palm offset 12cm | wrist 链距 box 表面 gap 2-9cm | 直接复用 |
| capsule radius 7cm | R023-fix2 实测 penetration 7cm 双手都接触 | 直接复用 |
| partner hand 加入后训练指标全面改善 | R024 reward 78→155 | partner 几何介入改变 contact graph，SPIDER CEM 也会受益 |

### 4.2 v6.0（Dynamic Partner PD）— 列为 E014/E015 的设计禁区

| 教训 | 实测 | 对 E014/E015 的含义 |
|------|------|---------------------|
| Dynamic body + 主动 PD 难调，partner 手飘走 | R036 kp=2000/max=200N 不够 | E015 加 dynamic body 时，**走 COLA 的 6-DoF joint coupling 路线**，不要走 v6.0 的"PD 直接控 hand pose"路线（前者 partner 是被 joint 约束的 mass，后者是 PD 力跟 pose）|
| Object attach to hands → robot 趴 box 上推 | R035 PD coupling 让 robot 学到"推" | 不要把 object weld 到 robot hand；E011 g1 已证 robot 会脱开 |
| 双约束训崩 | R037 (A+B) ep_len=3.7 | 不要同时加 partner constraint + object PD override |

### 4.3 跨版本共同失败 — E014/E015/E016 都要警惕

> **"机器人一直在用腿推箱子，从未学会用手搬箱子" — 推比搬能量上更优**

E010 在 SPIDER 也有这个味道（pad10_vmax2 hand 90.8% 但 floor 92.5%、xy 0.17）。**所有后续实验的决策门必须包含"推 vs 搬"检查**（leg_obj_contact + floor_contact 联合），不能只看 hand contact 数字。E011 在这一点上其实做对了（leg_obj 0%、floor 61%），所以这条防线已经初步证实可行。

## 五、下一步实验规划（本阶段视野，2 周）

按 "成本 / 信息价值" 排序。SBTO/horizon ablation 不在本阶段。路线 (d) 已移除。

### Step 0：E012 收尾（已计划，执行即可）

- 跑 8 个 variants（主线 + obj3 + guard）
- **关键诊断**：报告 `external_only_success` / `pose_closure_helped` / `robot_side_blocked` 分布
- **不要对 E012 的 obj_mean 抱期待**：spring 范式的 xy lag 是结构性的（§3.4），E012 改善预期 ≤ 0.10m，过 E081 gate 概率极低
- **E012 跑完即转入 §五.Step 2**，不在 spring 范式上再继续扫参

### Step 1：必做的"作弊上限"诊断（0.5 天，极便宜）

**E013 — true-freejoint object PD oracle**：在 freejoint scene 上启用 `object_pd_override=true`，让 object 完全按 ref 走（相当于无穷强外力 + 完美 ref tracking）。

- 期望产出：hand contact / pelvis err / leg intf / floor 的"作弊上限"
- **用途**（与 user 2026-05-18 确认）：E081 数字本身仍作为**软参考 target**（接近就行，不硬卡），oracle 主要用来**给 "接近" 划个量化范围**：
  - "接近 E081" 的工作定义 = `obj_mean ≤ max(E081_obj_mean + 0.05m, E013_oracle_obj_mean + 0.05m)`
  - 即：允许比 E081 差 5cm；如果 oracle 本身就比 E081 差很多，则放宽到 oracle + 5cm 那个量级
  - 其他指标（hand / leg / floor / pelvis）取 E081 与 oracle 中较宽松那个 + 容忍带（hand contact 容忍 -5pp、leg intf 容忍 +5pp、floor 容忍 +10pp、pelvis err 容忍 +0.1m）
- **必须在 E014 决策门 finalize 前完成**

### Step 2：主线 — 对齐 COLA，先 B 后 A

#### Step 2.1：E014 — COLA 特征 B 单独验证（6-DoF joint coupling，support body 用 kinematic）

**目标**：把 partner-object 连接从 spring force 改成 **6-DoF joint 约束**，测试位置约束范式能否消除 xy lag。**先不引入 dynamic body**（特征 A），保持改动最小：
support body 是 kinematic（mocap）driven by person2 hand mocap pose。

**实现方案**：

1. **Scene XML 改动**：
   - 在 partner 侧增加一个 kinematic body `partner_support`（mocap，无 freejoint 添加到 nq）
   - 这个 body 的 mocap pose 每步设为 person2 hand wrist mocap pose（来自 `partner_npz`）
   - 在 `partner_support` 与 `object` 之间增加 6-DoF coupling，初始实现用 **MuJoCo `weld` equality** + soft `solref`/`solimp`，等效于 6-DoF joint with stiffness/damping
   - **关键**：不增加 freejoint 到模型尾部，保持 `nq_obj=7` 在末尾（避免破坏 SPIDER 现有 `qpos[:, -nq_obj:]` 假设，progress.md 17:05 已警告过这个风险）
2. **Coupling 参数**：
   - `solref = (timeconst, dampratio)`：初始 `(0.02, 1)`（time constant 20ms，等效频率比 E011 的 spring τ=220ms 高 10×）
   - `solimp`：默认渐进
   - 后续扫 `solref` 的 timeconst：`(0.02, 1)` / `(0.05, 1)` / `(0.1, 1)`
3. **Relocalization 策略**：
   - 用与用户讨论得出的**方案 B（纯 ref-driven 不 relocalize）**：support body mocap pose 直接来自 person2 hand mocap pose 的 absolute 位置
   - 物理意图：让 partner_support 跟 ref 走 → sim box 落后时 partner_support 试图"穿过" sim box → joint constraint 产生大反力把 sim box 推到 ref 位置
   - 如果方案 B 数值不稳（solver 发散），回退方案 C（跟 robot drift），但需重新评估能否消除 lag

**变体**（约 5-6 runs）：

| Variant | solref timeconst | gravity scale | hold contact | Role |
|---------|------------------|---------------|--------------|------|
| `E014_box025_p2_jointB_t02` | 0.02 | 0.5 | no | main 基线 |
| `E014_box025_p2_jointB_t05` | 0.05 | 0.5 | no | 中等 stiffness |
| `E014_box025_p2_jointB_t02_g08` | 0.02 | 0.8 | no | 更强竖直补偿 |
| `E014_box025_p2_jointB_t02_hc1` | 0.02 | 0.5 | yes (hc=1.0) | robot-side 闭环加成 |
| `E014_box023_p2_jointB_t02` | 0.02 | 0.5 | no | guard |
| `E014_box023_p2_jointB_t02_hc1` | 0.02 | 0.5 | yes | guard with HC |

**决策门**（软 target 口径，参见 §Step 1）：

- **主门**：obj_mean 接近 oracle/E081 区间（具体阈值见 §Step 1 工作定义），且 hand contact 在容忍带内
- **"推 vs 搬"门**：leg-obj contact ≤ 15%，floor contact ≤ 70%（继承 holosoma 跨版本失败教训）
- **lag-free 验证**：obj_mean 必须显著低于 E011 k100 的 0.34m；若 ≥ 0.28m，说明 joint constraint 也没消除 lag → 方案 B 可能根本失败
- **作弊检测**：若 partner_support-object 反力均值 > 200N 或瞬时 > 800N，说明 partner 在硬塞物体 → 标记 `partner_overdrive`

**判定**：

- 通过所有门 → **B 范式成立**，直接进 Step 2.2 加 A
- **改善明显但未完全过门**（obj_mean 在 0.18-0.27m 之间，lag-free 满足、推 vs 搬通过）→ 进 **Step 2.1.5 内部 stiffness sweep**，不立即加 A
- 改善有限或方案 B 数值不稳（obj_mean ≥ 0.28m，或 solver 发散）→ 跳到 Step 2.2 加 A 看 dynamic body+PD 能否补足，**不直接退到 E016**

#### Step 2.1.5：E014b — B 范式内部 stiffness sweep（条件触发）

**触发条件**：E014 主轮显示明确改善（相对 E011 k100 的 0.34m 至少压到 0.27m）但未过 §Step 1 软 target。

**目的**：在不增加 A（dynamic body+PD）的工程代价下，先穷尽 B 范式自身的参数空间，确认改善上限。

**变体**（约 4-5 runs）：

| Variant | solref timeconst | solimp midpoint | gravity scale | hold contact | 备注 |
|---------|------------------|-----------------|----------------|--------------|------|
| `E014b_box025_p2_jointB_t01` | 0.01 | 默认 | 0.5 | no | 更紧 timeconst |
| `E014b_box025_p2_jointB_t01_g08` | 0.01 | 默认 | 0.8 | no | 紧 timeconst + 强竖直补偿 |
| `E014b_box025_p2_jointB_t01_hc2` | 0.01 | 默认 | 0.5 | yes (hc=2.0) | 加重 robot-side 闭环 |
| `E014b_box025_p2_jointB_t02_solimp_tight` | 0.02 | 调紧 (0.9→0.99) | 0.5 | no | 改 solimp 而不是 timeconst |
| `E014b_box025_p2_jointB_t005_unstable_probe` | 0.005 | 默认 | 0.5 | no | 极紧 timeconst，探边界稳定性 |

**判定**：

- 任一 E014b 变体过软 target → 锁定该配置作为 B-only baseline，进入 RL pipeline 对接
- 全部 E014b 仍未过门但 best-of-E014b 比 best-of-E014 又改善 ≥ 0.03m → 趋势确认 B 还在改进，再加 A 应当协同
- E014b 相对 E014 提升 < 0.02m → B 范式饱和，必须加 A 才有希望

#### Step 2.2：E015 — COLA 特征 A 补强（dynamic support body + PD command）

**前提**：E014 (B 单独) 已经过门或显示明显改善。

**目标**：把 E014 的 kinematic support body 升级为 **dynamic body（有 mass / inertia）+ PD command**，对齐完整 COLA。

**实现方案**：

1. **Scene XML 改动**：
   - 将 `partner_support` 改为 dynamic body，挂在 object 的 kinematic tree 下（**不是世界**），通过 6-DoF joint 连接，每个 DOF 有 stiffness/damping/limit
   - 这避免在末尾增加 freejoint 破坏 nq 布局（partner_support 的状态进入 object 之前，object freejoint 仍在末尾）
   - mass：初始 `2kg`（按论文 box 类物体 partner 端等效质量量级）
2. **PD command**：
   - support body 的 6 个 joint 目标值每步从 person2 hand mocap pose 推出
   - PD: `kp_pos = 500, kd_pos = 50, kp_rot = 50, kd_rot = 5`（按 COLA 论文量级）
   - 力上限 clamp，遵守人类合理范围（≤ 150N / 30Nm）
3. **保留 E014 的 6-DoF joint coupling** 作为物体侧 connection（不是 weld 了，是 mass+joint）

**变体**（约 4-5 runs）：

| Variant | support mass (kg) | PD kp_pos | gravity scale | Role |
|---------|-------------------|-----------|----------------|------|
| `E015_box025_p2_full_m2_kp500` | 2 | 500 | 0.5 | 基线 |
| `E015_box025_p2_full_m1_kp500` | 1 | 500 | 0.5 | 轻 partner |
| `E015_box025_p2_full_m2_kp1000` | 2 | 1000 | 0.5 | 硬 PD |
| `E015_box023_p2_full_m2_kp500` | 2 | 500 | 0.5 | guard |

**决策门**：与 E014 相同 + 一个 partner effort 门（partner-object interaction force mean ≤ 100N，max ≤ 250N）

**判定**：

- 过门 → **完整 COLA 成立**，进入 RL pipeline 对接阶段
- **改善明显但未过门**（obj_mean 在 0.15-0.25m 之间）→ 进 **Step 2.2.5 内部参数 sweep**，**不立即退 E016**
- 未过门也未比 E014 改善 → A 是冗余开销，**回退到 E014 best 配置**作为本阶段 reference 输出；同时启动 Step 3（E016）作为正交备选
- 训练崩溃 / 数值不稳（参考 holosoma R037 A+B 训崩教训）→ 排查 PD 参数后回到 E015 基线再判断；不是触发 E016 的条件

#### Step 2.2.5：E015b — A 引入后的参数优化（条件触发）

**触发条件**：E015 主轮显示明确改善（相对 E014 best 又压低 obj_mean ≥ 0.03m）但未过 §Step 1 软 target。

**目的**：在完整 COLA 范式内穷尽 dynamic body / PD / joint coupling 三轴的参数组合，确认 SPIDER 上完整 COLA 的实际上限。

**优化轴**（每轴 2-3 档，主线交叉约 6-8 runs，guard 各 1-2）：

| 优化轴 | 候选档位 | 物理意义 |
|--------|----------|----------|
| support mass | 0.5 / 2 / 5 kg | 太轻：PD 直接拉到位无 compliance；太重：PD 追不上 ref |
| PD kp_pos | 200 / 500 / 1500 | 决定 partner 跟 ref 的 bandwidth |
| PD kd_pos | 20 / 50 / 100 | damping，过低振荡，过高阻尼太重 |
| joint coupling solref timeconst | 0.02 / 0.05 / 0.1 | 物体侧 connection 的等效 stiffness |
| joint coupling limits / friction | 无限 / ±10cm 软限 / ±5cm + 摩擦 | 是否允许 partner 与 object 端 slip |

**实际 sweep 策略**：先固定其他三轴在 E015 best，单独扫 support mass + PD kp（最关键）共约 6 runs；如仍未过门，再各扫一轮 joint coupling 参数 4 runs；总预算 ≤ 10 runs / 1-2 天。

**判定**：

- 任一 E015b 变体过软 target → 完整 COLA 成立
- E015b best 仍未过门但比 E015 又压低 obj_mean ≥ 0.03m → 趋势仍在改进，**继续优化第三轮**（E015c：joint limits / friction / mass 更广范围；再消耗 ≤ 1 周）
- E015b 提升 < 0.02m → COLA 范式在 SPIDER 饱和；锁定 best 配置作为 reference 输出，进入 Step 3 E016 验证 kinematic+contact 是否能在不同范式下进一步改善

### Step 3（回退 / 正交验证）：E016 — kinematic + contact partner mocap hands

**触发条件**（与 user 2026-05-18 确认收紧）：必须满足**下列任一**才能启动 E016：

1. **完全失败路径**：E014 主轮 + E014b sweep + E015 主轮 + E015b sweep（含可能的 E015c）全部未过软 target，且 best 仍 < 0.03m 接近 target → COLA 范式被证伪
2. **正交验证路径**：E015 / E015b 已锁定一个 work 但偏弱的 reference（过门或接近过门），希望验证 kinematic+contact 是否能在不同 paradigm 下进一步改善

**不允许的触发条件**：E014 一轮失败就跳 E016；E015 训练崩溃就跳 E016（应先排查参数）。这些情况都要先完成本范式内 sweep。

**为什么不一开始就做 E016**：E016 是用 contact pair 实现"位置约束"的更弱版本，可控性比 6-DoF joint 差（接触可以"脱开"，joint 不行）；几何/relocalization 设计风险也更大（holosoma R023 / R023-fix 都失败过几次）。**先验证更强版本（joint），失败再退到更弱版本（contact），信息密度更高**。

**实现**（直接复用 holosoma v4.x R023-fix2 经验）：

- 两个 capsule mocap body，radius 7cm，length 15cm
- pose 来自 person2 双手 wrist mocap
- palm offset 12cm（沿 wrist→sim_object_center）
- relocalization：方案 B（纯 ref-driven，不跟 robot/sim drift）— SPIDER 单 case retarget 中 robot drift 比 holosoma RL 训练小，方案 B 是更直接的"用接触位置约束物体"实现
- 启用 hand-object contact pair，不用 weld

**约 5-7 variants**（radius / palm offset / 是否同时开 robot HC / main+guard），与 v2 文档 §五 Step 2 的 E014 设计一致（只是改名为 E016，因为它是回退而不是主线）。

**决策门**：与 E014 相同。

### Step 4：本阶段不做

- ❌ SBTO / horizon 扩展 — 锦上添花，移至下个阶段
- ❌ 双机器人 retarget — 长期备选
- ❌ 继续扫单点/多点 spring stiffness（E012 后续变体） — 边际信息低
- ❌ 仅靠改 reward 权重再循环 — E009 已证此路不通
- ❌ 路线 (d) "接受 0.34m reference 喂下游 RL" — 用户已 pass

### 里程碑（自适应，2-3 周视进度）

| 时间 | 计划 | 触发条件 |
|------|------|----------|
| Week 1 上半 | E012 完成；E013 oracle 完成；§Step 1 软 target 工作定义 finalize | 必做 |
| Week 1 下半 | E014 (COLA B / kinematic support + weld equality) 完成；初步判断 B 范式 | 必做 |
| Week 2 上半 | 若 E014 改善但未过门 → E014b stiffness sweep；若 E014 过门 → 直接进入 RL pipeline 对接 | 条件 |
| Week 2 下半 | E015 (COLA B+A / dynamic support + PD) 完成；判断完整 COLA | E014b 未过门或趋势仍改进时启动 |
| Week 3 上半 | 若 E015 改善但未过门 → E015b 参数 sweep；若 E015 过门 → 直接进入 RL pipeline 对接 | 条件 |
| Week 3 下半 | 三选一：① COLA 路线成功 → RL pipeline 对接；② COLA 范式饱和但有 work 配置 → 启动 E016 正交验证；③ COLA 完全失败 → 启动 E016 回退 | 条件 |

**总预算**：2-3 周；如果 COLA 在 E014 或 E015 主轮就过门，可压到 1-1.5 周；如果触发 E014b + E015 + E015b 全套 sweep，需要满 3 周。

## 六、user 决策记录（2026-05-18 已确认）

1. **E081 作为软 target，不硬卡**：以 E081 数字为参考目标，"接近就行"，具体阈值用 E013 oracle 划量化容忍带（详见 §五 Step 1）
2. **COLA B+A 工程代价可接受**：E014 改 scene XML 加 partner_support kinematic body + weld equality + E015 改 dynamic body + 6-DoF joint + PD 都执行
3. **不死磕也不轻易降级**：
   - E014 改善但未过门 → 先扫 stiffness（E014b），不立即加 A
   - E015 改善但未过门 → 先扫参数（E015b/E015c），不立即退 E016
   - E016 触发条件收紧为"全套 sweep 用完仍未接近 target"或"已有 work 配置需要正交验证"

## 七、关键判断变更记录

### v4（本版本，2026-05-18 用户决策反馈后）

- **E081 改为软参考 target**（不硬 gate）：用 E013 oracle 划量化容忍带，"接近 E081" 工作定义 = `obj_mean ≤ max(E081 + 0.05m, oracle + 0.05m)`
- **E014 / E015 各增加一轮内部 sweep**（E014b stiffness，E015b 参数）：改善明显但未过门时优先穷尽本范式参数，不立即升级或回退
- **E016 触发条件收紧**：必须 E014 + E014b + E015 + E015b 全套用完仍接近不上 target，或已有 work 配置需要正交验证；E015 训练崩溃要先排查参数，不是 E016 触发条件
- **里程碑改为 2-3 周自适应**：根据是否触发 sweep 弹性扩展

### v3（2026-05-18 与用户讨论后）

- **主线改为对齐 COLA B+A**（E014 先 B，E015 加 A）；holosoma kinematic+contact 降级为 E016 回退方案
- **明确 COLA 与 E001-E011 的核心差异是"位置约束 vs 力反馈"**（特征 B），而非"有无 dynamic body"（特征 A）；A 是 RL training 需要的 compliance，B 才是消除 xy lag 的根本机制
- **撤回 "E081 走路是假的" 措辞**：E081 物体由 actuator 驱动是"无延迟通道"，robot 仍然真实走路；E081 视频自然是合理的
- **移除路线 (d)**（接受 0.34m reference 喂下游 RL）— 用户明确 pass
- **xy lag 在 spring 范式内被诊断为结构性问题**（time constant τ=√(m/k) 物理推导 + E011 实测吻合）；E012 仍跑但成功概率极低
- 引用新文档 `docs/02_E011_k100_vs_E081_full_metric_comparison.md` 的细分指标证据

### v2

- 把 SBTO 从 Step 3 移出本阶段
- 利用 holosoma v4.x relocalization / palm offset / capsule radius 直接落地到 partner hands 设计，v6.0 Dynamic PD / object weld 列为禁区
- 区分 "COLA 思路 vs SPIDER short-horizon CEM 错配"（v3 进一步精化为"力反馈 vs 位置约束"差异）
- 决策门加 "推 vs 搬" 联合判定

### v1

- 初版评估，路线 (a)-(d) 全部列入候选，holosoma 经验未充分利用
