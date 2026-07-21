# 面向人机协作搬运的动力学重定向：从抓取式采样 MPC 到协作式 Loco-Manipulation 的方法扩展

> 论文式工作底稿（中文）· 2026-06-24
> 数据集：CORE4D 人-物-人协作动捕 · 机器人：Unitree G1 · 下游：SUGAR / Holosoma RL
> 基线方法：SPIDER（采样式 MPC 动力学重定向，面向单体灵巧抓取）、OmniRetarget（运动学重定向）
> 对标方法：DynaRetarget（SBTO）、ReActor（bilevel）

---

## 摘要

将人类动捕迁移为人形机器人可执行的运动（重定向）是低层运动控制的前置环节。运动学重定向方法（如 OmniRetarget）在接触丰富的人机协作搬运场景下物理合规性差——穿模、脚滑、接触失真，且 G1 较人矮、臂展更短，按比例压缩拓扑后手只够到箱底，无法侧面夹持。物理约束的动力学重定向方法 SPIDER 用采样式 MPC（SBMPC/CEM）在仿真中把运动学参考修正为动力学可行轨迹，但其奖励与约束栈是为**单体灵巧抓取**（围绕紧凑物体、以把手/接触点为目标）设计的，直接迁移到 CORE4D 的**双臂协作大箱搬运**（含 partner、行走、平面大接触面、抓取→搬运→释放相位）会在一系列具体失效模式上崩溃。

本文不更换 SBMPC 优化器，而是**重新设计其任务建模栈**，使其适配协作式 loco-manipulation。核心方法贡献包括：(1) **面接触带奖励**——用对称窄带 SDF 评分 `exp(-|sdf|/σ)` 取代抓取式点目标，适配无固定抓取点的平面接触；(2) **采样层穿透安全门 gateA**——在 CEM 采样层拒绝穿透样本，与接触层软硬度正交；(3) **释放相位衰减**——在规划 horizon 上对释放窗口衰减接触奖励，解决"放不开手"；(4) **参考相对姿态重排序**——以相对蹲姿参考的偏差（而非绝对地板高度）筛选 CEM 精英，修复"贴箱即摔"；(5) **协作与运动学建模**——物体 PD 驱动使 CEM 永不面对"抬不起"的箱、双手 AND 接触约束、足滑/平滑约束；(6) **下游对齐的 z-only 精修**——将上游跟踪/惩罚/平滑的几何与下游 RL 消费端的 z-only 终止门对齐。此外提出一套**公平评测协议**与一个**训练-free 的下游预测闸**，并给出"上游赢≠下游单 case 赢"的结构性解释——**可恢复性不对称**。

实验上，在统一物理口径下，本方法相对 OmniRetarget 把手-物物理穿透从 `0.576` 降到 `0.202`、真实接触从 `0.034` 升到 `0.153`（0 fall、跟踪不退化）；在两个独立下游 RL 项目上聚合超越 OmniRetarget：SUGAR 7-case 的 Holosoma-like 成功为 `56/51 vs 13`（约 4×），Holosoma handbox 5-case 平均成功 `48.1% vs 32.5%`（+15.6pp）。

---

## 1. 引言

### 1.1 动机

具身智能的价值不止于独立完成任务，更在于在家庭服务、康复护理、工业制造中与人**并肩工作**。人机物理协作（pHRC）——协作搬运、装配、递物——是一个持续、高频、动态接触的过程，要求机器人在各种地形上保持稳定控制并对人的动态变化做适应性响应。低层运动控制的前提是基于人类动捕的**动作重定向**，而协作场景对重定向参考的**物理合规性**要求远高于单体运动。

### 1.2 问题与缺口

运动学重定向（OmniRetarget [OmniRetarget]）通过 interaction mesh 保持相对空间拓扑，把人体动捕迁移到全身控制，但它是纯运动学的逐帧约束优化，不显式维护接触力，因而在协作搬运下产生穿模、脚滑、接触失真；叠加 G1 的形态可达性极限（手只够到箱底，0% 侧面夹持），运动学参考不足以支撑下游 RL。

动力学重定向 SPIDER [SPIDER] 用采样式 MPC（SBMPC，即 CEM/MPPI 家族）在 GPU 加速的 MuJoCo-Warp 中把运动学参考跟踪成动力学可行轨迹。然而 SPIDER（及其 HDMI/OMOMO 工作流 [HDMI]）的奖励与约束栈面向**单体灵巧抓取**：接触以物体表面的**预定义点目标**（如把手 `contact_target_offset`）为锚，奖励 `exp(-‖target-eef‖/σ)`（σ=0.3m），无显式抗穿透项、无释放语义、无姿态安全项——这些在单手抓取紧凑物体时无关紧要，但在**双手压平面大箱、边走边搬、含释放相位**的协作搬运中全部成为失效点。

与本文正交的另一条改进路线是 DynaRetarget [DynaRetarget]，它把 SBMPC 的**短 horizon 滑窗**升级为渐进增长 horizon 的 **SBTO**，在单体 OmniRetarget 数据上把成功率从 37.9% 提到 74.6% 并显著提升平滑度。DynaRetarget 改进的是**优化器**（面向单体可行性与平滑），本文改进的是**任务建模**（面向协作的接触/释放/姿态/partner/运动学/下游对齐），二者互补、可叠加。ReActor [ReActor] 则把重定向建模为 bilevel（参考与策略联合优化），并自陈其框架尚未覆盖 manipulation 与时变接触——这正是协作搬运的核心。

### 1.3 贡献

本文的贡献是一组**把抓取式 SBMPC 改造为协作式 loco-manipulation 动力学重定向**的方法，及其评测方法学：

- **C1 协作接触合规栈**（§4.1–4.4）：面接触带奖励（窄对称 SDF）、采样层穿透门 gateA、释放相位衰减、参考相对姿态重排序。四者联合解决"接触真实↑、穿透↓、放手干净、贴箱不摔"的耦合权衡。
- **C2 协作与运动学建模**（§4.5–4.7）：物体 PD 驱动（CEM 永不见"抬不起"）、双手 AND 接触约束、足滑/平滑约束、手部面接触代理几何。
- **C3 下游对齐的 z-only 精修**（§4.8）：将上游重定向目标与下游 RL 消费端的 z-only 终止几何（`BadTrackingZOnly`@0.25m）协同设计。
- **C4 评测方法学**（§5、§7）：弃用"自评"的公平评测协议 + staggered-phase 连续成功率；"可恢复性不对称"对上下游脱钩的结构性解释；训练-free 的 on-rails Isaac 下游预测闸。

实验（§6）在参考层物理指标与两个独立下游 RL 项目上均验证了相对 OmniRetarget 的聚合超越。

---

## 2. 相关工作

**运动学重定向.** OmniRetarget [OmniRetarget] 以 interaction-mesh Laplacian 形变能量为目标、穿透/限位/速度/脚不滑为硬约束，逐帧 SQP/SOCP 求解，产出运动学参考 qpos。它纯运动学（仅 `mj_forward`、无接触力），接触是几何带符号距离硬约束。GMR、PHC 等同属此类。

**物理约束的动力学重定向.** SPIDER [SPIDER] 用零阶采样优化（CEM）在物理仿真中跟踪运动学参考，奖励无需可微，可直接使用接触判定/穿透/SDF 门等不可微项；面向单体灵巧抓取。DynaRetarget [DynaRetarget] 用渐进 horizon 的 SBTO 提升单体长程可行性与平滑性。ReActor [ReActor] 用可微 bilevel 把"生成参考"与"跟踪"联合优化，消除脚滑/自穿透/地面穿透，但无物体、无 manipulation、参数时不变。

**人机/人形协作.** It Takes Two [ItTakesTwo] 学习两台人形机器人间的交互全身控制；CORE4D 提供人-物-人协作动捕。本文聚焦把单体动力学重定向扩展到**带 partner、带物体、强时变接触**的协作搬运——正是 ReActor 自陈的空白方向。

**本文定位.** 保留 SBMPC 优化器，重构其面向协作的任务建模与约束栈，并与下游 RL 消费端协同设计评测与目标。与 DynaRetarget（改优化器）、ReActor（改优化范式）互补。

---

## 3. 预备：SBMPC 与抓取式奖励为何在协作搬运下失效

### 3.1 SBMPC（CEM）

SBMPC 求解随机最优控制 `U* = argmax_U R(U)`，`U=(u_0,…,u_{H-1})` 为未来 horizon 的 PD 目标序列。一次迭代：在当前均值附近高斯采样 N 条候选 → 并行物理 rollout 评分 → 取 top-k 精英 → softmax 加权更新均值。它**从不计算 ∂R/∂U**，reward 仅用于排序/加权，故可用不可微奖励。这是本文所有"采样层门控/重排序"得以实现的基础。

### 3.2 抓取式接触奖励（HDMI/SPIDER 基线）

HDMI 接触奖励以物体表面**预定义点**为目标：`pos_rew=exp(-‖target_pos-contact_eef‖/σ)`（σ=0.3m），乘以常量力因子与 NPZ 接触 mask，非接触帧 reward=1.0。SPIDER/MJWP 将其改为**箱体 SDF 表面距离** `surface_dist=max(|hand-obj|-half_extents,0)`、取双手最小、`exp(-min_dist/σ)`、mask 由参考 FK 预计算（手距 <0.3m）。这一形态服务于"单手伸向紧凑物体"。

### 3.3 协作搬运下的失效模式（本文要解决的）

| 失效模式 | 抓取式基线为何处理不了 | 对应贡献 |
|---|---|---|
| 平面大箱**无固定抓取点** | 点目标/把手锚不适配平整大面 | §4.1 面接触带 |
| 双手压面**压入式穿透** | 无抗穿透项，CEM 利用软接触压入箱内刷接触 | §4.2 gateA |
| 抬升后**放不开手** | 无释放语义，接触奖励在释放窗口仍吸手 | §4.3 释放衰减 |
| 负载蹲姿下**贴箱即摔** | 无姿态安全；绝对地板门对蹲姿错误 | §4.4 姿态重排序 |
| 缺少 partner、需**双手协同** | 单体单手工作流无协作建模 | §4.5 协作建模 |
| **边走边搬**的脚滑/抖动 | 抓取基线基座静止，无运动学伪影 | §4.6 足约束/平滑 |
| 上游几何与**下游终止门不一致** | 上游 3D 跟踪与下游 z-only 门错配 | §4.8 z-only 精修 |

---

## 4. 方法

记号：`sdf` 为手部碰撞几何到物体表面的带符号距离（外正内负）；接触 mask 来自 CORE4D raw 3cm 接触检测；CEM horizon 上累加 reward。

**E167A 精确组成（来自 override 继承链 `E156_gateA → E163_narrowSurfaceBand → E167A`，配置核实）.** 本文拟定的 SPIDER 版本 **E167A** 由以下五个方法组件构成（其参考层成熟配置记为 **E163** = 前四项）：

| 组件 | 引入实验 | E167A 是否启用 | 关键配置 |
|---|---|:--:|---|
| ① 面接触带奖励（窄对称）| E158→E163 | ✅ | `surface_band_score_mode=symmetric_abs`, band `[-1,+3]mm`, σ=0.0015 |
| ② 采样层穿透门 gateA | E152/E153 | ✅ | `cem_hand_gate_enabled`, `min_sdf=-0.010`, `max_viol=0.10` |
| ③ 释放相位衰减 | E155/E161 | ✅ | `surface_band_decay_frac=0.15` |
| ④ 参考相对姿态重排序 | E160 | ✅ | `cem_posture_gate_enabled`, `mean/term/drop=0.10/0.12/0.18` |
| ⑤ z-only 精修 | E167A | ✅ | `e167_body_z_enabled`(w=2.0,@0.25m) + `e167_ground_z_enabled`(踝) |

并继承结构层改动：rubber-hand 面接触代理（`scene_act_E147_rubber_hull`，§4.7）、物体 PD 驱动与局部坐标跟踪（§4.5(a)、基线层）。

> **重要边界**：E166 的足约束+平滑（foot-slip XY / 3D ankle weight / 3D 平滑，§4.6）**不在 E167A 内**——E167A 显式关闭（`foot_slip_enabled=false`、`foot_ground_enabled=false`、`local_frame_ankle_weight=1.0`）。E167A 仅保留 z-only 形式的着地约束（属组件⑤）。在下游对比中 `spider_e166A_B2` 与 `spider_e167A` 是两个**并列**方法，§4.6 列出 E166 仅作为"探索但未纳入 E167A"的备选/消融。双手 AND 约束（E164，§4.5(b)）亦为负结果、未启用。

### 4.1 面接触带奖励（surface-band contact reward）

将抓取式点目标替换为**箱面 SDF 接触带**。最终采用**窄对称带 + 绝对值评分**（E163 `symmetric_abs`）：

```
band      : -1mm ≤ sdf ≤ +3mm          (surface_band_min_sdf=-0.001, width=0.003)
score     : exp(-|sdf| / σ),  σ = 0.0015   ← 峰值在表面 sdf=0，穿透与离面对称衰减
reward    : 1.5 · gate · score · 1[band]
```

它与早期单边带 `exp(-max(sdf,0)/σ)`（仅惩罚离面、容忍至 +30mm、不奖励"恰好贴面"）的关键差别在于：对称窄带**强制真实贴面**，既不奖励压入、也不容忍悬空。接触点采用箱体网格 SDF（800 采样点，与评测几何对齐），逐手（lh/rh）计算。

**演进与证据.** 单边宽带（E158/E159）虽把 in-mask 接触从 gateA 的 0.147 拉到 ~0.49，但要么破坏稳定（E158 摔 4/6），要么残留释放误接触（E159）。RL-safe 重评（§4.4、E162）发现宽带的 clean-3mm 接触虚高、而下游真正依赖的 **raw in-mask 接触回退**（box023 `0.9077→0.7538`）。收窄到对称 [-1mm,+3mm] 后（E163），box023 raw 接触恢复到 **0.8769**，3-case raw 均值 0.7505、物理穿透 0.1234、释放误接触 0.0000、跟踪 3/3、0 fall。

### 4.2 采样层穿透安全门 gateA

在 **CEM 采样层**对穿透样本做拒绝（非奖励项）。对每条 rollout，计算手部几何到箱体的逐帧 SDF；样本有效当且仅当

```
valid = (min_sdf_over_horizon ≥ hard_floor)  ∧  (frac_frames(sdf < min_sdf) ≤ max_viol)
```

`min_sdf` 是被计数的逐帧穿透容差，`hard_floor` 是单帧绝对下限（任何一帧低于即整条作废）。两个旋钮解耦后扫描（E153），最优工作点 **(min_sdf=-0.010, max_viol=0.10)** 是唯一 3/3 strict 且接触近零损失（物理接触 -0.017、box021 ±0）、深穿透 -0.172、门健康（valid 0.85–0.92）的组合。`min_sdf` 是主导的单调旋钮：更紧（-0.005）压穿透更狠但伤接触与门；更松（-0.015）保接触但 box021 真穿透回升。

**与抓取基线的关系.** 单手抓取紧凑物体很少产生深穿透，软接触 `solref` 即可；协作双手压平面大箱时，采样优化器会利用软接触**压入箱内**以最大化接触奖励。gateA 在采样层切断穿透样本，与接触层软硬度正交。统一物理口径（E156 clean8）下，相对 OmniRetarget：手-物物理穿透 **0.576→0.202**、in-mask 物理接触 **0.034→0.153**，0 fall、跟踪 8/8。

### 4.3 释放相位衰减（release-phase decay）

协作搬运含显式 **抓取→搬运→释放** 相位。接触/面带奖励在参考释放窗口仍吸手，导致"放不开手"（释放误接触）。在规划 horizon 上对释放窗口（末段 15%，`surface_band_decay_frac=0.15`）**衰减面带奖励**：

```
surface_reward_t  ←  surface_reward_t · decay(t),   t ∈ 末段释放窗口
```

关键在于衰减必须作用于 **horizon 释放帧**而非仅当前帧：仅按当前帧 mask 门控（strictMask 变体）不足，因为 CEM rollout 把未来接触帧平均进当前选择（实测 strictMask 残留释放奖励 0.076）。E155 确认 `decay` 优于 ramp/neutral（释放误接触 3mm `0.296→0.033`）；E161 clean8 将释放误接触 **0.188→0.030**（box021_029_p2 单 case `0.75→0.000`），in-mask 接触仅 -0.007、0 fall、跟踪 8/8。抓取基线无此相位，故无此项。

### 4.4 参考相对姿态重排序（reference-relative posture rerank）

面带奖励会把手拉上箱面而牺牲身体跟踪，导致负载蹲姿下摔倒（E159 box021_029_p2 摔）。但 CORE4D 参考本身就是**深蹲弯腰抬举**，绝对地板门（`pelvis_z>0.5m`）是错的。故以**相对参考根高的偏差**做 CEM 精英筛选/重排序（不引入新奖励项）：

```
per-frame :  z_err = |z_sim - z_ref|,   z_drop = z_ref - z_sim
per-sample:  valid = (mean z_err ≤ 0.10) ∧ (terminal z_err ≤ 0.12) ∧ (max z_drop ≤ 0.18)
fallback  :  若 <5% 样本通过，按 rews - 5.0·violation 重排
```

E160 在 E159 之上加入后：跟踪 2/3→3/3、摔 1→0、box021_029_p2 末端根高误差 0.042、保持高接触。**诚实边界**：姿态重排序修"摔"，但不修"放手"（box021 释放误接触 0.75）——这正是 §4.3 释放衰减与之联合的原因；E163 把四者合栈才同时满足接触、穿透、释放、姿态。

### 4.5 协作建模：物体 PD 驱动、双手约束、partner

**(a) 物体 PD 驱动（rollout 内物体跟随 GT）.** 物体自由关节替换为 6 个位置执行器（3 平移 + 3 旋转），每步在 CEM 写入控制后用 GT 参考位姿覆盖物体 `ctrl`（含重力补偿 `mg/kp` 的 z 偏置，kp=2000）。**协作语义**：CORE4D 的箱由 partner 提供横向搬运动力，单台 G1 无法复现；物体 6 DOF 被确定性 PD 驱到 GT，使 rollout **永不产生"掉箱/抬不起"**，CEM 得以专注优化机器人姿态/接触。其代价是 **CEM selection 永远看不到"抬不起来"**（§7 讨论：抬升语义必须进下游 RL reward，不能进 CEM）。

**(b) 双手 AND 接触约束（bimanual）.** 用统一搬运窗口 `global_mask = fill_holes(max(L,R))`，并把面带/接触奖励改为**双手同时在带内取最小**：`score = both_in_band ? min(left,right) : 0`。**诚实结果为负**：E164 双手 AND 在 raw 接触硬门上 0/3，过紧反而把 E163 已过的 box023/box026 拉回。结论：单纯"强制双手"会过约束接触配对，**双手协同应作为可选轴而非默认**。

**(c) partner 的三种表示.** 依次尝试运动学 mocap partner 身体（E011，物体 z_max 提升但 MPC 步间跳变致 G1 过倾摔倒）、partner 外力 wrench（E024，90% 重力卸载时"漂浮非搬运"）、双手约束（E164）。三者共同揭示：**短 horizon 关节空间 CEM 擅长身体跟踪，但不擅长接触丰富的协同操作**——这是与 DynaRetarget"长 horizon 提升可行性"互补的另一面动机。

### 4.6 运动学约束：足滑与平滑（locomotion）

协作是**边走边搬**：负载上身扰动步态、参考足部反复触地/离地，产生脚滑与抖动（抓取基线基座静止、无此伪影）。在采样层加入足部约束与平滑惩罚（从 `rews` 中扣减）：

```
foot-slip   : 参考 z ≤ 0.05m（着地）的踝, 罚仿真踝 XY 速度
foot-ground : 着地参考帧, 罚仿真踝 z 偏离参考
smoothness  : 踝位置有限差分 加速度/加加速度 的 P95（轴可选）
```

E166 将其分解为 A（CEM 足约束）/B1（CEM 平滑）/B2（后处理 CPU 平滑）三轴。**A_B2_postSmooth**（A 的 CEM 输出再做 B2 平滑）在 3-case 上最优：raw 接触 **0.777**、clean3 0.618、穿透 0.101、qpos jerk P95 **1907（↓57%）**。但**诚实边界**：在 remaining-4 下游中管线 4/4 通过、任务成功仅 1/4（box021_029_p2 `0.672→1.000`，box021_035_p1 `0.672→0.000` 回退），主残差为下游 3D `ee_body_pos`（踝+腕）门——这直接催生 §4.8。

### 4.7 手部面接触代理几何

把手部碰撞代理从 5cm 球替换为 G1 rubber-hand 凸包网格（`maxhullvert=64`）作为可选 CEM 轴。**协作语义**：协作用宽掌/指面贴平面大箱（面接触），球代理会穿入平面并以穿透虚增接触。E147/E148 显示：手穿透 `0.4456→0.2877`、深穿透 `0.0526→0.0007`，但物理接触 `0.4027→0.2780` 下降——**几何权衡而非净胜**，故保留为诊断轴/可选项，不设默认。

### 4.8 下游对齐的 z-only 精修（E167A）

上游 SPIDER 与下游 RL 消费端的**终止几何不一致**是 §4.6 残差的根因：SUGAR 的 `ee_body_pos` 是踝+腕的 **3D** 范数门（@0.30m），而 Holosoma WBT 的 `bad_motion_body_pos=BadTrackingZOnly` 只查 **z 误差**（@0.25m）。E167A 据此把上游身体跟踪/惩罚/平滑**从 3D 改为仅 z 轴**，而**物体跟踪保持 3D**：

```
body penalty :  z_err = |z_sim - z_ref| (踝+腕);  over = clamp(z_err - 0.25, 0)
smooth (B1)  :  仅对 pos[...,2] 做平滑;  XY 自由
handoff (B2) :  仅平滑 SUGAR body_pos_w[...,2] 与 root z;  要求 xy_max_abs_delta ≈ 0
```

**语义**：z-only 跟踪管控踝部离地与腕/手抓握高度（抬箱本质是 z 量），同时释放横向 XY 的过约束，使上游不再为对齐一个下游本就不惩罚的 XY 维度而牺牲解的多样性。E167A 的证据以下游为主（§6.2）：它把 box004_082/083_p1、box021_035_p1 等 case 明显改善，并取得最高的下游 completion/搬运距离；其在 box021_029_p2 的回退说明这是**重分布**（§6.4 讨论）。

> **方法学要点（C3）**：z-only 精修把"上游重定向目标"与"下游消费端的终止判据几何"显式协同设计——这正是 §7 可恢复性不对称给出的可落地方向之一（把下游真正卡人的维度放进上游目标）。

---

## 5. 公平评测协议

**问题.** 许多指标把各方法自身的 retarget reference 当 `qpos_ref`（OmniRetarget adapter 甚至 `qpos_ref=qpos`），使其 body/object tracking 误差天然≈0——只能诊断"是否贴近某条 reference"，不能横向证明谁更好。

**协议.**
1. **共同参考只能来自 CORE4D raw**：raw object pose、raw 接触 mask（3/5cm）、scene 几何、raw 足态；各方法只交 qpos，不提供 GT。
2. **分层报告**：参考层（穿透、物理接触、脚滑、pelvis/fall、leg/body-object 干涉、object SE(3) 误差、限位）与 rollout 层（同一 simulator/controller/reward/预算下的下游 RL 成功率、object progress/height、contact group）。
3. **分桶**（box/bucket/desk × carry/push/pull），避免聚合掩盖 failure mode。

**staggered-phase 连续成功率.** 早期"binary 64/64"是 frame-0 确定性 eval（64 env 行为近一致，std≈0.003，有效样本≈case 数），且是悬崖型（擦门 1.7–3cm 即 0/64），分辨率近零、不适合给上游方法排名。将 64 env 铺到不同参考起始相位后，成功率呈连续谱（实测三 spider case `box021 0.67 > box004 0.48 > box023 0.00`），§6.2 的 7-case 公平对照即建立在此口径上。

---

## 6. 实验

### 6.1 维度一：参考层物理指标（统一口径，clean8 / E156）

8 个 clean case、4 方法在同一 `core4d-e154-physics-contact-v1` 口径下评测（OmniRetarget 读各 case `trajectory_kinematic.npz` 转 `scene_act.xml` 后同口径）：

| method | tracked | fall | in-mask 物理接触↑ | 手-物物理穿透↓ | 手-物几何穿透↓ |
|---|:--:|:--:|---:|---:|---:|
| OmniRetarget | 8/8 | 0 | 0.034 | **0.576** | 0.586 |
| spider-rubberhand | 8/8 | 0 | 0.145 | 0.210 | 0.217 |
| **本方法 (+gateA)** | 8/8 | 0 | **0.153** | **0.202** | 0.211 |
| E155_decay（接触最大但穿透回升）| 8/8 | 0 | 0.314 | 0.313 | 0.374 |

相对 OmniRetarget：接触 **+0.119**、穿透 **−0.375**、几何穿透 **−0.375**，0 fall、跟踪不退化。OmniRetarget 的"接触"大半是压入式穿透（穿透 0.576 极高、真实接触 0.034 极低）；`E155_decay` 证明"接触越多越好"是错的——必须接触/穿透/释放联合权衡，故不设为默认。接触回退由 §4.1 窄对称带修复（box023 raw `0.7538→0.8769`）。

### 6.2 维度二：下游 RL（两个独立消费端项目，聚合均胜）

**SUGAR（refiner RL，7-case 同口径 staggered-phase + Holosoma-like success，每 case 64 attempts）.** Holosoma-like success = `carry_progress_ratio>0.60 ∧ height_success`。

| method | success（/448） | 相对 OmniRetarget |
|---|---:|---|
| OmniRetarget (e163) | 13 | baseline |
| **本方法 E163** | **56** | **≈4.3×** |
| **本方法 E167A**（z-only）| **51** | **≈3.9×** |
| 本方法 E166 A_B2（后平滑）| 5 | 后平滑未转化为下游 |

**Holosoma（WBT RL，handbox case-relative b04_g06，5-case，可信主线）.**

| 指标 | OmniRetarget | 本方法 (SPIDER) | Δ |
|---|---:|---:|---:|
| 5-case 平均 success | 32.5% | **48.1%** | **+15.6pp** |
| Box021 035 p1 | 0.0% | **51.6%** | +51.6pp |
| Box021 035 p2 | 4.7% | 29.7% | +25.0pp |

两个独立项目、不同 reward/eval 口径，**聚合方向一致指向本方法**。仍 case-dependent（§6.4）。

**评测完整性 caveat（Holosoma rubberhand 口径不可用）.** 另有一套 SUGAR rubber-hand（R150/R151）口径，但**不能用于 Omni-vs-SPIDER 裁定**：(i) Box004 的 OmniRetarget(R127)/SPIDER(R128) 的 RL export SHA 完全相同（SPIDER 导出的 processed trajectory 与 OmniRetarget source 逐字节相同，083 p1 亦然）——根本不是方法对照；(ii) 该口径允许腿/身体 shortcut，使 Box021 035 p1 的胜负相对 handbox 翻转（handbox SPIDER 51.6% vs Omni 0%；rubberhand SPIDER 43.8% vs Omni 59.4%，且 Omni 伴随 lower contact 0.594）。故 Holosoma 一律以 handbox 为准。该现象本身是 §6.4 / §7 "下游成败受消费端设计支配"的实证。

### 6.3 分步消融（按实验脉络，逐组件量化）

五个组件在不同阶段引入，case 集随阶段演进，故采用**分步消融**：每一步在其自身**一致的 case 集**上对比"加入该组件前后"（不强求各步同一组 case）。脉络顺序为 ②穿透门 → ①面接触带 → ④姿态重排序 → ③释放衰减 → ①窄带收尾 → ⑤z-only。所有数值取自 `results/` 下 strict eval 的 `*_method_summary.tsv` / `*_combo_summary.tsv`（source-of-truth）。**配套 xlsx：`workspace/core4d/report/0624/E167A_ablation.xlsx`**（sheet1=三版本对比，sheet2=主消融表，sheet3-9=各分步消融表）。

**表 A — 整条方法链在同一 3-case 上的统一重评（消融主表）.**
来源：`workspace/core4d/results/E163/narrow_surface_band/eval/full/e163_method_summary.tsv`（同一口径 `core4d-e154-physics-contact-v1`，同 3 case：`box023_person2 / box021_029_p2 / box004_083_p2`，逐行累加组件）。

| 配置（逐步加组件）| 引入实验 | raw接触↑ | clean3接触↑ | 物理穿透3mm↓ | 几何穿透2mm↓ | 关节误差°↓ | EEF误差cm↓ | 组件 |
|---|:--:|---:|---:|---:|---:|---:|---:|---|
| OmniRetarget（运动学基线）| 基线(E156) | 0.961† | 0.027 | 0.606 | 0.615 | 0.00‡ | 0.00‡ | — |
| spider-rubberhand（无门/无带）| E147/E148 | 0.607 | 0.118 | 0.283 | 0.294 | 3.86 | 12.75 | 基线 |
| +gateA | E152/E153 | 0.610 | 0.122 | 0.279 | 0.274 | 3.79 | 12.65 | ②穿透门 |
| +surfaceBand-A（宽单边带）| E158 | 0.683 | 0.518 | 0.120 | 0.018 | 5.02 | 23.42 | ①面接触带 |
| +surfaceBand-A2（去 penalty）| E159 | 0.616 | 0.460 | 0.105 | 0.030 | 4.56 | 15.97 | ① |
| +postureRerank | E160 | 0.668 | 0.519 | 0.106 | 0.038 | 4.24 | 14.65 | ④姿态重排序 |
| +releaseDecay | E161 | 0.716 | 0.517 | 0.137 | 0.047 | 4.49 | 14.87 | ③释放衰减 |
| **+narrowBand = E163** | E163 | 0.750 | 0.566 | 0.123 | 0.045 | 4.30 | 14.60 | ①窄对称带 |
| **+z-only = E167A（拟定版本）** | E167A | **0.778** | **0.644** | **0.097** | **0.042** | 4.29 | 14.05 | ⑤z-only |

†OmniRetarget 的 raw 接触含大量压入式穿透（physPen3=0.606）。‡OmniRetarget 关节/EEF 误差≈0 是 self-eval 偏置（`qpos_ref=qpos`），非真实跟踪优势。rubberhand→E163 各行取自 E163 full 统一重评；**E167A 行取自 E167 cem_metrics eval 在同 3-case 子集**（`results/E167/holosoma_zonly/eval/cem_metrics/full/e167_arm_metrics.tsv`，其 baseline 行与 E163 完全一致，验证可比）。"引入实验"列标注该组件首次引入/基准的实验编号。（jerk 指标判别力弱、与 EEF 类似，已移出主表；其数据见 xlsx「G_zonly参考层」与「H_平滑度E166」两表。）

读法：面接触带把 clean3 接触 `0.12→0.46~0.52`、几何穿透 `0.27→0.02`（带把手放在表面而非压入），**代价是 EEF 跟踪误差升高 `12.6→23.4cm`**（强 surface reward 牺牲身体跟踪）；姿态重排序把 EEF 误差拉回 14.6cm 并保接触，但带来 release 副作用（0.083→0.271）；释放衰减把释放误接触压回 0.006；窄对称带把 raw 接触恢复到 0.750 且释放归零（EEF 14.6cm、关节 4.3°）。**z-only（E167A）在参考层亦优于 E163**：raw `0.750→0.778`、clean3 `0.566→0.644`、物理穿透 `0.123→0.097`、EEF `14.60→14.05cm`。**但 E167A 下游为重分布**（表 F：聚合 56→51）——参考层更优而下游单 case 有升有降，再次印证可恢复性不对称（§7）。

**表 B — ②穿透门 gateA（多 case 集证据，3-case 与 8-case 分开列）.** gateA 的主效在**深穿透**。

| 口径 / case 数 | 对比 | clean3 接触 | 深穿透(<−5mm) / physPen3 | 几何穿透2mm | fall | 来源 |
|---|---|---:|---:|---:|:--:|---|
| 3-case（E152）| baseline → +gateA | 物理接触 0.349→0.311 | 深穿透 **0.553→0.347**（−0.206）| 0.294→0.269 | 0/0 | `results/E152/axis1_hand_object_physics_gate/eval/full/e152_method_summary.tsv` |
| 3-case 扫描（E153）| 门关→门开 sweet spot(−0.010,0.10) | −0.017（近零损失）| 深穿透 **−0.172** | — | 3/3 strict | `results/E153/gate_threshold_sweep/eval/full/e153_combo_summary.tsv` |
| 3-case（E156，按同 3-case 过滤）| rubberhand → +gateA | 0.118→0.122 | physPen3 0.283→0.279 | 0.294→0.274 | 0/3 | `results/E156/clean8_gate_decay/eval/full/e156_method_metrics.tsv` |
| clean8（E156，8 case，参考）| rubberhand → +gateA | 0.145→0.153 | physPen3 0.210→0.202 | 0.217→0.211 | 0/8 | 同上（8-case 与 3-case 主表不可直接比）|

**表 C — ④姿态重排序（3-case，E160，同 3 case：box021_029_p2/box004_083_p2/box023_person2）.**
来源：`results/E160/posture_rerank/eval/full/e160_method_summary.tsv`

| 配置 | tracked | fall | pz 末端误差↓ | clean3 接触 | 释放误接触 |
|---|:--:|:--:|---:|---:|---:|
| gateA+surfaceBand-A2 | 2/3 | 1 | 0.172 | 0.460 | 0.083 |
| +postureRerank | **3/3** | **0** | **0.017** | 0.519 | 0.271 |

读法：修复 fall（1→0、末端高度误差 0.172→0.017、保接触），但 release 副作用（0.083→0.271）正是引入组件③的动因。

**表 D — ③释放衰减（clean8，E161，8 case；M0→M1，对照 M2）.**
来源：`results/E161/surface_release_ablation/eval/full/e161_method_summary.tsv`

| 配置 | 释放误接触3mm↓ | clean3 接触 | physPen3 | tracked/fall |
|---|---:|---:|---:|:--:|
| M0 postureRerankA | 0.188 | 0.477 | 0.116 | 8/8, 0 |
| **M1 +releaseDecay** | **0.030**（−0.158）| 0.470（−0.007）| 0.128 | 8/8, 0 |
| M2 strictMask（仅当前帧 gate，对照）| 0.160 | 0.427 | 0.150 | 8/8, 0 |

读法：horizon 级衰减（M1）把释放误接触降 0.158 且接触几乎不损；仅当前帧 gate（M2）不足（残留 0.160），证明衰减必须作用于 horizon 释放帧。

**表 E — ①窄对称带收尾的 RL-safe 验证（clean8，E163，8 case）.**
来源：`results/E163/narrow_surface_band/eval/clean8/e163_method_summary.tsv`。E163 把 raw in-mask 接触均值从 releaseDecay 的 0.637 提到 **0.714**（rubberhand 0.481），clean3 0.470→0.544，释放误接触 0.030→0.012；8 case 中 7/8 过 raw-contact RL-safe 硬门（唯一未过 `box004_082_p1`，raw −0.098）。

**表 F — ⑤z-only 精修（下游 SUGAR 7-case，E163→E167A，同 7 case，64 attempts/case）.**
来源：`SUGAR-private/docs/log/CORE4D_OMNIRETARGET_VS_SPIDER_HOLOSOMA_LIKE_COMPARISON_CN.md`（Holosoma-like success）。

| case | E163 | E167A |
|---|---:|---:|
| box004_083_p2 | 0.078 | 0.016 |
| box004_082_p1 | 0.000 | **0.156** |
| box004_083_p1 | 0.000 | **0.312** |
| box021_029_p2 | **0.734** | 0.047 |
| box021_035_p1 | 0.062 | **0.266** |
| box021_035_p2 | 0.000 | 0.000 |
| box023_person2 | 0.000 | 0.000 |
| **合计** | **56/448** | **51/448** |

读法：z-only 是**重分布**——改善 box004_082/083_p1、box021_035_p1，但 box021_029_p2 回退；聚合 56→51 基本持平，而 E167A 的 completion/搬运距离最高（0.263/0.608m，§6.2），更易卡在下游 height/progress 门。z-only 的设计目标（对齐 Holosoma z 门）属下游消费端协同，参考层指标不是其主战场。

**表 G — 平滑度消融（jerk 指标唯一所在；E166 foot-smooth，3-case：box004_082_p1/box004_083_p2/box021_035_p2，排除 box023）.**
来源：`workspace/core4d/log/218_E166_foot_smooth_cem_results.md`（Summary 表）。**注意此 3-case 与主消融表不同**（排除 box023），故 jerk 不能并入主表。

| arm | pass | raw接触 | clean3接触 | pen3 | qpos_jerk_p95↓ | trackbody_jerk_p95↓ | ankle_acc_max↓ |
|---|:--:|---:|---:|---:|---:|---:|---:|
| baseline | 3/3 | 0.653 | 0.445 | 0.140 | 4388.1 | 579.0 | 46.0 |
| B1（CEM 平滑）| 3/3 | 0.654 | 0.503 | 0.105 | 4494.4 | 495.1 | 47.8 |
| B2（后平滑 only）| 2/3 | 0.625 | 0.457 | 0.115 | 1987.8 | 315.4 | 31.4 |
| A（CEM 足约束）| 3/3 | 0.748 | 0.548 | 0.131 | 4308.5 | 519.6 | 48.5 |
| **A_B2_postSmooth** | 3/3 | **0.777** | **0.618** | 0.101 | **1907.0** | **299.1** | 31.5 |
| AplusB（A+B1）| 3/3 | 0.730 | 0.594 | 0.095 | 4536.3 | 575.4 | 59.9 |

读法：A_B2_postSmooth 降抖最强（qpos jerk 4388→1907，**↓57%**；trackbody jerk 579→299）且接触最高。**但此分支不在 E167A 内**（§4.6 并列分支），且其下游 SUGAR 任务成功仅 1/4（box021_029_p2 0.672→1.000、box021_035_p1 回退）→ 残差为下游 3D `ee_body` 门 → 催生组件⑤ z-only。

**表 H — 探索但未纳入 E167A 的组件（诚实记录）.**

| 组件 | 结论 | 证据 / 来源 |
|---|---|---|
| 双手 AND 约束（E164）| 过约束，raw 接触硬门 0/3，回退已过 case | `log/210_E164_...`（负结果，未启用）|
| 后平滑 A_B2（E166）| 上游平滑↑（jerk P95 ↓57%）但下游任务仅 1/4，残差为 3D `ee_body` 门 → 催生 z-only | `log/218,219_E166_...`（混合，**不在 E167A**，见表 G）|
| rubber-hand 代理（E147/E148）| 手穿透 0.446→0.288 但物理接触 0.403→0.278，几何权衡 | `log/187,188`（E167A 用其几何，但非"净胜"组件）|
| E155_decay（放手段衰减替代）| clean8 接触最高但 release/穿透回升，不设默认 | `log/197_E156`（C2/C4 不成立）|

### 6.4 case 依赖与"重分布"

SUGAR 7-case 内，OmniRetarget 在 box004_r161、box021_035_p2 局部占优；box021_029_p2 是 E163 的决定性主胜（47/64），也是 E163 聚合领先 E167A 的主因。上游 E163→E167A 是**重分布而非整体提升**：E167A 明显改善 box004_082/083_p1、box021_035_p1，却把 box021_029_p2 从 47 跌到 3；E167A 的 completion/搬运距离最高（0.263/0.608m），但更易卡在 height/progress 门。box023_person2 四方法全 0（自碰撞 pathology，§7）。

---

## 7. 讨论

### 7.1 可恢复性不对称：为何聚合赢但单 case 抖

把误差维度按"下游 RL 能否自我修复"重排：

| 上游误差维度 | RL 可自修复 | 证据 | 对下游单 case 预测力 |
|---|:--:|---|:--:|
| 接触量/接触标签 | ✅ | 参考接触 0.036 → RL 自学到 0.54 | 弱（本方法一直在刷）|
| 动态可行性 vs 硬门余量 | ❌ | 接触好（IoU 0.63），仍因 obj_pos 越门 1.7cm 而 0/64 | 强（未度量）|
| 本体可执行性（末端/身体）| ❌ | 接触不差，被 ee_body 越门 3cm 击穿 | 强（未度量）|
| 任务语义（抬升高度）| ❌（不在梯度）| xy 进度 0.98、final err 0.041，但 z_max 0.19<0.34 | 强（未度量）|

**结论**：本方法（及抓取式 SPIDER）一路优化、并用硬门死卡的是 RL **最能原谅**的维度（接触清洁、穿透）；真正翻转单 case 成败的三维（离硬门动态余量、本体可执行性、抬升语义）一个都没进上游目标。这就是"聚合稳定向本方法、但单 case 抖"的结构性原因，也是 §4.8 z-only 精修（把下游 z 门放进上游）的理论依据。

### 7.2 接触是三种相反的病

复算逐帧力后，三个 case 是需**相反修法**的三种病：box021 健康（filtered recall 0.615、幻象力 0）；box004 源接触**未被复现**（recall 仅 0.058，embodiment gap，应修手部贴合/容错）；box023 接触力**虚假**（手贴髋自碰撞，frame0 net 高达 2459N，且 `spider 0.069m vs omni 0.073m` 几乎一致——**继承自源人体姿态、非 CEM 引入**，应改代理半径/碰撞过滤，不动 CEM）。**单一标量 gap 把两种反向病混为一种，必然误导修复方向**。

### 7.3 训练-free 的 on-rails 下游预测闸

唯一真正预测下游成败的接触信号是 policy-free 的 Isaac 运动学回放探针（物体 on-rails，只量接触几何）：

| case | filtered recall | phantom force rate | max init net force | 病型 | 下游 staggered |
|---|---:|---:|---:|---|---:|
| box021 | 0.615 | 0.00 | 0 N | 干净迁移 | 0.67 |
| box004 | 0.058 | 0.197 | 0 N | 源接触未复现 | 0.48 |
| box023 | 0.667 | 0.614 | 2459 N | 自碰撞穿透 | 0.00 |

**关键**：单标量不预测下游（box023 recall 0.667 却 0/64，败在自碰撞）；必须三正交标量联合分病（recall/phantom/init-net），不合成单一 gap。这是一个便宜（无需训练）、用消费端物理、且 RL 之前即能把 case 分开的 preflight 闸。

### 7.4 局限

1. 聚合胜但仍 case-dependent；E167A 在 SUGAR 上为重分布（聚合 56→51），case 数有限，需做成统计显著。
2. 两套下游 eval 口径不统一（SUGAR staggered / Holosoma handbox），方向一致但不能直接相加；OmniRetarget partner 未在两项目全 case 同口径补齐。
3. **数据完整性风险**（rubberhand 暴露）：Box004 的 SPIDER 导出选用了与 Omni 逐字节相同的 processed trajectory——需在 export 阶段强制 `sha256(spider)≠sha256(omni)`。
4. 上游目标缺三类 downstream-critical 信号（消费端接触可见性、离硬门动态余量、抬升语义）。
5. 工程 bug：box023 初始自碰撞穿透、box004 接触标签与代理几何不一致；box023 全方法全 0。
6. 纯 CEM 框架内"接触↑↔穿透↑↔稳定性↓"三元 tradeoff 难根除，本文是缓解而非根除；双手 AND 约束目前为负、后平滑下游仅 1/4。
7. 与 DynaRetarget 的长 horizon SBTO 尚未结合——本文改任务建模、未改优化器，二者正交可叠加（future work）。

---

## 8. 结论

本文把面向单体灵巧抓取的采样式 MPC 动力学重定向（SPIDER）系统性改造为面向**人机协作搬运**的方法：以面接触带奖励 + 采样层穿透门 + 释放相位衰减 + 参考相对姿态重排序构成协作接触合规栈，以物体 PD 驱动/双手约束/足约束-平滑构成协作与运动学建模，并以 z-only 精修与下游 RL 消费端协同设计。在统一物理口径下相对 OmniRetarget 取得穿透 −0.375、真实接触 +0.119 的明确优势，并在两个独立下游 RL 项目上聚合超越（SUGAR 56/51 vs 13；Holosoma handbox +15.6pp）。除指标外，本文给出公平评测协议、"可恢复性不对称"的结构性解释，以及一个训练-free 的下游预测闸——把下一步从"继续刷接触"重定向为"把消费端物理、动态余量、抬升语义放进上游目标与评测"。与 DynaRetarget（改优化器）、ReActor（改优化范式）的结合是自然的后续方向。

---

## 参考文献

- [SPIDER] Pan et al. *SPIDER: Scalable Physics-Informed Dexterous Retargeting.* 2026.
- [OmniRetarget] *OmniRetarget: Interaction-Mesh Kinematic Retargeting for Humanoid Whole-Body Control.*
- [DynaRetarget] Dhedin, Taouil, Omar et al. *DynaRetarget: Dynamically-Feasible Retargeting using Sampling-Based Trajectory Optimization.* TUM/MIRMI, 2026.
- [ReActor] Müller, Serifi, Christen, Grandia, Knoop, Bächer. *ReActor: Reinforcement Learning for Physics-Aware Motion Retargeting.* ACM TOG 45(4), 2026.
- [HDMI] *HDMI: Whole-Body Humanoid-Object Interaction Control.*
- [ItTakesTwo] Liu et al. *It Takes Two: Learning Interactive Whole-Body Control Between Humanoid Robots.* 2025.
- [CORE4D] *CORE4D: Human-Object-Human Collaborative Manipulation Dataset.*

---

## 附录 A：贡献—实验—代码对照（可复现性）

| 贡献 | 关键实验 | 代码位置 | 关键数值 |
|---|---|---|---|
| 面接触带（窄对称）| E158→E163 | `config.py:449-461`（`symmetric_abs`）、`mjwp.py` surfaceBand | box023 raw 0.7538→0.8769；clean8 7/8 |
| gateA 穿透门 | E152/E153 | `sampling.py:270-356`、`config.py:325-332` | sweet spot (-0.010,0.10) 3/3；深穿透 -0.172 |
| 释放相位衰减 | E155/E161 | `config.py:461`（`decay_frac=0.15`）| releaseF3 0.188→0.030 |
| 姿态重排序 | E160 | `sampling.py:358-409,872-894`、`mjwp.py:2082-2086` | 摔 1→0；末端根高误差 0.042 |
| 物体 PD 驱动 | E027b | `mjwp.py:2788-2827`、`config.py:153-162` | desk005 rot 90.3°→8.8° |
| 局部坐标跟踪 | E035 | `mjwp.py:515-527,844-856`、`config.py:486-507` | 100% stable、0% penetration |
| 双手 AND 约束 | E164 | `mjwp.py:1666-1699,1178-1195`、`config.py:206-212,455-467` | raw 接触 0/3（负结果）|
| 足约束/平滑 | E166 | `sampling.py:47-93,168-213`、`config.py:366-378` | A_B2 jerk P95 ↓57%；下游 1/4 |
| 手部代理几何 | E147/E148 | `scene_act_E147_rubber_hull.xml` | 手穿透 0.4456→0.2877（接触↓，非默认）|
| z-only 精修 | E167A | `sampling.py:102-165`、`config.py:379-401`、`mjwp.py:2145-2182` | 下游 SUGAR 51/448 |
| 公平评测 + staggered | E109 / E163_RL | `scripts/eval_omni_vs_spider`、SUGAR `holosoma_like` | 二值→连续谱 |
| 数据污染修复 | E103 | scene inertial audit | 87/198 scene 污染（mass=29.632kg）|

*底稿日期：2026-06-24 · 覆盖 E001–E167 · 核心方法 E152–E167A*
