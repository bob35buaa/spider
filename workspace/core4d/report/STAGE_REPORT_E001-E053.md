# CORE4D 人机协作动力学重定向 — 阶段总结 (E001 → E053)

> 时间：2026-04-30 → 2026-05-12 (13 天)
> 实验数：53 次主实验（含 100+ 子变体），跨 16 个 phase
> 涉及方法：5 类（CEM/SBMPC、Connect-焊接、xfrc 弹簧、HDMI contact-guidance、SBTO 开环）
> Simulator：MJWP / MJWP-EQ / HDMI
> 数据：CORE4D box023, box025, bucket005/010, chair022, desk005

---

## 0. TL;DR — 当前真实状态

**视觉复查后的结论是：53 个实验中，没有任何一个产生了视觉上可接受的 loco-manipulation 轨迹。**

| 维度 | 当前最好 | 是否可用于 RL | 备注 |
|---|---|---|---|
| Body tracking（关节角） | HDMI 5.3°（box025）、E035 9.7°（bucket010） | ✅（仅 body） | 机器人能复现弯腰/行走 |
| Stability（pelvis>0.6m） | 100%（多 case） | ⚠️（条件性） | 但常常是"丢下物体走"或"靠物体支撑" |
| 物体物理跟随 | 无 | ❌ | 全部依赖 PD/Connect "假搬运" |
| 接触真实性 | 无 | ❌ | 手背接触 / 推非搬 / 穿模 |

**核心瓶颈是物理接触的产生与维持，不是 body tracking、不是 reward 权重、不是 collision margin、也不是 simulator 选型。**

因此，**现在直接进入 RL 训练是无效的**：上层 RL 没有可信的接触参考轨迹，只能学到"丢下物体走"或"被 PD 拖着的物体"。需要先在重定向层把物理接触搞出来。

---

## 1. 目标回顾

将 CORE4D（双人协作搬运 mocap）重定向为单/双 G1 人形机器人的物理合规轨迹，要求：

1. **Body**：关节角/根位姿与 ref 接近（MPKPE < 15 cm，joint_err < 5°）
2. **物体**：物体被机器人**接触力**抬起/搬运并跟随 ref（obj_pos < 12 cm，obj_rot < 10°）
3. **接触**：手与物体保持持续物理接触（contact > 80%，无穿模）
4. **稳定性**：全程不摔倒（pelvis_z > 0.6 m，>90%）
5. **数据用途**：作为 Holosoma RL pHRC 的参考运动

**失败标准**：任何一项不满足都不能直接喂给 RL。

---

## 2. 实验路线（16 phase × 5 类方法）

### 2.1 Phase 0–4（E001–E018）：从单人到双机器人，发现 connect 假象

| Phase | 实验 | 核心动作 | 结论 |
|---|---|---|---|
| P0 数据管线 | E001 | holosoma → SPIDER NPZ + scene XML | ✅ 通过 |
| **P1 单人 + 强引导** | E002–E009 | 基线 / 强 PD / 阻尼 / forearm reward / partner mocap | ❌ obj_z ≤ 0.31 m，**根因：单 G1 臂展 0.5 m < box 长 0.61 m，几何不可解** |
| P2 单人 + Mocap Partner | E010–E012 | partner 提供物理力，导出 hybrid 轨迹 | ⚠️ 运动学可行（pelvis_err 0.083 m），但物体由运动学驱动 |
| P3 修复 + 高方差 | E013–E014 | Intra-rollout mocap、增大 partner 碰撞体 | ❌ CEM 高方差，partner 力混乱 |
| **P4 双机器人 (Gibbs CEM + Connect)** | E015–E018 | Gibbs CEM 优化 nq=79、soft 2-connect、task-space reward | **❌ 假突破** — `obj_z` 升到 0.6 m 但视频证实是 connect 把物体悬浮，机器人穿模 |

**P1–P4 教训**：**一切 connect/weld 约束产生的"成功"都是假的**。E016/E017/E018 的所有 obj_z 提升都来自约束力，机器人本体没有真接触。

### 2.2 Phase 5–8（E020–E031）：从假象退回，重新理解结构性瓶颈

| Phase | 实验 | 核心动作 | 结论 |
|---|---|---|---|
| **P5 多 case 诊断 + Anchor** | E020–E024 | 5 物体 × 2 模式扫描；引入 pelvis XY/Yaw anchor | ✅ Anchor 使 pelvis_err ↓47-72%；**关键发现：行走位移本身贡献了 87-96% 的 pelvis 误差** |
| P5+ partner force | E024 | 50–90% 重力补偿 partner | ❌ 90% 时物体失重飘起 = "假搬"；CEM 不主动产生接触 |
| P6 接触奖励 | E025–E027 | hand approach reward / sustained contact / 增加 sample/iter | ⚠️ 手能接近物体（dist 0→0），但仍是"推/碰静止物体"，不是"沿 ref 搬运" |
| P7 阻尼弹簧 + xfrc | E028–E030 | xfrc_applied 控制物体 spring/damper；anchored hybrid 导出 | ❌ 物体翻转（orientation 不可控）；hybrid 导出与 RL 因果关系断裂 |
| P8 双机器人泛化 | E031 | E018 双机器人 connect 在 4 case 上重测 | ❌❌ 全部失败：Gibbs+connect 拽倒机器人 |
| **P8 Object PD Override** | E027b | 给物体加 scene_act + grav_comp + relative_euler | ✅ desk005 obj_pos=0.10 m / obj_rot=8.8°（数值漂亮） |
| P8 Contact Guidance 移植 | E027c–E027d2 | 移植 OMOMO/HDMI 的 contact guidance 到 MJWP | ⚠️ body 站着走，但物体仍由 PD 驱动 |

**P5–P8 教训**：物体 PD 控制器（包括 contact guidance 的 PD-decay-to-0）能让物体跟随 ref，但 CEM 在 PD decay 后**没有维持接触的策略**——这是后续所有失败的根因。

### 2.3 Phase 9–12（E032–E047）：CEM reward 调优的极限

| Phase | 实验 | 核心动作 | 结论 |
|---|---|---|---|
| P9 reward sweep | E032–E033 | hand approach 权重 / σ sweep / CEM budget（2048 sample） | ✅ desk005 stable=95% + Contact<15cm=91%；**但增加 sample/horizon 反而恶化 stability — tradeoff 是根本性的** |
| P10 HDMI-style reward | E034–E036 | bounded qpos / stability_penalty / **local-frame body tracking** | ✅ E035 desk005 contact=94.8%，3/3 case 不摔倒；**但 E036 关掉 hand_approach 后 contact 暴跌 7%** ⇒ body tracking 与 contact 是 tradeoff，CEM 无法同时优化 |
| P11 Contact reward 重写 | E037–E041 | box-SDF 接触奖励、动态 per-frame target、orientation gating | 🐛 发现 config bug：contact reward 一直没生效！修复后 contact=85%，但**手粘连物体 / 手背接触 / 不自然行为持续出现** |
| **P11 当前最佳 (E041c)** | E041c | additive ori（w=0.3）+ stable=100% + MPKPE=1.4 cm | ⚠️ CEM 数值最佳，但视频显示：box025 趴箱、desk005 丢桌走、bucket010 推非搬 |
| P12 Wrist freeze + sigma + SBTO | E042–E047 | 冻结手腕噪声、收紧 σ、增大 wrist 权重、移植 DynaRetarget SBTO | ❌ 全部退化或崩溃；**SBTO 开环优化 + exp-kernel reward 不兼容**；contact 上限稳定在 ~64-66% |

**P9–P12 教训**：在当前的 reward + CEM 框架下，**contact 与 stability 是互斥的**——给 hand approach 加权或收紧 σ 都会牺牲下半身平衡，而放松又会让手离开物体。CEM 在 1024 sample × 32 iter 的搜索空间里**找不到既稳又接触的点**。

### 2.4 Phase 13–16（E048–E053）：HDMI 对照与"配置不可信"教训

| Phase | 实验 | 核心动作 | 结论 |
|---|---|---|---|
| P13 碰撞盒修复 | E048 | 21/21 case 碰撞盒修正为 mesh AABB×1.05 | ⚠️ 数值上 box023 ObjPos=14 cm 看似很好；**视觉复查发现机器人摔倒，14 cm 是因为物体没动** |
| P14 HDMI workflow 对照 | E049 | 完整跑通 HDMI on CORE4D | ✅ HDMI body tracking 完爆 MJWP（Joint 5.3° vs 21.5°）；❌ HDMI object tracking 在 box025 上漂 91 cm；**eval 脚本之前用错 channel，HDMI 之前的 0.3 cm 全部是假象** |
| P14 HDMI 优化移植 | E049a-d | apply_holosoma_pd + wrist_damping + zero_noise | ❌ Stability 直接崩到 12-57%；**HDMI 三优化是协同设计，不能单独移植** |
| P15 HDMI scene 重建 | E050–E052 | euler convention 修复 / suitcase 模板重建 scene / 4 个 scene 配置矩阵 | ❌❌ 全矩阵失败：修 euler 反而恶化 5×；**E048a baseline (24 cm/7.3°/100%) 就是 HDMI 在 CORE4D 上的极限** |
| P16 碰撞盒 margin sweep | E053 | 0.90 / 0.95 / 1.00 / 1.05 × 3 case | ⚠️ 没有全局最优 margin，box025 喜欢 0.90 / bucket010 需要 1.05；**碰撞盒不是搬运失败的根因** |

**P13–P16 教训**：
- 数值指标系统性误导（摔倒/物体没动 = 低 ObjPos）
- HDMI 在 CORE4D 上的"成功"是 body tracking，不是 object tracking
- 修配置反而破坏已适应的系统（E051b 修 euler 使结果 5× 恶化，E052c suitcase+正确 euler 是 2×2 矩阵最差组合）
- 单一 hyperparameter sweep 已经触顶

---

## 3. 视觉真实 vs 数值指标 — 系统性偏差全景

> 这是本阶段最重要的方法论教训。

| 数值指标 | 数字看起来 | 视觉真实 | 失真原因 |
|---|---|---|---|
| `obj_z=0.597 m` (E017d 双机器人) | 物体抬起 112% | connect 约束悬浮 | 非物理接触力 |
| `obj_pos=14 cm` (E041c box023) | 物体追踪好 | 机器人摔倒，物体没动 | 物体静止 = 低误差 |
| `Stability=100%` (E041c desk005) | 机器人稳定搬运 | 丢下桌子自己走 | 只看 pelvis_z |
| `Contact=93%` (HDMI box023) | 手触碰物体 | 物体方向错 178° | euler 错配下的"假接触" |
| `MPKPE=0.3 cm` (HDMI box025) | 极佳 body tracking | eval 脚本读错了 ref channel | 之前与"内部 ref"对比 |

**结论：任何不结合视频的数值结论都不可信。**之后所有实验必须强制视频复核（E048-E052 视觉复查项目已经写入 EXPERIMENT_TRACKER.md 经验库）。

---

## 4. 5 类方法的失败模式总结

| 方法族 | 代表实验 | 失败模式 | 根本限制 |
|---|---|---|---|
| **MJWP CEM 单机器人** | E041c, E048 | body 站立但物体没接触 / 摔倒 | CEM 无法在采样空间中产生 sustained contact |
| **HDMI Contact Guidance** | E048a, E049 | body tracking 完美，object 漂 24-91 cm | PD-decay-to-0 后 CEM 没接力 |
| **双机器人 + Connect/Weld** | E016-E018, E031 | obj_z 漂亮但穿模/悬浮，不能泛化 | Connect 是非物理 hack |
| **xfrc / Spring / Damper** | E028-E030 | 物体翻转，orientation 不可控 | xfrc_applied 无法稳定控制 freejoint orientation |
| **SBTO 开环优化** | E047a/b | 机器人摔倒，MPKPE=1.5 m | 无 MPC 闭环反馈，不兼容 exp-kernel reward |

**共同底层限制**：所有方法都没有**主动产生抓握的机制**——机器人不知道"要抓"，只在被引导接近物体后偶然碰一下。Reward 没有显式 force-closure 项，CEM 也没有对 grasp 维度的高效探索。

---

## 5. 根因分析（4 层）

### 5.1 算法层：CEM/SBMPC 的接触搜索能力极限

- 1024 sample × 32 iter 在 nq=29（单 G1）维度的 control noise 空间里采样
- 接触是**离散事件**：手接近 → 碰到 → 摩擦力撑住 → 抬起。中间任何一步失败，整个 trajectory reward 急剧下降，CEM 看不到梯度
- **现象佐证**：增加 sample 到 2048 / iter 到 64（E033d）反而恶化 stability —— CEM 在更大空间里更容易找"丢下物体走"这种局部最优
- **HDMI 的成功来自把 object 完全交给 PD**，让 CEM 只做 body tracking。但这在 object 大、需要长距离搬运时（box025 1.5 m），PD 也跟不住

### 5.2 数据层：CORE4D 是双人协作

- `box025`（72×72×89 cm）是双人对夹任务，G1 单人臂展 0.5 m **物理不可解**（E009）
- 所有需要 lift 的 case 都假设了 partner 的同时支撑——单人 retarget 在物理上是 ill-posed 问题
- **当前的"数据筛选"不够**：53 个实验里只在最简单的 desk005 上偶尔得到数值好看的结果，其他 case 几乎全军覆没

### 5.3 物理建模层：scene/PD/euler 的脆弱耦合

- E049-E052 证明 HDMI 的"错"配置（armature=1.0 虚高、euler=xyz 178° 偏、1-sphere hand）是 CEM 适应的"特征"，任何修改都打破现有平衡
- `apply_holosoma_pd` 移植到 E041c 直接让 stability 崩到 12%（E049a-d）
- 这意味着**没有可移植的"通用最优配置"**，每套 reward + scene + PD 是一个 fragile 的局部解

### 5.4 评估层：无 force-closure / wrench-based metric

- 所有现有 metric（contact distance、obj_pos、stability）都是**间接的**
- CEM 优化的是这些代理指标，因此可以通过"丢下物体走 + pelvis_z 漂亮"来 hack
- 真正应该评估的是**手对物体的合力/力矩是否能支撑 ref 加速度**——目前没有这个 metric

---

## 6. 现有最有价值的资产

虽然 53 个实验没产出可用 RL 数据，**但留下了非常完整的负面证据 + 几项可继续利用的子能力**：

| 资产 | 状态 | 用途 |
|---|---|---|
| **完整的 CORE4D → SPIDER 数据管线** | ✅ E001 | 21 case × 2 person 全部就绪 |
| **HDMI workflow 在 CORE4D 上跑通** | ✅ E048-E052 | body tracking 5-7° 可用作 RL body 参考 |
| **Local-frame body tracking 移植** | ✅ E035 | HDMI 不摔的核心 trick，已在 MJWP 复用 |
| **Hybrid 导出格式（pelvis_err 0.083）** | ✅ E012 | 单人 + 运动学物体备选轨迹 |
| **碰撞盒修复脚本（21/21）** | ✅ E048+E053 | mesh AABP + 可调 margin |
| **统一评估 + 视频复查工作流** | ✅ E048-E052 | 防止下次再被假指标误导 |
| **53 个实验的负向证据库** | ✅ EXPERIMENT_TRACKER.md | 直接告诉下一步什么不要做 |

---

## 7. 为什么现在不能进入 RL

| 进入 RL 需要的前提 | 当前状态 |
|---|---|
| 物理合规的 reference trajectory（手有真实接触力） | ❌ 全部 reference 物体由 PD 驱动 |
| Reward 信号能区分"真搬"与"靠物体支撑站着" | ❌ 现有 metric 不能 |
| 至少 2-3 个 case 上 retarget 成功，能给 RL 一个 demonstration prior | ❌ 没有任何 case 视觉合格 |
| Reference 不存在 connect/weld 这类 RL 学不到的约束 | ❌ 双机器人结果全是 connect |

**如果现在进 RL**：智能体要么学到"丢下物体走"（因为 reference body tracking 是这样的），要么学到"等待 PD 把物体送到手边"（reference object 是这样动的）——两者都是 reward hacking 的复制品。

---

## 8. 总账

| | 数量 |
|---|---|
| 主实验（E001-E053） | 53 |
| 子变体 / sweep run | 100+ |
| 视觉合格的 trajectory | **0** |
| 数值合格但视觉不合格 | 6 (E018d2, E041c, E035, E048a HDMI, E027b, E034d) |
| 给出"此路不通"的结构性负面结论 | 12+ |
| 已修复的代码 bug（全部记入 git） | 8 (碰撞盒、euler、scene_name、ref channel、config flag、3 个 contact_guidance bug) |

---

## 9. 下一步规划详见

→ `workspace/core4d/report/NEXT_PLAN_E054+.md`
