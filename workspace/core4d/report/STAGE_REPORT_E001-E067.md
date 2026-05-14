# CORE4D 人机协作动力学重定向 — 阶段总结 (E001 → E067)

> 时间：2026-04-30 → 2026-05-14 (15 天)
> 实验数：67 次主实验（含 130+ 子变体），跨 18 个 phase
> 涉及方法：6 类（CEM/SBMPC、Connect-焊接、xfrc 弹簧、HDMI contact-guidance、SBTO 开环、Hand-snap warmstart）
> Simulator：MJWP / MJWP-EQ / HDMI
> 数据：CORE4D box023, box025, bucket005/010, chair022, desk005, desk021

---

## 0. TL;DR — 当前真实状态

**67 个实验中，没有任何一个产生了视觉上完全合格的 loco-manipulation 轨迹**。但 R4-direct (log 87) 通过 HDMI vs MJWP 直接对比定位了根因 **不在 reward / dynamics / solver 层**, 而是 **sim t=0 pelvis quat 已经偏 ref 22° yaw** 的 init pose 流程 bug — 一个之前 53 个实验里都被忽略的问题.

| 维度 | 当前最好 | 是否可用于 RL | 备注 |
|---|---|---|---|
| Body tracking（关节角） | HDMI 5.3°（box025）、E035 9.7°（bucket010） | ✅（仅 body） | 机器人能复现弯腰/行走 |
| Stability（pelvis>0.6m） | HDMI box023 9s 稳在 0.62-0.75m ✓ | ⚠️（HDMI only） | MJWP 全部 fail |
| 物体物理跟随 | **HDMI box023 真搬运** (z 0.49→0.61m 维持) | ⚠️（HDMI only） | MJWP 全部依赖 PD/Connect "假搬" |
| 接触真实性 | 无 | ❌ | 手背接触 / 推非搬 / 穿模 |
| **MJWP 与 HDMI 行为差异根因** | **init pose mismatch (22° yaw)** | — | log 87 R4-direct 定位 |

**E054-E067 阶段的核心瓶颈不再是 reward 设计**, 而是 **MJWP env 初始化流程让 sim 第 1 帧就偏离 ref 22° yaw**. 任何 reward / actuator / solver 调参都是治标 — CEM 在 wrong init 上一直被迫找代偿姿态 (lunge / superman / handstand).

**进入 RL 仍然不可行**: 即使 HDMI 在 box023 pre-contact 完美 work, 它的 contact 阶段也失败 (用户判断 ref 左手位置发不上力), 真正的 hand-object force-closure 仍未产生.

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

## 2. 实验路线（18 phase × 6 类方法）

### 2.1 Phase 0–4（E001–E018）：从单人到双机器人，发现 connect 假象

| Phase | 实验 | 核心动作 | 结论 |
|---|---|---|---|
| P0 数据管线 | E001 | holosoma → SPIDER NPZ + scene XML | ✅ 通过 |
| **P1 单人 + 强引导** | E002–E009 | 基线 / 强 PD / 阻尼 / forearm reward / partner mocap | ❌ obj_z ≤ 0.31 m，**根因：单 G1 臂展 0.5 m < box 长 0.61 m，几何不可解** |
| P2 单人 + Mocap Partner | E010–E012 | partner 提供物理力，导出 hybrid 轨迹 | ⚠️ 运动学可行（pelvis_err 0.083 m），但物体由运动学驱动 |
| P3 修复 + 高方差 | E013–E014 | Intra-rollout mocap、增大 partner 碰撞体 | ❌ CEM 高方差，partner 力混乱 |
| **P4 双机器人 (Gibbs CEM + Connect)** | E015–E018 | Gibbs CEM 优化 nq=79、soft 2-connect、task-space reward | **❌ 假突破** — `obj_z` 升到 0.6 m 但视频证实是 connect 把物体悬浮，机器人穿模 |

**P1–P4 教训**：**一切 connect/weld 约束产生的"成功"都是假的**。

### 2.2 Phase 5–8（E020–E031）：从假象退回，重新理解结构性瓶颈

| Phase | 实验 | 核心动作 | 结论 |
|---|---|---|---|
| **P5 多 case 诊断 + Anchor** | E020–E024 | 5 物体 × 2 模式扫描；引入 pelvis XY/Yaw anchor | ✅ Anchor 使 pelvis_err ↓47-72%；**关键发现：行走位移本身贡献了 87-96% 的 pelvis 误差** |
| P5+ partner force | E024 | 50–90% 重力补偿 partner | ❌ 90% 时物体失重飘起 = "假搬"；CEM 不主动产生接触 |
| P6 接触奖励 | E025–E027 | hand approach reward / sustained contact / 增加 sample/iter | ⚠️ 手能接近物体（dist 0→0），但仍是"推/碰静止物体" |
| P7 阻尼弹簧 + xfrc | E028–E030 | xfrc_applied 控制物体 spring/damper | ❌ 物体翻转（orientation 不可控）|
| P8 双机器人泛化 | E031 | E018 双机器人 connect 在 4 case 上重测 | ❌❌ 全部失败 |
| **P8 Object PD Override** | E027b | 给物体加 scene_act + grav_comp + relative_euler | ✅ desk005 obj_pos=0.10 m / obj_rot=8.8°（数值漂亮） |
| P8 Contact Guidance 移植 | E027c–E027d2 | 移植 OMOMO/HDMI 的 contact guidance 到 MJWP | ⚠️ body 站着走，但物体仍由 PD 驱动 |

### 2.3 Phase 9–12（E032–E047）：CEM reward 调优的极限

| Phase | 实验 | 核心动作 | 结论 |
|---|---|---|---|
| P9 reward sweep | E032–E033 | hand approach 权重 / σ sweep / CEM budget | ✅ desk005 stable=95% + Contact<15cm=91%；**增加 sample/horizon 反而恶化 stability** |
| P10 HDMI-style reward | E034–E036 | bounded qpos / stability_penalty / **local-frame body tracking** | ✅ E035 desk005 contact=94.8%，3/3 case 不摔倒；**E036 关掉 hand_approach 后 contact 暴跌 7%** |
| P11 Contact reward 重写 | E037–E041 | box-SDF 接触奖励、动态 per-frame target、orientation gating | 🐛 contact reward 一直没生效 bug 修复后 contact=85% |
| **P11 当前最佳 (E041c)** | E041c | additive ori（w=0.3）+ stable=100% + MPKPE=1.4 cm | ⚠️ CEM 数值最佳，但视频显示：box025 趴箱、desk005 丢桌走、bucket010 推非搬 |
| P12 Wrist freeze + sigma + SBTO | E042–E047 | 冻结手腕噪声、收紧 σ、移植 SBTO | ❌ 全部退化或崩溃 |

### 2.4 Phase 13–16（E048–E053）：HDMI 对照与"配置不可信"教训

| Phase | 实验 | 核心动作 | 结论 |
|---|---|---|---|
| P13 碰撞盒修复 | E048 | 21/21 case 碰撞盒修正为 mesh AABB×1.05 | ⚠️ **视频复查发现机器人摔倒，14 cm 是因为物体没动** |
| P14 HDMI workflow 对照 | E049 | 完整跑通 HDMI on CORE4D | ✅ HDMI body tracking 完爆 MJWP；❌ HDMI object 在 box025 上漂 91 cm |
| P14 HDMI 优化移植 | E049a-d | apply_holosoma_pd + wrist_damping + zero_noise | ❌ Stability 直接崩到 12-57%；**HDMI 三优化是协同设计** |
| P15 HDMI scene 重建 | E050–E052 | euler convention 修复 / suitcase 模板 | ❌❌ 全矩阵失败 |
| P16 碰撞盒 margin sweep | E053 | 0.90 / 0.95 / 1.00 / 1.05 × 3 case | ⚠️ 没有全局最优 margin，碰撞盒不是搬运失败的根因 |

### 2.5 Phase 17（E054–E061）：Hand-snap warmstart 路线 + 3-box port 回归发现 ⭐ 新增

| Phase | 实验 | 核心动作 | 结论 |
|---|---|---|---|
| **P17a Case Tier 分析** | E054 | 21 case 几何可达性 + mocap 数据质量联合分析 | ✅ 分 4 tier, 优先级排序; box023 / bucket005_s2 进入 Tier-A 候选 |
| **P17b Hand-snap warmstart 首验证** | E055 | box023 hand-snap warmstart (Path B) | ✅ snap 后 contact=89%, hand 离物体 0.014m, box 抬起 |
| P17c Multi-case hand-face 诊断 | E056 | 多 case 检查 hand-face 对齐 | 🔬 发现部分 case 手方向反了 (palm_normal 错), 修正方向 |
| P17d bucket005_s2 hand-snap | E057 | bucket005_s2 重复 E055 流程 | ✅ 6/6 Claims, hand contact 100%, contact dist=0.014m |
| **P17e Path B-CEM 首跑** | E058 | bucket005_s2 + snap warmstart 喂入 CEM | ❌ **Claims 2/6** — 流水线 OK 但 sim 双方都摔倒, snap 不能弥补 CEM 接触失败 |
| P17e Path D 区分 | E059 | box023 + snap warmstart CEM | ❌ Baseline + Warm 都摔倒, **warmstart 不解决 CEM 接触搜索能力** |
| **P17f 数据层修复 baseline 重测** | E060.0 | E041c on box023+bucket005_s2 with fixed data | ❌ 数据层修复未实质改善 |
| P17f no-ori ablation | E060.1 | contact_hdmi_ori_weight=0 | ❌ FAIL 但反直觉 case-divergent 发现 |
| P17f case-correct palm | E060.2 | per-case palm_normal +x | ❌❌ CATASTROPHIC FAIL 全面退化 |
| **🚨 P17g 3-box port 回归发现** | log 74 | E060 系列对比时发现 box025 历史 baseline 也已 regression | 🚨 **CRITICAL — 3-box hand port (commit fa2e181) 在 E041c 校准的 box025 上也 regression, E060 整套对比基础崩塌** |
| **P17h sphere 复原** | E061 | 回退 hand collision sphere | ✅ pelvis_min 0.672m, Stable 100% — **决定性确认 fa2e181 的 3-box port 是 box025 regression 的唯一根因** |
| P17i 战略转向 | log 76/77 | 用户基于 4-cell 矩阵推翻 A/B 分析 | 🔄 **E041c 完全是 box025 sphere 过拟合, 没有任何泛化性; box025 真问题是 G1 臂展不够 (物理硬限制不易改善), 转 X1+X2 reward 泛化方向** |

**P17 教训**:
- **教训 #5**: hand-snap warmstart 解决 hand-object 几何对齐, 但**不解决 CEM 接触搜索能力** (E058/E059)
- **教训 #6** (regression discovery): **修一个东西时必须 verify baseline 不退化** — fa2e181 (3-box port) 改善某些 case 但破坏 box025 history baseline, 整个 E060 对比变得无意义
- **教训 #7**: E041c 是单 case 过拟合, 任何"baseline" 都需多 case verify 后才能信
- **教训 #X**: G1 臂展物理硬限制 — box025 双人协作 mocap 单人 retarget 是 ill-posed problem

### 2.6 Phase 18（E062–E067 + R4-diag）：X1+actuator+partition 三元 ablation 全失败 → init pose bug 定位 ⭐ 新增

#### E062: X1 Auto palm_normal on sphere — Mixed result (起点)

| 实验 | 核心动作 | box025 (regression guard) | box023 (target case) | 结论 |
|------|---------|---------------------------|----------------------|------|
| E062 | 用 `compute_palm_normal.py` 自动计算每 case palm_normal (audit §2.2 算法), box023 得 [+1,0,0] | pelvis_min 0.688m vs E061 0.672m (+1.6cm), stable 100% — **完美 self-consistency** | pelvis_min **0.058m** ❌ (E048 0.193m, **-13.5cm 退化**), 但 mean 0.602m + stable 77.9% + max 0.828m 全面改善, novel "carry attempt + fall + push-up recovery + stand" 行为 | ⚠️ box025 ✓, box023 PARTIAL — 真"尝试搬运"但摔倒 |

**重大里程碑** — E062 是 53 实验中**唯一一个 sim 真的尝试搬运 box023** 的实验. final box vs ref 仅偏 9.8cm. 但 carry→place 转换 (t=2.0-2.7s) 摔倒.

#### E063-E064: Reward weight 调参三次失败 — 触发 3-strike

| 实验 | 假设 / 改动 (vs E062) | box023 失败模式 | box025 regression |
|------|----------------------|-----------------|-------------------|
| **E063** Tier 1 | stability_penalty_scale 0.0→1.0 + task_obj 1.0→0.5 | "**superman lunge**" — pelvis_min 0.058→0.192m (+13.4cm 改善但 C1 仍 FAIL), sim 几乎平躺扑向 box | ✓ PASS (R1/R2/R3) |
| **E064** Tier 2 | + root_σ 0.5→0.3 + contact_gain 5.0→3.0 + threshold 0.55→0.65 | "**lie-down-and-stay**" — pelvis_min 0.178m, post-intent 永不起身, 单膝跪深 kneel | R1 边际 FAIL (0.643 vs 0.65, -7mm) |

**3-strike triggered**: E062 fall / E063 superman / E064 prone — 同 reward 框架内调 weight 3 次都让 CEM 找到不同 dead-end pit.

**用户视觉介入** (log 82): "box023 在弯腰之前 (pre-contact, t=0-2s) 一只脚就抬起 60cm 已经处于 lunge 单脚撑". 提取 trace_ref body 索引验证: box023 ref 全程双脚 z=0.00-0.05m, sim 全程一脚 z=0+另一脚 z=0.15-0.62m. **元凶不在 contact 阶段, 在 pre-contact body tracking**.

**用户提供 HDMI A/B 对照** (log 82 §10): `run_hdmi.py` 在 box023 pre-contact body tracking **完美** (sim 双脚平地弯腰), 同 ref motion + 同 G1 robot + body-tracking 公式逐字 port. 元凶必在 MJWP 比 HDMI 多的 reward + config.

#### E065-E067: 三元 ablation 全 FAIL — 揭示元凶在更深层

| Round | 实验 | 假设 (基于 log 82 HDMI diff) | B1 = max foot_z [0-2s] | 失败模式 |
|-------|------|------------------------------|------------------------|---------|
| baseline | E063 | (E062 + Tier 1) | **0.48m** | superman lunge |
| **R1 / E065** | A: drop task_obj (scale=0)<br>D: HDMI exp form `exp(-err/0.5)` | "MJWP `task_obj=-L2` unbounded 是元凶" | A=0.30m (改善 38% ✗)<br>D=0.69m (反恶化 ✗) | A: deep dive + recover<br>D: lunge + recover |
| **R2 / E066** | + 软 actuator (kp 500→20, kp_rot 50→0.3, decay 1.0→0.85) | "actuator 拉手→抬腿 lunge" | A=0.53m ✗<br>D=0.75m ✗ | A: 不搬箱 (C4 jumped 3.10→**112cm**)<br>D: C4=138cm 更糟 |
| **R3 / E067** | + narrow body partition (lower 12→6, upper 17→6) | "MJWP `error.mean()` 把脚约束稀释 2×" | N=**1.30m** ❌❌<br>NS=1.25m ❌❌ | N: **handstand** 倒立, C3=6.9% |

**3 ablation 全 FAIL, 每次产生新 failure mode** → 元凶不在 reward / dynamics / partition 任一单点.

**触发 ✋#3 + #5** (plan §6 暂停 trigger): handstand 是新 failure mode 第 1 次出现 + 4 round 累计协议.

#### R4-direct (log 87): HDMI vs MJWP 直接 trace 对比 — INIT POSE BUG smoking gun ⭐⭐

用户授权 R4-direct, 用现成 `workspace/core4d/results/E052/E052c_box023_euler_fix/trajectory_hdmi.npz` 不训练直接对比.

**关键发现 1**: HDMI 在 box023 pre-contact 完美 work
- B1 = **0.066m** ✓ (vs MJWP-E062 0.48m, **7× 高**)
- pelvis 9s 稳在 0.62-0.75m, 没摔
- 物体 z 0.49→0.61m **真搬起来 + 维持** 9 秒, 不是 actuator drag

**关键发现 2**: HDMI ctrl ≈ ctrl_ref + 13% residual (PPO learned 平衡 prior)
- HDMI: |ctrl-ctrl_ref|.mean=0.17, ratio 0.36 (PPO 学到小残差)
- MJWP CEM: 已 init `ctrls=ctrl_ref` ✓, noise scale 实际 0.025-0.05 比 HDMI residual 0.13 **还小** → **噪声不是元凶** (推翻 log 86 candidate C)

**关键发现 3 (smoking gun)**: t=0 pelvis quat
| Source | wxyz | euler yaw | 偏 ref |
|--------|------|-----------|--------|
| **ref** (kinematic.npz) | (0.657, -0.029, 0.006, 0.753) | +97.8° | 0° |
| **HDMI sim** (E052c) | (0.657, -0.034, 0.009, 0.753) | +97.7° | **-0.1°** ✓ |
| **MJWP-E062 sim** (E063 npz) | (0.504, -0.005, 0.027, 0.863) | +119.5° | **+21.7°** ❌❌ |

接下来 2 帧 (t=0.07s) MJWP 又再漂 30° 到 yaw=148°. HDMI 同期保持 ±0.5°.

**已 verify (排除)**: ref qpos[0] 数据正确, init code path (`d_act.qpos[:]=ref[0,:42]` mj_forward 后 pelvis xquat 完全对齐), warmstart='', ref qvel=0.

**剩余怀疑** (未 verified): 
- (a) `mjwarp.put_data` GPU 转换 lossy
- (b) 保存的 sim qpos[0,0,:] 是 post-1-step 物理 (不是 init)
- (c) **robot scene_act kp=500 + obj kp=20 在 init target≠state 蹬反作用力推 pelvis**
- (d) 物体 actuator init 不准 (object 通过手→pelvis 链反推 robot)

**P18 教训汇总** (新):
- **#9** (log 80): stability_penalty(threshold=0.55m) 只惩罚 pelvis 高度不约束 torso 朝向, CEM 找到"水平 superman lunge"局部最优
- **#10** (log 81): 同 reward 框架 weight 调参 3 次都让 CEM 找到 NEW local optimum, 触发 3-strike rule
- **#11** (log 82): 用户视觉 catch + HDMI A/B 对照是定位元凶最快的方法; reward 中 unbounded/总是 active 项扰动 CEM 比 weight 失衡更 fundamental
- **#12** (log 84): 调 reward 之前先对齐 dynamics (actuator/PD/joint limits) baseline
- **#13** (log 85, 推翻): "reward.mean 下 body 数加多反向稀释" — **实际反过来**, more bodies HELP CEM
- **#14** (log 85): soft actuator 是 trade-off, 必须配套 task_obj 加大补偿失去的 actuator-driven tracking
- **#15** (log 86): more body tracking constraints HELP CEM 即使有 mean dilution; HDMI 用少 body 因 PPO 已学 prior, MJWP CEM 没 prior 需要更多约束
- **#16** (log 86): 3 round ablation 全失败强烈暗示元凶在更深层 (solver / time res / noise / init), 不在 reward / dynamics 单点
- **#17** (log 87): **对比 sim 和 ref 的 init pose 必须 dump quaternion 全维度 (不只 pelvis_z)** — 我之前 (log 84) 说 "init 对齐 OK" 是因为只看 z; quat 22° 错误早就发生但没注意, 直到 R4-direct 才挖到

---

## 3. 视觉真实 vs 数值指标 — 系统性偏差全景

> 这是本阶段最重要的方法论教训 (覆盖 E001-E067 全部).

| 数值指标 | 数字看起来 | 视觉真实 | 失真原因 |
|---|---|---|---|
| `obj_z=0.597 m` (E017d 双机器人) | 物体抬起 112% | connect 约束悬浮 | 非物理接触力 |
| `obj_pos=14 cm` (E041c box023) | 物体追踪好 | 机器人摔倒，物体没动 | 物体静止 = 低误差 |
| `Stability=100%` (E041c desk005) | 机器人稳定搬运 | 丢下桌子自己走 | 只看 pelvis_z |
| `Contact=93%` (HDMI box023) | 手触碰物体 | 物体方向错 178° | euler 错配下的"假接触" |
| `MPKPE=0.3 cm` (HDMI box025) | 极佳 body tracking | eval 脚本读错了 ref channel | 之前与"内部 ref"对比 |
| **`pelvis_min=0.058m` (E062 box023)** | **catastrophic fall** | **sim 真的尝试搬运 task 完成 obj err=9.8cm, 然后摔倒+恢复** | **只看 min 漏掉了完整 task 行为** |
| **`pelvis_min_intent=0.192m` (E063)** | improved 13cm | "superman lunge" 水平扑向 box | stability_penalty 只惩罚 z 不惩罚 torso ori |
| **`stable%=93% C2/C3 PASS` (E066-A)** | 整体稳定 | sim "弃箱直立" 不搬 box (C4=112cm) | C2/C3 是后段 recovery, 没看 task completion |
| **`B1=0.30m` improvement (E065-A)** | reduce foot lift 38% | 仍 lunge, sim t=0 已经偏 ref 22° yaw | **真元凶在 init pose, B1 改善只是次级影响** |

**更新结论 (E054-E067 阶段)**: 任何不结合视频 + 不 dump init pose quaternion 的数值结论都不可信. 之后所有实验必须强制:
1. 视频复核 (规则 9)
2. dump init pose 全 quat 跟 ref 对比 (新)
3. 对比同 case 已 work workflow (HDMI vs MJWP 直接 trace, 不只看公式)

---

## 4. 6 类方法的失败模式总结

| 方法族 | 代表实验 | 失败模式 | 根本限制 |
|---|---|---|---|
| **MJWP CEM 单机器人** | E041c, E048 | body 站立但物体没接触 / 摔倒 | CEM 无法在采样空间中产生 sustained contact |
| **MJWP CEM + reward ablation** | E063-E067 (8 变体) | 三元素都失败: form/actuator/partition | **真元凶是 init pose 22° 偏离, reward 调参治标** |
| **HDMI Contact Guidance** | E048a, E049, E052c | body tracking 完美, object 漂 24-91 cm; box023 pre-contact 完美但 contact 阶段失败 (ref 左手位置不对) | PD-decay-to-0 后 CEM 没接力; ref 数据本身有问题 |
| **双机器人 + Connect/Weld** | E016-E018, E031 | obj_z 漂亮但穿模/悬浮 | Connect 是非物理 hack |
| **xfrc / Spring / Damper** | E028-E030 | 物体翻转 | xfrc_applied 无法稳定控制 freejoint orientation |
| **SBTO 开环优化** | E047a/b | 机器人摔倒, MPKPE=1.5 m | 无 MPC 闭环反馈, 不兼容 exp-kernel reward |
| **Hand-snap warmstart (Path B)** ⭐ 新 | E055/E057/E058/E059 | snap 后 hand-object 几何对齐 ✓, 但喂入 CEM 后双方仍摔倒 | snap 解决几何不解决 contact 搜索能力 |

**共同底层限制 (E054-E067 阶段更新)**:
- 之前结论 "所有方法都没有主动产生抓握的机制" 仍然成立
- **新发现**: 即使 reward / dynamics / solver 完全对齐 HDMI, MJWP env init 阶段 (sim t=0 pelvis 偏 ref 22°) 已经决定了 sim 必然偏离 — CEM 一直在试图代偿这个偏离, 找出 lunge / handstand / superman 都是代偿姿态

---

## 5. 根因分析（5 层 — 新增 init 层）

### 5.1 算法层：CEM/SBMPC 的接触搜索能力极限

(同 E001-E053 总结)
- 1024 sample × 32 iter 在 nq=29 维度的 control noise 空间里采样
- 接触是离散事件: 手接近 → 碰到 → 摩擦力撑住 → 抬起
- 增加 sample/iter 反而恶化 stability — CEM 在更大空间更易找局部最优

### 5.2 数据层：CORE4D 是双人协作

(同 E001-E053)
- box025 (72×72×89 cm) 双人对夹 G1 单人臂展 0.5m **物理不可解** (E009)
- box023 ref 左手位置不对 (用户判断, 待 R5 验证) → HDMI 即使 pre-contact 完美 contact 阶段也失败

### 5.3 物理建模层：scene/PD/euler 的脆弱耦合

(同 E001-E053)
- HDMI "错"配置是 CEM 适应的特征
- 没有可移植的"通用最优配置"

### 5.4 评估层：无 force-closure / wrench-based metric

(同 E001-E053 + 新)
- 现有 metric 都是间接的, 容易 hack
- **新增**: B1 (pre-contact 双脚高度) 是 P18 引入的新 metric, 对比 ref 双脚平地来抓 lunge — 但 B1 改善 ≠ 真 fix, init pose bug 才是元凶

### 5.5 ⭐ Init 层 (P18 新发现): MJWP env 初始化流程偏离

- sim t=0 pelvis quat 偏 ref 22° yaw (HDMI 同期 0.1°)
- 接下来 2 帧再漂 30° → yaw 148°
- CEM 一直在试图代偿这个 init 偏离, 找的"最优"都是代偿姿态
- 已排除 ref 数据 / init code path / warmstart / qvel 问题
- **剩余怀疑**: actuator 强 (kp 500/20) 在 init target≠state 蹬反作用力 / mjwarp put_data 转换 / 保存的 qpos[0,0,:] 是 post-1-step

---

## 6. 现有最有价值的资产 (E054-E067 更新)

| 资产 | 状态 | 用途 |
|---|---|---|
| **完整的 CORE4D → SPIDER 数据管线** | ✅ E001 | 21 case × 2 person 全部就绪 |
| **HDMI workflow 在 CORE4D 上跑通** | ✅ E048-E052 | body tracking 5-7° 可用作 RL body 参考 |
| **HDMI box023 trajectory** ⭐ 新 | ✅ E052c | pre-contact 完美 (B1=0.066m), 物体真搬 z 0.49→0.61, R4-direct 用作 ground-truth |
| **Local-frame body tracking 移植** | ✅ E035 | HDMI 不摔的核心 trick |
| **Hybrid 导出格式** | ✅ E012 | 单人 + 运动学物体备选 |
| **碰撞盒修复脚本 + sphere 复原** | ✅ E048+E053+E061 | mesh AABB + 可调 margin + 3-box port 回退 |
| **Case Tier 分析** ⭐ 新 | ✅ E054 | 21 case 几何 + mocap 数据质量分级 |
| **Hand-snap warmstart pipeline** ⭐ 新 | ✅ E055-E057 | hand-object 几何对齐, contact dist <2cm |
| **统一评估 + 视频复查工作流** | ✅ E048-E067 | + B1/B2 pre-contact foot z 新指标 (E065+) |
| **`compute_palm_normal.py` 自动算法** ⭐ 新 | ✅ E062 | audit §2.2 ref-motion proximity-windowed dot product |
| **`task_obj_use_exp` config flag (backward-compat)** ⭐ 新 | ✅ E065 commit 4b46d6d | spider/config.py + spider/simulators/mjwp.py |
| **R4-direct HDMI vs MJWP diagnosis pipeline** ⭐ 新 | ✅ log 87 | foot_z + pelvis + obj_z trace, init pose quat dump |
| **67 个实验的负向证据库** | ✅ EXPERIMENT_TRACKER.md | + 17 个教训 |

---

## 7. 为什么现在仍不能进入 RL

(同 E001-E053 + 新增)

| 进入 RL 需要的前提 | 当前状态 |
|---|---|
| 物理合规的 reference trajectory（手有真实接触力） | ❌ 全部 reference 物体由 PD 驱动 |
| Reward 信号能区分"真搬"与"靠物体支撑站着" | ❌ 现有 metric 不能 |
| 至少 2-3 个 case 上 retarget 成功 | ❌ HDMI 在 box023 pre-contact 完美但 contact 阶段失败 (ref 数据问题), MJWP 全失败 |
| Reference 不存在 connect/weld 这类 RL 学不到的约束 | ❌ 双机器人结果全是 connect |
| **Sim env init 状态跟 ref 一致** ⭐ 新 | ❌ MJWP sim t=0 已偏 ref 22° yaw |

---

## 8. 总账 (E001-E067)

| | 数量 |
|---|---|
| 主实验（E001-E067） | 67 |
| 子变体 / sweep run | 130+ |
| 视觉合格的 trajectory | **0** (但 HDMI box023 pre-contact 是首次"接近合格") |
| 数值合格但视觉不合格 | 8 (E018d2, E041c, E035, E048a HDMI, E027b, E034d, E063 superman, E066-A 弃箱) |
| 给出"此路不通"的结构性负面结论 | 18+ (新增 6: 3-box port regression, X1+X2 都不解决泛化, task_obj form 不是元凶, actuator 不是元凶, partition 反方向, init pose bug 才是真元凶) |
| 已修复的代码 bug（全部记入 git） | 9 (碰撞盒、euler、scene_name、ref channel、config flag、3 个 contact_guidance bug、3-box port 回退) |
| 教训 (Lessons #1-#17) | 17 |
| Git commits | 100+ on `feat/dual-robot-retarget` 分支 |

---

## 9. R5 候选方向 (R4-direct 后等用户决策)

按侵入性 / 验证速度排序:

| 方案 | 侵入性 | 验证速度 | 描述 |
|------|-------|---------|------|
| **E** Minimal isolation test | 极低 (新脚本) | **30s** | scene_act + ref qpos[0] + zero ctrl, 跑 N 步看 pelvis 漂. 换变量 (kp=0/500, ctrl=zero/ref) 隔离 (a)-(d). 不需训练 |
| **C** snap 第 1 frame 到 ref | 低 (yaml/小代码) | 33min (1 训练) | 第 1 sim_step 后强制 `data_wp.qpos[:]=qpos_ref[1]` 重设. 简单粗暴, 验证假设最快, 不诊断原因 |
| **B** init_kp_warmup ramp config | 中 (spider/mjwp.py) | 33min | 第 1-N frames actuator kp ramp 0→full, 给 sim settle. 如果 (c) 是真因会修好 |
| **A** yaml + 改 scene_act XML init kp | 高 (XML edit + git add -f + snapshot) | 33min | robot kp 500→100 + obj kp 20→2. 入侵性大, scene_act 要重新 snapshot, 但能直接验证 |
| **D** fallback 到 HDMI workflow | 极高 (换 solver) | — | HDMI 在 contact 阶段也失败 (ref 左手发不上力), 不是真 fix box023 |

**建议**: E (minimal test) 最快定位真原因, 然后视结果选 B/C/A.

---

## 10. 下一步规划详见

→ `workspace/core4d/log/87_R4_HDMI_diagnosis_init_pose_bug.md` (R4 诊断完整记录)
→ `workspace/core4d/EXPERIMENT_TRACKER.md` (滚动更新)
→ R5 待用户决策后展开
