# 下一阶段规划 — 在进入 RL 之前要做的事 (E054+)

> 前置阅读：`STAGE_REPORT_E001-E053.md`
> 核心约束：**暂不进入 RL 训练**。本阶段所有工作都在 retarget 层；目标是产出至少 1 个 case 的 视觉合格、物理真接触 reference，然后再考虑泛化和 RL。

---

## 1. 决策框架（先收口，再突破）

53 个实验的负面证据已经把 5 类方法全部排除：CEM-only / Connect-hack / xfrc-spring / 双机器人 Gibbs / SBTO 都触顶。继续在这些方向调参的边际收益≈0。

下一阶段必须做两件事：

1. **降低问题难度**：把 case 集合先收口到"几何上单人可解 + 接触简单"的子集，把"双人不可解的 case"显式从重定向 scope 移出
2. **改变 contact 产生机制**：不再依赖 CEM 在控制噪声里偶然找到 grasp，而是**主动注入 grasp 先验**（pose-grasp warmstart / 力闭合奖励 / 差分接触梯度）

不要再做：
- 新一轮 reward weight sweep（已扫了 40+ 组合，触顶）
- 新一轮 σ / sample / iter sweep（E033/E045 已证明无效）
- 新一轮 collision margin / euler convention 修复（E051-E053 已证明 fragile）
- 任何 connect/weld 双机器人方案（E031 证明不能泛化）

---

## 2. 优先级排序（4 条路径）

### Path A — 数据筛选 + 单臂可达 case 清单（**优先级最高，1-2 天内完成**）

**问题**：现在所有"难"的 case（box023/box025/chair022）都是双人协作，单 G1 物理不可解。继续在它们上调参是浪费。

**做法**：
1. 对 21 个 CORE4D case 跑一遍**几何可达性分析**：
   - 物体最大宽度 vs G1 双臂展（约 1.0 m）
   - 物体高度 vs G1 抬手最高点
   - Mocap partner 在搬运全程是否参与（如果 partner 也用力支撑 ≥ 30%，标为"双人 only"）
2. 输出 case 分级表（**预判，待 E054 量化确认**）：
   - **Tier 1 (单人可解)**：物体最大维度 ≤ 单臂展(50 cm) 或重量 < 5 kg，partner force 贡献 < 30%
     - 候选：**box023** (35-41 cm)、**bucket005**、**bucket010**、部分 **desk005** 序列
   - **Tier 2 (单人勉强)**：物体维度在 50-70 cm 之间，可推不可抬 — chair022, 矮 desk
   - **Tier 3 (结构性双人 only)**：物体最大维度 > 双臂夹持极限 (~70 cm) — **box025 (75 cm)**、高桌
     - 从单人 retarget scope 永久移出
3. 后续 E054+ 实验**只在 Tier 1**上验证，避免被 Tier 3 失败干扰

⚠️ **修正前期错误**：之前把 box023 误归 Tier 3 是错误的。box023 实测 mesh AABB 17×18×20 cm，几何上完全在 G1 单臂可达范围内。E041c 在 box023 上摔倒是 **方法失败**（CEM 无法产生接触），不是 **几何不可解**。两件事必须分开。

**成功判据**：交付 `workspace/core4d/results/case_tier_classification.csv`，明确列出每个 case 的可达性参数 + partner force 量化 + tier 分类。

**预期产出**：3-5 个 Tier 1 case 用于后续实验，**box023 与 bucket005 作为首批验证 case**。

---

### Path B — Intent-Driven Hand-Snap Warmstart（**优先级第二，3-5 天**）

**问题**：CEM 在 1024×32 采样里"偶然"碰到物体的概率太低，接触永远是脆弱的副产品。需要在 rollout 之前就让机器人手处于"已经握住"的姿态。

**核心难点（用户指出的关键）**：CORE4D mocap 本身不准 — 很多序列 ref 里 SMPL 手 mesh 与物体 mesh 距离从未 < 1 cm，凑合接近的帧也未必是真实接触帧。**"找 ref 里的真接触帧"是循环假设**：动力学重定向存在的理由就是要修正 mocap，不能反过来依赖 mocap 当 ground truth。

**重新设计的检测逻辑（不依赖"真接触"判断）**：

用 **意图检测 (Intent Detection) + 物体运动锚定 (Object-Motion Anchoring)** 替代"距离 < 1 cm"：

1. **物体运动锚定**（更可靠）— 物体的低频刚体运动比手指 mocap 准
   - 计算 ref 里物体的世界系加速度 `a_obj(t)`
   - 找到 `a_obj_z(t) > 0` 持续 > 5 帧的窗口 = 物体被向上抬 = **此时必然有外部支撑力 = 必然有接触**（无论 mocap 手在哪）
   - 这是 **物理必要性**给出的"intended grasp window"，与 mocap 手的精度无关

2. **手运动学意图**（辅助）— 即使手 mocap 不准，"手减速并停在物体附近"是稳健信号
   - 计算手相对物体的距离时间序列 `d(t)`
   - 找局部极小：`d(t)` 在 ≥ 10 帧内保持在 `d_min + δ` 内（δ ≈ 5 cm）
   - 同时 `|v_hand|(t) < threshold`（手速度低 = 不是路过）
   - 阈值用绝对距离的 **百分位数**（如 d 时间序列的 20%-quantile），不用 1 cm 硬阈值
   - 这避免"凑合接近 5 cm"被错过的问题

3. **两者交集**：物体抬升窗口 ∩ 手低速接近窗口 = **可信赖的 grasp window** `[t_start, t_end]`
   - 如果两者不重叠（数据极差），fallback 到只用物体运动窗口
   - 如果物体全程不动（推/拉任务），用手低速接近窗口

**Hand-Snap 步骤**：

4. **主导手判定（来自 E054 视频核实，必须执行）**：
   - CORE4D 是双人协作数据集，少数 case 上 person1 **只用一只手**参与（partner 出另一只）
   - E054 实测 + 视频核实 6 个 B+C case：**5 个 both-hand**（symmetry ≥ 0.94），**只有 bucket001_person1 是 single-hand**（symmetry 0.37, L=23cm vs R=62cm）
   - 判定指标：`hand_symmetry_ratio = min(L_mean, R_mean) / max(L_mean, R_mean)` 在 intent 窗口内取均值，< 0.55 = single-hand
   - **❌ 错误做法 1**：对所有 case 一律双手 snap → 会破坏 bucket001 这种单手 case（强行把右手扯到物体上）
   - **❌ 错误做法 2**：用 "L_frac"（L 比 R 更近的帧数）判定 → 在双手对称的 case 上会误判（box023 / bucket005_s2 这种两手都在 28cm 的 case，谁稍近就拿 96% 票，但实际两手都在搬）
   - **✅ 正确做法**：根据 E054 csv 的 `dominant_hand` 字段决定 snap 哪只手：
     - `dominant_hand = "L"` → 只 snap 左手到物体表面，右手保持原 ref qpos（不要碰）
     - `dominant_hand = "R"` → 只 snap 右手
     - `dominant_hand = "both"` → 双手都 snap，目标点选物体两侧

5. 在 `[t_start, t_end]` 窗口内，把选中手的 IK 目标点设为**物体当前姿态下距 ref 手位置最近的表面点**（沿表面法向外推 5 mm 防止穿模）：
   - 这一步是在主动**修正 mocap 误差** — 用物理几何（物体 mesh 表面）替代不可信的手 mocap 位置
   - IK 解的 qpos 替换 ref qpos 中的对应关节（仅 snap 手所属的链：肩-肘-腕）
6. 在窗口外（approach phase），用线性插值在原 ref qpos 与 IK 解之间过渡（避免 qpos 跳变）
7. 把这条 **修正后的 ref qpos 序列** 作为：
   - CEM 的初始 mean trajectory
   - 同时也作为 CEM body tracking reward 的新 ref（让 reward 也"信"这条修正轨迹）

**关键代码改动**：
- 复用 `workspace/core4d/scripts/analyze/case_tier_analysis.py` 输出的 csv 作为 intent window + 主导手来源（不重复实现意图检测）
- 新建 `spider/preprocess/hand_snap_ik.py` — 单/双手 IK 投影到物体表面（按 `dominant_hand` 字段切换策略）
- `spider/preprocess/ik.py`：复用现有 IK，新增 (a) surface-projection target，(b) 单链 IK（只解 shoulder-elbow-wrist 子链）
- `examples/run_mjwp.py`：新增 `warmstart_qpos_path` 与 `override_ref_qpos` 选项
- `spider/optimizers/sampling.py`：CEM 初始 mean 使用 warmstart 而不是 zero/ref

**成功判据**：在 1 个 Tier 1 case 上得到 contact > 80% × 持续 > 1 s × stable=100%，**且视频复查显示真实抓握**（不是手背贴近 / 不是 IK 把手卡在物体内部）。

**风险与对策**：
- IK 可能把手投影到物体内部 → 用法向 offset 5 mm 在表面外
- IK 可能违反 G1 关节限位 → 退化到"投影到最近的可达点"，并在 log 中标记为"几何亚最优"
- 物体 mocap 也有噪声 → 物体加速度阈值用低通滤波后的 `a_obj`
- **Tier 1 case 数据质量验证**：E054 完成时一并跑数据质量分析（每个 case 的 hand-obj 最小距离分布、obj 加速度信噪比），把"数据极差不可救"的 case 标出来 — 它们应该退化到只用 Path C（force-closure reward 学接触），跳过 Path B

**与 mocap 不可信的关系**：
> Path B 不再假设 ref 准。它假设的是更弱的命题 — "**物体被抬起 ≡ 必然有外力支撑**"（物理定律，与 mocap 精度无关），以及"**手长时间停在物体附近 ≡ 接触意图**"（运动学统计，对 mocap 噪声鲁棒）。最终 IK 是用物体几何替代手 mocap，**这正是动力学重定向应该做的事**。

---

### Path C — 力闭合 (Force-Closure) Reward（**优先级第三，5-7 天**）

**问题**：现有 contact reward 只看"距离 < 10 cm"，CEM 用手背就能满足；不约束法向力 / 摩擦锥 / 扭矩平衡，所以"假接触"无法被惩罚。

**做法**：
1. 在 mjwp 的 `get_reward()` 里加一个 wrench-based 项：
   - 读取 `data.contact.frame[i]` 与 `efc_force[i]`（MuJoCo 接触力）
   - 计算双手对物体的合力 `F_total`、合力矩 `M_total`
   - **目标**：合力 = 物体跟随 ref 加速度所需的 wrench `F_required = m·(a_ref - g)`
   - reward = `-‖F_total - F_required‖`，加权到 task reward
2. 评估时把 force-closure margin（合力是否在摩擦锥内）单独打印
3. 与 Path B 组合：warmstart 给 CEM 一个有真实接触力的初始解，force-closure reward 防止它退化

**关键代码**：
- `spider/simulators/mjwp.py`：新增 `_compute_grasp_wrench()` helper
- `spider/config.py`：新增 `force_closure_w` field
- 新建 `examples/config/override/core4d_e056_forceclosure.yaml`

**成功判据**：force-closure margin > 0（数学上能支撑物体），且视频显示物体被真实抬起。

**风险**：MuJoCo 接触力在 sub-step 之间不连续，可能 CEM 看到的力是噪声。备选：用 implicit contact model（differentiable contact）或 EWMA 平滑。

---

### Path D — 差分物理 (MJX gradient-based) 探索（**优先级第四，1-2 周，长线投资**）

**问题**：CEM 是 zero-th order 优化，对接触这种稀疏奖励无能为力。MJX 提供 differentiable physics，可以对接触力的"将要发生"做反向传播。

**做法**：
1. 把当前 retarget 问题写成 MJX trajectory optimization：
   - decision variables：每帧 ctrl
   - objective：tracking + force-closure
   - constraint：dynamics（自动满足 via MJX rollout）
2. 用 Adam / L-BFGS 优化，每步 backprop 通过 1-2 s rollout
3. 与 CEM 对比：在同一 case 上谁能产生持续接触
4. 如果有效，把 differentiable optimizer 作为 CEM 的 inner loop refinement

**关键依赖**：MJX 是否能高效跑 G1 (29 dof) 的 contact-rich 场景；可能需要 simplified 接触模型（soft sphere contact）

**成功判据**：在 1 个 Tier 1 case 上，差分优化产生 contact > 80% × force-closure margin > 0，**且收敛步数 < CEM 1/10**。

**风险**：MJX 的 contact gradient 在硬接触下不稳定；需要设计 contact smoothing。如果 1 周内做不出原型，降级到 Path A+B+C 的组合。

---

## 3. 推荐执行顺序

⚠️ Path B 与 Path C 不再是"先后"关系而是 **互补关系**：
- Path B 适用：物体有可识别的抬升/移动（接触可被物体运动反推） — bucket、box023 提起类
- Path C 适用：物体几乎不动 / mocap 信噪比极低（接触必须靠 reward 学出来） — 推、扶、按类
- E054 case 分析完成时，每个 Tier 1 case 应被标注"B-friendly"或"C-only"

```
Week 1 — 收口与建底
  Day 1-2: Path A 完成 case 分级 + 数据质量分析 → 锁定 3 个 Tier 1 case
           并标注 B-friendly / C-only
  Day 3-5: Path B 在 1 个 B-friendly case (推荐 box023 或 bucket005) 上 baseline

Week 2 — Reward 改造
  Day 6-9: Path C 实现 force-closure reward → 与 Path B 组合
  Day 10:  3 case 上完整评估，写阶段小结

Week 3 — 突破或转向
  Day 11-15: 如果 Path B+C 在 ≥ 1 case 上视觉合格 → 复制到其他 Tier 1 case，准备 RL
              如果失败 → 启动 Path D 差分物理原型
```

---

## 4. 每个实验的强制要求（吸取 E001-E053 教训）

1. **视频强制复查**：任何数值结论必须配 1 个 `mp4`，写入 log 时必须有逐帧描述（参考 `61_E048_E052_visual_reevaluation.md` 格式）
2. **多 case 同测**：单 case 上的"成功"不能作为结论（E018 box025 → E031 4 case 全失败）
3. **诚实指标**：报告 `obj_pos` 时必须同时报告 `obj_distance_traveled`（防止"物体没动 = 低误差"陷阱）
4. **物理力日志**：所有实验都必须 dump `data.contact` 与 `efc_force`，事后能离线判断接触真实性
5. **Git 纪律**：每个 E054+ 实验单独分支 `feat/E054-grasp-warmstart`，跑完合并到 `feat/dual-robot-retarget`
6. **不再调老 reward**：CEM hyperparameter 冻结在 E041c 配置；只通过新模块（warmstart / force-closure）改善

---

## 5. 何时可以进 RL

至少满足以下 3 项才考虑 RL：

- [ ] 在 ≥ 2 个 Tier 1 case 上视觉合格（持续接触 + 物体被真实搬起）
- [ ] Force-closure margin > 0 全程满足
- [ ] Reference trajectory 中无 connect/weld，物体的运动是接触力驱动的结果
- [ ] 有 ≥ 1 套统一的 reward + scene + PD 配置可以跨 case 工作（不再 per-case 调参）

只要这 4 项中有 1 项不满足，就继续在 retarget 层迭代，不要急于进 RL。

---

## 6. 失败回退方案

如果 Week 3 结束 Path B+C+D 都不成功：

- **回退方案 1**：接受 hybrid retarget（机器人物理 + 物体运动学），把 E012 的导出格式扩展到 Tier 1 全部 case，给 RL 提供 body-only demonstration（明确告诉 RL 物体接触需要它自己学）
- **回退方案 2**：转向 dual-robot 的物理协作（E018 路线），但要解决 connect 假象问题（用 force-closure 替换 connect，让 CEM 自己找接触）
- **回退方案 3**：考虑使用更大的人形机器人（H1, 1.8 m）让臂展不再是几何瓶颈，box023/025 也变成单人可解

---

## 7. 第一个具体实验（E054）

**E054 — Tier 1 Case 几何可达性 + 数据质量联合分析**

```bash
# 不需要跑物理仿真，只需要数据分析
uv run python workspace/core4d/scripts/analyze/case_tier_analysis.py \
    --cases all \
    --robot g1 \
    --output workspace/core4d/results/case_tier_classification.csv
```

输出列：

**几何分级（决定 Tier）**：
- case_name, person_id
- object_dim_max (m), object_height (m), object_mass (kg)
- robot_two_arm_span (m≈1.0), robot_max_reach_height (m)
- partner_force_contribution (% of total wrench)
- tier (1/2/3)

**Mocap 数据质量（决定方法选择 B-friendly / C-only）**：
- hand_obj_dist_min (m) — ref 里手到物体最近距离的最小值（多数 case 预期 > 1 cm）
- hand_obj_dist_p20 (m) — 20% 分位数（"接近"程度的稳健度量）
- obj_z_amplitude (m) — 物体 z 方向运动幅度（> 5 cm = 有抬升 = B-friendly）
- obj_acc_z_snr (dB) — 物体加速度信噪比（高 = 物体运动信号清晰）
- intent_window_count — 检测到的"接触意图窗口"数量（≥ 1 = B 可用）
- recommended_method:
  - `B+C` — 几何 Tier 1 + 物体有明显抬升 + intent window 可识别 → 走 hand-snap warmstart + force-closure
  - `C-only` — 几何 Tier 1 但物体几乎不动或 mocap 太差 → 跳过 warmstart, 直接 force-closure
  - `dual-robot` — Tier 3 双人协作 → 不在本阶段范围
  - `drop` — Tier 3 + mocap 不可救 → 永久排除

预期 1 天完成，作为后续 E055+ 实验的 ground truth。

**E054 同时回答两个独立问题**：
1. 哪些 case 物理上能做（几何 + partner force）
2. 哪些 case 的 mocap 质量足以让 Path B 工作（vs 必须靠 Path C 学接触）
