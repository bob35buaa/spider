# ReActor 对本课题（CORE4D 人机协作重定向 / loco-manipulation）的启发

阅读对象：`paper/Müller 等 - 2026 - ReActor Reinforcement learning for physics-aware motion retargeting.pdf`
作者：Disney Research（Müller, Serifi, Christen, Grandia, Knoop, Bächer），ACM TOG Vol.45 No.4 (2026)。
日期：2026-06-05
本课题相关 memory：`project_contact_metric_pitfall`（−2.9pp 恒定差距）、`project_physics_augmentation`（改另一侧力增广）、`project_spider_core4d`。

---

## 0. ReActor 一句话 + 核心方法

**把"运动重定向"建模成 bilevel 优化：上层优化"重定向参数 p"，下层用 RL 训练"跟踪策略 π_φ"，两者联合优化、互相适配，从而消除脚滑/自穿透/地面穿透等物理伪影，产出可直接喂给下游 imitation/RL 的高质量参考。**

关键设计：
1. **Retargeting ≠ Imitation**（论文 p2 明确区分）：imitation 是跟踪一个**给定**参考；retargeting 是**生成**一个适合目标形态的参考。ReActor 解决的是前置的"生成参考"问题。
2. **Bilevel 公式**（Eq.1）：`min_p L(p, φ*(p))  s.t.  φ*(p) = argmax_φ R(p,φ)`。用 single-loop TTSA（双时间尺度）在每个 RL 迭代里同时更新 p，并推导了一个**简化梯度估计**（Eq.4-7）绕开 implicit function theorem / 逆 Hessian。
3. **重定向参数化**（Fig.3, Eq.8-13）：用户只给**稀疏的 source-target 刚体对**（nominal/T-pose 下），系统自动抽全局 scale + nominal TF；再叠加可学习的 `p_pos, p_ori`（局部位姿偏移）+ **逐 motion 的竖直偏移 `p_z`**（专门修 noisy contact 导致的 floating/penetration）。约束在凸集 P 内（范数上界 `δ_pos/δ_ori/δ_z`，Eq.14）。
4. **下层 RL**（Sec.6, PPO）：action = 关节 PD setpoint + **作用于 robot root 的辅助 wrench（Residual Force Control, RFC）**（Eq.18），50Hz；对 wrench 用量做**惩罚 + 连续 deadband**（Eq.19）；引入 **retargeting phase ψ_t**（0→1，episode 开头暂停参考、把机器人挪到起始位姿，并用于 reward blending / 数据过滤）。RSI 被"学习初始化"替代。
5. **机器人**：Unitree G1（1.27m/35kg/29DoF）、Lima（小型自定义）、**ANYmal D 四足**（证明跨形态）。源是 SMPL/AMASS。
6. **指标**（Table 4）：Ground/Self penetration（time+depth）、Foot sliding（vel）、Foot floating（height）、**下游 RL success rate** + root pos/ori/joint RMSE。**success = 策略跑完不触发终止**（root 偏移>1m 或 geodesic 朝向>45°，附录 A，沿用 OmniRetarget/Yang2025a 判据）。
7. **结果**：在 penetration / foot 伪影 / 下游 RL 成功率上全面超 GMR、OmniRetarget（G1 success 97.45% vs Omni 95.51% vs GMR 89.93%），且 ground/self penetration **设计上为 0**。
8. **外力惩罚是最敏感超参**（Fig.8）：惩罚↑ → 物理真实性↑ 但极端动作失败↑；惩罚↓ → 重定向更准但物理可信度↓。是一个**可刻画的 realism-vs-feasibility 前沿**。

---

## 1. 最强启发：它正面回答了你"−2.9pp 恒定差距"的根因 ★★★

我们 memory（`project_contact_metric_pitfall`）已代码级证明：**SPIDER 把 OmniRetarget 的输出当作冻结的跟踪参考，于是在以该参考为锚的接近度指标上结构性地"逼近但不能超过"，差距恒定。**

**ReActor 的整篇论文就是这个问题的解药**：不要冻结参考，把参考本身变成**可学习参数**，与跟踪器联合优化。它的 RL-only 消融（Fig.6/7：参考参数静止）恰恰复现了我们的处境——tracking reward 更低、伪影更多；开了 bilevel（co-adapt 参考）后 reward 升、伪影降。

> **直接结论**：要真正抬高 SPIDER 的接触/接近度（而非被物理薄壳永久扣分），方向**不是调 reward 权重，而是修/重优化 reference**——这正是我们 v2 报告 B6 "ref repair / GT 指尖目标" 路线。ReActor 给了它一个**形式化的 bilevel 框架**和一个**可行的简化梯度**。

**落到 SPIDER（CEM 而非 RL）的可行近似**：SPIDER 下层是 sampling-MPC（CEM），没有梯度。可以做**交替优化**近似 ReActor 的 bilevel：
- 外层：优化一小组 reference 偏移参数（如 hand target offset、ReActor 式的逐 case 竖直 `p_z`、object 接触点偏移）——可用无梯度搜索或有限差分；
- 内层：固定参数重跑 CEM；
- 迭代到 reference 与物理解互相收敛。
这比直接移植 bilevel-RL 改动小得多，且与现有 CEM 基础设施兼容。

---

## 1.5 基础概念厘清：sampling-MPC 是什么、reward 怎么起作用、交替优化怎么做

（这一节展开 §1 落地所需的方法论前提，代码核查自 `spider/optimizers/sampling.py`。）

### (a) sampling-MPC 本质是一个优化问题吗？是。

它求解的是一个标准的最优控制问题（argmax）：

```
    U* = argmax_U  R(U) ,     U = (u_0, u_1, ..., u_{H-1})
```

- 决策变量 U：未来 horizon_steps 步的机器人控制序列，形状 (H, num_actions)。
- 目标 R(U)：把这串控制喂进物理仿真 rollout 出一条轨迹，按 reward 累加成一个标量。
- 约束：MuJoCo 物理动力学本身 + 可选的 SDF 穿透 safety gate。

和"梯度下降优化网络权重"是同一类问题（都是 argmax/argmin），区别只在用什么算法去解。

### (b) 是什么算法？有没有梯度下降？—— 是 CEM/MPPI，零阶，无梯度。

代码里叫 DIAL-MPC（`sampling.py:339`），属 CEM / MPPI 家族。一次 `optimize_once` 迭代（对照代码行）：

```
  ① 采样     sample_ctrls (:352)      在当前均值 ctrls 周围加高斯噪声，撒 num_samples(如2048) 条候选
  ② 评估     rollout (:364)           每条候选并行跑一遍物理仿真，得标量 reward rews
  ③ 选精英   topk (:252)              取 reward 最高的 elite_fraction(10%) 条
  ④ 加权     softmax (:257-259)       精英按 softmax(reward/temperature) 算权重，reward 越高权重越大
  ⑤ 更新均值 (weights*samples).sum (:451)  新控制序列 = 精英的加权平均 → 下一轮的中心
```

关键认知：**它从不计算 ∂R/∂U**。靠"多撒点 → 看谁好 → 把分布往好的那撮挪"逼近最优，是零阶 (derivative-free) 优化。
⇒ reward 只用于排序和加权，**不需要可微**。这正是 SPIDER 能用不可微奖励（接触判定、穿透、SDF 门）的原因；而 ReActor 那种 bilevel 才必须费力推导"简化梯度"。

  对比：
  - 梯度下降：算斜率，沿斜率走一步。需 reward 对参数可微。
  - CEM(SPIDER)：撒一片点，挑最好一撮，把采样分布往它们那儿挪。reward 只需能算数值、能比大小。

### (c) reward 怎么作用于整个系统？—— 只评分选精英，不反传、不更新任何网络。

与 RL 的 reward 作用方式完全不同：

- SPIDER 系统里**没有策略网络**；reward 不反向传播、不更新任何权重。
- reward 唯一作用 = 上面 ②→③→④ 的"评分 → 选精英 → 定权重"，决定这一轮 2048 个候选里均值往哪几条挪。
- 所以"调 reward 权重"在 SPIDER 的物理意义 = **改变什么样的控制序列会被评为'精英'**，从而改变 CEM 收敛到的那条轨迹。

这也解释 §1 的恒定差距：reward = 跟踪项(贴近 OmniRetarget 参考) + 穿透惩罚 → CEM 选出的精英永远是"贴近参考但不穿透"的那撮 → 收敛到参考外侧一层薄壳。**调权重改变不了"参考在哪"**，只能在参考周围微调取舍。要动参考本身，就得 (d)。

### (d) 交替优化（block coordinate descent）怎么做？

ReActor 用可微 bilevel（上层梯度 + 下层 RL 单循环同时更新）。SPIDER 无梯度，故用交替优化这个更朴素、目标等价的近似：**把"参考"也变成可优化变量，与"控制序列"轮流优化。**

现在 SPIDER 只优化一个变量（参考被冻结）：

```
    U* = argmax_U  R(U ; θ_ref = 冻结的 OmniRetarget)
```

交替优化扩成两个变量轮流解，迭代 k=0,1,2,... 直到收敛：

```
    内层(已有)   U^(k)     = argmax_U  R(U ; θ^(k))                  ← 现有 CEM，固定 ref 参数
    外层(新增)   θ^(k+1)   = argmin_θ  L( rollout(U^(k)) , GT动捕 )   ← 更新 ref 参数
```

最小可行落地：

1. 选一组**小而可解释**的 ref 参数 θ（别把整条参考设成自由变量，维度爆炸）。借 ReActor 设计先选几个：
   - 逐 case 竖直偏移 p_z（ReActor Eq.10，修 floating/penetration，最便宜）；
   - hand target offset（手接触目标相对 OmniRetarget wrist 的偏移，3~6 维）。
2. 外层目标 L 用**不以 OmniRetarget 为锚**的量：rollout 后物体 vs GT 动捕误差、hand-box 贴合带 frac(SDF∈[0,2cm])、穿透惩罚。这才是真正"修参考"的方向盘（绕开恒定差距）。
3. 外层同样**零阶**求解（因内层 CEM 不可微）：θ 维度小 → 坐标/网格搜索或有限差分；或再套一层小 CEM（在 θ 空间撒几十点，每点跑一次内层 CEM 得 L，挑最好挪均值）。θ 刻意压到个位数维度，否则外层跑不动（每个外层迭代要跑一次完整内层 CEM，很贵）。
4. 收敛判据：L 不再下降，或 θ 变化 < 阈值。

与 ReActor 的关系：ReActor 可微 bilevel 一步同时更新两层；交替优化分块轮流更新，目标一致（co-adapt 参考与解），代价是慢 + 局部最优近似，但**完全兼容现有 CEM，不需把 SPIDER 改成可微/RL**。这是在无梯度优化器上实现"参考可学习"的标准做法。

---

## 1.6 三个方法的优化器谱系：OmniRetarget vs SPIDER vs ReActor

（代码核查自 holosoma `src/interaction_mesh_retargeter.py` + `src/utils.py`；OmniRetarget 论文 §III-A Eq.3a-3e。这一节把 §1.5 的"优化器类别"扩到三方对比，弄清"参考是谁生成的、能在哪一层修参考"。）

### (a) OmniRetarget 也是优化问题 —— 但与 SPIDER 的 CEM 是方法论对立面

OmniRetarget 是**纯运动学的约束优化**，**逐帧**求解（论文 §III-A Eq.3a-3e）：

```
q*_t = argmin_{q_t}  Σ_i ‖L(p^source_{t,i}) − L(p^target_{t,i}(q_t))‖²  +  ‖q_t − q_{t-1}‖²_Q
  s.t.  φ_j(q_t) ≥ 0            非穿透/避碰 (硬约束)
        q_min ≤ q_t ≤ q_max      关节限位
        v_min·dt ≤ q_t−q_{t-1} ≤ v_max·dt   速度限位
        p^F_t = p^F_{t-1}        支撑脚不打滑
```

- **决策变量**：机器人**每帧的完整构型 q_t**（浮动基座 quat+平移 + 所有关节角）；代码里是增量 dqa（`interaction_mesh_retargeter.py:664`）。
- **目标**：interaction-mesh 的 **Laplacian 形变能量**（保住"谁挨着谁"的空间/接触关系，Eq.1-2）+ 时间平滑。
- **约束**：穿透/限位/速度/脚不滑——全是**硬约束**。

**算法 = SQP（序列二次规划），子问题是凸 SOCP，用 Clarabel 内点法解（CVXPY）。**
- 论文 Table I 列其 "Optimization Method" = "Sequential SOCP"（对比 PHC="Gradient Descent"）。
- 代码：`cp.CLARABEL`（`:819,825,885`），信赖域是二阶锥 `cp.SOC(step_size, dqa)`（`:756`），外层 SQP 循环 `iterate()`（`:969-1006`）。
- **有梯度吗？是一阶 Jacobian-based**：约束线性化 + 目标二次近似，用**解析 Jacobian**（`mj_jac`，`:1409`），不是 autodiff、不是普通梯度下降（那是 PHC）。内点法内部用二阶 Newton/KKT。
- paper/code 差异：论文原版用 **Drake + autodiff**；本仓代码用 **CVXPY+Clarabel+解析 Jacobian**。同一 SQP/SOCP 数学，不同后端。

**interaction mesh**：对关键关节 + 物体/环境表面采样点做 Delaunay 四面体化（`utils.py:405-416`），匹配 **Laplacian 坐标**（每点 − 邻居加权平均）→ 保住跨形态后的局部接触/空间结构。Laplacian 权重 `=10`（`:126`）。

**纯运动学,确认**：全程只调 `mj_forward`（FK + 几何距离），**无 `mj_step`、无动力学、无接触力**；碰撞是 `mj_geomDistance` 带符号距离硬约束（`:729-737`），不是物理接触。输出存成 `qpos` 轨迹——**一个给下游 RL 当 target 的运动学参考**。

### (b) 三方优化器谱系总表

| 维度 | OmniRetarget | SPIDER (CEM) | ReActor |
|---|---|---|---|
| 是优化问题 | 是，逐帧约束 NLP | 是，随机最优控制 | 是，bilevel |
| 决策变量 | 机器人构型 q_t（逐帧，warm-start） | 控制/动作序列 U（batch 采样） | 上层 ref 参数 p + 下层策略 φ（联合） |
| 求解器类别 | **一阶 Jacobian** 确定性（SQP→SOCP 内点法） | **零阶** 随机（CEM 撒点选精英） | **一阶梯度** bilevel（简化梯度 + PPO） |
| 需要可微代价 | **需要**（Jacobian） | **不需要**（黑盒 reward） | **需要**（上层简化梯度） |
| 物理 vs 运动学 | **纯运动学**（仅 mj_forward） | **物理**（rollout 过仿真，真实接触力） | **物理**（Isaac，RL rollout） |
| 接触处理 | 几何带符号距离**硬约束**，无力 | 物理接触求解器**涌现**，真实力 | 物理接触 + RFC root 辅助力 |
| 约束满足 | **硬约束精确满足** | 软约束（进 reward），统计满足 | 软（reward）+ 凸集投影约束 p |
| 参考是否可变 | **它就是参考的生产者**（可微，可直接改目标项） | 参考**冻结**（来自 OmniRetarget） | **参考可学习**（与策略 co-adapt） |
| 输出 | 运动学参考（qpos） | 动力学可行动作 | 动力学可行 + 已修好的参考 |

**一句话**：OmniRetarget = 一阶(Jacobian)确定性 + **运动学**，产**参考**；SPIDER = 零阶(采样)随机 + **物理**，跟踪那个冻结参考产**动作**；ReActor = 一阶梯度 bilevel，把"产参考"和"跟踪"**合一**。

### (c) 对 ref-repair 路线的关键影响：可以直接在 OmniRetarget 层修参考 ★★★

§1 说要"修参考"对抗恒定差距。本节查出一个更便宜的入口：**参考本来就是 OmniRetarget 用可微凸优化生成的**，所以修参考有**两条路**：

1. **SPIDER 外层交替优化**（§1.5(d)）：在无梯度 CEM 外面套零阶搜索 ref 偏移 → 慢、贵（每外层迭代跑一次完整 CEM）。
2. **直接在 OmniRetarget 这一层改**（更自然）：OmniRetarget 本身是**可微 SQP/SOCP**，目标里**已经有 Laplacian + 接触保持项**（Phase 4c `w_contact=20`，代码里默认关，`:792-813`）。要让参考的手更贴箱、少穿透，可以：
   - 打开/调高它的**接触保持项** `w_contact`（wrist→object 距离），让生成的参考本身就把手放得更贴；
   - 调 **Laplacian 权重** 或对 hand-object 边加权，强化接触局部结构；
   - 这些都是在**一阶可微优化**里直接加项，比在零阶 CEM 外层做搜索**便宜得多、可控得多**。
   - 注意：OmniRetarget 纯运动学，改完仍是"运动学参考"，物理可行性还得 SPIDER 的 CEM 兜底——但**起点更好** → SPIDER 收敛到的薄壳更贴、绝对接触更高。

> **结论更新**：对抗"−2.9pp 恒定差距"的最优路线,可能是 **(2) 先在 OmniRetarget 层把参考的接触改好（开 `w_contact` / 调 Laplacian），再让 SPIDER 跟踪**——而不是只在 SPIDER 外层套交替优化。两者可叠加：OmniRetarget 出更好的参考 → SPIDER 外层再微调残差。这把 ReActor 的"co-adapt 参考"思想落在**已有的可微参考生成器**上，工程代价最小。

---

## 2. 第二强启发：RFC + 外力惩罚 = 你"改另一侧力"增广的方法论模板 ★★★

ReActor 对 robot **root** 施加可学习的辅助 wrench（RFC，Eq.18），让本来动力学不可行的动作（如无手倒立）可跟踪，并**惩罚 wrench 用量**以保物理可信。

这和你的物理增广想法（`project_physics_augmentation`：对箱子**另一侧**施外力）是**同一类机制**——外部 wrench 让欠驱动/欠约束系统可行。ReActor 给你两样可直接借用的东西：

1. **外力惩罚的 trade-off 刻画方法（Fig.8）**：横轴 = force penalty weight，三条曲线 = 最大施加力/力矩、上层 loss、失败数。**这正是你做"改另一侧力增广"时该画的图**：扫描另一侧力的幅度，刻画"可行性 vs 物理真实性"前沿，从而**有原则地选力值/选保留哪些变体**，而不是拍脑袋。直接对应你 plan 里 §5 的"力扫 4~5 个值 + 多指标评估"。
2. **连续 deadband（Eq.19）**：让策略在不需要时输出**零外力**。对增广的启示：可设计"仅在必要相位（抬升/转身）施另一侧力"的 deadband，使增广出的轨迹在平稳段不依赖外力 → 更接近真实协作。

**但要注意两点差异（诚实）**：
- ReActor 的 RFC 作用在 **robot root**（是"作弊力"，故要罚到最小）；你的另一侧力作用在 **object**、且代表**真实的人类 partner 分担**（不是作弊）。所以你**不该**像 ReActor 那样把它一味罚到 0——而是把它当**可控的物理条件**来扫。ReActor 的"惩罚"思路适合"让增广轨迹不过度依赖外力"，但不适合"消灭外力"。
- ReActor 是 RL 学 wrench；你的 plan 是设定 `perturb_force` 后**重跑 CEM**。两者都满足"加力→重优化"的核心原则（与 `project_physics_augmentation` 原则 1 一致：不能固定 U 只改力）。

---

## 3. 第三启发：成功率指标 —— 印证你 Q1 的结论（loco-manip 不该用 object servo tracking）★★

ReActor 的 success rate = **下游 RL 跑完不触发 root 终止**（root pos>1m 或 ori>45°，附录 A），**完全不看 object**（它本就无物体）。这与 SPIDER 论文 Table1-3 的 object tracking（0.1m/0.5rad）是两套东西。

对你已 parked 的 Q1 是有力佐证：**loco-manip 社区（OmniRetarget/Yang2025a/ReActor）的成功率判据是 root/pelvis 稳定 + 下游 RL，不是物体伺服跟踪**——恰好等价于你已有的 pelvis 摔倒/稳定指标（`unified_replay_eval.py:431-449`）。也再次说明：你工作流里被 6 actuator 伺服的 object tracking 不能当成功率（degenerate）。

> 可借用：把 ReActor 的 root 终止判据（1m / 45°）作为你协作场景成功率的**稳定性分量**，再叠加**物体相关分量**（箱子是否被搬到目标、hand-box SDF 贴合、无穿透）——构成 loco-manip **协作**专用的多维成功率。注意 ReActor 没有物体，这一块是你要补的。

---

## 4. 第四启发：`p_z` 逐 motion 竖直偏移 —— 修 floating/penetration 的轻量手段 ★

ReActor 学一个**逐 motion 的竖直偏移 `p_z`**（Eq.10）专门修 AMASS 里 noisy contact 造成的 floating/penetration。对你的 CORE4D：
- 类似地可对 hand-box 或 foot-ground 引入**可学习/可搜索的偏移**修穿透——比改 reward 更直接。
- 这其实是 §1 "交替优化修 reference" 里最便宜的一个参数维度，建议作为 ref-repair 的第一个旋钮试。

---

## 5. 论文自己点名的研究空白 = 你的课题正好落点 ★★★

ReActor 结论（p9）原话：bilevel 框架"holds significant promise for more complex scenarios, such as obstacle avoidance, **manipulation**, or even automated robot design"。且：
- ReActor **没有物体、没有 manipulation、没有协作**——纯 loco 重定向；其 penetration 指标只有 ground + self，**没有 hand-object**。
- 它明确把**时变参数化**（params 现在 constant over time）列为 future work——而协作里 handover/转身的接触是**强时变**的。

> **这意味着**：把 ReActor 的"bilevel co-adapt 参考 + RFC + 外力惩罚刻画"扩展到**带物体的人机协作 loco-manipulation**，是论文自己指出、但未做的方向——正是你的课题。可作为你工作的**理论定位与 related-work 对标**：你 = ReActor 的 bilevel/physics-aware 思想 × manipulation/collaboration × 时变接触。

---

## 6. 哪些不能直接迁移（诚实边界）

| ReActor | 你的工作流 | 能否直接用 |
|---|---|---|
| 下层 RL (PPO) + 可微简化梯度 bilevel | SPIDER 下层 sampling-MPC (CEM)，无梯度 | ❌ 不能直接移植 bilevel-RL；只能用**交替优化**近似（§1） |
| RFC 作用于 robot root（作弊力，罚到 0） | 另一侧力作用于 object（真实 partner，应扫不应灭） | 概念借用 ✓，惩罚策略需反向理解（§2） |
| 无物体；penetration 只有 ground+self | 核心是 hand-object 接触 + 箱子动力学 | 指标体系要自己补物体维度（§3） |
| 时变参数 = future work | 协作接触强时变（handover） | 这是你要补的增量，不是现成方案 |
| success = root 终止 1m/45° | 需 root 稳定 + 物体搬运 + 接触 | 借 root 分量，补物体分量（§3） |

---

## 7. 可落地的下一步（结合现有 plan）

1. **ref-repair 作为对抗"恒定差距"的主线**（§1）：在 SPIDER 里做"优化少量 reference 偏移参数（先试 ReActor 式逐 case 竖直 `p_z` + hand target offset）→ 重跑 CEM"的交替循环。这比调 reward 更可能真正抬高接触/接近度，且有 ReActor 的 bilevel 理论背书。
2. **物理增广用 Fig.8 方法学刻画**（§2）：在 `project_physics_augmentation` 的 C1（改另一侧力）落地时，照 ReActor Fig.8 画"另一侧力幅度 vs 物体跟踪误差/穿透 vs 失败数"的前沿图，用它选力值、定可行变体阈值。
3. **协作成功率指标**（§3）：root 终止（1m/45°）+ 物体搬运到位 + hand-box SDF 贴合 + 无穿透，多维报告（mean+std+worst）。
4. **related work / 定位**：把课题写成"ReActor 的 physics-aware co-adaptation 思想在带物体的人机协作 loco-manipulation 上的扩展"，引 ReActor 结论自陈的 manipulation/time-varying 空白。

---

## 8. 关键引用定位（便于回看 PDF）
- Retargeting≠Imitation：p1 右栏、p2 左栏。
- Bilevel 公式 Eq.1：p2 右栏底。简化梯度 Eq.4-7：p3。
- 参数化 Fig.3 + Eq.8-14：p4。`p_z` 竖直偏移：Eq.10。
- RFC action Eq.18、deadband Eq.19、phase ψ_t Eq.20：p5。reward Table.1：p5。
- 指标定义 Table.3、结果 Table.4：p6、p8。
- bilevel 消融 Fig.6/7：p7。外力惩罚 trade-off Fig.8：p8。
- 用例（四足/交互动画/实机）+ 结论自陈空白：p8-9。success 判据：附录 A，p10。
