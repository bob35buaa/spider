# CORE4D 人-物重定向：核心方法技术总结（面向论文）

> 定位：本文档抽取我们方法**真正的算法贡献**（以最终 **E167A + PRG + G1** 线为准，参数经 override 链核实），并给出论文用的**可视化图**与 **LaTeX 算法伪代码**。
> 底座：我们**沿用** SPIDER 的采样式 MPC 主干（CEM + 退火协方差 + top-10% 精英）与 6D 伺服物体驱动；贡献是叠加在其目标层与约束层上的三点。
> 配套：`figures/contact_reward_landscape.pdf`、`figures/penetration_penalty_gate.pdf`（`plot_paper_figures.py` 可复现）、`algorithm.tex`（Alg.1/2）。

贡献按重要性由高到低：

| # | 贡献 | 层 | 一句话 |
|---|---|---|---|
| **1** | **Multi-Scale Contact Reward（多尺度接触奖励）** | 目标层 | 宽尺度锚点吸引（欧氏）+ 尖尺度 SDF 表面贴合，两项**同时叠加**、按距离平滑接管，把「够到」与「真实贴合」分离建模 |
| **2** | **Part-wise Penetration Constraints（分部位穿透约束）** | 目标层 + 约束层 | 每个部位组一条有符号净空约束 `φ_P≥θ_P`（阈值本身编码部位差异：leg 严禁碰、hand 允许贴），用**两层执行**：软层可微惩罚 + 硬层候选门（AND 组合、least-violation 回退） |
| **3** | **面向 CORE4D 的数据工程** | 参考生成 + 后处理 | OmniRetarget 重定向→SPIDER 转换、hand-snap IK 抓取先验、scene_act Euler 契约、物体增强、handoff 平滑与成功过滤（偏工程） |

---

## 1. 核心贡献 1：Multi-Scale Contact Reward（多尺度接触奖励）

### 1.1 问题
原始 SPIDER 的接触项是**手部 site 到参考 site 的欧氏距离** `−Σ mask·‖p_site − p_ref‖`。它无法区分两种几何上完全不同的状态：**贴合物体表面** 与 **穿进物体内部**——两者的 site 距离可以一样小。对 CORE4D 的抓取/搬运，这会导致「穿模式接触」被误当成好接触（reward hacking）。

### 1.2 方法：两个尺度同时叠加的接触奖励
我们用**两项互补、始终同时叠加**的接触奖励，覆盖从「远处够物」到「贴面锁定」的全过程（见 **图 1**）。注意这**不是**"先粗后细"的时序调度：两项每帧、每次迭代都在 reward 里，粗/细谁主导**由手到表面的距离自动决定**（远处细项为 0，只有粗项；贴近时细项尖峰点亮）。

**(a) 远/粗——HDMI 式锚点吸引**（`contact_hdmi`）
把接触目标建成**刚附着在物体上的锚点** `p^{⋆,e}_t = o_pos + R(o_quat)·offset`；末端接触点 `p^e_t = eef_pos + R(eef_quat)·offset`：

$$r^{\text{coarse}}_{t,e} \;=\; g\cdot\exp\!\Big(-\tfrac{\lVert p^e_t - p^{\star,e}_t\rVert}{\sigma_c}\Big)\quad(\text{接触相 }m^e_t{=}1),\qquad \sigma_c=0.30\text{ m},\ g=5.0$$

宽尺度（σ_c=30cm）提供**长程平滑梯度**，把手从远处稳定地吸向接触锚点；非接触相取中性值 1.0（HDMI 约定 `mask=0 → 1`）。另加 palm-normal 朝向项（`additive`, w=0.3），要求掌面朝向目标。

**(b) 近/细——SDF 薄带表面贴合**（`surface_band`）
用物体局部 **grid-SDF** `φ_e(x)` 度量手部几何到物体表面的**有符号距离**，只在一条**薄带** `B=[−1mm,+3mm]` 内给分：

$$r^{\text{fine}}_{t,e} \;=\; s\cdot\exp\!\Big(-\tfrac{|\varphi_e(x_t)|}{\sigma_f}\Big)\cdot \mathbf 1[\varphi_e\in B]\cdot m^e_t,\qquad \sigma_f=1.5\text{ mm},\ s=1.5$$

尖锐尺度（σ_f=1.5mm）**只奖励真实贴面**：略微穿透（>1mm）或悬空（>3mm）迅速掉出带外得 0。释放相（末段 15%）线性衰减，修「放手放不掉」。

**为何多尺度叠加有效**：粗项是一个**处处有梯度**的吸引势（解决"从哪来"），细项是一个**只在表面尖峰**的贴合势（解决"停在哪"）。单用细项，梯度只在 ±3mm 内存在，采样极难命中；单用粗项，最优点是锚点而非真实表面，易穿模。二者**同时相加**得到「远处被拉近、近处被锁在表面、穿透被推回」的复合势——粗到细的过渡是**空间(距离)驱动**的平滑接管，而非优化过程中的调度切换。

> **图 1**（`figures/contact_reward_landscape.png`）：左＝粗项在 0–40cm 的长程吸引；右＝近场放大，细项在 [−1,+3]mm 的尖峰贴合 + 穿透区（d<0）交由惩罚/门处理。

![contact reward](figures/contact_reward_landscape.png)

### 1.3 与文献接触项的差异
- vs SPIDER 的 site 距离：我们用**物体几何 SDF** 而非点-点距离，几何语义正确（内/外有符号）。
- vs 纯 HDMI（It Takes Two）：HDMI 只有锚点吸引（粗），我们补了 **SDF 表面细项**，把"接触真实性"显式建模。
- 备注：代码另有 `distance_continuation` score-mode（单项内即 `w_f·e^{-|d|/0.05}+w_n·e^{-|d|/0.015}` 的多尺度混合），最终线用的是 `symmetric_abs`（细项）+ 独立的 HDMI 粗项，**系统级**实现多尺度。
- 命名说明：粗项用的是**手接触点到物体锚点的欧氏距离**（非 SDF），只有细项用 SDF，故不叫 "SDF Contact Reward"；两项同时生效、按距离接管，故不叫 "Coarse-to-Fine"（避免"先粗后细"的时序误解）。定名 **Multi-Scale Contact Reward**。

---

## 2. 核心贡献 2：Part-wise Penetration Constraints（分部位穿透约束）

### 2.1 问题
接触奖励鼓励"靠近"，必须有对偶机制阻止"穿进去"，且不同身体部位对穿透的容忍度不同：**手**要贴面（允许接触级的极浅接触），**下肢/身体**则应与箱体保持净空（腿穿箱、身体压穿是 CORE4D 里最常见的物理病态）。

### 2.2 方法：一条 per-part 约束，两层执行（见 **图 2**）

核心是**为每个部位组 P 定义一条有符号净空约束**，阈值 `θ_P` 本身编码该部位的接触角色：

$$\varphi_P(x_t)\ \ge\ \theta_P,\qquad P\in\{\text{hand},\ \text{body},\ \text{leg}\}$$

最终线核实的阈值（**分部位、非对称**，这是"part-wise"的实质）：**leg** `≥+5mm`（**PRG-G**，腿严格离箱、绝不接触）、**body(safety)** `≥−5mm`、**hand** `≥−10mm` 且 **hard-floor −20mm**（手允许接触级浅穿）。同一条约束用**两层强度执行**——不是两个独立特性：

**① 软层——可微惩罚（进入 reward，样本级）**：约束的松弛,给"推离"梯度：

$$c^{\text{pen}}_{t,P} \;=\; w_P\cdot\max\!\big(0,\ \delta_P - \varphi_P(x_t)\big),\qquad w_P=2.0,\ \delta_P=2\text{ cm}$$

部位组落地为 {robot(身体)-物体, leg(下肢，PRG-R)-物体, hand-floor(手-地面)}；另含 **E167A body-z / ground-z** 穿地惩罚（样本级 z 分量）。

**② 硬层——候选门（精英选择前剔除，候选级，不可微）**：同一约束的可行性投影：

$$\text{valid}_P^{\,j} \;=\; \big[\min_t \varphi_P(x^j_t)\ge \text{floor}_P\big]\ \wedge\ \Big[\tfrac1H\textstyle\sum_t \mathbf 1[\varphi_P<\theta_P]\le \tau_P\Big]$$

**E153**：硬地板 `floor_P` 与 per-frame 阈值 `θ_P` 解耦——允许 ≤10% 帧落在 `[floor,θ)`，任何帧 `<floor` 直接否决。候选合法 ⟺ **所有部位 AND 通过**；合法样本 `< 2%·N` 时 **least-violation 回退**（按违规深度+比例排序），保证永远选得出精英。

> **两层是同一条约束的"软梯度 + 硬可行性",不是 1+1**：软层防止"轻微靠近"被过度惩罚、并给采样梯度；硬层从根上杜绝"压入式穿透被 tracking 高分掩盖"（软层单独会被 tracking 淹没）。

> **图 2**（`figures/penetration_penalty_gate.png`）：左＝软 hinge（w=2, δ=2cm）；右＝分部位硬门阈值（leg +5 / body −5 / hand −10，hand hard-floor −20mm）。

![penetration](figures/penetration_penalty_gate.png)

---

## 3. 核心贡献 3：面向 CORE4D 的数据工程（偏工程）

单主体人-物协作数据（CORE4D）落到 G1 需要一套清洗/过滤/对齐工程，独立于优化器：

**预处理（参考生成层）**
- **OmniRetarget 运动学重定向 → SPIDER 转换**：holosoma `robot_retarget` 产出 G1 `qpos(T,43)`，`spider/process_datasets/core4d.py` 经 MuJoCo FK 补 `qvel/ctrl/contact/contact_pos` 转为 `trajectory_kinematic.npz`。
- **物体交互增强**：接近段扰动物体位姿（平移/±yaw）+ 指数衰减重锚（`translation_tau=50 / rotation_tau=25`）保持操作终点不变；`omnirt_v2`（Phase-4 约束松弛）提升 IK 可行率。
- **hand-snap IK**：intent 窗口内把手掌 site 阻尼最小二乘投影到物体表面（仅改手臂 7DoF），注入几何抓取先验作为 CEM warmstart。
- **scene_act Euler 契约**（`scene_act_reference.py`）：对预编译 scene 做 fail-closed 的 Euler 约定 + sha256 校验，杜绝物体位姿在不同 XML 版本间静默漂移。

**后处理**
- **smooth handoff**（`postprocess/smooth_handoff*.py`）：CPU-only 沿时间轴平滑 handoff NPZ（跳过 mask/contact/time），不改输入。
- **成功过滤 / 快照**：按物体位姿误差 + 接触保真 + 穿透门通过率筛选成功案例；每次训练前 scene XML + sha256 快照（`scene_snapshot/`）保证可复现。

> 这一块是"让方法在 CORE4D 上跑得动、跑得可复现"的工程支撑，学术贡献度低于 1、2，但对结果的可信度不可或缺，建议在论文中作为"Implementation / Data pipeline"小节陈述。

---

## 4. 算法伪代码

完整 LaTeX 见 `algorithm.tex`（`algorithm` + `algpseudocode`，风格对齐 SPIDER Alg.1 与 DynaRetarget Alg.1/2）。要点：

- **Algorithm 1**：外层 receding-horizon 采样 MPC（沿用 SPIDER：退火协方差、早停），内层每次调用 `CostAndGate` 评估、`GatedElite` 精英重拟合。
- **Algorithm 2**：`CostAndGate`——局部系跟踪 + 物体位姿 + **Multi-Scale 接触奖励（粗=锚点欧氏，细=SDF 薄带）** + **分部位软穿透惩罚** + E167A z 项，并输出**分部位门统计**；`GatedElite`——合法候选内 softmax 精英，`<2%` 合法时 least-violation 回退。

> 说明：最终线使用**默认 receding-horizon MPC**（`use_sbto=False`），故 Alg.1 是 SPIDER 式外层；DynaRetarget 的 SBTO（增量地平线）仅在探索线出现，不写入最终算法。

---

## 5. 论文图使用建议（LaTeX）

图已同时导出 `.pdf`（矢量，供 `\includegraphics`）：

```latex
\begin{figure}[t]\centering
  \includegraphics[width=\linewidth]{figures/contact_reward_landscape.pdf}
  \caption{Multi-scale contact reward. A coarse anchor-attraction term
  (\emph{Euclidean} distance to an object-attached anchor, $\sigma_c{=}0.3$\,m)
  and a fine SDF surface-band term ($\sigma_f{=}1.5$\,mm, support $[-1,+3]$\,mm)
  are applied \emph{simultaneously}; (a) the coarse term gives a long-range
  gradient, (b) the fine term rewards true surface adherence, with a smooth
  distance-driven hand-off. Penetration ($d{<}0$) is handled by the penalty
  and hard gate.}
  \label{fig:contact}
\end{figure}

\begin{figure}[t]\centering
  \includegraphics[width=\linewidth]{figures/penetration_penalty_gate.pdf}
  \caption{Part-wise penetration control. (a) soft hinge penalty
  $-w\max(0,\delta-\mathrm{SDF})$; (b) per-part hard gate thresholds
  (leg $+5$, body $-5$, hand $-10$\,mm; hand hard floor $-20$\,mm),
  AND-combined with least-violation fallback.}
  \label{fig:penetration}
\end{figure}
```

---

## 6. 最终线参数（经 override 链核实，供表格/复现）

| 组件 | 参数 | 值 | 来源 override |
|---|---|---|---|
| HDMI 粗接触 | `gain / σ_c / ori(additive,w)` | 5.0 / 0.30 m / 0.3 | E084C / e041c |
| SDF 细接触 | `scale / σ_f / band / mode / decay` | 1.5 / 1.5 mm / [−1,+3] mm / symmetric_abs / 0.15 | E163 |
| 软穿透惩罚 | `w / δ`（robot/leg/hand-floor） | 2.0 / 2 cm | E084C / E199-PRG |
| E167A z-穿透 | body-z / ground-z | enabled | E167A |
| 硬门 leg (PRG-G) | `min_sdf` | +5 mm | E199-PRG |
| 硬门 body(safety) | `min_sdf / min_valid_frac` | −5 mm / 2% | E088A |
| 硬门 hand | `min_sdf / hard_floor / max_vio` | −10 / −20 mm / 10% | E163 |
| 优化器 | receding-horizon MPC, top-10% 门控精英 | `use_sbto=False` | 默认 |
| 物体驱动 | 6D 伺服 (contact_guidance) | 同 vanilla | — |

*配套详见 `spider_algorithm_full_comparison.md`（全面异同）；源码：`spider/rewards/surface_distance.py`、`spider/simulators/mjwp.py`（get_reward / geom_object_sdf_min）、`spider/optimizers/sampling.py`（GatedElite）、`spider/geometry/grid_sdf.py`。*
