# E198 结果：G1×A2 完整 2×2 因子 —— 两干预非可加，主要相互纠正对方的回退

_Core4D · Phase 61 · 2026-08-13 · 计划 [plan226](../plan/226_E198_g1xa2_factorial_and_E192_a2_expansion_plan.md) · evaluator `core4d-e154-physics-contact-v1`_

## TL;DR

- **执行闭合**：103/103（4 物体）+ 56/56（box001，[plan227](../plan/227_E198_box001_g1a2_a2_supplement_plan.md)）Full CEM 完成（本地 8 卡 priority 队列，与他人 job 叠加共跑，0 失败）；**5 物体** 4 臂（A0/G1/A2/G1+A2）共 348/348 用同一公共 evaluator 打分，0 error。
- **box001（第 5 物体，§5.6）是「无单臂回退→可加」的对照**：obj_ori 上 G1、A2 都单调改善（无 box023/box004 那种单臂崩溃），故 INT≈0（+0.180 CI 含 0），交互干净退化为可加；G1 的 lower_body 改善最强（A0→G1 +25pp，p=0.016）。**这条负结果强化了「相互救援只在某单臂回退时出现」的机制论断。** box001 用 E196 修正参考、运行时 fail-closed parity 通过。
- **C3 baseline parity PASS**：复用的 box021/023 A0+G1（88 行）重打分与冻结 E194 表逐 case z 差 `0.000000 cm`，交互项非 confounded。
- **z-tracking 由 G1 独占，A2 不贡献**：`track_obj_z_abs_err_cm_mean` 的改善几乎全部来自 G1（如 box024 5.574→3.229），A2 单加几乎不动 z；G1+A2 ≈ G1。交互项多数 CI 含 0（近似可加），仅 box021 `INT=−0.230 [−0.442,−0.039]` 轻微协同。
- **核心发现（gate 迁移）——两干预相互纠正对方的回退**：
  - **A2 单用会损伤姿态/朝向门**（A0→A2 全 59 例）：`root_ori −13.6pp`（exact p=0.021）、`hand_ori −16.9pp`（p=0.006）、`root_pos −6.8pp`、`hand_pos −5.1pp`。在四物体全集上复现了 E192 的 A2 姿态代偿/collapse 机制。
  - **叠加 G1 能救回 A2 的损伤**（A2→G1+A2）：`root_ori +15.2pp`（p=0.004，9 例 F→P）、`lower_body +13.6pp`（p=0.039，10 例 F→P）、`hand_ori +11.9pp`、`hand_pos +10.2pp`。
  - **在 G1 上叠加 A2 基本中性**（G1→G1+A2）：多数门 |Δ|<4pp；`object_ori +8.5pp`（5 例 F→P，p=0.06）——即 A2 叠在 G1 上不再引发它单用时的姿态塌陷。
- **A2 的降穿透收益在 G1 之上不叠加**：box024 A2 单用把 3mm 手物穿透 `0.378→0.308`，但 G1+A2=`0.320`≈G1（`INT=+0.069 [−0.034,0.170]`）。
- **object_ori 的物体特异协同**：box023 `INT=−3.07 [−5.70,−0.78]`、box004 `INT=−3.51 [−8.70,−0.29]`——G1 单用抬高 box023 朝向误差（5.11→7.96°），但 G1+A2 降回 4.94°（A2 纠正了 G1 的朝向回退）。
- **组合价值是「方差收缩」而非「均值提升」**（§5.3）：A2 单用崩 box004 朝向（11.5→15.2°）、G1 单用崩 box023 朝向（5.1→8.0°），G1+A2 把两次崩溃都削平但均值不超 G1——买到的是跨物体最坏情况更紧，不是更低均值。且 obj_ori 上「谁救谁」符号随物体翻转（box023 是 A2 救 G1、box004 是 G1 救 A2，§5.2）。
- **判决 `FACTORIAL_CHARACTERIZED`**：两干预**非独立**、主要通过相互抵消各自的副作用而非叠加各自的收益来交互；G1+A2 在大多数指标上接近 G1-only。**不升级 A2 或 G1+A2**；A2 的 `INCONCLUSIVE_GATE_COLLAPSE` governance 不被推翻。
- **交付**：四臂×物体对比工作簿 `E198_G1xA2_factorial.xlsx`（每物体 12 指标 × G1+A2/G1/A2/PRG + INT + 95%CI + gate 迁移 + 逐例）。

## 1. 设计与执行

完成 box004/box021/box023/box024 四物体各自的 2×2 因子（none=A0/PRG、G1=object gravcomp、A2=hand-gate 三字段、G1+A2）。G1、A2 定义与冻结不变量见 plan226。本轮只新增两组 GPU 运行：

| 实验 | 臂 | Cases | 结果目录 |
|---|---|---|---|
| E198 | G1+A2 | box024(9)+box004(6)+box021(28)+box023(16)=59 | `results/E198/s6_downstream/cem/full_g1a2/` |
| E192-ext | A2 | box021(28)+box023(16)=44 | `results/E192/s6_downstream/cem/full_a2_expansion/`（见 [log283](283_E192_a2_expansion_box021_box023_results.md)）|

历史 A0/G1/A2（box024/004）与 box021/023 的 A0/G1 复用既有 rollout，由本轮同一 `core4d-e154-physics-contact-v1` evaluator 重打分（单变量 C1、单 evaluator C3 均满足）。

执行：本地 8× GPU（0-7）统一 priority 队列，`PER_GPU_MEM_MIB=5000`、每卡 1 run，严格 P0(box024 G1+A2)→P1(A2)→P2(G1+A2)→P3(box004 G1+A2)；与 GPU0/2/4 上他人 job 叠加共跑，未 kill/抢占任何进程。canary 4/4 通过。

## 2. 交互项（`INT = M(G1+A2) − M(A2) − M(G1) + M(A0)`，paired bootstrap 95% CI）

主指标（cm/占比，越低越好；in-mask contact 越高越好）。完整六指标见 report。

### obj z |err| (cm)
| Object | n | A0 | G1 | A2 | G1+A2 | INT [95% CI] |
|---|---:|---:|---:|---:|---:|---:|
| box024 | 9 | 5.574 | 3.229 | 5.281 | 3.227 | 0.291 [−0.091, 0.635] |
| box021 | 28 | 6.237 | 4.938 | 6.429 | 4.901 | **−0.230 [−0.442, −0.039]** |
| box023 | 16 | 5.817 | 5.276 | 5.848 | 5.137 | −0.170 [−0.453, 0.164] |
| box004 | 6 | 4.892 | 4.483 | 5.094 | 4.540 | −0.145 [−0.451, 0.172] |

### hand 3mm penetration（越低越好）
| Object | n | A0 | G1 | A2 | G1+A2 | INT [95% CI] |
|---|---:|---:|---:|---:|---:|---:|
| box024 | 9 | 0.378 | 0.320 | 0.308 | 0.320 | 0.069 [−0.034, 0.170] |
| box021 | 28 | 0.176 | 0.192 | 0.195 | 0.212 | 0.001 [−0.037, 0.037] |
| box023 | 16 | 0.148 | 0.164 | 0.144 | 0.146 | −0.013 [−0.088, 0.052] |
| box004 | 6 | 0.147 | 0.098 | 0.107 | 0.114 | 0.055 [−0.001, 0.131] |

### obj ori err (deg)（越低越好）
| Object | n | A0 | G1 | A2 | G1+A2 | INT [95% CI] |
|---|---:|---:|---:|---:|---:|---:|
| box024 | 9 | 6.251 | 5.951 | 6.247 | 5.159 | −0.789 [−2.970, 1.406] |
| box021 | 28 | 6.091 | 5.873 | 6.261 | 5.948 | −0.095 [−0.600, 0.350] |
| box023 | 16 | 5.108 | 7.963 | 5.160 | 4.943 | **−3.072 [−5.698, −0.781]** |
| box004 | 6 | 11.534 | 11.325 | 15.179 | 11.465 | **−3.505 [−8.697, −0.286]** |

leg penetration / 3D pos / in-mask contact 的完整表见 `E198_G1xA2_factorial_report.md`；leg penetration 全部 CI 含 0（box024 `INT=−0.100 [−0.228,0.024]` 呈救援趋势但不显著）。

## 3. 12-gate 迁移（全 59 例，exact McNemar）

| Transition | 关键门 | Δpp | P→F / F→P | exact p |
|---|---|---:|---:|---:|
| A0→A2 | root_ori | −13.6 | 9 / 1 | **0.021** |
| A0→A2 | hand_ori | −16.9 | 11 / 1 | **0.006** |
| A0→A2 | root_pos | −6.8 | 5 / 1 | 0.219 |
| A0→G1 | lower_body | +13.6 | 4 / 12 | 0.077 |
| A0→G1 | hand_penetration | −10.2 | 9 / 3 | 0.146 |
| G1→G1+A2 | object_ori | +8.5 | 0 / 5 | 0.063 |
| A2→G1+A2 | root_ori | +15.2 | 0 / 9 | **0.004** |
| A2→G1+A2 | lower_body | +13.6 | 2 / 10 | **0.039** |
| A2→G1+A2 | hand_ori | +11.9 | 2 / 9 | 0.065 |

读法：A2 单用（A0→A2）显著砸 orientation/pose 门；把 G1 叠加到 A2 上（A2→G1+A2）把这些门大幅救回（大量 F→P）。这说明 G1+A2 的净行为主要由 G1 决定，A2 单用的姿态破坏被 G1 的下肢/支撑改善抵消。

## 4. Claims 判定

| Claim | 判定 | 证据 |
|---|---|---|
| C0 scope/provenance | PASS | 103 新 run + 4 物体 4 单元 authority 全可追溯；SHA parity 全过 |
| C1 单变量 intervention | PASS | 59 个 G1+A2 单变量 gravcomp 审计通过；A2 仅改 3 gate 字段；`A2_GATE==E192` |
| C2 execution/numeric closure | PASS | 103/103 Full；236/236 scored；error/non-finite/diverged=0 |
| C3 baseline parity | PASS | 复用 88 行重打分 vs 冻结表 z 差 `0.000000 cm` |
| C4 交互项估计 | PASS | 四物体 2×2 完整；六指标 INT+CI+四主效应逐物体报告 |
| C5 承重接触保留 | PASS（除 box024 边界） | G1+A2 in-mask contact vs A0：box021/004 ≥−0.05；box024 `0.308→0.373`（较 A0 +0.065）；box023 `0.473→0.458`（−0.015）|
| C6 物理安全无灾难 | PASS | 无新增 fall/non-finite/diverged |
| C7 gate migration 透明 | PASS | 4 transition × 12 门全报 P→F/F→P + McNemar |
| C8 device confound | PASS | 队列按空闲卡 round-robin，落卡 GPU id 逐 case 记录 |
| C9 证据闭合 | PASS | 数值/gate/interaction 全闭合；4-cell MP4 + viser 交互复核见 §6 |

## 5. 深入分析：交互不是「叠加收益」，是「相互抵消副作用」

汇总表见 `E198_G1xA2_factorial.xlsx`（sheet「四臂×物体对比」，G1+A2 / G1 / A2 / PRG 逐物体 12 指标 + INT + 95%CI；绿=最优臂，黄=CI 不含 0）。以下四点是从 xlsx 的 INT 符号结构里读出的、超出逐指标均值的结论。

### 5.1 按指标簇给交互定性——三种模式，机制各异

| 指标簇 | 交互模式 | 判据（INT） | 机制解释 |
|---|---|---|---|
| **物体 Z / 3D 位置** | 可加-零（G1 独占） | A2≈A0、G1+A2≈G1，4 物体 INT 近 0（仅 box021 −0.230 轻微协同） | A2 只改 hand-gate SDF 阈值，**不触及 object position servo**；z 误差的唯一杠杆是 G1 的 gravcomp，A2 无从贡献 |
| **手-物 3mm 穿透** | 次可加（竞争） | box024 A2 单用 0.378→0.308，但 G1+A2=0.320≈G1，**INT=+0.069（↓指标上 INT>0 即拮抗）**；box004 同向 +0.055 | G1 抬升物体、A2 收紧手-物间隙，二者争夺**同一手-物法向自由度**，收益不叠加 |
| **姿态/朝向门（root/hand/lower_body）** | **相互救援** | A0→A2 破坏（root_ori −13.6pp、hand_ori −16.9pp，p<0.05），A2→G1+A2 救回（root_ori +15.2pp、lower_body +13.6pp，p<0.05，大量 F→P） | A2 为满足更松的手-gate 引入姿态代偿→塌下肢；G1 的物体托举减轻手臂负荷→**下肢/根姿态被动恢复**，净效果 ≈ G1 的姿态 |
| **物体朝向 obj_ori** | **双向、物体特异救援** | box023 INT=**−3.07** [−5.70,−0.78]；box004 INT=**−3.51** [−8.70,−0.29] | 见 5.2——两个显著协同来自**相反方向**的救援 |

### 5.2 最深的一点：obj_ori 上「谁救谁」的符号随物体翻转

两个 object_ori 的显著负 INT（协同）看似同类，机制却相反：

- **box023：A2 救 G1。** G1 单用把朝向误差从 `5.11°` **抬高到 7.96°**（gravcomp 托举改变了接触力矩分布，复现 E194 的 G1 朝向回退），而 A2 收紧手-gate 后 G1+A2 拉回 `4.94°`——A2 在这里是 G1 副作用的解药。
- **box004：G1 救 A2。** A2 单用把朝向误差从 `11.53°` **炸到 15.18°**（A2 姿态代偿波及物体朝向），而 G1+A2=`11.47°≈G1`——这次是 G1 抵消 A2 的破坏。

即 **INT<0（超可加改善）在两物体上都成立，但一个是 A2→救 G1、一个是 G1→救 A2**。这解释了为什么 obj_ori 的协同强烈却**无法泛化成一个方向的机制断言**：它取决于在该物体上哪个单干预先出了 orientation 回退。

### 5.3 由此得到 G1+A2 的真实价值：方差收缩，而非均值提升

把 obj_ori 四物体并排看最清楚（单位 °）：

| 臂 | box024 | box021 | box023 | box004 | **最坏值** |
|---|---:|---:|---:|---:|---:|
| A2 单用 | 6.25 | 6.26 | 5.16 | **15.18** | 15.18 |
| G1 单用 | 5.95 | 5.87 | **7.96** | 11.33 | 7.96 |
| **G1+A2** | 5.16 | 5.95 | 4.94 | 11.47 | **11.47** |

每个单干预都在**某个物体上有一次朝向崩溃**（A2 崩 box004、G1 崩 box023）；G1+A2 把这两次崩溃都削平，代价是均值不比 G1 更好。**这才是组合的卖点：不是更低的均值，而是更紧的最坏情况包络——在物体分布上更鲁棒。** 姿态门层面同理：A2 单用有 9–11 例 orientation 门 P→F，G1+A2 把它们几乎全数 F→P 救回（§3），净门通过构成 ≈ G1。

### 5.4 决策含义

- **z / 位置 / 穿透**：G1+A2 相对 G1 无净收益（可加-零 + 竞争），这些指标上 A2 是纯冗余。
- **朝向/姿态鲁棒性**：G1+A2 的确削平了单干预的最坏情况，但收益**物体特异、方向不一致、均值不超 G1**。
- 因此 **不把 G1+A2 或 A2 升级为新默认**：为「削 obj_ori 最坏情况」多引入 3 个 hand-gate 字段的配置复杂度，性价比不足；G1-only 仍是首选，A2 的 `INCONCLUSIVE_GATE_COLLAPSE` governance 不被推翻。若未来确有「跨物体朝向鲁棒性」硬需求，可把 G1+A2 作为**候选鲁棒档**，但需先在更大物体集上确认 5.2 的救援方向可预测。

### 5.5 box004 去除 086 两例后的四臂对比（n=4，敏感性检查）

去掉 `box004_20231003_2_086_p1` / `box004_20231003_2_086_p2` 两例，box004 剩 4 例（082_p1/p2、083_p1/p2）。四臂逐指标均值（从 `e198_factorial_by_case.tsv` 重算，`INT=G1A2−G1−A2+A0`）：

| 指标 | 方向 | G1+A2 | G1 | A2 | PRG | INT |
|---|:--:|---:|---:|---:|---:|---:|
| 物体 Z |误差| (cm) | ↓ | 5.518 | 5.467 | 5.979 | 5.790 | −0.138 |
| 物体 3D 位置误差 (cm) | ↓ | 12.126 | 11.980 | 12.340 | 13.386 | +1.192 |
| 物体朝向误差 (°) | ↓ | 13.384 | 13.250 | **18.205** | 12.647 | **−5.425** |
| 手-物 3mm 穿透占比 | ↓ | 0.127 | 0.107 | 0.087 | 0.153 | +0.086 |
| 承重接触 3mm in-mask | ↑ | 0.487 | 0.503 | 0.513 | 0.490 | −0.038 |
| 腿穿透占比 | ↓ | 0.000 | 0.000 | 0.023 | 0.016 | −0.007 |
| 根位置误差 (cm) | ↓ | 16.042 | 15.820 | 16.216 | 14.334 | −1.661 |
| 根朝向误差 (°) | ↓ | 12.443 | 12.895 | **15.494** | 11.811 | **−4.136** |
| 末端位置误差 (cm) | ↓ | 15.699 | 15.473 | 15.304 | 15.455 | +0.377 |
| 末端朝向误差 (°) | ↓ | 21.054 | 20.672 | 22.019 | 18.354 | −3.282 |
| 关节 jerk p95 | ↓ | 3306.978 | 3080.836 | 3425.233 | 2853.465 | −345.626 |

**结论**：去除 086 两例后，5.2 的「**G1 救 A2 朝向崩溃**」结构不但保持、且**更强**——A2 单用把 obj_ori 从 12.65° 炸到 **18.21°**（全集 n=6 时为 15.18°），G1+A2 拉回 13.38°≈G1，INT 由全集的 **−3.51 加深到 −5.43**；根朝向同向（A2 单用 15.49° 崩、G1+A2 救回 12.44°，INT=−4.14）。说明该救援效应**不是由 086 这两例驱动**的，反而 086 在全集里是稀释项。其余指标结论不变：z/位置/穿透上 G1+A2≈G1、A2 冗余或轻微竞争（穿透 INT=+0.086）。

### 5.6 box001 扩充（n=28，E196 修正参考）——「无回退→可加」的对照物体

按 [plan227](../plan/227_E198_box001_g1a2_a2_supplement_plan.md) 把因子扩到第 5 物体 box001：新跑 **G1+A2(28)+A2(28)=56 条 Full CEM**（本地 8 卡 priority 队列，0 失败），A0=E173 PRG、G1=**E196 corrected(21)+E194 clean(7)** 复用重打分。**运行时确认**：CEM 日志 `scene-act-reference: convention=XZY xml_axis_sequence=XZY parity=pass`——box001 用 E196 修正参考、fail-closed parity 通过，orientation 未被 Euler 错配污染（C1/C3 满足）。

四臂均值（`INT=G1A2−G1−A2+A0`，paired bootstrap 95% CI）：

| 指标 | 方向 | G1+A2 | G1 | A2 | PRG | INT [95% CI] |
|---|:--:|---:|---:|---:|---:|---:|
| 物体 Z |err| (cm) | ↓ | 3.274 | 3.233 | 4.761 | 4.846 | +0.125 [−0.019, +0.286] |
| 物体 3D 位置 (cm) | ↓ | 9.615 | 9.645 | 10.966 | 11.257 | +0.261 [−0.149, +0.705] |
| 物体朝向 (°) | ↓ | 4.872 | 5.007 | 5.612 | 5.927 | +0.180 [−0.275, +0.621] |
| 手-物 3mm 穿透 | ↓ | 0.135 | 0.178 | 0.183 | 0.212 | −0.014 [−0.059, +0.031] |
| 承重接触 in-mask | ↑ | 0.544 | 0.540 | 0.517 | 0.488 | −0.025 [−0.077, +0.031] |
| 腿穿透 | ↓ | 0.026 | 0.021 | 0.046 | 0.073 | +0.031 [+0.001, +0.070] |

gate 迁移（exact McNemar）：`A0→G1 lower_body +25.0pp`（7 F→P，**p=0.016**）、`A2→G1+A2 lower_body +17.9pp`（5 F→P，p=0.06）。

**关键点——box001 是「无单臂回退」的对照物体，交互退化为可加**：
- **obj_ori 上没有任何单臂崩溃**：G1 单用 5.01°、A2 单用 5.61° 都**单调改善**了 A0 的 5.93°（不同于 box023 的 G1 崩到 7.96°、box004 的 A2 崩到 15.18°）。既然没有任一单臂的朝向回退需要"救"，INT 就落在 0 附近（+0.180，CI 含 0）——**可加，而非 §5.2 的相互救援**。G1+A2=4.87° 仍是最优，但靠的是两个各自的小改善叠加。
- **z / 3D 位置 = G1 独占**（G1+A2≈G1、A2≈A0，INT≈0），与四物体一致。
- **穿透单调降**（0.212→0.135），G1+A2 最优。
- **lower_body 是 G1 的强项**：A0→G1 +25pp（本实验所有物体中最强的 G1 下肢改善），复现 G1 的支撑收益。

**这条负结果反而强化了 §5.3 的机制论断**：G1×A2 的"相互救援"**只在某个单臂在该物体上引入回退时才出现**（box023 A2 救 G1、box004 G1 救 A2）；box001 两个单臂都不回退，于是交互干净地退化为可加。5 物体判决仍为 `FACTORIAL_CHARACTERIZED`，**不升级 A2/G1+A2**。

## 6. 可视化（已完成）

CEM 队列以 `save_video=false` 跑（吞吐优先），离线渲染另做。初始 `osmesa`/`egl` 均失败
（本机缺 `libOSMesa`、EGL 无 NVIDIA PLATFORM_DEVICE）；**安装 `libosmesa6` 后 osmesa 软件渲染恢复**，
据此离线渲染 2×2 四单元视频。

**产物**：
- **4-cell MP4**（A0 左上 / G1 右上 / A2 左下 / G1+A2 右下，带 arm 标签与 12-gate pass 标记）：
  box024 P0 全 9 例 + box004 全 6 例，`results/E198/s6_downstream/render/full_factorial/E198_{case}_4cell.mp4`。
- **viser 交互复核**：`bash workspace/core4d/scripts/eval/wrappers/review_player.sh E198 --port 8080`
  已接入（review_index 增加 E198 arm-sweep 条目），**live qpos 回放全部 236 个 arm-case（4 臂×59），
  236/236 playable**，不依赖本机 GL。

**四阶段人审**（grasp/lift/carry/place contact sheet，`render/full_factorial/keyframes/{case}_4phase.png`，
逐例观察见 `render/full_factorial/visual_review.tsv`）。两条代表性实际观察：
- `box024_026_p1`（P0 强制极端 case）：右列 G1/G1+A2 全程把长箱托水平，左列 A0/A2 箱体明显前倾下沉；
  **A2≈A0（不修下沉），G1+A2≈G1**——直观印证「z 由 G1 独占」。
- `box024_028_p2`（E192 的 A2 PASS→FAIL case）：**G1+A2 达 12/12 全门通过，A2 单臂 fail**；carry 阶段
  G1+A2 机器人更直立、A2 更下蹲前倾——**视觉印证「G1 救 A2 姿态门」**（§3 的 A2→G1+A2 F→P）。
box024 P0(9)+box004(6) 全 15 例四阶段帧均无摔倒/飞散/脱手；右列箱姿普遍比左列平稳，G1+A2≈G1。
box021/023 的 44 例 4-cell 已渲完。

**box001 补充（plan227）视觉复核**（28 例 4-cell + 四阶段 contact sheet 全部产出，`keyframes/box001_*_4phase.png`）：
- `box001_20231020_014_p1`（obj_ori A0=7.05→G1=3.80→A2=4.48→G1+A2=3.41，单调改善）：**G1+A2 达 12/12**，
  grasp→place 全程右列（G1/G1+A2）把大蓝箱托稳、朝向受控；无摔倒/穿透/脱手——直观印证 box001 上
  **G1、A2 都各自改善朝向、组合最优且为可加**（无单臂崩溃可救）。
- `box001_20231003_1_040_p1`（A0 lower_body FAIL→G1 PASS）：lift/carry 阶段**右列 G1/G1+A2 机器人明显更直立、
  腿部支撑更稳**，左列 A0/A2 更前倾下蹲——**视觉印证 §5.6 的「A0→G1 lower_body +25pp」下肢支撑收益**。
box001 全 28 例四阶段帧无摔倒/飞散/脱手。5 物体逐例可在 viser 交互细审：
`review_player.sh E198 --arm G1A2 --port 8082`（`--arm` 为 plan227 新增过滤，只看 G1+A2）。

## 7. 复现入口

```bash
# 1. 构建 103-row priority manifest + 快照 + 单变量审计
MUJOCO_GL=osmesa .venv/bin/python workspace/core4d/scripts/experiments/E198/build_g1a2_manifest.py --apply --snapshot
# 2. 本地 8 卡 priority 队列（canary 后 Full）
MODE=canary bash workspace/core4d/scripts/launch/active/run_E198_local_8gpu.sh
MODE=full GPUS=0,1,2,3,4,5,6,7 PER_GPU_MEM_MIB=5000 MAX_PER_GPU=1 bash workspace/core4d/scripts/launch/active/run_E198_local_8gpu.sh
# 3. 四臂 2×2 因子 eval + 报告
bash workspace/core4d/scripts/eval/wrappers/eval_E198_factorial.sh
PYTHONPATH=workspace/core4d/scripts .venv/bin/python workspace/core4d/scripts/eval/reports/gen_E198_factorial_report.py
```

## 8. 产物

- 评估目录：`workspace/core4d/results/E198/s6_downstream/eval/full_factorial/`
  - `e198_factorial_by_case.tsv` / `e198_factorial_by_object.tsv` / `e198_gate_migrations.tsv` / `e198_factorial_summary.json` / `e198_arm_cache.tsv`（236 行）
  - `E198_G1xA2_factorial_report.md`（markdown 报告）
  - **`E198_G1xA2_factorial.xlsx`**（工作簿 4 sheet：四臂×物体对比 / **G1+A2逐例**（59 例 12 指标 + 12-gate 通过明细，整体 19/59 过）/ Gate迁移(McNemar) / 逐例(by_case)，生成器 `scripts/eval/reports/gen_E198_xlsx.py`）
- CEM 输出：`results/E198/s6_downstream/cem/full_g1a2/`（59）；`results/E192/s6_downstream/cem/full_a2_expansion/`（44）
- Manifest/快照：`results/E198/s6_downstream/manifests/e198_priority_*_manifest.tsv`；`results/E198/scene_snapshot/g1a2/`、`results/E192/scene_snapshot/a2_expansion/`

## 9. 下一步

1. 补 viser 强制视觉复核（§6），回填实际观察后将 C9 置 PASS。
2. 若继续，优先解释 box023/box004 的 object_ori 协同机制，而非把 G1+A2 当默认推广。
3. 不重复 A2 单参数 seed；不叠加更多干预后声称单机制归因（延续 E192 下一步纪律）。
