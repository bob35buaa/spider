# 实验诊断 v3 — SPIDER 手物接触为何落后 OmniRetarget（after E143）

日期：2026-06-04
范围：复盘 E098–E143 的核心结论，深入分析 E143 `raw_mask_ref_fk` 24-case 对比表（含用户在 `Spider成功逐case` sheet 的人工备注 / 颜色标注 / RL 结果），回答用户的核心问题——**SPIDER 的接触落后于 OmniRetarget，到底是 (a) 动捕数据质量差、(b) OmniRetarget 算法本身不好、还是 (c) SPIDER 算法的问题**。
执行：spider 主仓库本地 GPU 节点，**read-only，未改任何 pipeline / 算法代码**（遵循任务约束 1）。
配套：用 subagent 做了 metric 定义审计和 E098–E143 日志/数据管线测绘，结论已并入本文（遵循约束 2）。

工作区：
```
workspace/exp_diagnostic_v3/
├── diagnostic_v3_report.md        ← 本文件
├── scripts/                       ← （本轮以分析为主，复用既有 eval 脚本，无新增算法脚本）
└── results/
    ├── frames/                    ← 从 mocap_omni_spider_cmp 三联视频抽的对比帧
    └── figures/                   ← 报告引用的关键证据图
```

---

## TL;DR（一句话结论）

**“SPIDER 接触落后 OmniRetarget” 这个命题，在当前评测口径下有一大半是伪命题。** 因为 E143 的“手物接触”指标是在**静态运动学回放（`mj_forward`，无物理求解）**下算的，一帧被判定为“接触”当且仅当**机器人手的碰撞几何嵌进了箱子内部**——它在数值上几乎等于“手物穿透”指标（24/24 case 完全相等，OmniRetarget/ref_fk 也有 23/24 相等）。所以 OmniRetarget 的高“接触”分**主要来自它把手插进箱子**；SPIDER 的物理优化把深穿透从 29.4% 压到 3.1%，机械地拉低了这个“接触=穿透”分。这不是 SPIDER 退步，而是**指标本身把穿透当成了接触**。

但这不等于 SPIDER 没问题。剔除穿透伪信号、只看**诚实的近场指标（5cm/10cm SDF band）**，SPIDER 在干净 case 上仍平均落后 OmniRetarget **5cm: −4.8pp，10cm: −2.7pp**。差距小、但真实，方向是“**手够到了物体附近，但没有稳定维持贴合接触**”。

三类归因的最终配比（22 个 Spider 成功 case）：
- **数据/任务定义问题（不该用来评判 SPIDER）：≈ 13/22**（8 mocap/object 质量 + 2 非目标传递动作 + 3 OmniRetarget/ref 语义已错）。
- **OmniRetarget 不好、SPIDER 救了碰撞但没救回接触：3/22**（leg/非手碰撞下降但 contact 更低——是“偏保守”而非“失败”）。
- **SPIDER 真实算法差距（干净 case 上够到但接触不持续）：≈ 6–9/22**。

**所以三个原因都成立，但权重和很多人直觉相反**：最大的单一来源是**评测指标缺陷 + 数据质量**，其次才是 SPIDER 的“接触保持”算法短板；OmniRetarget“算法不好”反而是 SPIDER 接触低的一个**正向副作用**（SPIDER 是在修 OmniRetarget 的穿透）。

---

## Part 0 — 必读：评测指标本身的陷阱（这是本轮最重要的发现）

### 0.1 “手物接触”在静态回放里 == “手物穿透”

E143 的对比表（`eval_E143_raw_mask_ref_fk_24case.py`）对每个 variant 的 qpos 序列做**逐帧 `mj_forward` 静态回放**，没有物理积分、没有接触求解器把物体推开。在这种回放里：

- **`手物穿透 hand_object_penetration`** = `frac(hand_SDF < 0)` = 手碰撞几何**嵌进箱子**的帧占比。
  （`eval_E105_box026_clean_cem.py:159-179` + `eval_E143...py:218`）
- **`手物接触 hand_object_contact`** = MuJoCo `data.ncon` 里出现“手 geom × 箱 geom”接触对的帧占比。
  （`eval_E105...py:227-238,282`）
- MuJoCo 的窄相碰撞**只有在两个 geom 互相穿插时**才会生成接触对。所以“产生 physics contact 的帧”≈“手嵌进箱子的帧”。

实测确认（`results/E143/.../e143_case_by_case.tsv`）：

| method | contact==penetration 的 case 数 | max\|差值\| |
|---|---|---|
| raw_mask_ref_fk (SPIDER) | **24 / 24** | 0.0000 |
| OmniRetarget | 23 / 24 | 0.064 |
| ref_fk (旧 Spider CEM) | 23 / 24 | 0.104 |

> 含义：在这套口径下，“接触越高”几乎严格等价于“穿透越多”。把 `hand_object_contact` 当成“接触质量”来比较 SPIDER vs OmniRetarget，**方向上是错的**——它奖励穿透。

### 0.2 拆 band 后真相浮现（E110 已量化，本轮复核）

E110 把接触按 SDF band 拆开（`140_E110_contact_metric_audit_results.md`）：

| method | physics contact | hand≤5cm | hand≤10cm | **deep pen** | shallow pen |
|---|---:|---:|---:|---:|---:|
| OmniRetarget | 0.544 | 0.646 | 0.671 | **0.294** | 0.253 |
| Spider CEM (ref_fk) | 0.431 | 0.618 | 0.657 | **0.031** | 0.404 |
| Δ (Spider−Omni) | −11.3pp | −2.8pp | **−1.4pp** | **−26.3pp** | +15.1pp |

读法：
- SPIDER 把 **deep penetration 从 29.4% 砍到 3.1%**（−26.3pp）——这是它该做的、正确的物理修正。
- 代价是“接触=穿透”分掉 11.3pp，但**诚实的 10cm 近场只掉了 1.4pp**。
- 也就是说 SPIDER 把 OmniRetarget 的“深插”转成了“浅插 / 贴近”，**手并没有离开物体**。
- E110 的 failure label：`penetration_removed_contact_not_recovered` 18/24——名字应该读成“**穿透被正确移除，但（穿透式的假）接触没在无穿透前提下补回**”。

### 0.3 诚实指标下 SPIDER 仍有真实差距

只看 5cm/10cm（不奖励穿透），SPIDER 仍落后（24-case 全集，vs OmniRetarget）：

- mean Δ near5 = **−9.1pp**（仅 1/24 case raw≥omni）
- mean Δ near10 = **−5.4pp**（仅 2/24 case raw≥omni）
- mean Δ leg_pen = **−2.9pp（SPIDER 更好）**，9/24 case SPIDER 腿穿透更低。

在**干净 case 子集**（8 个 valid-like）上差距收窄：near5 Δ = **−4.8pp**，contact Δ = −13.2pp，leg_pen Δ = +3.7pp（略差）。

> 结论：剔除穿透伪信号后，SPIDER 的真实短板是“**最后 5cm：够到了但没贴住/没持续承重**”，幅度是个位数百分点，不是几十个百分点。

---

## Part 1 — 用户三个假设的逐一裁决

用户问：接触落后是 (a) 动捕数据差、(b) OmniRetarget 算法差导致 SPIDER 更差（或 Omni 差但 SPIDER 救回一点）、还是 (c) SPIDER 算法问题。**答案是三者都有，按贡献排序如下。**

### 因素 1（最大）：评测指标缺陷 —— “接触”实为“穿透”

见 Part 0。这不在用户给的三个选项里，但它是**最大的混淆源**：它让 OmniRetarget 凭“插得更深”赢得名义接触分。任何后续对比都必须先换指标，否则会持续误判 SPIDER。

### 因素 2（很大）：动捕 / 物体 / 任务定义质量（用户假设 a，成立）

22 个 Spider 成功 case 里，**13 个**根本不该进“抱箱/搬箱接触”主比较（用户自己的颜色标注已经标出来了）：

| 颜色/类别 | case 数 | 代表 case | 问题 |
|---|---:|---|---|
| 浅紫（箱子旋转~180°） | 4 | box026_133_p1/p2, box026_*_139_p2×2 | 箱子大幅旋转，非稳定抱箱目标 |
| 黄（箱子轨迹非 GT） | 3 | box026_135_p1/p2, box026_137_p1 | 物体轨迹不是我们要的参考 |
| 灰（非期望序列） | 2 | box026_141_p1/p2 | 是“传递”动作，不是搬箱 |
| 深紫（走上前再搬） | 3 | box026_039_p2, box026_134_p1, box026_138_p2 | 含 walk-up，初始无稳定接触 |

这些 case 的 contact deficit 最大（mocap/object 类 −21.2pp、非目标类 −33.2pp、ref 问题类 −33.5pp），会**严重放大 SPIDER 的名义落后幅度**。其中 `box004_082_p1`（物体动捕抖动）却 RL 成功、contact 差值仅 −0.9pp，说明抖动数据并不必然拖垮下游。

**视觉证据**（`results/figures/fig3_box026_133_*.png`）：box026_133_p1 的 mocap 里人只是弯腰、箱子留在地上没被抬起——这种 case 两个 robot 都没接触箱子，拿它比接触毫无意义。

### 因素 3（中等）：OmniRetarget / ref 语义已错，SPIDER 被坏参考牵引（用户假设 b，部分成立）

3 个 case（box026_039_p2, box026_138_p2, box026_134_p1）的根因在 OmniRetarget/ref 之前就坏了：mocap 是“走上前再搬”，ref 在初始阶段已把手放到不合理接触点。这一组 leg_pen Δ = −8.3pp（**SPIDER 明显更安全**）但 contact Δ = −33.5pp。

读法正是用户假设的第三种子情形——“**OmniRetarget 不好，SPIDER 救回了一部分（碰撞/腿穿透降低），但没救回接触**”。机制：SPIDER 的优化偏保守，宁可避免坏接触也不强行重建一个语义本就错的接触。**这类 case 靠调 SPIDER reward 救不回来，得先做 ref repair / 重选 contact window。**

**视觉证据**（`results/figures/fig1_box026_138_*.png`，下方放大对比）：
- **OmniRetarget（中）**：右手整只**插进箱子顶面**（mesh 被压出凹陷），这就是它“接触分高”的来源——穿透。
- **SPIDER（右）**：手停在箱口边缘**不穿透**，但箱体已漂移、接触没建立。

这一帧是整份报告的缩影：Omni 的“接触”是穿透，SPIDER 的“无接触”是它拒绝穿透。

### 因素 4（真实但幅度小）：SPIDER 接触保持算法短板（用户假设 c，成立但被高估）

在干净 case（box021_035_p1/p2, box021_029_p2, box004_083_p1/p2, box023_person2 等）上：
- 大动作语义三列一致（人/Omni/SPIDER 都在搬箱，见 `fig2_box021_035_*.png`）；
- 5cm gap（−4.8pp）远小于 contact gap（−13.2pp）→ **不是 reach failure，是 contact persistence failure**；
- 表现为：手贴到侧面/下缘、轻微漂移、没有把“双手支撑 + 箱体同步运动”锁成强约束。

这是 SPIDER 唯一应该自己负责、且**值得投入优化**的部分。它的本质是 reward/CEM 当前更容易接受“手靠近 + 姿态稳 + 物体大致跟随”的解，而没有硬约束“接触建立时机（lift onset）/ 双手承重 / 手-物相对位姿锁定 / 接触段连续性”。

---

## Part 2 — 关键数据与证据汇总

### 2.1 方法均值（24-case 全集）

| method | 手物接触(=穿透) | 5cm | 10cm | 手物穿透 | 腿穿透 |
|---|---:|---:|---:|---:|---:|
| OmniRetarget | 0.544 | 0.646 | 0.664 | 0.547 | 0.113 |
| ref_fk (旧 Spider) | 0.431 | 0.618 | 0.647 | 0.435 | 0.100 |
| raw_mask_ref_fk (新 Spider) | 0.330 | 0.555 | 0.610 | 0.330 | **0.085** |

注意最后一列：SPIDER 腿穿透最低（最安全）。而前四列 SPIDER 最低——但前四列里“接触”和“穿透”是同一个东西，**SPIDER 接触低 = SPIDER 穿透低 = 物理上更干净**。

### 2.2 0/24 超过 OmniRetarget —— 但这是指标的胜利不是 SPIDER 的失败

`raw_exceeds_omni_contact = TRUE: 0/24`。在“接触=穿透”口径下，要超过 OmniRetarget 就必须**比它穿得更深**，而 SPIDER 的全部价值就是不穿透。所以 0/24 是**指标设计的必然结果**，E142（`171_E142...`）报的“0/12、0/7 超过 Omni”同理。

### 2.3 RL 结果交叉验证

4 个 RL 成功 case：box021_035_p1, box021_035_p2, box023_person2, box004_082_p1——**全部落在 valid-like 干净子集**，且都是 contact 名义落后 Omni 的 case。这反证：**名义接触分低 ≠ 下游不可用**。RL 能成功恰恰因为 SPIDER 的输出物理上自洽（不穿透），是可被策略跟踪的。

---

## Part 3 — 下一步规划（不改算法，先改评测与数据；再谈 reward）

遵循约束 1（本轮不改算法）。以下为**建议的后续实验设计**，按 ROI 排序。

### P0 — 立刻换评测口径（半天，纯分析，必须先做）

1. **废弃“physics contact / hand_object_contact”作为接触质量主指标**，因为它在静态回放下=穿透。改用**无穿透贴合度**：
   - 主指标：`frac(0 ≤ hand_SDF ≤ 2cm)`（贴而不穿）+ `frac(2cm < hand_SDF ≤ 5cm)`（近）。
   - 穿透单列为**惩罚**项（越低越好），不再混进“接触”。
2. **建一个“无穿透接触质量”复合分**：`good_contact = frac(SDF∈[0,2cm])`，对三方法重算 24-case。预期 SPIDER 在此指标上反超或持平 OmniRetarget（因为 Omni 的接触大多是 SDF<0 的穿透）。这是把“0/24 落后”翻译成真实结论的最小一步。
3. 复用现成脚本：`eval_omni_vs_spider/unified_replay_eval.py` 已经输出 `near_5/near_10/deep/shallow` band，只差一个 `[0,2cm]` band 的聚合——E110 的 `contact_metric_audit.py` 已注明缺 2cm 阈值，补上即可。

### P1 — 固定 clean benchmark（半天，数据筛选）

4. **冻结一个干净接触 benchmark**（6 个主 case）：box021_035_p1, box021_035_p2, box021_029_p2, box004_083_p1, box004_083_p2, box023_person2（注明箱小）。
   - dirty/旋转/walk-up/传递动作 case 单独成 “stress set”，**不进主平均**。
   - 谨慎组：box004_082_p1（抖动但 RL 过）、box026_139_p1（mixed）。
5. 所有后续 SPIDER vs OmniRetarget claim **只在 clean benchmark 上、用 P0 的无穿透指标** 下结论。

### P2 — 对 ref-bad case 做 ref repair（1 GPU 日，数据侧）

6. box026_039_p2 / box026_138_p2 / box026_134_p1 这类“ref 接触点已错”，**不在 SPIDER 端补**，而是回到 OmniRetarget 输入：重选 contact window / 用 raw mocap 指尖投票修初始接触点（接 v2 报告 B6 的根治路线）。验证：repair 后 ref 的 5cm band 是否回到 mocap 水平。

### P3 — 针对真实短板的 SPIDER 接触实验（多 GPU 日，需改 reward，**单独开实验、git 隔离**）

> 注意：这一步需要改算法代码，**不在本诊断范围内**；下面只列出可证伪的实验设计，供后续 E1xx 立项（按 `.claude/rules/experiment.md` 走 `feat/E{NNN}-*` 分支 + 实验记录）。

7. **contact persistence reward**：在 raw/ref contact mask 为真的窗口内，奖励**连续无穿透贴合**（run-length），而非逐帧 near。可证伪：clean benchmark 上 `frac(SDF∈[0,2cm])` 的最长连续接触段 ↑ ≥ X%，且 leg_pen 不退化。
8. **bilateral support + lift-onset gate**：抱箱 case 要求双手在物体起升前已建立接触。可证伪：lift onset 时刻双手 SDF≤2cm 的 case 比例 ↑。
9. **hand-object relative pose lock**：接触后手相对物体 local 位置漂移 < 阈值。可证伪：接触段内 hand-in-object-frame 位置方差 ↓。
   - 每条都必须：clean benchmark 上跑、报 mean+std+worst、配 A/B 视频（约束 5）。

### P4 — 防退化与可视化

10. 把 P0 的“无穿透贴合度”指标也接进 `mocap_omni_spider_cmp` 三联视频的角标，肉眼+数值同屏，避免再出现 v2 报的“指标好但视觉趴箱”盲区。

---

## Part 4 — 直接回答用户的提问

> “当前 spider 的接触落后于 omniretarget 的原因，是动捕数据本身质量差，还是 omniretarget 算法本身就不好从而导致 spider 更不好（或者 omniretarget 不好但 spider 救回一点了），还是 spider 算法的问题？”

**三者都有，但首要原因在选项之外——评测指标把穿透当接触：**

1. **评测口径缺陷（最大）**：静态回放下“手物接触”≡“手物穿透”（24/24 数值相等）。OmniRetarget 的高接触分≈它把手插进箱子；SPIDER 把深穿透 29.4%→3.1%，机械拉低该分。**0/24 超过 Omni 是指标的必然，不是 SPIDER 退步。**
2. **动捕/物体质量差（很大，= 用户假设 a）**：22 个里 13 个是箱子旋转/轨迹非 GT/传递动作/walk-up，根本不该进主比较，且贡献了最大的名义落后幅度。
3. **OmniRetarget/ref 不好、SPIDER 救了碰撞没救回接触（= 用户假设 b 的第二子情形）**：3 个 case，leg_pen 明显下降（SPIDER 更安全）但 contact 更低——是“偏保守”，需 ref repair 才能救。
4. **SPIDER 真实算法差距（= 用户假设 c，真实但被高估）**：剔除穿透伪信号后，干净 case 上 5cm 仅落后 4.8pp，是“够到了但接触不持续/不承重”，这是唯一值得调 SPIDER reward 的部分，幅度个位数百分点。

**一句话**：SPIDER 没有“落后”——它在用正确的物理把 OmniRetarget 的穿透式假接触换成无穿透的真接触，现有指标惩罚了这件正确的事。真正要做的是 (1) 换成无穿透贴合指标，(2) 在干净 benchmark 上重判，(3) 只针对“最后 5cm 接触保持”做小幅 reward 改进。

---

## 附录 — 证据图

- `results/figures/fig1_box026_138_penetration_vs_no_penetration.png` — **核心证据**：Omni 手插进箱顶（穿透=高接触分），SPIDER 手停箱口不穿透。
- `results/figures/fig2_box021_035_clean_case.png` — 干净 case，三列语义一致，差距在“接触保持”。
- `results/figures/fig3_box026_133_box_rotation_not_lifted.png` — 数据质量问题：mocap 箱子没被抬起，不该进比较。
- `results/figures/fig4_box004_082_jitter_rl_success.png` — 抖动数据但 RL 成功、contact 差值仅 −0.9pp。
- `results/frames/*_p40/p65/p90.png` — 各 case 三时点对比帧。

## 附录 — 关键 file:line 索引

| 内容 | 位置 |
|---|---|
| E143 对比表生成 | `workspace/core4d/scripts/eval/eval_E143_raw_mask_ref_fk_24case.py` |
| 手物穿透 = `frac(SDF<0)` | `eval_E105_box026_clean_cem.py:159-179` + `eval_E143...:218` |
| 手物接触 = `data.ncon` 物理接触对 | `eval_E105...:227-238,282` |
| 统一 replay SDF/contact | `workspace/core4d/scripts/eval_omni_vs_spider/unified_replay_eval.py:255-292,377-458` |
| CEM WORK/FAIL gate | `eval_E105_box026_clean_cem.py:66-105` |
| E110 band 拆解结论 | `workspace/core4d/log/140_E110_contact_metric_audit_results.md` |
| 既有失败分析（22 case 分类） | `workspace/core4d/results/E143/spider_contact_failure_analysis/report.md` |
| v2 报告（B6 接触语义错位根因） | `workspace/exp_diagnostic_v2/diagnostic_v2_report.md` |
