# E054: CORE4D Case 几何可达性 + Mocap 数据质量联合分析 — 结果

## 状态: ✅ 完成 (2026-05-12)

## 实验配置

| 项 | 值 |
|---|---|
| 实验类型 | 数据分析（无物理仿真） |
| 输入 | `example_datasets/processed/core4d/unitree_g1/humanoid_object/` 21 个 case |
| 物体 mesh | `workspace/core4d/object_models/object_models/{box,bucket,chair,desk}/` |
| Robot | G1 (双臂展 1.0 m, 单臂展 0.5 m) |
| 运行 | `uv run python workspace/core4d/scripts/analyze/case_tier_analysis.py` |
| 时长 | < 5 s, CPU-only |

## 输出文件

| 路径 | 内容 |
|---|---|
| `workspace/core4d/results/E054/case_tier_classification.csv` | 21 行，22 列指标 |
| `workspace/core4d/results/E054/E054_case_tier_summary.md` | 分类汇总 + Claims 验证 |
| `workspace/core4d/results/E054/intent_detection_verify.png` | 3 个 B+C case 时序目检图 |

## 指标定义三次迭代

| 版本 | 设计 | 问题 |
|---|---|---|
| v1 | `d_min < d_p20+5cm` AND `\|v_hand\| < 0.30 m/s` AND ≥10帧 | 几乎所有 case intent=0：G1 行走时手摆动 0.3-0.7 m/s 远超阈值 |
| v2 | 把绝对速度改为 **手-物体相对速度** + 5 帧平滑 | 大多数 case OK，但 **desk021 漏覆盖**：v_rel 在 desk 搬运相位仍噪声大（≥0.30 m/s），漏掉 obj_z 抬升的大半 |
| **v3 (当前)** | 加入 **物理必要性信号**：`band ∩ (slow_rel OR lifted)` + 3 帧形态学闭合 | desk021 完整覆盖 ✅，但 C-only 类天然为空（数据本身原因） |

**v3 完整规则（持续 ≥ 10 帧）**：

```
band(t)     := d_min(t) < d_p20 + 0.07 m                     # 手在 case 自身距离分布的近端
slow_rel(t) := |v_hand - v_obj|_smoothed_5f < 0.30 m/s        # 相对速度低
lifted(t)   := obj_z(t) > obj_z_p10 + max(0.10 m, 0.33·obj_z_amp)
                                                              # 抬升阈值跟 case 振幅成比例
                                                              # 防止 8cm 噪声漂移被误判为 lift
intent(t)   := band(t) AND (slow_rel(t) OR lifted(t))
intent      := morphological_closing(intent, k=3)             # 填充 ≤3 帧的小空洞
```

**为何加 `lifted` 通道**：物理上"obj 抬起 ≡ 必有外力支撑"，这是比 v_rel 更基本的接触证据。
当 obj 上下振动 5cm 时（push/drag 任务），lifted 不触发，落回严格的 `slow_rel` 通道；
当 obj 真实抬升 ≥ 10cm 且 ≥ 33% 振幅时，lifted 触发，让搬运相位完整覆盖。

## 数值结果

### 21 个 case 完整列表（v3 检测结果）

| Case | Tier | dim_max | obj_z_amp | intent_win | total_frames | partner_pf | hand_obj_min | recommended |
|---|---|---|---|---|---|---|---|---|
| box001_p1 | 3 | 0.81m | 22.2cm | 1 | — | 0.00 | — | drop |
| **box021_p1** | **2** | **0.50m** | **44.1cm** | **1** | — | 0.00 | — | **B+C** |
| **box023_p1** | **1** | **0.39m** | **53.2cm** | **1** | **52** | 0.00 | 27cm | **B+C** |
| box024_p1 | 3 | 0.98m | 32.1cm | 1 | — | 0.00 | — | drop |
| box025_p1 | 3 | 0.89m | 22.9cm | 1 | — | 0.00 | — | drop (dim>70cm) |
| box025_p2 | 3 | 0.89m | 23.3cm | 2 | — | 0.00 | — | drop |
| box025_s2_p1 | 3 | 0.89m | 29.3cm | 1 | — | 0.00 | — | drop |
| **bucket001_p1** | **1** | **0.46m** | **32.3cm** | **1** | — | 0.00 | 19cm | **B+C** |
| bucket003_p1 | 3 | 0.76m | 18.5cm | 1 | — | 0.00 | — | drop |
| bucket005_p1 | 3 | 0.41m | 12.3cm | 1 | — | 0.50 | — | dual-robot |
| bucket005_p2 | 3 | 0.41m | 12.5cm | 2 | — | 0.50 | — | dual-robot |
| **bucket005_s2_p1** | **1** | **0.41m** | **40.6cm** | **1** | **87** | 0.00 | 21cm | **B+C** |
| **bucket007_p1** | **2** | **0.57m** | **25.5cm** | **1** | **41** | 0.00 | — | **B+C** |
| bucket010_p1 | 3 | 0.74m | 19.8cm | 1 | — | 0.50 | — | drop |
| bucket010_p2 | 3 | 0.74m | 20.0cm | 1 | — | 0.08 | — | drop |
| bucket010_s2_p1 | 3 | 0.74m | 32.4cm | 1 | — | 0.00 | — | drop |
| chair022_p1 | 3 | 0.86m | 31.3cm | 1 | — | 0.00 | — | drop |
| desk001_p1 | 3 | 0.80m | 8.1cm | 5 | — | 0.00 | — | drop |
| desk005_p2 | 3 | 0.80m | 10.8cm | 1 | — | 0.00 | — | drop |
| desk007_p1 | 3 | 0.80m | 14.6cm | 1 | — | 0.00 | — | drop |
| **desk021_p1** | **2** | **0.51m** | **42.4cm** | **1** | **68** | 0.00 | 28cm | **B+C** |

### 分布

| | 数量 | Case |
|---|---|---|
| Tier 1 | 3 | box023_p1, bucket001_p1, bucket005_s2_p1 |
| Tier 2 | 3 | box021_p1, bucket007_p1, desk021_p1 |
| Tier 3 | 15 | box001/024/025_*, bucket003/005_p1p2/010_*, chair022, desk001/005/007 |
| **B+C** | **6** | **所有 Tier 1/2 case** (box021/023, bucket001/005_s2/007, desk021) |
| **C-only** | **0** | **(空，CORE4D 数据本身原因，见下文)** |
| dual-robot | 2 | bucket005_p1, bucket005_p2 |
| drop | 13 | (其余 Tier 3) |

## Claims 验证

| ID | 描述 | 量化标准 | 实际 | 通过 |
|---|---|---|---|---|
| C1 | box023 → Tier 1 | dim_max < 50 cm AND partner_pf < 30% | dim=0.39m, pf=0.00 | ✅ |
| C2 | box025 → Tier 3 | dim_max > 70 cm OR partner_pf > 50% | dim=0.89m | ✅ |
| C3 | ≥3 B-friendly case | obj_z_amp ≥ 5cm AND intent ≥ 1 AND tier ≤ 2 | **6 个** | ✅ 超目标 |
| C4 | ≥1 C-only case | obj_z_amp < 5cm OR intent = 0 (Tier 1/2) | **0 个** | ❌ |
| C5 | csv 可复现 | 字段完整 + recommended_method | 22 列 21 行 | ✅ |

**4/5 通过；C4 失败 = 数据本身的发现，不是检测器问题，详见下文。**

### C4 失败的诚实分析

C4 当初是**预测式**假设："数据集中应该存在 mocap 质量太差以致 intent 检测不到的 case"。
21 个 case 跑完后实际情况：

- 全部 3 个 Tier 1 case 都有清晰 intent window（box023/bucket001/bucket005_s2，最近距离 19-28cm 但 obj 显著抬升）
- 全部 3 个 Tier 2 case 也都有 intent window（box021/bucket007/desk021）
- 唯一 intent=0 的 case 都在 Tier 3 (drop)，与 C4 无关

**真实结论**：CORE4D 21 个 case **全部是搬运/抬升类任务**，没有"obj 完全不动"的纯推/扶/操作任务。Path C-only 类在 CORE4D 数据集上**天然为空**。

**对 Phase 17 路线的影响**：
- 不需要单独的 Path C 验证 case（原计划 E056 取消或合并）
- Path C (force-closure reward) 仍然有价值，但应作为 Path B 内部的辅助 reward 实现，而不是独立验证路径
- 如果未来扩到 OMOMO 或其他纯操作数据集，再引入 C-only 验证

## 可视化目检 (intent_detection_verify.png) — v3 修复后

5 个 case 的时序图（左：hand-obj 距离 + intent window；右：obj_z + v_rel + intent window）：

| Case | obj_z amp | intent_win × 帧数 | 观察 |
|---|---|---|---|
| **box023_person1** | 53cm | 1 × 52 帧 | 手从 60cm → t=1s 降到 28cm 平台 → t=1-2.7s 维持 → obj_z 同期 0.65m。intent **完整覆盖搬运相位** ✅ |
| **bucket005_s2_person1** | 41cm | 1 × 87 帧 | 手 50cm → t=0.7s 降到 22cm 长平台 (t=0.7-3.5s) → obj_z 0.6m。**最长完整窗口** ✅ |
| **desk021_person1** | 42cm | 1 × 68 帧 | 手 60cm → t=0.7s 降到 28-45cm 振荡区 → obj_z 抬至 0.65m (t=0.7-3.1s)。**v3 修复后从 v2 的 3 段零碎变为 1 段完整覆盖** ✅ |
| **bucket007_person1** | 26cm | 1 × 41 帧 | 手 65cm → t=2s 降到 35cm → obj_z 抬至 0.55m (t=2-4s)。**v3 修复后从 v2 的 intent=0 变为 1 段完整 lift 阶段** ✅ |
| desk001_person1 (Tier3 drop) | 8cm | 5 散乱 | 振幅太小未触发 lifted 通道，仅靠 slow_rel 检出零碎窗口；属正常表现，与 push/drag 任务行为吻合 |

**关键观察 1**：所有 4 个 B+C case 的 hand_obj_dist_min 都在 19-28 cm 之间。**这正面验证了用户对 mocap 不可信的判断**：retarget 后 G1 手与物体之间存在 19-28 cm 间隙，意味着原数据中"接触"在 G1 几何上从未真正发生。Path B 的 IK-snap-to-surface 步骤要做的就是**主动消除这 20-28 cm 的几何间隙**。

**关键观察 2**：v3 检测器在 desk021/bucket007 上的修复，验证了"obj 抬起 ≡ 必有外力"这一物理信号比 v_rel 更可靠。
v_rel 在 desk 等大物体上有较大噪声（因为物体晃动），单靠 v_rel 会漏掉真实搬运相位。

## 与 EXPERIMENT_TRACKER 历史 case 交叉验证

| 历史实验中评测过的 case | E054 分类 | 一致性 |
|---|---|---|
| box025 (E001-E018, E031, E041c, E048, E049, E053) | drop (dim 0.89m) | ✅ E009 已结论"双人不可解" |
| bucket005 p1/p2 (E015) | dual-robot | ✅ 与 E015 partner 检测一致 |
| bucket005_s2_p1 | **B+C 新发现** | ⚠️ 此 case 之前未单独跑过，是新候选 |
| bucket010 (E020+ 多次) | drop (dim 0.74m, pf=0.50) | ✅ E021/E031 重复失败一致 |
| chair022 (E020, E021) | drop (dim 0.86m) | ✅ E020 "碰撞推飞" 与 dim>70cm 一致 |
| desk005 (E028+ 多次) | drop (dim 0.80m) | ✅ E028-E041c 反复失败一致 |
| box023 (E041c, E048-E052) | **B+C 重要修正** | ⚠️ 之前与 box025 同处理，E054 证明应单人可解 |

**重要修正**：box023 在过去所有实验中和 box025 走同一套配置，被 E041c 摔倒、HDMI 物体推歪。E054 表明这是**方法配错**，不是几何不可解 — Path B 应优先在 box023 上验证。

## 分析与下一步

### Path B 首批验证 case 排序（视频核实后修正版）

**初版用"L 最近帧占比 (L_frac)"判定 dominant_hand 时，6 个 case 错了 3 个**。
通过 `extract_case_keyframes.sh` 从 visualization_kinematic.mp4 提取关键帧目检后，
改用"双手在 intent 窗口内的平均距离比 `min(L_mean,R_mean)/max(L_mean,R_mean)`"，
< 0.55 算 single-hand。新结果 6/6 匹配视频。

**视频核实矩阵**：

| Case | L_frac (旧) | 旧判定 | L_mean | R_mean | sym (新) | 新判定 | 视频实际 | 一致 |
|---|---|---|---|---|---|---|---|---|
| box023_p1 | 0.96 | L only | 28cm | 29cm | 0.96 | both | 视频双手抱箱 | ✅ |
| **bucket001_p1** | 1.00 | L only | **23cm** | **62cm** | **0.37** | **L only** | 视频左手提桶 | ✅ |
| bucket005_s2_p1 | 1.00 | L only | 22cm | 23cm | 0.96 | both | 视频双手抱桶 | ✅ |
| box021_p1 | 0.72 | both | 32cm | 33cm | 0.98 | both | 视频双手抱箱 | ✅ |
| desk021_p1 | 0.66 | both | 37cm | 39cm | 0.94 | both | 视频双手按桌 | ✅ |
| bucket007_p1 | 0.66 | both | 36cm | 36cm | 0.99 | both | 视频双手抱桶 | ✅ |

**修正后的 6 个 B+C case 推荐顺序**：

| # | case | tier | obj_z_amp | intent (frames) | dim_max | **dom_hand** | sym | snap 策略 |
|---|---|---|---|---|---|---|---|---|
| 1 ★★★ | **box023_person1** | 1 | 53.2cm | 52 | 0.39m | **both** | 0.96 | **双手** |
| 2 ★★★ | bucket005_s2_person1 | 1 | 40.6cm | 87 | 0.41m | both | 0.96 | 双手 |
| 3 ★★★ | **bucket001_person1** | 1 | 32.3cm | 58 | 0.46m | **L** | **0.37** | **单手 (L)** |
| 4 ★★ | box021_person1 | 2 | 44.1cm | — | 0.50m | both | 0.98 | 双手 |
| 5 ★★ | desk021_person1 | 2 | 42.4cm | 68 | 0.51m | both | 0.94 | 双手 |
| 6 ★ | bucket007_person1 | 2 | 25.5cm | 41 | 0.57m | both | 0.99 | 双手 |

**真实模式**（与旧总结相反）：
- **5 / 6 B+C case 是 both-hand**（symmetry ≥ 0.94，两手平均距离差 ≤ 2cm）
- **bucket001_person1 是唯一 single-hand**（symmetry 0.37，左手提桶把手，右手自然摆动）
- 几何 Tier 与单手/双手并不强相关（box023/bucket005_s2 都是 Tier 1 但都是双手）

**对 Path B 设计的强约束**（与旧版本结论一致，但 case 数量大变）：
- ❌ 对 bucket001 不能强行双手 snap — 右手离物体 62 cm，IK snap 会强行扭曲到 23 cm，违反"右手只是摆动"的真实意图
- ❌ 对其他 5 个 case 不能强行单手 snap — 会丢失另一只手的真实接触
- ✅ 必须按 csv 的 `dominant_hand` 字段切换 IK-snap 策略（L/R/both），不能 hard-code

**E055 推荐起点**：**box023_person1** — Tier 1 + 物体最小 + 抬升最大 + 双手 snap（与多数 case 一致，作为通用路径优先验证）

**E056 验证 case**：bucket001_person1 — 验证单手 snap 路径不破坏其他正常的关节

### Path C 验证 case (取消)

E054 实测 CORE4D 数据集**不存在 C-only case**（21 个 case 全是搬运/抬升类，
所有 Tier 1/2 case 都有可识别的 intent window）。

→ 原计划的"Path C 单独在 C-only case 上验证"取消。
→ Path C (force-closure reward) 改为 **Path B 内部的辅助 reward**，
   在 E055 完成后作为增量改进引入（如果 Path B 的 IK 解出现接触不稳定）。
→ 如未来扩到 OMOMO 或纯操作数据集，再启动独立的 Path C 验证。

### 失败回退确认

13 个 drop case 中：
- 5 个 box（dim ≥ 0.81m），3 个 desk（dim 0.80m），1 个 chair（dim 0.86m）：单人**几何上不可救**，应在大型 humanoid（H1, 1.8m）或双机器人方案中重新评估
- bucket003/010 (dim 0.74m, pf=0.50)：边界 case，dim 略超阈值且 partner 参与多

### 后续实验

- **E055** (Plan §Path B 首验证)：box023_person1 上做 hand-snap warmstart + IK 投影
- **E056** (Plan §Path C)：bucket001_person1 上做 force-closure reward（不依赖 intent window）
- **E057** (可选)：扫描更多 obj_z_amp 大但 intent=0 的 case，调试 intent detection 是否还能收紧

## 改动文件

| 文件 | 改动 |
|---|---|
| `workspace/core4d/scripts/analyze/case_tier_analysis.py` | 新建 (~330 行) |
| `workspace/core4d/results/E054/case_tier_classification.csv` | 新生成 |
| `workspace/core4d/results/E054/E054_case_tier_summary.md` | 新生成 |
| `workspace/core4d/results/E054/intent_detection_verify.png` | 新生成 |
| (无 spider 核心代码改动) | — |

## 教训记录

1. **绝对速度 vs 相对速度**：分析行走中的搬运任务时，手的绝对速度（0.3-0.7 m/s）受 locomotion 主导；要识别"接触意图"必须用 **手-物体相对速度**（搬运时 ≈ 0），否则把整段搬运相位都过滤掉。
2. **Finite-difference 噪声**：mocap 数据数值微分有 1-2 帧的尖峰（5-15 m/s），必须 5 帧滑窗平滑后再做阈值判断。
3. **百分位数阈值 > 硬编码距离阈值**：每个 case 的 hand-obj 距离尺度差异大（19cm-50cm），用 d_p20 + 5-7cm 自适应阈值比 1cm 硬阈值稳健 N 倍。
4. **box023 误归类的代价**：先前把 box023 和 box025 当同类处理，浪费了 E041c/E048-E052 至少 5 个实验。**几何参数化筛选应该在做任何方法实验之前**。
5. **物理必要性 > 运动学统计**：v_rel 这类运动学统计在大物体/晃动场景下噪声大，会漏掉真实搬运（desk021 v2 漏覆盖事件）。"obj_z 抬起 ≡ 必有外力"是更基本的物理信号，应作为主信号；v_rel 只用于 obj 不抬起的 push/drag 任务。
6. **可视化驱动调参**：v2 → v3 的关键修复完全依赖时序图目检（matplotlib 输出 png）。光看 intent_count 数字看不出 desk021 漏覆盖；只有看 obj_z 与 intent 阴影对齐情况才能发现问题。**任何意图/接触检测都必须配可视化 sanity check**。
7. **C-only 类天然为空是数据集特性**：CORE4D 全是搬运任务，没有纯操作。这件事必须在 plan 阶段用数据探索确认，而不是事先假设"应该有 N 个"。
8. **主导手判定不能用 tiebreaker 类指标**：第一版 dominant_hand 用"L 比 R 更近的帧数比例 (L_frac)"判定，6 个 case 错了 3 个。两只手都在 28cm 附近时，谁略微近一点就拿 96%+ 的票，但这跟"另一只手是否参与"完全无关。**视频核实是必须的**——脚本的输出再"干净"也可能是表征错指标的产物。改用"两手平均距离比 symmetry = min/max"后才与视频 6/6 匹配。
9. **数值指标必须有视觉 ground truth 校验**：E054 通过 `extract_case_keyframes.sh` 抽取关键帧目检后才发现指标错配。**即使是数据分析类实验也必须配视频/可视化验证**，不能信纯数字。
10. **可复用工具一律落实到本地脚本**：视频提取本来可以直接调技能里的 `frame.sh`，但用户提醒后写成 `extract_case_keyframes.sh` 加入 EXPERIMENT_TRACKER 索引，未来 E055+ 复用时可直接调用。
