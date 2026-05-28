# 数据筛选逻辑升级与候选 box 推荐

日期：2026-05-28
关联诊断：`workspace/exp_diagnostic/diagnostic_report.md`
对应任务：`workspace/exp_diagnostic.md` Q2 (是否需要更新筛选逻辑 + 找尺寸介于 box023 与 box025 之间的箱子)

## TL;DR

1. **现有 holosoma D001-D008 pipeline 缺少"G1 retarget 几何可行性"门**：D004/D005 只看 human ↔ box 几何，从不检查 OmniRetarget 把人映射到 G1 之后双手是否仍合理。这就是 D003 box021 13 case 全部 D005 pass 但 SPIDER 0/13 成功的根因。
2. **新增 G1-feasibility gate（命名建议 D005b）** 仅基于已有 `trajectory_kinematic.npz`，无需重新训练，5 个硬阈值：见 §2。在 12 个已知 case 上 calibrate：所有 E028/E082-E088 box021 D003 失败 case 都被拒绝；box023_p2、box025_p2 通过；并意外发现 **`box021_person1`（老 OmniRetarget，非 D003）反而通过**——是当前已有数据里唯一一条 "中等尺寸 + G1 可行 + 未跑过 SPIDER 动力学" 的候选。
3. **CORE4D 内符合"大于 box023 + 小于 box025"的箱型**有 4 种（按可用 seq 数排序）：Box026 (52)、Box021 (50)、Box022 (8)、Box004 / box004 (20)；其中只有 Box021 跑过 D003 OmniRetarget。Box026 与 Box022 都没有 OmniRetarget 输出，需要先跑 holosoma stage2b 才能用新 gate 评估。
4. **下一步推荐顺序**：
   - **优先** 用 SPIDER 动力学跑 `box021_person1` 看是否真能通过——零额外重定向成本；
   - 在 holosoma 重做 D003 box021 OmniRetarget，强制双手投到 box 顶面 +5 cm（与新 gate 一致），重 evaluate 13 case；
   - 用 stage2b 给 Box026 (52 seq) 跑 OmniRetarget 后跑新 gate，预期能挑出几条干净的 medium-large box 候选；
   - Box022 数据量少（8 seq）但形状最特殊（窄长），可作 stress test。

## 1. 现有 pipeline 与缺失环节

`workspace/v3/data_construction` 的现状（holosoma）：

| Stage | 作用 | 包含的检查 | 缺失 |
|---|---|---|---|
| D001 | candidate inventory | object size band（按 box023/box025 体积比）、raw motion z 范围、frames、template availability | — |
| D002 | raw contact | 人-物 contact frac (2cm/3cm)、both-hands、longest run | 不看 G1 |
| D003 | omniretarget+spider production | 跑 OmniRetarget→SPIDER → 输出 trimmed npz、scene XML | 没有自动 gate，全部进 D004 |
| D004 | visual QC | sheet/preview/diagnostic 图 + raw_target contact rate + obj 速度 | 只看 raw target，**不看 G1 retarget** |
| D005 | handbox surface gate | hand-box face counts、edge/corner frac、top/bottom frac、L/R patch sep、torso/leg shortcut frac | **签字面 + shortcut 都是人体侧的，没有 G1 wrist 在 box 局部坐标的几何检查** |
| D006 | top case bank | aggregate score + risk penalty | — |
| D007/D008 | dynamic gate / smoke trainability | smoke 训练 1 epoch + RL 可训练性 | 这里出现失败才被发现，已经花了大量算力 |

**核心缺口**：D004/D005 把"人-物" 几何当成 surrogate for "G1-物" 几何。当 G1 (1.32 m, 短臂) 与 SMPL 人 (1.7 m, 长臂) 的 retarget scale 在某个 case 上偏差较大时，**人体侧通过了，但 G1 双手实际落在 box 内部 / 远低于 pelvis**，这种 case 在 D005 是 review/pass 的，进到 D008/动力学才崩。E028→E082-E088 是该缺口直接产物。

## 2. 新增 G1-Feasibility Gate（D005b 草案）

只读已有 SPIDER `trajectory_kinematic.npz` + `scene.xml`，无新依赖、无 GPU、O(T) 复杂度。脚本：`workspace/exp_diagnostic/scripts/g1_feasibility_gate.py`。

### 计算（每 case，per hand）

针对 G1 IK retarget wrist（左右），加 wrist+5cm 前向 eef_offset，做 `mj_forward(qpos_ref[t])`，把 wrist 世界点投到 object local frame：

| 指标 | 阈值 | 物理含义 |
|---|---|---|
| `ik_wrist_inside_box_frac` | ≤ 10% | wrist + eef_offset 落在 box 体内的帧比；> 10% 表示 IK 强制嵌入物体，CEM 不可能既 contact 又 no-pen |
| `ik_wrist_signed_dist_to_face` mean | ≥ 0.03 m | 沿所选 face 法线方向 wrist 离 box 外有多少 cm；过紧说明只能贴皮接触 |
| `wrist_below_pelvis_gap` mean | ≤ 0.30 m | pelvis_z 减去 wrist_world_z；过大表示 G1 必须深度前倾去够 wrist 目标 |
| `pelvis_z_min` (per case) | ≥ 0.60 m | ref 强制 G1 蹲到 <60cm 时 CEM 几乎一定不稳 |
| `trajectory_T` | ≥ 80 帧 (≈ 2.7 s @ 30Hz) | 太短给 CEM 太少 context |
| `top_face_frac_either_hand` (≥ 1 手) | ≥ 20% | 至少一只手 ≥20% 帧落在 box +z 面（上沿/顶面）；CORE4D 搬箱主流姿态是"双手抓上沿"，没有 top-face 接触的 case 几乎都是 side-press / hug / 错face |

### Calibration（12 个已知 case）

| Case | 历史结论 | 新 gate | 拒绝原因 |
|---|---|---|---|
| `d003_box021_20231018_029_p2` | E082-E088 FAIL | **REJECT** | R wrist 33% 在 box 内 + T=75<80 |
| `d003_box021_20231011_035_p2` | E082-E088 FAIL | **REJECT** | pelvis 跌到 0.58m + 无 top-face hand |
| `d003_box021_20231020_019_p1` | E082-E088 FAIL | **REJECT** | 无 top-face hand |
| `box023_person2` | E082 guard PASS | **PASS** | top_face 51% |
| `box025_person2` | E080 partial | **PASS** | top_face 100% (边缘 signed dist 0.034m) |
| `box025_person1` | E080 false positive | **REJECT** | L 23% / R 39% 在 box 内 |
| `box025_s2_person1` | 未测 | **REJECT** | L 41% / R 69% 在 box 内 |
| `box024_person1` | 未测 | **REJECT** | L 28% / R 57% 在 box 内 + 无 top |
| `box023_person1` | E054 标 Tier1 但其实从未跑通 | **REJECT** | 无 top-face hand |
| `box021_person2` (老) | 未测 | **REJECT** | T=75 + 无 top-face hand |
| `box001_person1` (老) | 未测 | **REJECT** | wrist 16% 在 box 内 |
| **`box021_person1` (老)** | E054 标 "press not lift"，从未跑 SPIDER | **PASS** | top_face 70%，wrist 4.5% 在 box 内，pmin 0.72 |

新 gate 决策与已观察到的 SPIDER 失败 100% 一致，并把 box025_p2 这种边缘 partial pass 也通过。**关键意外发现**：`box021_person1`（5 kg 老 scene，pre-D003）几何上是 G1 可行的；可视化 `findings/04_overlay_box021_person1_PASS_NEW_GATE.png` 显示双手稳定聚在 +z 顶面，与 box023_p2 同模式。

### 集成到现有 pipeline

最小侵入式接入：

1. 把 `g1_feasibility_gate.py` 移到 `holosoma/workspace/v3/data_construction/scripts/check_g1_feasibility.py`；
2. D003 产出 `holosoma_{case}/spider_qpos_*.npz` 之后立刻调用，写 `d005b_g1_feasibility.tsv`；
3. 在 D006 scoring 里加入 `g1_feasibility_score_25`（pass = 25，每条 reject reason 扣 7 分），把已 SPIDER 端验证过的物理可达性纳入 top3 排序；
4. D007/D008 跑前必须 g1_feasibility_pass，从源头杜绝 box021 D003 这类无意义 dynamic-smoke 浪费。

## 3. CORE4D 内"大于 box023 + 小于 box025"的箱型清单

来源：`holosoma/workspace/v3/data_construction/results/d001_stage0_inventory_v2/candidate_inventory_v2.json`。

| Object | n_seq | 体积 m³ | extents (m) | vs box023 vol | vs box025 vol | D001 size_band | 已 OmniRetarget? |
|---|---:|---:|---|---:|---:|---|---|
| Box023 (ref small) | 46 | 0.036 | 0.30 × 0.30 × 0.39 | 1.00× | 0.108× | small_above_bucket | yes (person2 PASS) |
| Box004 / box004 | 2 + 18 | 0.041 | 0.35 × 0.26 × 0.45 | 1.14× | 0.123× | target_medium | **no** |
| **Box021** | 50 | 0.059 | 0.39 × 0.30 × 0.50 | 1.64× | 0.178× | target_medium | yes (D003 13 case, all FAIL); 老 box021_person1 PASS gate |
| Box022 | 8 | 0.083 | 0.27 × 0.67 × 0.46 | 2.30× | 0.248× | target_medium | **no** |
| **Box026** | 52 | 0.116 | 0.63 × 0.39 × 0.47 | 3.23× | 0.349× | target_medium | **no** |
| Box024 / box024 | 46+ | 0.253 | 0.51 × 0.51 × 0.98 | 7.01× | 0.757× | near_box025_large_review | yes (person1 REJECT gate) |
| Box001 / box001 | 72+30 | 0.256 | 0.50 × 0.63 × 0.81 | 7.10× | 0.767× | near_box025_large_review | yes (person1 REJECT gate) |
| Box025 (ref large) | 42 | 0.333 | 0.61 × 0.61 × 0.89 | 9.25× | 1.00× | hard_reject_size | yes (person2 PASS) |

注：Box020 (0.013 m³) 比 box023 更小，已被 D001 hard_reject。

### 用户 "中等" 目标的子集

严格在 (box023, box025) 体积区间且属于 `target_medium` band：**Box004 / box004 / Box021 / Box022 / Box026**。

按 unique seq 数（粗略 = 可挑选 case 数）排序：
- **Box026: 52 seq** —— 最值得做的方向，体积约 box023 的 3.2 倍 / box025 的 35%。Extents 0.63×0.39×0.47 是 "厚饼干形"，整体接近一个标准搬运纸箱。
- **Box021: 50 seq** —— 当前有 D003 OmniRetarget 但全 fail；老 box021_person1 PASS。建议同时尝试 (a) 跑 SPIDER 给 box021_person1 (b) 重做 D003 box021 OmniRetarget 让双手投到顶面。
- **Box004: 20 seq** —— 0.35×0.26×0.45，比 box023 略大，可能是"扁长方"。
- **Box022: 8 seq** —— 0.27×0.67×0.46，"长条窄高"，最特殊，少量数据但 stress-test 几何泛化用。

## 4. 当前已有数据的 G1-Feasibility 评估结果（按可行性排序）

跑 `workspace/exp_diagnostic/scripts/g1_feasibility_gate.py` over 全部 12 个已 retarget 的 box case：

| 排名 | task | size band | 已知动力学结果 | 新 gate | 备注 |
|---:|---|---|---|---|---|
| 1 | `box023_person2` | small | E082 guard PASS | **PASS** | 已是 known-good baseline |
| 2 | `box021_person1` | **medium** | **从未跑** SPIDER 动力学 | **PASS** | top 70%, 老 scene；用户最值得马上试这一条 |
| 3 | `box025_person2` | large | E080 partial | **PASS** | 已 partial pass，可继续 |
| — | 9 条其他 | mixed | 0 通过 | REJECT | 含全部 D003 box021 |

中等尺寸 + gate PASS + 现成数据 = 仅 `box021_person1` 一条。

## 5. 推荐行动序列

### A 路（零额外重定向，最快验证）

1. 跑 SPIDER 动力学 on `box021_person1`（已有 trajectory_kinematic.npz + scene_act.xml），先 24-step smoke（~5 min）确认 gate 预测准确，然后 full CEM（~30 min）。
2. 如果 head/upper penetration 低于 box021 D003 失败 case 的 1/3，**新 gate 即被 SPIDER 端动力学验证**，作为推荐到 D008 的 medium box 候选。
3. 如失败，意味着 gate 的 5 个阈值是必要但不充分；进入 B 路。

### B 路（重做 D003 box021 OmniRetarget）

1. 在 holosoma OmniRetarget 加 constraint：双手 wrist target 在 box local frame 必须满足 `+z >= half_z + 5cm`（顶面外 +5cm），否则向上投影 + IK 限位求解。可在 `holosoma/.../d003_omniretarget` 阶段直接加。
2. 重新对 box021 50 seq 跑 D003，并立刻应用新 D005b gate；预期 13 case 中至少有 30-50% 能 pass（因为 raw mocap 本来人就在上沿附近）。
3. 取 top3 进 SPIDER smoke。

### C 路（开新箱型）

1. 给 Box026 (52 seq)、Box022 (8 seq)、Box004 (20 seq) 跑 holosoma stage2b OmniRetarget；
2. 直接接 D005b G1-feasibility gate，按 pass + 体积排序；
3. Box026 是首选——seq 数最多 + 形状接近"普通纸箱"。

### 三路对比

| 路 | 算力 | 期望产出 | 风险 |
|---|---|---|---|
| A | 极小 (~30 min CEM) | 第 1 条可用 medium-box case | gate 不充分 → 没有新发现 |
| B | 中 (重跑 D003，~小时级) | 多达 5-10 条 medium-box case，且与 box023/025 同一 OmniRetarget 版本 | 改 OmniRetarget 约束可能影响 raw contact 质量 |
| C | 大 (开新箱，重新 holosoma stage2b + stage3 + 全部 D 系列) | Box026 这种 "标准搬运纸箱" 的全新 candidate 池 | 投入大但能根治"中等箱只有 Box021"的尴尬 |

推荐：**A + B 并行**（A 立刻就能开始；B 在 holosoma 端开线程；C 等 B 出第一批结果再决定是否要更高变样态）。

## 6. 产出清单

- `workspace/exp_diagnostic/scripts/g1_feasibility_gate.py` — 新 D005b gate 实现
- `workspace/exp_diagnostic/scripts/inventory_boxes.py` — box mesh inventory（spider 本地）
- `workspace/exp_diagnostic/findings/05_box_inventory.json` — spider 本地 box 尺寸
- `workspace/exp_diagnostic/findings/06_g1_feasibility_gate.json` — calibration 完整数据
- `workspace/exp_diagnostic/findings/06_g1_feasibility_gate.txt` — calibration 控制台打印
- `workspace/exp_diagnostic/findings/07_gate_all_existing_boxes.txt` — 12 case sweep
- `workspace/exp_diagnostic/findings/04_overlay_box021_person1_PASS_NEW_GATE.png` — 候选 1 视觉证据
- `workspace/exp_diagnostic/data_filter_recommendation.md` — 本文档
