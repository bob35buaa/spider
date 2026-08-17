# log289 · E200 · noPRG(E167A) arm 结果整理 + noPRG vs PRG 对比（E201 14-gate 宽窄口径）

_Core4D · Phase 63 · Run **R288**（noPRG arm）· 承接 [plan230](../plan/230_E200_augmentation_prg_g1a2_and_noprg_arms_plan.md) · 口径 = [E201 funnel_config 14-gate 宽/窄](../plan/231_E201_data_filter_funnel_plan.md) · 2026-08-18_

## Context / 目标

E200 把 E199 已建的 **249 条平移增强**（trans0/1/2，83 box case，arm-independent trajectory）复用到两个新下游 CEM arm。**noPRG(E167A) arm 已 249/249 全部完成**（另一台机器跑完后挂机，零损失）；PRG+G1+A2 arm 仍在本机跑（见 progress）。

本 log 交付用户所需：**整理 noPRG 结果 + 与 PRG（E199 aug）做同口径对比 + 出 xlsx**。对比采用**新口径 E201 14-gate 宽/窄漏斗**（此前 log 内部草算用的 12-gate 已作废）。

## 方法（口径一致性）

- **同一打分函数**：noPRG 249 rollout 用 `eval_E199_augmentation.score()` + `EvalConfig()` 打分（`eval_E200_arm_augmentation.py`），与 PRG（E199 fullscale aug）**逐字段同 contract**（`METHOD=E167A_zOnlyBody`、rubber_hull 手、`metric_standard_id=core4d-e154-physics-contact-v1`）。两 arm 是**同一批 249 aug 输入**，1:1 配对 `(case_id, aug_variant)`。
- **分层口径 = E201 `funnel_config`（单一真源）**：14 门 = 4 硬门（fall / body_z≤0.20 / ankle_jerk<1000 / obj_speed<3，无宽窄）+ 10 带门（宽/窄双阈值）。三层：L1 弃（硬门或任一带门 fail 宽）/ L2 人审（过宽、≥1 fail 窄）/ L3 接受（过窄；family 仲裁分 auto/review）。
  - **窄口径 leg_pen≤0.20 / hand_pen≤0.32 刻意松于 E178 canonical（0.10/0.30）**；接触门宽=0.40 窄=0.50；tracking 宽=窄+1。
  - noPRG 用 `classify_funnel.py`（扩展 `EXPS` 加 `E200_noprg`，纯新增不动 E199）分层；PRG 直接复用 E199 funnel rollout。
- **noPRG orig 锚 = E190 38-case noPRG 导出**（`spider_method_id=E167A_zOnlyBody_noPRG`，rubber_hull，omnirt_v1 orig）——与 E200 noPRG trans **同 arm**，且与 PRG 侧复用 E198 A0（同为 v1 orig）**对称**。用同一个 `score()` 打分 37/38（box021_035_p1 因 contact_mask 帧数 88≠qpos 129 跳过，属 E190 已知 box021 口径差异，继承 E190 审计 note），加入 noPRG case_metrics（group=orig）→ 这 37 个 case 的 family 有 orig 锚，classify_funnel 的 L3 family 仲裁与 PRG 侧对等。
  - 仍有 caveat：noPRG 只 37 个 case 有 orig（E190 覆盖面），其余 46 个 aug case 的 family 无 orig（partial，只 3 trans）；PRG 侧 83 case 全有 orig。故 **L3_auto/review 划分在无-orig 的 46 个 case 上仍不完全对等**；最干净的对比仍是**逐条带门宽/窄通过率**（同 249 aug、同打分，完全对齐）。
  - box021 orig 口径差异（E168 CEM 超参 / 2 bridge case，见 E190 log267）→ box021 的 layer 结论单列参考。

## 结果：noPRG vs PRG（OVERALL，249 vs 249 aug）

### 漏斗分层
| Layer | noPRG | PRG | Δpp |
|---|---|---|---|
| L1 弃 | 150 (60.2%) | 137 (55.0%) | +5.2 |
| L2 人审 | 48 (19.3%) | 36 (14.5%) | +4.8 |
| L3_auto | 12 (4.8%) | 15 (6.0%) | −1.2 |
| L3_review | 39 (15.7%) | 61 (24.5%) | −8.8 |
| **L3 接受(auto+review)** | **51 (20.5%)** | **76 (30.5%)** | **−10.0** |

> layer 已用**含 orig 的完整 family** 仲裁（见方法段：noPRG 补入 E190 37 个同 arm orig 作 family 锚）→ 两 arm 的 aug layer 口径对等。补 orig 前 noPRG 是 L3_auto 18/L3_review 33（无 orig 锚，偏松）；补后 12/39（6 个 case 因其 orig 不达标，其 aug 由 auto 降 review）。L3 接受总数 51 不变。

### 整体接受率
| 口径 | noPRG | PRG | Δpp |
|---|---|---|---|
| hard_all_pass | 0.827 | 0.807 | +2.0 |
| **WIDE_all_pass（不进 L1）** | **0.398** | **0.450** | **−5.2** |
| **NARROW_all_pass** | **0.205** | **0.305** | **−10.0** |

### 逐门（挑关键；完整 14 门 × 宽/窄见 xlsx）
| 门 | noPRG | PRG | Δpp |
|---|---|---|---|
| **leg_pen 窄(≤0.20)** | **0.570** | **0.823** | **−25.3** |
| **leg_pen 宽(≤0.40)** | **0.859** | **0.972** | **−11.2** |
| eef_ori 窄(≤20) | 0.514 | 0.574 | −6.0 |
| root_ori 窄(≤20) | 0.811 | 0.859 | −4.8 |
| obj_pos 窄(≤20) | 0.936 | 0.916 | +2.0 |
| obj_ori 窄(≤10) | 0.956 | 0.932 | +2.4 |
| contact 窄(≥0.50) | 0.888 | 0.892 | −0.4 |
| hand_pen 窄(≤0.32) | 0.876 | 0.884 | −0.8 |
| body_z 硬(≤0.20) | 0.948 | 0.932 | +1.6 |

### 关键指标 mean（noPRG − PRG）
- **leg_pen: 0.196 vs 0.086（+0.110）** ← 唯一大差异
- obj_pos: 12.74 vs 13.51 cm（−0.77）；obj_ori: 5.82 vs 6.11°（−0.29）；contact: 0.720 vs 0.719；hand_pen: 0.172 vs 0.177；root_ori: 16.30 vs 13.87°（+2.43）；eef_ori: 24.95 vs 22.17°（+2.79）。

## 结论

1. **PRG 明显优于 noPRG**：NARROW 全通过 30.5% vs 20.5%（+10pp），L3 接受 30.5% vs 20.5%。作为下游 RL 数据源，PRG（及 PRG+G1+A2）产出更高比例的高质量 rollout。
2. **差距几乎完全来自腿穿透**：leg_pen 是唯一 Δ>10pp 的带门（窄 −25.3pp / 宽 −11.2pp，mean 0.196 vs 0.086）。正是 PRG 的 16 腿-物碰撞对 + leg-gate 的作用；noPRG（E167A）无此约束 → 腿-物穿透显著更高。
3. **其余 13 门两 arm 基本等价**（±6pp 内）：tracking / contact / release / hand_pen / 硬门全接近；noPRG 因约束更少，obj tracking 反而略好（obj_pos/obj_ori +2~2.4pp），但 root_ori/eef_ori 略差。
4. **下游取舍**：若 RL 对腿-物穿透敏感 → 选 PRG；若不敏感且看重 obj tracking / 省 leg-gate 计算 → noPRG 亦可用（但需接受 ~2.3× 腿穿透）。三 arm（+PRG+G1+A2）完整对比待 PRG+G1+A2 CEM 跑完补。

## 产物路径
| 类型 | 路径 |
|---|---|
| noPRG 打分（249，同 contract） | `results/E200/s6_downstream/eval/noprg/e200_noprg_case_metrics.tsv` + `summary.json` |
| noPRG funnel 分层（14-gate 宽窄+layer） | `results/E201/funnel/E200_noprg_funnel_rollout.tsv` |
| PRG funnel（复用 E199） | `results/E201/funnel/E199_funnel_rollout.tsv` |
| **对比 xlsx（交付）** | **`results/E200/s6_downstream/eval/E200_noprg_vs_prg_funnel.xlsx`**（detail: 每 rollout 两 arm 相邻 × 14 门值+宽/窄 PASS+layer；summary: OVERALL+逐物体 noPRG vs PRG） |
| 脚本 | `eval/runners/eval_E200_arm_augmentation.py`、`eval/reports/gen_E200_arm_funnel_compare_xlsx.py`；`experiments/E201/classify_funnel.py`（EXPS +E200_noprg，纯新增） |

## 逐物体（L3 接受率 noPRG / PRG，leg_pen 窄通过率）
| 物体 | noPRG L3接受 | PRG L3接受 | leg_pen窄 noPRG/PRG |
|---|---|---|---|
| box001 | 见 xlsx | — | — |
| box004 | 15/18 L1(83%) 最差 | — | — |
> 逐物体完整数字见 xlsx summary 各物体 block。box004 两 arm 都最差（L1≈83%，主因 tracking/contact，非 arm 差异）。

## 待办 / 下一步
- PRG+G1+A2 arm CEM 跑完（本机，预计明晨）→ 补**三 arm（noPRG / PRG / PRG+G1+A2）**同口径对比。
- **视觉复核（rule 9，本 log 暂缺）**：需对 noPRG 抽 L1（尤其 leg_pen 触发）+ L3 各 ≥2 case 渲染关键帧，与 PRG 同 case side-by-side，确认腿穿透差异肉眼可见、L3 接受项无致命 artifact。→ render QC 待补，补后回填本 log「实际观察」。
- git commit（E200 eval/对比脚本 + classify_funnel EXPS 扩展）。

## 口径变更说明（重要）
本 log 用 **E201 14-gate 宽/窄漏斗**（非早期 12-gate）。窄口径 leg_pen≤0.20 / hand_pen≤0.32 **刻意松于 E178 canonical（0.10/0.30）**，因下游 RL 对二者容忍度高（见 plan231 决策）。同口径对两 arm 一致应用 → 对比公平。
