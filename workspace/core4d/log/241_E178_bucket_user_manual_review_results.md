# E178 人工终审结果：bucket003 / bucket007

_Core4D Phase 41 · 2026-07-24 ·
[E178 主计划](../plan/194_E178_bucket_contact_aligned_top_segment_plan.md) ·
[Tracking gates 追加计划](../plan/196_E178_tracking_error_numeric_gates_plan.md) ·
[前置 Full 结果](239_E178_local_5090_hybrid_rebalance.md) ·
[12 门评测修订](240_E178_tracking_error_numeric_gates.md)_

## 0. 结论

用户完成了 E178 中 bucket003 全 9 条和 bucket007 全 14 条的完整时间序列
人工审查，共覆盖 `23/27`：

| 人工结论 | Quality label | 数量 |
|---|---|---:|
| 可以用 | `USE / CLEAN` | 8 |
| 勉强能用 | `USE / MINOR_ACCEPTABLE` | 4 |
| 不能用 | `DO_NOT_USE / UNUSABLE` | 11 |
| 未审查（bucket004） | `PENDING` | 4 |

在已审 23 条中，人工最终可用为 `12/23`，不能用为 `11/23`。当前 12 门
numeric gate 在同一子集上通过 `8/23`：

- 8 个 numeric PASS 全部被用户判为可用，自动 PASS precision=`100%`；
- 11 个人工不能用全部被 numeric gate 拒绝，DNU recall=`100%`；
- 4 个用户认可的 case 被 numeric gate 额外拒绝，人工 USE recall=`66.7%`；
- 二分类一致率为 `(8+11)/23=82.6%`。

因此，当前 12 门适合充当**高置信自动通过门**，但不应替代人工终审：
它没有在已审子集上误放明显坏例，却会保守淘汰一部分视觉上仍可用的结果。
由于 bucket004 四条尚未收到人工裁决，E178 全量最终状态仍是
`PENDING_USER_REVIEW`，不能写成 27/27 人工终审完成。

## 1. 审查范围与记录口径

### 1.1 用户输入解析

用户原始输入覆盖：

| Object | Full rows | 已审查 | 未审查 |
|---|---:|---:|---:|
| bucket003 | 9 | 9 | 0 |
| bucket004 | 4 | 0 | 4 |
| bucket007 | 14 | 14 | 0 |
| **Overall** | **27** | **23** | **4** |

用户消息末尾的
`bucket007_20231023_075_p1bucket007_20231003_1_021_p2` 按两个合法且均存在
于 E178 manifest/metrics 的 case ID 解析为：

1. `bucket007_20231023_075_p1`
2. `bucket007_20231003_1_021_p2`

集合校验结果：submitted=`23`、unique=`23`、missing from eval=`0`、
duplicate=`0`。未提交项精确等于 bucket004 的四条，不做推断性填充。

### 1.2 标签映射

沿用 E173 已落盘的人工审查语义：

| 用户表述 | `manual_use_decision` | `manual_quality_label` |
|---|---|---|
| 可以用 | `USE` | `CLEAN` |
| 勉强能用 | `USE` | `MINOR_ACCEPTABLE` |
| 不能用 | `DO_NOT_USE` | `UNUSABLE` |

`manual_failure_taxonomy` 保持空值，因为用户只给出了总体可用性裁决，没有给
逐 case 的具体视觉病因；日志不根据 numeric failure mode 反向伪造人工观察。

## 2. 人工审查逐 Case 结果

### 2.1 bucket003

| Case | 用户结论 | Authority label | 12门 numeric | Numeric failure modes |
|---|---|---|---:|---|
| `bucket003_20231018_001_p1` | 可以用 | `USE / CLEAN` | PASS | — |
| `bucket003_20231018_001_p2` | 勉强能用 | `USE / MINOR_ACCEPTABLE` | PASS | — |
| `bucket003_20231018_003_p1` | 不能用 | `DO_NOT_USE / UNUSABLE` | FAIL | root_pos, root_ori, hand_pos, hand_ori |
| `bucket003_20231018_003_p2` | 勉强能用 | `USE / MINOR_ACCEPTABLE` | PASS | — |
| `bucket003_20231018_005_p1` | 不能用 | `DO_NOT_USE / UNUSABLE` | FAIL | root_pos, root_ori, hand_ori |
| `bucket003_20231018_005_p2` | 可以用 | `USE / CLEAN` | FAIL | release |
| `bucket003_20231020_064_p1` | 不能用 | `DO_NOT_USE / UNUSABLE` | FAIL | hand_penetration, root_pos, root_ori, hand_pos, hand_ori |
| `bucket003_20231020_068_p1` | 勉强能用 | `USE / MINOR_ACCEPTABLE` | FAIL | hand_penetration |
| `bucket003_20231020_068_p2` | 不能用 | `DO_NOT_USE / UNUSABLE` | FAIL | fall, body_z, lower_body, root_pos, root_ori, hand_pos, hand_ori |

bucket003 汇总：

| 指标 | 数量 |
|---|---:|
| 人工 CLEAN | 2 |
| 人工 MINOR_ACCEPTABLE | 3 |
| 人工 UNUSABLE | 4 |
| 人工 USE | `5/9` |
| 12门 numeric PASS | `3/9` |
| Numeric PASS 且人工 USE | `3/3` |
| 人工 USE 但 numeric FAIL | 2 |

### 2.2 bucket007

| Case | 用户结论 | Authority label | 12门 numeric | Numeric failure modes |
|---|---|---|---:|---|
| `bucket007_20231003_1_021_p1` | 不能用 | `DO_NOT_USE / UNUSABLE` | FAIL | hand_penetration |
| `bucket007_20231003_1_021_p2` | 可以用 | `USE / CLEAN` | PASS | — |
| `bucket007_20231003_2_021_p1` | 可以用 | `USE / CLEAN` | PASS | — |
| `bucket007_20231003_2_021_p2` | 可以用 | `USE / CLEAN` | FAIL | hand_penetration, hand_ori |
| `bucket007_20231003_2_023_p1` | 不能用 | `DO_NOT_USE / UNUSABLE` | FAIL | contact, release |
| `bucket007_20231018_019_p1` | 不能用 | `DO_NOT_USE / UNUSABLE` | FAIL | hand_penetration, lower_body, root_pos, hand_pos, hand_ori, object_ori |
| `bucket007_20231018_019_p2` | 不能用 | `DO_NOT_USE / UNUSABLE` | FAIL | contact, root_pos, hand_pos, hand_ori |
| `bucket007_20231018_021_p1` | 不能用 | `DO_NOT_USE / UNUSABLE` | FAIL | root_pos, hand_pos, hand_ori |
| `bucket007_20231018_021_p2` | 不能用 | `DO_NOT_USE / UNUSABLE` | FAIL | hand_penetration, lower_body, root_pos, root_ori, hand_pos, hand_ori |
| `bucket007_20231020_055_p1` | 不能用 | `DO_NOT_USE / UNUSABLE` | FAIL | root_pos, hand_pos, hand_ori |
| `bucket007_20231020_059_p1` | 勉强能用 | `USE / MINOR_ACCEPTABLE` | FAIL | hand_ori |
| `bucket007_20231023_073_p1` | 可以用 | `USE / CLEAN` | PASS | — |
| `bucket007_20231023_075_p1` | 可以用 | `USE / CLEAN` | PASS | — |
| `bucket007_20231023_075_p2` | 可以用 | `USE / CLEAN` | PASS | — |

bucket007 汇总：

| 指标 | 数量 |
|---|---:|
| 人工 CLEAN | 6 |
| 人工 MINOR_ACCEPTABLE | 1 |
| 人工 UNUSABLE | 7 |
| 人工 USE | `7/14` |
| 12门 numeric PASS | `5/14` |
| Numeric PASS 且人工 USE | `5/5` |
| 人工 USE 但 numeric FAIL | 2 |

### 2.3 bucket004 待审列表

以下四条没有收到用户裁决，annotation/XLSX 中均保持 `PENDING`：

- `bucket004_20231002_021_p1`
- `bucket004_20231002_021_p2`
- `bucket004_20231003_1_012_p1`
- `bucket004_20231003_1_012_p2`

## 3. 人工裁决与 12 门 Numeric Gate 的关系

### 3.1 交叉表

| 人工标签 | Numeric PASS | Numeric FAIL | 合计 |
|---|---:|---:|---:|
| `USE / CLEAN` | 6 | 2 | 8 |
| `USE / MINOR_ACCEPTABLE` | 2 | 2 | 4 |
| `DO_NOT_USE / UNUSABLE` | 0 | 11 | 11 |
| **合计** | **8** | **15** | **23** |

把 numeric PASS 视为自动推荐 USE：

| 指标 | 结果 |
|---|---:|
| 自动 PASS precision | `8/(8+0)=100%` |
| 人工 USE recall | `8/12=66.7%` |
| 人工 DNU recall | `11/11=100%` |
| 二分类一致率 | `19/23=82.6%` |

这个子集上没有出现“numeric PASS 但人工不能用”的假阳性。主要误差是四个
人工认可 case 被 numeric gate 保守淘汰。

### 3.2 四个人工救回 Case

| Case | 人工标签 | Numeric 失败门 | 实际值 | 阈值 |
|---|---|---|---:|---:|
| `bucket003_20231020_068_p1` | `MINOR_ACCEPTABLE` | hand_penetration | `0.4896` | `≤0.3` |
| `bucket003_20231018_005_p2` | `CLEAN` | release | `0.5517` | `≤0.3` |
| `bucket007_20231020_059_p1` | `MINOR_ACCEPTABLE` | hand_ori | `20.605°` | `≤20°` |
| `bucket007_20231003_2_021_p2` | `CLEAN` | hand_penetration, hand_ori | `0.3158`, `23.772°` | `≤0.3`, `≤20°` |

这里存在两类现象：

1. `bucket007_...059_p1` 的 hand orientation 只超阈值 `0.605°`，
   属于明显的边界型拒绝；
2. `bucket007_...2_021_p2` 的 penetration 仅超 `0.0158`，但 hand
   orientation 超 `3.772°`；另两条的 penetration/release 超阈值幅度较大，
   说明视觉可用性与单项物理统计并不总是等价。

因此不能简单通过全局放松全部阈值解决 false negative。更稳妥的消费方式是：

- 12 门全过：作为高置信自动 PASS；
- 单门或小幅越界：进入人工 rescue queue；
- 人工明确 `DO_NOT_USE`：保持最终否决权；
- 是否调阈值需积累更多物体、更多人工标签后再做 held-out calibration。

### 3.3 人工不能用 Case 的失败门分布

11 个 `UNUSABLE` case 的 numeric failure mode 计数为：

| Failure mode | Count / 11 |
|---|---:|
| root_pos | 9 |
| hand_ori | 9 |
| hand_pos | 8 |
| root_ori | 5 |
| hand_penetration | 4 |
| lower_body | 3 |
| contact | 2 |
| fall | 1 |
| body_z | 1 |
| release | 1 |
| object_ori | 1 |

root/hand tracking 门在人工坏例上具有最强覆盖，支持把新增 tracking gates
保留为自动筛选的重要组成部分。该计数是相关性证据，不单独证明视觉失败由
某一个 gate 对应的物理机制造成。

## 4. 可视化 → 实际观察

本轮人工标签直接来自用户对 E178 完整时间序列的审查，而不是 Codex 根据
中间帧代判：

- bucket003：4 条不能用、3 条勉强能用、2 条可以用；
- bucket007：7 条不能用、1 条勉强能用、6 条可以用；
- 两个物体合计：11 条不能用、4 条勉强能用、8 条可以用。

用户没有为每条不能用结果提供具体动作级病因，所以本日志只记录总体可用性；
numeric failure modes 单列展示，不冒充人工视觉观察。此前 log 239 已验证
27 个视频均可播放、ref/physics 双画面可读、物体与机器人拓扑正确。

可复现启动入口：

```bash
bash workspace/core4d/scripts/eval/wrappers/review_player.sh
```

## 5. 落盘产物

| 类型 | 路径 | 状态 |
|---|---|---|
| 人工 annotation authority | `workspace/core4d/results/E178/s6_downstream/eval/full/user_manual_review_filled.tsv` | 23 reviewed / 4 pending |
| 完整评测指标 | `workspace/core4d/results/E178/s6_downstream/eval/full/e178_case_metrics.tsv` | 27/27 |
| 评测 summary | `workspace/core4d/results/E178/s6_downstream/eval/full/summary.json` | 12门 numeric 10/27 |
| Review workbook | `workspace/core4d/results/E178/s6_downstream/eval/full/E178_buckets_prg_full_validation.xlsx` | 已同步重建 |
| Full 视频 | `workspace/core4d/results/E178/s6_downstream/render/full/` | 27/27 playable |

XLSX 同步结果：

```text
User reviewed            = 23
Manual USE               = 12
Manual DO_NOT_USE        = 11
Machine recommendation   = PENDING_USER_REVIEW
Final promotion decision = PENDING_USER_REVIEW
LibreOffice formulas     = 716
Formula errors           = 0
Workbook SHA256          = a379891febdca1def181fcba791baedf6729371c39888e0c75513300fb18d6c9
```

## 6. 执行与验证

本轮是 review-only 归档，没有重跑 CEM、训练或物理仿真，也没有修改 27 个
Full NPZ、scene、proxy、target 或视频。使用既有 canonical report generator
同步工作簿：

```bash
python3 workspace/core4d/scripts/eval/reports/gen_E178_bucket_prg_xlsx.py

python3 /home/ubuntu/.codex/skills/xlsx/scripts/recalc.py \
  workspace/core4d/results/E178/s6_downstream/eval/full/\
E178_buckets_prg_full_validation.xlsx 60

PYTHON_BIN=.venv/bin/python \
  bash workspace/core4d/scripts/eval/wrappers/review_player.sh --check
```

验证结果：

| 检查 | 结果 |
|---|---|
| Annotation row/ID uniqueness | ✅ 23 rows、23 unique、0 missing |
| Label/status/reviewer schema | ✅ |
| Pending set | ✅ 精确为 bucket004 四条 |
| Review player index | ✅ E178 reviewed `23/27`、playable `27/27` |
| XLSX cached-value assertions | ✅ |
| LibreOffice formula scan | ✅ 716 formulas、0 errors |
| XLSX ZIP integrity | ✅ |

## 7. Claims 验证

| Claim | 结果 |
|---|---|
| C1 用户清单无损落盘 | ✅ 23 个 case 全部一一对应，无重复、无漏项 |
| C2 标签语义一致 | ✅ 不能用/勉强/可以用映射遵循 E173 既有 contract |
| C3 未审项不猜测 | ✅ bucket004 四条保持 PENDING |
| C4 Consumer consistency | ✅ TSV、review player、XLSX 均为 reviewed 23 / USE 12 / DNU 11 |
| C5 Workbook validity | ✅ 716 formulas、0 errors |
| C6 Numeric-vs-human 分析可复核 | ✅ 23 条交叉表与四个 disagreement case 均来自正式 metrics |

## 8. 当前决策与下一步

当前状态：

```text
E178 Full artifacts        = COMPLETE (27/27)
E178 numeric evaluation    = COMPLETE (10/27)
User review bucket003/007  = COMPLETE (23/23)
User review bucket004      = PENDING (0/4)
Overall user review        = PARTIAL (23/27)
Final promotion decision   = PENDING_USER_REVIEW
```

下一步只需对 bucket004 四条补充 `可以用 / 勉强能用 / 不能用` 裁决。补齐后
可重新生成同一 XLSX，并给出 E178 27/27 的最终人工 USE 集合与 release 决策；
当前证据不要求重跑 CEM。
