# E178 结果修订：Tracking Error Numeric Gates

_Core4D Phase 41 · 2026-07-24 ·
[计划 196](../plan/196_E178_tracking_error_numeric_gates_plan.md)_

## 0. 结论

按用户指定阈值把 root、hand/EFF、object 的平均位置/姿态误差加入 E178
numeric release 判定后，E178 Full numeric pass 从原 6 物理门下的
`16/27` 收紧为 12 门下的 `10/27`：

| Object | 原 6 门 | 新 12 门 |
|---|---:|---:|
| bucket003 | `5/9` | `3/9` |
| bucket004 | `3/4` | `2/4` |
| bucket007 | `8/14` | `5/14` |
| **Overall** | **`16/27`** | **`10/27`** |

本次只重算评测派生产物，没有重跑 CEM，27 个 Full NPZ 与 27 个 MP4 均未改。

## 1. 新增 Gate Contract

六项为独立 gate，等于阈值时 PASS，缺失或非有限值时 FAIL：

| Failure mode | Metric | Threshold | Pass / 27 | Fail / 27 |
|---|---|---:|---:|---:|
| `root_pos` | root mean position error | `≤20cm` | 17 | 10 |
| `root_ori` | root mean orientation error | `≤20°` | 22 | 5 |
| `hand_pos` | EFF mean position error | `≤20cm` | 17 | 10 |
| `hand_ori` | EFF mean orientation error | `≤20°` | 14 | 13 |
| `object_pos` | object mean position error | `≤20cm` | 27 | 0 |
| `object_ori` | object mean orientation error | `≤10°` | 25 | 2 |

最终 `numeric_release_pass` 精确等于原 6 个物理门与这 6 个 tracking 门的
交集。27×12 gate cells 全部为明确 true/false，逐 row intersection
mismatch=`0`。

## 2. 被 Tracking Gates 额外淘汰的原 PASS

| Case | 新增失败门 |
|---|---|
| `bucket003_20231018_003_p1` | root_pos, root_ori, hand_pos, hand_ori |
| `bucket003_20231018_005_p1` | root_pos, root_ori, hand_ori |
| `bucket004_20231003_1_012_p1` | hand_pos, hand_ori, object_ori |
| `bucket007_20231018_021_p1` | root_pos, hand_pos, hand_ori |
| `bucket007_20231020_055_p1` | root_pos, hand_pos, hand_ori |
| `bucket007_20231020_059_p1` | hand_ori |

新增门中最强的单项瓶颈是 hand orientation：`13/27` FAIL。object position
为 `27/27 PASS`，当前阈值下没有额外筛除 case。

## 3. 实现与兼容性

- `eval_E176_lowgeom.py` 将 tracking gates 实现为 opt-in；E176 历史默认仍只
  使用原 6 门。
- `eval_E178_lowgeom.py` 强制启用并传入 `20,20,20,20,20,10`。
- summary 记录 `tracking_gates_enabled=true`、六项 threshold 和 pass count。
- review index/player 注册六个 `*_gate_pass` 列；E178 显示 12 门，旧实验
  忽略不存在的新列，仍显示原 gate 数。
- review player 跟踪误差红线同步为 root `20/20`、hand `20/20`、object
  `20/10`。

单元级验证覆盖：

- E176 legacy 不生成 tracking gate columns；
- 六项数值等于阈值时全部 PASS；
- `hand_ori=20.0001°` FAIL；
- 缺失 object position FAIL。

## 4. 重评完整性

```text
evaluated/expected = 27/27
paired            = 27/27
not_ready         = 0
errors            = 0
missing_baseline  = 0
summary status    = pass
```

正式结果：

```text
workspace/core4d/results/E178/s6_downstream/eval/full/
  e178_case_metrics.tsv
  e178_vs_e174_paired_deltas.tsv
  e178_group_summary.tsv
  summary.json
```

`gate_health_pass=0/27` 仍是 result NPZ 内部 PRG leg-gate health 的独立诊断，
不等于最终 numeric pass；最终 12 门 numeric pass 为 `10/27`。

## 5. Review Player 与 XLSX

review player headless check：

```text
E178 indexed/evaluated/numeric/playable = 27/27/10/27
```

XLSX 已在原路径重建：

```text
workspace/core4d/results/E178/s6_downstream/eval/full/
  E178_buckets_prg_full_validation.xlsx
```

Overview 已更新为 numeric=`10/27`、pass rate=`37.037%`、
bucket003/004/007=`3/9,2/4,5/14`，并新增六项 tracking failure counts。
LibreOffice recalc：formulas=`716`、errors=`0`；27 个视频与 27 个 keyframe
路径仍全部存在。

## 6. Claims

| Claim | 结果 |
|---|---|
| C1 threshold fidelity | ✅ summary 与 adapter 精确为 20/20、20/20、20/10 |
| C2 independent failures | ✅ 六个 failure modes 与六列一一对应 |
| C3 backward compatibility | ✅ E176 legacy truth-test PASS |
| C4 E178 completeness | ✅ 27/27，0 error/not-ready |
| C5 consumer consistency | ✅ TSV/summary/review/XLSX 均为 10/27 |
| C6 workbook validity | ✅ LibreOffice 716 formulas，0 error |

### 可视化 → 实际观察

本次没有修改轨迹或视频，因此沿用 log 239 已检查的 27-case midpoint
montage：renderer/replay 均可读，但存在人体/物体跟踪偏差。新增 tracking
gates 正是将这些误差从 display-only 指标升级为正式 numeric 约束；完整时间
序列人工审阅仍未完成。
