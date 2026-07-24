# E178 追加计划：Tracking Error Numeric Gates

_Core4D Phase 41 · 2026-07-24 · 追加到
[E178 主计划](194_E178_bucket_contact_aligned_top_segment_plan.md)_

## Context

E178 Full 已完成 27/27 并按原物理门得到 numeric pass 16/27。用户要求把
root、hand/EFF、object 的平均位置与姿态跟踪误差正式加入 numeric release
判定，而不再只作为 review player 的 display-only 指标。

## Gate Contract

六项均为独立 gate；任一项失败都进入 `numeric_failure_modes`：

| Failure mode | Metric | Unit | Pass threshold |
|---|---|---:|---:|
| `root_pos` | `track_root_pos_err_cm_mean` | cm | `≤20` |
| `root_ori` | `track_root_ori_err_deg_mean` | degree | `≤20` |
| `hand_pos` | `track_eef_pos_err_cm_mean` | cm | `≤20` |
| `hand_ori` | `track_eef_ori_err_deg_mean` | degree | `≤20` |
| `object_pos` | `track_obj_pos_err_cm_mean` | cm | `≤20` |
| `object_ori` | `track_obj_ori_err_deg_mean` | degree | `≤10` |

E178 的最终 `numeric_release_pass` 是原 6 个物理门与上述 6 个 tracking 门的
交集，共 12 gates。指标缺失或非有限值按 FAIL 处理。

## Claims

| Claim | 验证标准 |
|---|---|
| C1 threshold fidelity | summary 与逐 row gate columns 精确记录 20/20、20/20、20/10 |
| C2 independent failures | 六个 failure mode 与六个 `*_gate_pass` 列一一对应 |
| C3 backward compatibility | E176 不显式启用时仍使用历史 6-gate numeric contract |
| C4 E178 completeness | 27/27 evaluated、0 error、0 not-ready、27 paired |
| C5 consumer consistency | review player、summary、XLSX 对 numeric pass/fail 解释一致 |
| C6 workbook validity | LibreOffice recalc 0 formula errors，Overview 与 TSV 汇总一致 |

## Implementation

1. 参数化 `eval_E176_lowgeom.py` 的 tracking gate enable/thresholds；
2. `eval_E178_lowgeom.py` 强制启用六门和用户给定阈值；
3. review index 注册六个 gate columns，player 红线改为 20/20、20/20、20/10；
4. 重跑 E178 Full strict eval；
5. 更新 E178 workbook generator、重新生成并 recalc XLSX；
6. 写入新结果 log，更新 Tracker。

## Commands

```bash
bash workspace/core4d/scripts/eval/wrappers/eval_E178_lowgeom.sh \
  full --require-all

python3 workspace/core4d/scripts/eval/reports/gen_E178_bucket_prg_xlsx.py

python3 /home/ubuntu/.codex/skills/xlsx/scripts/recalc.py \
  workspace/core4d/results/E178/s6_downstream/eval/full/\
E178_buckets_prg_full_validation.xlsx 60
```

## Non-goals

- 不重跑 CEM，不修改 NPZ、proxy、reward 或 target route；
- 不追溯改写 E174/E176 的历史 metrics；
- 不把 tracking gate 失败解释成碰撞 proxy 的单一因果证据。
