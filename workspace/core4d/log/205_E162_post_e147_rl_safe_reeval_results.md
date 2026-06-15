# E162 — post-E147 RL-safe unified re-evaluation results

日期：2026-06-14

计划文件：`workspace/core4d/plan/171_E162_post_E147_rl_safe_reeval_plan.md`

## 1. 目标

E161 `surfaceBandReleaseDecay` 在 clean8 aggregate 下看似改善，但下游 RL 在 `box023_person2`
失败。E162 统一重评 E147 之后的所有可复用轨迹，baseline 固定为：

```text
E147 spider-rubberhand
```

本轮不重跑 CEM/RL，只用现有轨迹重新计算当前 shared metrics。

## 2. 口径修正

执行中核对 Holosoma 诊断后确认，用户指出的 `box023` 数值 `0.91 -> 0.75` 对应：

```text
hand_object_physics_contact_in_mask_frac
```

即真实 3cm contact mask 内是否存在物理接触。它不是 clean 3mm contact
`hand_object_physics_contact_3mm_in_mask_frac`。RL 对穿透不敏感，主要依赖接触 gate 是否存在，
因此 E162 的 hard gate 使用 raw in-mask physical contact：

```text
contact_inmask_delta_vs_E147 < -0.05 -> contact_regression_fail
```

clean 3mm/5mm contact 和 penetration 继续作为诊断列保留，不能抵消 raw contact 回退。

## 3. 实现与产物

新增/修改：

| 文件 | 内容 |
|---|---|
| `scripts/experiments/E162/build_post_e147_reeval_manifest.py` | 统一 E147/E148/E150/E151/E152/E153/E155/E156/E158/E159/E160/E161 manifest |
| `scripts/eval/runners/eval_E162_post_e147_rl_safe_reeval.py` | 统一重评、E147 delta、RL-safe gate、XLSX |
| `scripts/eval/wrappers/eval_E162_post_e147_rl_safe_reeval.sh` | 固定评测入口 |
| `scripts/eval/core/core_metrics.py` | 新增 Table4 风格 tracking fields |
| `scripts/eval/core/METRICS_STANDARD.md` | 记录 raw in-mask RL gate 与 Table4 fields |

运行命令：

```bash
python3 -m py_compile workspace/core4d/scripts/eval/core/core_metrics.py \
  workspace/core4d/scripts/experiments/E162/build_post_e147_reeval_manifest.py \
  workspace/core4d/scripts/eval/runners/eval_E162_post_e147_rl_safe_reeval.py
bash -n workspace/core4d/scripts/eval/wrappers/eval_E162_post_e147_rl_safe_reeval.sh
bash workspace/core4d/scripts/eval/wrappers/eval_E162_post_e147_rl_safe_reeval.sh full
```

输出：

```text
workspace/core4d/results/E162/post_e147_rl_safe_reeval/eval/full/
```

关键文件：

| 文件 | 内容 |
|---|---|
| `e162_method_metrics.tsv` | 181 rows per-case metrics |
| `e162_delta_vs_E147.tsv` | 每 row 相对 E147 baseline delta |
| `e162_case_failures.tsv` | hard gate / tracking / fall / baseline missing 失败列表 |
| `e162_method_summary.tsv` | method-level RL-safe summary |
| `e162_baseline_missing.tsv` | E147 baseline 缺失或 baseline mask 缺失 rows |
| `E162_post_E147_RL_safe_reeval.xlsx` | `method_summary/per_case_metrics/delta_vs_E147/case_failures/tracking_table4/baseline_missing` |

完整性：

```text
manifest_rows=181
metric_rows=181
missing_rows=0
baseline_missing_rows=91
case_failure_rows=107
```

`baseline_missing_rows=91` 的含义：这些 rows 没有 E147 同 case baseline，或 E147 baseline 缺 contact mask；
它们只报告绝对指标，不判 RL-safe pass。

## 4. 关键结果

最终 case status：

| status | rows |
|---|---:|
| pass | 74 |
| contact_regression_fail | 14 |
| fall_fail | 2 |
| needs_baseline_or_manual_review | 91 |

### 4.1 E161 结论修订

E161 aggregate 的 “inmaskC3/penetration 更好” 结论不能作为 RL-safe 结论。按 raw in-mask contact gate：

| method | case | E147 raw contact | run raw contact | delta | clean3mm delta | status |
|---|---|---:|---:|---:|---:|---|
| M0 postureRerankA | `box023_person2` | 0.9077 | 0.6308 | -0.2769 | +0.3538 | fail |
| M1 releaseDecay | `box023_person2` | 0.9077 | 0.7538 | -0.1538 | +0.4769 | fail |
| M2 strictMask | `box023_person2` | 0.9077 | 0.8000 | -0.1077 | +0.3385 | fail |
| M1 releaseDecay | `box004_082_p1` | 0.6721 | 0.6066 | -0.0656 | +0.2459 | fail |

这正好解释 downstream RL failure：clean 3mm contact/penetration 看起来改善，但 raw physical contact
gate 在高接触 case 上下降超过 0.05。

### 4.2 其他 contact regression

被 hard gate 标红的 rows：

| method | case | raw contact delta |
|---|---|---:|
| `E150_off08` | `box021_035_p2` | -0.0874 |
| `E150_off11` | `box023_person2` | -0.0615 |
| `E150_off11` | `box004_082_p1` | -0.0656 |
| `E151_b2_sup` | `box023_person2` | -0.0615 |
| `E152_gateA` | `box023_person2` | -0.1077 |
| `E158_gateA_surfaceBandA` | `box023_person2` | -0.3077 |
| `E159_gateA_surfaceBandA2` | `box023_person2` | -0.2308 |
| `E160_gateA_surfaceBandA2_postureRerankA` | `box023_person2` | -0.2769 |
| `E161_*` | `box023_person2` / `box004_082_p1` | see table above |

`E156_gateA` passes the raw contact gate on E147-covered cases, despite failing under the earlier clean-3mm-only
interpretation; this is expected because it preserves raw contact while reducing deep/dirty contact.

## 5. Table4 tracking

新增 fields 均已写入 `e162_method_metrics.tsv` 和 XLSX `Tracking指标` sheet：

```text
track_joint_err_deg_mean
track_eef_pos_err_cm_mean
track_eef_ori_err_deg_mean
track_root_pos_err_cm_mean
track_root_ori_err_deg_mean
track_obj_pos_err_cm_mean
track_obj_ori_err_deg_mean
```

181/181 rows 这些字段均非 NaN。GT 为固定 SPIDER 输入 `trajectory_kinematic.npz`。

## 6. Claims

| Claim | 结果 |
|---|---|
| C1: E147 作为统一 baseline | 部分成立；E147-covered rows 可比较，91 rows 缺 E147 baseline/mask 单独列出 |
| C2: contact regression 不再被 aggregate 掩盖 | pass；14 rows 进入 `case_failures` |
| C3: E161 box023 regression 被抓到 | pass；releaseDecay `box023_person2` raw contact delta = -0.1538 |
| C4: Table4 tracking 可用 | pass；7 个 Table4 fields 181/181 非 NaN |
| C5: 不重跑 CEM | pass；仅重放现有 trajectory |

## 7. 结论

E161 不能作为 RL-safe 默认候选。它降低了 release false / clean contact penetration，但在 `box023_person2`
和 `box004_082_p1` 这类 E147 高 raw-contact case 上触发 hard contact regression。

后续方法选择应先满足 raw in-mask contact 的 per-case do-no-harm gate，再比较 clean 3mm contact、
penetration、release false-contact 和 Table4 tracking。对于缺 E147 baseline 的 clean8 rows，
需要补 E147-style rubberhand baseline 或单独人工审查，不能直接标 RL-safe pass。

## 8. XLSX 可读性更新

按用户反馈重写 `E162_post_E147_RL_safe_reeval.xlsx` 的可见 sheet：

| sheet | 内容 |
|---|---|
| `主表` | 方法级摘要；只保留 RL-safe、几何接近/穿透、物理接触/穿透、Table4 tracking |
| `逐case` | case 级状态、raw 接触 gate、clean3 诊断、核心几何/物理/tracking |
| `Tracking指标` | 对齐 SPIDER Table 4 命名的 tracking 明细 |
| `说明` | E147/E148 区别、raw contact hard gate、clean3 contact 含义、baseline missing 规则 |

原始 `method_summary_full/per_case_full/delta_full/failures_full/baseline_missing_full`
仍保留在 workbook 中，但设为 hidden，避免默认打开时被 100+ 内部列干扰。

E147/E148 说明已写入 `说明` sheet：E147 是本轮统一 baseline（原始 10-case
`spider-rubberhand`）；E148 是 E143 24-case rubberhand 扩展，重叠 case 复用 E147，新 case
由 E148 跑出，E148 本身不是本轮 baseline。

## 9. 分层交集报表

用户进一步澄清：E147/E148 算法层面是同一个 `SPIDER+rubberhand` 方法，不应在方法表里拆成两行；
缺 baseline 的 row 也不应进入主比较表。已保留原始大表不动，新增两个分层交集报表：

| 文件 | 交集 | 方法层 |
|---|---|---|
| `E162_layer1_foundation_intersection.xlsx` | 3 case: `box004_083_p2,box021_029_p2,box023_person2` | OmniRetarget、SPIDER+rubberhand、+gateA、+b1、+gateA+b1、E153 gateA+b1 threshold sweep |
| `E162_layer2_post_E156_intersection.xlsx` | clean6: `box004_083_p1,box004_083_p2,box021_029_p2,box021_035_p1,box021_035_p2,box023_person2` | E156 之后 surface/posture/release 系列 |

这两个表的 `主表/逐case` 中所有 delta/status 都相对同 case 的 `SPIDER+rubberhand`，不再相对 E147。
E147/E148/E156 只作为数据来源记录在 `source/method_id` 和 `交集审计` 中。

为让 OmniRetarget 也有 Table4 tracking 字段，重跑了轻量评测：

```bash
python3 workspace/core4d/scripts/eval/runners/eval_E154_omniretarget_reference.py
bash workspace/core4d/scripts/eval/wrappers/eval_E156_clean8_gate_decay.sh full
python3 workspace/core4d/scripts/eval/reports/gen_E162_layered_intersection_tables.py
```

验证：两个新 XLSX 均无公式错误样式单元格；原 `E162_post_E147_RL_safe_reeval.xlsx` 未重写。
