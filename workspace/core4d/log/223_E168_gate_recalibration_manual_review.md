# E168 门槛重标定与 Box021 人工核验

日期：2026-07-17

状态：评测与工作簿已刷新；CEM production 仍在运行，本记录是 `29/40` 可评测产物快照，不是最终 freeze。

## 用户决策

E168 成功判定改用以下门槛：

| gate | 主指标 | 新门槛 |
|---|---|---:|
| z-only | `body_z_err_p95_m` | `<=0.20m` |
| lower body | `leg_penetration_frac` | `<=0.10` |
| physics penetration | `hand_object_physics_penetration_3mm_frame_frac` | `<=0.30` |
| raw contact | `hand_object_physics_contact_in_mask_frac` | `>=0.50` |

`body_z_err_peak_m` 继续保留为诊断指标，不再决定 z gate。release false-contact 门槛不变，仍为 `<=0.30`。

## 人工核验

用户逐一核验旧工作簿中的 16 条 Box021：

| 人工决定 | 质量标签 | 数量 | 含义 |
|---|---|---:|---|
| `USE` | `NO_ISSUE` | 5 | 没有明显问题，可用 |
| `USE` | `MINOR_ACCEPTABLE` | 3 | 有小瑕疵但可接受，可用 |
| `DO_NOT_USE` | `MAJOR_ISSUE` | 8 | 存在大问题，不能用 |

权威人工清单：

```text
workspace/core4d/results/E168/s6_downstream/cem/eval/manual_review/e168_user_visual_review.tsv
```

人工结论与 numeric gates 正交保存，不用数值失败覆盖用户的 `USE`。8 条人工可用 case 中，5 条同时 numeric pass；以下 3 条保留 `USE`，同时保留数值告警：

| case | 人工标签 | numeric 告警 | 指标值 |
|---|---|---|---|
| `box021_20231011_034_p1` | `MINOR_ACCEPTABLE` | lower body | `0.266 > 0.10` |
| `box021_20231011_034_p2` | `MINOR_ACCEPTABLE` | penetration | `0.517 > 0.30` |
| `box021_20231011_038_p2` | `MINOR_ACCEPTABLE` | penetration + lower body | `0.462 > 0.30`；`0.521 > 0.10` |

## 最新评测快照

运行入口：

```bash
bash workspace/core4d/scripts/eval/wrappers/eval_E168_e167a_metrics.sh
```

当前 `29/40` rows 可评测，errors `0`：

| gate | pass |
|---|---:|
| all numeric | `8/29` |
| tracking | `28/29` |
| fixed-reference z p95 | `25/29` |
| raw contact | `23/29` |
| release | `28/28 applicable`，另 `1 N/A` |
| penetration | `26/29` |
| lower body | `11/29` |
| no fall | `28/29` |

人工状态为 reviewed `16`、USE `8`、DO_NOT_USE `8`、pending `13`。新增完成但未被用户核验的 rows 保持 `PENDING_MANUAL_REVIEW`。

## Excel 产物

工作簿：

```text
workspace/core4d/results/E168/s6_downstream/cem/eval/e167a_aligned_available/E168_E167A_aligned_available_case_metrics.xlsx
```

- 10 sheets，新增“人工核验”页。
- “门控总览”前置人工决定、人工质量、视觉通过、核验状态与备注。
- “完整指标”为 `29 rows x 192 columns`。
- 人工页公式读回为 reviewed `16`、USE `8`、DO_NOT_USE `8`。
- LibreOffice 重算 `876` 个公式，扫描结果 `0` formula errors。

## 结论

- 新门槛已进入 evaluator、plan、summary TSV/JSON/Markdown 和 xlsx。
- 16 条 Box021 人工结论已落盘，且不会被 numeric gates 覆盖。
- production 尚未完成；后续新增 case 需继续人工核验，`40/40` 后再冻结最终表。
