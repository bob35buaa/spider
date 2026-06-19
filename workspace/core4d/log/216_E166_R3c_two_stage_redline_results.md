# E166-R3c two-stage redline 结果

日期：2026-06-18
前置：
- `log/214_E166_redline_predictive_results.md`
- `log/215_E166_R3b_redline_pathology_split_results.md`
产物：`workspace/core4d/results/E166/redline_predictive/`

## 结论

**R3c 不放行 GPU。**

two-stage classifier 在当前 7 个 downstream labeled case 上可以拟合到 `precision=1.0 / recall=1.0`，但 leave-one 稳健性只有 `3/7=0.429`。这说明阈值强烈依赖单个 case 数值，不能作为 E166 Phase1/2/3 的硬 gate。

因此仍不启动：
- 9 条 E166 CEM；
- 12 条 SUGAR RL；
- `run_E166_remote.sh`。

## Two-stage 拟合规则

拟合选出的规则：

1. Stage0：`pathology_self_collision == true` 直接 redline。
2. Stage1：对非 self-collision case，满足任一条件则 redline：
   - `obj_speed_max >= 4.353317102380466`
   - `trackbody_jerk_p95 >= 3772.954977620963`

拟合结果：

| precision | recall | F1 | TP | FP | FN |
|---:|---:|---:|---:|---:|---:|
| 1.000 | 1.000 | 1.000 | 4 | 0 | 0 |

Case decisions：

| case | actual_fail | pred_fail | trigger |
|---|---:|---:|---|
| box021_029_p2 | false | false | — |
| box021_035_p1 | false | false | — |
| box021_035_p2 | true | true | `obj_speed_max>=4.353` |
| box004_083_p2 | false | false | — |
| box004_082_p1 | true | true | `obj_speed_max>=4.353` |
| box004_083_p1 | true | true | `trackbody_jerk_p95>=3773` |
| box023_person2 | true | true | `stage0:self_collision` |

## 稳健性审计

Leave-one audit 结果：`3/7` correct。

失败项：
- hold out `box021_035_p1`：训练集会选择更低 `jerk>=2955.6`，误拦该中等成功 case。
- hold out `box021_035_p2`：训练集把 `obj_speed` 阈值推到 `5.6979`，漏掉该失败 case。
- hold out `box004_082_p1`：训练集把 `ankle_acc` 阈值推到 `151.3` 或更保守规则，漏掉该失败 case。
- hold out `box004_083_p1`：训练集改用 `obj_speed>=4.353`，漏掉极端 jerk 失败 case。

这符合并行审查子任务的风险判断：当前 n=7（Stage1 n=6）太小，阈值贴着单 case 数值，不能作为放行 GPU 的可泛化规则。

## 关键产物

新增/更新：
- `redline_two_stage_thresholds.json`
- `redline_two_stage_grid.tsv`
- `redline_two_stage_case_decisions.tsv`
- `redline_two_stage_leave_one.tsv`
- `redline_summary.md`

## 验证

```bash
python -m py_compile workspace/core4d/scripts/eval/runners/eval_E166_redline_predictive.py
git diff --check -- workspace/core4d/scripts/eval/runners/eval_E166_redline_predictive.py
bash workspace/core4d/scripts/eval/wrappers/eval_E166_redline.sh
```

## 下一步

R3c 的可执行下一步不是启动 E166 CEM/RL，而是补强验证集：

1. 完成缺失的 `box026_139_p1` downstream staggered 标签，当前本机 SUGAR outputs 只有 contact probe/data_qc，没有训练/eval。
2. 或增加至少一个新 labeled case，用于验证 `obj_speed/jerk/self-collision` two-stage rule。
3. 若仍要推进 A/B CEM 消融，需要把它明确降级为探索性实验，并修改 plan 177 的硬门/成功标准。
