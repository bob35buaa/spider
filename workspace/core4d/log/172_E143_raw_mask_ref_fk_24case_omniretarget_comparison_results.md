# E143 raw_mask_ref_fk 24-case OmniRetarget 对比结果

## 目标

按用户纠正后的目标，只评估 SPIDER/CEM 重定向的手物接触是否能超过 OmniRetarget；本轮不进入 Holosoma/RL，不启动 PPO，不触碰 Holosoma 项目。

E143 作为新的 24-case 实验，使用 `raw_mask_ref_fk` 方法补齐 E109 24-case workset 中未跑过的 case，并与 `OmniRetarget`、`ref_fk` 在同 case、同指标口径下比较。

## 运行与产物

| item | path / value |
|---|---|
| plan | `workspace/core4d/plan/152_E143_raw_mask_ref_fk_24case_omniretarget_comparison_plan.md` |
| manifest | `workspace/core4d/scripts/E143/variants.tsv` |
| preflight | `workspace/core4d/results/E143/preflight/raw_mask_ref_fk_24case_preflight.tsv` |
| train runner | `workspace/core4d/scripts/train/train_E143_raw_mask_ref_fk_24case.sh` |
| remote runner | `workspace/core4d/scripts/run_E143_remote.sh` |
| remote pull | `workspace/core4d/scripts/pull_E143_remote_results.sh` |
| evaluator | `workspace/core4d/scripts/eval/eval_E143_raw_mask_ref_fk_24case.py` |
| xlsx | `workspace/core4d/results/E143/raw_mask_ref_fk_24case_omniretarget_comparison/E143_raw_mask_ref_fk_24case_omniretarget_comparison.xlsx` |

Manifest 共 24 行：21 个 `to_run`，3 个 `already_done` 复用 E112 `raw_mask_ref_fk`：

- `box004_082_p1`
- `box021_035_p1`
- `box026_135_p1`

本地 `local-gpu0` 完成 7/7，远端 `spider-remote` 完成 14/14；回收后 E143 root NPZ/MP4 均为 21 个。最终 evaluator 成功 join 24/24 raw rows，`missing_rows=0`。

## 方法平均

| method | cases | 手物接触 | 5cm | 10cm | 手物穿透 | 腿穿透 |
|---|---:|---:|---:|---:|---:|---:|
| OmniRetarget | 24 | 54.4% | 64.6% | 66.4% | 54.7% | 11.3% |
| ref_fk | 24 | 43.1% | 61.8% | 64.7% | 43.5% | 10.0% |
| raw_mask_ref_fk | 24 | 33.0% | 55.5% | 61.0% | 33.0% | 8.5% |

## 只看 Spider 成功 case

用户指出：Spider/ref_fk 明显不 work 的 case 不应作为“raw_mask 是否超过 Spider”的比较分母。因此 E143 同步输出 Spider 成功过滤表，过滤依据为 E109 `fair_eval/method_case_metrics.tsv` 中 Spider rows 的 `cem_status`；`WORK` 和旧批次 `pass` 视为成功，明确 `fail` 排除。

排除的明确 fail case：

- `bucket004_20231002_022_p1`
- `bucket004_20231003_1_012_p1`

Spider 成功过滤后为 22 个 case：

| method | cases | 手物接触 | 5cm | 10cm | 手物穿透 | 腿穿透 |
|---|---:|---:|---:|---:|---:|---:|
| OmniRetarget | 22 | 56.8% | 64.1% | 65.9% | 56.8% | 11.7% |
| ref_fk | 22 | 43.4% | 61.2% | 64.1% | 43.4% | 10.9% |
| raw_mask_ref_fk | 22 | 34.4% | 54.4% | 60.1% | 34.4% | 9.1% |

在这 22 个 Spider 成功 case 中，`raw_mask_ref_fk` 有 5 个 case 的手物接触超过 Spider/ref_fk，但仍是 0 个超过 OmniRetarget：

| case | Spider status | Spider | raw_mask_ref_fk | delta vs Spider | OmniRetarget | delta vs Omni |
|---|---|---:|---:|---:|---:|---:|
| `box026_137_p1` | WORK | 40.0% | 52.2% | +12.2pp | 62.2% | -10.0pp |
| `box004_082_p1` | pass | 37.6% | 46.8% | +9.2pp | 47.7% | -0.9pp |
| `box021_035_p1` | WORK | 70.5% | 75.2% | +4.7pp | 79.1% | -3.9pp |
| `box021_029_p2` | WORK | 45.3% | 49.3% | +4.0pp | 70.7% | -21.3pp |
| `box004_083_p1` | pass | 32.4% | 33.3% | +1.0pp | 44.1% | -10.8pp |

xlsx 不再单独放 `Spider成功最差case` sheet；`Spider成功逐case` 只比较 `raw_mask_ref_fk` 和 `OmniRetarget`，并按 `raw_mask_ref_fk - OmniRetarget` 手物接触差值从高到低排序。

## 结论

- `raw_mask_ref_fk` 没有达到目标：24 个 case 中 `0/24` 在 hand-object contact 上超过 OmniRetarget。
- `raw_mask_ref_fk` 的平均手物接触为 `33.0%`，低于 OmniRetarget 的 `54.4%`，也低于 `ref_fk` 的 `43.1%`。
- 按 Spider 成功 case 过滤后，`raw_mask_ref_fk` 是 `5/22` 超过 Spider/ref_fk，但平均仍低于 Spider/ref_fk（34.4% vs 43.4%），并且仍是 `0/22` 超过 OmniRetarget。
- `raw_mask_ref_fk` 的手物穿透也随接触一起下降到 `33.0%`，说明当前 contact objective/mask route 没有带来更好的有效接触，而是整体减少了接触/近场。
- 5cm/10cm 近场指标同样低于 OmniRetarget：`55.5%/61.0%` vs `64.6%/66.4%`。
- 腿穿透均值低于 OmniRetarget：`8.5%` vs `11.3%`，但这不是本轮主目标，且不能抵消手物接触显著下降。

## 验证

```bash
python3 -m py_compile workspace/core4d/scripts/eval/eval_E143_raw_mask_ref_fk_24case.py
bash workspace/core4d/scripts/eval/eval_E143_raw_mask_ref_fk_24case.sh full
```

最终检查：

- `raw evaluated rows = 24`
- `missing rows = 0`
- xlsx sheets: `方法平均`, `逐case对比`, `Spider成功方法平均`, `Spider成功逐case`
- `方法平均` 4 行 7 列，`逐case对比` 25 行 18 列，`Spider成功方法平均` 4 行 7 列，`Spider成功逐case` 23 行 19 列
- `Spider成功逐case` 列结构：case 标识 + 5 个核心指标，每个指标包含 `raw_mask_ref_fk`、`OmniRetarget`、`raw-Omni差值`，最后一列为 `npz/视频位置`
- `Spider成功逐case` 排序检查：按 `手物接触_raw-Omni差值` 降序，顶部为 `box004_082_p1`、`box021_035_p1`、`box026_139_p1`，底部为 `box026_133_p1`、`box026_141_p2`、`box026_138_p2`
