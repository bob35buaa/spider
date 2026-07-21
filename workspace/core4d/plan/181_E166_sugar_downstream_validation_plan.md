# E166 SUGAR downstream validation plan

日期：2026-06-18

## Context

E166 CEM full 已完成并记录在 `log/218_E166_foot_smooth_cem_results.md`：

- `A`：3/3 SPIDER gate pass，raw/clean3 contact 显著提升。
- `AplusB`：3/3 SPIDER gate pass，clean3 contact 最高、3mm penetration 最低。
- `B1`：3/3 pass，但平滑收益有限。
- `B2`：2/3 pass，`box021_035_p2` raw contact regression，不作为主候选。

因此下游验证先跑 **A/AplusB × 3 case = 6 条 SUGAR RL**，复用 E163/E165D 的训练 recipe 和 staggered eval。`box023_person2` 继续作为 pathology 特例排除；`box026_139_p1` 不补 label。

## Claims

| Claim | 最低证据 |
|---|---|
| C-A1 downstream | A 在 `box004_083_p2` 的 staggered 从 E163 baseline `0.484` 提升到 `>=0.60`，且 failed_windows 中踝主导 `ee_body_pos` 占比下降 |
| C-AplusB | AplusB 在 `box021_035_p2`、`box004_082_p1` 至少一条超过 A，并达到计划目标 `0.10/0.20` |
| no-regression | A/AplusB 不复现 E165D 的 0/64 collapse；staggered eval 必须真实去同步并写出 `success_vs_phase.csv` |
| diagnosis | 每条 eval 必须有 `success_vs_phase.csv`；失败 case 必须有 `failed_windows.csv` 或明确 64/64 success |

## Scope

本轮进入：

1. 新建 E166 RL export：读取 `variants.tsv` + `e166_arm_metrics.tsv`，输出 A/AplusB 六条 `rl_export_input.tsv`。
2. Holosoma export：生成 `/home/ubuntu/Workspace/holosoma/workspace/v3/data/R171_E166_A_AplusB_threecase_rl`。
3. SUGAR converter：输出独立 data 目录，命名含 `E166`、arm、case，避免覆盖 E163/E165D。
4. SUGAR launch/pull/summarize：本地 1 卡 + 远程 2 卡并行；每条训练后自动 staggered eval。

本轮不进入：

1. 不启动 `B2` 主线训练。
2. 不把 `box023_person2` 或 `box026_139_p1` 加入 E166 主线。
3. 不复用或覆盖 E163/E165D 的 SUGAR data/output 目录。

## GPU split

第一批 6 条：

| worker | cases |
|---|---|
| local GPU0 | `box021_035_p2/A`, `box021_035_p2/AplusB` |
| remote GPU0 | `box004_082_p1/A`, `box004_082_p1/AplusB` |
| remote GPU1 | `box004_083_p2/A`, `box004_083_p2/AplusB` |

## Success criteria

| Item | Evidence |
|---|---|
| Export ready | E166 `rl_export_input.tsv` has 6 rows, all `RL_EXPORT_READY` |
| SUGAR data ready | 6 data folders each contain `robot_50hz.npz`, `obj_motion_global_50hz.pkl`, `contact_labels_50hz.npy` |
| Launch safe | local/remote scripts pass `bash -n`; no output paths overlap E163/E165D |
| Training complete | 6 `model_5999.pt` |
| Eval complete | 6 `success_vs_phase.csv`; failed windows present unless 64/64 success |
| Report | summary TSV/MD compares A/AplusB vs E163 baseline target thresholds |

## Commands

```bash
bash workspace/core4d/scripts/launch/active/run_E166_sugar_export.sh

# after data preflight
bash /home/ubuntu/Workspace/Loco-Manipulation/SUGAR/scripts/sugar_rl/launch_core4d_e166_refiner.sh local
bash /home/ubuntu/Workspace/Loco-Manipulation/SUGAR/scripts/sugar_rl/launch_core4d_e166_refiner.sh remote_gpu0
bash /home/ubuntu/Workspace/Loco-Manipulation/SUGAR/scripts/sugar_rl/launch_core4d_e166_refiner.sh remote_gpu1

bash /home/ubuntu/Workspace/Loco-Manipulation/SUGAR/scripts/sugar_rl/pull_core4d_e166_refiner_remote.sh
python /home/ubuntu/Workspace/Loco-Manipulation/SUGAR/scripts/sugar_rl/summarize_core4d_e166_failed_windows.py
```
