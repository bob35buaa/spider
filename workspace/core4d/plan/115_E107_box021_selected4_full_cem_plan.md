# E107 Phase 2 计划：Box021 selected-4 full CEM

日期：2026-06-01
上游：

- E107 Phase 1：`workspace/core4d/log/134_E107_box021_clean_reconstruction_gate_results.md`
- 选中 case：`workspace/core4d/results/E107/selected_case_to_cem.json`
- 执行方式参考 E106：本地 1 卡 + 远程 2 卡，full CEM，`ref_fk_clean` route。

## 目标

对 `selected_case_to_cem.json` 中 4 个 Box021 clean target 跑 full CEM，作为 E107 第二步，输出可对齐 E106 的数值评估、视频、失败模式和 RL strict candidate 结论。

## 候选

| ordinal | selected id | derived task | split |
|---:|---|---|---|
| 1 | `d003_box021_20231011_034_p1_e107` | `d003_box021_20231011_034_p1_e107_clean` | local-gpu0 |
| 2 | `d003_box021_20231011_035_p1_e107` | `d003_box021_20231011_035_p1_e107_clean` | remote-gpu0 |
| 3 | `d003_box021_20231011_035_p2_e107` | `d003_box021_20231011_035_p2_e107_clean` | remote-gpu1 |
| 4 | `d003_box021_20231018_029_p2_e107` | `d003_box021_20231018_029_p2_e107_clean` | remote-gpu0 |

远程 GPU0 串行 2 个，远程 GPU1 跑 1 个，本地 GPU0 跑 1 个。

## 验证声明

| claim | 验证方式 | 成功标准 |
|---|---|---|
| C1 selected-4 manifest 与 E107 clean gate 对齐 | 读取 `selected_case_to_cem.json` 和 gate summary | 4/4 target 存在且 `cem_ready=True` |
| C2 full CEM 使用 ref-FK clean route | override 审计 | 4/4 override 为 `contact_hdmi_target_source: ref_fk`，无 external target |
| C3 跑前可视化与 medium subagent 审查完成 | pre-CEM replay + REVIEW.md | 4/4 `Status: PASS` 或 `PASS_WITH_NOTES` |
| C4 三卡并行启动 full CEM | 本地/远程 tmux + logs | 本地 1 卡、远程 2 卡启动；同卡串行 |
| C5 结果可用且评估完整 | eval summary + MP4/NPZ | 4/4 CEM 输出存在，生成 full eval summary |

## 实现

新增/更新：

- `workspace/core4d/scripts/E107/e107_common.py`
- `workspace/core4d/scripts/E107/build_selected4_cem_manifest.py`
- `workspace/core4d/scripts/E107/render_pre_cem_replays.py`
- `workspace/core4d/scripts/train/train_E107_box021_selected4_full.sh`
- `workspace/core4d/scripts/run_E107_remote.sh`
- `workspace/core4d/scripts/sync_E107_remote.sh`
- `workspace/core4d/scripts/pull_E107_remote_results.sh`
- `workspace/core4d/scripts/eval/eval_E107_box021_selected4.py`
- 4 个 `examples/config/override/core4d_E107C*.yaml`

## 执行命令

准备：

```bash
python workspace/core4d/scripts/E107/build_selected4_cem_manifest.py
python workspace/core4d/scripts/E107/render_pre_cem_replays.py --overwrite --all
```

本地：

```bash
bash workspace/core4d/scripts/train/train_E107_box021_selected4_full.sh local full 0
```

远程：

```bash
bash workspace/core4d/scripts/sync_E107_remote.sh
ssh spider-remote "cd /home/xiayb/pHRI_workspace/spider && tmux new-session -d -s e107_remote_full_$(date +%Y%m%d_%H%M%S) 'bash workspace/core4d/scripts/run_E107_remote.sh full'"
```

回收与评估：

```bash
bash workspace/core4d/scripts/pull_E107_remote_results.sh full
bash workspace/core4d/scripts/train/train_E107_box021_selected4_full.sh eval full 0
```

## 成功标准

- 4 个 full CEM root NPZ 和 MP4 均存在。
- `workspace/core4d/results/E107/cem/full/full_eval_summary.{json,csv,md}` 生成。
- log 记录每个 case 的 upper WORK、replay gate、lower-body strict、身体/腿部干涉、最终是否可进 RL。

## 停止规则

若 pre-CEM medium subagent 审查任一 case `FAIL_PRE_CEM_VISUAL`，该 case 不启动 CEM；若远程/本地某个 CEM 失败，记录错误并优先修执行链路，不把执行失败归因给算法。
