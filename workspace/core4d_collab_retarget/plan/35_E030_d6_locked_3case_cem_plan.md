# E030 Plan: D6 locked 3-case direct CEM

日期：2026-05-27

## Context

E029 的正式 gate 结论是 `stop_before_full_sanity_failed`：5 个 D003 Box021 candidates 中，best gate-eligible no-training sanity 只达到 `3/5`，没有满足 plan 34 的 `>=4/5` full CEM 条件。

用户现在明确要求：先不要继续方法语义重构，直接对已经通过 sanity 的 3 个物体跑 CEM，并行使用本地 1 卡 + 远程 2 卡。当前机器上已有其他程序占用显存，但剩余显存足够；执行期间不得 kill 其他程序。远程执行按 `experiment-planning-zh/remote-execution.md`，并在完成后回收远程结果。后续视频视觉审查交给 high subagent。

本实验因此作为 E030 exploratory direct-CEM 分支，不改写 E029 的 gate 结论。

## Variants

只跑 E029 D6 locked raw sanity 已通过的 3 个 variants：

| 分配 | Variant | Source task |
|---|---|---|
| 本地 GPU0 | `E029_d003_box021_20231018_029_p2_d6_locked` | `d003_box021_20231018_029_p2` |
| 远程 GPU0 | `E029_d003_box021_20231011_035_p2_d6_locked` | `d003_box021_20231011_035_p2` |
| 远程 GPU1 | `E029_d003_box021_20231020_019_p1_d6_locked` | `d003_box021_20231020_019_p1` |

## Claims

| Claim | 验证方式 |
|---|---|
| C1 只运行 sanity 通过的 3 个 D6 locked case | variant list 固化在 `scripts/train/train_E030.sh` 和 `scripts/run_E030_remote.sh` |
| C2 不清理其他 GPU 程序 | 启动前只检查 `nvidia-smi`，不执行 kill/reset |
| C3 本地 1 卡 + 远程 2 卡并行启动 | 本地 shell/tmux 日志 + 远程 tmux 输出 |
| C4 结果可回收 | `scripts/pull_E030_remote_results.sh` 回收远程 `npz/mp4/log` |
| C5 full CEM 输出完整 | 3/3 `trajectory_mjwp.npz`、root copy、online video、运行日志存在 |
| C6 视觉审查由 high subagent 完成 | subagent 审查 3 个 online videos 并输出文字判断 |

## Outputs

| 类型 | 路径 |
|---|---|
| CEM 结果 | `workspace/core4d_collab_retarget/results/E030/d6_locked_cem/` |
| 在线视频 | `workspace/core4d_collab_retarget/results/E030/d6_locked_cem/online_video/` |
| 关键帧 | `workspace/core4d_collab_retarget/results/E030/d6_locked_cem/keyframes/` |
| 运行日志 | `logs/core4d_collab_retarget/E030/` |
| 远程脚本 | `workspace/core4d_collab_retarget/scripts/run_E030_remote.sh` |
| 回收脚本 | `workspace/core4d_collab_retarget/scripts/pull_E030_remote_results.sh` |

## Commands

```bash
# 本地 representative case
bash workspace/core4d_collab_retarget/scripts/train/train_E030.sh local 0

# 远程 2 卡
ssh spider-remote "cd /home/xiayb/pHRI_workspace/spider && tmux new-session -d -s E030_d6_locked_3case 'bash workspace/core4d_collab_retarget/scripts/run_E030_remote.sh'"

# 回收远程结果
bash workspace/core4d_collab_retarget/scripts/pull_E030_remote_results.sh
```

## Success Criteria

- 本地和远程进程均正常结束；
- 3/3 root `*.npz` 存在；
- 3/3 `online_video/*.mp4` 存在；
- 3/3 运行日志存在且没有未解释的 traceback；
- high subagent 完成视觉审查，记录每个 case 的接触、物体漂移、机器人姿态和是否值得继续。

## Stop Conditions

- 任一 case CUDA OOM 或数值崩溃时，记录错误，不 kill 其他已有进程；
- 如果远程 repo dirty 阻止 `git pull`，不清理远程 dirty files，改用按需 `rsync/scp` 同步本次 E030 所需代码和数据；
- 若三条均失败，不继续扩大到 5-case 或调参 sweep。
