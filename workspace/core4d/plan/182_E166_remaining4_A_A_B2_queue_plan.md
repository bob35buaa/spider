# E166 remaining4 A/A_B2 queue plan

日期：2026-06-19

## Context

E166 三个代表 case 已完成 SPIDER/CEM 侧 `A`、`A_B2_postSmooth` 对比；`A_B2_postSmooth` 在 CEM 侧同时改善接触与平滑度。当前三 case 的 SUGAR downstream 已在跑 `A`，并已从 `AplusB` 切换为后续接 `A_B2_postSmooth`。

用户要求在当前三 case 的 `A` 与 `A_B2_postSmooth` 后面，排 clean8 中除已有三 case 与 `box026_139_p1` 外的四个 case，两版重定向为 `A` 和 `A_B2_postSmooth`，并启动 `A_B2_postSmooth` 的 SUGAR-RL。所有后续 GPU 作业三卡并行；可以先写代码和 smoke/preflight，等待当前三 case 完成后再启动；检查间隔 20 分钟，并回收远程结果。

## Scope

新增 remaining4 case：

| case | source |
|---|---|
| `box023_person2` | E163 clean8 |
| `box021_029_p2` | E163 clean8 |
| `box021_035_p1` | E163 clean8 |
| `box004_083_p1` | E163 clean8 |

排除：

- 已在当前 E166 downstream 中覆盖：`box021_035_p2`、`box004_082_p1`、`box004_083_p2`
- 用户明确不补：`box026_139_p1`

新增 arms：

- `A`: CEM foot constraints。
- `A_B2_postSmooth`: 先跑 `A`，再用同款 B2 CPU 后平滑。

SUGAR-RL 只跑 `A_B2_postSmooth`；`A` 作为 CEM/postprocess source 与 SPIDER 侧对照。

## Claims

| claim | 最低证据 |
|---|---|
| C-rem4-A | remaining4 的 `A` CEM 能三卡并行完成，且产物完整 | 4/4 root npz、outdir trajectory、mp4、config_act 存在 |
| C-rem4-A_B2 | remaining4 的 `A_B2_postSmooth` CPU 后处理能从 A 产物生成 | 4/4 postprocess npz + smooth report 存在 |
| C-rem4-RL | remaining4 的 `A_B2_postSmooth` 能进入 SUGAR downstream | 4/4 SUGAR data 三件套存在，三卡排队启动，最终产生 model_5999 与 success_vs_phase |
| C-queue | 后续作业不抢占当前三 case A/A_B2 训练 | watcher 等当前 A/A_B2 6 条完成后才启动 remaining4 |

## Implementation

新增或扩展脚本：

| 文件 | 作用 |
|---|---|
| `workspace/core4d/scripts/experiments/E166/build_remaining4_A_A_B2_manifest.py` | 生成 remaining4 × `A/A_B2_postSmooth` manifest、A override、preflight |
| `workspace/core4d/scripts/launch/active/run_E166_remaining4_local.sh` | 单机按 split 跑 A CEM |
| `workspace/core4d/scripts/launch/active/run_E166_remaining4_remote.sh` | 同步代码/数据到远程并三卡并行跑 A CEM |
| `workspace/core4d/scripts/launch/active/pull_E166_remaining4_remote_results.sh` | 回收远程 CEM/postprocess/SUGAR 结果 |
| `workspace/core4d/scripts/launch/active/run_E166_remaining4_postprocess.sh` | 对 A 产物生成 A_B2_postSmooth |
| `workspace/core4d/scripts/launch/active/run_E166_remaining4_sugar_export.sh` | 导出 remaining4 `A_B2_postSmooth` SUGAR 输入 |
| `workspace/core4d/scripts/launch/active/watch_E166_remaining4_queue.sh` | 每 20 分钟检查当前三 case 完成；随后启动 remaining4 CEM/postprocess/export/SUGAR 并回收远程 |

SUGAR 侧新增或扩展：

- `scripts/data_preprocess/convert_core4d_e166_remaining4_manifest_to_sugar.py`
- `scripts/sugar_rl/launch_core4d_e166_remaining4_refiner.sh`
- `scripts/sugar_rl/eval_staggered_phase_e166_remaining4.sh`
- `scripts/sugar_rl/pull_core4d_e166_remaining4_refiner_remote.sh`
- `scripts/sugar_rl/summarize_core4d_e166_remaining4_failed_windows.py`

## Run Plan

1. 先写脚本并做 smoke/preflight，不启动 remaining4 GPU full。
2. watcher 以 `POLL_INTERVAL=1200` 运行，等待当前三 case 的 `A/A_B2_postSmooth` 共 6 条 SUGAR eval 完成。
3. 当前三 case 完成后，三卡并行启动 remaining4 `A` CEM：
   - local GPU0: `box023_person2`, `box021_035_p1`
   - remote GPU0: `box021_029_p2`
   - remote GPU1: `box004_083_p1`
4. A CEM 完成并回收后，运行 CPU postprocess 生成 remaining4 `A_B2_postSmooth`。
5. 导出 remaining4 `A_B2_postSmooth` SUGAR data，并三卡并行启动 SUGAR-RL：
   - local GPU0: `box023_person2`, `box021_035_p1`
   - remote GPU0: `box021_029_p2`
   - remote GPU1: `box004_083_p1`
6. watcher 每 20 分钟 pull 远程，最终生成 remaining4 SUGAR summary。

## Success Criteria

- remaining4 manifest preflight `ok=true`。
- smoke/preflight 不启动长期 GPU 作业；不影响当前三 case SUGAR 训练。
- queue watcher 存在且 `POLL_INTERVAL=1200`。
- 当前三 case 完成前，不出现 remaining4 full CEM/SUGAR 训练进程。
- 启动后远程结果可被 pull 回本地。

