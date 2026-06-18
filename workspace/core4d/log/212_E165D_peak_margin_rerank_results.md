# E165D Phase 3 结果：CEM peak-margin rerank 首轮

日期：2026-06-18
Plan：[plan/176_E165_rl_safe_eval_levers_plan.md](../plan/176_E165_rl_safe_eval_levers_plan.md)
结果路径：`workspace/core4d/results/E165/peak_margin_rerank/`
方法：`E165D peakMargin025`

## 目的

验证 Phase 3 / Claim C-D：把 SUGAR hard-gate 相关的动态余量接到 CEM sample-level elite selection 上，而不是离线重排已有结果。首轮只覆盖三条 failure-focused case：

| case | split | 目的 |
|---|---|---|
| `box023_person2` | local GPU0 | 主目标：观察 `ee_body_pos` / root tracking 是否改善 |
| `box021_029_p2` | remote GPU0 | 检查 anchor/root/posture margin，不回归 fall |
| `box004_083_p2` | remote GPU1 | 检查不破坏接触/抬升相关诊断 |

## 实现摘要

- CEM rerank 对象是每轮 `ctrls_samples[i]` 的 rollout，不是离线 `.npz`。
- 新增默认关闭的 `cem_peak_margin_*` 配置；本轮开启 `cem_peak_margin_enabled=true`。
- `ee_body_pos` 覆盖双腕/双踝：`left_ankle_roll_link`、`right_ankle_roll_link`、`left_wrist_yaw_link`、`right_wrist_yaw_link`。
- `anchor_pos` 使用 `torso_link`；`anchor_ori` 仅记录诊断，不作为强 penalty。
- 不把 `obj_pos/obj_ori` 纳入 CEM rerank 主项，因为 SPIDER CEM 中物体 on-rails/GT。
- safe threshold 固定为 `0.25m`，SUGAR hard gate 仍是 `0.30m`；buffer 为 `0.03m`，即约 `0.22m` 后开始 soft 扣分。
- posture 项复用 E160 的 root-z guard：mean `0.10m`、terminal `0.12m`、max drop `0.18m`。

## 运行与回收

```bash
# manifest / preflight
.venv/bin/python workspace/core4d/scripts/experiments/E165/build_peak_margin_manifest.py

# smoke
WAIT_FOR_GPU_IDLE=0 LOCAL_GPU=0 \
  bash workspace/core4d/scripts/launch/active/run_E165D_local.sh smoke

# full: 本地 1 卡
tmux new-session -d -s E165D_local_full_012753 \
  "cd /home/ubuntu/Workspace/spider && WAIT_FOR_GPU_IDLE=0 LOCAL_GPU=0 bash workspace/core4d/scripts/launch/active/run_E165D_local.sh full"

# full: 远程 2 卡
bash workspace/core4d/scripts/launch/active/run_E165D_remote.sh full

# pull
bash workspace/core4d/scripts/launch/active/pull_E165D_remote_results.sh full

# strict eval
bash workspace/core4d/scripts/eval/wrappers/eval_E165D_peak_margin_rerank.sh full
```

远程执行遵循 `experiment-planning-zh/remote-execution.md`：SSH alias `spider-remote`，远程路径 `/home/xiayb/pHRI_workspace/spider`，GPU0/GPU1 分别跑 `box021_029_p2` / `box004_083_p2`。远程首次 preflight 曾因缺少 E163 baseline artifacts 失败，已通过同步 baseline artifacts 并修正 remote launcher 的 `rsync -R` 路径解决。

## Artifact 检查

| 项 | 结果 |
|---|---:|
| root `.npz` | 3/3 |
| full `.mp4` | 3/3 |
| `trajectory_mjwp_act.npz` | 3/3 |
| `config_act.yaml` | 3/3 |
| strict eval | `metric_rows=6, missing=0, all_artifacts_ok=true` |

关键输出：

- `workspace/core4d/results/E165/peak_margin_rerank/eval/full/e165d_method_metrics.tsv`
- `workspace/core4d/results/E165/peak_margin_rerank/eval/full/e165d_method_summary.tsv`
- `workspace/core4d/results/E165/peak_margin_rerank/eval/full/e165d_artifact_checks.tsv`
- `workspace/core4d/results/E165/peak_margin_rerank/eval/full/e165d_eval_summary.json`

## 汇总指标

| method | n | tracked | fall | raw contact in mask | clean 3mm contact in mask | phys pen 3mm | geom pen 2mm | eef cm | root cm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E163 narrowSurfaceBand | 3 | 3/3 | 0 | 0.7505 | 0.5660 | 0.1234 | 0.0452 | 14.60 | 15.83 |
| E165D peakMargin025 | 3 | 3/3 | 0 | 0.7533 | 0.4423 | 0.2090 | 0.0827 | 12.63 | 10.22 |

解释：

- tracking 侧：E165D 明显改善 mean EEF/root tracking，且 3/3 tracked、0 fall。
- contact/penetration 侧：raw contact 均值基本持平，但 clean 3mm contact 明显下降，3mm 物理穿透和 2mm 几何穿透上升。
- 因此本轮不能宣称 E165D 已是 RL-safe 改进；它证明了 CEM sample-level peak-margin rerank 管线可用，但暴露了 contact tradeoff。

## Per-case 对比

| case | raw contact Δ | clean 3mm contact Δ | phys pen3 Δ | geom pen2 Δ | EEF cm Δ | root cm Δ | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| `box023_person2` | -0.1692 | -0.2923 | +0.0588 | +0.0441 | -3.57 | -9.99 | tracking 明显改善，但主目标 case 接触严重回归 |
| `box021_029_p2` | +0.1455 | +0.0182 | +0.0933 | -0.0267 | -3.47 | -6.47 | tracking/contact 改善，物理深穿透上升 |
| `box004_083_p2` | +0.0323 | -0.0968 | +0.1048 | +0.0952 | +1.14 | -0.36 | raw contact 略升，但 clean contact 与穿透回归 |

## Peak-margin health

| case | valid frac | selected valid frac | fallback used | ee peak mean | anchor peak mean | violation mean |
|---|---:|---:|---:|---:|---:|---:|
| `box023_person2` | 0.698 | 0.793 | 0.165 | 0.179m | 0.132m | 1.051 |
| `box021_029_p2` | 0.285 | 0.415 | 0.525 | 0.309m | 0.203m | 4.288 |
| `box004_083_p2` | 0.628 | 0.797 | 0.150 | 0.234m | 0.172m | 1.780 |

`box021_029_p2` 的 valid set 明显最紧，fallback 使用率约 0.53；说明 `0.25m` 对部分 case 已经是比较强的约束。`box023_person2` 的 health 看起来健康，但接触退化，说明单独优化 body/anchor/root margin 会把解推向更好 tracking、但可能牺牲手-物接触姿态。

## Claim C-D 判定

| Claim | 结果 | 判定 |
|---|---|---|
| C-D peak-margin rerank 降低下游 hard-gate 擦边风险 | CEM health 字段齐全；tracking 平均改善；但 clean contact/penetration 回归，且尚未跑 SUGAR staggered failed_windows 对比 | **部分支持 / 不足以推广** |

当前证据只支持：

1. CEM sample-level rerank 接线正确，能改变 elite selection。
2. `ee_body_pos/anchor/root-z` 约束确实影响 tracking 和 selected set。
3. 仅靠 peak-margin 不足以保证 RL-safe，因为 contact quality 会被牺牲。

尚不能支持：

- `box023/spider` 下游 `ee_body_pos` failed_windows 下降。
- E165D 可作为 E163 后继导出到 SUGAR。

## 诊断性 RL Export

为后续 SUGAR failed_windows 对比预备输入，已执行诊断性 RL export。注意：该 export 只证明 handoff artifact 完整，不改变上面的“不推广”判定。

```bash
bash workspace/core4d/scripts/launch/active/run_E165D_peakMargin_rl_export.sh
```

输出：

- result root: `workspace/core4d/results/E165/peak_margin_rerank/rl_export`
- `rl_export_input.tsv`: `3` rows，`RL_EXPORT_READY=3`
- partner OmniRetarget: `3/3 pass`
- summary: `workspace/core4d/results/E165/peak_margin_rerank/rl_export/summary.md`

Partner 对应关系：

| source | partner | status |
|---|---|---|
| `box023_person2` | `box023_20231008_045_p1` | pass |
| `d003_box021_20231018_029_p2` | `box021_20231018_029_p1` | pass |
| `e091_box004_20231003_2_083_p2` | `box004_20231003_2_083_p1` | pass |

## 下一步

1. 不直接推广 E165D peakMargin025。
2. 若继续 Phase3，建议做 E165D2：peak-margin penalty 只作为 tie-breaker 或降低 `lambda`，并加入 contact-preservation guard（例如 raw/clean contact proxy 不低于 E163）。
3. 真正验证 C-D 仍需把 E165D 三条 handoff 跑 SUGAR staggered failed_windows，对比 `ee_body_pos`、`anchor_pos`、`obj_pos/obj_ori` 失败计数。
