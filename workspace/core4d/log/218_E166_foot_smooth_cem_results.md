# E166 foot/smooth CEM results

日期：2026-06-18

## Scope

本轮完成 E166 的 SPIDER/CEM 侧因子化消融与 xlsx 对比表，不停止正在跑的 SUGAR downstream。

- 主线 case：`box021_035_p2`、`box004_082_p1`、`box004_083_p2`
- 排除：`box023_person2`，作为 pathology/self-collision/init-penetration 特例
- 不补：`box026_139_p1` downstream label
- arms：`baseline`、`B1`、`B2`、`A`、`A_B2_postSmooth`、`AplusB`
- 术语边界：`AplusB` = A+B1 的 CEM arm；`A_B2_postSmooth` = 先跑 A CEM，再对 A 输出套 B2 CPU 后平滑，不是新 CEM arm

## Artifacts

结果根目录：

- `workspace/core4d/results/E166/foot_smooth_retarget/cem/full/`
- `workspace/core4d/results/E166/foot_smooth_retarget/postprocess/full/`
- `workspace/core4d/results/E166/foot_smooth_retarget/eval/full/`
- 完整 xlsx：`workspace/core4d/results/E166/foot_smooth_retarget/eval/full/E166_foot_smooth_vs_E163_three_case_eval.xlsx`

完整性：

| artifact | count |
|---|---:|
| CEM root npz | 9/9 |
| CEM full mp4 | 9/9 |
| CEM outdir trajectory | 9/9 |
| CEM config_act | 9/9 |
| postprocess npz (`B2` + `A_B2_postSmooth`) | 6/6 |
| postprocess smooth report (`B2` + `A_B2_postSmooth`) | 6/6 |
| evaluated artifact rows | 18/18 |

最终 manifest：

```text
rows=18
cem_to_run=0
postprocess_to_run=0
preflight_ok=True
split_counts={'local-gpu0': 0, 'remote-gpu0': 0, 'remote-gpu1': 0}
```

最终 strict eval：

```text
E166 eval: stage=full expected=18 evaluated=18 missing=0
```

xlsx readback 已验收：

- sheets：`主表`、`逐case`、`SmoothFoot健康度`、`E166产物检查`、`说明`、`原始metrics`
- `主表` 6 行，`逐case` 18 行，`原始metrics` hidden
- `summary.json` 记录 xlsx 路径
- formula-error-like check：0

## Summary

| arm | pass | raw contact | clean3 contact | pen3 | qpos jerk p95 | track jerk p95 | ankle acc max | foot slip | failed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| baseline | 3/3 | 0.653 | 0.445 | 0.140 | 4388.1 | 579.0 | 46.0 | 0.969 | |
| B1 | 3/3 | 0.654 | 0.503 | 0.105 | 4494.4 | 495.1 | 47.8 | 1.105 | |
| B2 | 2/3 | 0.625 | 0.457 | 0.115 | 1987.8 | 315.4 | 31.4 | 1.014 | `box021_035_p2` |
| A | 3/3 | 0.748 | 0.548 | 0.131 | 4308.5 | 519.6 | 48.5 | 1.120 | |
| A_B2_postSmooth | 3/3 | 0.777 | 0.618 | 0.101 | 1907.0 | 299.1 | 31.5 | 1.122 | |
| AplusB | 3/3 | 0.730 | 0.594 | 0.095 | 4536.3 | 575.4 | 59.9 | 1.048 | |

`pass` 是 SPIDER 侧 gate：tracking pass、no fall、raw/clean contact 相对 baseline 不低于 `-0.05`、3mm penetration 不高于 `+0.05`。

## A_B2_postSmooth detail

| case | status | raw contact | raw delta | clean3 delta | pen3 delta | qpos jerk delta | track jerk delta | ankle acc delta |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `box021_035_p2` | pass | 0.835 | +0.078 | +0.194 | -0.098 | -784.8 | -238.0 | -14.8 |
| `box004_082_p1` | pass | 0.705 | +0.131 | +0.148 | -0.009 | -5794.4 | -469.3 | -17.3 |
| `box004_083_p2` | pass | 0.790 | +0.161 | +0.177 | -0.010 | -864.4 | -132.3 | -11.3 |

## Main Findings

1. **A_B2_postSmooth 是当前 SPIDER/CEM 侧最强候选。**
   它三 case 全部通过，mean raw contact `0.777`、clean3 `0.618`、qpos jerk p95 `1907.0`、trackbody jerk p95 `299.1`。相对 baseline 同时提升接触并显著降抖，没有复现 B2-only 在 `box021_035_p2` 的 contact regression。

2. **A 仍是最强 pure CEM reward arm。**
   A 三 case 全部通过，raw contact `+0.094`、clean3 `+0.103`、3mm penetration `-0.009`。如果只考虑不做 CPU postprocess 的 CEM 输出，A 是最稳的脚约束候选。

3. **AplusB 通过但不是 A_B2。**
   AplusB 是 A+B1，clean3 `0.594`、pen3 `0.095`，但 qpos jerk、trackbody jerk、ankle acc 均不如 A_B2_postSmooth。它仍适合当前 SUGAR downstream 对照，但不能代表“后平滑 B2”的效果。

4. **B2-only 仍不能直接推广。**
   B2-only 把 qpos jerk p95 `4388.1→1987.8`、trackbody jerk `579.0→315.4`，但 `box021_035_p2` raw contact delta `-0.068`，触发 contact regression gate。A 后再做 B2 才消除了这个问题。

5. **foot-slip FK 诊断仍未被当前 A 系列压低。**
   A/A_B2/AplusB 的接触和 penetration gate 明显改善，但 `foot_slip_max_m` 仍高于 baseline。当前收益更像 contact/physical gate 与运动平滑改善，是否真正降低下游 `ee_body_pos` 踝失败，需要看 SUGAR failed_windows。

## Claims

| claim | 结论 | 证据 |
|---|---|---|
| C-B1 | 部分支持 | B1 降低 FK trackbody jerk 且不牺牲接触，但 qpos jerk 未降、foot slip 上升 |
| C-B2 | 条件支持 | B2-only 显著降抖但一例接触回归；A 后接 B2 的 `A_B2_postSmooth` 3/3 pass |
| C-A1/A3 | SPIDER 侧支持，downstream 待验证 | A/A_B2/AplusB 3/3 pass，接触/penetration 改善；foot-slip 诊断未下降 |
| C-disentangle | CEM 侧部分成立 | `box004_083_p2` 上 A/A_B2/AplusB 明显改善接触，B1/B2 只小幅变化；最终仍需 SUGAR staggered 验证 |
| contact no-regression | A、A_B2_postSmooth、AplusB、B1 通过；B2-only 不通过 | strict eval 18/18，B2 fail case 为 `box021_035_p2` |

## Next

2026-06-19 更新：用户决策暂停 `AplusB` downstream，优先让 `A_B2_postSmooth` 接替验证。

当前执行状态：

- 三路 `A` 训练继续运行，未 kill。
- 当前无 `AplusB` 训练进程；本地/远程 AplusB SUGAR data 已移到 `data/paused_E166_AplusB_20260619/`，防止旧 launcher 在 A 后误启动。
- 已生成 `A_B2_postSmooth` 三 case SUGAR data，并把 box004 两路同步到远程。
- watcher 已切换为等待 `A` + `A_B2_postSmooth` 共 6 条；after-A 监控会在各自 A eval 完成后启动 A_B2-only worker。

建议后续：

1. 当前 SUGAR 先等 `A` 完成并自动/监控启动 `A_B2_postSmooth`。
2. 暂不恢复 `AplusB`，除非后续需要 A+B1 downstream 对照。
3. 下游重点看 `box004_083_p2` 是否由 A 系列提升，以及 failed_windows 中 `ee_body_pos` 的踝主导占比是否下降。
