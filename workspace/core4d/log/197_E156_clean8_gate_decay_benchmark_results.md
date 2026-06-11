# E156 — clean8 gate/decay benchmark 结果

> 计划:`workspace/core4d/plan/165_E156_clean8_gate_decay_benchmark_plan.md`
> 状态:**完成(13/13 新 full CEM + 24/24 strict eval)**
> 指标标准:`core4d-e154-physics-contact-v1`
> 对比方法:`spider-rubberhand` / `+gateA` / `E155_decay`

## 0. 一句话结论

E156 把 E155 的 3-case `decay` 放大到 clean8 后，结论不能推广为默认策略。

三种方法都保持 **tracking 8/8、0 fall**。`E155_decay` 的 in-mask 物理接触最高
(`inmaskC3/5=0.314/0.479`)，但它同时显著增加了放手段 false contact
(`release_false3/5=0.128/0.185`) 和 3mm 物理穿透(`physPen3=0.313`)。相对 `+gateA`，
`decay` 的 `release_false3` **+0.115**、`physPen3` **+0.111**，没有满足升级为 clean8 默认策略的门槛。

当前 clean8 默认候选应保留为 **`+gateA`**：tracking 8/8，`release_false3=0.013`，
`physPen3=0.202`，并相对 `spider-rubberhand` 轻微降低几何/物理穿透。

## 1. 执行

Benchmark 使用 E149/E150 固定的 `relaxed8_valid_like` clean8:

| case | 备注 |
|---|---|
| `box021_035_p1` | clean6 primary |
| `box021_035_p2` | clean6 primary |
| `box021_029_p2` | E155 selected |
| `box004_083_p1` | clean6 primary |
| `box004_083_p2` | E155 selected |
| `box023_person2` | E155 selected |
| `box004_082_p1` | relaxed8 added |
| `box026_139_p1` | relaxed8 added |

跑量:

| 方法 | 覆盖 | 来源 |
|---|---:|---|
| `spider-rubberhand` | 8/8 | 复用 E148/E147 rubberhand artifacts |
| `+gateA` | 8/8 | E156 新跑 |
| `E155_decay` | 8/8 | 复用 E155 3 条，新跑 5 条 |

新增 full CEM 13/13 完成。回收后本地检查:

| artifact | 数量 |
|---|---:|
| root npz | 13 |
| `trajectory_mjwp_act.npz` | 13 |
| full mp4 | 13 |

远程结果已通过 `workspace/core4d/scripts/launch/active/pull_E156_remote_results.sh full` 回收。
本地/远程日志尾部有 MuJoCo EGL 析构噪声，artifact 完整，strict eval 正常通过。

## 2. clean8 聚合

| method | tracked | fall | rel_false3 | rel_false5 | inmaskC3 | inmaskC5 | physPen3 | physPen5 | pen2 | pen5 | legPen | objErr |
|---|:--:|:--:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| spider-rubberhand | 8/8 | 0 | **0.013** | **0.013** | 0.145 | 0.247 | 0.210 | **0.146** | 0.217 | 0.140 | 0.095 | 0.00834 |
| +gateA | 8/8 | 0 | 0.013 | 0.038 | 0.153 | 0.224 | **0.202** | 0.153 | **0.211** | **0.130** | **0.094** | **0.00825** |
| E155_decay | 8/8 | 0 | 0.128 | 0.185 | **0.314** | **0.479** | 0.313 | 0.195 | 0.374 | 0.151 | 0.130 | 0.00879 |

说明:
- `tracked` = no fall 且 terminal pelvis-z tracking <=0.08m。
- `rel_false3/5` 越低越好；`inmaskC3/5` 越高越好；`physPen3/5`、`pen2/5`、`legPen`、`objErr` 越低越好。
- XLSX 中按同一方向规则标出最优黑色加粗、次优下划线。

## 3. 相对 `+gateA` 的主结论

| 比较 | delta |
|---|---:|
| `E155_decay` release_false3 | +0.115 |
| `E155_decay` inmaskC3 | +0.161 |
| `E155_decay` physPen3 | +0.111 |
| `E155_decay` hand pen2 | +0.162 |
| `E155_decay` legPen | +0.035 |
| `E155_decay` objErr | +0.00054 |

`decay` 的主要收益是接触量更足，但代价是放手 false contact 和穿透都明显升高。这个 tradeoff 与 E155
3-case 结论相反，说明 E155 的 selected cases 对 clean8 不够代表，不能直接推广。

## 4. Claims 验证

| Claim | 标准 | 结果 | 裁定 |
|---|---|---|---|
| C1 success_tracked | `E155_decay >= 7/8` | 8/8 | **成立** |
| C2 release_false3 改善 | 相对 `+gateA` mean delta <= -0.05 | +0.115 | **不成立** |
| C3 inmaskC3 不明显下降 | 相对 `+gateA` mean delta >= -0.10 | +0.161 | **成立** |
| C4 physPen3 不明显上升 | 相对 `+gateA` mean delta <= +0.03 | +0.111 | **不成立** |

最终 promotion: `promote_decay=false`。

## 5. 结论与下一步

- **不要把 `E155_decay` 设为 clean8 默认策略**。它在 8 case 上接触更强，但 false contact/穿透代价过大。
- **保留 `+gateA` 作为当前 clean8 默认候选**。它在 E156 中没有牺牲 tracking，且相比 baseline 穿透略降。
- 后续若继续做 release 策略，应避免单纯扩大接触窗口或尾段衰减；需要把 release false contact 直接纳入优化目标或做 per-case/phase-aware gating。
- E155 selected-3 可以继续作为诊断子集，但不能再替代 clean8 benchmark 作默认策略决策。

## 6. 结果路径

| 类型 | 路径 |
|---|---|
| manifest | `workspace/core4d/scripts/experiments/E156/variants.tsv` |
| CEM full | `workspace/core4d/results/E156/clean8_gate_decay/cem/full/` |
| strict eval | `workspace/core4d/results/E156/clean8_gate_decay/eval/full/` |
| XLSX | `workspace/core4d/results/E156/clean8_gate_decay/eval/full/E156_clean8_gate_decay_benchmark.xlsx` |
| eval runner | `workspace/core4d/scripts/eval/runners/eval_E156_clean8_gate_decay.py` |
| eval wrapper | `workspace/core4d/scripts/eval/wrappers/eval_E156_clean8_gate_decay.sh` |
| local/remote launch | `workspace/core4d/scripts/launch/active/run_E156_{local,remote}.sh` |
| remote pull | `workspace/core4d/scripts/launch/active/pull_E156_remote_results.sh` |

