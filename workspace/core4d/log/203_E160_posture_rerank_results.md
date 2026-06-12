# E160 — gateA + surfaceBand-A2 + postureRerankA 结果记录

日期：2026-06-13

计划文件：`workspace/core4d/plan/169_E160_surfaceBandA2_posture_rerank_plan.md`

## 1. 实验目标

E159 `gateA+surfaceBand-A2` 能在 clean6 上显著改善 E158 的稳定性，但仍有两个关键问题：

- `box021_029_p2` fall，terminal pelvis-z tracking 明显失控。
- `box004_083_p2` release false contact 仍偏高。

E160 在 E159 A2 的 reward 上不再加新 reward，而是在 CEM elite selection 层加入相对参考轨迹的 root-z posture gate/rerank：

```text
gateA+surfaceBand-A2+postureRerankA
```

核心约束不是绝对 `pelvis_z > 0.5m`，而是与 kinematic reference 的 `qpos[:,2]` 对齐：

```text
mean_z_err <= 0.10m
terminal_z_err <= 0.12m
max_z_drop <= 0.18m
fallback_score = reward - 5.0 * posture_violation
```

## 2. 运行与回收

本轮按计划跑 3 个 failure-focused case：

| case | split | 作用 |
|---|---|---|
| `box021_029_p2` | local-gpu0 | E159 fall case，首要验证 |
| `box004_083_p2` | remote-gpu0 | E159 release false case |
| `box023_person2` | remote-gpu1 | 正常弯腰 sanity case |

执行方式：

| 类型 | 入口/会话 |
|---|---|
| local full | `workspace/core4d/scripts/launch/active/run_E160_local.sh full`，tmux `E160_local_full_021305` |
| remote full | `workspace/core4d/scripts/launch/active/run_E160_remote.sh full`，tmux `E160_full_021313` |
| pull | `workspace/core4d/scripts/launch/active/pull_E160_remote_results.sh full` |
| eval | `workspace/core4d/scripts/eval/runners/eval_E160_posture_rerank.py full` |

远程机器通过 rsync 同步当前分支所需代码/脚本/数据，没有执行 `git pull`，避免覆盖远程已有脏工作区。

结果完整性：

| artifact | 状态 |
|---|---|
| root result npz | 3/3 |
| full mp4 | 3/3 |
| outdir `trajectory_mjwp_act.npz` | 3/3 |
| outdir `config_act.yaml` | 3/3 |

结果目录：

```text
workspace/core4d/results/E160/posture_rerank/cem/full/
workspace/core4d/results/E160/posture_rerank/eval/full/
workspace/core4d/results/E160/posture_rerank/diagnostics/frames/
```

## 3. 评测产物

Strict eval 输出：

```text
E160 eval: metric_rows=18 methods=6 delta_vs_gate=9 missing=0
```

主要文件：

| 文件 | 内容 |
|---|---|
| `e160_method_metrics.tsv` | per-case metrics |
| `e160_method_summary.tsv` | method summary |
| `e160_delta_vs_gateA.tsv` | 相对 `+gateA` delta |
| `e160_delta_vs_omniretarget.tsv` | 相对 OmniRetarget delta |
| `e160_delta_vs_spider_rubberhand.tsv` | 相对 spider-rubberhand delta |
| `e160_eval_summary.json` | claims 自动判定 |
| `E160_posture_rerank_3case.xlsx` | 汇总表 |

命名说明：实际 override/variant 使用 `E160_*_gateA_postureRerankA`，这是为了与 E156/E159 的 `+gateA` lineage 和结果文件名保持一致；override 内容已固定写入 surfaceBand-A2 参数，包括 `surface_band_rew_scale=1.5`、`surface_band_penalty_scale=0.0`、`surface_band_min_sdf_m=-0.001`。

XLSX 检查：

- sheets: `method_summary`, `per_case_metrics`, `delta_vs_omniretarget`, `delta_vs_baseline`, `delta_vs_gateA`, `summary`
- `method_summary` 中最优值有黑色加粗，次优值有下划线。

## 4. 关键结果

### 4.1 3-case mean

| method | tracked | fall | inmaskC3 | physPen3 | geomPen2 | releaseF3 | pz terminal |
|---|---:|---:|---:|---:|---:|---:|---:|
| `+gateA` | 3/3 | 0 | 0.122 | 0.279 | 0.274 | 0.000 | 0.0117 |
| `gateA+surfaceBand-A2` | 2/3 | 1 | 0.460 | 0.105 | 0.030 | 0.083 | 0.1720 |
| `gateA+surfaceBand-A2+postureRerankA` | 3/3 | 0 | 0.519 | 0.106 | 0.038 | 0.271 | 0.0173 |

相对 E159 A2：

| metric | delta |
|---|---:|
| `inmaskC3` | +0.059 |
| `physPen3` | +0.001 |
| `geomPen2` | +0.008 |
| `releaseF3` | +0.188 |

相对 `+gateA`：

| metric | delta |
|---|---:|
| `inmaskC3` | +0.397 |
| `physPen3` | -0.173 |
| `geomPen2` | -0.235 |
| `releaseF3` | +0.271 |

### 4.2 E160 per-case

| case | tracked | fall | pz terminal | inmaskC3 | physPen3 | geomPen2 | releaseF3 | posture valid | fallback |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `box021_029_p2` | true | false | 0.042 | 0.455 | 0.200 | 0.013 | 0.750 | 0.870 | 0.0217 |
| `box004_083_p2` | true | false | 0.007 | 0.565 | 0.067 | 0.057 | 0.062 | 0.933 | 0.0063 |
| `box023_person2` | true | false | 0.003 | 0.538 | 0.051 | 0.044 | 0.000 | 0.954 | 0.0000 |

## 5. Claims 验证

`e160_eval_summary.json` 中 `success.checks` 全部为 true：

| Claim | 判定 | 证据 |
|---|---|---|
| C1 修复 `box021_029_p2` fall | pass | tracked=true, fall=false, terminal pz err=0.042 <= 0.08 |
| C2 不牺牲接触 | pass | mean inmaskC3 相对 E159 A2 +0.059，没有下降 |
| C3 不增加穿透 | pass | mean physPen3 +0.001、geomPen2 +0.008，均 <= +0.03 |
| C4 不恶化 `box004_083_p2` release | pass | `box004` releaseF3 相对 E159 A2 -0.188 |
| C5 不误杀 `box023_person2` | pass | tracked=true，fallback=0 |

自动 summary 给出：

```text
promote_to_clean6_probe = true
missing = []
```

## 6. 视觉检查

Frame sheets：

| case | 文件 | 观察 |
|---|---|---|
| `box021_029_p2` | `diagnostics/frames/box021_029_p2_sheet.jpg` | 视角较远，但没有 E159 那种明显趴倒/塌陷；定量 tracking 已恢复 |
| `box004_083_p2` | `diagnostics/frames/box004_083_p2_sheet.jpg` | 弯腰和搬箱过程稳定，无明显 fall |
| `box023_person2` | `diagnostics/frames/box023_person2_sheet.jpg` | 正常弯腰和搬运未被 posture gate 误杀 |

## 7. 结论

E160 的 posture rerank 达到了本轮核心目的：修复 `box021_029_p2` 的 fall，并保持 E159 A2 的高 mask 内接触和低穿透水平。3 个 case 全部 tracked，`fall=0/3`，strict eval `missing=0`。

但 E160 不能直接作为默认方法升级。主要风险是：

- `box021_029_p2` 的 `releaseF3=0.75`，导致 3-case mean `releaseF3` 相对 E159 A2 反而 +0.188。
- 这说明 posture rerank 解决的是“为了贴物而牺牲 body tracking/fall”的问题，不解决放手语义。

建议下一步：

1. 可以进入 clean6 probe，但需要把 release false 作为硬诊断项，而不是只看 C1-C5。
2. 优先设计 release-side gate 或 mask-aware surface reward，使 surfaceBand 在 reference release 窗口不继续吸附。
3. 对 `box021_029_p2` 单独画 release 窗口内的 hand-object SDF/contact 曲线，确认 0.75 release false 的具体时间段。
