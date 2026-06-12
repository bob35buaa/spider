# E159 `box021_029_p2` fall failure analysis

> 关联实验：`E159 gateA+surfaceBand-A2`
> 目标：定位 clean6 中唯一 fall case `box021_029_p2` 的失败原因
> 状态：完成诊断；不改训练代码

## 0. 结论

`box021_029_p2` 的 fall 不是物体 tracking 失败，也不是 hand gate 崩掉；根因是 surface reward 在接触窗口内持续奖励“手贴箱体表面”，使优化器接受明显下降的 body/qpos tracking。A2 降低 scale 后比 E158 A 晚倒、更轻，但仍在约 `2.13s` 触发 fall。

一句话：**surfaceBand-A2 仍然把局部最优推成“贴物优先”，缺少姿态/高度 guard 或 surface reward 末段衰减。**

## 1. 诊断产物

| artifact | 路径 |
|---|---|
| overview curve | `workspace/core4d/results/E159/gate_surface_band_A2/diagnostics/box021_029_p2_failure/box021_029_p2_failure_overview.png` |
| reward/tracking curve | `workspace/core4d/results/E159/gate_surface_band_A2/diagnostics/box021_029_p2_failure/box021_029_p2_reward_tracking_breakdown.png` |
| frame sheet | `workspace/core4d/results/E159/gate_surface_band_A2/diagnostics/box021_029_p2_failure/frames/box021_029_p2_frame_sheet.jpg` |
| per-frame TSV | `workspace/core4d/results/E159/gate_surface_band_A2/diagnostics/box021_029_p2_failure/box021_029_p2_failure_timeseries.tsv` |
| interval stats | `workspace/core4d/results/E159/gate_surface_band_A2/diagnostics/box021_029_p2_failure/box021_029_p2_interval_stats.tsv` |
| summary TSV | `workspace/core4d/results/E159/gate_surface_band_A2/diagnostics/box021_029_p2_failure/box021_029_p2_failure_summary.tsv` |

对比方法：

- `+gateA`：E156 成功基线。
- `E155_decay`：接触增强但不 fall 的对照。
- `surfaceBand-A`：E158 强 surface reward + penalty，fall。
- `surfaceBand-A2`：E159 无 penalty、`scale=1.5`，fall。

## 2. 关键时间点

| method | first `|root_z-kin_z|>8cm` | first fall (`pelvis_z<0.45m`) | terminal pelvis-z err | pelvis min |
|---|---:|---:|---:|---:|
| `+gateA` | 0.67s | none | 0.025m | 0.620m |
| `E155_decay` | 0.73s | none | 0.029m | 0.641m |
| `surfaceBand-A` | 0.97s | 2.03s | 0.554m | 0.144m |
| `surfaceBand-A2` | 0.97s | 2.13s | 0.493m | 0.154m |

解释：

- `+gateA` / `decay` 也会在早期短暂超过 8cm，但随后恢复。
- `surfaceBand-A/A2` 在 `~0.97s` 后没有恢复，`1.8s` 后 pelvis 持续下沉，最终 fall。
- A2 比 A 晚约 0.10s fall，说明降 scale 有效，但不足以从根上阻止姿态崩坏。

## 3. 分段统计

### 3.1 `+gateA` 成功对照

| interval | pelvis_z mean/min | pz_err mean/max | root_err mean/max | hand_sdf mean | cleanC3 | qpos_rew | contact_hdmi |
|---|---:|---:|---:|---:|---:|---:|---:|
| contact_early 0.55-1.20s | 0.697 / 0.620 | 0.064 / 0.105 | 0.134 / 0.167 | -0.8mm | 0.00 | 2.63 | 4.32 |
| contact_mid 1.20-1.80s | 0.782 / 0.757 | 0.023 / 0.051 | 0.390 / 0.499 | 6.3mm | 0.06 | 2.60 | 3.99 |
| fall_onset 1.80-2.25s | 0.707 / 0.682 | 0.028 / 0.053 | 0.089 / 0.233 | 7.8mm | 0.00 | 3.08 | 4.00 |
| terminal 2.25-2.50s | 0.723 / 0.685 | 0.020 / 0.033 | 0.066 / 0.081 | 72.2mm | 0.00 | 3.04 | 4.27 |

`+gateA` 接触很少，但 body tracking 可以恢复，末端也能站住。

### 3.2 E159 A2 失败段

| interval | pelvis_z mean/min | pz_err mean/max | root_err mean/max | hand_sdf mean | cleanC3 | surface_rew | qpos_rew | contact_hdmi |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| contact_early 0.55-1.20s | 0.703 / 0.670 | 0.058 / 0.091 | 0.132 / 0.178 | 5.1mm | 0.21 | 1.17 | 2.25 | 3.99 |
| contact_mid 1.20-1.80s | 0.742 / 0.716 | 0.060 / 0.084 | 0.366 / 0.458 | ~0.0mm | 0.28 | 1.14 | 1.74 | 3.33 |
| fall_onset 1.80-2.25s | 0.552 / 0.244 | 0.183 / 0.474 | 0.307 / 0.630 | ~0.1mm | 0.43 | 0.87 | 1.06 | 2.18 |
| terminal 2.25-2.50s | 0.178 / 0.154 | 0.565 / 0.608 | 0.725 / 0.755 | 0.7mm | 0.14 | 1.15 | 0.97 | 1.85 |

关键观察：

- A2 在 `contact_mid` 后 hand SDF 长期贴近 `0mm`，正好落在 reward 最优区。
- `fall_onset` 段 body tracking 已明显坏掉：`pz_err mean=0.183m`、`root_err mean=0.307m`。
- 即使 qpos/contact_hdmi reward 已经显著下降，surface reward 仍保持 `0.87-1.15`，继续奖励贴物状态。
- 这说明当前 reward 没有足够机制阻止“贴住箱体但身体下沉”的局部最优。

## 4. 视频核对

关键帧 sheet：

`workspace/core4d/results/E159/gate_surface_band_A2/diagnostics/box021_029_p2_failure/frames/box021_029_p2_frame_sheet.jpg`

视觉结论：

- `0.70s`：surface 系列比 `+gateA` 更积极贴箱。
- `1.60s`：surface 系列仍贴住箱面，姿态已比 `+gateA` 更低、更斜。
- `2.10s`：A/A2 都进入明显下沉姿态；A2 稍晚但趋势一致。
- `2.40s`：A2 已经靠箱/下沉，无法恢复站姿。

## 5. 不是主要原因的项

- 不是 hand gate 崩掉：A2 `hand_gate_valid_frac_mean=0.919`，`fallback=0`。
- 不是 object tracking 崩掉：A2 `obj_err_mean=0.00999`，略优于 `+gateA` 的 `0.01064`。
- 不是 release false：该 case A2 `releaseF3=0`。
- 不是显式 surface penalty：A2 penalty 为 0。
- 也不是单纯物理穿透：A2 相对 `+gateA` 的 `physPen3 delta=-0.053`、`geomPen2 delta=-0.147`，穿透反而下降。

## 6. 对下一轮的含义

下一轮不要只调大/调小 surface scale；问题不是单点权重，而是缺少 guard。建议优先做：

1. **Tracking-gated surface reward**：当 `root_z`/pelvis tracking error 或 pelvis z 低于阈值时，surface reward 乘衰减。目标是保留 early contact gain，但禁止 fall_onset 段继续吃 surface reward。
2. **Terminal/recovery gate**：在接触窗口后半段或 `t>1.6s`，surface reward 逐步 decay，避免末端仍贴箱。
3. **Posture guard reranking**：CEM sample selection 加 `pelvis_min_z` 或 terminal `root_z` 约束，先做 selection-level guard，比直接改 reward 更容易验证。
4. 如果做简单 sweep，`scale=1.0/0.75` 可作为对照，但应视为弱化方案，不是根因修复。

优先级：先做 tracking/posture gate，再考虑 scale sweep。
