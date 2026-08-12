# 五类 box Full CEM：OmniRetarget 与 PRG 七指标对比

_Core4D E197 离线统一重评 · 87 个 Full CEM case · 2026-08-12_

---

## 📋 摘要

本报告覆盖 `87` 个进入 Full CEM 的唯一 case：box001/004/021/023/024 分别为 `28/6/28/16/9`。OmniRetarget kinematic replay 与 PRG CEM rollout 使用同一 scene_act、3cm raw mask、person index 和帧域，并由 Spider 公共评测模块统一重算。

以 object-balanced macro average 为主，PRG 相对 OmniRetarget 的四项 direction-aware improvement 为：3mm in-mask 接触 `+31.7 pp`，raw in-mask 接触 `-18.1 pp`，手物 >3mm 穿透 `+33.0 pp`，lower-body 穿透 `+2.7 pp`。正 improvement 始终代表 PRG 更好。

## 🔬 方法与口径

```mermaid
flowchart LR
    accTitle: E197 Paired Evaluation Flow
    accDescr: The same 87 Full CEM cases are replayed as OmniRetarget references and PRG rollouts under a shared scene and contact mask before public-core metrics are paired and aggregated.

    authority["📥 Freeze 87 cases"] --> omni["⚙️ Replay OmniRetarget"]
    authority --> prg["⚙️ Replay PRG"]
    omni --> public_core["📊 Public-core metrics"]
    prg --> public_core
    public_core --> paired["🔗 Pair same cases"]
    paired --> report["✅ Report results"]

    classDef input fill:#ede9fe,stroke:#7c3aed,stroke-width:2px,color:#3b0764
    classDef process fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#1e3a5f
    classDef success fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#14532d

    class authority input
    class omni,prg,public_core,paired process
    class report success
```

### 指标映射

| 用户指标 | Spider 公共字段 | 方向 |
| --- | --- | --- |
| 3mm in-mask 接触 | `hand_object_physics_contact_3mm_in_mask_frac` | 越高越好 |
| Raw in-mask 接触 | `hand_object_physics_contact_in_mask_frac` | 越高越好 |
| 手物穿透 >3mm | `hand_object_physics_penetration_3mm_frame_frac` | 越低越好 |
| Lower-body 穿透 | `leg_penetration_frac` | 越低越好 |
| Foot slip max | `foot_slip_max_m` | 越低越好 |
| Object speed max | `obj_speed_max` | 越低越好 |
| Ankle jerk P95 | `ankle_jerk_p95` | 越低越好 |

前四项 physics 指标是 frame fraction；后三项 motion-health 指标分别使用 m、m/s、m/s³。`Delta = PRG − OmniRetarget`；接触的 improvement 等于 delta，其余越低越好的指标 improvement 等于 `−delta`。因此 improvement 为正时，一律表示 PRG 改善。

### 输入与转换审计

输入路径与 SHA 审计为 `87/87` pass。OmniRetarget freejoint object pose 转为 PRG scene_act 的 slide/hinge 参数时，Euler 顺序直接由 compiled object hinge axes 推导；不使用默认 `XYZ`。87 条 world-pose round-trip 最大 orientation error 为 `0.00000296°`，最大 position error 为 `1.450e-13 cm`。

## 📊 结果

### 按物体对比

下表为均值；括号内为 `PRG − OmniRetarget`。穿透 delta 为负代表 PRG 穿透更少。

| 物体 | n | 3mm in-mask 接触 | Raw in-mask 接触 | 手物 >3mm 穿透 | Lower-body 穿透 | Foot slip max | Object speed max | Ankle jerk P95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| box001 | 28 | 2.0% → 48.8% (+46.7 pp) | 96.2% → 78.9% (-17.3 pp) | 64.4% → 21.2% (-43.2 pp) | 11.1% → 7.3% (-3.8 pp) | 1.293 m → 0.910 m (-0.383 m) | 2.320 m/s → 1.259 m/s (-1.062 m/s) | 1500.2 m/s³ → 657.4 m/s³ (-842.8 m/s³) |
| box004 | 6 | 6.6% → 45.1% (+38.5 pp) | 85.7% → 69.4% (-16.2 pp) | 48.0% → 14.7% (-33.3 pp) | 4.1% → 5.6% (+1.5 pp) | 1.099 m → 0.898 m (-0.200 m) | 3.174 m/s → 1.452 m/s (-1.722 m/s) | 2162.5 m/s³ → 685.0 m/s³ (-1477.5 m/s³) |
| box021 | 28 | 22.6% → 42.6% (+20.0 pp) | 88.5% → 67.1% (-21.5 pp) | 45.9% → 17.6% (-28.3 pp) | 4.7% → 12.0% (+7.4 pp) | 1.296 m → 0.886 m (-0.410 m) | 3.685 m/s → 1.792 m/s (-1.892 m/s) | 2060.7 m/s³ → 861.9 m/s³ (-1198.8 m/s³) |
| box023 | 16 | 24.9% → 47.3% (+22.4 pp) | 88.9% → 71.2% (-17.7 pp) | 38.2% → 14.8% (-23.4 pp) | 3.1% → 3.1% (+0.0 pp) | 1.218 m → 0.733 m (-0.485 m) | 3.372 m/s → 1.792 m/s (-1.580 m/s) | 2736.0 m/s³ → 844.9 m/s³ (-1891.0 m/s³) |
| box024 | 9 | 0.1% → 30.8% (+30.7 pp) | 99.9% → 81.9% (-18.0 pp) | 74.7% → 37.8% (-36.9 pp) | 22.5% → 3.9% (-18.6 pp) | 1.271 m → 0.679 m (-0.591 m) | 3.204 m/s → 1.544 m/s (-1.661 m/s) | 2373.6 m/s³ → 779.9 m/s³ (-1593.7 m/s³) |

### 总体汇总

| 聚合 | 指标 | OmniRetarget | PRG | Delta | Improvement | 95% CI | W/T/L |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Case-weighted micro | 3mm in-mask 接触 | 13.0% | 44.4% | +31.4 pp | +31.4 pp | [+26.0 pp, +36.7 pp] | 74/1/12 |
| Case-weighted micro | Raw in-mask 接触 | 92.0% | 73.3% | -18.7 pp | -18.7 pp | [-23.0 pp, -14.3 pp] | 8/0/79 |
| Case-weighted micro | 手物穿透 >3mm | 53.6% | 20.1% | -33.4 pp | +33.4 pp | [+29.4 pp, +37.5 pp] | 84/0/3 |
| Case-weighted micro | Lower-body 穿透 | 8.2% | 7.6% | -0.6 pp | +0.6 pp | [-2.5 pp, +3.6 pp] | 34/27/26 |
| Case-weighted micro | Foot slip max | 1.264 m | 0.845 m | -0.419 m | +0.419 m | [+0.334 m, +0.508 m] | 72/0/15 |
| Case-weighted micro | Object speed max | 3.103 m/s | 1.571 m/s | -1.532 m/s | +1.532 m/s | [+1.296 m/s, +1.776 m/s] | 84/0/3 |
| Case-weighted micro | Ankle jerk P95 | 2043.9 m/s³ | 772.3 m/s³ | -1271.6 m/s³ | +1271.6 m/s³ | [+1024.1 m/s³, +1624.0 m/s³] | 87/0/0 |
| Object-balanced macro | 3mm in-mask 接触 | 11.2% | 42.9% | +31.7 pp | +31.7 pp | [+27.0 pp, +36.4 pp] | 74/1/12 |
| Object-balanced macro | Raw in-mask 接触 | 91.8% | 73.7% | -18.1 pp | -18.1 pp | [-22.3 pp, -13.7 pp] | 8/0/79 |
| Object-balanced macro | 手物穿透 >3mm | 54.2% | 21.2% | -33.0 pp | +33.0 pp | [+28.8 pp, +37.4 pp] | 84/0/3 |
| Object-balanced macro | Lower-body 穿透 | 9.1% | 6.4% | -2.7 pp | +2.7 pp | [+0.6 pp, +4.8 pp] | 34/27/26 |
| Object-balanced macro | Foot slip max | 1.235 m | 0.821 m | -0.414 m | +0.414 m | [+0.322 m, +0.506 m] | 72/0/15 |
| Object-balanced macro | Object speed max | 3.151 m/s | 1.568 m/s | -1.583 m/s | +1.583 m/s | [+1.273 m/s, +1.914 m/s] | 84/0/3 |
| Object-balanced macro | Ankle jerk P95 | 2166.6 m/s³ | 765.8 m/s³ | -1400.8 m/s³ | +1400.8 m/s³ | [+1070.9 m/s³, +1805.5 m/s³] | 87/0/0 |

## 💡 解读

- PRG 的 3mm in-mask 接触均值在 `5/5` 个物体上更高：box001, box004, box021, box023, box024
- PRG 的 raw in-mask 接触均值在 `0/5` 个物体上更高：无
- PRG 的手物 >3mm 穿透均值在 `5/5` 个物体上更低：box001, box004, box021, box023, box024
- PRG 的 lower-body 穿透均值在 `2/5` 个物体上更低：box001, box024

总体 micro average 会被 28-case 的 box001/box021 主导；跨物体判断应优先参考 macro average，逐 case 排查则使用 XLSX 的 `Case Comparison` sheet。

## 🎯 下游 RL 宽口径过滤

这是一个只读取 OmniRetarget 指标的绝对阈值候选预过滤；PRG 的任何指标都不参与 gate 判定。7 项条件均通过才标记为 `RL_CANDIDATE_OMNI_WIDE_GATE_PASS`；仍需人工 USE、partner、alignment 和下游 contract。`obj_speed_max` 保留在统一报告和 Viser 指标栏中，但用户未提供阈值，故不参与本版 gate。P05/P50/P95 仅展示 87 条 Omni reference 的分布证据。

| Gate | Omni P05 | Omni P50 | Omni P95 | Absolute threshold | Rule |
| --- | ---: | ---: | ---: | ---: | --- |
| 3mm in-mask 接触 | 0 | 0.05738 | 0.5265 | 0.01 | OmniRetarget ≥ 0.01 |
| 3mm in-mask 接触 | 0 | 0 | 0.006 | 0 | OmniRetarget ≥ 0 (box024) |
| Raw in-mask 接触 | 0.5155 | 1 | 1 | 0.5 | OmniRetarget ≥ 0.5 |
| 手物穿透 >3mm | 0.1944 | 0.5833 | 0.8003 | 0.8 | OmniRetarget ≤ 0.8 |
| Lower-body 穿透 | 0 | 0.009091 | 0.3494 | 0.3 | OmniRetarget ≤ 0.3 |
| Foot slip max | 0.6608 | 1.195 | 1.944 | 1.9 | OmniRetarget ≤ 1.9 |
| Ankle jerk P95 | 1189 | 1573 | 4115 | 4000 | OmniRetarget ≤ 4000 |

OmniRetarget 宽 gate 通过 `52/87` 条。通过 case 清单见 `e197_omni_absolute_wide_gate_filter.tsv`，其中 `RL_CANDIDATE_OMNI_WIDE_GATE_PASS` 只表示数值预筛通过，不代表可直接导出 RL。按物体通过数为 box001/004/021/023/024=`9/3/22/12/6`。最常见的过滤原因是 raw contact 未达 50%（`5` 条），其次是 foot slip 超过 1.90 m（`7` 条）。

## ⚠️ 限制

- OmniRetarget 是 kinematic reference replay，PRG 是 Full CEM dynamic rollout；本比较衡量结果差异，不单独识别 P/R/G 各组件因果贡献
- CEM 只有单 seed；bootstrap 对 case 重采样，不能估计 optimizer seed 方差
- 五物体 case 数不均，因此同时报告 case-weighted micro 与 object-balanced macro
- 本轮不改变任何人工 USE/DNU、numeric gate 或 RL export 决策

## 👁️ OmniRetarget Viser review

只读播放器已生成：`workspace/core4d/scripts/eval/review/viser_e197_omnirt_player.py`，启动 wrapper 为 `workspace/core4d/scripts/eval/wrappers/review_E197_omnirt_player.sh`。它直接加载 `omni_converted_qpos/*.npz` 和对应 scene_act XML，支持 87 个 case 的case/object/gate 筛选、逐帧播放、碰撞体显示和七项指标查看；`--check` 已审计`87/87` playable，Viser smoke server 在 `8097` 正常监听。

## 🎬 52 个宽 Gate case 的独立视频

视频审计时工作区原有视频文件数为 `0`，因此这 52 个通过 `E197-omni-absolute-wide-v4` 的 case 均无可认定的独立 Omni 视频。已补渲染 `52/52` 个 MP4，全部只使用 `omni_converted_qpos/*.npz` 与对应 `scene_act` XML，不叠加 Spider/PRG rollout 或 reference ghost。

- 视频目录：`workspace/core4d/analysis/E197_full_cem_omnirt_vs_prg_metrics/omnirt_videos/`
- 清单：`omnirt_video_manifest.tsv`（case、物体、帧数、fps、内容边界）
- 渲染摘要：`omnirt_video_render_summary.json`
- 渲染脚本：`workspace/core4d/scripts/eval/reports/render_E197_omnirt_videos.py`
- 编码检查：`52/52` 为 H.264、`720×480`、30 fps，可由 ffprobe 读取；代表帧已用 video-frames 抽查。

### 1 fps 视频视觉 QC

对上述 52 个 MP4 按 `1 fps` 抽帧检查明显摔倒和明显/大范围抖动；对动作变化点额外复核邻近帧。结果为 `52/52` 个 case 均未发现明确摔倒，`52/52` 均未发现明显/大范围抖动。该结果是视频视觉 QC，不替代数值 gate 或人工 USE/partner/alignment gate。

- 汇总报告：`omnirt_video_visual_audit.md`
- 逐 case TSV：`omnirt_video_visual_audit.tsv`
- 汇总 JSON：`omnirt_video_visual_audit_summary.json`

## 🔗 产物

- `E197_full_cem_omnirt_vs_prg_metrics.xlsx`：README、公式化 Summary、87-case 对比、174-row method metrics、输入审计与指标定义
- `e197_case_comparison.tsv`：87 条同 case 对比
- `e197_summary_by_object.tsv`：按物体、micro 与 macro 汇总
- `e197_method_metrics.tsv`：174 条统一重算 method metrics
- `e197_input_audit.tsv`：输入路径与 SHA authority
