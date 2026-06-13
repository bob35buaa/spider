# E161 — surfaceBand release ablation clean8 结果记录

日期：2026-06-13

计划文件：`workspace/core4d/plan/170_E161_surface_release_ablation_clean8_plan.md`

## 1. 实验目标

E160 的 `gateA+surfaceBand-A2+postureRerankA` 修复了 `box021_029_p2` fall，但 `box021_029_p2 releaseF3=0.75`。E161 在 clean8 上验证两个 release-side ablation：

| 方法 | 目的 |
|---|---|
| M0 `gateA+surfaceBand-A2+postureRerankA` | E160 方法扩到 clean8；复用 E160 3 case，补跑 5 case |
| M1 `+surfaceBandReleaseDecay` | 末段 15% 对 surface reward 做 tail decay |
| M2 `+surfaceBandStrictMask` | surface reward 只用当前帧 reference contact gate |

总新跑 full CEM：`5 + 8 + 8 = 21`。

## 2. 运行与修正

执行方式：

| 类型 | 入口/会话 |
|---|---|
| local full | `workspace/core4d/scripts/launch/active/run_E161_local.sh full` |
| remote full | `workspace/core4d/scripts/launch/active/run_E161_remote.sh full` |
| pull | `workspace/core4d/scripts/launch/active/pull_E161_remote_results.sh full` |
| eval | `workspace/core4d/scripts/eval/runners/eval_E161_surface_release_ablation.py full` |

中途发现并修正了一个口径问题：E161 override 最初继承 E156/E159 base，导致 box021 case 走 rotated-SDF mask，而 box004 case 走 `core4d_3cm` mask。已停止 E161 自己的首次 tmux（未动其他任务），并在 E161 builder/override 显式固定：

```text
contact_hdmi_mask_source: core4d_3cm
contact_hdmi_mask_path: <E143 clean8 mask>
contact_hdmi_mask_person_idx: <person_idx>
contact_hdmi_mask_time_axis: auto
```

修正后重跑 smoke 和 full。首条日志确认 box021/box004 均使用 `E078 core4d_3cm` mask。

提交记录：

| commit | 内容 |
|---|---|
| `862019d` | E161 计划、脚本、reward/eval plumbing |
| `dfb3b3c` | 固定 E161 contact mask source |

## 3. 完整性与产物

full 新跑完整性：

| artifact | 状态 |
|---|---|
| root result npz | 21/21 |
| full mp4 | 21/21 |
| outdir `trajectory_mjwp_act.npz` | 21/21 |
| outdir `config_act.yaml` | 21/21 |

Strict eval：

```text
E161 eval: metric_rows=62 methods=8 delta_vs_M0=16 missing=0
```

主要产物：

| 文件 | 内容 |
|---|---|
| `results/E161/surface_release_ablation/eval/full/e161_method_metrics.tsv` | per-case metrics |
| `results/E161/surface_release_ablation/eval/full/e161_method_summary.tsv` | method summary |
| `results/E161/surface_release_ablation/eval/full/e161_delta_vs_M0.tsv` | M1/M2 相对 M0 delta |
| `results/E161/surface_release_ablation/eval/full/E161_surface_release_ablation_clean8.xlsx` | XLSX 汇总表，最优黑色加粗、次优下划线 |
| `results/E161/surface_release_ablation/diagnostics/release_curves/` | release 诊断曲线 PNG/CSV |

XLSX 检查：sheets 为 `method_summary`, `per_case_metrics`, `delta_vs_omniretarget`, `delta_vs_baseline`, `delta_vs_gateA`, `delta_vs_M0`, `summary`；`method_summary` 有加粗与下划线样式。

## 4. 关键结果

### 4.1 clean8 mean

| method | tracked | fall | releaseF3 | inmaskC3 | physPen3 | geomPen2 | pz terminal |
|---|---:|---:|---:|---:|---:|---:|---:|
| M0 postureRerankA | 8/8 | 0 | 0.188 | 0.477 | 0.116 | 0.049 | 0.016 |
| M1 releaseDecay | 8/8 | 0 | 0.030 | 0.470 | 0.128 | 0.055 | 0.014 |
| M2 strictMask | 8/8 | 0 | 0.160 | 0.427 | 0.150 | 0.064 | 0.023 |

相对 M0：

| method | releaseF3 delta | inmaskC3 delta | physPen3 delta | geomPen2 delta |
|---|---:|---:|---:|---:|
| M1 releaseDecay | -0.158 | -0.007 | +0.012 | +0.007 |
| M2 strictMask | -0.028 | -0.050 | +0.034 | +0.015 |

### 4.2 Claims

| Claim | M1 releaseDecay | M2 strictMask |
|---|---|---|
| full8 | pass | pass |
| tracked 不差超过 1 case | pass, 8/8 | pass, 8/8 |
| fall 不差超过 1 case | pass, 0 | pass, 0 |
| release mean 降低至少 0.10 | pass, -0.158 | fail, -0.028 |
| `box021_029_p2 releaseF3 <= 0.25` | pass, 0.000 | pass, 0.000 |
| inmaskC3 下降不超过 0.08 | pass, -0.007 | pass, -0.050 |
| physPen3 上升不超过 0.03 | pass, +0.012 | fail, +0.034 |
| geomPen2 上升不超过 0.03 | pass, +0.007 | pass, +0.015 |

自动 summary 中 `pz_delta_le_0_03=false` 是因为 delta summary 里该项为 NaN；绝对 pz terminal mean 分别为 M0 0.016、M1 0.014、M2 0.023，均没有 tracking/fall 风险。

## 5. 诊断解释

M1 `surfaceBandReleaseDecay` 是本轮有效方案：

- clean8 `releaseF3` 从 0.188 降到 0.030。
- in-mask contact 基本保持：0.477 → 0.470。
- 物理穿透只小幅上升：physPen3 +0.012、geomPen2 +0.007。
- 8/8 tracked、0 fall。

M2 `surfaceBandStrictMask` 不如预期：

- 当前帧 gate 诊断为 0，但 `surface_band_release_rew_mean` 仍有 0.076 mean。
- 原因是 `surface_band_*_mean` 来自 CEM rollout horizon 均值；即使当前帧 reference release，horizon 内未来接触帧 reward 仍可能影响当前 selection。
- 结果上 releaseF3 只从 0.188 降到 0.160，同时 inmaskC3 下降 0.050、physPen3 上升 0.034。

诊断曲线已输出：

```text
workspace/core4d/results/E161/surface_release_ablation/diagnostics/release_curves/
```

覆盖 `box021_029_p2`、`box021_035_p2`、`box004_083_p2`、`box026_139_p1`。曲线同时画 reference contact、release window、surface reward、rollout gate mean、decay factor，以及 strictMask 的当前帧 gate。

## 6. 结论

E161 支持把 `surfaceBandReleaseDecay` 作为下一轮默认候选。它解决了 E160 最核心的 release false 问题，同时保持 clean8 的 tracking 稳定性和大部分 mask 内接触。

`surfaceBandStrictMask` 不建议继续作为主线。它说明“只关当前帧 gate”不足以消除 CEM horizon 里的 release-side 吸附，需要要么对 horizon 内 release 段整体衰减/关断，要么把 selection-level reward 分解成当前帧/未来帧不同权重。

下一步建议：

1. 以 M1 `gateA+surfaceBand-A2+postureRerankA+surfaceBandReleaseDecay` 做 clean8 视觉抽查和下游 RL handoff 候选。
2. 如继续优化 release，可做 `horizon-aware release decay`：对 rollout horizon 内 reference release 帧同步衰减 surface reward，而不只是按全局 tail time 衰减。

## 7. 追加：releaseDecay clean8 下游 RL handoff

用户确认 `surfaceBandReleaseDecay` 作为当前可用优化后，已在 E161 内追加 S6/RL handoff 导出，不新开实验编号。

新增脚本：

| 文件 | 内容 |
|---|---|
| `scripts/experiments/E161/export_releaseDecay_rl_handoff.py` | 从 E161 `variants.tsv` 与 strict metrics 读取 M1 releaseDecay clean8 源行，生成 S5/S6/RL export |
| `scripts/launch/active/run_E161_releaseDecay_rl_export.sh` | 正式导出入口，默认执行 partner OmniRetarget，允许单 case 失败并记录 |

结果根目录：

```text
workspace/core4d/results/E161/releaseDecay_rl_export/
```

关键产物：

| 文件 | 内容 |
|---|---|
| `manifest/releaseDecay_source_rows.tsv` | 8 个 releaseDecay 源行快照 |
| `s5_handoff/handoff_manifest.tsv` | S5 handoff manifest |
| `s6_downstream/evidence/downstream_evidence_manifest.tsv` | S6 CEM evidence |
| `s6_downstream/rl_export/rl_export_input.tsv` | 下游 RL 唯一输入索引 |
| `s6_downstream/rl_export/partner_omnirt/rl_partner_omnirt_manifest.tsv` | partner OmniRetarget 结果索引 |
| `summary.md` | 导出汇总 |

校验结果：

```text
rl_export_rows=8
rl_export_decision_counts={'RL_EXPORT_READY': 8}
rl_missing_required=[]
partner_rows=8
partner_status_counts={'pass': 7, 'missing_outputs': 1}
```

Partner OmniRetarget 结果：

| source | partner | status | trimmed frames |
|---|---|---|---:|
| `d003_box021_20231011_035_p1` | `box021_20231011_035_p2` | pass | 133 |
| `d003_box021_20231011_035_p2` | `box021_20231011_035_p1` | pass | 127 |
| `d003_box021_20231018_029_p2` | `box021_20231018_029_p1` | pass | 71 |
| `e091_box004_20231003_2_083_p1` | `box004_20231003_2_083_p2` | pass | 105 |
| `e091_box004_20231003_2_083_p2` | `box004_20231003_2_083_p1` | pass | 102 |
| `box023_person2` | `box023_20231008_045_p1` | pass | 134 |
| `e091_box004_20231003_2_082_p1` | `box004_20231003_2_082_p2` | missing_outputs | - |
| `e091_box026_20231023_139_p1` | `box026_20231023_139_p2` | pass | 141 |

`box004_20231003_2_082_p2` 失败原因仍是 Holosoma `robot_retarget.py` 中 CVXPY solve `infeasible`，与 E157 历史记录一致。为避免该失败阻断后续 case，E161 exporter 已改为在 `--allow-partner-failure` 下逐 source case 执行 partner OmniRetarget，最后重建完整 8 行 partner manifest 并按输出文件存在性标记 `pass/missing_outputs`。

静态与文件级检查：

```text
python3 -m py_compile workspace/core4d/scripts/experiments/E161/export_releaseDecay_rl_handoff.py
bash -n workspace/core4d/scripts/launch/active/run_E161_releaseDecay_rl_export.sh
git diff --check
```

### 7.1 方法版本字段修正

用户指出 `target_variant_id=ref_fk` 不能表达 `surfaceBandReleaseDecay` 方法版本。确认后修正如下：

- `target_variant_id` 继续只表示 target route，本实验仍为 `ref_fk`。
- `spider_method_id` 新增为 CEM/SPIDER 方法版本，本实验为 `gateA_surfaceBandA2_postureRerankA_surfaceBandReleaseDecay`。
- `source_exp_id` 新增为来源实验编号，本实验为 `E161`。

已更新通用 S6 脚本 `export_rl_inputs.py` 与 `record_downstream_evidence.py`，并重跑 E161 export 刷新结果：

```text
s5_handoff/handoff_manifest.tsv: source_exp_id=E161, spider_method_id=gateA_surfaceBandA2_postureRerankA_surfaceBandReleaseDecay, target_variant_id=ref_fk
s6_downstream/evidence/downstream_evidence_manifest.tsv: 同上
s6_downstream/rl_export/rl_export_input.tsv: 同上，8/8 RL_EXPORT_READY
```

同时更新 v3 docs 与 `.codex/skills/data-construction-v3-zh/SKILL.md`：后续 RL handoff 必须同时保留 `source_exp_id` 与 `spider_method_id`，不能把 CEM/reward 方法写入 `target_variant_id`。
