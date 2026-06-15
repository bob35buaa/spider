# E163 — narrow symmetric surfaceBand three-case results

日期：2026-06-14

计划文件：`workspace/core4d/plan/172_E163_narrow_surface_band_three_case_plan.md`

## 1. 目标

E162 确认 E161 `surfaceBandReleaseDecay` 的核心问题是 raw in-mask physical contact 退化：

```text
box023_person2:
  SPIDER+rubberhand raw contact = 0.9077
  E161 releaseDecay raw contact = 0.7538
  delta = -0.1538
```

E163 只做三 case probe，验证把 surfaceBand 收窄到 `[-1mm, +3mm]` 且使用
`exp(-abs(sdf)/sigma)` 后，能不能恢复 raw contact，同时不牺牲 tracking/fall。

## 2. 实现

核心改动：

| 文件 | 内容 |
|---|---|
| `spider/config.py` | 新增 `surface_band_score_mode: str = "one_sided"`，默认保持历史行为 |
| `spider/simulators/mjwp.py` | `one_sided` 保持旧公式；`symmetric_abs` 使用 `exp(-abs(sdf)/sigma)` |
| `scripts/experiments/E163/build_narrow_surface_band_manifest.py` | 生成三 case manifest 和 override |
| `scripts/launch/active/run_E163_local.sh` | 本地 GPU0 launcher |
| `scripts/launch/active/run_E163_remote.sh` | 远程 GPU0/GPU1 launcher |
| `scripts/launch/active/pull_E163_remote_results.sh` | 远程结果回收 |
| `scripts/eval/runners/eval_E163_narrow_surface_band.py` | 三 case RL-safe 评测 |
| `scripts/eval/wrappers/eval_E163_narrow_surface_band.sh` | 固定评测入口 |

E163 override 固定：

```text
surface_band_min_sdf_m = -0.001
surface_band_width_m = 0.003
surface_band_sigma = 0.0015
surface_band_score_mode = symmetric_abs
contact_hdmi_mask_source = core4d_3cm
```

旧 E158-E161 默认仍为 `one_sided`，不会被 E163 污染。

## 3. 运行记录

静态检查与 manifest：

```bash
python3 -m py_compile spider/config.py spider/simulators/mjwp.py \
  workspace/core4d/scripts/experiments/E163/build_narrow_surface_band_manifest.py \
  workspace/core4d/scripts/eval/runners/eval_E163_narrow_surface_band.py
bash -n workspace/core4d/scripts/launch/active/run_E163_local.sh \
  workspace/core4d/scripts/launch/active/run_E163_remote.sh \
  workspace/core4d/scripts/launch/active/pull_E163_remote_results.sh \
  workspace/core4d/scripts/eval/wrappers/eval_E163_narrow_surface_band.sh
python3 workspace/core4d/scripts/experiments/E163/build_narrow_surface_band_manifest.py
```

manifest 结果：

```text
rows=3
to_run_total=3
preflight_ok=True
split_counts={'local-gpu0': 1, 'remote-gpu0': 1, 'remote-gpu1': 1}
```

smoke：

```bash
CASE_METHODS=box023_person2:narrowSurfaceBand LOCAL_GPU=0 \
  bash workspace/core4d/scripts/launch/active/run_E163_local.sh smoke
bash workspace/core4d/scripts/eval/wrappers/eval_E163_narrow_surface_band.sh smoke --allow-missing
```

smoke 产出 root npz、`trajectory_mjwp_act.npz`、`config_act.yaml`、smoke mp4；config 和 trajectory
均确认 narrow symmetric surfaceBand plumbing 生效。

full 三卡并行：

| split | session | case |
|---|---|---|
| local GPU0 | `E163_local_full_235312` | `box023_person2` |
| remote GPU0 | `E163_full_235320` | `box021_029_p2` |
| remote GPU1 | `E163_full_235320` | `box004_083_p2` |

回收与评测：

```bash
bash workspace/core4d/scripts/launch/active/pull_E163_remote_results.sh full
bash workspace/core4d/scripts/eval/wrappers/eval_E163_narrow_surface_band.sh full
```

pull 输出：

```text
root_npz_count=3
video_count=3
```

strict eval 输出：

```text
E163 eval: stage=full e163_rows=3 all_rows=27 missing=0 e163_pass=3/3
xlsx=workspace/core4d/results/E163/narrow_surface_band/eval/full/E163_narrow_surface_band_three_case_eval.xlsx
```

## 4. 结果

输出目录：

```text
workspace/core4d/results/E163/narrow_surface_band/
```

关键文件：

| 文件 | 内容 |
|---|---|
| `eval/full/e163_case_status.tsv` | per-case 状态与核心指标 |
| `eval/full/e163_method_summary.tsv` | 方法级摘要 |
| `eval/full/e163_artifact_check.tsv` | 产物/config hard check |
| `eval/full/e163_eval_summary.json` | strict eval 摘要 |
| `eval/full/E163_narrow_surface_band_three_case_eval.xlsx` | 可读表格 |

三 case hard gate：

| case | rubberhand raw | E161 releaseDecay raw | E163 raw | delta vs rubberhand | raw 下限 | 状态 |
|---|---:|---:|---:|---:|---:|---|
| `box023_person2` | 0.9077 | 0.7538 | 0.8769 | -0.0308 | 0.8577 | 通过 |
| `box021_029_p2` | 0.3818 | 0.7818 | 0.7455 | +0.3636 | 0.3318 | 通过 |
| `box004_083_p2` | 0.5323 | 0.6129 | 0.6290 | +0.0968 | 0.4823 | 通过 |

E163 关键诊断：

| case | clean3接触 | 物理穿透3mm | 几何穿透2mm | release误接触3mm | tracked | fall |
|---|---:|---:|---:|---:|---|---|
| `box023_person2` | 0.8000 | 0.0368 | 0.0441 | 0.0000 | true | false |
| `box021_029_p2` | 0.3818 | 0.2667 | 0.0533 | 0.0000 | true | false |
| `box004_083_p2` | 0.5161 | 0.0667 | 0.0381 | 0.0000 | true | false |

方法级对比：

| 方法 | pass | mean raw接触 | worst raw delta | mean clean3 | mean 物理穿透3mm | mean 几何穿透2mm | mean releaseF3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `SPIDER+rubberhand` | 3/3 | 0.6073 | 0.0000 | 0.1180 | 0.2828 | 0.2940 | 0.0000 |
| `E161 releaseDecay` | 2/3 | 0.7162 | -0.1538 | 0.5165 | 0.1372 | 0.0467 | 0.0062 |
| `E163 narrowSurfaceBand` | 3/3 | 0.7505 | -0.0308 | 0.5660 | 0.1234 | 0.0452 | 0.0000 |

产物检查：三 case 的 root npz、full mp4、`trajectory_mjwp_act.npz`、`config_act.yaml` 均存在；
`config_act.yaml` 中 narrow band、`symmetric_abs`、`core4d_3cm` mask source 全部通过检查。
XLSX 有 `主表/逐case/E163产物检查/说明` 四个 sheet，未发现公式错误样式单元格。

视频已经生成；本轮只做产物存在性和量化评测，没有做逐帧人工 visual QC。

## 5. Claims 验证

| Claim | 结果 |
|---|---|
| C1: 恢复 `box023_person2` raw contact | pass；E163 `0.8769 >= 0.8577`，且高于 E161 `0.7538` |
| C2: 不触发 RL contact hard regression | pass；三 case raw contact delta 均 `>= -0.05`，最差 `-0.0308` |
| C3: 不牺牲基础稳定性 | pass；tracked 3/3，fall 0/3 |
| C4: penetration 清理不完全丢失 | pass/tradeoff；box023 物理穿透3mm `0.0368`，仍远低于 rubberhand `0.3456` |
| C5: 不污染历史实验 | pass；默认 `one_sided` 保持旧行为，E163 显式打开 `symmetric_abs` |

## 6. 结论

E163 三 case RL-safe probe 通过。它修复了 E161 在 `box023_person2` 上的 raw contact 回退：
`0.7538 -> 0.8769`，同时保住 tracked 3/3、fall 0/3，并保持 penetration 明显低于 rubberhand。

结论边界：E163 现在只证明三 case probe 成立，可以进入 clean8 probe；还不能直接升级为默认方法。
