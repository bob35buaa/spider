# E163 — narrow surfaceBand clean8 extension results

日期：2026-06-15

计划文件：`workspace/core4d/plan/174_E163_narrow_surface_band_clean8_extension_plan.md`

## 1. 目标

把 E163 narrow symmetric surfaceBand 从三 case probe 扩展到 E156 clean8 benchmark。

本轮保持方法名和结果根目录不变：

```text
workspace/core4d/results/E163/narrow_surface_band/
```

三条已有 full 结果复用：

```text
box023_person2
box021_029_p2
box004_083_p2
```

本轮新补五条：

```text
box021_035_p1
box021_035_p2
box004_083_p1
box004_082_p1
box026_139_p1
```

## 2. 方法

方法与 E163 三 case 完全一致，只使用 narrow symmetric surfaceBand：

```text
surface_band_min_sdf_m = -0.001
surface_band_width_m = 0.003
surface_band_sigma = 0.0015
surface_band_score_mode = symmetric_abs
surface_band_score = exp(-abs(sdf) / sigma)
surface_band_rew_scale = 1.5
surface_band_penalty_scale = 0.0
surface_band_decay_frac = 0.15
contact_hdmi_mask_source = core4d_3cm
```

## 3. 运行

预检：

```bash
python3 -m py_compile \
  workspace/core4d/scripts/experiments/E163/build_narrow_surface_band_manifest.py \
  workspace/core4d/scripts/eval/runners/eval_E163_narrow_surface_band.py
bash -n \
  workspace/core4d/scripts/launch/active/run_E163_local.sh \
  workspace/core4d/scripts/launch/active/run_E163_remote.sh \
  workspace/core4d/scripts/launch/active/pull_E163_remote_results.sh \
  workspace/core4d/scripts/eval/wrappers/eval_E163_narrow_surface_band.sh
git diff --check -- workspace/core4d/scripts/experiments/E163/build_narrow_surface_band_manifest.py \
  workspace/core4d/scripts/eval/runners/eval_E163_narrow_surface_band.py \
  workspace/core4d/scripts/launch/active/run_E163_local.sh \
  workspace/core4d/scripts/launch/active/run_E163_remote.sh \
  workspace/core4d/scripts/launch/active/pull_E163_remote_results.sh \
  workspace/core4d/scripts/eval/wrappers/eval_E163_narrow_surface_band.sh
python3 workspace/core4d/scripts/experiments/E163/build_narrow_surface_band_manifest.py
bash workspace/core4d/scripts/eval/wrappers/eval_E163_narrow_surface_band.sh clean8 --allow-missing
```

预检结果：

```text
manifest: rows=8, to_run_total=5, reuse_existing=3, preflight_ok=true
split_counts={local-gpu0:1, remote-gpu0:2, remote-gpu1:2}
allow-missing eval: e163_rows=3, all_rows=63, missing=5, ref_missing=9, e163_pass=3/8
```

启动命令均为叠加运行，`WAIT_FOR_GPU_IDLE=0`，未 kill 其他程序。

```bash
tmux new-session -d -s E163_clean8_local_013352 \
  "cd /home/ubuntu/Workspace/spider && WAIT_FOR_GPU_IDLE=0 E163_SPLIT=local-gpu0 LOCAL_GPU=0 CASE_METHODS='box021_035_p1:narrowSurfaceBand' bash workspace/core4d/scripts/launch/active/run_E163_local.sh full"

WAIT_FOR_GPU_IDLE=0 SESSION=E163_clean8_full_013352 \
REMOTE_GPU0_WORK="box021_035_p2:narrowSurfaceBand box004_083_p1:narrowSurfaceBand" \
REMOTE_GPU1_WORK="box004_082_p1:narrowSurfaceBand box026_139_p1:narrowSurfaceBand" \
bash workspace/core4d/scripts/launch/active/run_E163_remote.sh full
```

远程结果回收：

```bash
bash workspace/core4d/scripts/launch/active/pull_E163_remote_results.sh full
```

回收后本地：

```text
root npz = 8/8
full mp4 = 8/8
trajectory_mjwp_act.npz + config_act.yaml = 8/8
```

严格评测：

```bash
bash workspace/core4d/scripts/eval/wrappers/eval_E163_narrow_surface_band.sh clean8
```

输出：

```text
e163_rows=8
all_rows=68
missing=0
ref_missing=4
e163_pass=7/8
xlsx=workspace/core4d/results/E163/narrow_surface_band/eval/clean8/E163_narrow_surface_band_clean8_eval.xlsx
```

`ref_missing=4` 只来自 E158/E159 在 `box004_082_p1`、`box026_139_p1` 两个扩展 case 上没有历史结果，不影响 E147/E148 rubberhand baseline、E156/+gateA、E161 releaseDecay 和 E163 主比较。

## 4. E163 clean8 case 结果

| case | 状态 | raw contact | 下限 | Δ vs rubberhand | clean3 contact | physPen3 | geomPen2 | tracking | fall |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| `box023_person2` | 通过 | 0.8769 | 0.8577 | -0.0308 | 0.8000 | 0.0368 | 0.0441 | true | false |
| `box021_029_p2` | 通过 | 0.7455 | 0.3318 | 0.3636 | 0.3818 | 0.2667 | 0.0533 | true | false |
| `box004_083_p2` | 通过 | 0.6290 | 0.4823 | 0.0968 | 0.5161 | 0.0667 | 0.0381 | true | false |
| `box021_035_p1` | 通过 | 0.8200 | 0.1900 | 0.5800 | 0.7100 | 0.0930 | 0.0078 | true | false |
| `box021_035_p2` | 通过 | 0.7573 | 0.4063 | 0.3010 | 0.5243 | 0.1880 | 0.0602 | true | false |
| `box004_083_p1` | 通过 | 0.6349 | 0.1087 | 0.4762 | 0.5079 | 0.0784 | 0.0686 | true | false |
| `box004_082_p1` | 接触退化 | 0.5738 | 0.6221 | -0.0984 | 0.2951 | 0.1651 | 0.0734 | true | false |
| `box026_139_p1` | 通过 | 0.6711 | 0.4500 | 0.1711 | 0.6184 | 0.1127 | 0.1127 | true | false |

## 5. 方法级摘要

| 方法 | case数 | 通过case | 失败case | raw contact mean | clean3 contact mean | physPen3 mean | geomPen2 mean | releaseF3 mean | worst Δ |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| `SPIDER+rubberhand` | 8 | 8 | - | 0.4811 | 0.1490 | 0.2096 | 0.2173 | 0.0000 | 0.0000 |
| `surfaceBand releaseDecay` | 8 | 5 | `box023_person2,box004_082_p1,box026_139_p1` | 0.6374 | 0.4698 | 0.1282 | 0.0553 | 0.0302 | -0.1538 |
| `E163 narrowSurfaceBand` | 8 | 7 | `box004_082_p1` | 0.7136 | 0.5442 | 0.1259 | 0.0573 | 0.0117 | -0.0984 |

## 6. Claims

| Claim | 结果 | 说明 |
|---|---|---|
| C1 产物完整 | pass | 8/8 root npz、full mp4、outdir trajectory、config 均存在 |
| C2 raw contact hard gate | fail | `box004_082_p1` raw contact 0.5738，低于同 case rubberhand 下限 0.6221 |
| C3 tracking / fall | pass | E163 8/8 `success_tracked=true`、8/8 `fall=false`、Table4 tracking 全完整 |
| C4 box023 sanity | pass | `box023_person2` raw contact 0.8769，高于 0.8577 下限 |
| C5 secondary diagnostics | tradeoff | 均值上接触/穿透/release false 改善，但不能抵消 `box004_082_p1` raw contact fail |

## 7. 结论

E163 clean8 扩展完成，但不能直接作为 clean8 RL-safe 默认版本。

关键结论：

```text
artifact complete = 8/8
tracking pass = 8/8
fall = 0/8
raw contact hard gate = 7/8 pass
failed case = box004_082_p1
```

`box004_082_p1` 是唯一 blocker。它相对 rubberhand baseline 的 raw in-mask contact 从 0.6721 降到 0.5738，超过允许降幅 0.05；虽然 clean3/clean5 接触和穿透指标相对 rubberhand 更好，但根据 E162 后的 RL-safe 口径，这个 case 必须判接触退化。

下一步建议不要导出 clean8 RL-ready；先针对 `box004_082_p1` 做 case-specific 诊断或小范围参数修复，再重新跑该 case 和受影响 case。

## 8. 追加：downstream RL contact 口径

按 Holosoma downstream exporter 的 `object_contact` 口径追加了 RL contact 指标，并重跑了 E156 full eval 与 E163 clean8 eval。

口径来自：

```text
/home/ubuntu/Workspace/holosoma/workspace/v3/scripts/data/export_rl_motion_from_spider_tsv.py
```

等价逻辑：

```text
spider_contact_mask_3cm[:, person_idx, :]
-> max(L, R)
-> duplicate to both hands
-> fill internal false gaps with length <= 5 source frames
-> mean on the eval qpos time axis
```

新增字段：

```text
rl_object_contact_ref_frac
rl_object_contact_filled_frame_count
hand_object_physics_contact_in_rl_mask_frac
hand_object_physics_contact_3mm_in_rl_mask_frac
hand_object_physics_contact_5mm_in_rl_mask_frac
```

E163 clean8 表已更新：

```text
workspace/core4d/results/E163/narrow_surface_band/eval/clean8/E163_narrow_surface_band_clean8_eval.xlsx
```

`box004_082_p1` 结果：

| 方法 | raw contact | RL contact | RL mask占比 | RL补洞帧数 | RL Δ vs rubberhand |
|---|---:|---:|---:|---:|---:|
| `SPIDER+rubberhand` | 0.6721 | 0.6613 | 0.5688 | 1 | 0.0000 |
| `surfaceBand releaseDecay` | 0.6066 | 0.5968 | 0.5688 | 1 | -0.0645 |
| `E163 narrowSurfaceBand` | 0.5738 | 0.5645 | 0.5688 | 1 | -0.0968 |

结论不变：downstream RL mask 会补掉 `box004_082_p1` 的 1 帧短断口，但 E163 相对 rubberhand 的 RL contact 仍低 `0.0968`，超过 `0.05` hard gate。
