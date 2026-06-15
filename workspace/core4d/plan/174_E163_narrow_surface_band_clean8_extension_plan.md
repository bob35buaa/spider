# E163 — narrow surfaceBand clean8 extension plan

日期：2026-06-15

## 0. 背景

E163 narrow symmetric surfaceBand 三 case probe 已通过：

```text
box023_person2 raw contact: E161 0.7538 -> E163 0.8769
E163 three-case: pass 3/3, missing=0, tracked 3/3, fall 0/3
```

E163 三 case 已导出下游 RL-ready 数据，`RL_EXPORT_READY=3/3`、partner OmniRetarget `3/3 pass`。

下一步把同一个 E163 方法扩展到 E156 clean8 benchmark。用户要求：

```text
仍命名为 E163
只补缺的 5 条 CEM
多卡并行
叠加运行
不要 kill 其他程序
```

本计划只设计实验，不启动 CEM。

## 1. 实验范围

clean8 全量 case：

| case | E163 状态 | 动作 |
|---|---|---|
| `box023_person2` | 已完成 full | reuse |
| `box021_029_p2` | 已完成 full | reuse |
| `box004_083_p2` | 已完成 full | reuse |
| `box021_035_p1` | 缺 E163 | 新跑 |
| `box021_035_p2` | 缺 E163 | 新跑 |
| `box004_083_p1` | 缺 E163 | 新跑 |
| `box004_082_p1` | 缺 E163 | 新跑 |
| `box026_139_p1` | 缺 E163 | 新跑 |

补跑数：

```text
total clean8 rows = 8
reuse existing E163 rows = 3
new CEM rows = 5
```

结果仍归档在：

```text
workspace/core4d/results/E163/narrow_surface_band/
```

已有三 case 评测表 `eval/full/` 保持不动；clean8 扩展新出：

```text
workspace/core4d/results/E163/narrow_surface_band/eval/clean8/
```

## 2. 方法定义

与 E163 三 case 完全一致，不改 reward：

```text
base = E161 surfaceBandReleaseDecay stack
change = narrow symmetric surfaceBand only

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

保持 E163 三 case 的其他 gate：

```text
cem_hand_gate_min_sdf_m = -0.010
cem_hand_gate_max_violation_pct = 0.10
cem_hand_gate_hard_floor_m = -0.020

cem_posture_gate_mean_z_err_m = 0.10
cem_posture_gate_terminal_z_err_m = 0.12
cem_posture_gate_max_z_drop_m = 0.18
cem_posture_gate_terminal_frac = 0.15
cem_posture_gate_min_valid_frac = 0.05
cem_posture_gate_fallback_lambda = 5.0
```

默认历史行为仍保持 `surface_band_score_mode=one_sided`；只有 E163 override 显式打开 `symmetric_abs`。

## 3. 代码与脚本改动

### 3.1 Manifest builder

修改：

```text
workspace/core4d/scripts/experiments/E163/build_narrow_surface_band_manifest.py
```

要求：

1. `TARGET_CASES` 扩展到 clean8。
2. 已完成三 case 如果 full artifacts 齐全，`run_status` 标为 `reuse_e163_full`。
3. 缺的五 case 标为 `to_run`。
4. summary 里报告：

```text
method_rows = 8
reuse_existing = 3
to_run_total = 5
preflight_ok = true
split_counts = local/remote0/remote1
```

### 3.2 Split 分配

叠加运行，不等待 GPU idle，不 kill 现有程序。

建议分配：

| split | case | 说明 |
|---|---|---|
| local-gpu0 | `box021_035_p1` | 本地跑一条，方便快速看日志 |
| remote-gpu0 | `box021_035_p2` | clean8 box021 pair |
| remote-gpu0 | `box004_083_p1` | box004 p1 |
| remote-gpu1 | `box004_082_p1` | E161 曾 raw contact fail 的 case，需要补 E163 |
| remote-gpu1 | `box026_139_p1` | clean8 额外 case |

已完成三 case 的 split 可保留原值或标 `reuse`，但 runner 不能重新跑它们。

### 3.3 Launcher

复用并小改现有：

```text
workspace/core4d/scripts/launch/active/run_E163_local.sh
workspace/core4d/scripts/launch/active/run_E163_remote.sh
workspace/core4d/scripts/launch/active/pull_E163_remote_results.sh
```

要求：

1. `run_E163_local.sh full` 只选择 `run_status=to_run`。
2. `is_complete()` 继续作为防重跑保护。
3. `WAIT_FOR_GPU_IDLE=0` 是本轮正式启动命令的默认要求。
4. 不执行任何 `tmux kill-session`、`pkill`、`nvidia-smi --gpu-reset` 或类似 kill 动作。
5. 如果 OOM 或资源冲突，记录失败并回报，不杀其他程序。

### 3.4 Evaluator

修改：

```text
workspace/core4d/scripts/eval/runners/eval_E163_narrow_surface_band.py
workspace/core4d/scripts/eval/wrappers/eval_E163_narrow_surface_band.sh
```

要求：

1. clean8 target cases 从 E163 `variants.tsv` 或显式 clean8 list 读取。
2. 新输出目录为：

```text
workspace/core4d/results/E163/narrow_surface_band/eval/clean8/
```

3. 保留原 `eval/full/` 三 case 结果，不覆盖。
4. 新 workbook 命名：

```text
E163_narrow_surface_band_clean8_eval.xlsx
```

5. 对比方法至少包含：

```text
OmniRetarget
SPIDER+rubberhand
+gateA
E155 decay
E161 releaseDecay
E163 narrowSurfaceBand
```

E158/E159/E160 也可继续保留为参考，但主结论以 E163 vs `SPIDER+rubberhand` 的 raw contact hard gate 为准。

## 4. 成功判据

### C1: 产物完整

8 个 E163 rows 必须都有：

```text
root npz
trajectory_mjwp_act.npz
config_act.yaml
full mp4
```

其中五条新跑产物必须由本轮产生；三条旧产物允许 reuse。

### C2: raw contact hard gate

每个 case 使用同 case `SPIDER+rubberhand` baseline：

```text
hand_object_physics_contact_in_mask_frac_delta_vs_rubberhand >= -0.05
```

任何 case 低于 `-0.05`，该 case 直接判 `contact_regression_fail`，不能由 clean3 接触、penetration、release false 或 tracking 抵消。

### C3: tracking / fall

```text
success_tracked = true for 8/8
fall = false for 8/8
Table4 tracking fields non-NaN for 8/8
```

### C4: box023 sanity

旧关键 case 必须保持：

```text
box023_person2 raw contact >= 0.8577
```

如果 clean8 evaluator 里 box023 变低，先查复用路径/评测口径，不接受静默覆盖旧结果。

### C5: secondary diagnostics

以下只做诊断，不抵消 raw contact fail：

```text
clean3 / clean5 contact
physPen3 / physPen5
geomPen2 / geomPen5
release false contact 3mm/5mm
Table4 tracking
gate/posture health
```

如果 E163 clean8 raw contact 全过，但 release false 或 penetration 比 E161 releaseDecay 明显退化，要标记为 tradeoff，不直接 promote。

## 5. 运行命令

### 5.1 Preflight

```bash
python3 workspace/core4d/scripts/experiments/E163/build_narrow_surface_band_manifest.py
python3 -m py_compile \
  workspace/core4d/scripts/experiments/E163/build_narrow_surface_band_manifest.py \
  workspace/core4d/scripts/eval/runners/eval_E163_narrow_surface_band.py
bash -n \
  workspace/core4d/scripts/launch/active/run_E163_local.sh \
  workspace/core4d/scripts/launch/active/run_E163_remote.sh \
  workspace/core4d/scripts/launch/active/pull_E163_remote_results.sh \
  workspace/core4d/scripts/eval/wrappers/eval_E163_narrow_surface_band.sh
```

预期：

```text
rows=8
to_run_total=5
reuse_existing=3
preflight_ok=true
```

### 5.2 Allow-missing eval

正式启动前跑一次 allow-missing：

```bash
bash workspace/core4d/scripts/eval/wrappers/eval_E163_narrow_surface_band.sh clean8 --allow-missing
```

预期只缺五条新 E163 rows。

### 5.3 叠加启动 full

本地：

```bash
WAIT_FOR_GPU_IDLE=0 \
E163_SPLIT=local-gpu0 \
LOCAL_GPU=0 \
CASE_METHODS="box021_035_p1:narrowSurfaceBand" \
bash workspace/core4d/scripts/launch/active/run_E163_local.sh full
```

远程：

```bash
WAIT_FOR_GPU_IDLE=0 \
SESSION=E163_clean8_full_$(date +%H%M%S) \
REMOTE_GPU0_WORK="box021_035_p2:narrowSurfaceBand box004_083_p1:narrowSurfaceBand" \
REMOTE_GPU1_WORK="box004_082_p1:narrowSurfaceBand box026_139_p1:narrowSurfaceBand" \
bash workspace/core4d/scripts/launch/active/run_E163_remote.sh full
```

注意：

```text
不 kill 其他程序
不等待 GPU idle
如果已有训练/CEM 占用 GPU，本轮按用户要求叠加运行
```

### 5.4 回收与 strict eval

```bash
bash workspace/core4d/scripts/launch/active/pull_E163_remote_results.sh full
bash workspace/core4d/scripts/eval/wrappers/eval_E163_narrow_surface_band.sh clean8
```

strict eval 预期：

```text
e163_rows=8
missing=0
e163_pass_cases reported out of 8
```

## 6. 输出

新增或更新：

| 路径 | 内容 |
|---|---|
| `workspace/core4d/scripts/experiments/E163/variants.tsv` | clean8 manifest，8 rows |
| `workspace/core4d/results/E163/narrow_surface_band/preflight/` | clean8 preflight |
| `workspace/core4d/results/E163/narrow_surface_band/cem/full/` | 3 old + 5 new full CEM artifacts |
| `workspace/core4d/results/E163/narrow_surface_band/eval/clean8/` | clean8 strict eval |
| `logs/E163/cem/full/` | local/remote CEM logs |

完成后写新结果 log：

```text
workspace/core4d/log/208_E163_narrow_surface_band_clean8_results.md
```

并更新：

```text
workspace/core4d/progress.md
workspace/core4d/EXPERIMENT_TRACKER.md
workspace/core4d/log/INDEX.md
```

## 7. 下一步边界

本轮只做 clean8 benchmark，不自动刷新 RL export。

如果 clean8 8/8 raw contact hard gate + tracking/fall 全过，再单独把 E163 RL export 从 3 rows 扩到 clean8 8 rows。
