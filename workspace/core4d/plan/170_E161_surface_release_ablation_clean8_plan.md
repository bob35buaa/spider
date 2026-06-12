# E161 — surfaceBand release ablation clean8 实验计划

日期：2026-06-13

## 0. 背景

E160 `gateA+surfaceBand-A2+postureRerankA` 在 3 个 failure-focused case 上修复了 E159 的
`box021_029_p2` fall：

```text
tracked = 3/3
fall = 0/3
box021_029_p2 terminal pelvis-z err = 0.042m
```

但它暴露了新的 release 问题：

```text
box021_029_p2 releaseF3 = 0.75
```

这个数的具体含义是：`box021_029_p2` 的 reference contact window 为 frame `16..70`，release
window 只有 frame `71..74` 共 4 帧；`releaseF3=0.75` 表示最后 4 帧中有 3 帧仍存在 3mm clean
hand-object physics contact。它不是 B1 回归，因为 E160 的 `hand_support_rew_scale=0.0`，活跃的是
`surface_band_rew`。因此 E161 不回退到 B1，而是在 E160 基础上做 **surfaceBand release-aware ablation**。

## 1. 实验目标

在 clean8 benchmark 上比较：

| 方法 | 说明 | new full run |
|---|---|---:|
| M0 `gateA+surfaceBand-A2+postureRerankA` | E160 方法扩到 clean8；已有 3 case，补 5 case | 5 |
| M1 `+surfaceBandReleaseDecay` | 在 M0 上对 surfaceBand reward 做 tail decay | 8 |
| M2 `+surfaceBandStrictMask` | 在 M0 上让 surfaceBand 在 ref mask=0 时硬关闭 | 8 |

总新跑：

```text
5 + 8 + 8 = 21 full CEM
```

运行资源仍按三卡并行：

```text
local-gpu0 + remote-gpu0 + remote-gpu1
```

## 2. clean8 benchmark

沿用 E156 clean8：

| case | 备注 |
|---|---|
| `box021_035_p1` | clean8 |
| `box021_035_p2` | clean8 |
| `box021_029_p2` | E160 已跑 M0；releaseF3=0.75 风险 case |
| `box004_083_p1` | clean8 |
| `box004_083_p2` | E160 已跑 M0；E159 release false 风险 case |
| `box023_person2` | E160 已跑 M0；正常弯腰 sanity case |
| `box004_082_p1` | clean8 / E157 RL export case |
| `box026_139_p1` | clean8 |

M0 复用 E160 已有 3 case：

```text
box021_029_p2
box004_083_p2
box023_person2
```

M0 需要补跑 5 case：

```text
box021_035_p1
box021_035_p2
box004_083_p1
box004_082_p1
box026_139_p1
```

## 3. 方法定义

### 3.1 M0: E160 full8 baseline

基于 E160：

```text
gateA + surfaceBand-A2 + postureRerankA
```

核心参数：

```text
cem_hand_gate_enabled = true
cem_hand_gate_min_sdf_m = -0.010
cem_hand_gate_max_violation_pct = 0.10
cem_hand_gate_hard_floor_m = -0.020

surface_band_rew_scale = 1.5
surface_band_penalty_scale = 0.0
surface_band_width_m = 0.030
surface_band_min_sdf_m = -0.001
surface_band_sigma = 0.015

cem_posture_gate_enabled = true
cem_posture_gate_mean_z_err_m = 0.10
cem_posture_gate_terminal_z_err_m = 0.12
cem_posture_gate_max_z_drop_m = 0.18
cem_posture_gate_terminal_frac = 0.15
cem_posture_gate_min_valid_frac = 0.05
cem_posture_gate_fallback_lambda = 5.0
```

### 3.2 M1: surfaceBandReleaseDecay

目标：保留 M0 搬运窗口内的贴面接触增益，但在 episode 尾部逐渐衰减 surfaceBand，缓解放手惯性。

新增配置建议：

```text
surface_band_decay_frac = 0.15
```

公式：

```text
if t > (1 - decay_frac) * T:
    decay_factor = (T - t) / (T * decay_frac)
else:
    decay_factor = 1

surface_band_rew *= clamp(decay_factor, 0, 1)
surface_band_penalty *= clamp(decay_factor, 0, 1)  # 当前 A2 penalty=0，保留一致语义
```

预期：

- releaseF3 比 M0 明显下降。
- 比 strict mask 更平滑，mask 边界不容易导致接触突然掉。
- 风险是尾部仍有 residual reward，`box021_029_p2` 这种 release window 很短的 case 可能降不彻底。

### 3.3 M2: surfaceBandStrictMask

目标：reference mask=0 的 release 帧完全不给 surfaceBand reward，语义更干净。

新增 gate source 建议：

```text
surface_band_gate_source = contact_mask_strict_current
```

实现要求：

- 只使用当前 timestep 的 `approach_mask_val`，不要使用 horizon 内任意未来/过去 mask。
- 当前 timestep 目标人的两手 mask 都为 0 时，`surface_band_gate=0`。
- 当前 timestep 任一目标手 mask 为 1 时，`surface_band_gate=1`。
- 增加 diagnostic 字段，便于在 release window 验证：

```text
surface_band_release_gate_mean
surface_band_release_rew_mean
surface_band_release_active_frac
```

预期：

- releaseF3 应比 M0 更明显下降，尤其是 `box021_029_p2`。
- 风险是 contact mask 边界突变，可能牺牲 mask 尾段接触或造成 CEM 选到更弱接触样本。

## 4. 运行矩阵与三卡分配

总共 21 条 new full run，均衡分到三卡，每卡 7 条。

### local-gpu0

| method | case |
|---|---|
| M0 | `box021_035_p1` |
| M0 | `box004_082_p1` |
| M1 | `box021_029_p2` |
| M1 | `box023_person2` |
| M1 | `box026_139_p1` |
| M2 | `box021_035_p2` |
| M2 | `box004_083_p2` |

### remote-gpu0

| method | case |
|---|---|
| M0 | `box021_035_p2` |
| M0 | `box026_139_p1` |
| M1 | `box021_035_p1` |
| M1 | `box004_083_p1` |
| M1 | `box004_082_p1` |
| M2 | `box021_029_p2` |
| M2 | `box023_person2` |

### remote-gpu1

| method | case |
|---|---|
| M0 | `box004_083_p1` |
| M1 | `box021_035_p2` |
| M1 | `box004_083_p2` |
| M2 | `box021_035_p1` |
| M2 | `box004_083_p1` |
| M2 | `box004_082_p1` |
| M2 | `box026_139_p1` |

## 5. 文件规划

建议在新分支执行实现和 full run：

```text
experiment/E161-surface-release-ablation
```

| 类型 | 路径 |
|---|---|
| plan | `workspace/core4d/plan/170_E161_surface_release_ablation_clean8_plan.md` |
| manifest/scripts | `workspace/core4d/scripts/experiments/E161/` |
| overrides | `examples/config/override/core4d_E161_*_{postureRerankA,releaseDecay,strictMask}.yaml` |
| CEM results | `workspace/core4d/results/E161/surface_release_ablation/cem/full/` |
| eval results | `workspace/core4d/results/E161/surface_release_ablation/eval/full/` |
| eval runner | `workspace/core4d/scripts/eval/runners/eval_E161_surface_release_ablation.py` |
| eval wrapper | `workspace/core4d/scripts/eval/wrappers/eval_E161_surface_release_ablation.sh` |
| launch | `workspace/core4d/scripts/launch/active/run_E161_local.sh`, `run_E161_remote.sh`, `pull_E161_remote_results.sh` |

Evaluator 应同表比较：

```text
OmniRetarget
spider-rubberhand
+gateA
E155_decay
gateA+surfaceBand-A2
gateA+surfaceBand-A2+postureRerankA
gateA+surfaceBand-A2+postureRerankA+surfaceBandReleaseDecay
gateA+surfaceBand-A2+postureRerankA+surfaceBandStrictMask
```

## 6. 实现范围

### 6.1 Config

新增默认关闭字段：

```text
surface_band_decay_frac: float = 0.0
```

若采用独立 gate source：

```text
surface_band_gate_source: str
```

需要扩展支持：

```text
contact_mask_strict_current
```

默认值不变，旧实验不受影响。

### 6.2 Simulator reward

在 `spider/simulators/mjwp.py` 的 `surface_band_rew` 计算后加入：

```text
surface_band_rew *= surface_band_decay_factor
surface_band_penalty *= surface_band_decay_factor
```

并输出：

```text
surface_band_decay_factor
surface_band_release_gate
surface_band_release_rew
```

M2 的 strict mask 应确保 release window 中 `surface_band_gate=0`，这是 E161 的关键 plumbing 检查。

### 6.3 Manifest / launch / eval

Manifest 需要支持三类 row：

```text
reuse_e160_m0   # E160 已有 3 case
to_run_m0       # M0 需补跑 5 case
to_run_decay    # M1 8 case
to_run_strict   # M2 8 case
```

Preflight summary 至少输出：

```text
reuse_e160 = 3
to_run_m0 = 5
to_run_decay = 8
to_run_strict = 8
to_run_total = 21
split_counts = {local-gpu0: 7, remote-gpu0: 7, remote-gpu1: 7}
```

## 7. 成功标准与 Claims

### C1: M0 clean8 baseline 补齐

M0 full8 strict eval：

```text
missing = 0
success_tracked >= 7/8
fall <= 1/8
```

如果 M0 在 clean8 上失败严重，则 M1/M2 的意义变成局部 ablation，不直接晋级。

### C2: release false 下降

相对 M0 full8：

```text
mean releaseF3 delta <= -0.10
worst-case releaseF3 <= M0 worst-case
box021_029_p2 releaseF3 <= 0.25   # 4 release frames 中最多 1 帧仍接触
```

若 M1 或 M2 只在 mean 上改善但 `box021_029_p2` 仍为 0.75，则不算解决 E160 暴露的问题。

### C3: 不牺牲 mask 内接触

相对 M0 full8：

```text
mean inmaskC3 drop <= 0.08
mean inmaskC3 still >= +gateA + 0.15
```

### C4: 不增加物理穿透

相对 M0 full8：

```text
mean physPen3 <= M0 + 0.03
mean geomPen2 <= M0 + 0.03
```

### C5: 不破坏 posture / tracking

```text
success_tracked >= M0 - 1 case
fall <= M0 + 1 case
track_pelvis_z_err_terminal_m mean <= M0 + 0.03
```

### C6: M2 gate 语义验证

对 `surfaceBandStrictMask`：

```text
release window surface_band_gate_mean <= 0.05
release window surface_band_rew_mean <= 0.05
```

如果 C6 不成立，说明 strict mask plumbing 没有真正生效，不能解释 release 指标。

## 8. 判定逻辑

| 结果 | 决策 |
|---|---|
| M1 和 M2 都满足 C2-C5 | 选 releaseF3 更低且 inmaskC3 更高者，进入 clean8 推荐候选 |
| 只有 M1 满足 | 采用 decay，说明平滑过渡比硬 mask 稳 |
| 只有 M2 满足 | 采用 strict mask，说明 release 语义硬关闭必要 |
| 两者 release 下降但接触明显掉 | 下一轮做 mask-tail ramp，而不是全尾 decay/硬关 |
| 两者 release 不降 | 需要显式 release repulsion reward：mask=0 时奖励 hand-object SDF > 3cm |

## 9. 验证流程

静态检查：

```bash
python -m py_compile spider/config.py spider/simulators/mjwp.py
python -m py_compile workspace/core4d/scripts/experiments/E161/*.py
python -m py_compile workspace/core4d/scripts/eval/runners/eval_E161_surface_release_ablation.py
bash -n workspace/core4d/scripts/launch/active/run_E161_local.sh
bash -n workspace/core4d/scripts/launch/active/run_E161_remote.sh
bash -n workspace/core4d/scripts/launch/active/pull_E161_remote_results.sh
bash -n workspace/core4d/scripts/eval/wrappers/eval_E161_surface_release_ablation.sh
```

Smoke：

1. 先跑 `box021_029_p2` 的 M1 和 M2 smoke。
2. 检查 M1 的 `surface_band_decay_factor` 尾部下降。
3. 检查 M2 的 release window `surface_band_gate/reward` 接近 0。

Full：

1. 本地 1 卡 + 远程 2 卡并行跑 21 条。
2. 回收远程结果。
3. strict eval `missing=0`。
4. 生成 TSV/JSON/XLSX，XLSX 继续使用黑色加粗表示最优、下划线表示次优。
5. 对 `box021_029_p2`、`box004_083_p2`、`box026_139_p1` 生成 release-focused 诊断曲线。

## 10. 风险

- `box021_029_p2` release window 只有 4 帧，`releaseF3` 对单帧非常敏感；必须同时看 per-frame 曲线。
- M1 tail decay 可能对短序列尾部过强，导致搬运末段接触下降。
- M2 strict mask 可能在 mask 边界造成奖励突变，触发接触掉点或 posture tradeoff。
- 如果 M0 扩到 clean8 后本身在新 5 case 上不稳定，需要先分析 M0 clean8 的 posture/fall，再解释 M1/M2。
