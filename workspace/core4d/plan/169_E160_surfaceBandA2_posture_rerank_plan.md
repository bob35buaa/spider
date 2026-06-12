# E160 — gateA + surfaceBand-A2 + posture rerank 计划

## 0. 分支与背景

当前分支：

```text
experiment/E160-posture-rerank
```

E159 `gateA+surfaceBand-A2` 结论是：A2 比 E158 A 稳定很多，但 `box021_029_p2` 仍 fall，`box004_083_p2` 仍有 release false。针对 `box021_029_p2` 的 failure analysis 表明：

- fall 不是 object tracking、hand gate、release false 或穿透导致。
- 根因是 surface reward 持续奖励 hand SDF≈0mm 的贴物状态，使 CEM 接受明显变差的 body/qpos tracking。
- 绝对 `pelvis_z > 0.5m` 不适合作为统一安全门，因为有些 case 的参考动作本来就会大幅弯腰。

因此 E160 不用绝对 pelvis 高度门槛，而是在 **CEM elite selection** 阶段增加“相对参考轨迹”的 posture rerank / safety gate。

## 1. 方法定义

E160 方法名：

```text
gateA+surfaceBand-A2+postureRerankA
```

基于 E159 A2：

```text
surface_band_rew_scale = 1.5
surface_band_penalty_scale = 0.0
surface_band_width_m = 0.030
surface_band_min_sdf_m = -0.001
surface_band_sigma = 0.015
```

保留 E156/E159 hand gate：

```text
cem_hand_gate_enabled = true
cem_hand_gate_min_sdf_m = -0.010
cem_hand_gate_max_violation_pct = 0.10
cem_hand_gate_hard_floor_m = -0.020
```

新增 sample-level posture gate，只影响 CEM elite selection，不直接改每 timestep reward：

```text
z_err[t] = abs(sim_root_z[t] - ref_root_z[t])
z_drop[t] = ref_root_z[t] - sim_root_z[t]

mean_z_err = mean_t(z_err[t])
terminal_z_err = mean_last_15pct(z_err[t])
max_z_drop = max_t(z_drop[t])
```

valid 条件：

```text
mean_z_err <= 0.10m
terminal_z_err <= 0.12m
max_z_drop <= 0.18m
```

这里的 `max_z_drop` 只惩罚 sim 比 ref 低太多，不惩罚 ref 本身大幅弯腰。

最终 elite selection mask：

```text
sample_valid =
  hand_gate_valid
  AND posture_gate_valid
```

## 2. Fallback 规则

如果 posture + hand gate 后 valid sample 太少，不直接使 CEM 失败，使用 violation rerank：

```text
posture_violation =
    relu(mean_z_err - 0.10) / 0.05
  + relu(terminal_z_err - 0.12) / 0.05
  + relu(max_z_drop - 0.18) / 0.05

fallback_score = reward - 5.0 * posture_violation
```

初始参数：

| 参数 | 值 | 说明 |
|---|---:|---|
| `cem_posture_gate_enabled` | `true` | E160 新增 |
| `cem_posture_gate_mean_z_err_m` | `0.10` | 全 horizon root-z tracking |
| `cem_posture_gate_terminal_z_err_m` | `0.12` | last 15% root-z tracking |
| `cem_posture_gate_max_z_drop_m` | `0.18` | sim 比 ref 低的最大允许量 |
| `cem_posture_gate_terminal_frac` | `0.15` | terminal window |
| `cem_posture_gate_min_valid_frac` | 复用或对齐 `cem_safety_gate_min_valid_frac` | valid sample 太少时 fallback |
| `cem_posture_gate_fallback_lambda` | `5.0` | fallback rerank penalty |

## 3. Benchmark cases

本轮只跑 failure-focused 3 case，不直接 clean6：

| user alias | canonical case id | 目的 |
|---|---|---|
| `box021_029_p2` | `box021_029_p2` | E159 唯一 fall case，首要验证 |
| `box004_083_p2` | `box004_083_p2` | E159 release false case，检查 rerank 是否破坏/改善 release |
| `box023_p2` | `box023_person2` | E159 成功且 tracking 很好，用作不过度约束的 sanity case |

对比方法：

| 方法 | 来源 |
|---|---|
| `+gateA` | E156 |
| `E155_decay` | E156/E155 |
| `gateA+surfaceBand-A2` | E159 |
| `gateA+surfaceBand-A2+postureRerankA` | E160 新跑 |

## 4. Claims

| Claim | 成功标准 |
|---|---|
| C1 修复 box021 fall | `box021_029_p2` `success_tracked=true`, `fall=false`, `track_pelvis_z_err_terminal_m <= 0.08` |
| C2 不牺牲接触 | E160 在 3 case mean 上，`inmaskC3` 相对 E159 A2 下降不超过 `0.10` |
| C3 不增加穿透 | E160 在 3 case mean 上，`physPen3` 与 `geomPen2` 不高于 E159 A2 `+0.03` |
| C4 不恶化 release | `box004_083_p2` `releaseF3 <= E159 A2`，最好降到 `0` |
| C5 不误杀正常弯腰 | `box023_person2` 仍 `success_tracked=true`，且 posture fallback 不应长期触发 |

判定：

- C1-C5 全满足：进入 clean6 复核候选。
- C1 满足但 C2 明显下降：posture gate 太强，下一轮放宽阈值或只做 terminal gate。
- C1 不满足：说明 selection-level rerank 不足，需 reward-side tracking gate 或更强 terminal/recovery decay。
- C5 失败：说明相对 ref gate 实现或阈值仍误杀大弯腰，需要优先修 gate，而不是调 surface reward。

## 5. 实现范围

### 5.1 配置

在 `spider/config.py` 新增默认关闭字段：

```text
cem_posture_gate_enabled: bool = False
cem_posture_gate_mean_z_err_m: float = 0.10
cem_posture_gate_terminal_z_err_m: float = 0.12
cem_posture_gate_max_z_drop_m: float = 0.18
cem_posture_gate_terminal_frac: float = 0.15
cem_posture_gate_min_valid_frac: float = 0.05
cem_posture_gate_fallback_lambda: float = 5.0
```

默认关闭，保证旧实验不受影响。

### 5.2 Rollout info

在 CEM rollout 内从 sample qpos 和 ref qpos 计算：

```text
cem_posture_z_err
cem_posture_z_drop
```

或直接在 `sampling.py` / `sampling_fast.py` 的 `info_combined` 阶段从 recorded qpos/ref 计算 sample-level fields：

```text
sample_posture_mean_z_err
sample_posture_terminal_z_err
sample_posture_max_z_drop
sample_posture_violation
sample_posture_valid_mask
```

实现应同时覆盖普通 `sampling.py` 和 deterministic `sampling_fast.py`，避免 fast/full 路径不一致。

### 5.3 Elite selection

扩展现有 `_compute_sample_gate_info()`：

```text
sample_gate_valid_mask =
  existing_body_hand_gate
  AND sample_posture_valid_mask
```

fallback 时使用：

```text
fallback_score = reward - cem_posture_gate_fallback_lambda * sample_posture_violation
```

注意：

- 不要把绝对 `pelvis_z > 0.5` 写死。
- 不要修改 E159 surface reward 公式。
- 不要改变 E154+ metrics 口径。

### 5.4 Health fields

trajectory 中至少记录：

```text
cem_posture_gate_valid_frac
cem_posture_gate_selected_valid_frac
cem_posture_gate_fallback_used
cem_posture_mean_z_err_mean
cem_posture_terminal_z_err_mean
cem_posture_max_z_drop_mean
cem_posture_violation_mean
```

评估表应纳入这些 health fields，方便判断 gate 是否过紧。

## 6. 文件规划

| 类型 | 路径 |
|---|---|
| plan | `workspace/core4d/plan/169_E160_surfaceBandA2_posture_rerank_plan.md` |
| manifest/scripts | `workspace/core4d/scripts/experiments/E160/` |
| overrides | `examples/config/override/core4d_E160_*_surfaceBandA2_postureRerankA.yaml` |
| CEM results | `workspace/core4d/results/E160/posture_rerank/cem/full/` |
| eval results | `workspace/core4d/results/E160/posture_rerank/eval/full/` |
| eval runner | `workspace/core4d/scripts/eval/runners/eval_E160_posture_rerank.py` |
| eval wrapper | `workspace/core4d/scripts/eval/wrappers/eval_E160_posture_rerank.sh` |
| launch | `workspace/core4d/scripts/launch/active/run_E160_local.sh`, `run_E160_remote.sh`, `pull_E160_remote_results.sh` |

## 7. 验证流程

静态检查：

```bash
python -m py_compile spider/config.py spider/optimizers/sampling.py spider/optimizers/sampling_fast.py
python -m py_compile workspace/core4d/scripts/experiments/E160/*.py
python -m py_compile workspace/core4d/scripts/eval/runners/eval_E160_posture_rerank.py
bash -n workspace/core4d/scripts/launch/active/run_E160_local.sh
bash -n workspace/core4d/scripts/launch/active/run_E160_remote.sh
bash -n workspace/core4d/scripts/launch/active/pull_E160_remote_results.sh
bash -n workspace/core4d/scripts/eval/wrappers/eval_E160_posture_rerank.sh
```

Smoke：

- 先跑 `box021_029_p2` smoke。
- 检查 `cem_posture_gate_valid_frac` 非 0，`fallback` 不应全程触发。
- 检查 trajectory config 中 E160 posture gate 参数已写入。

Full：

- 3 case full complete。
- strict eval `missing=0`。
- 输出 TSV/JSON/XLSX，对比 E156/E159。

## 8. 预期风险

- 阈值太紧会降低 surface contact gain，特别是 `box023_person2` 这种弯腰幅度较大的 sanity case。
- 阈值太松则无法阻止 `box021_029_p2` fall。
- fallback penalty 如果太小，bad posture sample 仍可能进 elite；如果太大，会退化成只做 tracking。

优先保持 E160 是小而可诊断的 3-case 实验，不直接扩 clean6。
