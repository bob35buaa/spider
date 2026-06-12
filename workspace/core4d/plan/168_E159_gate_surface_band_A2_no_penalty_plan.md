# E159 — gateA + surfaceBand-A2 no-penalty clean6 计划

## 0. 背景

E158 `gateA+surfaceBand-A` 的结论是：surface band 能强力提高 mask 内干净物理接触，并大幅降低几何穿透，但 reward 太强且带显式 penalty 后会破坏整体运动稳定性：

| 方法 | tracked | fall | inmaskC3 | physPen3 | geomPen2 | releaseF3 |
|---|---:|---:|---:|---:|---:|---:|
| `+gateA` | 6/6 | 0 | 0.147 | 0.188 | 0.194 | 0.000 |
| `E155_decay` | 6/6 | 0 | 0.346 | 0.314 | 0.386 | 0.043 |
| `gateA+surfaceBand-A` | 1/6 | 4 | 0.520 | 0.151 | 0.013 | 0.115 |

E158 证明“强 surface reward + 穿透 penalty”会把优化推向贴物/拖垮姿态。E159 只验证一个更保守的形状项：不惩罚，低权重，只在接近表面且不深穿透时给奖励。

## 1. 方法定义

E159 方法名：

```text
gateA+surfaceBand-A2
```

基于 E156 `+gateA`：

```text
cem_hand_gate_enabled=true
cem_hand_gate_geom_names=["lh","rh"]
cem_hand_gate_min_sdf_m=-0.010
cem_hand_gate_max_violation_pct=0.10
cem_hand_gate_hard_floor_m=-0.020
```

surfaceBand-A2 reward：

```text
sdf = min signed distance from hand geoms to object
gate = true 3cm contact mask

reward_band = -0.001 <= sdf <= 0.030
surface_score = exp(-max(sdf, 0) / 0.015)
surface_reward = 1.5 * gate * surface_score * 1[reward_band]

surface_penalty = 0
```

配置参数：

| 参数 | 值 | 说明 |
---|---:|---|
| `surface_band_rew_scale` | `1.5` | E158 为 `3.0`，本轮减半 |
| `surface_band_penalty_scale` | `0.0` | 取消显式穿透 penalty |
| `surface_band_width_m` | `0.030` | 外侧 30mm 内有奖励 |
| `surface_band_sigma` | `0.015` | 每远离表面 15mm，score 衰减到约 37% |
| `surface_band_min_sdf_m` | `-0.001` | 新增或等效实现：允许 -1mm 浅接触仍有奖励 |

实现注意：

- 不要写成单纯 `sdf > -1mm`；必须保留上界 `sdf <= 30mm`，避免远离物体也有奖励。
- `sdf < -1mm` 不给奖励，但也不额外惩罚。
- 物理安全仍主要依赖 `gateA` CEM hand gate，而不是 reward penalty。

## 2. Benchmark

沿用 E158 clean6，保证可直接对比：

| case | split |
|---|---|
| `box021_035_p1` | local-gpu0 |
| `box021_035_p2` | local-gpu0 |
| `box021_029_p2` | remote-gpu0 |
| `box004_083_p1` | remote-gpu0 |
| `box004_083_p2` | remote-gpu1 |
| `box023_person2` | remote-gpu1 |

对比方法：

| 方法 | 来源 |
|---|---|
| `OmniRetarget` | 复用 E156/E158 reference eval |
| `spider-rubberhand` | 复用 E156/E148 |
| `+gateA` | 复用 E156 |
| `E155_decay` | 复用 E156/E155 |
| `gateA+surfaceBand-A` | 复用 E158 |
| `gateA+surfaceBand-A2` | E159 新跑 6 条 |

## 3. Claims

E159 不追求一次性超过 E158 的接触量；目标是找一个“接触提升但不摔”的方向。

| Claim | 成功标准 |
|---|---|
| C1 tracking | `success_tracked=6/6`, `fall=0/6` |
| C2 release guard | 相对 `+gateA`，mean `releaseF3 <= +0.03` |
| C3 contact gain | 相对 `+gateA`，mean `inmaskC3 >= +0.10` |
| C4 penetration guard | 相对 `+gateA`，mean `physPen3 <= +0.05` 且 `geomPen2 <= +0.02` |
| C5 improves over decay tradeoff | `inmaskC3` 接近/超过 `E155_decay`，同时 `geomPen2` 和 `releaseF3` 明显低于 `E155_decay` |

判定：

- C1-C4 全满足：进入 clean8 复核候选。
- C1-C2 满足但 C3 不足：说明 A2 太弱，下一轮只小幅加 scale 或调 sigma。
- C3 满足但 C1/C2 失败：说明 surface reward 仍主导姿态，需要加 tracking/posture guard 或 terminal/release gating。

## 4. 实现范围

需要改动：

1. 在 `spider/config.py` 增加 `surface_band_min_sdf_m`，默认建议 `0.0` 保持 E158/Earlier 兼容。
2. 在 `spider/simulators/mjwp.py` 将 reward band 从固定 `0 <= sdf <= width` 改为：

```text
surface_band_min_sdf_m <= sdf <= surface_band_width_m
```

并使用：

```text
score = exp(-max(sdf, 0) / sigma)
```

3. 新增 E159 manifest/override/launch/eval wrapper，复用 E158 evaluator 结构，新增 A2 method。

不做：

- 不启用 B1 `hand_support_rew`。
- 不启用 E155 decay。
- 不改 E154 metrics 定义。
- 不重新跑 E156/E158 对照方法。

## 5. 文件规划

| 类型 | 路径 |
|---|---|
| plan | `workspace/core4d/plan/168_E159_gate_surface_band_A2_no_penalty_plan.md` |
| manifest/scripts | `workspace/core4d/scripts/experiments/E159/` |
| overrides | `examples/config/override/core4d_E159_*_gateA_surfaceBandA2.yaml` |
| CEM results | `workspace/core4d/results/E159/gate_surface_band_A2/cem/full/` |
| eval results | `workspace/core4d/results/E159/gate_surface_band_A2/eval/full/` |
| eval runner | `workspace/core4d/scripts/eval/runners/eval_E159_gate_surface_band_A2.py` |
| eval wrapper | `workspace/core4d/scripts/eval/wrappers/eval_E159_gate_surface_band_A2.sh` |
| launch | `workspace/core4d/scripts/launch/active/run_E159_local.sh`, `run_E159_remote.sh`, `pull_E159_remote_results.sh` |

## 6. 验证流程

静态检查：

```bash
python -m py_compile spider/config.py spider/simulators/mjwp.py
python -m py_compile workspace/core4d/scripts/experiments/E159/*.py
python -m py_compile workspace/core4d/scripts/eval/runners/eval_E159_gate_surface_band_A2.py
bash -n workspace/core4d/scripts/launch/active/run_E159_local.sh
bash -n workspace/core4d/scripts/launch/active/run_E159_remote.sh
bash -n workspace/core4d/scripts/launch/active/pull_E159_remote_results.sh
bash -n workspace/core4d/scripts/eval/wrappers/eval_E159_gate_surface_band_A2.sh
```

Smoke：

- 跑 `box021_035_p1` 单 case smoke。
- 检查 `surface_band_penalty_mean` 应为 0 或接近 0。
- 检查 `surface_band_rew_mean` 非零，且 `surface_band_sdf_mean` 不被推到深负值。

Full：

- clean6 6/6 CEM complete。
- strict eval `missing=0`。
- 输出 TSV/JSON/XLSX。

## 7. 预期风险

- 即使取消 penalty，surface reward 仍可能导致 sticky release；因此 release false 是硬门槛。
- `scale=1.5` 仍可能过强；如果 fall 仍明显，下一轮应先降到 `0.5/1.0`，而不是改回 penalty。
- A2 若 contact gain 不足，但 tracking 稳定，则说明方向有效，后续再做小范围 scale/sigma sweep。
