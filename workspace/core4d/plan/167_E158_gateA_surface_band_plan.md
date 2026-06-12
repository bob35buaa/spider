# E158 — gateA + surfaceBand-A clean6 实验计划

## 0. 背景

E156 clean8 显示 `spider-rubberhand`、`+gateA`、`E155_decay` 都明显优于 `OmniRetarget` 的物理穿透，
但离“高物理接触 + 低物理穿透”仍不够。`+gateA` 是当前默认候选，但它只是 CEM sample filter，
不能主动生成干净表面接触；`E155_decay` 能显著提升 mask 内接触，却同步增加 release false contact
和 3mm 物理穿透，说明它仍在用吸附/穿透换接触。

E151/E156 暴露的核心问题是 B1 `hand_support_rew` 使用 `abs(sdf)` 近零奖励，外侧贴面和浅穿透同分。
E158 改为 one-sided `surfaceBand-A`：只奖励物体外侧近表面，并对超过 3mm 的穿透直接惩罚。

## 1. Benchmark

本轮只跑 E156 clean6 primary，保证与 E156 已有四方法对比完全可复用：

| case | 备注 |
|---|---|
| `box021_035_p1` | clean6 primary |
| `box021_035_p2` | clean6 primary |
| `box021_029_p2` | clean6 primary / E155 selected |
| `box004_083_p1` | clean6 primary |
| `box004_083_p2` | clean6 primary / E155 selected |
| `box023_person2` | clean6 primary / E155 selected |

对比方法：

| 方法 | 来源 |
|---|---|
| `OmniRetarget` | 复用 E156 reference 评测逻辑 |
| `spider-rubberhand` | 复用 E156/E148 rubberhand |
| `+gateA` | 复用 E156 full |
| `E155_decay` | 复用 E156/E155 full |
| `gateA+surfaceBand-A` | E158 新跑 6 条 |

## 2. 方法定义

`surfaceBand-A` 使用 `["lh","rh"]` hand geom 到 object 的最小 signed distance：

```text
sdf = min signed distance from hand geoms to object
gate = true 3cm contact mask

reward band: 0 <= sdf <= 0.030
surface_score = exp(-sdf / 0.015) inside reward band, else 0
surface_reward = 3.0 * gate * surface_score

penetration_penalty = -200.0 * gate * max(-sdf - 0.003, 0)
```

叠加 E156 `+gateA`：

```text
cem_hand_gate_enabled=true
cem_hand_gate_geom_names=["lh","rh"]
cem_hand_gate_min_sdf_m=-0.010
cem_hand_gate_max_violation_pct=0.10
cem_hand_gate_hard_floor_m=-0.020
```

不启用 B1 `hand_support_rew`，不启用 `E155_decay`。

## 3. Stage A：诊断可视化

先不跑 CEM，只读取 E156 clean6 的 `spider-rubberhand`、`+gateA`、`E155_decay` 轨迹，输出：

1. per-case time curve：`sdf_min(t)`、真实 contact mask、`0mm`、`-3mm`、`-5mm`、`+30mm` 阈值线。
2. per-case method comparison：同一 case 下三方法 `sdf_min(t)` 对齐比较。
3. stacked band fraction：mask 内和 release 段分别统计：
   - `far`: `sdf > 30mm`
   - `surface`: `0 <= sdf <= 30mm`
   - `shallow`: `-3mm <= sdf < 0`
   - `pen3`: `-5mm <= sdf < -3mm`
   - `deep`: `sdf < -5mm`

输出目录：

```text
workspace/core4d/results/E158/gate_surface_band/diagnostics/
```

## 4. Stage B：Full CEM

新增 6 条 full CEM：

| 资源 | 任务 |
|---|---|
| local GPU0 | `box021_035_p1`, `box021_035_p2` |
| remote GPU0 | `box021_029_p2`, `box004_083_p1` |
| remote GPU1 | `box004_083_p2`, `box023_person2` |

输出：

```text
workspace/core4d/results/E158/gate_surface_band/cem/full/
```

## 5. 指标与成功标准

固定使用：

```text
core4d-e154-physics-contact-v1
```

必须满足：

| Claim | 标准 |
|---|---|
| C1 tracking | `success_tracked = 6/6` |
| C2 penetration guard | 相对 `+gateA`，`physPen3 <= +0.02` 且 `pen2 <= +0.02` |
| C3 release guard | 相对 `+gateA`，`release_false3 <= +0.03` |
| C4 contact gain | 相对 `+gateA`，`inmaskC3 >= +0.05` |

判定：

- C1-C4 全满足：进入 clean8 复核候选。
- C1-C3 满足但 C4 不满足：物理合规有效但接触驱动不足，下一轮加大 band reward 或改 selection ranking。
- C4 满足但 C2/C3 失败：仍在用穿透或 release sticky 换接触，下一轮加强 penalty 或做 Pareto selection。

## 6. 实现文件

| 类型 | 路径 |
|---|---|
| manifest/builder | `workspace/core4d/scripts/experiments/E158/` |
| diagnostics | `workspace/core4d/scripts/experiments/E158/diagnose_e156_surface_band.py` |
| full CEM | `workspace/core4d/results/E158/gate_surface_band/cem/full/` |
| eval | `workspace/core4d/results/E158/gate_surface_band/eval/full/` |
| local/remote/pull | `workspace/core4d/scripts/launch/active/run_E158_{local,remote}.sh`, `pull_E158_remote_results.sh` |
| evaluator | `workspace/core4d/scripts/eval/runners/eval_E158_gate_surface_band.py` |
| eval wrapper | `workspace/core4d/scripts/eval/wrappers/eval_E158_gate_surface_band.sh` |

## 7. 验证

静态检查：

```bash
.venv/bin/python -m py_compile spider/config.py spider/simulators/mjwp.py
.venv/bin/python -m py_compile workspace/core4d/scripts/experiments/E158/*.py
.venv/bin/python -m py_compile workspace/core4d/scripts/eval/runners/eval_E158_gate_surface_band.py
bash -n workspace/core4d/scripts/launch/active/run_E158_local.sh
bash -n workspace/core4d/scripts/launch/active/run_E158_remote.sh
bash -n workspace/core4d/scripts/launch/active/pull_E158_remote_results.sh
bash -n workspace/core4d/scripts/eval/wrappers/eval_E158_gate_surface_band.sh
```

Preflight：

- clean6 E156 对照 artifact 6/6 完整。
- E143 contact mask 6/6 存在。
- E158 override 6/6 写入 `surface_band_*` 与 `cem_hand_gate_*`。

Smoke：

- 先跑 `box021_029_p2` 和 `box004_083_p2` 各 1 条。
- 检查 `surface_band_rew`、`surface_band_penalty` 非零，`hand_gate_valid_frac` 不塌。

Full：

- 新增 CEM 6/6 complete。
- strict eval method rows = 30 行。
- 生成 TSV/JSON/XLSX 和结果 log。

## 8. 范围约束

- 不修改 E154/E156 指标定义。
- 不重跑 E156 四个对照方法。
- 不跑 surfaceBand-B/tolerant 或 B1+penalty 修补版。
- 不把 E158 结论直接推广到 clean8；clean8 复核另开后续实验。
