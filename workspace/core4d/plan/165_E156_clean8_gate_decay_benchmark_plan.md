# E156 — clean8 benchmark: spider-rubberhand / +gateA / E155_decay

## 0. 背景

E155 在 3 个 selected case 上验证了 `decay` 放手平滑策略：在 `core4d-e154-physics-contact-v1`
评测口径下，`release_false_3mm/5mm` 明显下降，同时 tracking 保持 3/3。

当前需要把 case 范围扩大到历史 clean benchmark 的 8 个 case，判断该结论是否能从 3 case 推广到
E149/E150 固定过的 `relaxed8_valid_like` benchmark。

本实验只比较 3 个方法：

| 方法 | 名称 | 定义 |
|---|---|---|
| baseline | `spider-rubberhand` | E148 rubber hand CEM 结果，不加 gate、不加 B1/decay |
| +gateA | `+gateA` | rubberhand + hand SDF CEM gate |
| E155_decay | `E155_decay` | gateA+B1 hand-support + carry-union mask + tail decay |

## 1. Benchmark

E156 使用 E149/E150 的 `relaxed8_valid_like` 8 case：

| case | 来源说明 |
|---|---|
| `box021_035_p1` | clean6 primary |
| `box021_035_p2` | clean6 primary |
| `box021_029_p2` | clean6 primary，E155 selected |
| `box004_083_p1` | clean6 primary |
| `box004_083_p2` | clean6 primary，E155 selected |
| `box023_person2` | clean6 primary，E155 selected |
| `box004_082_p1` | relaxed8 added |
| `box026_139_p1` | relaxed8 added |

所有 case 必须使用 E143 canonical contact mask：

```text
workspace/core4d/results/E143/contact_masks/<case>/raw_contact_mask_3cm.npz
```

## 2. 方法定义

### 2.1 `spider-rubberhand`

- 复用 E148 rubberhand 8/8 结果。
- 显示名固定为 `spider-rubberhand`。
- 如果实现时发现 E148 某个 artifact 缺失，只补跑缺失 baseline，不重跑已完整 case。

### 2.2 `+gateA`

- 基础配置：对应 case 的 E147/E148 rubberhand override。
- 新增 hand-object CEM gate：
  - `cem_hand_gate_enabled=true`
  - `cem_hand_gate_geom_names=["lh","rh"]`
  - `cem_hand_gate_min_sdf_m=-0.010`
  - `cem_hand_gate_max_violation_pct=0.10`
  - `+cem_hand_gate_hard_floor_m=-0.020`
- 统一使用 E153/E155 后固定的 `max_viol=0.10` 标准。
- 不复用 E152 旧纯 `gateA` 结果，因为 E152 旧阈值是 `max_viol=0.05`。

### 2.3 `E155_decay`

- 基础配置：对应 case 的 rubberhand override。
- 加入 E151 B1 hand-support reward：
  - `contact_hdmi_dynamic_target=true`
  - `contact_hdmi_target_source=ref_fk`
  - `contact_hdmi_target_uses_eef_offset=true`
  - `contact_hdmi_gain=5.0`
  - `hand_support_rew_scale=3.0`
  - `hand_support_sigma=0.015`
  - `hand_support_margin_m=0.01`
  - `hand_support_gate_source=contact_mask`
  - `hand_support_geom_names=["lh","rh"]`
- 同时加入 `+gateA` 的 hand SDF gate。
- 运行时加：
  - `+contact_hdmi_mask_carry_union=true`
  - `+hand_support_decay_frac=0.15`
  - `+cem_hand_gate_hard_floor_m=-0.020`
- 复用 E155 已有 3 case decay：`box021_029_p2`、`box004_083_p2`、`box023_person2`。

## 3. 跑量与复用

| 方法 | 8-case 覆盖 | 复用 | 新跑 |
|---|---:|---:|---:|
| `spider-rubberhand` | 8/8 | 8 | 0 |
| `+gateA` | 8/8 | 0 | 8 |
| `E155_decay` | 8/8 | 3 | 5 |

总新增 full CEM 跑量：13 条。

建议 split：

| 资源 | 任务 |
|---|---|
| local GPU0 | `box021_035_p1 gateA/decay`, `box021_035_p2 gateA/decay`, `box021_029_p2 gateA` |
| remote GPU0 | `box004_082_p1 gateA/decay`, `box004_083_p1 gateA/decay` |
| remote GPU1 | `box004_083_p2 gateA`, `box023_person2 gateA`, `box026_139_p1 gateA/decay` |

## 4. 实现计划

新增 canonical 文件：

| 类型 | 路径 |
|---|---|
| manifest/builder | `workspace/core4d/scripts/experiments/E156/` |
| full CEM 输出 | `workspace/core4d/results/E156/clean8_gate_decay/cem/full/` |
| eval 输出 | `workspace/core4d/results/E156/clean8_gate_decay/eval/full/` |
| local/remote/pull | `workspace/core4d/scripts/launch/active/run_E156_local.sh`, `run_E156_remote.sh`, `pull_E156_remote_results.sh` |
| evaluator | `workspace/core4d/scripts/eval/runners/eval_E156_clean8_gate_decay.py` |
| eval wrapper | `workspace/core4d/scripts/eval/wrappers/eval_E156_clean8_gate_decay.sh` |

实现步骤：

1. 写 E156 manifest builder，从 `workspace/core4d/scripts/experiments/E148/variants.tsv` 读取 clean8 元数据。
2. 生成 `variants.tsv`，包含 24 method rows，并标注 `reuse_e148` / `reuse_e155` / `to_run`。
3. 为 `+gateA` 和缺失的 `E155_decay` 生成 Hydra override。
4. 写 local/remote/pull 脚本；支持 `STAGE=smoke|full`、`CASE_METHODS=case:method ...`、已完成自动 skip。
5. 写 evaluator，直接 import `eval.core.core_metrics`，使用 `core4d-e154-physics-contact-v1`。
6. 输出 TSV/JSON/XLSX；XLSX 中最优黑色加粗、次优下划线。

## 5. 指标与成功标准

评测标准固定为：

```text
core4d-e154-physics-contact-v1
```

主指标：

| 指标 | 方向 |
|---|---|
| `success_tracked` | 越高越好 |
| `hand_object_release_false_contact_3mm_frac` | 越低越好 |
| `hand_object_release_false_contact_5mm_frac` | 越低越好 |
| `hand_object_physics_contact_3mm_in_mask_frac` | 越高越好 |
| `hand_object_physics_contact_5mm_in_mask_frac` | 越高越好 |
| `hand_object_physics_penetration_3mm_frame_frac` | 越低越好 |
| `hand_object_physics_penetration_5mm_frame_frac` | 越低越好 |
| `hand_geom_penetration_2mm_frac` | 越低越好 |
| `hand_geom_penetration_5mm_frac` | 越低越好 |
| `leg_penetration_frac` | 越低越好 |
| `obj_err_mean_m` | 越低越好 |

`E155_decay` 升级为 clean8 默认策略的判据：

1. `success_tracked >= 7/8`
2. 相对 `+gateA`，mean `release_false_3mm` 下降至少 0.05
3. 相对 `+gateA`，mean `inmaskC3` 下降不超过 0.10
4. 相对 `+gateA`，mean `phys_pen3` 上升不超过 0.03

若不满足，E156 仍完成 clean8 benchmark 扩展，但不把 `E155_decay` 作为默认策略。

## 6. 验证

### 6.1 静态检查

```bash
.venv/bin/python -m py_compile workspace/core4d/scripts/experiments/E156/build_clean8_gate_decay_manifest.py
.venv/bin/python -m py_compile workspace/core4d/scripts/eval/runners/eval_E156_clean8_gate_decay.py
bash -n workspace/core4d/scripts/launch/active/run_E156_local.sh
bash -n workspace/core4d/scripts/launch/active/run_E156_remote.sh
bash -n workspace/core4d/scripts/launch/active/pull_E156_remote_results.sh
bash -n workspace/core4d/scripts/eval/wrappers/eval_E156_clean8_gate_decay.sh
```

### 6.2 Preflight

- clean8 8/8 case 都能从 E148 manifest 匹配。
- baseline 8/8 artifact 存在。
- E143 contact mask 8/8 存在。
- E155 decay 3/3 复用 artifact 存在。
- `variants.tsv` 中 `to_run` 正好 13 条。

### 6.3 Smoke

先跑 1 条 `+gateA` 和 1 条 `E155_decay`：

- 日志确认 hand gate 解析到 `lh/rh`。
- `config_act.yaml` 写入 `cem_hand_gate_*`、`cem_hand_gate_hard_floor_m`。
- decay run 额外确认 `contact_hdmi_mask_carry_union=true`、`hand_support_decay_frac=0.15`。
- 产出 root npz、mp4、`trajectory_mjwp_act.npz`。

### 6.4 Full + Eval

- full CEM 新跑 13/13 complete。
- strict eval `missing=[]`。
- 输出 method summary、per-case metrics、delta vs baseline、delta vs gateA。
- XLSX 无 Excel error cells，best/second formatting 存在。

## 7. 记录与提交

实验完成后：

- 写 `workspace/core4d/log/<next>_E156_clean8_gate_decay_benchmark_results.md`
- 更新 `workspace/core4d/progress.md`
- 更新 `workspace/core4d/EXPERIMENT_TRACKER.md`
- commit 并 push 到当前分支

## 8. 范围约束

- 不改 E154+ metric 定义。
- 不改 E148/E155 既有结果。
- 不把 E152 `max_viol=0.05` 旧 gateA 混入 E156。
- 不新增除 `spider-rubberhand`、`+gateA`、`E155_decay` 之外的方法。
