# E023 实验计划：retarget kinematic geometry repair

日期：2026-05-20

## Context

E020 将 `box025_p1` 与 `bucket007_p2` 归为 `retarget_kinematic`：object-side E018b support proxy 已经成立，但 kinematic reference 本身有高比例腿/物体干涉。

本轮用户要求忽略 `desk021_p1` 和 `box021_*`，并按 cause 分组优化剩余 case。E023 只处理 ref-side lower-body geometry 问题，不做 controller reward sweep。

### Baseline evidence

| Case | E018b task | Anchor | E020 ref leg/object interference | Ref hand contact |
|---|---|---|---:|---:|
| `box025_p1` | `box025_person1_freejoint_legobj_e018b` | `-x`, `[-0.3768, 0, 0.290904]` | `66.53%` | `77.42%` |
| `bucket007_p2` | `bucket007_person2_freejoint_legobj_e018b` | `+y`, `[0, 0.2997, 0.21328]` | `66.32%` | `100.0%` |

主要 offending ref geoms：

- `box025_p1`: `lf3`, `left_thigh_collision`, `left_shin_collision`, `rf2`, `left_linkage_brace_collision`
- `bucket007_p2`: `right_thigh_collision`, `left_thigh_collision`, `right_linkage_brace_collision`, `rf2`, `right_shin_collision`

## Claims

| Claim | 最低证据 |
|---|---|
| C1 ref geometry repair 明确降低腿/物体干涉 | `full_ref_leg_box_interference_frames_pct < 15%`，并记录 case-window 指标 |
| C2 object-side support proxy 不回退 | E018b row 的 canonical anchor、`support_proxy_point_local`、`support_proxy_gravity_scale`、`support_weld_anchor` 语义不变 |
| C3 physical rollout 可恢复完整 retarget | contact preservation `>=70%`、deep penetration `<=20%`、max penetration `<=5cm`、no fall |
| C4 true-freejoint invariant 保持 | `contact_guidance=false`、`object_action_dims=0`、`object_actuator_ids=[]`、无 direct wrench / object override |
| C5 E023 可复现 | assets/overrides/train/eval/scene snapshot/MP4/comparison 全部落盘 |

## 改动

### 1. Derived task variants

新建 `workspace/core4d_collab_retarget/scripts/E023/variants.tsv`。

| Variant | Case | Patch mode | 用途 |
|---|---|---|---|
| `E023_box025_p1_baseline_replay` | `box025_p1` | none | E023 plumbing baseline |
| `E023_bucket007_p2_baseline_replay` | `bucket007_p2` | none | E023 plumbing baseline |
| `E023_box025_p1_legpair_off` | `box025_p1` | remove lower-body/object contact pairs | 诊断：确认 failure 是否来自 lower-body collision pairs |
| `E023_bucket007_p2_legpair_off` | `bucket007_p2` | remove lower-body/object contact pairs | 诊断：确认 failure 是否来自 lower-body collision pairs |
| `E023_box025_p1_lowerbody_proxy_min` | `box025_p1` | shrink offending lower-body collision proxies | strict candidate |
| `E023_bucket007_p2_lowerbody_proxy_min` | `bucket007_p2` | shrink offending lower-body collision proxies | strict candidate |

`legpair_off` 只作 diagnostic，不作为最终 strict pass 候选，因为它可能通过移除真实腿/物体接触来掩盖 artifact。`lowerbody_proxy_min` 是首批主候选：只缩小 thigh/shin/linkage/foot lower-body collision proxy，不改 hand/object/object_collision。

### 2. Asset generator

**文件**：`workspace/core4d_collab_retarget/scripts/E023/generate_e023_assets.py`

设计：

- 读取 `results/E018b/manifest.tsv` 的 `E018b_box025_p1_canonical_t02` 与 `E018b_bucket007_p2_canonical_t02`。
- 从 E018b derived task 复制到 E023 task copy，例如：
  - `box025_person1_freejoint_legobj_e023_lowerbody_proxy_min`
  - `bucket007_person2_freejoint_legobj_e023_lowerbody_proxy_min`
- 保持 object qpos、support weld anchor、canonical support point、contact masks 不变。
- XML patch 只作用于 lower-body/object contact pairs 或 offending lower-body geoms：
  - `legpair_off`: 删除 `LEG_FOOT_GEOMS` 对 object collision geom 的 contact pairs。
  - `lowerbody_proxy_min`: thigh/shin/linkage geom radius/size 降到 `0.005`，foot spheres 降到 `0.001`；不修改 hand geoms。
- 输出 `workspace/core4d_collab_retarget/results/E023/manifest.tsv`，记录 patch mode、disabled pairs、shrunk geoms、support proxy invariants hash。

### 3. Override / train / remote

**文件**：

- `workspace/core4d_collab_retarget/scripts/E023/generate_e023_overrides.py`
- `workspace/core4d_collab_retarget/scripts/run_E023_preprocess.sh`
- `workspace/core4d_collab_retarget/scripts/train/train_E023.sh`
- `workspace/core4d_collab_retarget/scripts/train/train_E023_remote_tmux.sh`
- `workspace/core4d_collab_retarget/scripts/run_E023_remote.sh`
- `workspace/core4d_collab_retarget/scripts/pull_E023_remote_results.sh`

Key config invariants：

| 字段 | 值 |
|---|---|
| `contact_guidance` | `false` |
| `object_pd_override` | `false` |
| `object_kinematic_override` | `false` |
| `object_action_dims` | `0` |
| `object_actuator_ids` | `[]` |
| `partner_force_scale` | `0.0` |
| `partner_force_spring_kp` | `0.0` |
| `support_proxy_enabled` | `true` |
| `support_proxy_mode` | `mocap_pad` |
| `support_proxy_mocap_body_name` | `support_weld_anchor` |
| `support_proxy_mocap_quat_mode` | `object_ref` |

### 4. Eval

**文件**：`workspace/core4d_collab_retarget/scripts/eval/eval_E023.py`

复用 E018b evaluator，并追加：

- `E023_ref_geometry_repair_pass`
- `E023_support_proxy_unchanged`
- `E023_lowerbody_patch_mode`
- `E023_disabled_leg_object_pairs`
- `E023_shrunk_leg_geoms`
- `full_ref_leg_box_interference_frames_pct`
- `case_window_ref_leg_box_interference_frames_pct`
- `case_window_sim_leg_box_interference_frames_pct`

## 成功标准

| 指标 | E018b baseline | E023 目标 |
|---|---:|---:|
| full ref leg/object interference | `66.5%` / `66.3%` | `<15%` |
| object Epos | `<0.10m` 已成立 | `<0.10m` |
| object Erot | `<25deg` 已成立 | `<25deg` |
| transport | pass | pass |
| contact preservation 5cm | 低于 strict | `>=70%` |
| deep penetration duration | case-dependent | `<=20%` |
| max penetration | case-dependent | `<=5cm` |
| robot fall | false | false |

## 训练命令

```bash
bash workspace/core4d_collab_retarget/scripts/run_E023_preprocess.sh
bash workspace/core4d_collab_retarget/scripts/train/train_E023.sh smoke 0

# full
bash workspace/core4d_collab_retarget/scripts/train/train_E023.sh local 0
bash workspace/core4d_collab_retarget/scripts/run_E023_remote.sh
bash workspace/core4d_collab_retarget/scripts/pull_E023_remote_results.sh

# eval
.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E023.py --all
```

## 决策规则

- 如果 `lowerbody_proxy_min` 把 ref interference 降到 `<15%` 且 physical rollout 达成 contact/artifact/object gates，则 E023 通过，进入 E024。
- 如果 ref interference 降低但 contact 仍低，说明 geometry repair 只修掉 ref-side artifact，case 转入 E025 contact closure。
- 如果 `legpair_off` 通过但 `lowerbody_proxy_min` 不通过，不把 `legpair_off` 当最终结果；记录为 collision geometry modeling 问题，需要更合理的 lower-body proxy 或 per-geom collision filtering。
