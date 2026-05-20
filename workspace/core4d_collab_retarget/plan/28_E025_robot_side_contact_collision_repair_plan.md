# E025 实验计划：robot-side contact closure + collision penalty

日期：2026-05-20

## Context

E020 将 `box023_p2`、`bucket005_s2_p1`、`bucket005_s2_p2`、`bucket007_p1` 归为 `algo_contact`。这组 case 的 object-side transport 已成立，robot 不摔倒，但 contact/collision behavior 没有达到完整 retargeting gate。

Post-E022 update：`box023_p1` 的 contact mask semantics 已修正，但 contact preservation 仍只有约 `25%`，因此作为 low-contact closure case 并入 E025；这不是重复 E022 mask sweep。

E025 目标不是继续调 anchor，而是分离两类问题：

1. `box023_p1/p2`: contact preservation low，主要是 contact closure gap。
2. bucket / bucket007 cases: contact 高但 deep penetration 高，说明当前 reward 可以通过“手伸进物体”来维持 contact，需要 explicit robot-object penetration penalty。

## Baseline evidence

| Case | Contact 5cm | Deep pen duration | Max pen | Leg interference | Dominant issue |
|---|---:|---:|---:|---:|---|
| `box023_p1` | `25.3%` | `0.0%` | `<5cm` | `0.0%` | mask fixed but contact closure gap remains |
| `box023_p2` | `28.6%` | `3.3%` | `2.37cm` | `0.0%` | contact closure gap |
| `bucket005_s2_p1` | `97.6%` | `88.2%` | `5.12cm` | `23.7%` | hand deep penetration + leg shortcut |
| `bucket005_s2_p2` | `95.9%` | `74.4%` | `7.94cm` | `9.9%` | hand deep penetration |
| `bucket007_p1` | `79.0%` | `68.5%` | `8.40cm` | `2.0%` | hand deep penetration |

Existing code:

- `spider/config.py`: contact closure knobs already exist (`contact_hdmi_gain`, `contact_hdmi_sigma`, `contact_hdmi_ori_weight`, `contact_hdmi_ori_mode`, `hold_contact_rew_scale`, `hold_contact_sigma`).
- `spider/simulators/mjwp.py`: HDMI per-EEF contact reward and E074C hold-contact reward reward proximity to surface, but do not penalize being inside object.
- `workspace/core4d_collab_retarget/scripts/eval/paper_metrics.py`: penetration metrics are eval-only today.

## Claims

| Claim | 最低证据 |
|---|---|
| C1 contact closure 对 low-contact case 有效 | `box023_p1/p2` contact preservation `>=70%`，deep pen 不恶化 |
| C2 explicit penetration penalty 降低 bucket artifact | bucket cases deep penetration duration `<15%`，max penetration `<=5cm` |
| C3 no object-side regression | all cases object Epos `<0.10m`、Erot `<25deg`、transport pass |
| C4 no stability regression | all cases no fall |
| C5 至少部分泛化成立 | 5 case 至少 `2/5` strict pass，未通过 case 有明确下一步 |

## 改动

### 1. Reward implementation gap

E025 需要新增训练期 penetration penalty；当前只有 eval 指标。建议最小实现：

**文件**：

- `spider/config.py`
- `spider/simulators/mjwp.py`

新增 knobs：

| 字段 | 默认 | 说明 |
|---|---:|---|
| `robot_object_penalty_scale` | `0.0` | 手/前臂等 robot-object penetration penalty |
| `robot_object_penalty_margin_m` | `0.0` | surface margin |
| `robot_object_penalty_deep_threshold_m` | `0.02` | deep penetration hinge threshold |
| `leg_object_penalty_scale` | `0.0` | lower-body/object guard |
| `leg_object_penalty_margin_m` | `0.02` | lower-body guard margin |
| `leg_object_penalty_geom_names` | `[]` | optional explicit lower-body geom list |

实现原则：

- 对 object box/cylinder SDF 或现有 paper_metrics 中的 geometry distance 口径做训练期近似，不做字符串 ad-hoc parsing。
- 对 hand contact geoms 的 inside-object negative SDF 加 hinge penalty；不要惩罚正常 surface contact。
- leg guard 只对 E081 `LEG_FOOT_GEOMS` 类 lower-body geoms 生效。
- 默认 scale 为 `0.0`，保证 E018b/E022-E024 不受影响。

### 2. Variants

新建 `workspace/core4d_collab_retarget/scripts/E025/variants.tsv`。

| Variant | Cases | Params | 目的 |
|---|---|---|---|
| `E025_box023_p1_hc2_gain8_sigma20_ori_nf` | `box023_p1` | `hold_contact=2.0`, `gain=8.0`, `sigma=0.20`, `ori_weight=0.3`, `ori_mode=near_field` | post-E022 low-contact closure |
| `E025_box023_p2_hc2_gain8_sigma20_ori_nf` | `box023_p2` | `hold_contact=2.0`, `gain=8.0`, `sigma=0.20`, `ori_weight=0.3`, `ori_mode=near_field` | low-contact closure |
| `E025_bucket_penalty_lite_hc1` | bucket005/bucket007 | `robot_object_penalty_scale=2.0`, `deep_threshold=0.02`, `hold_contact=1.0` | reduce hand penetration |
| `E025_bucket_leg_guard_penalty` | `bucket005_s2_p1` | add `leg_object_penalty_scale=2.0`, margin `0.02` | reduce leg shortcut |
| `E025_bucket005_s2_p2_penalty_s4_hc1` / `E025_bucket007_p1_penalty_s4_hc1` | selected bucket cases | `robot_penalty=4.0`, `hold_contact=1.0` | stronger anti-penetration if lite is insufficient |

E025 可以先按 case-group 展开，不要求每个 case 每个 variant 全组合；避免把 5 case x 多 sweep 直接放大到不可控运行量。

### 3. Scripts

**文件**：

- `workspace/core4d_collab_retarget/scripts/E025/generate_e025_overrides.py`
- `workspace/core4d_collab_retarget/scripts/run_E025_preprocess.sh`
- `workspace/core4d_collab_retarget/scripts/train/train_E025.sh`
- `workspace/core4d_collab_retarget/scripts/train/train_E025_remote_tmux.sh`
- `workspace/core4d_collab_retarget/scripts/run_E025_remote.sh`
- `workspace/core4d_collab_retarget/scripts/pull_E025_remote_results.sh`
- `workspace/core4d_collab_retarget/scripts/eval/eval_E025.py`

Reuse E018b derived tasks and canonical support proxy. Do not copy or patch scene XML unless E025 later needs a lower-body collision-pair diagnostic.

### 4. Eval

复用 E018b evaluator，追加：

- `E025_contact_closure_pass`
- `E025_penetration_guard_pass`
- `E025_leg_guard_pass`
- `E025_strict_success`
- reward params columns (`robot_object_penalty_scale`, `leg_object_penalty_scale`, `hold_contact_rew_scale`, `contact_hdmi_gain`, `contact_hdmi_sigma`)

## 成功标准

| 指标 | 目标 |
|---|---|
| Contact preservation | `>=70%` |
| Deep penetration duration | `<15%` |
| Max penetration | `<=5cm` |
| Fall | false |
| Object | Epos `<0.10m`, Erot `<25deg`, transport pass |
| Strict count | `>=2/5` cases strict pass |

## 训练命令

```bash
bash workspace/core4d_collab_retarget/scripts/run_E025_preprocess.sh
bash workspace/core4d_collab_retarget/scripts/train/train_E025.sh smoke 0

# full
bash workspace/core4d_collab_retarget/scripts/train/train_E025.sh local 0
bash workspace/core4d_collab_retarget/scripts/run_E025_remote.sh
bash workspace/core4d_collab_retarget/scripts/pull_E025_remote_results.sh

# eval
.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E025.py --all
```

## 决策规则

- 如果 `box023_p1/p2` contact remains low after high contact closure reward, it should move to dynamic target/contact timing diagnosis, not more mask or penalty sweep.
- If bucket contact remains high but deep penetration does not fall under `<15%`, increase penalty only after checking video/keyframes; otherwise the reward may simply choose a different penetration path.
- If penetration improves but contact collapses, tune contact/penalty balance; do not count it as pass because contact preservation is a primary task objective.
- If leg shortcut persists only in `bucket005_s2_p1`, keep E025 result partial and split a later lower-body collision geometry experiment.
