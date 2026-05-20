# E027 results: Contact timing + data / retarget quality diagnosis

日期：2026-05-21

## 目标

按 `plan/32_E027_contact_timing_data_quality_plan.md`，E027 第一阶段只做离线诊断：

1. 聚合 E020/E026/Holosoma/E018b 证据，给 13case 打 `quality_label`。
2. 对 low-contact 且未弃用的 case 做 contact timing panel。
3. 根据证据选择最多 6 个 E027 full variants；如果没有证据支持，也要明确停止理由。

本阶段不改 reward、不改 `spider/config.py` / `spider/simulators/mjwp.py`，不占 GPU。

## 结果路径

| 产物 | 路径 |
|---|---|
| Plan | `workspace/core4d_collab_retarget/plan/32_E027_contact_timing_data_quality_plan.md` |
| Scripts | `workspace/core4d_collab_retarget/scripts/E027/`, `workspace/core4d_collab_retarget/scripts/eval/eval_E027.py` |
| Data quality CSV | `workspace/core4d_collab_retarget/results/E027/data_quality/case_quality_audit.csv` |
| Data quality MD | `workspace/core4d_collab_retarget/results/E027/data_quality/case_quality_audit.md` |
| Timing summary | `workspace/core4d_collab_retarget/results/E027/timing_panels/timing_summary.csv` |
| Timing panels | `workspace/core4d_collab_retarget/results/E027/timing_panels/*.md` |
| Full candidates | `workspace/core4d_collab_retarget/results/E027/full_variant_candidates.tsv` |
| Offline summary | `workspace/core4d_collab_retarget/results/E027/offline_summary.md` |

## 执行命令

```bash
.venv/bin/python -m py_compile \
  workspace/core4d_collab_retarget/scripts/E027/e027_common.py \
  workspace/core4d_collab_retarget/scripts/E027/audit_case_data_quality.py \
  workspace/core4d_collab_retarget/scripts/E027/diagnose_contact_timing.py \
  workspace/core4d_collab_retarget/scripts/E027/select_full_variants.py \
  workspace/core4d_collab_retarget/scripts/eval/eval_E027.py

.venv/bin/python workspace/core4d_collab_retarget/scripts/E027/audit_case_data_quality.py --all
.venv/bin/python workspace/core4d_collab_retarget/scripts/E027/diagnose_contact_timing.py --all
.venv/bin/python workspace/core4d_collab_retarget/scripts/E027/select_full_variants.py
.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E027.py --offline
```

## Data / retarget quality 结果

| Label | Count | Cases |
|---|---:|---|
| `usable_algorithmic_failure` | 6 | `box021_p1`, `box021_p2`, `box025_p2`, `bucket005_s2_p1`, `bucket005_s2_p2`, `bucket007_p1` |
| `usable_with_caveat` | 4 | `box023_p1`, `box023_p2`, `bucket001_p1`, `bucket001_p2` |
| `retarget_questionable` | 2 | `box025_p1`, `bucket007_p2` |
| `discard_from_success_denominator` | 1 | `desk021_p1` |

当前 P0 仍沿用 E026 的 9case 排除：`box021_p1`, `box021_p2`, `bucket001_p1`, `desk021_p1` 不进 P0 主分母。E027 进一步给出严格弃用：

- `desk021_p1`: `discard_from_success_denominator=True`
  - E020 root cause = `raw_data`
  - Holosoma / OmniRetarget kinematic 缺失，原因是 `desk021_p1` SOCP infeasible
  - raw/data 与前置 retarget 两层证据同时不可靠
  - 仍保留在 P1/caveat table，不从完整报告中删除

`box025_p1` 和 `bucket007_p2` 被标为 `retarget_questionable`，但不直接弃用，因为仍有 object tracking / no-fall / Holosoma available 的 counter-evidence；后续应进入 retarget geometry / lower-body control 修复，而不是简单丢弃。

## Contact timing 结果

E027 对四个 priority low-contact case 输出 timing panel。contact timing 口径使用 5cm hand-object SDF，而不是 MuJoCo contact count。

| Case | Label | Current overlap | Best shift | Expected gain | Dominant miss | Decision |
|---|---|---:|---:|---:|---|---|
| `box023_p1` | `usable_with_caveat` | `49.23%` | `0` | `0.00pp` | `timing_or_wrong_side` | E030 geometry/control |
| `box023_p2` | `usable_with_caveat` | `54.96%` | `0` | `0.00pp` | `timing_or_wrong_side` | E030 geometry/control |
| `box025_p1` | `retarget_questionable` | `68.78%` | `0` | `0.00pp` | `raw_or_retarget_questionable` | retarget fix / E030 |
| `bucket007_p2` | `retarget_questionable` | `38.95%` | `0` | `0.00pp` | `raw_or_retarget_questionable` | retarget fix / E030 |

结论：当前 evidence 不支持做 E027 phase-shift / hold-window full rollout。四个 case 的最佳 shift 都是 `0`，expected gain 都是 `0pp`。`box023_p1/p2` 不是简单时间相位错，更像 side/surface/geometry/control 问题；`box025_p1` 和 `bucket007_p2` 则优先是前置 retarget geometry 问题。

## Full variant candidates

`workspace/core4d_collab_retarget/results/E027/full_variant_candidates.tsv` 为结构化空表，仅含表头。

原因：

- 没有 case 满足 “timing shift 预测带来 `>=10pp` contact gain”。
- `box025_p1`、`bucket007_p2` 被标为 `retarget_questionable`，不进入 E027 reward/timing sweep。
- `box023_p1/p2` 虽保留为 P0 可用 case，但 timing panel 指向 wrong-side / surface / geometry-control，而不是 phase shift。

## Claims 验证

| Claim | 结果 |
|---|---|
| C1: 13case 全部有可复核 quality label | 通过：`case_quality_audit.csv` 13 行 |
| C2: 弃用/降级 case 有多证据支持 | 通过：`desk021_p1` 为唯一 denominator discard，具备 raw_data + Holosoma infeasible 双证据；`box025_p1/bucket007_p2` 只降级为 retarget_questionable，不弃用 |
| C3: low-contact timing 与 data/retarget failure 可分离 | 通过：4 个 timing panel；box023 为 timing_or_wrong_side，box025/bucket007 为 retarget_questionable |
| C4: 产出有界 full candidates | 通过：0 个 candidates，结构化空 TSV；停止理由明确 |
| C5: 不以算法失败反向证明数据差 | 通过：box021/bucket stability 和 bucket penetration cases 仍标为 usable_algorithmic_failure 或 usable_with_caveat，没有因 rollout 差被弃用 |

## 下一步

1. **E028 hard no-penetration / surface feasibility**：处理 `bucket005_s2_p1`, `bucket005_s2_p2`, `bucket007_p1`, `bucket001_p2`。这些 case 数据/retarget 基本可用，失败主要是 high-contact penetration shortcut。
2. **E030 retarget geometry / lower-body control**：处理 `box025_p1`, `bucket007_p2`，并把 `box023_p1/p2` 的 wrong-side / surface-control 问题纳入 E030 或后续 surface target 诊断。
3. **不启动 E027 full rollout**：当前没有 timing phase-shift 证据，继续跑 E027 full variants 会变成无依据 sweep。
4. **P1 caveat**：`desk021_p1` 保留在 P1 表中，但从主 success denominator 中弃用；`box021_p1/p2`、`bucket001_p1` 仍按 E026 P0 规则排除。

## 遇到的错误

| 错误 | 尝试次数 | 解决方案 |
|---|---:|---|
| 初版 timing script 用 `sim_total_contact_count` 估计 contact overlap，导致 box023 overlap 虚高 | 1 | 改为严格使用 `sim_min_hand_sdf_m/ref_min_hand_sdf_m <= 0.05m` 的 5cm hand-object SDF 口径，并重跑 timing/eval |
