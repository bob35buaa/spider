# E027 实验计划：Contact Timing + Data / Retarget Quality Diagnosis

日期：2026-05-21

## Context

E026 完整评测后，当前 best dynamic 是 `spider_best_E018b_E022_E025`：

- P0 9case：object `4.97cm`、contact `65.81%`、deep penetration `30.52%`、fall `0`、strict `1/9`。
- P1 13case：object `5.33cm`、contact `55.87%`、deep penetration `26.37%`、fall `3`、strict `1/13`。

E022-E025 已证明普通 reward sweep 不足：`box023_p1` mask overclaim 修掉后 contact 仍只有 `25.30%`；`box023_p1/p2` 加强 contact reward 后仍不过 strict contact；bucket 系列 object tracking 很好但 high contact 经常来自 penetration shortcut；`box025_p1`、`bucket007_p2` 的 kinematic reference 已有 lower-body/object interference。

E020 已做过一版 S1-S6 failure attribution，产物在：

- `workspace/core4d_collab_retarget/results/E020_audit/root_cause_attribution.csv`
- `workspace/core4d_collab_retarget/results/E020_audit/anchor_vs_raw.csv`
- `workspace/core4d_collab_retarget/results/E020_audit/mask_vs_raw.csv`
- `workspace/core4d_collab_retarget/results/E020_audit/ref_physics.csv`
- `workspace/core4d_collab_retarget/results/E020_audit/sim_ref_overlay.csv`
- `workspace/core4d_collab_retarget/results/E020_audit/per_case/*/attribution_panel.png`

E027 不重做全部 E020。它复用 E020/E026/Holosoma，并补上两个缺口：

1. 将 `root_cause` 升级为面向数据筛选的 `quality_label` / `discard` 决策。
2. 对 low-contact 且未弃用的 case 做 contact timing / target phase 诊断，决定是否值得进入 full rollout variants。

### 根因分析

当前失败至少有三层来源：

| 层级 | 可能问题 | 证据来源 |
|---|---|---|
| raw CORE4D | mocap/object/contact annotation 跳变或与视觉冲突 | E020 raw contact cache、mask audit、keyframes |
| OmniRetarget / Holosoma | SOCP infeasible、kinematic jitter、ref leg/object interference、contact phase 不可达 | `holosoma_v2_kinematic/comparison.csv`、E020 ref physics |
| SPIDER physical | contact timing、surface feasibility、stability/control 不足 | E026 best dynamic metrics、E018b/E022-E025 rollout |

如果一个 case 的 raw 与 kinematic reference 已经不可信，就不应继续把它当作 SPIDER algorithmic failure 强行优化。相反，如果 raw/reference 基本可信，E027 应给出下一步机制实验：timing shift、surface target、hard no-penetration 或 lower-body control。

### 关键 insight

E027 是一个离线诊断实验，先不做核心 CEM/reward 改动。它应先回答：

- 哪些 case 仍然是有效算法优化目标？
- 哪些 case 应该降级为 caveat 或从 P0 success denominator 移除？
- 对保留的 low-contact case，contact miss 是相位错、target 错、surface/side 错，还是控制不可达？

只有回答这些问题后，才进入 E027 full variants 或 E028 hard no-penetration。

## Claims

| Claim | 最低证据 |
|---|---|
| C1: E027 能给 13case 全部打上可复核的数据/retarget 可用性标签 | `case_quality_audit.csv` 有 13 行，且每行包含 `quality_label`、`discard_from_p0`、`discard_from_success_denominator`、证据字段和 counter-evidence |
| C2: 弃用或降级 case 必须有多证据支持 | 任一 `discard_from_success_denominator=True` case 至少满足 2 类 independent evidence failure，并在 Markdown 中列出关键帧/视频路径 |
| C3: E027 能分离 low-contact 的 timing failure 与 data/retarget failure | 对 `box023_p1`、`box023_p2`、`box025_p1`、`bucket007_p2` 至少输出 timing panel；如果某 case 被弃用，要说明不做 timing/full variant 的原因 |
| C4: E027 能产出下一步 full variants 的有界候选集 | `full_variant_candidates.tsv` 不超过 6 个 full variants，并附每个 variant 的 evidence rationale；若没有值得跑的 variants，也要给出停止理由 |
| C5: E027 不以算法失败反向证明数据差 | `case_quality_audit.md` 必须列出 counter-evidence；只有算法 rollout 差但 raw/retarget 证据好时，标签应为 `usable_algorithmic_failure` |

## Scope

### 13case audit

沿用 E026 P1 union：

`box021_p1`, `box021_p2`, `box023_p1`, `box023_p2`, `box025_p1`, `box025_p2`, `bucket001_p1`, `bucket001_p2`, `bucket005_s2_p1`, `bucket005_s2_p2`, `bucket007_p1`, `bucket007_p2`, `desk021_p1`

### Timing panel priority

第一优先：

- `box023_p1`
- `box023_p2`
- `box025_p1`
- `bucket007_p2`

第二优先，只有 audit 发现 contact annotation / retarget quality 可疑时纳入：

- `desk021_p1`
- `bucket001_p1`
- `box021_p1/p2`

bucket penetration cases (`bucket005_s2_p1/p2`, `bucket007_p1`, `bucket001_p2`) 的主线仍是 E028 hard no-penetration；E027 只做 quality label，不做 timing sweep，除非 audit 发现 contact phase 明显异常。

## 数据源

| 数据源 | 路径 | 用途 |
|---|---|---|
| E020 attribution | `workspace/core4d_collab_retarget/results/E020_audit/root_cause_attribution.csv` | 初始 root cause、S1-S4 pass/fail、evidence 字符串 |
| E020 raw/mask/ref tables | `anchor_vs_raw.csv`, `mask_vs_raw.csv`, `ref_physics.csv`, `sim_ref_overlay.csv` | raw contact、mask mismatch、ref interference、sim/ref alignment |
| E020 panels/keyframes | `results/E020_audit/per_case/*` | 视觉/counter-evidence |
| E026 full metrics | `workspace/core4d_collab_retarget/results/E026_full_eval/method_case_metrics.csv` | best dynamic、E081、OmniRetarget 统一指标 |
| E026 best selection | `workspace/core4d_collab_retarget/results/E026_full_eval/best_dynamic_selection.csv` | 每 case 当前 best dynamic variant |
| Holosoma kin metrics | `workspace/core4d_collab_retarget/results/holosoma_v2_kinematic/comparison.csv` | kinematic feasibility、cost、smoothness、28cm contact、missing case |
| E018b manifest | `workspace/core4d_collab_retarget/results/E018b/manifest.tsv` | scene、mask、source task、online video、support point meta |
| E018b rollout | `workspace/core4d_collab_retarget/results/E018b/*.npz` | sim qpos for timing diagnostics |
| E018b task snapshot | `workspace/core4d_collab_retarget/results/E018b/scene_snapshot/*/0/trajectory_kinematic.npz` | ref qpos/contact mask |
| Processed mask | `workspace/core4d_collab_retarget/results/E018b/contact_masks/*/raw_contact_mask_3cm.npz` | raw 3cm contact source |

## 改动

### 1. E027 detailed audit scripts

| # | 文件 | 改动 |
|---|---|---|
| 1 | `workspace/core4d_collab_retarget/scripts/E027/audit_case_data_quality.py` | 聚合 E020/E026/Holosoma/E018b，输出 13case quality labels |
| 2 | `workspace/core4d_collab_retarget/scripts/E027/diagnose_contact_timing.py` | 对 low-contact case 输出 per-frame timing / contact miss panel |
| 3 | `workspace/core4d_collab_retarget/scripts/E027/select_full_variants.py` | 根据 audit + timing panel 生成最多 6 个 full candidates |
| 4 | `workspace/core4d_collab_retarget/scripts/eval/eval_E027.py` | 汇总 E027 离线诊断产物；若后续有 full variants，再接入 paper metrics |
| 5 | `workspace/core4d_collab_retarget/scripts/train/train_E027.sh` | 预留 full rollout wrapper；第一阶段不使用 GPU |
| 6 | `workspace/core4d_collab_retarget/scripts/run_E027_remote.sh` | 仅当 full variants `>=3` 时使用 |

第一阶段只实现 1-4，不改 `spider/config.py` / `spider/simulators/mjwp.py`。如果 timing panel 明确需要 runtime phase shift / surface projection，再写下一版 E027b 或 E028 plan，并考虑新分支。

### 2. Quality label 规则

| Label | 判定 |
|---|---|
| `usable_algorithmic_failure` | raw/contact/kinematic 证据基本可信，但 physical rollout 失败 |
| `usable_with_caveat` | 有一类 data/retarget 警告，但仍可支持算法比较 |
| `retarget_questionable` | Holosoma infeasible、高 cost/smoothness outlier、ref leg/object interference、kinematic contact 不可信中至少两项成立 |
| `raw_data_questionable` | raw contact / object / visual 证据与 processed annotation 或 support assumption 冲突 |
| `discard_from_success_denominator` | raw 与 retarget 两层均有强证据失败，或 raw/visual 已无法支持单 G1 physical retarget 评测 |

最低弃用协议：

- 至少 2 类 independent evidence failed；
- 需要 `primary_evidence`、`secondary_evidence`、`counter_evidence`；
- 需要 `visual_evidence_path` 或明确说明视觉证据缺失；
- P1 full table 保留，P0 denominator 是否移除必须单独列 `discard_from_p0`。

### 3. Timing panel 字段

逐帧 CSV 至少包含：

- `frame`
- `ref_contact_left/right`
- `raw_contact_left/right`（如可用）
- `robot_contact_5cm_left/right`
- `sim_hand_object_dist_left/right_m`
- `ref_hand_object_dist_left/right_m`
- `sim_min_hand_sdf_m`
- `ref_min_hand_sdf_m`
- `object_pos_lag_m`
- `support_proxy_lag_m`（如可从 manifest/scene 推出，否则留空并说明）
- `contact_miss_reason`
- `phase_shift_suggestion_frames`

Markdown panel 至少包含：

- case-level label / discard decision
- contact miss breakdown
- top-3 suspected failure reasons
- recommended next action: `discard`, `retarget_fix`, `E027_full_variant`, `E028_barrier`, `E030_geometry`

## 结果路径

| 产物 | 路径 |
|---|---|
| Plan | `workspace/core4d_collab_retarget/plan/32_E027_contact_timing_data_quality_plan.md` |
| Data quality CSV | `workspace/core4d_collab_retarget/results/E027/data_quality/case_quality_audit.csv` |
| Data quality MD | `workspace/core4d_collab_retarget/results/E027/data_quality/case_quality_audit.md` |
| Per-case audit CSV | `workspace/core4d_collab_retarget/results/E027/data_quality/per_case/*.csv` |
| Timing panels | `workspace/core4d_collab_retarget/results/E027/timing_panels/*.md` |
| Timing CSV | `workspace/core4d_collab_retarget/results/E027/timing_panels/*.csv` |
| Full candidates | `workspace/core4d_collab_retarget/results/E027/full_variant_candidates.tsv` |
| Log | `workspace/core4d_collab_retarget/log/27_E027_contact_timing_data_quality_results.md` |

## Reward 权重

不适用。E027 第一阶段是离线诊断，不改 reward。

如果 E027 产生 full variants，必须先追加 plan 小节或 E027b plan，明确 phase shift / hold window / surface target 参数，再开跑。

## 执行命令

### 第一阶段：离线 audit，不占 GPU

```bash
.venv/bin/python workspace/core4d_collab_retarget/scripts/E027/audit_case_data_quality.py --all
.venv/bin/python workspace/core4d_collab_retarget/scripts/E027/diagnose_contact_timing.py --all
.venv/bin/python workspace/core4d_collab_retarget/scripts/E027/select_full_variants.py
.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E027.py --offline
```

### 第二阶段：如需 full variants

只有 `full_variant_candidates.tsv` 非空且包含 `>=3` 个独立 full variants 时，才启用远程 2 卡 + 本地 1 卡：

```bash
# local critical variant
RUN_TIMEOUT_SECONDS=2400 RUN_STALL_TIMEOUT_SECONDS=300 \
  bash workspace/core4d_collab_retarget/scripts/train/train_E027.sh one 0 <variant>

# remote non-blocking variants
git push
ssh spider-remote "cd /home/xiayb/pHRI_workspace/spider && git pull"
bash workspace/core4d_collab_retarget/scripts/run_E027_remote.sh
```

远程 wrapper 必须：

- GPU0/GPU1 串行分队列；
- 每条 variant 独立 video/npz 路径；
- 默认带 `RUN_TIMEOUT_SECONDS` / `RUN_STALL_TIMEOUT_SECONDS`；
- pull 前检查不是 smoke `T=2`。

## 成功标准

| 指标 | 本次目标 |
|---|---|
| data audit coverage | `13/13` case 有 quality label |
| discard protocol | 任一弃用 case 有至少 2 类 independent evidence，并保留 counter-evidence |
| timing panel coverage | 第一优先 low-contact cases 全部有 timing panel，除非已被明确弃用 |
| full candidates | 输出 `0-6` 个候选；每个候选有 evidence rationale |
| no premature core change | 第一阶段不改 `spider/config.py` / `spider/simulators/mjwp.py` |
| reproducibility | 所有命令固化为脚本；log 记录输入/输出路径 |

## 决策规则

| 情况 | 后续动作 |
|---|---|
| `quality_label=discard_from_success_denominator` | 不进入 E028-E031 主优化分母，P1/caveat 保留 |
| `quality_label=retarget_questionable` | 优先回到 OmniRetarget/data reference 修复或 E030 geometry；不做 SPIDER reward sweep |
| `quality_label=raw_data_questionable` | 降级为 caveat；只有用户确认需要 salvage 时才继续 |
| low-contact 且 timing shift 预测有效 | 生成 E027 full variant |
| low-contact 但 timing 无效、data/retarget 可信 | 转 E030 geometry/control |
| high-contact penetration | 转 E028 hard no-penetration |

## Git / Branch 策略

- 当前 E027 第一阶段只新增 workspace scripts/plan/log，不开新分支。
- 每个阶段必须单独 commit：
  1. E027 plan commit；
  2. E027 offline audit scripts commit；
  3. E027 offline results/log commit；
  4. 如进入 full variants，train/eval/remote scripts commit；
  5. full results/log/tracker commit。
- 如果后续需要修改 `spider/config.py` / `spider/simulators/mjwp.py` 实现 runtime phase shift 或 surface target，先从当前分支新开实验分支，再改核心代码。

## 风险

| 风险 | 应对 |
|---|---|
| 过度弃用困难 case | 弃用必须多证据；P1 full table 保留；记录 counter-evidence |
| E020 旧 audit 口径不足 | E027 不覆盖旧结论，新增 Holosoma/E026 evidence 交叉验证 |
| Timing panel 缺 raw 字段 | 字段留空并标注 missing source；不能用缺失字段支持弃用 |
| 离线诊断无法推出 full variant | 输出停止理由，直接转 E028/E030，不强造 sweep |
| 远程不稳定 | E027 第一阶段不依赖远程；后续 full wrapper 带 stall guard |
