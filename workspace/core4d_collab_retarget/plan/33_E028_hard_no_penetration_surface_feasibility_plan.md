# E028 实验计划：Hard No-Penetration / Surface Feasibility

日期：2026-05-21

分支：`exp/core4d-collab-retarget-e028-hard-penetration`

## Context

E027 已完成 contact timing + data/retarget quality diagnosis。结论对 E028 很关键：

- `bucket005_s2_p1`、`bucket005_s2_p2`、`bucket007_p1` 被标为 `usable_algorithmic_failure`，不是数据/retarget 弃用对象。
- `bucket001_p2` 被标为 `usable_with_caveat`，但 object tracking、no-fall、Holosoma available 等反证仍支持继续作为 P0 hard-penetration 优化目标。
- `desk021_p1` 是唯一 `discard_from_success_denominator`，不进入 E028。
- E027 没有生成 phase-shift full candidates，因此 E028 不做 timing sweep。

E025 已证明 soft collision penalty 有工程信号但强度不够：

| Case | E026/E025 best | Contact 5cm | Deep pen | Max pen | 判断 |
|---|---|---:|---:|---:|---|
| `bucket005_s2_p1` | E018b / E025 lite | `97.59-99.47%` | `88.15-92.89%` | `5.03-5.12cm` | high contact 基本来自 hand/object penetration shortcut |
| `bucket005_s2_p2` | `E025_bucket005_s2_p2_penalty_s4_hc1` | `96.42%` | `64.53%` | `7.06cm` | scale 4 有改善但远未达 `<15%` |
| `bucket007_p1` | `E025_bucket007_p1_penalty_s4_hc1` | `84.13%` | `35.57%` | `5.16cm` | 最强正信号，但仍略高于 max pen gate |
| `bucket001_p2` | `E024_bucket001_p2_root025_gain2_stab_t065` | `77.53%` | `59.60%` | `8.08cm` | stability 可修，剩余主要是 penetration |
| `box025_p2` guard | `E018b_box025_p2_canonical_t02` | `86.93%` | `0.00%` | `1.14cm` | strict pass guard，不应回退 |

E028 的目标不是继续提高 contact 数字，而是阻止 CEM 选择“高接触 + 手伸进物体内部”的轨迹。只要 no-penetration 约束使 contact 下降，也必须判断这是更真实的 surface contact 还是 target 本身不可行，不能用穿透换成功。

## Claims

| Claim | 最低证据 |
|---|---|
| C1: E028 的 hard feasibility 能显著压低 bucket penetration shortcut | 目标 4case mean deep penetration 相比 E026/E025 best 降低 `>=25pp`，且至少 `2/4` case deep pen `<=15%` |
| C2: hard barrier 不能靠丢 object transport 达成 | 每个目标 case object pos `<=8cm`，若 `>8cm` 必须在 log 中标为 barrier/object tradeoff failure |
| C3: surface feasibility 与 contact preservation 要同时报告 | 每个 variant 同时报 contact 5cm、hand deep pen、leg deep pen、max pen、object、fall；禁止只按 contact 选 best |
| C4: pass guard 不回退 | `box025_p2` guard 在 E028 机制下仍保持 object `<=8cm`、contact `>=70%`、deep pen `<=15%`、no fall |
| C5: E028 不把 E027 数据质量 caveat 当算法收益 | `bucket001_p2` 保留 caveat 标记；如果失败，不能反向改成数据弃用，除非新增 independent raw/retarget 证据 |

## Scope

### Target cases

主目标：

- `bucket005_s2_p1`
- `bucket005_s2_p2`
- `bucket007_p1`
- `bucket001_p2`

Guard：

- `box025_p2`

不纳入 E028：

- `box023_p1/p2`：E027 指向 surface/geometry/control，不是 penetration hard barrier 主线。
- `box025_p1`、`bucket007_p2`：E027 标为 `retarget_questionable`，转 E030。
- `desk021_p1`：E027 已 `discard_from_success_denominator`。
- `bucket001_p1`、`box021_p1/p2`：P1 stability/data diagnostic，不作为 E028 成功分母。

## 改动

### 1. Core reward / feasibility knobs

E028 预计需要核心代码改动，因此已从 `exp/core4d-collab-retarget` 切出新分支。

| 文件 | 改动 |
|---|---|
| `spider/config.py` | 新增默认关闭的 E028 knobs：`robot_object_barrier_*`、`cem_penetration_score_cap_*`、`contact_penetration_gate_*`、`penetration_staged_contact_*` |
| `spider/simulators/mjwp.py` | 复用 E025 `geom_box_sdf_min`，实现非线性 barrier、score cap / rejection 近似、contact reward gate、staged contact schedule |
| `workspace/core4d_collab_retarget/scripts/E028/generate_e028_overrides.py` | 生成 E028 manifest + Hydra overrides，复用 E018b/E024/E025 best source variants |
| `workspace/core4d_collab_retarget/scripts/E028/variants.tsv` | 固定 6 个 full variants，避免无依据大 sweep |
| `workspace/core4d_collab_retarget/scripts/train/train_E028.sh` | 本地/远程/smoke/eval entrypoint |
| `workspace/core4d_collab_retarget/scripts/run_E028_remote.sh` | 远程 2-GPU 队列 |
| `workspace/core4d_collab_retarget/scripts/eval/eval_E028.py` | 汇总 E028 指标、与 E026/E025 best baseline 对比、输出 strict gate |

所有新增 knobs 默认值必须保持旧实验行为不变。

### 2. Hard feasibility 机制

E025 现有 penalty 实际是线性 soft hinge：

`robot_object_penalty = -scale * clamp(deep_limit - robot_sdf, 0)`

E028 改成三种更硬机制，先做小而有区分度的 ablation：

| 机制 | 作用 | 预期风险 |
|---|---|---|
| `barrier_quad` | 对 `sdf < margin` 施加 squared barrier，margin 从物体表面外 `1-2cm` 开始，而不是只惩罚 deep threshold | 可能降低 contact，需要看是否换来真实 surface contact |
| `score_cap` | 如果 trajectory 当前 frame 出现 `sdf < -1cm/-2cm`，对 sample reward 加大幅 cap penalty，近似 CEM rejection | 可能让 object tracking 变差，需 guard object |
| `contact_gate` | 当 hand/object SDF 为负或低于 margin 时，contact reward 不再继续给正收益，避免 penetration 抵消 penalty | 可能暴露 target 不可行，contact 下降时要记录原因 |
| `staged_contact` | 前半段先优化 object/posture/no-penetration，后半段再打开 contact preservation | 可能错过早期接触，需要和 timing panel 区分 |

### 3. Surface target

第一版 E028 不做复杂 mesh-level surface projection，只做 box/collision-geom SDF 约束，原因是当前 runtime E025 SDF 已用 object collision box 且可直接跑在 MJWarp reward 中。

如果 hard barrier 使 contact 大幅下降但 object/fall 保持，E028 log 中将其记录为 surface target 需求，并把真正的 object-specific surface projection 推到 E028b 或 E030，而不是在同一实验里继续扩展。

## Variants

控制 full variants 不超过 6 个：

| Variant | Source | Case | Queue | 机制 | 关键参数 |
|---|---|---|---|---|---|
| `E028_bucket007_p1_barrier_quad_m02` | `E025_bucket007_p1_penalty_s4_hc1` | `bucket007_p1` | local | `barrier_quad` | margin `2cm`, scale high, no score cap |
| `E028_bucket005_s2_p2_barrier_quad_m02` | `E025_bucket005_s2_p2_penalty_s4_hc1` | `bucket005_s2_p2` | remote_gpu0 | `barrier_quad` | margin `2cm`, scale high, no score cap |
| `E028_bucket005_s2_p1_contact_gate_m02` | `E018b_bucket005_s2_p1_canonical_t02` | `bucket005_s2_p1` | remote_gpu1 | `contact_gate + barrier_quad` | contact reward gated when SDF `<0` |
| `E028_bucket001_p2_contact_gate_m02` | `E024_bucket001_p2_root025_gain2_stab_t065` | `bucket001_p2` | remote_gpu0 | `contact_gate + barrier_quad` | preserve stability settings |
| `E028_bucket007_p1_scorecap_m01` | `E025_bucket007_p1_penalty_s4_hc1` | `bucket007_p1` | remote_gpu1 | `score_cap` | cap if min SDF `<-1cm` |
| `E028_box025_p2_guard_barrier_m02` | `E018b_box025_p2_canonical_t02` | `box025_p2` | local_after_main | guard | strict-pass case regression check |

执行顺序：

1. 本地先跑 `bucket007_p1_barrier_quad_m02`，因为 E025 scale 4 已有最清楚正信号。
2. 远程 GPU0 跑 `bucket005_s2_p2_barrier_quad_m02` 后接 `bucket001_p2_contact_gate_m02`。
3. 远程 GPU1 跑 `bucket005_s2_p1_contact_gate_m02` 后接 `bucket007_p1_scorecap_m01`。
4. `box025_p2` guard 在本地主 variant 完成后跑；如果主 variant 已证明 core wiring 错误，先不跑 guard，先修 smoke。

## 本地 / 远程并行

E028 有 `>=3` 个独立 full variants，满足远程并行触发条件。计划使用本地 1 卡 + 远程 2 卡：

```bash
# 先生成 variants / overrides
.venv/bin/python workspace/core4d_collab_retarget/scripts/E028/generate_e028_overrides.py

# smoke：本地确认所有 overrides 和新 knobs 可解析
RUN_TIMEOUT_SECONDS=600 RUN_STALL_TIMEOUT_SECONDS=180 \
  bash workspace/core4d_collab_retarget/scripts/train/train_E028.sh smoke 0

# full：本地关键 case
RUN_TIMEOUT_SECONDS=2400 RUN_STALL_TIMEOUT_SECONDS=300 \
  bash workspace/core4d_collab_retarget/scripts/train/train_E028.sh local 0

# full：远程 2 卡，需先 git push
git push -u origin exp/core4d-collab-retarget-e028-hard-penetration
ssh spider-remote "cd /home/xiayb/pHRI_workspace/spider && git fetch && git switch exp/core4d-collab-retarget-e028-hard-penetration && git pull"
RUN_TIMEOUT_SECONDS=2400 RUN_STALL_TIMEOUT_SECONDS=300 \
  bash workspace/core4d_collab_retarget/scripts/run_E028_remote.sh

# 结果回收后本地统一评估
.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E028.py --all
```

如果远程 stale，按 E025 经验先杀掉该 tmux 队列，回收已完成 NPZ，剩余 variants 本地顺序补跑，不重复同一失败配置。

## Evaluation

`eval_E028.py` 必须输出：

| 产物 | 路径 |
|---|---|
| Plan | `workspace/core4d_collab_retarget/plan/33_E028_hard_no_penetration_surface_feasibility_plan.md` |
| Variants | `workspace/core4d_collab_retarget/scripts/E028/variants.tsv` |
| Manifest | `workspace/core4d_collab_retarget/results/E028/manifest.tsv` |
| Results | `workspace/core4d_collab_retarget/results/E028/*.npz`, `online_video/*.mp4` |
| Comparison | `workspace/core4d_collab_retarget/results/E028/comparison.csv` |
| Aggregate | `workspace/core4d_collab_retarget/results/E028/aggregate_summary.json` |
| Baseline delta | `workspace/core4d_collab_retarget/results/E028/baseline_delta.csv` |
| Log | `workspace/core4d_collab_retarget/log/28_E028_hard_no_penetration_surface_feasibility_results.md` |

核心指标：

- `paper_omniretarget_contact_preservation_5cm_pct`
- `paper_omniretarget_robot_object_deep_penetration_duration_pct`
- `paper_omniretarget_hand_object_deep_penetration_duration_pct`
- `paper_omniretarget_leg_object_deep_penetration_duration_pct`
- `paper_omniretarget_robot_object_max_penetration_cm`
- `paper_object_Epos_case_m`
- `paper_object_Erot_case_deg`
- `E018b_robot_fall_detected`
- `case_window_sim_min_hand_sdf_mean_m`
- `case_window_sim_min_hand_sdf_max_m`

E028 strict pass 定义：

```text
contact_5cm >= 70%
robot_object_deep_penetration <= 15%
robot_object_max_penetration <= 5cm
object_Epos <= 8cm
object_Erot <= 25deg
no fall
support_proxy unchanged
```

## 成功标准

| 指标 | 目标 |
|---|---|
| target coverage | 4/4 target cases 至少各有 1 个 full E028 result |
| deep penetration | 4case mean 相比 best baseline 降低 `>=25pp` |
| case pass | 至少 `2/4` target cases deep pen `<=15%` 且 max pen `<=5cm` |
| strict | 至少 `2/4` target cases E028 strict pass |
| object no-regression | 4/4 target object pos `<=8cm`，guard `box025_p2` 不回退 |
| contact realism | contact `>=70%` 时必须同时满足 non-penetration gate；低 contact 不能被穿透 contact 替代 |

Stretch：

- `bucket007_p1` 从 E025 best deep pen `35.57%` 压到 `<=15%` 并 strict pass。
- `bucket005_s2_p2` 从 E025 best deep pen `64.53%` 压到 `<=25%`，即使未 strict pass，也证明 hard feasibility 明显优于 soft penalty。

## 停止条件

| 情况 | 动作 |
|---|---|
| smoke 中新 knobs 解析失败或旧 default 行为改变 | 先修 wiring，不跑 full |
| barrier 使 object pos `>10cm` 且 contact 下降 `>20pp` | 不继续加 scale，转 surface target / reachable surface projection |
| score cap 导致所有 samples 低分、轨迹停滞 | 降级为 barrier/contact_gate，不重复 scorecap full |
| `box025_p2` guard 回退 | 优先修 core gating 的默认/条件逻辑，暂停 E028 结论 |
| 4 个 target 中 `0/4` deep pen 降低 `>=15pp` | 宣告 hard reward 不足，下一步必须做 projection 或 CEM-level candidate filtering |

## Git 策略

E028 至少分三次提交：

1. `plan(core4d_collab): start E028 hard penetration plan`
2. `feat(core4d_collab): add E028 hard penetration controls`
3. `log(core4d_collab): record E028 hard penetration results`

如果实现过程中发现需要 mesh-level projection 或改 scene XML，另起 E028b plan，不把范围混入本实验。
