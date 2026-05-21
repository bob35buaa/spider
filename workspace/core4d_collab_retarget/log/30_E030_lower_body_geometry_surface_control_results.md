# E030 results: Lower-body geometry / surface-control repair

日期：2026-05-21

分支：`exp/core4d-collab-retarget-e030-geometry-surface-control`

## 目标

E030 验证 E027/E028/E029 之后的下一条主线：更细 lower-body collision proxy 加已有 surface/contact/penetration gates，是否能修复 `box025_p1`、`bucket007_p2` 这类 retarget geometry 可疑 case，并用 `box023_p1/p2`、`bucket005_s2_p1`、`box025_p2` 做 surface/guard 诊断。

第一版不改 `spider/` 核心代码，只复用已有默认关闭 knobs 和 derived scene 机制。成功必须满足：不删除 leg/object collision pairs、不靠高 penetration 获得 contact、不破坏 clean guard。

## 结果路径

| 产物 | 路径 |
|---|---|
| Plan | `workspace/core4d_collab_retarget/plan/35_E030_lower_body_geometry_surface_control_plan.md` |
| Variants | `workspace/core4d_collab_retarget/scripts/E030/variants.tsv` |
| Manifest | `workspace/core4d_collab_retarget/results/E030/manifest.tsv` |
| Train scripts | `workspace/core4d_collab_retarget/scripts/train/train_E030.sh`, `workspace/core4d_collab_retarget/scripts/run_E030_remote.sh` |
| Eval script | `workspace/core4d_collab_retarget/scripts/eval/eval_E030.py` |
| Results | `workspace/core4d_collab_retarget/results/E030/` |
| Comparison | `workspace/core4d_collab_retarget/results/E030/comparison.csv` |
| Baseline delta | `workspace/core4d_collab_retarget/results/E030/baseline_delta.csv` |
| Ref interference | `workspace/core4d_collab_retarget/results/E030/ref_interference.csv` |
| Sim interference | `workspace/core4d_collab_retarget/results/E030/sim_interference.csv` |
| Aggregate | `workspace/core4d_collab_retarget/results/E030/aggregate_summary.json` |
| Online videos | `workspace/core4d_collab_retarget/results/E030/online_video/` |
| Keyframes | `workspace/core4d_collab_retarget/results/E030/keyframes/` |
| Scene snapshots | `workspace/core4d_collab_retarget/results/E030/scene_snapshot/` |
| Logs | `logs/core4d_collab_retarget/E030/` |

## 执行命令

```bash
.venv/bin/python -m py_compile \
  workspace/core4d_collab_retarget/scripts/E030/generate_e030_assets.py \
  workspace/core4d_collab_retarget/scripts/eval/eval_E030.py

bash -n \
  workspace/core4d_collab_retarget/scripts/run_E030_preprocess.sh \
  workspace/core4d_collab_retarget/scripts/train/train_E030.sh \
  workspace/core4d_collab_retarget/scripts/train/train_E030_remote_tmux.sh \
  workspace/core4d_collab_retarget/scripts/run_E030_remote.sh \
  workspace/core4d_collab_retarget/scripts/pull_E030_remote_results.sh

bash workspace/core4d_collab_retarget/scripts/run_E030_preprocess.sh

RUN_TIMEOUT_SECONDS=600 RUN_STALL_TIMEOUT_SECONDS=180 \
  bash workspace/core4d_collab_retarget/scripts/train/train_E030.sh smoke 0

RUN_TIMEOUT_SECONDS=2400 RUN_STALL_TIMEOUT_SECONDS=300 \
  bash workspace/core4d_collab_retarget/scripts/train/train_E030.sh local 0

RUN_TIMEOUT_SECONDS=2400 RUN_STALL_TIMEOUT_SECONDS=300 \
  bash workspace/core4d_collab_retarget/scripts/run_E030_remote.sh

bash workspace/core4d_collab_retarget/scripts/pull_E030_remote_results.sh
```

远端独立 worktree 需要额外链接 ignored source artifacts。中途修复并提交了三次远程 worktree 软链问题：

```text
b360f9e fix(core4d_collab): link E030 remote worktree artifacts
16bdc9a fix(core4d_collab): link E030 remote source result dirs
029b486 fix(core4d_collab): link E030 remote source tasks
```

远端 GPU1 的 `E030_box023_p2_surface_hold_gate` 在 `100/272` 后被 `300s` stall watchdog 误杀。该分支由本地 fallback 完成：

```bash
RUN_TIMEOUT_SECONDS=3600 RUN_STALL_TIMEOUT_SECONDS=900 \
  bash workspace/core4d_collab_retarget/scripts/train/train_E030.sh one 0 E030_box023_p2_surface_hold_gate

.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E030.py --all
```

## 执行状态

- [x] Plan：`plan/35_E030_lower_body_geometry_surface_control_plan.md`
- [x] Implementation：E030 第一版脚本/overrides/derived scenes 完成；没有核心 `spider/` 改动。
- [x] Smoke：6/6 variants 4-step smoke 通过。
- [x] Full：6/6 variants 得到 full NPZ + MP4。
- [x] Remote：远端 3 条成功；`box023_p2` 远端误杀，由本地 fallback 完成。
- [x] Eval：`eval_E030.py --all` 完成，`num_results=6`。
- [x] Visual：6/6 online MP4/keyframes 到位，并人工复核关键帧。

## 量化结果

读表规则：

- Contact 5cm 越高越好。
- Deep pen / Max pen / Object pos / Ref-Sim leg artifact 越低越好。
- Fall 为 `true` 表示检测到 fall。
- `Result` 是 E030 对应 role 的 success/signal/guard 判定。

| Variant | Case | Role | Contact 5cm | Deep pen | Max pen | Obj pos | Fall | Ref leg int. | Sim leg int. | Result |
|---|---|---|---:|---:|---:|---:|---|---:|---:|---|
| `E030_box025_p1_tinygeom_surface_gate` | `box025_p1` | target geometry | `2.50%` | `0.00%` | `0.68cm` | `7.43cm` | true | `21.37%` | `8.06%` | fail |
| `E030_bucket007_p2_tinygeom_surface_gate` | `bucket007_p2` | target geometry | `57.71%` | `0.00%` | `1.89cm` | `5.57cm` | false | `43.68%` | `24.74%` | fail |
| `E030_box023_p1_surface_hold_gate` | `box023_p1` | diagnostic surface | `29.32%` | `9.59%` | `2.70cm` | `4.30cm` | false | `0.00%` | `0.00%` | no signal |
| `E030_box023_p2_surface_hold_gate` | `box023_p2` | diagnostic surface | `8.33%` | `0.00%` | `0.67cm` | `4.27cm` | true | `0.00%` | `12.50%` | no signal |
| `E030_bucket005_s2_p1_leg_guard_surface` | `bucket005_s2_p1` | shortcut guard | `99.47%` | `94.31%` | `5.69cm` | `3.76cm` | false | `30.41%` | `5.41%` | rejected |
| `E030_box025_p2_guard_surface` | `box025_p2` | clean guard | `0.00%` | `0.00%` | `0.00cm` | `5.68cm` | false | `22.98%` | `0.00%` | fail |

Aggregate:

| Metric | Value |
|---|---:|
| `num_results` | `6` |
| `num_target_geometry_success` | `0/2` |
| `num_diagnostic_surface_signal` | `0/2` |
| `num_clean_guard_pass` | `0/1` |
| `num_shortcut_guard_pass` | `0/1` |
| `num_no_pair_deletion_pass` | `6/6` |
| `num_E030_success` | `0/6` |
| `best_target_contact5_pct` | `57.71%` |
| `best_target_ref_leg_interference_pct` | `21.37%` |

Baseline deltas:

| Case | E030 contact | Baseline contact | ΔContact | E030 deep pen | Baseline deep pen | Ref int. change |
|---|---:|---:|---:|---:|---:|---:|
| `box025_p1` | `2.50%` | `66.07%` | `-63.57pp` | `0.00%` | `5.06%` | `66.53% -> 21.37%` |
| `bucket007_p2` | `57.71%` | `29.75%` | `+27.96pp` | `0.00%` | `18.42%` | `66.32% -> 43.68%` |
| `box023_p1` | `29.32%` | `25.30%` | `+4.02pp` | `9.59%` | `0.00%` | `0.00% -> 0.00%` |
| `box023_p2` | `8.33%` | `28.57%` | `-20.24pp` | `0.00%` | `3.33%` | `0.00% -> 0.00%` |
| `bucket005_s2_p1` | `99.47%` | `97.59%` | `+1.87pp` | `94.31%` | `88.15%` | `30.41% -> 30.41%` |
| `box025_p2` | `0.00%` | `86.93%` | `-86.93pp` | `0.00%` | `0.00%` | `22.98% -> 22.98%` |

## 可视化观察

关键帧路径：

- `workspace/core4d_collab_retarget/results/E030/keyframes/E030_box025_p1_tinygeom_surface_gate/f160.jpg`
- `workspace/core4d_collab_retarget/results/E030/keyframes/E030_box025_p1_tinygeom_surface_gate/f204.jpg`
- `workspace/core4d_collab_retarget/results/E030/keyframes/E030_bucket007_p2_tinygeom_surface_gate/f160.jpg`
- `workspace/core4d_collab_retarget/results/E030/keyframes/E030_bucket007_p2_tinygeom_surface_gate/f180.jpg`
- `workspace/core4d_collab_retarget/results/E030/keyframes/E030_box023_p1_surface_hold_gate/f160.jpg`
- `workspace/core4d_collab_retarget/results/E030/keyframes/E030_box023_p1_surface_hold_gate/f204.jpg`
- `workspace/core4d_collab_retarget/results/E030/keyframes/E030_box023_p2_surface_hold_gate/f160.jpg`
- `workspace/core4d_collab_retarget/results/E030/keyframes/E030_box023_p2_surface_hold_gate/f180.jpg`
- `workspace/core4d_collab_retarget/results/E030/keyframes/E030_box023_p2_surface_hold_gate/f204.jpg`
- `workspace/core4d_collab_retarget/results/E030/keyframes/E030_bucket005_s2_p1_leg_guard_surface/f160.jpg`
- `workspace/core4d_collab_retarget/results/E030/keyframes/E030_bucket005_s2_p1_leg_guard_surface/f204.jpg`
- `workspace/core4d_collab_retarget/results/E030/keyframes/E030_box025_p2_guard_surface/f160.jpg`
- `workspace/core4d_collab_retarget/results/E030/keyframes/E030_box025_p2_guard_surface/f204.jpg`

实际观察：

- `box025_p1` 在 `f160/f204` 中 sim 侧物体与 robot hand 没有形成有效跟随接触，橙色 EEF marker 掉到物体下方或视野边缘；虽然 deep penetration 被压低，contact 只剩 `2.50%`，并且 fall=true。
- `bucket007_p2` 在 `f160/f180` 中姿态看起来接近 ref，但腿/身体仍绕 bucket 形成 artifact；contact 提升到 `57.71%`，但 ref interference `43.68%`、sim interference `24.74%`，仍不是 geometry 修复。
- `box023_p1` 在 `f160` 有短暂表面接近，到 `f204` 已与 box 分离；contact 只比 baseline 高 `4.02pp`，低于 `>=10pp` diagnostic signal 门槛。
- `box023_p2` 在 `f160/f180/f204` 中 sim 侧翻到 box 上或离开物体，contact `8.33%` 且 fall=true；surface-hold gate 没有稳定住可用接触。
- `bucket005_s2_p1` 在 `f160/f204` 中手/脚 marker 与 bucket 表面交叠，量化为 contact `99.47%`、deep pen `94.31%`；这是 high-contact high-penetration 坏解，被 E030 guard 拒绝。
- `box025_p2` 在 `f160/f204` 中身体与 box 相对姿态尚可，但 EEF marker 未稳定贴在目标接触面；contact 从 baseline `86.93%` 掉到 `0.00%`，clean guard 明确回退。

## Claims 验证

| Claim | 结果 | 说明 |
|---|---|---|
| C1: 更细 lower-body proxy geometry 能进一步降低 target ref interference | 部分通过但不足 | `box025_p1` 从 `66.53%` 降到 `21.37%`，`bucket007_p2` 从 `66.32%` 降到 `43.68%`；均未到 `<15%`，也未形成 success |
| C2: geometry 改善必须不制造 sim leg/body shortcut | 失败 | `bucket007_p2` sim interference `24.74%`，`box025_p1` contact/fall 失败；没有 target geometry success |
| C3: `box023_p1/p2` 低接触不是 phase shift，应验证 surface/side-control 信号 | 通过负结论 | `box023_p1` 只提升 `+4.02pp`，`box023_p2` 下降 `-20.24pp`；两者都低于 `>=10pp` 门槛 |
| C4: E030 不破坏 clean guard | 失败 | `box025_p2` object/no-fall/penetration 过，但 contact 从 `86.93%` 掉到 `0.00%` |
| C5: bucket/leg shortcut guard 能识别 high-contact high-penetration 坏解 | 通过 | `bucket005_s2_p1` contact `99.47%`，但 deep pen `94.31%`、max pen `5.69cm`，未进入 success |

## 数据可用性结论

E030 不新增任何 `discard_from_success_denominator=True` case。理由：

- `box025_p1`、`bucket007_p2` 的 ref leg/object interference 仍是 retarget geometry caveat，但 E030 只是证明“XML shrink + 现有 gates”不足，不足以反向证明 raw data 应弃用。
- `box023_p1/p2` ref interference 为 `0%`，失败更像 surface target / control objective 不足；不能归为 mocap 质量差。
- `bucket005_s2_p1` 高 contact 高 penetration 是 algorithmic shortcut 证据，正好说明需要 candidate-level rejection/projection，不是数据弃用证据。
- `box025_p2` clean guard 回退由 E030 gate 组合造成，不能作为数据问题。
- 当前唯一主分母弃用 case 仍是 E027 已判定的 `desk021_p1`。

## 结论

E030 是明确的负结果。更细 lower-body proxy 能降低一部分 ref interference，尤其 `box025_p1`，但没有转化为可用 contact；`bucket007_p2` contact 有提升，却仍伴随过高 ref/sim leg artifact；`box023_p1/p2` 的 surface-hold gate 没有产生诊断信号；`box025_p2` clean guard 回退说明这套 gate 组合不能直接推广。

因此，下一步不应继续做 case-specific XML 微调、普通 contact gain sweep 或单纯提高 barrier scale。需要进入 runtime surface target / candidate-level rejection-projection：

1. 为 object mesh/box/bucket 建立 explicit surface target 或 signed-distance/contact normal target，让 EEF 追踪“正确表面点”，而不是只靠动态 contact mask 和 support proxy。
2. 在 CEM candidate 层拒绝或投影 high-penetration、leg/body artifact、fall candidate；避免 reward 事后惩罚无力阻止坏解。
3. 对 `bucket001_p1`、`box023_p1/p2` 做 reachability/support timing audit，判断 support proxy motion 与 robot reachable surface 是否脱耦。
4. 继续沿用 E027 多证据数据协议：只有 raw mocap、OmniRetarget kinematics、visual、metric 多方面同时支持质量差时才弃用 case；当前 E030 不触发新的弃用。
