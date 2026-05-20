# E028 results: Hard no-penetration / surface feasibility

日期：2026-05-21

分支：`exp/core4d-collab-retarget-e028-hard-penetration`

## 目标

按 `plan/33_E028_hard_no_penetration_surface_feasibility_plan.md`，E028 处理 E025 之后仍未解决的 high-contact penetration shortcut：

- `bucket005_s2_p1`
- `bucket005_s2_p2`
- `bucket007_p1`
- `bucket001_p2`

核心问题不是 contact 数字低，而是 CEM 会选择“手或机器人进入物体内部”来换高 contact preservation。E028 因此新增默认关闭的 hard feasibility knobs：SDF barrier、score cap、contact penetration gate、staged contact gate。`box025_p2` 作为 guard，验证已经 pass 的 case 不应回退。

## 结果路径

| 产物 | 路径 |
|---|---|
| Plan | `workspace/core4d_collab_retarget/plan/33_E028_hard_no_penetration_surface_feasibility_plan.md` |
| Variants | `workspace/core4d_collab_retarget/scripts/E028/variants.tsv` |
| Reward implementation | `spider/config.py`, `spider/simulators/mjwp.py` |
| Train scripts | `workspace/core4d_collab_retarget/scripts/train/train_E028.sh`, `workspace/core4d_collab_retarget/scripts/run_E028_remote.sh` |
| Eval script | `workspace/core4d_collab_retarget/scripts/eval/eval_E028.py` |
| Results | `workspace/core4d_collab_retarget/results/E028/` |
| Comparison | `workspace/core4d_collab_retarget/results/E028/comparison.csv` |
| Baseline delta | `workspace/core4d_collab_retarget/results/E028/baseline_delta.csv` |
| Aggregate | `workspace/core4d_collab_retarget/results/E028/aggregate_summary.json` |
| Online videos | `workspace/core4d_collab_retarget/results/E028/online_video/` |
| Keyframes | `workspace/core4d_collab_retarget/results/E028/keyframes/` |
| Scene snapshot | `workspace/core4d_collab_retarget/results/E028/scene_snapshot/` |

## 执行命令

```bash
.venv/bin/python -m py_compile \
  spider/config.py \
  spider/simulators/mjwp.py \
  workspace/core4d_collab_retarget/scripts/E028/generate_e028_overrides.py \
  workspace/core4d_collab_retarget/scripts/eval/eval_E028.py

bash -n \
  workspace/core4d_collab_retarget/scripts/run_E028_preprocess.sh \
  workspace/core4d_collab_retarget/scripts/train/train_E028.sh \
  workspace/core4d_collab_retarget/scripts/train/train_E028_remote_tmux.sh \
  workspace/core4d_collab_retarget/scripts/run_E028_remote.sh \
  workspace/core4d_collab_retarget/scripts/pull_E028_remote_results.sh

bash workspace/core4d_collab_retarget/scripts/run_E028_preprocess.sh

RUN_TIMEOUT_SECONDS=600 RUN_STALL_TIMEOUT_SECONDS=180 \
  bash workspace/core4d_collab_retarget/scripts/train/train_E028.sh smoke 0

RUN_TIMEOUT_SECONDS=2400 RUN_STALL_TIMEOUT_SECONDS=300 \
  bash workspace/core4d_collab_retarget/scripts/train/train_E028.sh local 0

bash workspace/core4d_collab_retarget/scripts/run_E028_remote.sh
bash workspace/core4d_collab_retarget/scripts/pull_E028_remote_results.sh

RUN_TIMEOUT_SECONDS=2400 RUN_STALL_TIMEOUT_SECONDS=600 \
  bash workspace/core4d_collab_retarget/scripts/train/train_E028.sh one 0 E028_bucket001_p2_contact_gate_m02

.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E028.py --all
```

远程主 worktree 因 E026 遗留的 modified/untracked 文件无法直接 `git switch`。未清理远程主目录，改用独立 worktree `/home/xiayb/pHRI_workspace/spider_e028_20260521_041811`，并 symlink 主 repo 的 `.venv`、`workspace/core4d_collab_retarget/results`、`logs`。

## 执行状态

- [x] Plan：E028 计划写入 `plan/33_E028_hard_no_penetration_surface_feasibility_plan.md`。
- [x] Implementation：新增默认关闭的 E028 knobs，旧实验默认行为不变。
- [x] Preprocess：6 个 overrides 与 manifest 生成。
- [x] Smoke：6/6 4-step smoke 通过；smoke 只验证 wiring。
- [x] Full：6/6 full variants 完成并写出 NPZ + MP4。
- [x] Eval：`eval_E028.py --all` 完成，`num_results=6`。
- [x] Visual：6/6 online MP4 与 keyframes 到位，并已人工复核关键帧。

## 量化结果

读表规则：

- Contact 5cm 越高越好。
- Deep pen、Max pen、Object pos 越低越好。
- `ΔDeep` 是相对进入 E028 前 best baseline 的 deep penetration 改善，正数表示 E028 降低了穿透。

| Variant | Case | 机制 | Contact 5cm | Deep pen | Hand pen | Leg pen | Max pen | Object | Ori | Fall | Strict | ΔDeep |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|
| `E028_bucket007_p1_barrier_quad_m02` | `bucket007_p1` | barrier | `1.11%` | `0.67%` | `0.00%` | `0.67%` | `2.69cm` | `5.50cm` | `7.46deg` | true | false | `+34.90pp` |
| `E028_bucket005_s2_p2_barrier_quad_m02` | `bucket005_s2_p2` | barrier | `97.80%` | `88.18%` | `88.18%` | `0.00%` | `8.78cm` | `6.01cm` | `8.72deg` | false | false | `-23.65pp` |
| `E028_bucket005_s2_p1_contact_gate_m02` | `bucket005_s2_p1` | contact gate + barrier | `99.20%` | `92.89%` | `92.89%` | `0.00%` | `6.33cm` | `4.01cm` | `4.62deg` | false | false | `-4.74pp` |
| `E028_bucket001_p2_contact_gate_m02` | `bucket001_p2` | contact gate + barrier | `2.81%` | `0.00%` | `0.00%` | `0.00%` | `1.70cm` | `3.54cm` | `8.70deg` | false | false | `+59.60pp` |
| `E028_bucket007_p1_scorecap_m01` | `bucket007_p1` | score cap | `82.66%` | `67.79%` | `67.79%` | `0.00%` | `10.58cm` | `6.52cm` | `8.17deg` | false | false | `-32.21pp` |
| `E028_box025_p2_guard_barrier_m02` | `box025_p2` | guard barrier | `0.00%` | `0.00%` | `0.00%` | `0.00%` | `0.00cm` | `5.68cm` | `2.12deg` | false | false | `+0.00pp` |

Aggregate:

| Metric | Value |
|---|---:|
| `num_results` | `6` |
| `num_E028_strict_success` | `0` |
| `num_guard_results` | `1` |
| `num_guard_strict_success` | `0` |
| `num_target_results` | `5` |
| `num_target_E028_strict_success` | `0` |
| `num_target_penetration_guard_pass` | `2` |
| `num_target_object_no_regression_pass` | `5` |
| `num_target_no_fall_pass` | `4` |
| `mean_target_deep_pen_improvement_pp` | `6.78pp` |

`num_target_results=5` 是因为 `bucket007_p1` 有 barrier 与 score-cap 两个 target variants。按 4 个 target case 取 case-best deep penetration 时：

| Case | Best E028 variant by deep pen | Deep pen | Max pen | Contact 5cm | Object | Fall | Strict | ΔDeep |
|---|---|---:|---:|---:|---:|---|---|---:|
| `bucket005_s2_p1` | `E028_bucket005_s2_p1_contact_gate_m02` | `92.89%` | `6.33cm` | `99.20%` | `4.01cm` | false | false | `-4.74pp` |
| `bucket005_s2_p2` | `E028_bucket005_s2_p2_barrier_quad_m02` | `88.18%` | `8.78cm` | `97.80%` | `6.01cm` | false | false | `-23.65pp` |
| `bucket007_p1` | `E028_bucket007_p1_barrier_quad_m02` | `0.67%` | `2.69cm` | `1.11%` | `5.50cm` | true | false | `+34.90pp` |
| `bucket001_p2` | `E028_bucket001_p2_contact_gate_m02` | `0.00%` | `1.70cm` | `2.81%` | `3.54cm` | false | false | `+59.60pp` |

Case-best summary:

| Criterion | Result |
|---|---:|
| Mean case-best deep improvement | `16.53pp` |
| Deep pen `<=15%` and max pen `<=5cm` | `2/4` |
| Object pos `<=8cm` | `4/4` |
| No fall | `3/4` |
| Contact `>=70%` on best-deep variant | `2/4` |
| Strict success | `0/4` |

## 与基线对比：穿透下降是否换来了真实接触

E028 不是简单负结果，而是把失败模式分清楚了：

| Case / variant | Baseline contact | E028 contact | Baseline deep pen | E028 deep pen | 判断 |
|---|---:|---:|---:|---:|---|
| `bucket007_p1_barrier` | `84.13%` | `1.11%` | `35.57%` | `0.67%` | 穿透被压下，但 contact 几乎消失且 fall；不是 surface-contact success |
| `bucket001_p2_contact_gate` | `77.53%` | `2.81%` | `59.60%` | `0.00%` | 穿透被压下，但接触丢失；object 还能跟踪，说明 reward 可避开物体而非贴表面 |
| `bucket005_s2_p1_contact_gate` | `97.59%` | `99.20%` | `88.15%` | `92.89%` | 高 contact 仍主要来自 hand-object 内部 shortcut |
| `bucket005_s2_p2_barrier` | `96.42%` | `97.80%` | `64.53%` | `88.18%` | barrier 反而更差，未阻止 bucket 内部穿透 |
| `bucket007_p1_scorecap` | `84.13%` | `82.66%` | `35.57%` | `67.79%` | score cap 保持 contact，但穿透明显回退 |
| `box025_p2_guard` | `86.93%` | `0.00%` | `0.00%` | `0.00%` | object/penetration 不坏，但 pass case contact 完全回退 |

核心结论：第一版 hard feasibility 有两种失败模式。强 barrier 能让优化器“不要碰物体”，从而降低穿透；弱/score-cap 路径则仍允许“高 contact + 内部穿透”。当前没有形成“高 contact + 非穿透 surface contact”的中间解。

## Claims 验证

| Claim | 结果 |
|---|---|
| C1: hard feasibility 显著压低 bucket penetration shortcut | 不通过。按 5 个 target variants 平均只改善 `6.78pp`；按 4 个 target case-best 也只有 `16.53pp`，低于 `>=25pp`。虽然 `2/4` case deep pen `<=15%`，但这两个都伴随 contact collapse 或 fall |
| C2: 不能靠丢 object transport 达成 | 通过但有 caveat。4/4 target case-best object pos `<=8cm`；object 没丢，但 `bucket007_p1` fall、`bucket001_p2` contact collapse，说明 object no-regression 不足以代表真实交互 |
| C3: surface feasibility 与 contact preservation 同时报 | 通过。每个 variant 都输出 contact、hand/leg/robot penetration、max penetration、object、fall、strict gate |
| C4: `box025_p2` guard 不回退 | 不通过。guard deep pen 仍 `0%`、object `5.68cm`、no fall，但 contact 从 `86.93%` 降到 `0%`，不满足 contact `>=70%` |
| C5: 不把 E027 数据质量 caveat 当算法收益 | 通过。`bucket001_p2` 保留 `usable_with_caveat`；E028 没把它改成弃用，也没有把 contact collapse 解释为数据问题 |

## 可视化观察

关键帧来自 `workspace/core4d_collab_retarget/results/E028/keyframes/`，与 online MP4 一致。

| Variant | 观察 |
|---|---|
| `E028_bucket007_p1_barrier_quad_m02` | robot 很快与 bucket 分离，后段出现明显姿态失稳/摔倒；低穿透来自避开物体，不是有效表面接触 |
| `E028_bucket007_p1_scorecap_m01` | robot 仍贴近 bucket，视频上能看到持续接触，但指标显示 hand deep penetration `67.79%`，score cap 没有真正阻断内部接触 |
| `E028_bucket005_s2_p1_contact_gate_m02` | sim 能跟随 bucket 运动，但手/手臂位置仍贴在 bucket 内部区域；高 contact 与 `92.89%` hand deep penetration 一致 |
| `E028_bucket005_s2_p2_barrier_quad_m02` | 与 p1 类似，object tracking 维持，但接触集中在 bucket 侧壁/内部，未形成干净外表面支撑 |
| `E028_bucket001_p2_contact_gate_m02` | bucket 能保持大致位置，但 robot 与 bucket 的有效表面接触很少；低穿透主要来自 contact 被切断 |
| `E028_box025_p2_guard_barrier_m02` | robot 与 box 基本不建立有效接触，解释了 contact `0%`；这是明确 guard regression |

## 遇到的错误

| 错误 | 尝试次数 | 解决方案 |
|---|---:|---|
| 远程主 worktree 有 E026 modified/untracked 文件，`git switch` 会覆盖本地改动 | 1 | 不清理远程主目录，创建独立 worktree `/home/xiayb/pHRI_workspace/spider_e028_20260521_041811` 跑 E028 |
| 远程 GPU0 `E028_bucket001_p2_contact_gate_m02` 在 `224/244` stall，且本地已有 smoke 小 NPZ 会阻止 rsync 覆盖 | 1 | 修 `pull_E028_remote_results.sh` 去掉 `--ignore-existing`；删除 stale smoke NPZ；本地单条 full 重跑完成 |
| `eval_E028.py --all` 曾被单条重试覆盖为 1 行 comparison | 1 | 在 6 个 full NPZ 到位后重新执行 `eval_E028.py --all`，恢复 6/6 汇总 |

## 结论

E028 的工程实现和实验执行完成，但实验结论是负结果：

1. Hard barrier / contact gate 确实能在 `bucket007_p1`、`bucket001_p2` 上压低 penetration，但代价是接触坍塌，甚至出现 fall。
2. `bucket005_s2_p1/p2` 的高接触高穿透没有被第一版 SDF barrier/contact gate 修掉，说明当前 box-SDF reward 路径仍不足以表达“手贴表面但不进入物体”的可达目标。
3. `score_cap` 不是当前优先方向：它保留了 contact，却让 penetration 明显变差。
4. `box025_p2` guard 回退说明 E028 第一版机制不能直接并入主 pipeline。

下一步不应继续简单加大 barrier scale。更合理的方向是新开 E028b 或并入 E030：

- 做 object-specific surface target：把 contact target 从“靠近 object center/box 内部区域”改成“靠近最近表面点，且 SDF 非负”。
- 做 CEM candidate-level rejection/projection：在 sample 级别丢弃或投影进入物体内部的候选，而不是只在 reward 末端扣分。
- bucket005 优先检查 collision box / bucket 内外表面语义：当前 metric 与视频都显示优化器仍能把手放到 bucket 内部。
- `bucket001_p2` 保留 E027 caveat，但不是弃用对象；它说明 no-penetration 可行，缺的是保持 surface contact。
- `box025_p2` guard 必须作为 E028b/E030 的第一类 regression test，contact 不能再从 pass 退到 `0%`。
