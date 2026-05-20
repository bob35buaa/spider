# E029 results: Bucket001 stability / posture-valid contact control

日期：2026-05-21

分支：`exp/core4d-collab-retarget-e029-stability-control`

## 目标

E029 针对 `bucket001_p1` 在 E024 中持续 fall / contact `0%` 的问题，验证更硬的 posture feasibility 是否能替代继续扫线性 stability penalty。核心新增机制均默认关闭：

- pelvis upright barrier
- pelvis score cap
- root tilt penalty
- foot support penalty
- posture-valid contact gate

`bucket001_p2` 与 `box025_p2` 是 guard：前者验证 E024 p2 no-fall/contact 不被新 gate 破坏，后者验证已通过的 strict-ish case 不回退。

## 结果路径

| 产物 | 路径 |
|---|---|
| Plan | `workspace/core4d_collab_retarget/plan/34_E029_bucket001_stability_control_plan.md` |
| Variants | `workspace/core4d_collab_retarget/scripts/E029/variants.tsv` |
| Manifest | `workspace/core4d_collab_retarget/results/E029/manifest.tsv` |
| Reward implementation | `spider/config.py`, `spider/simulators/mjwp.py` |
| Train scripts | `workspace/core4d_collab_retarget/scripts/train/train_E029.sh`, `workspace/core4d_collab_retarget/scripts/run_E029_remote.sh` |
| Eval script | `workspace/core4d_collab_retarget/scripts/eval/eval_E029.py` |
| Results | `workspace/core4d_collab_retarget/results/E029/` |
| Comparison | `workspace/core4d_collab_retarget/results/E029/comparison.csv` |
| Baseline delta | `workspace/core4d_collab_retarget/results/E029/baseline_delta.csv` |
| Aggregate | `workspace/core4d_collab_retarget/results/E029/aggregate_summary.json` |
| Online videos | `workspace/core4d_collab_retarget/results/E029/online_video/` |
| Keyframes | `workspace/core4d_collab_retarget/results/E029/keyframes/` |
| Logs | `logs/core4d_collab_retarget/E029/` |

## 执行命令

```bash
.venv/bin/python -m py_compile \
  spider/config.py \
  spider/simulators/mjwp.py \
  workspace/core4d_collab_retarget/scripts/E029/generate_e029_overrides.py \
  workspace/core4d_collab_retarget/scripts/eval/eval_E029.py

bash -n \
  workspace/core4d_collab_retarget/scripts/run_E029_preprocess.sh \
  workspace/core4d_collab_retarget/scripts/train/train_E029.sh \
  workspace/core4d_collab_retarget/scripts/train/train_E029_remote_tmux.sh \
  workspace/core4d_collab_retarget/scripts/run_E029_remote.sh \
  workspace/core4d_collab_retarget/scripts/pull_E029_remote_results.sh

bash workspace/core4d_collab_retarget/scripts/run_E029_preprocess.sh

RUN_TIMEOUT_SECONDS=600 RUN_STALL_TIMEOUT_SECONDS=180 \
  bash workspace/core4d_collab_retarget/scripts/train/train_E029.sh smoke 0

RUN_TIMEOUT_SECONDS=2400 RUN_STALL_TIMEOUT_SECONDS=300 \
  bash workspace/core4d_collab_retarget/scripts/train/train_E029.sh local 0

RUN_TIMEOUT_SECONDS=2400 RUN_STALL_TIMEOUT_SECONDS=300 \
  bash workspace/core4d_collab_retarget/scripts/run_E029_remote.sh
```

远端主 worktree 仍有 E026 dirty/untracked 文件，未清理、不覆盖，改用独立 worktree：

```text
/home/xiayb/pHRI_workspace/spider_e029_20260521_055150
```

远端 GPU1 的 `E029_bucket001_p2_guard_posture_gate` 在 `162/244` 后日志不再刷新，watchdog 判定 stall：

```text
[06:31:26] ERROR E029_bucket001_p2_guard_posture_gate stalled: no log update for 300s
```

本地 GPU 空闲后补跑同一 variant，放宽 stall 阈值：

```bash
RUN_TIMEOUT_SECONDS=3600 RUN_STALL_TIMEOUT_SECONDS=900 \
  bash workspace/core4d_collab_retarget/scripts/train/train_E029.sh one 0 E029_bucket001_p2_guard_posture_gate
```

回收远端结果时排除了远端失败的 p2 artifacts，保留本地 fallback p2：

```bash
rsync -av --exclude='*E029_bucket001_p2_guard_posture_gate*' \
  spider-remote:/home/xiayb/pHRI_workspace/spider/workspace/core4d_collab_retarget/results/E029/ \
  workspace/core4d_collab_retarget/results/E029/

rsync -av --exclude='E029_bucket001_p2_guard_posture_gate.log' \
  spider-remote:/home/xiayb/pHRI_workspace/spider/logs/core4d_collab_retarget/E029/ \
  logs/core4d_collab_retarget/E029/

.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E029.py --all
```

## 执行状态

- [x] Plan：`plan/34_E029_bucket001_stability_control_plan.md`
- [x] Implementation：新增默认关闭的 E029 posture/stability knobs，旧默认行为不变。
- [x] Smoke：6/6 variants 4-step smoke 通过。
- [x] Full：6/6 variants 得到 full NPZ + MP4。
- [x] Remote：远端 3 条成功，1 条 p2 guard stall；p2 guard 由本地 fallback 完成。
- [x] Eval：`eval_E029.py --all` 完成，`num_results=6`。
- [x] Visual：6/6 online MP4/keyframes 到位，并人工复核关键帧。

## 量化结果

读表规则：

- Contact 5cm 越高越好。
- Deep pen / Max pen / Object pos 越低越好。
- `ΔPelvis`、`ΔContact` 相对 source baseline；正的 `ΔPelvis` 表示 pelvis min 变高。

| Variant | Case | 机制 | Pelvis min | ΔPelvis | Contact 5cm | ΔContact | Deep pen | Max pen | Obj pos | Fall | Stability | Useful | Guard |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| `E029_bucket001_p1_upright_barrier_t055` | `bucket001_p1` | upright barrier | `0.584m` | `+0.428m` | `0.00%` | `+0.00pp` | `0.00%` | `0.00cm` | `3.33cm` | false | true | false | false |
| `E029_bucket001_p1_posture_gate_t055` | `bucket001_p1` | posture gate | `0.527m` | `+0.372m` | `0.00%` | `+0.00pp` | `0.00%` | `0.00cm` | `3.33cm` | false | true | false | false |
| `E029_bucket001_p1_scorecap_t045` | `bucket001_p1` | score cap + foot support | `0.533m` | `+0.377m` | `0.00%` | `+0.00pp` | `0.00%` | `0.00cm` | `3.33cm` | false | true | false | false |
| `E029_bucket001_p1_tilt_gate_t055` | `bucket001_p1` | root tilt + foot support | `0.534m` | `+0.378m` | `0.00%` | `+0.00pp` | `0.00%` | `0.00cm` | `3.33cm` | false | true | false | false |
| `E029_bucket001_p2_guard_posture_gate` | `bucket001_p2` | p2 guard | `0.510m` | `-0.216m` | `99.44%` | `+21.91pp` | `63.64%` | `9.00cm` | `3.23cm` | false | true | false | false |
| `E029_box025_p2_guard_posture_gate` | `box025_p2` | strict-pass guard | `0.756m` | `-0.003m` | `96.73%` | `+9.80pp` | `5.78%` | `2.64cm` | `5.63cm` | false | true | false | true |

Aggregate:

| Metric | Value |
|---|---:|
| `num_results` | `6` |
| `num_target_results` | `4` |
| `num_target_stability_pass` | `4` |
| `num_target_useful_signal` | `0` |
| `num_target_p1_strict_target` | `0` |
| `best_target_pelvis_min_m` | `0.584m` |
| `best_target_contact5_pct` | `0.00%` |
| `num_guard_results` | `2` |
| `num_guard_pass` | `1` |
| `num_E029_success` | `1` |

## 可视化观察

关键帧路径：

- `workspace/core4d_collab_retarget/results/E029/keyframes/E029_bucket001_p1_upright_barrier_t055/f100.jpg`
- `workspace/core4d_collab_retarget/results/E029/keyframes/E029_bucket001_p1_upright_barrier_t055/f160.jpg`
- `workspace/core4d_collab_retarget/results/E029/keyframes/E029_bucket001_p1_upright_barrier_t055/f204.jpg`
- `workspace/core4d_collab_retarget/results/E029/keyframes/E029_bucket001_p1_posture_gate_t055/f160.jpg`
- `workspace/core4d_collab_retarget/results/E029/keyframes/E029_bucket001_p1_tilt_gate_t055/f160.jpg`
- `workspace/core4d_collab_retarget/results/E029/keyframes/E029_bucket001_p2_guard_posture_gate/f160.jpg`
- `workspace/core4d_collab_retarget/results/E029/keyframes/E029_bucket001_p2_guard_posture_gate/f204.jpg`
- `workspace/core4d_collab_retarget/results/E029/keyframes/E029_box025_p2_guard_posture_gate/f160.jpg`

实际观察：

- `bucket001_p1` 的 hard upright 类机制确实阻止了 E024 那种 pelvis 掉到 `0.16m` 的 fall；但 robot 与 bucket 分离，bucket 仍按 support proxy / object-side target 运动，hand 没有建立 5cm 接触。
- `upright_barrier` 在 `f100` 仍近似站立，但 object 已偏离手；到 `f160/f204`，robot 转为弯腰/手撑地类姿态，离 bucket 仍远。这解释了 `pelvis` 指标变好但 contact 仍 `0%`。
- `posture_gate` 与 `tilt_gate` 在 `f160` 显示相似模式：robot 保持非摔倒姿态，但把手/身体用于姿态稳定，而不是接近 bucket 表面。
- `bucket001_p2_guard` 在 `f160/f204` 显示 robot 身体斜撑、头/手部贴近或进入 bucket 区域；量化上 contact 高达 `99.44%`，但 deep penetration `63.64%`、max penetration `9.00cm`，不是 guard success。
- `box025_p2_guard` 在 `f160` 与 ref 对齐较好，contact 不退化，deep pen 虽从 `0%` 增到 `5.78%`，但仍在 guard 门限内；这是 E029 唯一完整成功项。

## Claims 验证

| Claim | 结果 | 说明 |
|---|---|---|
| C1: p1 fall 需要 hard posture feasibility | 部分通过 | 4 个机制不同 p1 full variants 都完成；hard posture 机制能修 pelvis/fall，但不能修 contact |
| C2: hard upright / score-cap 能阻止 CEM 选择趴地姿态 | 通过 | p1 pelvis min 从 E024 `0.156m` 提升到 `0.527-0.584m`，4/4 no-fall |
| C3: posture-valid contact gate 不能靠切断接触伪造稳定 | 通过负结论 | p1 4/4 contact 仍 `0%`，全部只算 stability-only partial，不算 useful signal |
| C4: object-side support proxy 不回退 | 通过 | p1 object `3.33cm`，p2 object `3.23cm`，box025 object `5.63cm`，均在门限内 |
| C5: guard 不回退 | 部分失败 | `box025_p2` guard 通过；`bucket001_p2` no-fall/object/contact 过，但 pelvis 低于 `0.55m` 且 penetration 高，guard fail |

## 结论

E029 是一个有用的负结果：

1. `bucket001_p1` 的 fall 可以被 hard posture feasibility 修掉，但 contact 仍为 `0%`。这说明 p1 不是继续加 stability scale / score cap / root tilt 就能解决；稳定后暴露出的主问题是 contact reachability、support/reference timing、或 lower-body/control feasibility。
2. `bucket001_p2` 再次复现 “高 contact + 高 penetration” 的坏解。E028 与 E029 共同说明 bucket surface contact 不能靠末端 reward gate 或普通 posture gate 闭合，需要 surface target、candidate-level rejection/projection，或更强的 reference/geometry 修复。
3. `box025_p2` guard 通过，说明 E029 的 posture gate 本身不是全局破坏性机制；问题集中在 bucket cases 的 surface/contact geometry 与 control feasibility。

## 下一步

- 不继续 E029b 形式的普通 reward sweep；p1 的 stopping condition 已触发：4 个机制 variants contact 仍 `0%`。
- 对 `bucket001_p1` 做 reference/support reachability audit：比较 ref hand、sim hand、bucket surface target、support proxy object motion，判断是否 object-side proxy 与 robot reach target 相互脱耦。
- 对 bucket cases 做 surface-contact target 或 CEM candidate rejection/projection，而不是继续提高 penetration barrier scale。
- E030 应优先处理 E027 标记的 `box025_p1` / `bucket007_p2` retarget geometry，以及 bucket001 的 surface/control feasibility，保持用户要求的数据判定原则：不能把 E029 的算法失败反向解释成数据差，除非新增 raw / annotation / kinematic / visual 多证据。
