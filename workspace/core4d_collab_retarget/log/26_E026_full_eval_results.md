# E026 full eval: OmniRetarget + Spider dynamics 13case

日期：2026-05-20 开始，2026-05-21 完成

## 背景

本轮按 `workspace/core4d_collab_retarget/task_full_eval.md` 做完整评估，目标是把 E019 统一评测框架、OmniRetarget kinematic 结果、E018b、E022-E025 优化变体，以及 E081 baseline 放到同一张 13case 账上。

两个版本：

- **P0 9case**：排除数据/retarget 问题较重的 `desk021_p1`、`box021_p1`、`box021_p2`、`bucket001_p1`。
- **P1 13case**：完整 union，包含上述 4 个问题 case。

核心输出：

- `workspace/core4d_collab_retarget/results/E026_full_eval/summary_9case.md`
- `workspace/core4d_collab_retarget/results/E026_full_eval/summary_13case.md`
- `workspace/core4d_collab_retarget/results/E026_full_eval/method_case_metrics.csv`
- `workspace/core4d_collab_retarget/results/E026_full_eval/best_dynamic_selection.csv`
- `workspace/core4d_collab_retarget/results/E026_full_eval/omni_threshold_sweep.md`
- `workspace/core4d_collab_retarget/results/E026_full_eval/visual_metric_audit.md`
- `workspace/core4d_collab_retarget/results/E026_E081_full/comparison.csv`

## Pre-eval 核对

### OmniRetarget / Holosoma kinematic 来源

当前 kinematic 轨迹来自本机 Holosoma workspace：

- `/home/ubuntu/Workspace/holosoma`
- 适配器：`workspace/core4d_collab_retarget/scripts/eval/adapters/kinematic_to_common.py`
- 评估脚本：`workspace/core4d_collab_retarget/scripts/eval/eval_holosoma_kinematic.py`
- 输出：`workspace/core4d_collab_retarget/results/holosoma_v2_kinematic/comparison.csv`

适配器读取 Holosoma 结果里的 robot `qpos`、`human_joints`、`fps`、`cost`，并结合 companion object poses 转为 E019 common schema。已补 fallback root，兼容旧 `/mnt/ali-sh-1/.../holosoma` 和当前 `/home/ubuntu/Workspace/holosoma`。

本轮重跑命令：

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl .venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_holosoma_kinematic.py --all
```

结果：OmniRetarget / Holosoma kinematic 覆盖 `12/13`，缺 `desk021_p1`，原因是 Holosoma SOCP infeasible。

### OmniRetarget 28cm contact threshold 来源

28cm 不是本轮新调出来的阈值，而是继承自 Holosoma v1 eval 实现：

- `/home/ubuntu/Workspace/holosoma/workspace/v1/scripts/eval_paper_metrics.py`
- 代码里使用 `CONTACT_RADIUS = 0.28`
- `/home/ubuntu/Workspace/holosoma/workspace/v1/README.md` 也记录了 0.28m binary contact 口径

这个指标在 object-local frame 下比较 demo wrist 与 robot wrist 是否都进入同一个 28cm 接触半径，并按 `preservation = 1 - miss_frames / T` 计算。它能复现原实现口径，但作为 paper-facing 指标需要 caveat：大物体上可能因为 demo wrist 根本不进入阈值而变成 trivial，小物体上才更有判别力。

本轮扫阈值：

| Radius cm | N | Mean preservation % | Mean demo contact frames | Min preservation % |
|---|---:|---:|---:|---:|
| 5 | 12 | 100.00 | 0.00 | 100.00 |
| 10 | 12 | 100.00 | 0.00 | 100.00 |
| 15 | 12 | 87.15 | 15.67 | 29.51 |
| 20 | 12 | 75.25 | 32.58 | 27.87 |
| 28 | 12 | 53.59 | 61.92 | 21.33 |
| 35 | 12 | 70.08 | 65.17 | 18.67 |
| 50 | 12 | 78.91 | 91.58 | 33.87 |

结论：28cm 可以作为“复现 Holosoma/OmniRetarget eval code”的口径，但不能单独作为强物理接触证据；需要和 5cm SPIDER-side contact、penetration、video audit 一起解释。

## E081 full rerun

旧 E081 只有 2 个 legacy rows：`box025_p2` 与 `box023_p2`，覆盖不足。因此本轮新增 E026_E081 full rerun：

- variants：`workspace/core4d_collab_retarget/scripts/E026/e081_full_variants.tsv`
- 本地/远端脚本：
  - `workspace/core4d_collab_retarget/scripts/run_E026_e081_preprocess.sh`
  - `workspace/core4d_collab_retarget/scripts/train/train_E026_e081_full.sh`
  - `workspace/core4d_collab_retarget/scripts/run_E026_e081_remote.sh`
  - `workspace/core4d_collab_retarget/scripts/pull_E026_e081_remote_results.sh`
- 输出：`workspace/core4d_collab_retarget/results/E026_E081_full/`

执行中修了两个工程问题：

- `workspace/core4d/scripts/eval/eval_E081.py` 将 `RESULTS` / `VARIANTS_FILE` 解析为 repo 内绝对路径，避免远端相对路径触发 `npz_path.relative_to(REPO)` 报错。
- `workspace/core4d/scripts/train/train_E081.sh` 的 keyframe 抽帧加 `ffmpeg -nostdin` 和 `KEYFRAME_TIMEOUT_SECONDS`，避免 tmux 后台 ffmpeg 因 stdin/job-control stopped。

最终 E081 full rerun 覆盖 `13/13`。

2026-05-21 修正：用户指出 E081 full rerun 缺 paper metrics 会让对比无意义，因此已把 `eval_E081.py` 接入 E019 `paper_metrics.add_paper_metrics`。E081 的 object 不是 trailing freejoint，而是 3 个 slide + 3 个 hinge joint；本轮同时把 `paper_metrics` 的 object pose / MuJoCo penetration object 检测改为 MuJoCo FK object body 口径，避免“列补齐但 object pose 读错”。重跑后 `results/E026_E081_full/comparison.csv` 中 `paper_metrics_version=2026-05-21-P3` 为 `13/13`，物体位置/姿态、5cm contact、robot-object deep penetration、MJ penetration、smoothness、SPIDER body/joint metrics 均已写入。E081 的 strict 仍是 legacy leg-object strict proxy，不等同于 E018b/E022-E025 的 strict gate。

下面这张表保留 legacy E081 gate 视角，paper metrics 聚合见后续 P0/P1 表。

| Case | Obj mean m ↓ | Contact proxy % ↑ | Pelvis min m ↑ | Case-window | Strict proxy |
|---|---:|---:|---:|---:|---:|
| box021_p1 | 0.376 | 30.26 | 0.129 | False | False |
| box021_p2 | 1.446 | 28.00 | 0.195 | False | False |
| box023_p1 | 0.177 | 56.79 | 0.702 | True | True |
| box023_p2 | 0.156 | 74.07 | 0.698 | True | True |
| box025_p1 | 0.274 | 74.07 | 0.717 | True | False |
| box025_p2 | 0.199 | 100.00 | 0.764 | True | True |
| bucket001_p1 | 0.265 | 0.00 | 0.182 | False | False |
| bucket001_p2 | 0.827 | 30.86 | 0.328 | False | False |
| bucket005_s2_p1 | 0.207 | 76.54 | 0.748 | True | False |
| bucket005_s2_p2 | 0.232 | 100.00 | 0.720 | True | False |
| bucket007_p1 | 0.210 | 56.79 | 0.695 | False | False |
| bucket007_p2 | 0.153 | 93.83 | 0.751 | True | True |
| desk021_p1 | 0.125 | 83.95 | 0.732 | True | False |

Aggregate：case-window success `8/13`，strict proxy `4/13`。

## P0 9case 结果

P0 cases：`box023_p1`, `box023_p2`, `box025_p1`, `box025_p2`, `bucket001_p2`, `bucket005_s2_p1`, `bucket005_s2_p2`, `bucket007_p1`, `bucket007_p2`。

| Method | N | Missing | Obj Pos cm ↓ | Obj Ori deg ↓ | Contact 5cm/proxy ↑ | Kin Contact 28cm ↑ | Deep Pen % ↓ | MJ Pen % ↓ | Smoothness ↓ | Pelvis min m ↑ | Falls ↓ | Strict ↑ |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| omniretarget_kinematic | 9/9 | - | 0.00 | 0.00 | - | 57.29 | - | 0.00 | 37196 | 0.71 | 0 | 0/9 |
| spider_E081 | 2/9 | 7 cases | 15.32 | - | 77.84 | - | - | - | - | 0.72 | 0 | 1/2 |
| spider_E081_full_rerun | 9/9 | - | 21.27 | 19.04 | 42.17 | - | 16.25 | 0.00 | 15147 | 0.64 | 1 | 4/9 |
| spider_E018b | 9/9 | - | 5.14 | 4.99 | 63.61 | - | 35.83 | - | 39591 | 0.68 | 1 | 1/9 |
| spider_best_E018b_E022_E025 | 9/9 | - | 4.97 | 4.24 | 65.81 | - | 30.52 | 0.00 | 39426 | 0.71 | 0 | 1/9 |

P0 结论：

- OmniRetarget kinematic 覆盖完整，但 28cm contact preservation 只有 `57.29%`，且 28cm 阈值需 caveat。
- E081 full rerun 补齐 paper metrics 后，P0 object mean error `21.27cm`、5cm contact `42.17%`、deep penetration `16.25%`、strict proxy `4/9`；object tracking 和 contact 都明显差于 E018b / best dynamic，strict 只是 E081 proxy。
- E018b/E022-E025 best selection 在 object tracking、fall、deep penetration 上更稳：P0 object `4.97cm`、fall `0`、deep pen `30.52%`。但 strict full-retargeting 仍只有 `1/9`。

## P1 13case 结果

P1 cases：13case union。

| Method | N | Missing | Obj Pos cm ↓ | Obj Ori deg ↓ | Contact 5cm/proxy ↑ | Kin Contact 28cm ↑ | Deep Pen % ↓ | MJ Pen % ↓ | Smoothness ↓ | Pelvis min m ↑ | Falls ↓ | Strict ↑ |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| omniretarget_kinematic | 12/13 | desk021_p1 | 0.00 | 0.00 | - | 53.59 | - | 0.00 | 36048 | 0.71 | 0 | 0/12 |
| spider_E081 | 2/13 | 11 cases | 15.32 | - | 77.84 | - | - | - | - | 0.72 | 0 | 1/2 |
| spider_E081_full_rerun | 13/13 | - | 27.10 | 15.53 | 36.23 | - | 12.39 | 0.00 | 14029 | 0.55 | 4 | 4/13 |
| spider_E018b | 13/13 | - | 5.45 | 5.22 | 54.35 | - | 30.04 | - | 37781 | 0.58 | 4 | 1/13 |
| spider_best_E018b_E022_E025 | 13/13 | - | 5.33 | 4.70 | 55.87 | - | 26.37 | 0.00 | 37544 | 0.61 | 3 | 1/13 |

P1 结论：

- OmniRetarget / Holosoma 仍缺 `desk021_p1`，原因是 SOCP infeasible。
- E081 full rerun 覆盖 `13/13` 且已补齐 paper metrics；full union 上 object error `27.10cm`、5cm contact `36.23%`、fall `4`，只适合作为 scene-actuator baseline，不是当前 best dynamic 方法。
- E018b/E022-E025 best selection 相比 E081 full rerun 保持更好的 object tracking 和 penetration 指标，但 strict 成功仍很低，说明 post-E020 robot-side problem 尚未解决。

## Best dynamic selection

`spider_best_E018b_E022_E025` 只在 E018b 与 E022-E025 之间按 deterministic score 选每 case 最好结果，不把 E081 baseline 混入候选。选择逻辑优先 strict success、object/transport、no fall、contact、low penetration。

代表选择：

- `box023_p1` 选 E022 `raw3_eval_axis`，因为 mask semantics 修复后不回退 object，但 contact 仍低。
- `bucket001_p1/p2` 选 E024 stability variants，p2 stability 有改善但 penetration 仍失败，p1 仍 fall。
- `bucket005_s2_p2`、`bucket007_p1` 选 E025 stronger penalty variants，deep penetration 有方向性下降但没过 strict gate。
- 多数其他 case 保持 E018b，说明 E022-E025 是局部修补而不是全面胜出。

## 视觉与指标一致性

`workspace/core4d_collab_retarget/results/E026_full_eval/visual_metric_audit.md` 汇总了已有在线视频和抽帧。核对结果：

- fall cases 的 pelvis/upright failure 与视频标签一致。
- low-contact cases 在视频中表现为手与物体明显脱离。
- bucket high-contact cases 的失败不是 object tracking，而是 penetration / artifact shortcut。

因此，本轮量化指标和已有可视化判断一致，没有发现“指标说过、视频明显不过”的反例。

## 最终判断

1. **OmniRetarget kinematic**：覆盖 `12/13`，缺 `desk021_p1`；28cm contact preservation 是 Holosoma v1 代码继承口径，不是强物理接触指标，需要与阈值 sweep 一起解释。
2. **E081 baseline**：旧结果只有 `2/13`，本轮 E026 full rerun 补齐 `13/13` 并接入 paper metrics。补齐后可直接比较 paper-facing 列，但 E081 strict 仍是 legacy proxy；它的 object error / 5cm contact / fall 明显较差，不能作为当前 best dynamic 方法。
3. **Spider dynamics 当前最好账面**：E018b/E022-E025 best selection 在 P1 达到 object `5.33cm`、deep pen `26.37%`、fall `3`，优于 E081 full rerun 的 object/fall 账面，但 strict success 仍只有 `1/13`。
4. **下一步不应继续普通 reward sweep**：E022-E025 与 E026 full eval 都指向同一结论：robot-side contact timing、hard no-penetration / surface feasibility、lower-body geometry/control、bucket001 p1 stability 是后续主线。

## Commands

```bash
# Holosoma / OmniRetarget kinematic
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl .venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_holosoma_kinematic.py --all

# E026 full eval summary
.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E026_full_eval.py --all

# E081 full rerun setup
bash workspace/core4d_collab_retarget/scripts/run_E026_e081_preprocess.sh
bash workspace/core4d_collab_retarget/scripts/run_E026_e081_remote.sh
RUN_TIMEOUT_SECONDS=7200 bash workspace/core4d_collab_retarget/scripts/train/train_E026_e081_full.sh local 0

# Local resume after shell-script edit interrupted old local wrapper
RUN_TIMEOUT_SECONDS=7200 bash workspace/core4d_collab_retarget/scripts/train/train_E026_e081_full.sh single 0 E026_E081_bucket005_s2_p1_legobj
RUN_TIMEOUT_SECONDS=7200 bash workspace/core4d_collab_retarget/scripts/train/train_E026_e081_full.sh single 0 E026_E081_bucket007_p1_legobj

# Pull remote + rebuild 13case E081 comparison
bash workspace/core4d_collab_retarget/scripts/pull_E026_e081_remote_results.sh
VARIANTS_FILE=workspace/core4d_collab_retarget/scripts/E026/e081_full_variants.tsv RESULTS=workspace/core4d_collab_retarget/results/E026_E081_full .venv/bin/python workspace/core4d/scripts/eval/eval_E081.py
```
