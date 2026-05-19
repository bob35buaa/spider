# E024 实验计划：bucket001 robot stability repair

日期：2026-05-20

## Context

E020 将 `bucket001_p1` 和 `bucket001_p2` 归为 `algo_stability`。用户已指定忽略 `box021_*`，因此 E024 的 stability scope 只包含这两个 bucket case。

E018b 已证明 object-side support proxy 对 bucket001 仍可工作：object Epos 分别约 `0.033m` / `0.038m`，transport pass；失败来自 robot fall / low pelvis posture，而不是 canonical anchor 或 support proxy。

## Baseline evidence

| Case | E018b issue | Contact | Deep pen | Pelvis min | Root cause |
|---|---|---:|---:|---:|---|
| `bucket001_p1` | fall | `0.0%` | `0.0%` | `0.145m` | stability |
| `bucket001_p2` | fall + artifact | `66.3%` | `64.6%` | `0.439m` | stability |

E018b bucket overrides already use:

- `support_proxy_mode: mocap_pad`
- `support_proxy_point_local: [0, 0.1891, 0.156922]`
- `support_proxy_gravity_scale: 0.5`
- `support_proxy_connector_kp: 0.0`
- `contact_guidance: false`
- `object_action_dims: 0`
- `object_actuator_ids: []`

E024 必须保持这些 object-side fields 不变。

## Claims

| Claim | 最低证据 |
|---|---|
| C1 bucket001 fall 修复 | 两个 bucket case `E018b_robot_fall_detected == false` |
| C2 robot posture 明显改善 | `full_pelvis_z_min_m >= 0.45`，目标 `>=0.55` |
| C3 object-side 不回退 | object Epos `<0.10m`、Erot `<25deg`、transport pass |
| C4 artifact 不转移 | deep penetration `<=20%`，目标 `<=15%`；leg/object interference `<=15%` |
| C5 E024 可复现 | variants/overrides/train/eval/remote/scene snapshot/MP4/comparison 全部落盘 |

## 改动

### 1. Variants

新建 `workspace/core4d_collab_retarget/scripts/E024/variants.tsv`。

| Variant | Case | Key params | 目的 |
|---|---|---|---|
| `E024_bucket001_p1_baseline_replay` | `bucket001_p1` | E018b replay | plumbing baseline |
| `E024_bucket001_p2_baseline_replay` | `bucket001_p2` | E018b replay | plumbing baseline |
| `E024_bucket001_p1_stab_s1_t055` | `bucket001_p1` | `stability_penalty_scale=1.0`, `threshold=0.55` | isolate height penalty |
| `E024_bucket001_p2_stab_s1_t055` | `bucket001_p2` | same | isolate height penalty |
| `E024_bucket001_p1_root03_gain3_stab_t065` | `bucket001_p1` | `root_sigma=0.3`, `contact_gain=3.0`, `stab_t=0.65` | main candidate |
| `E024_bucket001_p2_root03_gain3_stab_t065` | `bucket001_p2` | same | main candidate |
| `E024_bucket001_p1_root025_gain2_stab_t065` | `bucket001_p1` | `root_sigma=0.25`, `contact_gain=2.0`, `stab_t=0.65` | stronger fallback |
| `E024_bucket001_p2_root025_gain2_stab_t065` | `bucket001_p2` | same | stronger fallback |

可选 regression guard：如果运行预算允许，增加 `E024_box025_p2_root03_gain3_stab_t065`，因为 `box025_p2` 是 E018b 唯一 strict pass。

### 2. Override generator

**文件**：`workspace/core4d_collab_retarget/scripts/E024/generate_e024_overrides.py`

设计：

- 读取 `results/E018b/manifest.tsv` 的 bucket001 rows。
- 不复制/修改 scene XML；直接复用 E018b derived tasks，除非后续需要 task-specific metadata。
- 从 E018b override 继承 support proxy fields，并显式断言不漂移。
- 只改 robot-side optimizer/reward 参数：
  - `stability_penalty_scale`
  - `stability_penalty_threshold`
  - `local_frame_root_sigma`
  - `contact_hdmi_gain`

### 3. Train / remote

**文件**：

- `workspace/core4d_collab_retarget/scripts/run_E024_preprocess.sh`
- `workspace/core4d_collab_retarget/scripts/train/train_E024.sh`
- `workspace/core4d_collab_retarget/scripts/train/train_E024_remote_tmux.sh`
- `workspace/core4d_collab_retarget/scripts/run_E024_remote.sh`
- `workspace/core4d_collab_retarget/scripts/pull_E024_remote_results.sh`

本地优先跑 `root03_gain3_stab_t065` 的一个 case，远程 2 GPU 跑剩余 variants。训练前必须 snapshot bucket001 E018b task scene XML。

### 4. Eval

**文件**：`workspace/core4d_collab_retarget/scripts/eval/eval_E024.py`

复用 E018b evaluator，追加：

- `E024_stability_pass`
- `E024_pelvis_min_pass`
- `E024_object_no_regression_pass`
- `E024_artifact_guard_pass`
- `E024_support_proxy_unchanged`
- 每 variant 的 stability/root/contact params

## Reward / config 变更

| 参数 | E018b | E024 sweep |
|---|---:|---:|
| `stability_penalty_scale` | `0.0` | `1.0` |
| `stability_penalty_threshold` | default | `0.55` / `0.65` |
| `local_frame_root_sigma` | inherited | `0.30` / `0.25` |
| `contact_hdmi_gain` | `5.0` | `3.0` / `2.0` |
| `support_proxy_mode` | `mocap_pad` | unchanged |
| `support_proxy_point_local` | `[0, 0.1891, 0.156922]` | unchanged |

注意：height-only stability 可能诱发 lunge/superman artifact，所以 `stab_s1_t055` 只作 isolate；主候选用更紧 pelvis/root tracking 和较低 contact pull。

## 成功标准

| 指标 | 目标 |
|---|---|
| Fall | `bucket001_p1/p2` 均 false |
| Pelvis min | `>=0.45m`，目标 `>=0.55m` |
| Object | Epos `<0.10m`，Erot `<25deg`，transport pass |
| Deep penetration | `<=20%`，目标 `<=15%` |
| Leg/object interference | `<=15%` |
| Contact | p1 从 `0%` 改善；p2 保持 `>=50%`，目标 `>=70%` |

## 训练命令

```bash
bash workspace/core4d_collab_retarget/scripts/run_E024_preprocess.sh
bash workspace/core4d_collab_retarget/scripts/train/train_E024.sh smoke 0

# full
bash workspace/core4d_collab_retarget/scripts/train/train_E024.sh local 0
bash workspace/core4d_collab_retarget/scripts/run_E024_remote.sh
bash workspace/core4d_collab_retarget/scripts/pull_E024_remote_results.sh

# eval
.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E024.py --all
```

## 决策规则

- 如果 `root03_gain3_stab_t065` 两 case 均 no-fall 且 artifact/object gates pass，则 E024 通过。
- 如果 p1 no-fall 但 p2 仍 deep penetration 高，p2 转入 E025 collision-penalty path，不继续扩大 stability sweep。
- 如果所有 stability variants 都摔倒，下一步不是继续调 contact gain，而是新增 upright/root-pose terminal gate 或 lower-body control regularizer。
