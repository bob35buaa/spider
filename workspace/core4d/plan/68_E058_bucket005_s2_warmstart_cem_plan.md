# E058 实验计划：bucket005_s2 + Snap Warmstart 喂入 CEM (Path B-CEM 首跑)

> 前置阅读：`workspace/core4d/log/67_E057_bucket005_s2_hand_snap_results.md`、`workspace/core4d/log/61_E048_E052_visual_reevaluation.md`、`examples/config/override/core4d_e041c.yaml`

## Context

### 前置结论 (E057)

- E057 在 bucket005_s2 上跑通 hand-snap 流水线 (6/6 Claims)，产出 `warmstart_qpos.npz`：
  - `qpos_ref` (T=148, nq=43) — mocap 原始
  - `qpos_snap` (T=148, nq=43) — intent (20-107) 内手臂关节被 IK 投影到 bucket ±yz 两侧表面
  - `snap_mask` (T,) — 88 个 True 帧
- 视觉 keyframe 5/5 通过：intent mid 双手对称在 bucket ±yz 两侧（教科书对侧握）
- C6 face 验证：snap 后 L=-yz (77% close)、R=+yz (100% close)，与 E056 诊断完全一致

### 53 个 CEM 实验失败的核心原因 (E048-E052 视觉复查)

CEM (1024×32) 在 bucket005 上从未产生真实搬运：所有"Contact高"指标实为机器人摔倒后偶然碰桶。**根因**：mocap 原始 ref 把双手"放在桶旁边但没贴住"，CEM body tracking reward 锁死在错的几何上，contact reward 用 5cm-球-到-mesh 距离不足以覆盖差距，CEM 没有动力把手"搬"到桶上。

### Path B-CEM 假设

把 ref 中 intent 窗口的手臂 qpos **替换成 snap 后的对侧握姿**，让 body tracking reward "信"修正后的 ref，CEM 围绕这个 ref 采样，初始 mean 也来自 snap。预期：
- CEM 不再需要"自己发现"双手该贴桶 — snap 已经放好了，CEM 只要维持
- 物理 step 时双手已在 bucket 表面 5cm 内，contact reward 顺其自然给正反馈
- 接触面与 E056 诊断 (L=-yz, R=+yz) 一致，不退化到推/碰静止物体

### 关键 insight

**E057 已经把"几何不可能"变成"几何可行"。E058 验证物理可行**：把 snap 后 ref 喂 CEM，看 CEM rollout 后双手能否保持在桶两侧（而不是 CEM 把它"破坏"）。这是 Path B 的物理验证一关。

## Claims

| ID | Claim | 量化标准 | 测量方法 |
|----|-------|---------|---------|
| C1 | warmstart hook 在 run_mjwp.py 上跑通，CEM 收敛无 NaN | 输出 `trajectory_mjwp_act.npz` + 视频，CEM 32 iter 全部正常 | log 解析 + npz 存在 |
| C2 | warmstart vs baseline 的 contact% 提升 ≥ 10 个百分点 | intent 窗口内 palm 距 bucket < 5cm 帧比例 | 离线 face dist 重算 (复用 E056 算法) |
| C3 | warmstart 的 stability% ≥ 80% | 整个 episode pelvis_z ≥ 0.5m 帧比例 ≥ 80% | npz qpos[:,2] 阈值 |
| C4 | warmstart 后 main face 仍是 -yz / +yz (CEM 没破坏 snap 几何) | 重跑 verify_snap_face 在 CEM rollout qpos 上, L_main=-yz AND R_main=+yz | 复用 E057 verify_snap_face.py |
| C5 | 视频 A/B：5 keyframe 中 ≥ 3 帧 warmstart 视觉优于 baseline | 双方各抽 5 帧目检, 统计 warmstart 看起来更像"双手抱桶"的帧数 | 关键帧目检 |
| C6 | 整套流水线一键复现 | `bash workspace/core4d/scripts/train/train_E058.sh` 端到端 | 训练 + 评估 + 视频 |

> **本计划 scope 限定为 bucket005_s2 单 case**。推广到其他 case 留给 E059。

## 改动

### 1. `spider/config.py` — 加 1 个字段

**新增字段**（约第 ~313 行附近）：
```python
warmstart_qpos_path: str = ""  # path to .npz with qpos_snap + snap_mask;
                               # if set, replaces qpos_ref slices in intent window
```

### 2. `examples/run_mjwp.py` — 加 warmstart 应用 hook

在 `load_data` 调用之后、`ref_data = ...` 赋值之前，加 ~25 行：

```python
if config.warmstart_qpos_path:
    ws = np.load(config.warmstart_qpos_path)
    qpos_snap = torch.from_numpy(ws["qpos_snap"]).to(qpos_ref.device).to(qpos_ref.dtype)
    snap_mask = torch.from_numpy(ws["snap_mask"]).to(qpos_ref.device)
    assert qpos_snap.shape == qpos_ref.shape, \
        f"warmstart shape mismatch: {qpos_snap.shape} vs {qpos_ref.shape}"
    # overwrite qpos_ref where snap_mask is True
    qpos_ref = torch.where(snap_mask.unsqueeze(-1), qpos_snap, qpos_ref)
    # also overwrite ctrl_ref initial guess (CEM mean trajectory)
    ctrl_ref = qpos_ref[:, : config.nu]  # follows existing convention
    loguru.logger.info(
        "Applied warmstart from {}: replaced {} frames of qpos_ref",
        config.warmstart_qpos_path, snap_mask.sum().item(),
    )
```

> **不动 `load_data` / `Config.process_config` / 其他 simulator 文件**。warmstart 是 ref 层面的 surgical 修改。

### 3. 新建 `examples/config/override/core4d_e058_baseline.yaml`

继承 E041c reward stack，task = bucket005_s2_person1，**不**带 warmstart：

```yaml
# @package _global_
defaults:
  - core4d_e041c
task: bucket005_s2_person1
warmstart_qpos_path: ""
video_output_path: ""  # 由 train script 设置
```

### 4. 新建 `examples/config/override/core4d_e058_warm.yaml`

同上但 + warmstart：

```yaml
# @package _global_
defaults:
  - core4d_e041c
task: bucket005_s2_person1
warmstart_qpos_path: workspace/core4d/results/E057/bucket005_s2_person1/warmstart_qpos.npz
video_output_path: ""
```

### 5. 新建 `workspace/core4d/scripts/train/train_E058.sh`

单 GPU 串行跑 baseline + warm，输出到 `workspace/core4d/results/E058/`：

```bash
#!/usr/bin/env bash
# E058: baseline (no warmstart) vs warm (snap warmstart) on bucket005_s2
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

GPU="${1:-0}"
RESULTS=workspace/core4d/results/E058
LOGS=logs/E058
mkdir -p "$RESULTS" "$LOGS"

run_one() {
  local name=$1 cfg=$2
  echo "[$(date '+%H:%M:%S')] === $name ==="
  CUDA_VISIBLE_DEVICES=$GPU MUJOCO_GL=egl uv run examples/run_mjwp.py \
    +override=$cfg \
    video_output_path="$RESULTS/${name}.mp4" \
    > "$LOGS/${name}.log" 2>&1
  cp "example_datasets/processed/core4d/unitree_g1/humanoid_object/bucket005_s2_person1/0/trajectory_mjwp_act.npz" \
     "$RESULTS/${name}.npz" || true
  echo "[$(date '+%H:%M:%S')] $name done"
}

run_one E058_baseline core4d_e058_baseline
run_one E058_warm     core4d_e058_warm

echo "=== E058 done. Results: $RESULTS ==="
ls -lh "$RESULTS"
```

### 6. 新建 `workspace/core4d/scripts/eval/eval_E058.py`

复用 E056 face 算法 + E057 verify 逻辑，对两个 npz 算：
- contact% (intent 内 palm 距 bucket < 5cm 帧比例)
- stability% (pelvis_z ≥ 0.5m 帧比例)
- main face (L/R) intent 内
- 输出 `workspace/core4d/results/E058/eval_summary.csv` + 4-panel face dist png

### 7. 新建 `workspace/core4d/scripts/eval/extract_E058_keyframes.sh`

从 baseline.mp4 + warm.mp4 各抽 5 帧（覆盖 intent 起 / 中 / 末 + boundary）做 A/B 目检。

### 8. 一键脚本 `workspace/core4d/scripts/run_E058.sh`

train → eval → keyframes 串起来。

## 文件清单

| # | 文件 | 改动 |
|---|------|------|
| 1 | `spider/config.py` | 加 1 字段 `warmstart_qpos_path: str = ""` |
| 2 | `examples/run_mjwp.py` | 加 warmstart hook (~25 行) |
| 3 | `examples/config/override/core4d_e058_baseline.yaml` | 新建 (extends e041c) |
| 4 | `examples/config/override/core4d_e058_warm.yaml` | 新建 (extends e041c + warmstart) |
| 5 | `workspace/core4d/scripts/train/train_E058.sh` | 新建 |
| 6 | `workspace/core4d/scripts/eval/eval_E058.py` | 新建 (~150 行) |
| 7 | `workspace/core4d/scripts/eval/extract_E058_keyframes.sh` | 新建 |
| 8 | `workspace/core4d/scripts/run_E058.sh` | 新建 (一键) |
| 9 | `workspace/core4d/EXPERIMENT_TRACKER.md` | 加 E058 行 + 引用 |
| 10 | `workspace/core4d/log/68_E058_*.md` | 实验完成后写 |

**核心 spider/ 改动严格限制在 1 个新字段 + run_mjwp.py 的 25 行 warmstart hook**，向后兼容（不设 warmstart_qpos_path 时行为完全不变）。

## Reward 权重

复用 E041c 不动（base_pos_rew_scale=10, contact_hdmi_gain=5, ori additive 0.3）。本实验**不**调权重 — 隔离变量是 ref / 初始 mean，权重一旦动就无法判断改善来自 warmstart 还是权重。

## 训练命令

```bash
bash workspace/core4d/scripts/run_E058.sh   # 端到端 (train baseline + warm + eval)
# 或仅 train:
bash workspace/core4d/scripts/train/train_E058.sh 0   # GPU 0
```

预期单 case 跑 ~15-25 min（1024 samples × 32 iter，bucket005 中等规模）。两个 case 串行 ~30-50 min。

## 成功标准

| 指标 | baseline (E058_baseline) | warm (E058_warm) | 通过条件 |
|------|---------|------|---------|
| contact% (intent 内) | 实测 | 实测 | warm − baseline ≥ 10pp (C2) |
| stability% (整段) | 实测 | 实测 | warm ≥ 80% (C3) |
| main face L | 实测 | -yz | warm L=-yz AND R=+yz (C4) |
| main face R | 实测 | +yz | 同上 |
| 视频 keyframe 优势 | — | ≥ 3/5 帧 | warm 视觉更像抱桶 (C5) |

**核心成功条件**：C2 (contact +10pp) AND C4 (face 保持) AND C5 (视觉)。C3 是 floor。

## 风险与回退

| 风险 | 概率 | 应对 |
|------|------|------|
| warm 反而比 baseline 更差 (CEM 把 snap 抛弃) | 中 | 视频 + face 验证暴露; 下一步加 SDF contact reward 锁住 ±yz |
| ref 跳变导致 body tracking reward 爆炸 | 中 | snap_box023 已经验证 boundary blend 平滑; intent 边界 10 帧线性插值 (E055 已有) |
| ctrl_ref 跳变破坏 contact_guidance PD | 中 | 同上, 且 CEM 第一步会平滑掉 |
| nq mismatch (43 vs 42 scene_act 转换) | 低 | E041c 已有自动 quat→euler 转换 (run_mjwp.py:514), 在 warmstart hook 之后执行, 自然兼容 |
| GPU OOM (1024 samples × 88 frames) | 低 | E041c 已用同 budget 在 bucket010 上跑通 |

## 后续衔接

如果 C2-C5 全通过：
- **E059**：把 snap → CEM 流水线推广到其他 valid case (box023 / bucket007 / desk021 / bucket001)
- **E060**：加 face SDF reward (锁住 -yz / +yz)，对比"无 face reward"是否仍能保持

如果 C2 < 10pp 但 C4 通过（face 保持但 contact 没显著提升）：
- **E058b**：诊断 contact reward — 是 sigma 太小？hand 球半径不匹配？

如果 C4 失败（CEM 把双手解到错面）：
- **E058c**：加 face anchoring reward 强制双手停在 ±yz
