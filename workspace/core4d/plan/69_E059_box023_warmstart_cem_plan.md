# E059 实验计划：box023 + Snap Warmstart CEM (区分"E041c 不行" vs "warmstart 不行")

> 前置阅读: `workspace/core4d/log/68_E058_bucket005_s2_warmstart_cem_results.md`、`workspace/core4d/log/65_E055_box023_hand_snap_results.md`

## Context

E058 在 bucket005_s2 上 baseline + warm 都摔倒 (pelvis_min ≈ 0.11m), Claims 2/6 通过。**关键不确定性**: 失败来自
- (a) **E041c reward stack** 在 bucket005_s2 上不工作 (case 适配问题), 还是
- (b) **warmstart hook** 把 ref 改坏 (warmstart 设计问题)

E058 没法区分: 没有"已知工作"的 baseline。

### Path D 解决方案

在 **box023_person1** 上重做 E058 设计:
- box023 是 E055 已做过 snap 的 case (warmstart_qpos.npz 已有, 复用 0 成本)
- E041c 在 E048 baseline 评估中 box023 数据是有的 (虽然视觉也不理想), 但**至少不一致摔倒**
- box023 的 grasp_type=垂直 (-xy 托底 + -yz 扣远), intent 58 帧, 比 bucket005_s2 短 1/3

### 假设

| 情景 | 结论 |
|------|------|
| baseline ≈ 站立 + warm ≈ 站立, contact 提升 | warmstart 设计 OK, bucket005_s2 是 case 问题 → E060 修 baseline |
| baseline ≈ 站立 + warm 摔倒 | warmstart 设计破坏 baseline → E060 改 warmstart 边界条件 (减弱 ctrl_ref 替换 / 加 blend 帧) |
| baseline 摔倒 + warm 摔倒 | E041c 在 box023 也不行 → 整个 reward stack 需要重做 (E060 走 stability_penalty 路线) |

## Claims

| ID | Claim | 量化标准 |
|----|-------|---------|
| C1 | 流水线 OK (3 步: train baseline + warm, eval, keyframes) | 2 npz + 2 mp4 + 10 jpg + csv 全有 |
| C2 | baseline pelvis_min_intent ≥ 0.40m (未摔) | 区分情景 1+2 (站立) vs 3 (倒) |
| C3 | warm pelvis_min_intent ≥ 0.40m | 区分情景 1 (warm 站立) vs 2 (warm 倒) |
| C4 | warm contact (both) ≥ baseline + 5pp (warmstart 有正效) | 区分 warmstart 设计是否真有正贡献 |
| C5 | warm main_face = (-xy, -yz) 与 E055/E056 一致 | warmstart 几何在 CEM rollout 后是否保留 |
| C6 | 视频核实: warm 5 keyframe 中 ≥ 3 帧视觉优于 baseline | 与 E058 同标准 |

> **本计划 scope = 1 case, 2 runs, 4-8h GPU**。结果用来定 E060 方向, 不期待 E059 自己产生"成功搬运"。

## 改动

复用 E058 的 train/eval/keyframes 模板, 改 case + 路径。

### 1. `workspace/core4d/scripts/train/train_E059.sh`

克隆 `train_E058.sh`, 改:
- TASK = box023_person1
- WARMSTART = workspace/core4d/results/E055/box023_person1/warmstart_qpos.npz
- RESULTS = workspace/core4d/results/E059
- LOGS = logs/E059

### 2. `workspace/core4d/scripts/eval/eval_E059.py`

克隆 `eval_E058.py`, 改:
- CASE = box023_person1
- INTENT = (21, 78) (来自 E055/E056)
- EXPECTED_L = -xy (box023 是垂直握, L 在 box 底面)
- EXPECTED_R = -yz (box023 R 在远侧 yz 面)
- 输出路径 → E059

### 3. `workspace/core4d/scripts/eval/extract_E059_keyframes.sh`

克隆 `extract_E058_keyframes.sh`, 改:
- 视频路径 → E059
- 时间戳改用 box023 的: 0.40 / 0.70 / 1.65 / 2.60 / 3.00 (E055 keyframe 同款)

### 4. `workspace/core4d/scripts/run_E059.sh`

一键: train → eval → keyframes。

## 文件清单

| # | 文件 | 改动 |
|---|------|------|
| 1 | `workspace/core4d/scripts/train/train_E059.sh` | 新建 (克隆 E058) |
| 2 | `workspace/core4d/scripts/eval/eval_E059.py` | 新建 (克隆 E058 + 改 case 常量) |
| 3 | `workspace/core4d/scripts/eval/extract_E059_keyframes.sh` | 新建 |
| 4 | `workspace/core4d/scripts/run_E059.sh` | 新建 |
| 5 | `workspace/core4d/EXPERIMENT_TRACKER.md` | E059 行 + scripts |
| 6 | `workspace/core4d/log/69_E059_*.md` | 实验后写 |

**spider/ 不动**, hook 已在 E058 写好。

## 训练命令

```bash
bash workspace/core4d/scripts/run_E059.sh        # 端到端
bash workspace/core4d/scripts/train/train_E059.sh parallel 0 1   # 仅 train
```

## 风险

| 风险 | 应对 |
|------|------|
| 两 GPU 被 zombie 占用 | 11GB+ 空闲应够; 真不够时切 serial 1 GPU |
| warm 又 7x 慢 (E058 现象) | 接受, 真实信号; warm 4h 串行总 5h |
| box023 baseline 也摔 | 这恰好是情景 3, 重要发现 |

## 后续衔接 (依实际结果)

- 情景 1 (warm 改善 contact): E060 修 bucket005_s2 baseline
- 情景 2 (warm 破坏稳定): E060 检查 ctrl_ref/qpos_ref 替换的边界平滑度
- 情景 3 (双方都摔): E060 走 stability_penalty + 弱 sigma 的 reward 重设, **暂停 Path B 推广**
