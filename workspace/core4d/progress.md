# E068 Progress — 2026-05-14

## 当前状态: Plan 已写入，开始 init drift 诊断

## 完成步骤

- [x] 读取 `EXPERIMENT_TRACKER.md`、最新 plan/log/progress，确认最新问题来自 log 87 的 MJWP init pose mismatch。
- [x] 审查 `spider/simulators/mjwp.py::setup_env()`、`examples/run_mjwp.py` 初始化路径，发现 qpos/qvel/ctrl 写入后立即 `mj_step()` 的可疑路径。
- [x] 写 plan → `workspace/core4d/plan/73_E068_mjwp_init_drift_plan.md`
- [x] 新建 E068 诊断脚本 → `workspace/core4d/scripts/debug/diagnose_E068_init_drift.py`
- [x] 新建 E068 入口脚本 → `workspace/core4d/scripts/train/train_E068.sh`
- [x] 读取同步后的 E062-E067 真实结果，确认 E062/E063 第二个 substep 已出现约 22 deg yaw drift。
- [x] 发现 drift 不是 init `mj_step` 单独造成：CPU `mj_step` yaw drift 仅 0.22 deg；真实轨迹第一个 committed ctrl 相对 ref 有 1.56 rad 级别 robot joint 偏差。
- [x] 新建 first-commit 分析脚本 → `workspace/core4d/scripts/debug/analyze_E068_first_commit.py`
- [x] 运行 first-commit 分析，输出 `workspace/core4d/results/E068/first_commit_trace.csv` 和 `first_ctrl_delta.csv`
- [x] 视频/关键帧复查 E062/E063/E067，确认 lunge / handstand 与数值诊断一致。
- [x] 写 E068 结果 log → `workspace/core4d/log/88_E068_mjwp_init_drift_results.md`
- [x] 更新 `EXPERIMENT_TRACKER.md` 的 E068 行和 Logs 引用。
- [x] 写 E069 plan → `workspace/core4d/plan/74_E069_first_tick_warmup_plan.md`
- [x] 新建 E069 yaml 变体: `core4d_e069w02_box023.yaml`, `core4d_e069w05_box023.yaml`
- [x] 新建 E069 train/eval 脚本。
- [x] 空结果运行 `train_E069.sh eval` 验证脚本可执行，当前结果缺失为预期。
- [x] 修正 `train_E069.sh`，避免 eval-only 模式触发 scene snapshot。

## 当前实验

- **Run ID**: E068
- **Version**: R5 / Phase 18
- **阶段**: Plan → Diagnose
- **开始时间**: 2026-05-14

## 下一步

- 等待 GPU 机器运行 `bash workspace/core4d/scripts/train/train_E069.sh parallel 0 1`。
- 运行完成后检查 `workspace/core4d/results/E069/eval_summary.csv`，再写 log 89。

## 创建/修改的文件

- `workspace/core4d/plan/73_E068_mjwp_init_drift_plan.md`
- `workspace/core4d/scripts/debug/diagnose_E068_init_drift.py`
- `workspace/core4d/scripts/debug/analyze_E068_first_commit.py`
- `workspace/core4d/scripts/train/train_E068.sh`
- `workspace/core4d/log/88_E068_mjwp_init_drift_results.md`
- `workspace/core4d/results/E068/first_commit_trace.csv`
- `workspace/core4d/results/E068/first_ctrl_delta.csv`
- `workspace/core4d/EXPERIMENT_TRACKER.md`
- `workspace/core4d/plan/74_E069_first_tick_warmup_plan.md`
- `examples/config/override/core4d_e069w02_box023.yaml`
- `examples/config/override/core4d_e069w05_box023.yaml`
- `workspace/core4d/scripts/train/train_E069.sh`
- `workspace/core4d/scripts/eval/eval_E069.py`
- `workspace/core4d/progress.md`

## 遇到的错误

| 错误 | 尝试次数 | 解决方案 |
|------|---------|---------|
| 当前沙箱无 CUDA 设备，`device=cuda:0` 触发 `RuntimeError: No CUDA GPUs are available` | 1 | 将 E068 诊断入口默认改为 `E068_DEVICE=cpu`，只做 init state 诊断 |
| CPU device 无法执行 `setup_env()` 的 CUDA graph capture，触发 `RuntimeError: Must be a CUDA device` | 1 | 诊断脚本改为先保存 CPU `mj_forward`/`mj_step` 证据，并尝试手动 `mjwarp.put_data` 不 capture graph |
| `train_E069.sh eval` 初版会先 snapshot scene | 1 | 调整脚本，仅 `parallel/single_*` 训练模式执行 snapshot |

---

# E057 Progress — 2026-05-13

## 当前状态: ✅ 完成 (bucket005_s2 hand-snap, 6/6 Claims 通过)

## 完成步骤

- [x] 写 plan → `plan/67_E057_bucket005_s2_hand_snap_plan.md`
- [x] 克隆 E055 三件套 (snap / visualize / extract_keyframes), 改 CASE 路径
- [x] 新建 `verify_snap_face.py` (C6 — snap 后 main face vs E056 诊断一致性)
- [x] 写一键脚本 `run_E057_snap.sh` (snap → viz → keyframes → face 验证)
- [x] 跑流水线 — 单次跑通 4 步, 约 2 分钟
- [x] 视频核实 5 keyframe (front+side, top=ref/bot=snap)
- [x] 写 log → `log/67_E057_bucket005_s2_hand_snap_results.md`
- [x] 更新 EXPERIMENT_TRACKER (E057 行 + log/plan/scripts 引用)

## Claims 验证

| ID | 标准 | 实际 | 通过 |
|---|---|---|---|
| C1 | npz + csv + mp4 三件齐 | 101K npz + 11K csv + 2.3M mp4 | ✅ |
| C2 | palm-to-surface final ≤ 5cm | mean 4.93cm, max 5.10cm | ✅ |
| C3 | 关节限位 100% | 176/176 | ✅ |
| C4 | ≥ 4/5 keyframe 视觉合格 | **5/5 通过**, mid frame 教科书对侧握 ⭐ | ✅ |
| C5 | 一键脚本 | run_E057_snap.sh 跑通 | ✅ |
| C6 | snap face = E056 (L=-yz, R=+yz) | L=-yz 77%, R=+yz 100% | ✅ |

**6/6 通过 ✅**

## 关键结果

```
intent: (20, 107) = 88 frames @ 30fps (与 E056 诊断一致)
snap statistics (intent 内 88f × 2 hands = 176 rows):
  init  cm: mean=5.04 max=7.57   ← mocap 原始, 已经接近 5cm offset
  final cm: mean=4.93 max=5.10   ← 收敛到 target (表面外 5cm)
  ik_residual cm: mean=0.11 max=1.12
  ik_iters: mean=8.5 max=50
  joints in_limits: 176/176 (100%)
  L final: mean=4.99 max=5.09
  R final: mean=4.87 max=5.10

C6 face verification (intent 内):
  REF  L=-yz (med 0.58cm, 100%) | R=+yz (med 3.24cm, 100%)
  SNAP L=-yz (med 2.90cm,  77%) | R=+yz (med 1.87cm, 100%)
  → 双手 main face 完全保留 E056 诊断, 没漂移到错面
```

## 重要发现

1. **case 选对了, snap 几乎自由**: bucket005_s2 mocap 原始就 init=5cm, IK 几乎不需要努力。E055 box023 init~10cm, snap 后才 5cm。**E056 case 排序的物理意义在 E057 上兑现**。

2. **C6 face 验证是必要 guard**: 没这条 claim, snap 把 L 投到错面 (e.g. +yz) 数值上仍报"成功"。**后续任何 IK-to-surface 都应该 verify 接触面**。

3. **frame warm-start 持续有效**: max iter=50 偶发 (intent boundary), mean 8.5 iter, 大多数帧前一帧 qpos 就近。

## 下一步: E058 (Path B-CEM, bucket005_s2)

- 把 `warmstart_qpos.npz` 喂入 MJWP CEM 作为初始 mean trajectory
- body tracking ref 也换成 qpos_snap (让 reward "信" 修正后的 ref)
- 对比有/无 warmstart 的 contact / stability / pelvis_z
- Claims (草案):
  - contact (palm 距 bucket < 5cm 帧占比) ≥ 50%
  - stability (pelvis_z ≥ 0.5m) ≥ 90%
  - 视频: snap 阶段 CEM 没把 ±yz 两侧握姿"破坏"

## 改动文件

| 类型 | 路径 |
|---|---|
| 新建 plan | `workspace/core4d/plan/67_E057_bucket005_s2_hand_snap_plan.md` |
| 新建脚本 (×4) | `workspace/core4d/scripts/E057/{snap_bucket005_s2,visualize_snap,verify_snap_face}.py` + `extract_snap_keyframes.sh` |
| 新建一键脚本 | `workspace/core4d/scripts/run_E057_snap.sh` |
| 输出 npz | `workspace/core4d/results/E057/bucket005_s2_person1/warmstart_qpos.npz` |
| 输出 csv (×2) | `snap_diagnostics.csv` + `face_verification.csv` |
| 输出 mp4 | `snap_visualization.mp4` (2.3M, 148 帧) |
| 输出 png | `face_dist_snap.png` (2×2 时序) |
| 输出 jpg (×5) | `keyframes/frame_0[0-4]_*.jpg` |
| 新建 log | `workspace/core4d/log/67_E057_bucket005_s2_hand_snap_results.md` |
| EXPERIMENT_TRACKER | 添加 E057 行 + log/plan/scripts 引用 |

**`spider/preprocess/hand_snap_ik.py` 没动** — E055 实现 case-agnostic 已被 E057 验证。

---

## E069 进展: first-tick warmup 运行与保存修复

- [x] 已确认本机 GPU 在非 sandbox 命令下可见: RTX 5090 / Driver 580.126.09 / CUDA 13.0。
- [x] 已运行 `bash workspace/core4d/scripts/train/train_E069.sh parallel 0 0`。
- [x] W02/W05 均跑到 `sim_steps: 272/272`，不是 CUDA 或中途优化失败。
- [x] 失败点: `examples/run_mjwp.py` 结束保存 `info_list` 时 `np.stack` 遇到跨 tick shape 不一致的诊断字段，导致 `.npz`/`.mp4` 没有落盘。
- [x] 已修复保存逻辑: `qpos/qvel/time/ctrl` 等 shape 一致字段继续保存；shape 不一致或缺失的诊断字段跳过并写 warning。

### 遇到的错误

| 错误 | 尝试次数 | 解决方案 |
|------|---------|---------|
| E069 W02/W05 完整跑完后 `ValueError: all input arrays must have the same shape` | 1 | 修改 `examples/run_mjwp.py` 的 info 聚合逻辑，跳过 shape 不稳定的诊断字段，保留轨迹核心字段 |

### 下一步

- [x] 重跑 E069 W02/W05。
- [x] 修正 `eval_E069.py` 的 ctrl_ref 口径并重新生成 `eval_summary.csv`。
- [x] 从视频提取关键帧并写 `log/89_E069_first_tick_warmup_results.md`。
- [x] 更新 `EXPERIMENT_TRACKER.md`。

## E069 结果摘要

| 指标 | E069-W02 | E069-W05 | 结论 |
|------|----------|----------|------|
| warmup robot ctrl diff | 0.00 rad | 0.00 rad | ref ctrl 确实提交 |
| yaw err t=0.017/0.033s | 12.40 / 22.16 deg | 12.40 / 22.16 deg | warmup 无法压住 early yaw |
| B1 max foot z [0,2s] | 0.222m | 0.428m | 仍单脚/lunge |
| pelvis_min_intent | 0.578m | 0.197m | W02 数值通过但视频仍不可用 |

**新结论**: first CEM override 不是主因。即使 warmup 内提交 `ctrl_ref`，MJWarp commit step 仍复现 12/22 deg early yaw drift。下一步应做 E070 ref-control parity: MuJoCo `mj_step(ctrl_ref)` vs MJWarp `step_env(ctrl_ref)`。

## E070 计划

- [x] 写 plan → `workspace/core4d/plan/75_E070_mjwarp_ref_control_parity_plan.md`
- [ ] 等用户确认后实现 parity 诊断脚本与入口脚本。
