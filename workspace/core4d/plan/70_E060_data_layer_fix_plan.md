# Plan 70: E060 — Data Layer Fix → Baseline Validation → Reward Ablation (Conditional)

## Context

承接 E058/E059 双双 baseline 摔倒 + pre-E060 audit (`workspace/core4d/log/70_pre_E060_audit_E041c_data_and_reward.md`)。

E059 原 log 把根因归到 `stability_penalty_scale: 0.0`，audit 证明这判断窄了。真根因是**多层叠加**：
1. 数据层：`spider/assets/robots/unitree_g1/robot.xml` `hand_collision` 是 5cm sphere（HDMI 3-box 没推广）→ 物理上手心手背等价 → E041 ori reward 是视觉补丁
2. 数据层：box023 collision margin 1.05x，E053 box025 sweep 明确结论 0.85-0.90 最优，box023 一直没 sweep
3. Reward：`contact_hdmi_palm_normal [0,∓1,0]` 是 box025 motion fingerprint，audit §2.2 数值证明对 box023 L 是噪声 (mean dot 0.04)
4. 基础设施：scene XML 之前不在 git 里，已修（commit b190785, fa2e181）

## Goal-Driven Hypothesis Tree

```
Phase 1 (data layer) → Phase 2 (baseline test E060.0)
    ├─ pelvis_min_intent ≥ 0.40m + main_face match → ✅ DONE, 推广其他 case (E061)
    ├─ pelvis_min_intent ≥ 0.40m but main_face miss → reward 部分有问题, 进 Phase 3
    └─ pelvis_min_intent < 0.40m (still falls) → reward 是元凶, 进 Phase 3 全套
```

## Claims (E060.0)

| ID | 描述 | 量化 | 通过含义 |
|----|------|------|---------|
| C1 | 流水线完整 | snapshot + npz + mp4 + 10 jpg + csv 全有 | infra OK |
| C2 | box023 baseline pelvis_min_intent ≥ 0.40m | vs E059 0.14m | ⭐ 数据层修复有效 |
| C3 | bucket005_s2 baseline pelvis_min_intent ≥ 0.40m | vs E058 0.11m | ⭐ 数据层修复有效 |
| C4 | box023 main_face L=-xy AND R=-yz | E055/E056 expected | reward 也不是元凶 |
| C5 | bucket005_s2 main_face L=-yz AND R=+yz | E056 expected | reward 也不是元凶 |
| C6 | 视频 5/5 keyframe 站立 + 双手持物 | 目检 | 真搬运 |

**通过 = C1+C2+C3+C4+C5 (≥5/6)**：E041c reward 没问题，数据层就是元凶。
**部分通过 = C2+C3 但 C4/C5 miss**：站住但握姿错，进 Phase 3 排查 reward。
**不通过**：进 Phase 3 全套 ablation。

## Phase 1: Data Layer Fix ✅ 已完成 (commit fa2e181)

### 1A. 3-box hand
- `spider/assets/robots/unitree_g1/robot.xml`：`hand_collision` sphere → 3 box per side (lh/lh2/lh3 + rh/rh2/rh3)
- 9 个 active case 的 `scene.xml` + `scene_act.xml` 通过 `patch_hand_3box.py` 同步替换（scene 是自包含的）
- Pair 数 4 → 12 per case
- Source: `example_datasets/processed/hdmi/.../move_suitcase/scene/mjlab scene.xml`

### 1B. box023 collision margin
- `box023_person1/scene.xml` + `scene_act.xml`：margin 1.05 → 0.90 (用 `set_collision_margin.py`)
- 体积 -37%，half-size 17.86×18.30×20.60 → 15.31×15.68×17.66 cm
- bucket005_s2 不动（E053 bucket010 表明 1.05 反而最优）

### 1C. Snapshot
- `workspace/core4d/results/E060/scene_snapshot/{box023,bucket005_s2}_person1/`
- manifest 含 git HEAD `91ec2c2` + sha256

### 1D. Smoke test
- 9 case × 2 file × mujoco load = 18/18 通过
- 6 hand geoms + 34 contact pairs per case 一致

## Phase 2: E060.0 Baseline Validation (待执行)

### 训练
```bash
bash workspace/core4d/scripts/run_E060_0.sh parallel 0 1
```
- 内部第 1 步：`snapshot_scenes.sh E060` 锁状态
- GPU0=box023, GPU1=bucket005_s2，并行
- 都不带 warmstart（pure baseline）
- 完全 E041c 配置 (`+override=core4d_e041c`)
- 预期 wall time: ~30-40 min（参考 E059 baseline 32 min；3-box hand 12 pairs vs 4 可能略增）

### 评估
- `eval_E060_0.py`：每 case 算 contact% / stable% / pelvis_min / main_face，输出 `eval_summary.csv` + 2 个 face_dist PNG
- `extract_E060_0_keyframes.sh`：每 case 5 帧，共 10 jpg
- 视频 + face dist 联合判定

### 改动文件
- `workspace/core4d/scripts/train/train_E060_0.sh` - 训练
- `workspace/core4d/scripts/eval/eval_E060_0.py` - 评估
- `workspace/core4d/scripts/eval/extract_E060_0_keyframes.sh` - 关键帧
- `workspace/core4d/scripts/run_E060_0.sh` - 一键
- 输出: `workspace/core4d/results/E060/E060_0_*.{npz,mp4,jpg}` + eval_summary.csv + 2 face_dist*.png

## Phase 3: Reward Ablation (Conditional)

只在 Phase 2 失败时进入。

### E060.1 — kill ori reward
```
+override=core4d_e041c task=<case> contact_hdmi_ori_weight=0.0 +use_torch_compile=false
```
- 在两个 case 上各跑（GPU0/1 并行）
- 通过 = pelvis_min_intent 改善 ≥0.10m 或 contact_both 改善 ≥10pp

### E060.2 — case-correct palm normal (audit §2.2)
- 新 yaml `examples/config/override/core4d_e060_2.yaml`，clone E041c，override:
  ```yaml
  contact_hdmi_palm_normal_left:  [1.0, 0.0, 0.0]
  contact_hdmi_palm_normal_right: [1.0, 0.0, 0.0]
  ```
- 仅 box023 跑（bucket005_s2 L 已经匹配 [0,-1,0]）

### E060.3 — last resort
```
contact_hdmi_ori_weight=0.0 stability_penalty_scale=1.0 stability_penalty_threshold=0.55
```

## 风险

| 风险 | 概率 | 缓解 |
|------|------|------|
| 12 pair 让 plan time 爆炸 (>30s/iter) | 中 | 监控前 50 iter，超时降级 2-box (去 lh3/rh3) |
| box023 0.90 margin 物体过早穿透 | 低 | E053 box025 在 0.90 表现良好 |
| Phase 3 全套不达标 | 中 | 触发三次失败协议，跳出 Path B-CEM 进 RL/IL |

## 估算预算

| Phase | 时间 | GPU |
|-------|------|-----|
| Phase 1 | ✅ done | 0 |
| Phase 2 | ~40 min wall | 2 |
| Phase 3 (worst case) | ~2 h | 2 |
| 总 (best/worst) | 40 min / 3 h | 2 |
