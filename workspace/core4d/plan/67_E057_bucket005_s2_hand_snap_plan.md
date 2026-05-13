# E057 实验计划：bucket005_s2_person1 Hand-Snap Warmstart (Path B 第 2 case)

> 前置阅读：`workspace/core4d/log/65_E055_box023_hand_snap_results.md`、`workspace/core4d/log/66_E056_multi_case_diagnosis_results.md`

## Context

### 前置结论

**E055** 在 box023 上首次跑通 hand-snap 流水线（DLS IK + frame warm-start + 5 cm offset），双手 palm 投影到 box023 表面，IK 全帧关节限位通过、residual ≤ 1 cm。E056 多 case 诊断后**纠正**：box023 的握姿是 `垂直 (-xy / -yz)`（L 托底 + R 扣远），是 valid 但非教科书"对侧"。

**E056** 6 case face 诊断结论（intent 内 L/R main face + grasp_type）：

| Case | grasp_type | intent | L_main | R_main | 备注 |
|------|-----------|--------|--------|--------|------|
| **bucket005_s2** ⭐ | **对侧 (-yz / +yz)** | 88f | -yz (+0.6cm) | +yz (+3.2cm) | 唯一真"对侧"，最长 intent |
| box023 | 垂直 (-xy / -yz) | 58f | -xy (+2.6cm) | -yz (+1.3cm) | E055 已验证 |
| bucket007 | 垂直 (+xy / +yz) | 42f | — | — | 待验证 |
| desk021 | 垂直 (+xz / -xy) | 69f | — | — | 待验证 |
| bucket001 | 单手 (L on +xz) | 58f | — | — | 待验证 |
| box021 | 同面 (+xy / +xy) | 68f | — | — | ❌ 移出 Path B |

### 为什么是 bucket005_s2

E056 排序后**唯一的"对侧"握姿**：双手在同一平面（yz）的两侧，物理上是 force-closure 候选，是搬运动作的教科书姿势。intent 88 帧也是 6 case 中最长。视频核实 t=2.5s 双手对称在 bucket 左/右两侧。

### 关键 insight (来自 E055/E056)

1. snap 流水线本身在 box023 上已验证可用 — 只需把 case 路径换成 bucket005_s2，逻辑不变
2. 物体 collision box half-sizes 必须从 `model.geom_size` 读（E056 修正过 hardcode bug） — `hand_snap_ik.py` 已用 `find_object_mesh()` 读 mesh 路径，IK target 投影到 mesh 表面，无需改动
3. surface_offset=5 cm 在 box023 上把 hand 球穿模降到 8 cm，bucket005_s2 (15.8×16.2×23.1 cm half) 与 box023 (17.9×18.3×20.6 cm half) 大小接近，沿用 5 cm offset
4. **新增 Claim (C6)**: snap 后 main face 必须与 E056 诊断一致（L 解到 -yz, R 解到 +yz），否则说明 IK 选错最近点
   - 没这条 claim 的话，snap 可能把 L 投到 +yz 或 +xy（物理上无效），而 csv 里 final_d ≈ 0 仍报"成功"

## Claims

| ID | Claim | 最低证据 |
|----|-------|---------|
| C1 | 整套 snap 流水线在 bucket005_s2 上跑通，产出 npz + csv + mp4 | `warmstart_qpos.npz` + `snap_diagnostics.csv` + `snap_visualization.mp4` 三件齐全 |
| C2 | 双手 palm-to-surface final ≤ 5 cm（与 E055 同标准，5 cm = G1 hand 球半径） | csv 中所有 intent 帧 final_d ≤ 0.05 m |
| C3 | 关节限位 100% 满足 | csv 中 `joint_in_limits=True` 全帧 |
| C4 | 视觉合格：5 keyframe 目检中 ≥ 4 帧双手贴在 bucket 两侧、无穿模 | front + side 两视角，对 5 帧关键帧目检 ≥ 4/5 通过 |
| C5 | 一键脚本可复现 | `bash workspace/core4d/scripts/run_E057_snap.sh` 端到端跑通 |
| **C6** | **post-snap 接触面与 E056 诊断一致**：L 解到 -yz，R 解到 +yz | 重跑 face_distance 诊断，intent 内 L/R main_face 与 E056 一致，median \|dist\| ≤ 7 cm 且 ≥ 60% 帧 close |

> **本计划只验证 snap 在 bucket005_s2 几何可行 + 接触面正确**。是否能改善 CEM 留给 E058 (Path B-CEM)。

## 改动

### 1. 新建 `workspace/core4d/scripts/E057/snap_bucket005_s2.py`

**克隆** `workspace/core4d/scripts/E055/snap_box023.py`，改：
- `CASE = "bucket005_s2_person1"`
- 路径常量 `OUT_DIR = "workspace/core4d/results/E057/bucket005_s2_person1"`
- 其余逻辑（intent v3 detector + snap_hands_to_object 调用）完全不变

### 2. 新建 `workspace/core4d/scripts/E057/visualize_snap.py`

**克隆** `workspace/core4d/scripts/E055/visualize_snap.py`，改 `CASE` 与 `RESULTS` 路径。

### 3. 新建 `workspace/core4d/scripts/E057/extract_snap_keyframes.sh`

**克隆** `workspace/core4d/scripts/E055/extract_snap_keyframes.sh`，改视频路径 + 时间戳。bucket005_s2 intent=20-107 (88f @ 30fps) → 关键帧 t = 0.5/0.85/1.95/3.30/3.80s（pre / start / mid / end / post）。

### 4. 新建 `workspace/core4d/scripts/E057/verify_snap_face.py` — C6 验证

**新逻辑**：在 snap 后的 qpos 上重跑 `multi_case_face_diagnosis.py` 的 `compute_face_distances + find_main_face`，对比 ref vs snap 的 main face：
- 输入：`workspace/core4d/results/E057/bucket005_s2_person1/warmstart_qpos.npz`
- 输出：`face_verification.csv` (ref_L_face / snap_L_face / ref_R_face / snap_R_face / matches_E056)
- 期望：snap_L_face = -yz, snap_R_face = +yz, matches_E056 = True

### 5. 新建 `workspace/core4d/scripts/run_E057_snap.sh`

一键脚本：snap → 可视化 → 关键帧 → face 验证。

### 6. 文件清单

| # | 文件 | 改动 |
|---|------|------|
| 1 | `workspace/core4d/scripts/E057/snap_bucket005_s2.py` | 新建（克隆 E055，改路径） |
| 2 | `workspace/core4d/scripts/E057/visualize_snap.py` | 新建（克隆 E055，改路径） |
| 3 | `workspace/core4d/scripts/E057/extract_snap_keyframes.sh` | 新建（克隆 E055，改时间戳） |
| 4 | `workspace/core4d/scripts/E057/verify_snap_face.py` | 新建（C6 验证 ~80 行） |
| 5 | `workspace/core4d/scripts/run_E057_snap.sh` | 新建（一键流水线） |
| 6 | `workspace/core4d/EXPERIMENT_TRACKER.md` | E057 行 + scripts 引用 |
| 7 | `workspace/core4d/log/67_E057_bucket005_s2_hand_snap_results.md` | 实验完成后写 |

**不动 `spider/preprocess/hand_snap_ik.py`** — E055 已验证 box023 + bucket005_s2 大小接近，5 cm offset 通用。

## Reward 权重

不适用 — 本实验只生成 warmstart 轨迹，不跑 CEM。

## 训练命令

```bash
bash workspace/core4d/scripts/run_E057_snap.sh
```

预期 < 3 分钟（snap 88 帧 ×2 手 + 离屏渲染 153 帧 ×2 cam）。

## 成功标准

| 指标 | 参考 (E055 box023) | 本次目标 (E057 bucket005_s2) |
|------|-------------------|------------------------------|
| palm-to-surface final 均值 | 0.X cm | ≤ 5 cm（与 E055 同） |
| joint_in_limits | 100% | 100% |
| 视觉合格 keyframe | 5/5 | ≥ 4/5 |
| **post-snap main face** | (未做这条) | **L=-yz ✅, R=+yz ✅** |

## 风险与回退

| 风险 | 概率 | 应对 |
|------|------|------|
| IK 把 L 解到 +yz（错面） | 中 | C6 暴露后改 `snap_hands_to_object` 加 `face_constraint` 参数（限制 closest point 只在指定面上找） |
| bucket005_s2 mesh 加载失败 | 低 | scene.xml 已确认存在，`find_object_mesh` 已通过 `<compiler meshdir>` 解析 |
| intent 88 帧 snap 慢 | 低 | 已用 frame warm-start，预期 < 60 s |
| 关节限位违反 | 低 | bucket005_s2 比 box023 略小，G1 单臂应可达 |

## 后续衔接

如果 C1-C6 全过：
- **E058**: 把 `warmstart_qpos.npz` 传入 MJWP CEM 作为初始 mean，对比有/无 warmstart 的 contact / stability / pelvis_z（仍只在 bucket005_s2 上）
- **E059**: 把 snap 推广到 bucket007/desk021/bucket001（剩余 valid case），验证流水线 case-agnostic

如果 C6 失败（face 不一致）：
- **E057-fix**: 改 `snap_hands_to_object` 加 `target_faces=("-yz", "+yz")` 强制 closest point 只在指定面查找，重跑
