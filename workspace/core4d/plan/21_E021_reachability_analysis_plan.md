# E021: 参考动作可达性分析 + 行走适配方案

## Context

E020 多 Case 诊断的真正根因揭示:

**问题不是 IK 手部可达性，而是行走跟踪失败。**

| Case | ref pelvis 位移 | ref pelvis 旋转 | sim 实际跟踪距离 | pelvis_err |
|------|----------------|----------------|-----------------|-----------|
| box025 | 2.07m (y方向) | 2.8° | 0.83m (50%) | 0.660m |
| bucket010 | 2.66m + 旋转 | 69.0° | ~1.1m (41%) | 0.692m |
| chair022 | 4.00m (复杂路径) | 106.9° | ~1.0m (25%) | 0.787m |
| desk005 | 2.29m (y方向) | 9.1° | ~0.56m (24%) | 0.666m |
| bucket005 (对照) | ~0.3m (原地) | <5° | 原地 | **0.157m** |

**结论**: SPIDER CEM (horizon=1.6s, ~48 sim steps) 无法规划长距离步行序列。机器人在短视野内选择"站稳不动"的安全策略，导致 pelvis 逐渐被参考拉开。

**这与 E001-E009 的"几何不可解"诊断互补**: 即使物体够得着（如 desk005），机器人也跟不上参考的行走路径。

## 两个方向

### 方向 A: 分析 — 量化可达性 gap
- 逐帧计算: 参考中 G1 手到物体的距离 vs 当前 sim pelvis 位置下手到物体的距离
- 确认: 如果 sim 能走到正确位置，手是否能碰到物体?
- 分解 pelvis_err 为: 行走位移误差 + 局部姿态误差

### 方向 B: 适配 — 减少行走需求的预处理方案
- **方案 B1 (行走裁剪)**: 截取参考轨迹中 pelvis 位移最小的片段 (站定搬运阶段)
- **方案 B2 (重置 pelvis)**: 每个 MPC step 将 ref pelvis 对齐到 sim pelvis (相对跟踪)
- **方案 B3 (分段优化)**: 先解行走，再在到达位置后解操作

## Claims (成功标准)

| Claim | 定义 | 阈值 |
|-------|------|------|
| C1: 行走是主因 | pelvis_err 的行走分量 > 局部姿态分量 | walk_err > 3× pose_err |
| C2: 站定片段可达 | 在站定片段内, pelvis_err ≤ 0.20m | 至少 2/4 case |
| C3: 裁剪后 body tracking 改善 | 裁剪版 pelvis_err 优于全序列 | ↓50% |
| C4: 手部可达验证 | 在正确 pelvis 位置下, 手到物体距离 < 0.30m | 至少 2/4 case |

## 执行计划

### Phase A: 运动学分析脚本 (无需跑 MJWP)

```
Step 1: 逐帧分解 pelvis_err
  - 行走分量: pelvis xy 位移误差
  - 局部姿态分量: pelvis z + rotation + joint 误差
  → 验证 C1

Step 2: 识别"站定片段"
  - 找 pelvis xy 速度 < 0.1 m/s 的连续帧区间
  - 输出每个 case 的站定区间 [start, end]
  
Step 3: 手-物体距离分析
  - 在参考轨迹上做 FK, 计算手 site 到 object 质心的距离
  - 区分: 参考位置下的距离 vs sim 实际位置下的距离
  → 验证 C4
```

### Phase B: 行走裁剪 retargeting

```
Step 4: 截取站定片段, 生成 trajectory_kinematic_static.npz
  → 只保留低位移帧
  
Step 5: 用裁剪后的轨迹跑 SPIDER MJWP
  → 对比 pelvis_err 全序列 vs 裁剪版
  → 验证 C2, C3
```

### Phase C: pelvis 相对跟踪 (如果 B1 有效)

```
Step 6: 实现 ref pelvis 对齐
  - 每帧将 ref qpos 的 pelvis xy 平移到 sim pelvis xy
  - 保持 pelvis z, rotation, joint angles 不变
  → 测试: 这是否让 CEM "只优化局部姿态" 而非行走

Step 7: 验证效果
  → pelvis_err, obj interaction, 视频验证
```

## 代码改动清单

| 步骤 | 文件 | 描述 |
|------|------|------|
| A1 | `scripts/eval/analyze_reachability.py` | 新脚本: 逐帧分解 pelvis_err + FK 手距离 |
| B1 | `scripts/convert/trim_static_segment.py` | 新脚本: 识别站定片段并裁剪 |
| B2 | 4 × `trajectory_kinematic_static.npz` | 裁剪后的参考 |
| B3 | 4 × `core4d_{case}_static.yaml` | 裁剪版配置 (修改 ref frames) |
| C1 | `examples/run_mjwp.py` (或新配置) | pelvis xy 对齐逻辑 |

## Results 目录

```
workspace/core4d/results/E021_reachability/
├── analysis/
│   ├── pelvis_err_decomposition.csv
│   ├── static_segments.json
│   └── hand_object_distance.csv
├── box025_static/
│   ├── bodyonly.npz
│   └── bodyonly.mp4
├── bucket010_static/
├── chair022_static/
├── desk005_static/
└── metrics_static_vs_full.csv
```

## 风险

1. **站定片段可能太短** (< 30 帧 = 1s) — 如果 case 全程都在走路则 B1 无法应用
2. **相对跟踪可能打破 contact** — pelvis 对齐后物体位置可能不一致
3. **方向 B3 (分段优化) 需要更大改动** — 可能超出当前实验范围
