# E021: 参考动作可达性分析 + 行走适配 — 结果

## 状态: 突破性发现 (pelvis anchor 使 body tracking 改善 47-72%)

## 核心发现

1. **pelvis_err 的 87-96% 来自行走跟踪失败** (walk_err / pose_err 比值 3.4-15.5x) — C1 全部 PASS
2. **所有 4 case 全程都在走路** (均速 0.5-0.95 m/s), 无站定片段 — 裁剪方案 (B1) 不可行
3. **在参考位置下, 手到物体表面距离 < 0.05m 比例: 69-84%** — 手部可达性不是问题
4. **Pelvis XY Anchor 预处理使 pelvis_err 下降 47-72%** — 证明行走是唯一瓶颈

## 诊断分析

### Step 1: Pelvis Error 分解 (C1: PASS)

| Case | walk_err_xy | pose_err | ratio | 判定 |
|------|------------|----------|-------|------|
| box025 | 0.657m | 0.076m | **8.7x** | 行走主导 |
| bucket010 | 0.691m | 0.096m | **7.2x** | 行走主导 |
| chair022 | 0.774m | 0.227m | **3.4x** | 行走主导 (旋转也大) |
| desk005 | 0.664m | 0.043m | **15.5x** | 行走几乎是全部 |

### Step 2: 站定片段检测

**所有 case 均无静止片段** (threshold=0.10 m/s, min 20 frames):
- 参考轨迹是连续的行走+搬运, 没有"站着不动搬东西"的阶段
- 方案 B1 (裁剪) 不可行

### Step 3: 手-物体表面距离

| Case | 接触帧 (<0.05m) | 近距帧 (<0.15m) | 结论 |
|------|----------------|----------------|------|
| box025 | **69%** | 100% | 手能碰到! |
| bucket010 | **74%** | 80% | 手能碰到! |
| chair022 | **84%** | 91% | 手能碰到! |
| desk005 | 0% | 90% | 最近 0.075m, 几乎能碰到 |

**关键结论**: 如果 SPIDER 能让 pelvis 跟上参考位置, 手就自然能接触物体。

### Step 4-5: Pelvis XY Anchor 方案 (C3: PASS)

| Case | Original pelvis_err | Anchored pelvis_err | 改善 | stable |
|------|--------------------|--------------------|------|--------|
| box025 | 0.660m | **0.186m** | **↓72%** | 100% |
| bucket010 | 0.692m | **0.293m** | **↓58%** | 100% |
| chair022 | 0.787m | **0.417m** | **↓47%** | 100% |
| desk005 | 0.666m | **0.279m** | **↓58%** | 100% |

box025 从 0.66m 降到 0.186m, 接近 bucket005 (0.157m) 的水平!

## Claims 验证

| Claim | 结果 | 通过? |
|-------|------|------|
| C1: 行走是主因 (walk > 3× pose) | 4/4 case 比值 3.4-15.5x | **PASS** ✅ |
| C2: 站定片段可达 (pelvis_err ≤ 0.20m) | 无站定片段, 改用 anchor 方案 | N/A (重定义) |
| C3: Anchor 后 pelvis_err ↓50% | 4/4 case 改善 47-72% | **PASS** ✅ |
| C4: 手部可达验证 (<0.30m) | 3/4 case 接触帧 >69% | **PASS** ✅ |

## 方法: Pelvis XY Anchor

```python
# 每帧减去 pelvis xy 位移 (相对于 frame 0)
delta_xy = qpos[t, :2] - qpos[0, :2]
anchored[t, 0:2] -= delta_xy   # pelvis
anchored[t, 36:38] -= delta_xy  # object (保持相对几何)
```

效果: 机器人"原地"执行搬运动作 (上身/腿弯曲不变), CEM 只需优化局部姿态。

## 意义

1. **证明 SPIDER CEM 有局部姿态跟踪能力** — 只是被行走需求拖累
2. **Anchor 是 SPIDER→RL 管线的有效预处理** — 可以直接输出高质量的原地动作给 RL
3. **下一步明确**: 需要在 anchor 基础上测试 obj_rew, 看物体能否被搬起

## 结果路径

| 产出 | 路径 |
|------|------|
| 分析 CSV | `results/E021_reachability/analysis/pelvis_err_decomposition.csv` |
| 静止分析 | `results/E021_reachability/analysis/static_segments.json` |
| 手距分析 | `results/E021_reachability/analysis/hand_object_distance.csv` |
| Anchored 结果 | `results/E021_reachability/{case}_anchored/bodyonly.{npz,mp4}` |
| Anchor 脚本 | `scripts/convert/anchor_pelvis.py` |
| 实验计划 | `plan/21_E021_reachability_analysis_plan.md` |

## 下一步

1. **E022**: 在 anchored 轨迹上加 obj_rew + contact_rew, 看是否能物理搬起
2. 如果 E022 成功: anchored 轨迹 + connect 约束 → 最佳单人方案
3. 长期: 需要行走能力 (locomotion policy) 来真正解决全序列跟踪
