# E022: Anchored 轨迹 + Object Reward — 验证去除行走后的物体交互能力

## Context

E021 证明: Pelvis XY Anchor 将 body tracking 从 0.66-0.79m 降到 0.19-0.42m (↓47-72%)。
在参考位置下, 手到物体表面 <0.05m 占 69-84% — 手部可达性确认。

**核心问题**: 去除行走干扰后, CEM 能否利用更好的 body tracking 产生有效物体接触和搬运?

E020 已知: 原始轨迹上 obj_rew 效果不一致 (对 desk005 有帮助 +37%, 对 bucket010 有害)。
假设: 行走误差导致手偏离物体 → obj_rew 方向错误 → 反而恶化。Anchor 后手位置更正确, obj_rew 应更有效。

## 实验矩阵

| Run | Case | 轨迹 | pos_rew | contact_rew | base_pos | 目的 |
|-----|------|------|---------|-------------|----------|------|
| E022-a1 | box025 | anchored | 3.0 | 1.0 | 10.0 | baseline obj interaction |
| E022-a2 | box025 | anchored | 5.0 | 2.0 | 10.0 | 强 obj reward |
| E022-b1 | bucket010 | anchored | 3.0 | 1.0 | 10.0 | baseline |
| E022-b2 | bucket010 | anchored | 5.0 | 2.0 | 10.0 | 强 |
| E022-c1 | desk005 | anchored | 3.0 | 1.0 | 10.0 | baseline |
| E022-c2 | desk005 | anchored | 5.0 | 2.0 | 10.0 | 强 |
| E022-d1 | chair022 | anchored | 3.0 | 1.0 | 10.0 | baseline |
| E022-d2 | chair022 | anchored | 5.0 | 2.0 | 10.0 | 强 |

## Claims (成功标准)

| Claim | 定义 | 阈值 |
|-------|------|------|
| C1: Anchor 提升 obj interaction | anchored+obj_rew 的 lift% > original+obj_rew | 至少 2/4 case 改善 |
| C2: 物体有效位移 | obj_Δz > ref_lift * 30% | 至少 1/4 case |
| C3: 稳定性保持 | pelvis_z ≥ 0.50m | 所有 case ≥ 80% 帧 |
| C4: 视频确认手接触 | 视觉上手碰到物体 | 至少 2/4 case |

## 执行计划

```
Step 1: 跑 8 runs (4 case × 2 reward 强度), 使用 anchored trajectory
Step 2: 提取 metrics (pelvis_err, obj_z, lift%, stable%)
Step 3: 对比 original vs anchored (有obj_rew)
Step 4: 提取关键帧, 视觉验证手接触
Step 5: 写 log + 更新 tracker
```

## Results 目录

```
workspace/core4d/results/E022_anchored_objrew/
├── box025/
│   ├── a1_baseline.{npz,mp4}
│   └── a2_strong.{npz,mp4}
├── bucket010/
├── chair022/
├── desk005/
├── keyframes/
└── metrics_summary.csv
```
