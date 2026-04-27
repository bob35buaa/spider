# R007 Results: Full Pipeline Fix (Decimation + PD Gains + Freejoint Init)

## Summary
在 R006 的基础上修复 freejoint 初始化（使用 motion data 而非 HDMI env 默认位姿），使 robot 和 suitcase 在同一世界坐标系。

## Configuration
- N=1024, max_iter=32, max_sim_steps=250, data_id=1
- physics_dt=0.002, decimation=10
- HDMI PD gains override (Kp=20~200)
- Freejoint init from motion data (not HDMI env default)
- CEM: noise=0.05, threshold=0.005, check_steps=3, temp=0.3

## Results

| 指标 | R006c | **R007** | HF (data_id=1) | 目标 |
|------|-------|---------|------|------|
| rew_mean | 4.13 | **5.33** | 5.90 | > 4.0 ✅ |
| tracking | 2.25 | **2.20** | 2.19 | > 1.5 ✅ |
| obj_track | 1.88 | **3.13** | 3.71 | > 2.5 ✅ |
| opt_steps | 4.84 | **8.52** | 3.12 | > 1.5 ✅ |
| total time | 273s | **577s** | — | — |

## Phase breakdown (10-step windows)

| 时段 | rew | tracking | obj_track | opt_steps |
|------|-----|----------|-----------|-----------|
| 0.0-1.0s | 5.66 | 2.81 | 2.85 | 4.3 |
| 1.0-2.0s | 5.64 | 2.79 | 2.85 | 4.2 |
| 2.0-2.8s | 5.58 | 2.71 | 2.88 | 5.0 |
| **2.8-3.6s** | **5.90** | **2.28** | **3.62** | **7.0** |
| **3.6-4.4s** | **4.32** | **1.19** | **3.12** | **26.4** |
| 4.4-5.0s | 4.48 | 0.73 | 3.75 | 5.9 |

## 视频帧分析

| 时刻 | ref | sim | 分析 |
|------|-----|-----|------|
| t=0s | 站立+箱子在脚边 | 站立+箱子在脚边 | 初始化正确 ✅ |
| t=2s | 弯腰靠近箱子 | 弯腰靠近箱子 | 姿态匹配 ✅ |
| t=3s | 蹲下抱箱 | 蹲下抱箱 | 开始偏差但还行 |
| **t=4s** | **站立搬箱行走** | **趴在箱子上** | **❌ 摔倒！** |

## Pelvis Z 高度追踪

| 时间 | pelvis_z | 状态 |
|------|----------|------|
| 0.0s | 0.835 | 站立 |
| 2.0s | 0.792 | 站立 |
| 2.5s | 0.702 | 弯腰 |
| 3.0s | 0.323 | **摔倒** |
| 3.6s | 0.426 | 恢复中 |
| 4.0s | 0.617 | 部分恢复 |
| 4.4s | 0.370 | **再次摔倒** |
| 5.0s | 0.180 | 完全倒地 |

## Claims 验证

| Claim | 状态 | 结果 |
|-------|------|------|
| rew_mean > 4.0 | ✅ 通过 | 5.33 |
| tracking > 1.5 | ✅ 通过 | 2.20 (全程均值) |
| obj_track > 2.5 | ✅ 通过 | 3.13 |
| 视频不摔倒 | ❌ 未通过 | t=3s 开始摔倒 |

## 根因分析

**Suitcase 是纯 freejoint 无执行器**。参考轨迹中机器人抓起箱子并搬运，但在物理仿真中：
1. 箱子是自由刚体，不受任何执行器控制
2. 机器人手指没有建模（G1_29dof_nohand），无法抓取
3. CEM 优化只控制 robot joints，无法让箱子跟随
4. 当机器人弯腰尝试"抓取"时，箱子不配合 → 失衡 → 摔倒

**解决方案**: Contact Guidance — 给 suitcase 加 6 个 PD 执行器（xyz+rpy），引导它沿参考轨迹移动。MJWP 已有此机制 (`scene_act.xml` + `humanoid_object_act.yaml`)。

## 下一步: R008
- 在 `setup_env` 中用 MjSpec 给 suitcase 加 6 actuators
- ctrl_ref 扩展为 35 维 (29 robot + 6 object)
- 增益衰减：每迭代 `gain *= decay_ratio`，最后一步 gain=0
