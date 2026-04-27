# R008 Results: Contact Guidance 使箱子可被搬运

## Summary
给 suitcase 加 6 个 PD 执行器 (xyz+rpy)，用衰减增益引导物体跟踪参考轨迹。机器人不再在拾箱时摔倒。

## Configuration
- N=1024, max_iter=32, max_sim_steps=250, data_id=1
- contact_guidance=true, guidance_decay_ratio=0.8
- init_pos_kp=10.0, init_rot_kp=0.3
- nu=35 (29 robot + 6 object), improvement_threshold=0.0

## Results

| 指标 | R007 | **R008** | HF | 达标率 |
|------|------|---------|-----|--------|
| rew_mean | 5.33 | **5.63** | 5.90 | 95% ✅ |
| tracking | 2.20 | **2.52** | 2.19 | 115% ✅ |
| obj_track | 3.13 | **3.10** | 3.71 | 84% ✅ |
| opt_steps | 8.52 | **32.0** | 3.12 | — |

## Phase breakdown

| Quarter | rew | tracking | obj_track |
|---------|-----|----------|-----------|
| Q1 (0-1.25s) | 5.25 | 2.87 | 2.38 |
| Q2 (1.25-2.5s) | 5.15 | 2.77 | 2.38 |
| Q3 (2.5-3.75s) | 5.47 | 2.29 | 3.18 |
| Q4 (3.75-5.0s) | **6.61** | **2.18** | **4.43** |

Q4 rew=6.61 是全程最高！tracking Q4=2.18 (R007 是 0.93) — 后半程稳定。

## 视频帧验证

| 时刻 | R007 | R008 |
|------|------|------|
| t=3s | 开始摔倒 | 蹲下抱箱，姿态匹配 ref ✅ |
| t=4s | 趴在箱子上 | **站立搬箱行走** ✅ |

## Claims 验证

| Claim | 状态 | 结果 |
|-------|------|------|
| 机器人不摔倒 | ✅ 通过 | pelvis_z 全程 > 0.3m (R007 最低 0.18m) |
| rew_mean > 5.5 | ✅ 通过 | 5.63 |
| tracking > 2.0 | ✅ 通过 | 2.52 |

## 关键改动

1. `_make_contact_guidance_model()`: 用 ElementTree 修改 scene XML，替换 suitcase freejoint 为 6 slide/hinge joints + position actuators
2. `get_reference()`: ctrl_ref 扩展到 35 维，object 部分用 xyz + quat→euler(rpy)
3. `load_env_params()`: 支持更新 object actuator gains (Kp/Kd)
4. `run_hdmi.py`: env_params_list 设递减增益 `gain * 0.8^iter`，最后一步 gain=0
5. `hdmi.yaml`: contact_guidance=true + 相关参数
