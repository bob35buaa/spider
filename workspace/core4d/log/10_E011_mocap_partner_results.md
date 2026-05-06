# E011: Mocap Partner 协作重定向 — 结果

## 状态: 部分成功 (C1 通过, C2 失败)

## 实验结果

| 变体 | pos_rew | base_pos_rew | obj z_max | pelvis_min | C1 | C2 |
|------|---------|-------------|-----------|-----------|----|----|
| E011 (full reward) | 3.0 | 3.0 | **0.460** | 0.116 | PASS | FAIL |
| E011b (body only) | 0.0 | 3.0 | 0.320 | **0.690** | FAIL | PASS |
| E011c (moderate) | 1.0 | 5.0 | 0.307 | 0.192 | FAIL | FAIL |

## 核心发现

1. **Mocap partner 确实提供物理支撑**：E011 的 obj_z=0.460 是所有非 kinobj 方案中最好的
2. **C1/C2 存在不可调和的 trade-off**：强 obj 奖励→物体抬起但机器人塌；弱→稳定但物体不动
3. **架构限制**：mocap body 只在 MPC 步之间更新（非 rollout 内），partner 在单次 rollout 中是静止的
4. **改进方向**：需要在 rollout 内动态更新 mocap（CUDA graph 限制），或使用不同架构

## 工程产出

- `scene_mocap_partner.xml` — 带 2 个 partner hand mocap body 的场景
- `trajectory_kinematic_partner.npz` — 重采样的 partner 轨迹（30fps）
- `spider/simulators/mjwp.py` — 添加 mocap partner loading + sync_env 更新
- `spider/config.py` — 添加 `mocap_partner_trajectory` 字段

## 可视化观察 (`visualization_mjwp.mp4`, E011 full reward 版本)

| 时间 | ref | sim |
|------|-----|-----|
| t=1.0s | 弯腰抱箱 | G1 弯腰，双手接触箱顶面，可见橙色 partner 碰撞体在箱右侧 |
| t=1.5s | 扶箱起身 | G1 站直，箱子向右偏移，partner 碰撞体可见 |
| t=2.0s | 抱箱行走 | G1 身体后仰，箱子被推向远处 |
| t=2.5s | 继续抱箱 | **G1 摔倒** — 头朝下倒在箱旁 |
| t=3.5s | 弯腰放箱 | G1 完全倒地，头部和躯干贴地 |

**视觉结论**:
1. Partner mocap 碰撞体在画面中可见（橙色胶囊），确实与箱子产生接触
2. G1 为追踪物体过度前倾 → t≈2.5s 完全摔倒
3. 箱子被推向远处而非抬起
4. Partner 手在 MPC 步间跳变，不够平滑

## 与 E010 的综合结论

| 问题 | E010 回答 | E011 回答 |
|------|---------|---------|
| G1 运动学够吗？ | **YES** (pelvis stable with kinobj) | N/A |
| 有 partner 能物理抬起吗？ | N/A | **部分 YES** (0.460) 但不稳定 |
| 最佳 Layer 1 策略？ | kinobj (PD驱动物体 + CEM body) | mocap partner 提供额外 +0.15m |

## E012 方向

**最佳实际策略**：E005 hybrid approach（已验证有效）+ E010/E011 提供的信心
- 机器人运动：SPIDER CEM 优化的物理合规轨迹
- 物体轨迹：运动学参考
- 导出格式：Holosoma RL compatible NPZ
- 新增：partner hand 轨迹一并导出（供 Holosoma 用 interaction reward）
