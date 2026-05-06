# E002-E003: 单人Box025动力学重定向 — 结果

## 状态: 完成（发现物体跟踪根本性问题）

## 实验结果汇总

| 指标 | E002 (无引导) | E003 (首次,sites缺失) | E003b (修复sites) |
|------|-------------|---------------------|-------------------|
| obj_pos_err | 0.8254m | 0.8255m | 0.8255m |
| obj_quat_err | 0.0818 | 0.0789 | 0.0789 |
| pelvis_err | 0.100m | 0.068m | 0.068m |
| joint_err | 0.073 rad | 0.064 rad | 0.064 rad |
| 物体抬起? | 否(z=0.305) | 否 | 否 |
| 机器人站立? | 是(z≈0.81) | 是(z≈0.82) | 是(z≈0.82) |
| 运行时间 | 26s | 164s | 150s |

## 根因分析

### 成功点
- **机器人身体跟踪良好**: pelvis误差0.07-0.10m, 关节误差0.064-0.073 rad
- **机器人保持站立**: pelvis z 始终在0.80-0.82m
- **MPC优化器工作正常**: 收敛到有物理合规性的姿态

### 失败点: 物体从未被抬起
**根因**: G1 的球形手碰撞体(r=0.05m)物理上无法抓握 61×61×89cm 的箱子

1. **无抓握机制**: 两个半径5cm的球无法在61cm宽的箱子上产生足够的法向力
2. **重力主导**: 5kg 箱子落到地面后, 仅靠摩擦无法提起
3. **Contact guidance 衰减过快**: decay_ratio=0.85, 42步后增益衰减到 20×0.85^42≈0.016
4. **即使有增益, 箱子也被地面约束**: PD弹簧力 vs 重力+地面法向力

### Bug 修复记录
1. **配置缺少 `# @package _global_`** → 添加后 Hydra 正确加载覆盖
2. **trace_dt 不能被 sim_dt 整除** → 添加 `trace_dt: 0.0333333`
3. **contact site 数量不匹配** (4 vs 2) → task_info.json 只保留手部 sites
4. **contact guidance 找不到 track sites** → 添加 `track_hand_left/right` sites

## 下一步方案

### 方案 A: 强制物体跟踪（不衰减增益）
- `guidance_decay_ratio: 1.0`（增益不衰减）
- `init_pos_actuator_gain: 100`（更强的位置弹簧）
- 本质上使物体变成运动学控制
- 优点: 快速验证管线, 身体动作仍是物理合规的
- 缺点: 物体运动非物理

### 方案 B: 等式约束焊接（weld constraint）
- 在接触相位用 MuJoCo weld constraint 把箱子焊接到手上
- 使用 `scene_eq.xml` + `num_dyn=2` 实现渐进约束
- 更物理合理但实现复杂

### 方案 C: 调整接触几何
- 增大手碰撞体（capsule/box 替代 sphere）
- 增加摩擦系数到极高值
- 减小物体质量到 0.5kg

### 建议: 先做方案 A 验证管线完整性，再做方案 B 追求物理合规性

## 结果路径

| 产出 | 路径 |
|------|------|
| E002 NPZ | `example_datasets/processed/core4d/unitree_g1/humanoid_object/box025_person1/0/trajectory_mjwp.npz` |
| E002 视频 | `example_datasets/processed/core4d/unitree_g1/humanoid_object/box025_person1/0/visualization_mjwp.mp4` |
| E003 NPZ | `example_datasets/processed/core4d/unitree_g1/humanoid_object/box025_person1/0/trajectory_mjwp_act.npz` |
| E003 视频 | `example_datasets/processed/core4d/unitree_g1/humanoid_object/box025_person1/0/visualization_mjwp_act.mp4` |

## 可视化观察

**E002 (visualization_mjwp.mp4)**:
- 机器人站立稳定，整体姿态跟踪参考动作
- 双臂有正确的搬运姿态变化（伸出→弯曲）
- 箱子在第一帧后立即落到地面，之后静止不动
- 无脚滑现象，脚底与地面接触稳定
- pelvis 高度始终保持 ~0.81m，无摔倒趋势

**E003 (visualization_mjwp_act.mp4)**:
- 机器人姿态与 E002 类似，关节跟踪略好
- 箱子仍在第一帧后落地，contact guidance 未能改变物体轨迹
- 机器人手臂朝箱子方向运动，但未达到物理接触
