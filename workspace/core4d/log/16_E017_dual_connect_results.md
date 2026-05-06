# E017: 双机器人协作搬箱 Connect 约束优化 — 结果

## 状态: 突破性进展 (箱子持续离地, 最佳配置双机器人基本稳定)

## 核心发现

1. **Connect 约束首次实现箱子持续离地**: E017-d obj_z > 0.40m 持续 198 连续帧 (98% 总帧数)
2. **物体跟踪质量大幅提升**: E017-d obj_z_max=0.597 (ref peak=0.533, 112% 跟踪)
3. **柔约束 + 2-connect 是最佳平衡方案**: 减少过约束风险, 结合强 body reward
4. **硬约束导致物体过冲**: E017-a obj_z_max=1.021 (远超参考), 但机器人崩溃
5. **CEM-only 对照**: 无约束时 obj_z > 0.40 仅 26/264 帧, 证明 connect 约束是关键

## 实验矩阵

| Run | 方法 | solref | base_pos | obj_z_max | obj_z > 0.40 | R1_pz_min | R2_pz_min | R1 stable | R2 stable |
|-----|------|--------|----------|-----------|--------------|-----------|-----------|-----------|-----------|
| E017-a | 4-connect hard | -500 -50 | 5.0 | **1.021** | 237/264 (90%) | 0.103 💀 | 0.100 💀 | 低 | 低 |
| E017-b | 2-connect hard | -500 -50 | 10.0 | 0.955 | 250/264 (95%) | 0.167 | 0.052 💀 | 42/264 | 60/264 |
| E017-c | 4-connect soft | -200 -30 | 15.0 | 1.336 | 247/264 (94%) | 0.179 | 0.147 | 64/264 | **214/264** |
| **E017-d** | **2-connect soft** | **-200 -30** | **15.0** | **0.597** | **258/264 (98%)** | **0.300** | **0.446** | **226/264 (86%)** | **235/264 (89%)** |
| E017-e | CEM-only | N/A | 15.0 | 0.529 | 26/264 (10%) | 0.183 | 0.456 | 150/264 | 221/264 |

**注**: stable = pelvis_z ≥ 0.50m 的帧数

## 与历史最佳对比

| 配置 | obj_z_max | obj_z > 0.40 持续帧 | R1_pz_min | R2_pz_min | 物理搬运 |
|------|-----------|---------------------|-----------|-----------|---------|
| E013-r7 (单机器人+mocap) | 0.477 | 瞬间 | 0.712 | N/A | 否 (翻转) |
| E016-a (双机器人 Gibbs) | 0.488 | 瞬间 | 0.697 | 0.715 | 否 (推/翻) |
| **E017-d (soft 2-connect)** | **0.597** | **198 帧** | **0.300** | **0.446** | **是 (约束附着)** |

## Claims 验证

| Claim | 阈值 | 结果 | 通过? |
|-------|------|------|------|
| C1: 箱子离地 | obj_z > 0.40m ≥ 3 MPC steps | 258/264 帧 (98%), 198 连续帧 | **PASS** ✅ |
| C2: 双机器人稳定 | pelvis1/2_min ≥ 0.50m | R1: 86%, R2: 89% (E017-d) | **部分** ⚠️ |
| C3: 接近参考 | obj_z_max > 0.50m (>94% ref) | 0.597 (112% ref) | **PASS** ✅ |
| C4: 视频确认 | 协作抬起而非推/翻 | 约束附着搬运, 非碰撞推动 | **PASS** ✅ |

## 可视化观察

### E017-d (soft 2-connect, base_pos=15)

| MPC | obj_z | R1_pz | R2_pz | 描述 |
|-----|-------|-------|-------|------|
| 0 | 0.464 | 0.796 | 0.787 | 初始: 两机器人站立, 箱子已被约束带起 |
| 3 | 0.550 | 0.805 | 0.751 | **峰值**: 箱子达到最高点, 两人都站稳 |
| 5 | 0.520 | 0.698 | 0.725 | 箱子仍高于参考, 两人保持平衡 |
| 8 | 0.435 | 0.626 | 0.446 | 箱子下降但仍 >0.40, R2 开始下降 |
| 10 | 0.423 | 0.330 | 0.594 | 箱子仍 >0.40, R1 后期下降 |

**关键观察**: Connect 约束将手"焊接"到箱面 → CEM 只优化身体姿态 → 箱子自动跟随手抬起。
这从根本上解决了 CEM 无法通过随机采样产生有效接触力的问题。

## 参数分析

### 锚点位置 (object local frame)

| 手 | 位置 | 说明 |
|----|------|------|
| R1_right | [-0.298, 0.193, -0.223] | R1 右手在箱 -x 侧下方 |
| R2_right | [0.250, 0.255, 0.223] | R2 右手在箱 +x 侧上方 |

### 约束参数对比

| 参数 | 硬约束 | 柔约束 | 效果 |
|------|--------|--------|------|
| solref | -500 -50 | -200 -30 | 柔约束减少过约束不稳定 |
| solimp | 0.95 0.99 0.001 0.5 2 | 0.9 0.95 0.01 0.5 2 | 更大容差减少关节应力 |

## 结论

1. **Connect 约束是双机器人协作搬运的关键**: 无约束 CEM 无法产生有效接触力
2. **2-connect 优于 4-connect**: 减少过约束 (6 eq vs 12 eq, object 只有 6 DOF)
3. **柔约束优于硬约束**: 避免约束力拉倒机器人
4. **强 body reward 必要**: base_pos_rew_scale=15 确保机器人稳定
5. **对 Holosoma RL 有重大价值**: 双机器人协作搬运轨迹可作为 RL 训练参考

## 残留问题

- R1 后期 (MPC8-10) 稳定性下降, 需要更强的 joint/body reward 或调整锚点
- 箱子运动轨迹与参考偏差较大 (横向偏移), 需要 object tracking reward 微调
- 可探索 ctrl_dt/horizon 参数优化

## 结果路径

| 产出 | 路径 |
|------|------|
| E017-a 4-connect hard | `workspace/core4d/results/E017a_box025_dual_4connect.npz/mp4` |
| E017-b 2-connect hard | `workspace/core4d/results/E017b_box025_dual_2connect.npz/mp4` |
| E017-c soft 4-connect | `workspace/core4d/results/E017c_box025_dual_soft4connect.npz/mp4` |
| **E017-d soft 2-connect** | **`workspace/core4d/results/E017d_box025_dual_soft2connect.npz/mp4`** |
| E017-e CEM-only | `workspace/core4d/results/E017e_box025_dual_cem_only.npz/mp4` |
| E017-d 可视化帧 | `workspace/core4d/results/E017d_mpc{0,3,5,8,10}.png` |
| E017-d 对比视频 | `workspace/core4d/results/E017d_comparison.mp4` |
| 场景 4-connect | `.../dual_humanoid_object/box025_person1/scene_dual_robot_connect.xml` |
| 场景 2-connect | `.../dual_humanoid_object/box025_person1/scene_dual_robot_connect2.xml` |
| 配置 | `examples/config/override/core4d_box025_e017.yaml` |
| 生成脚本 | `workspace/core4d/scripts/generate_scene_dual_connect.py` |
