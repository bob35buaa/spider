# E016: 双机器人交替优化 (Box025) — 结果

## 状态: 通过 (双机器人 Gibbs CEM 实现, 两机器人均保持稳定)

## 核心发现

1. **双机器人 Gibbs 交替优化首次实现成功**: nq=79, nu=58, 两个 G1 + 共享 box025
2. **两机器人均保持稳定**: R1 pelvis_z≥0.697, R2 pelvis_z≥0.715 (body-focused config)
3. **物体最高达 0.488m (ref peak=0.533, 92% 跟踪)**: 比单机器人 E013-r7(0.477) 高
4. **但物体抬升仍为碰撞推动/翻转, 非协作搬运**: 与 Phase 1-3 诊断一致
5. **object tracking reward 仍导致崩溃**: E016-b R1 在 MPC9 摔倒 (pelvis→0.155)

## 架构设计

### Scene XML (scene_dual_robot.xml)

```
worldbody
  ├── pelvis (robot1): G1 freejoint + 29 joints
  ├── r2_pelvis (robot2): G1 freejoint + 29 joints (所有 body/joint/geom/site 加 r2_ 前缀)
  └── object: box025 freejoint (共享)

nq = 7 + 29 + 7 + 29 + 7 = 79
nv = 6 + 29 + 6 + 29 + 6 = 76
nu = 29 + 29 = 58
```

### 参考轨迹合并

```
person1 qpos(124,43): pelvis(7)+joints(29)+object(7)
person2 qpos(124,43): pelvis(7)+joints(29)+object(7)
→ merged qpos(124,79): robot1(36)+robot2(36)+object(7)
→ merged ctrl(124,58): robot1_joints(29)+robot2_joints(29)
```

### Gibbs 交替优化

```python
# examples/run_mjwp.py — dual_humanoid_object Gibbs
half_nu = config.nu // 2  # 29 per robot
robot1_ids = [0..28]
robot2_ids = [29..57]

for each MPC step:
    noise_scale[robot2_ids] = 0  →  optimize robot1 only
    noise_scale[robot1_ids] = 0  →  optimize robot2 only
    restore full noise_scale
```

## 实验矩阵

| Run | pos_rew | base_pos | gibbs | R1_pelvis_min | R2_pelvis_min | obj_z_max | 评价 |
|-----|---------|----------|-------|---------------|---------------|-----------|------|
| **E016-a** | **1.0** | **5.0** | **yes** | **0.697** | **0.715** | **0.488** | **最佳 — 双机器人稳定** |
| E016-b | 3.0 | 5.0 | yes | 0.047 (CRASH) | 0.709 | 0.448 | R1 崩溃 (MPC9) |

### 与单机器人最佳对比

| 配置 | R1_pelvis_min | obj_z_max | 物理搬运 |
|------|---------------|-----------|---------|
| E013-r7 (单机器人+mocap) | 0.712 | 0.477 | 否 (翻转) |
| E012 (body-only) | ≥0.73 | N/A | 否 |
| **E016-a (双机器人)** | **0.697** | **0.488** | **否 (但最高)** |

## Claims 验证

| Claim | 阈值 | 结果 | 通过? |
|-------|------|------|------|
| C1: 场景加载 | nq=79, nu=58 | 正确 | **PASS** |
| C2: 两机器人都稳定 | pelvis1/2_min ≥ 0.50m | R1≥0.697, R2≥0.715 | **PASS** |
| C3: 物体改善 | obj_z > E013-r7 (0.477) | 0.488 > 0.477 | **PASS** (微幅改善) |
| C4: 视频确认协作 | 两机器人手在箱旁 | 手接近但未有效抬起 | **部分** |

## 可视化观察

### E016-a (body-focused, Gibbs)

| MPC | 描述 |
|-----|------|
| 0 | 两 G1 面对面站在 box025 两侧，直立。R1 在左 (-x), R2 在右 (+x) |
| 2 | 两机器人略弯腰，手伸向箱子。箱子仍在原位 (obj_z=0.313) |
| 3 | **最佳瞬间**: R2 手接触箱侧面, 箱子被推高 obj_z=0.439。两人都站稳 |
| 5 | 箱子回落 (obj_z=0.319), 两机器人站直 |
| 10 | 两机器人直立, 箱子在旁 |

**关键观察**: MPC3 ref 中两人协作抬箱 (obj_z=0.458), sim 中箱子也达到 0.439 但是被推/翻而非抬起。

## 代码改动

| 文件 | 改动 |
|------|------|
| `spider/config.py` | +`scene_name` 字段; process_config 支持 `dual_humanoid_object` (nq_obj, data_path, scene_name) |
| `spider/simulators/mjwp.py` | +`dual_humanoid_object` branches in: `setup_mj_model`, `_weight_diff_qpos`, `_diff_qpos`, `get_terminate`, `load_env_params` |
| `examples/run_mjwp.py` | Gibbs sampling 扩展: `dual_humanoid_object` → robot1/robot2 nu split |
| `examples/config/override/core4d_box025_dual.yaml` | 双机器人配置 |
| Scene XML | `scene_dual_robot.xml` (nq=79, nu=58) |
| Trajectory | `trajectory_kinematic_dual.npz` (merged person1+person2) |

## 结论

1. **双机器人 Gibbs CEM 架构可行**: 实现从 bimanual 到 dual-humanoid 的泛化
2. **物体抬升略有改善**: 0.488 > 0.477 (E013-r7), 但差异小
3. **CEM horizon 限制不变**: 0.4s 内无法规划协作搬运序列
4. **Body-focused reward 仍是最佳策略**: object tracking 导致单侧崩溃
5. **对 Holosoma RL 有价值**: 双机器人 body-only 轨迹 + 物体参考 → RL 训练数据

## 结果路径

| 产出 | 路径 |
|------|------|
| E016-a best | `workspace/core4d/results/E016a_box025_dual_robot_gibbs.npz/mp4` |
| E016-b objtrack | `workspace/core4d/results/E016b_box025_dual_robot_objtrack.npz/mp4` |
| 配置 | `examples/config/override/core4d_box025_dual.yaml` |
| 场景 | `scene_dual_robot.xml` |
| 合并轨迹 | `trajectory_kinematic_dual.npz` |
