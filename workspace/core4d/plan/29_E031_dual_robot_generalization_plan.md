# E031: 双机器人 Connect 约束泛化 — 计划

## Context

### E017/E018 已做的
- **仅在 box025 上验证**: soft 2-connect + task-space rewards (DynaRetarget/Harmanoid)
- **E018-d2 最佳结果**: obj_pos_err=0.308m, R1=95%/R2=100% stable, 物体持续离地 83%
- **关键机制**: connect 约束将手"焊接"到箱面 → CEM 只需优化身体 → 物体自动跟随

### E017/E018 存在的问题

1. **仅验证了 box025**: 其他 3 case (bucket010, chair022, desk005) 没有双机器人 scene/trajectory
2. **Gibbs sampling 未生效**: `run_mjwp.py` line 540 只对 `bimanual` 启用 Gibbs, `dual_humanoid_object` 被跳过。E018 实际在 58 维空间直接采样（29+29），未做交替优化
3. **obj_pos_err=0.308m 仍偏大**: DynaRetarget 报告 <0.10m
4. **horizon=1.6s 太短**: 无法规划完整搬运序列
5. **connect 锚点硬编码 box025 half_extents**: `generate_scene_dual_connect.py` 中 `box_half = [0.305, 0.305, 0.446]` 对其他物体不适用

### 论文启发

- **SPIDER Figure 10 (OMOMO)**: SPIDER 可以在单人搬运上做好物体跟踪 (obj_pos=0.18), 用的是 contact guidance
- **Section 4.2**: SPIDER 输出 feedforward control $u_t^{SPIDER}$, RL 只学 residual $\pi_\theta(o_t)$
- **Table 4 (OMOMO)**: joint_err=0.83°, pos_err=0.20cm, ori_err=0.17° — 非常好的质量

## Claims (可验证)

1. **C1 (数据生成)**: 对 bucket010, chair022, desk005 生成双机器人 scene + trajectory (dual_kinematic.npz)
2. **C2 (Gibbs 修复)**: 在 MPC loop 中为 `dual_humanoid_object` 启用 Gibbs 交替优化，观察是否提升
3. **C3 (4 case 覆盖)**: 使用 E018 配置 + connect 约束在 4 case 上运行，3/4 stable (chair022 可能仍失败)
4. **C4 (质量提升)**: obj_pos_err < 0.25m (比 E018 的 0.308 降低 20%)

## 成功标准

- C1: 4 case 都有 `scene_dual_robot_connect2.xml` + `trajectory_kinematic_dual.npz`
- C2: Gibbs 代码修复, 有/无 Gibbs 对比数据
- C3: 至少 3/4 case pelvis stable (R1+R2 ≥ 80% frames)
- C4: obj_pos_err 均值 < 0.25m

## 实验步骤

### Step 1: 数据生成 — 3 case 双机器人 scene + trajectory

1. 对 bucket010, chair022, desk005 运行 `spider/process_datasets/core4d.py` 生成双人轨迹
   - 需要原始 holosoma 双人 retarget 数据 (检查是否存在)
   - 如不存在, 用单人轨迹 + mirror/copy 生成伪双人轨迹
2. 扩展 `generate_scene_dual_connect.py` 使锚点自适应物体尺寸 (从 scene.xml 读取 geom size)
3. 生成 `scene_dual_robot.xml` + `scene_dual_robot_connect2.xml` for 3 new cases

### Step 2: Gibbs 修复

在 `examples/run_mjwp.py` MPC loop 中:
```python
# Line 540: 修改条件
gibbs_enabled = config.gibbs_sampling and config.embodiment_type in ["bimanual", "dual_humanoid_object"]
```
为 `dual_humanoid_object` 添加 robot1/robot2 split (half_nu 方式)

### Step 3: 4 Case 全覆盖运行

- 使用 E018-d2 配置模板
- 每个 case 调整 connect 锚点位置 (来自参考轨迹中手-物体距离)
- 保存视频 + NPZ

### Step 4: 质量优化 (如 C4 未达标)

- 提高 task_obj_pos_rew_scale (40→80)
- 增加 num_samples (2048→4096)
- 尝试更长 horizon (1.6s→2.4s)

## 改动范围

| 文件 | 改动 |
|------|------|
| `spider/process_datasets/core4d.py` | 可能需要扩展支持 dual 模式 |
| `workspace/core4d/scripts/generate_scene_dual_connect.py` | 泛化锚点计算 (不硬编码 box_half) |
| `examples/run_mjwp.py` | Gibbs 修复 (line 540 条件) |
| `examples/config/override/core4d_{case}_e031.yaml` | 4 case 配置 |

## 风险

1. 其他 case 可能没有原始双人 holosoma 数据 → 需要生成
2. chair022 的参考动作对 G1 运动学不友好 → 可能仍然失败
3. bucket010/desk005 的物体形状不是 box → connect 锚点计算需要适配
