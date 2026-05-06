# E024: Partner-Force Assisted Retargeting — 单侧支撑下的接触重定向

## Context

E001-E023 的总结论: SPIDER CEM 无法单独完成 CORE4D 协作搬运。
但这不是正确的问题设定 — 部署时另一侧是真人, 不需要 learning。

**正确的问题**: 在 partner 提供外力支撑的条件下, SPIDER 能否让 G1 通过物理接触完成自己那侧的协作搬运?

**与 E017-E018 的区别**:
- E017: 双机器人 + connect (手焊死到箱面 = 作弊, 不是接触重定向)
- E024: partner 力支撑 + G1 自由接触 (G1 必须自己产生接触力)

**与 E011 (mocap partner) 的区别**:
- E011: partner 用碰撞体建模 → 太刚性, gap/混沌问题
- E024: partner 用外力 (xfrc_applied) 直接施加到物体 → 干净, 可控, 物理正确

## 物理建模

```
Object (box025, 5kg):
  - 重力: 5 * 9.81 = 49.05N 向下
  - Partner 侧支撑: ~25N 向上 (承担 50% 重力)
  - G1 侧: SPIDER 优化, 期望通过接触产生 ~25N 向上

实现方式 (MuJoCo xfrc_applied):
  - 在每个仿真步, 对 object body 施加 partner 侧的外力
  - 力的大小/位置从 Person2 参考数据推导
  - 或简化为: 恒定向上力 = 物体重力的 50%
```

## 实验矩阵

| Run | Partner Force | Object Reward | Contact Reward | Case | 目的 |
|-----|--------------|---------------|----------------|------|------|
| E024-a | 50% gravity (constant) | 3.0 | 1.0 | box025 | 最简单: 恒力支撑 |
| E024-b | 50% gravity + spring to ref | 3.0 | 1.0 | box025 | 弹簧拉向参考位置 |
| E024-c | Person2 ref force (时变) | 3.0 | 1.0 | box025 | 从参考数据推导partner力 |
| E024-d | 最佳方案 | 5.0 | 2.0 | box025 | 强 obj_rew |
| E024-e | 最佳方案 | 3.0 | 1.0 | bucket010 | 泛化到其他 case |

## Claims (严格标准)

| Claim | 定义 | 阈值 |
|-------|------|------|
| C1: 物体持续离地 | obj_z > init+0.05m 连续帧 | ≥ 30 帧 (1s) |
| C2: G1 手接触物体 | hand-obj surface < 0.05m 连续帧 | ≥ 20 帧 (0.67s) |
| C3: 非碰撞推动 | obj 运动方向与 ref 一致 (非翻转/推飞) | obj_z 曲线与 ref 相关 > 0.5 |
| C4: G1 稳定 | pelvis_z ≥ 0.50m | ≥ 95% 帧 |
| C5: 视频确认协作搬运 | G1 手在箱面, 箱子被抬起而非推/翻 | 定性判断 |

## 实现方案

### 方案 A: 恒定外力 (最简单)
在 mjwp.py 的 step 函数中, 每步对 object body 施加恒定向上力:
```python
# In step_env or similar
xfrc = torch.zeros(nworld, nbody, 6)
xfrc[:, obj_body_id, 2] = 0.5 * obj_mass * 9.81  # 50% gravity comp
data.xfrc_applied = xfrc
```

### 方案 B: 弹簧约束到参考
在 object 上加一个虚拟弹簧, 拉向参考轨迹:
```python
obj_pos_ref = ref_qpos[t, 36:39]  # reference object position
obj_pos_sim = sim_qpos[:, 36:39]
spring_force = kp * (obj_pos_ref - obj_pos_sim)  # PD on position
xfrc[:, obj_body_id, :3] = spring_force
```
这等价于 contact_guidance 的 object actuator, 但力更可控。

### 方案 C: 从 Person2 数据推导
Person2 的手力 ≈ 物体在 Person2 侧受到的支撑:
```python
# Person2 hand positions from reference
p2_hand_pos = ref_partner_data[t]
# Force direction: from person2 hand toward object center, magnitude = share
force_dir = normalize(obj_pos - p2_hand_pos)
force_mag = 0.5 * obj_mass * 9.81
xfrc[:, obj_body_id, :3] = force_dir * force_mag
```

## 代码改动

| 文件 | 改动 |
|------|------|
| `spider/simulators/mjwp.py` | 新增 partner_force 逻辑 in step_env |
| `spider/config.py` | 新增 partner_force_mode, partner_force_scale 字段 |
| `examples/config/override/core4d_box025_partner_force.yaml` | E024 配置 |
| 脚本 | `scripts/retarget/retarget_e024_partner_force.sh` |

## 使用 anchored 轨迹

为避免行走问题干扰, 使用 fullanchor 版本轨迹。
对 box025: 用 xy-anchor (yaw 旋转小, 不需要 full anchor)。

## 风险

1. xfrc_applied 在 mjwarp (GPU) 上是否支持? 需确认 API
2. 恒力过大可能让物体飘起来 (不是被 G1 搬起)
3. 需要区分: 物体移动是 partner 力导致还是 G1 接触导致
