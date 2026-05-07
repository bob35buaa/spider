# Phase 7 计划: Partner 弹簧优化 + 4 Case 全覆盖 (E028opt-E030)

## Context

E028 初步验证了阻尼弹簧方案:
- bucket010: obj_z=86% ref, hand<5cm=82%, stable=100%
- chair022: obj_z=93% ref, hand<5cm=91%, 但物体翻转 (orientation 不受控)
- **Orientation spring 实现了但导致爆炸** (torque 方向/增益问题)
- **Ramp-up (0.5s)** 有效避免了初始弹射

核心差距:
1. Object orientation 不受控 → 物体翻倒
2. obj_pos_err 仍 0.35m (xy 跟踪不足, 弹簧强度与 robot 碰撞 tradeoff)
3. 仅测了 bucket010 和 chair022, 未覆盖 box025/desk005

## 实验系列

### E028-opt: 修复 Orientation Spring + 参数优化

**子实验 A: 修复 orientation torque**

问题诊断: torque 可能在世界坐标系计算但施加在物体坐标系 (MuJoCo xfrc_applied torque 是世界系)。
- 检查: xfrc_applied[:, body_id, 3:6] 是世界系 torque → quaternion error 也应在世界系 → 应该正确
- 可能原因: kp_rot = kp * 0.1 仍然太大 (kp=30 → kp_rot=3.0, 对 4kg 物体 torque=3*2π≈19 Nm)
- 修复: kp_rot = 1.0 (固定小值) + kd_rot subcritical

**子实验 B: 4 Case 全覆盖 (position-only spring)**

| Case | kp | pf | 预期 |
|------|-----|-----|------|
| box025 | 30 | 0.85 | 物体跟踪好, 手可能仍够不到 |
| bucket010 | 30 | 0.85 | 已验证: z=86%, hand=82% |
| desk005 | 30 | 0.85 | 手偏远, 可能需 approach=10 |
| chair022 | 30 | 0.85 | z=93%, 但翻转 → 加 orientation 修复 |

**子实验 C: kp 调优 (bucket010)**

| kp | 预期效果 |
|----|---------|
| 20 | 弱跟踪, robot 能影响物体 |
| 40 | 中等, 平衡跟踪与交互 |
| 60 | 强跟踪, robot 影响小 |
| 100 | 几乎运动学 (接近 kinobj) |

**Claims (严格)**:
- C1: 4 case 全部运行成功 (无 NaN/爆炸)
- C2: 物体 z 方向跟踪率 ≥ 60% (sim_lift/ref_lift)
- C3: 物体 xy 轨迹 RMSE < 物体横移距离的 50%
- C4: 手-表面距离 <10cm ≥ 50% 帧 (至少 bucket010 + chair022)
- C5: Robot 稳定 (pelvis ≥ 0.50m ≥ 95%)
- C6: 视频确认物体在"被搬运" (非翻转/弹射), 且机器人手在物体旁

---

### E029: Contact Guidance 复用 (若 E028 不够)

**条件**: E028-opt C6 不过 (视频仍不像搬运)

**思路**: 利用 SPIDER 现有 `contact_guidance` 机制:
- 给 object 添加 position actuator (gain = fixed, 不衰减)
- Actuator target = ref pos → 物体被 PD 驱动
- 机器人通过碰撞与"正在运动的物体"交互
- 区别于 spring: actuator 通过 joint 力控制, 更稳定

**实现**: 修改 scene.xml 添加 object actuator, 或复用 `contact_guidance=True` + `guidance_decay_ratio=0.0`

---

### E030: Hybrid 轨迹导出 (成功后)

**条件**: E028 或 E029 的视频通过 C6

**产出**: 对通过的 case 导出 hybrid NPZ:
- robot qpos: SPIDER 物理合规 (sim)
- object qpos: ref 轨迹 (或 spring-tracked)
- partner data: 双手世界坐标 (供 RL)

格式与 E012 相同, 供 Holosoma RL 训练。

---

## 实施顺序

```
1. 修复 orientation spring (降低 kp_rot, 验证 torque 方向)
2. 4 Case 全跑 position-only spring (E028-opt-B)
3. 视频分析: 判断哪些 case 通过 C6
4. 对通过的 case 加 orientation spring (E028-opt-A)
5. kp 调优找最佳 (E028-opt-C)
6. 若仍不够 → E029 contact_guidance
7. 导出 hybrid 轨迹 (E030)
```

## 代码改动

### E028-opt-A (orientation spring 修复):
```python
# 降低 kp_rot: 固定 1.0 而非 kp*0.1
kp_rot = 1.0 * ramp  # 固定小增益
kd_rot = 0.5  # 固定小阻尼 (不用 critical damping — 过大)
```

### E028-opt-B (4 case 配置):
创建 `core4d_{case}_e028.yaml` × 4

### E029 (contact_guidance 复用):
```yaml
contact_guidance: true
guidance_decay_ratio: 0.0  # 不衰减 → 物体一直被驱动
hand_approach_rew_scale: 5.0
```

## 成功标准

**Phase 7 整体成功 = 至少 2 个 case 通过以下全部**:
1. 物体沿 ref 方向移动 (z ≥ 60%, xy 方向正确)
2. 机器人手在物体旁 (<10cm ≥ 50% 帧)
3. 视频看起来像"跟随物体/协助搬运" (非推倒/弹射)
4. Robot 稳定

## 风险

| 风险 | 缓解 |
|------|------|
| Orientation spring 仍不稳定 | 用极小增益 (kp_rot=0.5); 或放弃 orientation, 接受纯 position spring |
| box025 手仍够不到 | 预期如此 — 记录为"body-only case", 物体由 spring 搬运, 机器人做 body tracking |
| 高 kp 使机器人碰撞无意义 | 这是可接受的 — 目的是生成训练数据, 不是证明 CEM 能搬运 |
| 视频仍不像搬运 | E029 contact_guidance 作为 fallback |

## 数据路径

| 产出 | 路径 |
|------|------|
| E028-opt 结果 | `workspace/core4d/results/E028_damped_spring/{case}/` |
| E029 结果 | `workspace/core4d/results/E029_contact_guidance/{case}/` |
| E030 导出 | `workspace/core4d/results/E030_hybrid_export/{case}/` |
| 配置 | `examples/config/override/core4d_{case}_e028.yaml` |
