# Phase 7 计划: Partner 动态建模 + 接触协同 (E028-E030)

## 现状总结 (E028 初步结果后更新)

### 四 Case 核心数据

| Case | 物体 mass | 手-表面 min | 物体横移 | 物体 lift | 核心障碍 |
|------|-----------|-----------|---------|----------|---------|
| box025 | 5.0kg | 0.102m | 0.10m | 0.22m | 臂展不足 (min>10cm) |
| bucket010 | 3.0kg | 0.003m | **0.87m** | 0.20m | Partner 驱动横向 0.87m |
| desk005 | 5.0kg | 0.110m | 0.20m | 0.11m | 手偏远 + 重 |
| chair022 | 4.0kg | 0.000m | **1.04m** | 0.31m | Partner 驱动横向 1.04m, 但手很近 |

### E028 初步发现

1. **阻尼弹簧解决了 E026 不稳定问题** — 物体不再爆炸/NaN
2. **物体确实在移动** — bucket010 横移 0.47m (54% ref), chair022 lift 29.2cm (93% ref)
3. **手接触维持良好** — bucket010 82%, chair022 91%
4. **问题**: 物体 orientation 不受控 → 椅子翻转, bucket 倾斜

### 待解决

1. 物体 orientation 弹簧 (四元数 PD)
2. 弹簧初始力过大 (t=0 时 ref pos ≠ sim pos → 大力弹射)
3. 需要 ramp-up 机制或位置初始化对齐

## 核心思路: 物体跟随 ref + 机器人接触

**目标**: 物体按参考轨迹运动 (模拟 partner 贡献), 机器人在运动中维持手-物体接触

这等价于: "partner 在另一侧搬运物体, 机器人在这一侧跟随+施力"

### 方案对比

| 方案 | 物体如何动 | 机器人如何跟 | 优点 | 缺点 |
|------|----------|----------|------|------|
| A: 阻尼弹簧 | PD 控制器拉向 ref pos | hand_approach + body tracking | 物体可物理交互 | E026 不稳定; 需调参 |
| B: 运动学物体 | 直接设为 ref pos (无物理) | hand_approach + body tracking | 简单稳定 | 手碰到物体=穿透 (无接触力) |
| C: Mocap body | MuJoCo mocap 追踪 ref | hand_approach + body tracking | 物理碰撞有效 | 无限刚性 (E014 教训) |
| D: 弱 PD actuator | Object freejoint 有 PD gains | CEM 优化 robot + object reward | 物体有物理+可交互 | 需要添加 actuator |

### 推荐: 方案 A (阻尼弹簧) + 方案 D (PD actuator) 双线探索

---

## E028: 阻尼弹簧 — 物体跟随 ref 运动 (修复 E026 不稳定)

**假设**: E026 弹簧不稳定是因为无阻尼。临界阻尼弹簧 (c = 2√(mk)) 可以稳定拉动物体沿 ref 轨迹移动。

**方法**:
```python
# 阻尼弹簧: F = kp * (ref_pos - sim_pos) - kd * vel
spring_force = kp * (ref_pos - obj_pos) - kd * obj_vel
# kd = 2 * sqrt(mass * kp)  — 临界阻尼
```

**实现**:
- 修改 `_apply_partner_force()`: 添加 `partner_force_spring_kd` 阻尼项
- 使用 `data_wp.qvel` 获取物体速度
- 临界阻尼: kd = 2√(m·kp)

**测试矩阵** (bucket010, 4 cases 都做):
- E028-a: kp=20, kd=临界阻尼, + hand_approach=5.0
- E028-b: kp=50, kd=临界阻尼, + hand_approach=5.0
- E028-c: kp=20, kd=临界阻尼, 无 hand_approach (验证弹簧单独效果)
- E028-d: 最佳 kp 在 chair022 上验证

**Claims (严格)**:
- C1: 物体 pos 跟踪 ref pos, RMSE < 0.15m 全程 (对比 ref 横移 0.87m, 15% 误差)
- C2: 物体 z 跟踪 ref z, 峰值误差 < 30% (ref lift=0.20m → sim lift>0.14m)
- C3: 手-物体表面距离 < 0.05m 持续≥50% 帧 (与运动中的物体保持接触)
- C4: 视频确认: 物体在移动, 机器人手在物体上跟随
- C5: pelvis 稳定 (≥0.50m ≥95%)

---

## E029: PD Actuator 驱动物体 — 物体有物理且可交互

**假设**: 给物体 freejoint 添加弱 PD actuator (类似 contact_guidance 的 object_pos_actuator), 但增益设为只够驱动物体跟随 ref, 机器人碰触时能影响物体。

**方法**:
- 在 scene.xml 中给 object 添加 position actuator (kp 足够驱动物体, 但远小于 robot 力)
- CEM 不优化 object actuator (gains 固定, 目标 = ref pos)
- 机器人通过 hand_approach 接触物体, 接触力与 PD 力叠加

**优点**: 物体有真实物理碰撞, 机器人的接触力有意义

**实现**:
- 修改 scene.xml: 添加 object position actuator (ref trajectory as target)
- 或复用 `contact_guidance` 机制但 gains 设为恒定 (不衰减)

**测试矩阵**:
- E029-a: bucket010, object PD kp=100, + hand_approach
- E029-b: chair022, object PD kp=100, + hand_approach
- E029-c: desk005, object PD kp=150 (更重), + hand_approach

---

## E030: 全 Case 验证 + 最佳方案对比

**前提**: E028 或 E029 有一个成功

**测试**:
- 4 个 case × 最佳方案
- 对比 E020 body-only baseline
- 严格 claims: 物体轨迹跟踪 + 手接触 + body 稳定

---

## 实施顺序

1. **E028 (阻尼弹簧)** — 最小改动 (只改 `_apply_partner_force`), 可直接验证
2. **若 E028 成功** → E030 全 case 验证
3. **若 E028 失败** → E029 PD actuator (改动更大, 需要修改 scene.xml)

## Case 特殊策略

| Case | 策略 | 期望 |
|------|------|------|
| box025 | 弹簧拉物体 + approach (即使手碰不到, 弹簧保证物体移动) | 物体跟踪好, 手接触可能仍不足 — 记录为 body-only 可行 |
| bucket010 | 弹簧拉物体 (横向 0.87m) + approach | 最佳候选: 手近+物体在运动 |
| desk005 | 弹簧拉物体 + 强 approach | 手较远, 可能需要更高 approach 权重 |
| chair022 | 弹簧拉物体 + approach | 手极近 (69% <5cm) — 应该最容易维持接触 |

## 风险

| 风险 | 缓解 |
|------|------|
| 阻尼弹簧仍然不稳定 | 测试多个 kp 值; 使用 sub-critical 阻尼 (更保守) |
| 弹簧太强→物体不受 robot 影响 | 降低 kp, 让 robot 的接触力有贡献 |
| 弹簧太弱→物体不跟踪 ref | 提高 kp; 折中: 弹簧保证 70% 跟踪, robot 补剩余 |
| hand_approach 与移动物体冲突 (物体在跑, 手追不上) | 提高 approach sigma (更远距离也有梯度); 或跟踪物体预测位置 |
