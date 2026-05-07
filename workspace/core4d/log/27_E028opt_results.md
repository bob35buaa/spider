# E028-opt: 阻尼弹簧 4 Case 全覆盖 + Orientation 控制探索

## 状态: 部分成功 (位置跟踪+手接触有效, orientation 不受控)

## 4 Case 结果 (position-only damped spring + hand_approach)

| Case | kp | pf | stable | obj_z_track | obj_xy_track | hand<10cm | 视频 C6 |
|------|-----|-----|--------|------------|-------------|-----------|--------|
| box025 | 30 | 0.85 | **100%** | 21% | 45% | **82%** | 物体微动, 手碰到 |
| bucket010 | 30 | 0.85 | **100%** | 38% | 64% | **82%** | 物体横移, 手跟随, 翻转 |
| desk005 | 15 | 0.85 | **100%** | **84%** | **83%** | **90%** | **desk 翻倒** ❌ |
| chair022 | 15 | 0.85 | **100%** | 148%(过冲) | 37% | 73% | 椅子弹射/翻转 ❌ |

## Claims 验证

| Claim | 结果 | 通过? |
|-------|------|------|
| C1: 4 case 无 NaN/爆炸 | 全部正常运行 | **PASS** ✅ |
| C2: obj_z ≥ 60% | desk005=84%✅, 其余<60% | **PARTIAL** (1/4) |
| C3: obj_xy <50% err | bucket010=64%✅, desk005=83%✅ | **PARTIAL** (2/4) |
| C4: hand<10cm ≥50% (bucket+chair) | bucket=82%, chair=73% | **PASS** ✅ |
| C5: pelvis stable | 全部 100% (kp=15时) | **PASS** ✅ |
| C6: 视频像搬运 | **全部翻转** — orientation 不受控 | **FAIL** ❌ |

## Orientation 控制尝试

| 方法 | 结果 |
|------|------|
| Quaternion PD (kp_rot=kp*0.1) | NaN 爆炸 |
| Quaternion PD (kp_rot=1.0) | NaN 爆炸 |
| Angular velocity damping (kd=0.5) | 稳定但 pos_err 恶化 |
| 无 orientation 控制 | 最稳定, 物体翻转但跟踪好 |

**根因**: xfrc_applied torque + MuJoCo 物理步进 + 大 timestep (0.0167s) → quaternion 积分不稳定。需要更底层的解决方案 (如 MuJoCo equality constraint 或 joint damping)。

## 结论

**位置弹簧作为 partner 模型**: 技术可行, 数值稳定, hand_approach 有效引导接触。
**核心限制**: 物体翻转使得视频不合格, 但数值指标良好。

**对 RL 训练数据生成的意义**:
- Robot body motion 是物理合规的 (pelvis 100% stable, 合理姿态)
- 物体轨迹应使用 **reference** (非 sim 中翻转的物体)
- 可以导出 hybrid: sim_robot + ref_object + partner_data

## 后续方向

1. **接受当前方案导出 hybrid 轨迹** (E030): sim robot + ref object, 供 RL 训练
2. **使用 MuJoCo equality constraint** (weld 约束 + 弱化) 控制 orientation
3. **转向 E017-style connect + 弹簧**: weld hand→object + 弹簧 object→ref

## 结果路径

| 产出 | 路径 |
|------|------|
| 4 case kp=30 | `workspace/core4d/results/E028_damped_spring/{case}/e028_kp30.{npz,mp4}` |
| desk005/chair022 kp=15 | `.../{case}/e028_kp15.{npz,mp4}` |
| desk005 frames | `.../desk005/frames_kp15/` |
| 计划 | `workspace/core4d/plan/27_E028opt_E030_spring_optimization_plan.md` |
