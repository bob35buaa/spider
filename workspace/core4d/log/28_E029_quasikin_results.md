# E029: Quasi-Kinematic 位置弹簧 (kp=100) — 确认 xfrc_applied 根本限制

## 状态: FAIL (xfrc_applied 无法控制 freejoint orientation)

## 核心发现

1. **即使 kp=100 + 100% gravity comp, 物体仍然翻转** — 不是参数问题, 是机制问题
2. **位置跟踪改善**: box025 pos_err=0.07m(优秀), desk005=0.13m(好)
3. **chair022 pelvis 崩溃 (27%)**: 强弹簧拉动物体时撞击 robot
4. **根本结论**: xfrc_applied 在 body 质心施力, 无法阻止 freejoint 的自由旋转

## 实验矩阵

| Case | kp | pf | stable% | obj_pos_err | obj_quat_err | hand<10cm | 视频 |
|------|-----|-----|---------|-------------|-------------|-----------|------|
| box025 | 100 | 1.0 | **100%** | **0.072** | 0.33 | 82% | 微翻 |
| bucket010 | 100 | 1.0 | **100%** | 0.317 | 1.14 | 82% | 翻转 |
| desk005 | 100 | 1.0 | **100%** | **0.126** | 0.56 | 60% | **翻倒** |
| chair022 | 100 | 1.0 | **27%** | 0.613 | 1.85 | 91% | 弹射+robot 崩溃 |

## 视频分析

### desk005 (40%): 
- ref: 机器人站在 desk 旁, desk 四平八稳
- sim: desk **侧翻 90°** — 弹簧拉住中心但 desk 绕中心旋转

### desk005 (65%):
- ref: 准备弯腰
- sim: desk **完全倒地**, 面朝 robot

## 根因分析: 为什么 xfrc_applied 不能控制 orientation

```
MuJoCo freejoint = 3 DOF translation + 4 DOF quaternion (球形旋转)

xfrc_applied[:, body_id, 0:3] = 力 (作用于质心)
xfrc_applied[:, body_id, 3:6] = 力矩 (绕质心)

问题 1: 力矩是世界坐标系, 但 quaternion 积分是在 body 坐标系
问题 2: quaternion → axis-angle → torque 在大角度时有奇异性
问题 3: 无阻尼状态的 torque 控制需要极精确的 kp/kd 匹配

对比: MuJoCo position actuator on hinge joint
- 每个旋转轴独立 (euler 分解, 无 gimbal lock 问题在小角度)
- PD 控制器内置在 solver 中, 天然闭环稳定
- 不需要手动计算 quaternion error
```

## 结论: 需要 joint-level 物体驱动

xfrc_applied 方案已穷尽:
- kp=20: 跟踪弱
- kp=30-50: 跟踪中, 翻转
- kp=100: 跟踪好, 仍翻转
- orientation spring: 任何增益都 NaN
- angular damping: 也 NaN 或使 pos 跟踪恶化

**下一步: 生成 scene_act.xml (3 slide + 3 hinge + 6 position actuator), 用 SPIDER 现有 contact_guidance 机制驱动物体。**

## 结果路径

| 产出 | 路径 |
|------|------|
| 4 case 结果 | `workspace/core4d/results/E029_quasikin/{case}/result.{npz,mp4}` |
| 关键帧 | `workspace/core4d/results/E029_quasikin/{desk005,box025}/frames/` |
