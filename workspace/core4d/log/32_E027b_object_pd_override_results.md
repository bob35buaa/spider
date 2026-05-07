# E027b: Object PD Override — 物体沿 Ref 6DOF 运动 + Robot CEM

## 状态: 部分成功 — desk005/box025 高质量, bucket010/chair022 待改善

## 核心思路

解决 E025-E027 提出的"partner 横向搬运动作缺失"问题：
- 使用 `scene_act.xml`（6 position actuators: 3 slide + 3 hinge）
- 在 `step_env` 中每步覆盖 object actuator ctrl 为 PD 目标（从 ref 查表）
- CEM 只优化 robot joints，object ctrl 被确定性覆盖
- 加入 gravity compensation offset: `target_z += mg/kp`

## 运行命令

```bash
# Step 1: 重新生成 scene_act.xml (armature=2.0, damping=100/20, kp=2000, per-case euler)
uv run workspace/core4d/scripts/convert/generate_scene_act.py

# Step 2: 运行 4 cases
for TASK in box025_person1 bucket010_person1 desk005_person2 chair022_person1; do
    MUJOCO_GL=egl uv run examples/run_mjwp.py +override=core4d_e027b task=$TASK \
        data_path=example_datasets/processed/core4d/unitree_g1/humanoid_object/${TASK}/0/trajectory_kinematic_anchored.npz
done
```

## 数量结果 (最终版: relative euler + per-case convention)

| Case | stable% | pos_err | rot_err° | z_track% | 视觉 |
|------|---------|---------|----------|----------|------|
| box025 | 100% | 0.101 | **7.6°** | 64% | ★★ |
| bucket010 | 100% | 0.547 | 61.5° | 127% | ★ |
| desk005 | 100% | **0.100** | **8.8°** | **99%** | ★★★ |
| chair022 | 100% | 0.559 | 110.6° | 126% | ✗ |

## Bug 修复历程

### Bug 1: Euler 约定 (intrinsic vs extrinsic)
- **问题**: 初始用 `"xyz"` (intrinsic)，MuJoCo hinge 需要 extrinsic
- **修复**: 改为 extrinsic 大写约定
- **效果**: box025 rotation 匹配 (rot_err 从 90°+ → 5°)

### Bug 2: Slide position 偏移
- **问题**: freejoint qpos 是世界坐标, scene_act slide joints 是相对于 body_pos 的偏移
- **修复**: `slide_pos = world_pos - body_pos`
- **效果**: position 初始值正确 (pos_err 从 0.37 → 0.09)

### Bug 3: Gimbal lock
- **问题**: 固定 XYZ 约定对某些物体初始方向接近 gimbal lock (bucket010 |mid|=77°)
- **修复**: `generate_scene_act.py` 自动选择使 max|mid| 最小的约定, 生成对应轴序的 hinge joints
- **效果**: 所有 case max|mid| < 8°, 但未解决根本问题

### Bug 4: body_quat 非 identity (根本原因!)
- **问题**: bucket010/desk005/chair022 的 object body 在 XML 中有非 identity 的初始 `body_quat`。直接将世界旋转转 euler 赋给 hinge joints 是错误的——hinge 控制的是**相对于 body frame** 的旋转。
- **修复**: `R_joint = R_body.inv() * R_world`, 然后 `euler = R_joint.as_euler(convention)`
- **效果**: desk005 rot_err 90.3° → **8.8°**, bucket010 143.7° → **61.5°**

## 视频观察

### desk005 ★★★ (最佳)
- **t=0**: 机器人站在桌旁, 桌子方向/位置与 ref 完美匹配
- **t=50**: 桌子保持正确方向, 机器人姿态与 ref 接近(略前倾), 手在桌面高度
- **首次实现"机器人配合运动中物体"的完整视觉效果**, pos+rot 均正确

### box025 ★★
- **t=0**: 箱子方向完美匹配 ref
- **t=50**: 箱子有轻微倾斜 (7.6°), 机器人姿态合理但 z_track 只有 64% (箱子没抬到 ref 高度)
- 问题: 箱子太大 (0.61m), 机器人碰撞压低物体

### bucket010 ★
- **t=0**: 桶方向正确
- **t=50**: rot_err=61.5° — 桶被机器人碰撞推转。kp_rot=2000 不够抵抗接触力矩
- pos_err=0.547 — 横向偏移大

### chair022 ✗
- rot_err=110.6° — body_quat fix 对 chair 没明显改善, 可能还有其他问题

## 技术改动

| 文件 | 改动 |
|------|------|
| `spider/config.py` | +object_pd_override, object_pd_kp_pos=2000, object_pd_kp_rot=2000, nq_obj=6 override |
| `spider/simulators/mjwp.py` | +_apply_object_pd_override() in step_env: 读 ref euler, 加 grav_comp, 覆盖 ctrl |
| `examples/run_mjwp.py` | +E027b ref 转换: freejoint→(slide_pos + relative_euler), 读 scene_act_meta.json euler convention, R_body.inv()*R_world |
| `workspace/core4d/scripts/convert/generate_scene_act.py` | per-case 最佳 euler 约定, armature=2.0, kp=2000, +scene_act_meta.json |
| `examples/config/override/core4d_e027b.yaml` | 新配置 |

## 关键发现

1. **CUDA graph 不会更新 model params**: runtime `wp.copy` to `model_wp.actuator_gainprm` 无效! gains 必须 bake 进 XML。
2. **MuJoCo body_quat 非 identity**: 当 body 有初始旋转时, hinge joints 控制的是相对旋转, 不是世界旋转。
3. **Per-case euler convention**: 不同物体需要不同 euler 约定避免 gimbal lock (bucket=XZY, box=YXZ, desk/chair=XZY)。

## 下一步

1. **导出 desk005 hybrid 轨迹** — 质量足够支撑 RL (pos=0.10, rot=8.8°, stable=100%)
2. **增加 kp_rot** 解决 bucket/chair — 但需要更大 armature 保持稳定
3. **移除 hand_approach_rew** — 对 desk005 不需要且可能有害 (box025 中导致机器人趴上去)
4. **修复 chair022** — 可能需要检查 body_quat 计算的正确性

## 结果路径

| 产出 | 路径 |
|------|------|
| 视频 | `workspace/core4d/results/E027b_{box025,bucket010,desk005,chair022}.mp4` |
| 帧截图 | `workspace/core4d/results/E027b_frames/` |
| 配置 | `examples/config/override/core4d_e027b.yaml` |
| Plan | `workspace/core4d/plan/30_E027b_object_pd_override_plan.md` |
| scene_act 生成 | `workspace/core4d/scripts/convert/generate_scene_act.py` |
| meta | `example_datasets/.../scene_act_meta.json` (每个 case) |
