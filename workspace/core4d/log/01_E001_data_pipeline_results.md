# E001: 数据管线验证 — 结果

## 状态: 通过

## 结果

### Claims 验证
1. **qpos 无损转换**: qpos 完全一致 (`np.allclose` = True)
2. **场景 XML 正确**: scene.xml nq=43,nv=41,nu=29; scene_act.xml nq=42,nv=41,nu=35

### 输出数据
| Key | Shape | 说明 |
|-----|-------|------|
| qpos | (124, 43) | pelvis(7) + 29 joints + object(7) |
| qvel | (124, 41) | pelvis(6) + 29 joints + object(6) |
| ctrl | (124, 29) | 29 joint PD targets |
| contact | (124, 2) | left/right hand (all ones) |
| contact_pos | (124, 2, 3) | hand site world positions |

### 关键指标
- 帧数: 124 (30fps → 4.13s 序列)
- 手-物体中心距: 0.6-1.0m (物体半径~0.3m)
- 视频输出: `visualization_kinematic.mp4`

### 文件路径
- 数据: `example_datasets/processed/core4d/unitree_g1/humanoid_object/box025_person1/0/trajectory_kinematic.npz`
- 场景: `example_datasets/processed/core4d/unitree_g1/humanoid_object/box025_person1/scene.xml`
- 视频: `example_datasets/processed/core4d/unitree_g1/humanoid_object/box025_person1/0/visualization_kinematic.mp4`

## 下一步
- E002: 运行 SPIDER MJWP 优化器 (无 contact guidance)
- E003: 运行 SPIDER MJWP 优化器 (有 contact guidance)
