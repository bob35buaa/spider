# E030: Orientation Control + Hybrid Export — 结果

## 状态: Orientation 失败; Hybrid export 质量不足以支撑 RL 训练

## 摘要

三部分工作：
1. **Orientation torque debug**: xfrc torque CPU验证OK，Warp batch失败（正反馈），weld约束被碰撞力压倒
2. **Body-only retargeting + hybrid export**: 4 case全覆盖，anchored后4/4 stable
3. **质量评估**: hybrid export 中机器人姿态与搬运动作不匹配，**不足以作为 RL 训练参考**

## Part 1: Orientation Control (失败)

### CPU torque PD 验证

```bash
python workspace/core4d/scripts/debug/test_torque_cpu.py
```

| kp_rot | stable | final_angle | 说明 |
|--------|--------|-------------|------|
| 1.0 | NO | 7.01° | 太弱 |
| 5.0 | OK | 0.00° | 最佳下限 |
| 10.0 | OK | 0.00° | 60°倾斜也能恢复 |
| 20.0 | OK | 0.00° | |
| 50.0 | NO | 63.54° | 过强震荡 |

**结论**: xfrc torque PD 数学正确，CPU 单环境稳定。

### Warp batch 移植 (失败)

```bash
# kp_rot=10, 无保护
MUJOCO_GL=egl python examples/run_mjwp.py +override=core4d_box025_e030 partner_force_spring_kp_rot=10.0
# → pos=29862, quat=2.17, NaN 3880/4096 — 完全发散

# kp_rot=5, NaN保护+clamp
MUJOCO_GL=egl python examples/run_mjwp.py +override=core4d_box025_e030 partner_force_spring_kp_rot=5.0
# → pos=3309, quat=1.49, 中间爆 NaN — 仍不稳定

# kp_rot=5, 无pos spring (隔离测试)
MUJOCO_GL=egl python examples/run_mjwp.py +override=core4d_box025_e030 partner_force_spring_kp=0.0 partner_force_spring_kp_rot=5.0
# → pos=0.73, quat=1.59, 无NaN — 单独torque不爆但orientation跟踪差

# pos-only baseline (无orientation)
MUJOCO_GL=egl python examples/run_mjwp.py +override=core4d_box025_e030 partner_force_spring_kp_rot=0.0
# → pos=0.76, quat=1.43 — E028 baseline
```

**根因**: Position spring (F=kp*err, up to 150N) + orientation torque 产生正反馈:
position force → 接触旋转 → torque 纠正 → 力矩臂产生线性力 → 放大 → 发散

### Weld Equality Constraint (部分有效但不可靠)

```bash
# 生成 scene_weld.xml
python workspace/core4d/scripts/convert/generate_scene_weld.py

# Weld only (最好情况)
MUJOCO_GL=egl python examples/run_mjwp.py +override=core4d_box025_e030_weld partner_force_scale=0.0
# → pos=0.83, quat=0.13 (最好) / quat=0.92 (重复运行) — 高方差

# Weld + gravity comp
MUJOCO_GL=egl python examples/run_mjwp.py +override=core4d_box025_e030_weld partner_force_scale=1.0
# → pos=0.76, quat=0.27-0.53 — gravity comp 干扰 weld

# bucket010 weld
MUJOCO_GL=egl python examples/run_mjwp.py +override=core4d_box025_e030_weld task=bucket010_person1 partner_force_scale=0.0
# → quat=1.14 — 不可靠
```

**视频验证 (box025 weld-only)**: 50%帧时箱子被机器人推倒~45°，75%帧机器人摔倒。碰撞力远超 weld 恢复力。

### Orientation Control 结论
> CEM 随机采样的机器人运动产生的碰撞力远超任何合理的 orientation 恢复力。
> xfrc torque、weld constraint 均不能可靠控制 freejoint 物体 orientation。结构性限制。

## Part 2: Body-only Retargeting + Hybrid Export

### 物体碰撞影响分析

```bash
# 有碰撞 (scene.xml)
MUJOCO_GL=egl python examples/run_mjwp.py +override=core4d_e030_bodyonly task=box025_person1
# → pelvis_z_min=0.769

# 无碰撞 (scene_bodyonly.xml)
python workspace/core4d/scripts/convert/generate_scene_bodyonly.py
MUJOCO_GL=egl python examples/run_mjwp.py +override=core4d_e030_bodyonly task=box025_person1
# (config中 scene_name: scene_bodyonly)
# → pelvis_z_min=0.287 — 摔倒！
```

| Case | 有碰撞 pelvis_z | 无碰撞 pelvis_z | 说明 |
|------|---------------|---------------|------|
| box025 | **0.769** ✓ | 0.287 ✗ | 碰撞必需——机器人靠箱子维持平衡 |
| bucket010 | 0.762 ✓ | 0.736 ✓ | 几乎无影响 |
| chair022 | 0.474 ✗ | **0.631** ✓ | 碰撞有害——椅子干扰平衡 |
| desk005 | 0.741 ✓ | 0.747 ✓ | 几乎无影响 |

**关键发现**: box025 ref动作中人实际靠在箱子上，移除碰撞后机器人前倾摔倒。

### Anchored Retargeting (fullanchor + 保留碰撞)

```bash
# 生成 fullanchor 轨迹
python workspace/core4d/scripts/convert/anchor_pelvis_full.py

# 4 case 全覆盖
for TASK in box025_person1 bucket010_person1 chair022_person1 desk005_person2; do
    DATA_PATH="example_datasets/processed/core4d/unitree_g1/humanoid_object/${TASK}/0/trajectory_kinematic_fullanchor.npz"
    MUJOCO_GL=egl python examples/run_mjwp.py +override=core4d_e030_anchored task=$TASK data_path=$DATA_PATH
done
```

| Case | pelvis_z_min | stable | obj_pos_err |
|------|-------------|--------|-------------|
| box025_person1 | 0.764 | ✓ | 0.163 |
| bucket010_person1 | 0.766 | ✓ | 0.426 |
| chair022_person1 | 0.680 | ✓ | 0.354 |
| desk005_person2 | 0.779 | ✓ | 0.131 |

## Part 3: 质量评估 — 不足以支撑 RL 训练

### 视频验证

```bash
# Hybrid 渲染
MUJOCO_GL=egl python workspace/core4d/scripts/export/render_hybrid.py
```

**视频观察**:
- **box025**: 机器人站立→弯腰趴在箱顶。pelvis 虽然稳定 (z=0.764)，但**机器人是趴在箱子上**而非正常搬运姿态。与 ref 中人站着扶箱的姿态差距大。
- **bucket010**: 机器人弯腰手在桶上方，但**手没有抓住桶**，姿态像是在桶旁站着而非搬运。
- **chair022**: 椅子在物理中被推倒卡在腿间，虽然 hybrid 替换了物体位姿，但**机器人动作与搬椅子无关**。
- **desk005**: 最自然，机器人站桌旁手伸向桌面。但仍是**站在旁边**而非搬运。

### 核心问题

1. **Body-only retargeting 没有搬运语义**: CEM 只优化身体跟踪，不关心手是否接触/搬起物体
2. **Hybrid export 的"物理robot + ref物体"是割裂的**: 机器人身体动作（站/弯腰）与物体运动（被搬起）之间没有因果关系
3. **作为 RL 参考轨迹质量不够**: RL 需要看到机器人**主动搬运**的 demo，而不是机器人在物体旁做类似姿态

### 与论文 OMOMO 结果对比

论文 Table 4 中 SPIDER 在 OMOMO 上 obj_pos_err=0.18, obj_ori_err=0.06。
OMOMO 是**单人搬运**，SPIDER 的 contact guidance 能让手指/手与物体建立正确接触。
CORE4D 是**双人协作**，缺少 partner 力 → 单机器人 CEM 无法产生搬运。

### 结论

> Hybrid export（body-only + ref 物体替换）**不足以作为 RL 训练参考**。
> 机器人姿态只是"在物体旁做类似动作"，不是"主动搬运"。
> 需要转向**双机器人 + connect 约束**方案（E017/E018 已验证可行），
> 这是目前唯一能产生有因果关系的搬运轨迹的方法。

## 结果路径

| 产出 | 路径 |
|------|------|
| Body-only (有碰撞) | `workspace/core4d/results/E030_bodyonly/{case}.{mp4,npz}` |
| Body-only (无碰撞) | `workspace/core4d/results/E030_bodyonly_v2/{case}.{mp4,npz}` |
| Anchored retargeting | `workspace/core4d/results/E030_anchored/{case}.{mp4,npz}` |
| Hybrid 渲染帧 | `workspace/core4d/results/E030_anchored/{case}_hybrid_f{0,1,2}.png` |
| scene_bodyonly.xml | `example_datasets/.../{case}/scene_bodyonly.xml` |
| scene_weld.xml | `example_datasets/.../{case}/scene_weld.xml` |
| CPU torque test | `workspace/core4d/scripts/debug/test_torque_cpu.py` |
| Anchor脚本 | `workspace/core4d/scripts/convert/anchor_pelvis_full.py` |
| Hybrid export | `workspace/core4d/scripts/export/export_hybrid.py` |
| Weld生成器 | `workspace/core4d/scripts/convert/generate_scene_weld.py` |
| Bodyonly生成器 | `workspace/core4d/scripts/convert/generate_scene_bodyonly.py` |
| 配置 | `examples/config/override/core4d_e030_{bodyonly,anchored,hybrid}_*.yaml` |
| 配置 | `examples/config/override/core4d_box025_e030{,_weld}.yaml` |

## 代码改动
| 文件 | 改动 |
|------|------|
| `spider/config.py` | +partner_force_spring_kp_rot, kd_rot, rot_clamp |
| `spider/simulators/mjwp.py` | +orientation torque(disabled in practice), +_update_object_weld_target, +scene_weld dispatch in step_env |
| `examples/run_mjwp.py` | 扩展 ref setup 条件 (scene_weld) |
