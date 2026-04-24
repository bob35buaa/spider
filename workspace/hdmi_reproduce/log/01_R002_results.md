# R002 实验结果

**日期**: 2026-04-25
**对应Plan**: `workspace/hdmi_reproduce/plan/01_R002_fix_init_alignment_plan.md`

## 1. 背景

R001 首次运行 HDMI Warp batch stepping，机器人 0.2 秒内倒地。根因：qpos layout 不匹配（MuJoCo 实际是 [obj, pelvis, joints]，代码假设 [pelvis, joints, obj]）。

## 2. R002: 修复 qpos 布局 + sim/ref 对齐

**改动**:
- get_reference: 按 MuJoCo qpos 实际布局，用 mj_name2id 逐个映射 joint
- setup_env: 用 qpos_ref[0] + 正确 joint address 初始化 sim
- _diff_qpos: 参数化 pelvis_qpos_adr/obj_qpos_adr

### 结果 (100 sim_steps = 50 ctrl_steps = 2 秒)

| 指标 | R001 | **R002** | HF 参考 |
|------|------|---------|---------|
| Pelvis Z (mean) | 0.008m | **0.789m** | 0.786m |
| Pelvis Z (range) | [-0.006, 0.83] | **[0.776, 0.830]** | [0.775, 0.828] |
| 持续时间 | 0.2s (5步) | **2s (50步)** | 10s (250步) |
| opt_steps (典型) | 32 | **1** | N/A |
| 每步耗时 | ~40s | **~1.3s** | N/A |
| 总时间 | 167s | **76s** | N/A |

### kinematic 对比
- get_reference 输出与 HF trajectory_kinematic.npz **逐元素 diff = 0**（重排序后完全一致）
- 确认 qpos 布局差异：R002 用 MuJoCo 物理顺序 [obj(7), pelvis(7), joints(29)]，HF 用逻辑顺序 [pelvis(7), joints(29), obj(7)]

### 物理轨迹对比
- Pelvis Z 逐帧差距 < 0.01m
- 整体 qpos L2 距离 mean=1.808, 随时间增大到 2.6（物体跟踪发散）
- opt_steps=1 说明优化几乎没探索就停了（improvement < threshold）

### 视频观察
- 机器人正确站立，全程不倒
- 箱子跟踪有偏差（飞起/翻转）
- kinematic 侧渲染有错（qpos 布局不匹配 render 脚本）

## 3. Claims 验证

| Claim | 结果 |
|-------|------|
| kinematic npz 与 HF 一致 (diff < 1e-3) | **通过** — diff = 0 |
| Pelvis Z > 0.5m 持续 100 步 | **通过** — min=0.776m |
| qpos_dist < 0.5m | **未通过** — mean=1.808m（物体跟踪差） |

## 4. 下一步

1. **物体跟踪发散**: qpos L2 距离主要来自物体。需检查 _diff_qpos 对物体部分的权重和计算是否正确
2. **opt_steps=1 过低**: improvement_threshold=0.02 可能太低，或奖励尺度有问题，导致第一次迭代就 "收敛"
3. **渲染脚本适配**: render_trajectory_video.py 的 kinematic 侧需要 reorder qpos 到 MuJoCo 布局
4. **跑完整 500 步轨迹**: 验证长时间稳定性
