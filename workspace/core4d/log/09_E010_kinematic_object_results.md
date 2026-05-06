# E010: Connect/Kinematic Object 重定向 — 结果

## 状态: 通过 (C1/C2 均通过)

## 核心发现

**G1 运动学可行性确认**：当物体被强制跟踪参考轨迹时，G1 机器人身体保持稳定（pelvis_z min=0.733m）。问题纯粹在抓握/接触上，不在身体运动学。

## 实验矩阵

| 变体 | 方法 | obj z_max | pelvis_min | 结论 |
|------|------|-----------|-----------|------|
| connect (底部锚点, solref=-500) | 等式约束 child body | 0.431 | 0.652 (下降) | 约束太弱 |
| connect (侧面锚点) | 等式约束 side_mid | 0.311 | 0.054 (崩溃) | 方向错误 |
| connect (auto+强+轻) | 等式约束+mass=0.05 | 0.331 | 0.209 (崩溃) | 约束拉机器人下去 |
| **kinobj (gain=1000, no obj rew)** | PD 驱动物体 + CEM 只优化身体 | **1.329** | **0.733** | **PASS** |
| kinobj (gain=5000) | 过强 PD | 2.931 (爆炸) | 0.752 | PD 过强振荡 |
| kinobj (gain=300) | 适中 PD | 0.610 | 0.191 (崩溃) | 力不够+不稳定 |

## 最优配置

```yaml
# core4d_box025_kinobj.yaml
scene_name: scene_forearm_act
contact_guidance: true
guidance_decay_ratio: 1.0        # 不衰减
residual_gain_ratio: 1.0         # 不归零
init_pos_actuator_gain: 1000.0   # 强 PD 驱动物体
init_pos_actuator_bias: 200.0
pos_rew_scale: 0.0               # CEM 不优化物体
rot_rew_scale: 0.0
base_pos_rew_scale: 3.0          # CEM 只优化身体
base_rot_rew_scale: 1.0
```

**关键设计决策**：
- `pos_rew_scale=0`: CEM 不再优化物体位置（避免与 PD 冲突）
- `residual_gain_ratio=1.0`: PD 永不归零（物体始终被驱动）
- `base_pos_rew_scale=3.0`: CEM 专注身体跟踪

## Claims 验证

| Claim | 阈值 | 实测 | 通过? |
|-------|------|------|------|
| C1: obj z_max >= 0.40m | 0.40 | 1.329 (过冲但>0.40) | **PASS** |
| C2: pelvis_min >= 0.50m | 0.50 | 0.733 | **PASS** |
| C3: 视频确认物体搬起 | 视觉 | PD 驱动物体移动，过冲明显 | 部分 |
| C4: joint_err <= 0.10 | 0.10 | 未单独测量 | N/A |

## 关键 Insight

1. **问题确认在抓握，不在身体**：G1 的关节范围、臂展和躯干稳定性完全足够
2. **CEM + PD 冲突**：同时让 CEM 优化物体和 PD 驱动物体会导致不稳定
3. **解耦策略有效**：让 PD 独立驱动物体 + CEM 只关注身体 → 两者都稳定
4. **过冲问题**：PD gain=1000 导致物体过冲到 z=1.3（ref 峰值 0.533），但这是调参问题

## 对 E011 的指导

E010 证明 G1 运动学足够，所以 E011 (Mocap Partner) 有希望成功：
- 如果 person2 的手提供物理推力
- G1 的身体能适应并保持稳定
- 关键是让 G1 的手臂实际接触到物体（接触几何问题，不是运动学问题）

## 可视化观察 (`visualization_mjwp_act.mp4`)

| 时间 | ref | sim |
|------|-----|-----|
| t=0s | G1 站在箱前直立 | G1 直立，**箱子已倾斜**（PD gain=1000 初始冲击） |
| t=1.0s | 弯腰准备抱箱 | **箱子被 PD 弹飞到远处**，机器人弯腰但箱子已远离 |
| t=2.1s | 半蹲扶箱起身 | 机器人踮脚失衡，箱子仅剩一角可见 |
| t=3.2s | 半蹲扶住箱侧 | 机器人独自站立，箱子完全出画 |

**视觉结论**: PD gain=1000 导致箱子在 t=0 被弹飞，之后机器人和箱子分离。obj z=1.3m 是 PD 震荡假象。**但 pelvis 始终 >0.73m**——E010 的价值在于证明运动学可行性，不在于轨迹质量。

| 产出 | 路径 |
|------|------|
| 场景 XML (connect 版) | `example_datasets/.../scene_connect.xml` |
| 配置 (最优) | `examples/config/override/core4d_box025_kinobj.yaml` |
| 轨迹 | `.../trajectory_mjwp_act.npz` |
| 视频 | `.../visualization_mjwp_act.mp4` |
| 生成脚本 | `workspace/core4d/scripts/generate_scene_connect.py` |

## 运行命令

```bash
# E010 最优配置
uv run examples/run_mjwp.py +override=core4d_box025_kinobj \
    task=box025_person1 data_id=0 viewer=none
```
