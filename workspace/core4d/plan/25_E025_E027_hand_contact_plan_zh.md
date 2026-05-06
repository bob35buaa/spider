# Phase 6 计划: 手部接触引导 (E025-E027)

## 背景

E024 (Phase 5 最终实验) 确认了 SPIDER CEM **无法**通过以下方式产生接触操作:
- Partner force (重力补偿)
- 单纯 body tracking
- 单纯 object reward

根因: CEM 在关节空间采样, 无法发现"伸手→接触→施力"的因果序列。

## 关键数据发现 (计划前分析)

锚定参考轨迹中手到物体**表面**距离:

| Case | 平均距离 | <5cm 帧数 | <10cm 帧数 | 可行性 |
|------|---------|-----------|------------|--------|
| bucket010 | 0.074m | **50%** | **75%** | **首选目标** |
| desk005 | 0.145m | 0% | 0% | 次选 (需要 approach 引导) |
| box025 | 0.175m | 0% | 2% | 排除 (臂展限制) |

**bucket010 是最佳目标**: 参考轨迹中 G1 手腕已经在物体表面附近, hand-approach reward 可以提供梯度补上剩余的 ~5-7cm。

## 实验系列

### E025: 手部接近奖励 (主实验)

**假设**: 添加显式的手到物体表面距离奖励, 为 CEM 提供朝向接触的梯度, 克服"盲目采样"的限制。

**方法**:
```python
# 新奖励分量: hand_approach_rew
hand_approach_rew = hand_approach_rew_scale * exp(-sigma * min_hand_surface_dist)
```

与 `contact_rew` (仅在接触后激活) 不同, 该奖励**始终生效** — 从任意距离提供梯度。与 `task_body_rew` (跟踪参考位置) 不同, 该奖励直接最小化手-物体距离, 不依赖参考。

**实现步骤**:
1. 在 Config 中添加 `hand_approach_rew_scale` 和 `hand_approach_sigma`
2. 在 `get_reward()` 中计算手到物体距离并添加指数衰减奖励
3. 使用物体碰撞几何体的包围盒 (`model.geom_size`) 近似表面距离
4. 添加 partner force (50% 重力补偿) 降低物体重量要求

**测试矩阵** (bucket010 锚定轨迹 + partner force 50%):
- E025-a: hand_approach=5.0, sigma=5.0, partner_force=0.5
- E025-b: hand_approach=10.0, sigma=10.0, partner_force=0.5
- E025-c: hand_approach=5.0, sigma=5.0, partner_force=0.0 (无 partner, 纯 approach)
- E025-d: hand_approach=5.0 + task_body_rew=1.0 (hand weight=20.0), partner_force=0.5

**Claims (严格验证)**:
- C1: 仿真中手-表面距离 < 0.03m 持续 ≥30 连续帧 (持续接近)
- C2: 视频确认手明显伸向物体 (不是停在体侧)
- C3: 物体 z 方向位移与手部接近在时间上相关 (因果检验)
- C4: pelvis_z ≥ 0.50m ≥95% 帧数 (稳定性保持)
- C5: 相对 E024 基线有改善 (仿真中手距 < E024-a1 的手距)

**成功标准**: C1 + C2 + C4 全部通过 → 进入 E026 (approach + push)

---

### E026: 手部接近 + 物体弹簧 (方向 3)

**前提条件**: E025 实现手部接近 (C1 通过) 但物体不动。

**假设**: 如果手到达物体但无法产生足够力, 一个拉向参考位置的弱弹簧提供"初始推力" — CEM 只需维持接触, 无需发起抬升。

**方法**:
```python
# 组合: hand_approach (引导手接近) + spring (辅助物体运动)
partner_force_spring_kp = 50.0  # 弱弹簧拉向参考位置
hand_approach_rew_scale = 5.0
```

**实现**: 使用 E024 已有的 `partner_force_spring_kp` 基础设施。

**测试矩阵** (bucket010 锚定轨迹):
- E026-a: approach=5.0 + spring_kp=50 (弱弹簧)
- E026-b: approach=5.0 + spring_kp=100 (中等弹簧)
- E026-c: approach=10.0 + spring_kp=50 + obj_rew=3.0

**Claims**:
- C1: obj_z > 初始值+0.05m 持续 ≥30 帧 (50fps 控制频率下约 0.6s)
- C2: 物体运动时手-物体有接触 (非纯弹簧驱动)
- C3: 视频确认物体运动时手在物体上
- C4: 稳定性保持

---

### E027: 多 Case 验证 + 高权重手部跟踪

**前提条件**: E025 或 E026 在 bucket010 上有效。

**假设**: 成功方案可通过参数调整迁移到 desk005。

**测试矩阵**:
- E027-a: 最佳 E025/E026 配置应用于 desk005 (更长接近距离)
- E027-b: 组合 hand_approach + task_body_rew (hand_weight=30.0) 在 bucket010
- E027-c: 组合方案应用于 desk005

---

## 代码改动

### `spider/config.py` 新增字段:
```python
# E025: 手部接近奖励 — 引导手趋向物体表面
hand_approach_rew_scale: float = 0.0
hand_approach_sigma: float = 5.0  # 指数衰减陡度
hand_approach_body_names: list[str] = field(default_factory=lambda: ["left_wrist_yaw_link", "right_wrist_yaw_link"])
hand_approach_body_ids: list[int] = field(default_factory=list)  # 运行时解析
```

### `spider/simulators/mjwp.py` 在 `get_reward()` 中新增:
```python
# E025: 手部接近奖励 — 手到物体表面距离的指数衰减
hand_approach_rew = torch.zeros(N, device=config.device)
if config.hand_approach_rew_scale > 0.0 and config.hand_approach_body_ids:
    xpos_sim = wp.to_torch(env.data_wp.xpos)  # (N, nbody, 3)
    hand_pos = xpos_sim[:, config.hand_approach_body_ids]  # (N, 2, 3)
    obj_pos = xpos_sim[:, obj_body_id:obj_body_id+1]  # (N, 1, 3)
    # 使用物体几何体半尺寸近似表面距离
    delta = torch.abs(hand_pos - obj_pos)  # (N, 2, 3)
    half_ext = torch.tensor(obj_half_extents, device=config.device)  # (3,)
    surface_dist = torch.clamp(delta - half_ext, min=0.0)  # (N, 2, 3)
    min_dist = surface_dist.norm(dim=-1).min(dim=1).values  # (N,) — 取较近手
    hand_approach_rew = config.hand_approach_rew_scale * torch.exp(-config.hand_approach_sigma * min_dist)
```

### 配置 YAML 模板 (`core4d_bucket010_e025.yaml`):
```yaml
dataset_name: core4d
task: bucket010_person1
data_id: 0
robot_type: unitree_g1
embodiment_type: humanoid_object
data_path: "...trajectory_kinematic_anchored.npz"

# 手部接近奖励
hand_approach_rew_scale: 5.0
hand_approach_sigma: 5.0
hand_approach_body_names: ["left_wrist_yaw_link", "right_wrist_yaw_link"]

# Partner force (50% 重力补偿)
partner_force_scale: 0.5

# Body tracking (中等权重)
base_pos_rew_scale: 10.0
joint_rew_scale: 3.0
```

## 实施顺序

1. 添加配置字段 → `spider/config.py`
2. 添加 `hand_approach_rew` 到 `get_reward()` → `spider/simulators/mjwp.py`
3. 在 `process_config()` 中解析 `hand_approach_body_ids` → `spider/config.py`
4. 创建 E025 各变体的配置 YAML
5. 在 bucket010 上运行 E025-a 到 E025-d
6. 分析结果 (视频 + 指标)
7. 若 E025 通过 → E026, 否则 → 重新评估

## 数据路径

| 类型 | 路径 |
|------|------|
| 输入轨迹 | `example_datasets/processed/core4d/.../bucket010_person1/0/trajectory_kinematic_anchored.npz` |
| 实验结果 | `workspace/core4d/results/E025_hand_approach/bucket010/{a,b,c,d}.{npz,mp4}` |
| 配置文件 | `examples/config/override/core4d_bucket010_e025.yaml` |
| 日志 | `workspace/core4d/log/23_E025_hand_approach_results.md` |

## 风险评估

| 风险 | 缓解措施 |
|------|---------|
| CEM 仍忽略 approach 奖励 (相对 body tracking 太弱) | 扫描权重: 5/10/20 |
| 手到达物体但无法产生接触力 | E026 弹簧辅助 |
| 物体半尺寸近似不准确 (非 box 形状) | 使用模型中实际 geom 尺寸 |
| bucket010 成功但不泛化到其他 case | E027 验证泛化性 |
| 奖励冲突 (approach 将手拉离 body 参考位置) | E025-d 测试组合权重 |
