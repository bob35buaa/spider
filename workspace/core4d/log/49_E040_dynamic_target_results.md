# E040: Dynamic Per-Frame Contact Target — 解决 "手粘连物体" 问题

## 状态: 失败 — 手背接触物体问题未解决, 动态 target 无法修复 position-only reward 的方向缺陷

## 背景

E039b/E039c 修复 config bug 后, contact reward 首次正确生效 (box025: 85%, bucket010: 76%)。但发现严重的 **"手粘连物体"** 问题:
- 固定 `contact_target_offset` 假设手始终在物体同一个点
- CORE4D 任务中人围绕物体活动, 手在物体表面位置随时变化
- 固定 target → 手被拉回固定点 → 手腕反关节 + 身体扭转

E040 用 ref 中每帧手相对物体的实际位置作为该帧的 target。

## 核心改动

```python
# 预计算: 每帧 ref 中手在物体局部坐标系的位置
for t in range(T):
    target_np[t, ei] = obj_mat[t].T @ (hand_pos_ref[t] - obj_pos_ref[t])

# 运行时: 每帧使用当帧的 target (已随 ref_data 按时间步索引)
target_world = obj_pos_sim + quat_apply(obj_quat_sim, target_offset_t)
```

## 结果

| Case | E036 (无contact) | E039b (固定target) | **E040 (动态target)** | Ref 上限 | Stability |
|------|------|------|------|------|------|
| **box025** | 56% | **85%** | 64% | 69% | **100%** |
| **bucket010** | 2% | **76%** | 66% | 76% | **100%** |
| **desk005** | 7% | 81% (stable=20%) | 4% | 0% | **81%** |

| Case | MPKPE | Joint Err | Obj Pos Err |
|------|-------|-----------|-------------|
| box025 | 1.3cm | 0.8° | 0.7cm |
| bucket010 | 1.2cm | 0.8° | 0.9cm |
| desk005 | 1.9cm | 0.9° | 1.0cm |

### Contact Preservation (ref 接触帧中 sim 也接触的比例)

| Case | sim<10cm \| desired | sim<5cm \| desired |
|------|-------|-------|
| **box025** | **88.8%** | 69.7% |
| **bucket010** | **95.4%** | 4.6% |
| desk005 | 6.8% | 0.0% |

## 可视化分析

### box025 — Contact 64%, Stability 100%, MPKPE 1.3cm

| 帧 | ref | sim | 观察 |
|------|-----|-----|------|
| t=0.8s | 弯腰手搭箱面 | 弯腰手在箱面附近 | 手在箱面附近 |
| t=1.7s | 站箱旁手扶箱面 | 站箱旁手在箱面附近 | 姿态有改善但仍有问题 |
| t=2.0s | 站着手扶箱面 | **弯腰趴向箱顶, 手背朝向箱子** | ❌ **严重不自然: 手背接触** |
| t=3.3s | 弯腰趴箱面 | 弯腰手在箱面 | 接近 ref |

**❌ 问题未解决**: 视频 t≈2s 处清晰可见 sim 机器人弯腰趴向箱顶, **手背朝向物体** (而非手掌)。这与 E039b 的 "手粘连" 是同一类问题 — position-only reward 导致 CEM 用不自然的手腕旋转来满足距离约束。动态 target 只改变了粘连的位置, 没有解决根本的方向问题。

### bucket010 — Contact 66%, Stability 100%, MPKPE 1.2cm

| 帧 | ref | sim | 观察 |
|------|-----|-----|------|
| t=0.8s | 弯腰手搭桶顶 | 弯腰手伸向桶 | 手在桶附近 |
| t=1.7s | 站着手扶桶面 | 站着手在桶面 | 手贴近桶面 |
| t=2.5s | 站着手搭桶顶 | 站着手在桶旁 | 距离略大 |
| t=3.3s | 站着手在桶侧 | 站着手在桶旁 | 可接受 |

bucket010 表现相对较好, 但也存在手背朝物体的帧。

### desk005 — Contact 4%, Stability 81%

| 帧 | ref | sim | 观察 |
|------|-----|-----|------|
| t=0.8s | 站桌旁 | 站桌旁 | 匹配, 手未接触 |
| t=1.7s | 走向桌边 | **摔倒!** 脚离地 | stability 问题 |
| t=2.5s | 站桌旁推桌 | 恢复站立, 手在桌面附近 | 从摔倒中恢复 |

desk005 stability 从 E039b 的 20% 提升到 81% (动态 target 不会持续强拉), 但仍有不稳定段。

## Claims 验证

| Claim | 阈值 | 结果 | 状态 |
|-------|------|------|------|
| C1: Contact<10cm ≥70% (box025) | ≥70% | 64% | ❌ 未达 |
| C1: Contact<10cm ≥70% (bucket010) | ≥70% | 66% | ❌ 未达 |
| C2: MPKPE <3cm | <3cm | 1.2-1.9cm | ✅ 达成 |
| C3: Stability >90% | >90% | 81-100% | ⚠️ desk005 未达 |
| C4: 无手粘连/反关节 | 视频验证 | **❌ 手背接触仍存在** | ❌ 未达 |

## 关键发现

### 1. ❌ 动态 target 未解决手背接触问题

**根因**: position-only reward (仅约束点-到-点距离) **缺少方向约束**。

```
reward 只关心: ‖contact_point - target‖₂ 小
不关心: 手掌是否朝向物体表面
```

CEM 发现: 旋转手腕让 `wrist + [0.05, 0, 0]` 的点靠近 target, 而不管手掌朝向 → 手背接触。

动态 target 只是让"粘"的位置随时间变化 (不再固定在一个点), 但 **手掌方向错误** 的根因没有触及。

### 2. 与 HDMI 的根本差异

| | HDMI | SPIDER CEM |
|---|---|---|
| 优化方法 | RL policy (学习自然行为) | CEM (只优化距离) |
| 手部方向 | Policy 自然学会手掌朝向物体 | CEM 无方向 inductive bias |
| 解空间 | Policy 约束为自然动作分布 | CEM 在关节空间自由搜索 |

**HDMI 不需要显式方向约束**, 因为 RL 训练中 policy 自然趋向合理行为。CEM 没有这种归纳偏置, 必须**显式添加 orientation reward**。

### 3. Contact Preservation 指标高但不可信

之前认为 Contact Preservation 89-95% 是好结果, 但实际上这些 "接触" 帧中很多是 **手背接触** — 指标数字好看, 行为不合理。纯距离指标无法反映接触质量。

### 4. E040 vs E039b: 都有不自然问题, 只是表现形式不同

| 维度 | E039b (固定target) | E040 (动态target) |
|------|---|---|
| Contact<10cm | **85%/76%** (高) | 64%/66% (中) |
| 不自然行为 | 手粘连固定点 + 手腕反关节 | **手背接触物体** + 不自然弯腰 |
| Stability | 100%/100%/20% | 100%/100%/81% |
| MPKPE | 1.1-2.1cm | 1.2-1.9cm |
| 结论 | ❌ 不可用 | ❌ 不可用 |

**两者都不可用于论文**: E039b 手粘连, E040 手背接触。根本原因相同 — position-only reward 没有手掌方向约束。

## 下一步方向: E041 Hand Orientation Reward

**核心思路**: 添加 orientation 约束, 要求手掌法向量朝向物体表面。

```python
# 手掌法向 (wrist local frame 的某个轴, 如 +x 或 -z)
palm_normal_world = quat_apply(eef_quat, palm_normal_local)

# 物体表面法向 (从 contact_point 指向 target 的方向)
surface_normal = normalize(target_world - contact_point)

# Orientation reward: dot product 越大越好 (手掌朝向物体)
ori_rew = dot(palm_normal_world, surface_normal)
# 或: ori_rew = exp(-(1 - dot) / sigma)
```

**需要确定**:
1. G1 wrist frame 中哪个轴代表 "手掌朝向" (需要可视化确认)
2. 用 dot product reward 还是 cos similarity
3. orientation gain 相对 position gain 的权重

## 改动文件

| 文件 | 改动 |
|------|------|
| `spider/config.py` | +`contact_hdmi_dynamic_target: bool = False` |
| `examples/run_mjwp.py` | +E040 per-frame target 预计算 (~20行) + ref_data 9 元组 |
| `spider/simulators/mjwp.py` | get_reward 支持 9-tuple + 动态 target 分支 |
| `examples/config/override/core4d_e040.yaml` | E040 配置 |

## 结果路径

| 产出 | 路径 |
|------|------|
| box025 | `workspace/core4d/results/E040/E040_box025.{npz,mp4}` |
| bucket010 | `workspace/core4d/results/E040/E040_bucket010.{npz,mp4}` |
| desk005 | `workspace/core4d/results/E040/E040_desk005.{npz,mp4}` |
| 计划 | `workspace/core4d/plan/49_E040_dynamic_target_plan.md` |
| 配置 | `examples/config/override/core4d_e040.yaml` |
