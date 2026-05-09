# E041: Hand Orientation Reward — 手掌方向约束 + Sweep

## 状态: additive 模式(E041c)是最佳变体, 但未根本解决不自然姿态 (CEM 结构性限制)

## 背景

E039b~E040 的 contact reward 存在 "手背接触物体" 的不自然行为。根因: position-only reward 没有方向约束, CEM 通过旋转手腕让 contact_point 靠近 target, 导致手背 (而非手掌) 朝向物体。

E041 在 contact reward 中添加 orientation 约束: 要求手掌法向量朝向物体表面。

## Palm Normal 分析

通过分析 box025 ref 中 G1 wrist 在多个接触帧 (t=30~90) 的旋转矩阵与 wrist→object 方向的点积:

```
Left wrist:  -y 轴一致指向物体 (dot = -0.61 ~ -0.87)
Right wrist: +y 轴一致指向物体 (dot = +0.67 ~ +0.96)
```

结论: palm_normal_left = `[0, -1, 0]`, palm_normal_right = `[0, +1, 0]`

## 三种 Orientation 模式

```python
# Mode 1: multiply (E041, E041b) — 严格乘法门控
pos_rew = pos_rew * clamp(dot(palm_world, dir_to_target), min=0)

# Mode 2: additive (E041c) — 柔性加权
pos_rew = (1 - w) * pos_rew + w * ori_rew   # w=0.3

# Mode 3: near_field (E041d) — 仅近距离时约束方向
near = (dist < 0.15).float()
pos_rew = pos_rew * (1 - near + near * ori_rew)
```

## Sweep 完整结果 (本地 + 远程 2-GPU 并行)

### box025

| 变体 | Mode | Gain | Contact<10cm | Stability | MPKPE | Preservation | ObjPos |
|------|------|------|------|------|------|------|------|
| E036 (no contact) | - | - | 56% | 100% | 1.4cm | - | 0.9cm |
| E040 (pos-only) | - | 5.0 | 64% | 100% | 1.3cm | 88.8% | 0.7cm |
| E041 (multiply) | multiply | 5.0 | 62% | 100% | 1.6cm | 86.5% | 0.8cm |
| E041b (multiply) | multiply | 3.0 | 52% | **85%** ❌ | 1.5cm | 74.4% | 0.7cm |
| **E041c (additive)** | additive(0.3) | 5.0 | **66%** | **100%** | **1.4cm** | **88.2%** | 0.8cm |
| E041d (near_field) | near_field | 5.0 | 58% | 98% | 1.5cm | 73.5% | 0.9cm |

### bucket010

| 变体 | Mode | Gain | Contact<10cm | Stability | MPKPE | Preservation | ObjPos |
|------|------|------|------|------|------|------|------|
| E036 (no contact) | - | - | 2% | 100% | 1.3cm | - | 0.8cm |
| E040 (pos-only) | - | 5.0 | **66%** | 100% | 1.2cm | 95.4% | 0.9cm |
| E041 (multiply) | multiply | 5.0 | 56% | **90%** | 1.8cm | 80.5% | 1.0cm |
| E041b (multiply) | multiply | 3.0 | 40% | **90%** | 1.7cm | 58.1% | 0.9cm |
| **E041c (additive)** | additive(0.3) | 5.0 | 57% | **100%** | **1.4cm** | **81.6%** | 0.9cm |

## 可视化分析 (t=2.0s 关键帧对比, box025)

t=2.0s 是 E040 中手背接触最明显的帧 (用户指出), 以下逐变体对比:

**ref**: 站在箱侧, 手自然搭在箱面

### E040 (position-only, 无 ori) — Contact 64%

sim: 弯腰前倾, 手在箱顶, **手背朝向物体**。手腕反转, 不是搬运的姿态。

### E041 (multiply, gain=5) — Contact 62%

sim: 弯腰头趴箱顶, 手在箱面上。前倾程度与 E040 类似, 手掌方向不太明确 (视角不清晰)。乘法门控使 CEM 难以同时满足 position + orientation → MPKPE 退化到 1.6cm。

### E041b (multiply, gain=3) — Contact 52%

sim: 站姿相对直立, 手在箱面附近但距离较远。gain 太低, 接触激励弱 → Contact 大幅退化 (52%), Stability 也降到 85%。**最差变体**。

### E041c (additive, w=0.3) — Contact 66% ★最佳

sim: 弯腰但程度比 E040/E041 轻, 手在箱面附近。additive 模式不会完全 zero-out reward → CEM 保持接近行为, 同时 orientation 提供软性方向信号。MPKPE 恢复到 1.4cm (与 E036 baseline 一致)。**仍有前倾但程度减轻**。

### E041d (near_field) — Contact 58%

sim: 站姿较直, 手伸向箱面。只在 dist<0.15m 时才激活 ori 约束 → 远距离正常靠近, 近距离方向校正。但总 Contact 偏低 (58%), 可能是近距离校正使 CEM 在最后一步"犹豫"。

### E041c bucket010 可视化

| 帧 | ref | sim | 观察 |
|------|-----|-----|------|
| t=0.8s | 弯腰手搭桶顶 | 弯腰手伸向桶, 步幅较大 | 手伸向桶的方向大致正确 |
| t=1.7s | 站着手扶桶壁 | 弯腰手在桶面, 手搭在桶壁上 | 手掌大致朝向桶面 |
| t=2.5s | 站着手搭桶顶 | 站着手在桶旁 | 姿态大致匹配, 距离略远 |

bucket010 的 E041c 姿态整体合理, 但 Contact 从 E040 的 66% 降到 57% — orientation 约束增加了 CEM 负担。

## Claims 验证 (以 E041c 最佳变体为准)

| Claim | 阈值 | E041c 结果 | 状态 |
|-------|------|------|------|
| C1: Contact<10cm ≥60% | ≥60% | 66%/57% | ⚠️ bucket010 未达 |
| C2: MPKPE <3cm | <3cm | 1.4cm | ✅ 达成 |
| C3: Stability >90% | >90% | 100%/100% | ✅ 达成 |
| C4: 手掌朝向改善 | 视频验证 | 前倾减轻但仍有不自然帧 | ⚠️ 部分达成 |

## 关键分析

### 1. 各模式的效果排序

**additive > near_field > multiply > multiply+低gain**

| 模式 | 原理 | 对 CEM 的压力 | 效果 |
|------|------|-------------|------|
| additive(0.3) | `0.7*pos + 0.3*ori` | 低 (ori 只占 30%) | ★★★ 保持 Contact, 软性引导方向 |
| near_field | dist<15cm 时 `pos*ori` | 中 (远距离无压力) | ★★ Stability 好, Contact 偏低 |
| multiply | `pos * ori` | 高 (必须同时满足) | ★ Contact/Stability 退化 |
| multiply + gain=3 | `pos * ori`, gain 减半 | 高 + 激励弱 | ✗ 全面退化 |

### 2. CEM 优化预算分析

CEM 每步有 1024 samples × 32 iterations 的预算来优化 reward:

| Reward component | Max | 维度 |
|------|------|------|
| local_frame_rew (body tracking) | 3.5 | 7 terms (upper/lower/root × pos/ori + joint) |
| task_obj_rew | penalty | 2 terms (pos + rot) |
| contact_hdmi_rew (E041c) | 5.0 | **2 terms × (position + orientation)** = 4 维 |

总共 ~13 个约束维度。CEM 用 1024 个高斯采样来搜索 nu 维关节空间 (nu≈20+), 同时满足 13 个约束。**这超出了 CEM 的有效搜索能力**。

### 3. 根本性结论 (重要修正!)

**之前认为 "HDMI 用 RL, SPIDER 用 CEM → CEM 能力不足" — 这是错误的!**

核实 `examples/run_hdmi.py` 发现: HDMI workflow 在 SPIDER 中**也是 CEM** (sampling-based MPC), 使用完全相同的 `make_optimize_fn`。且 HDMI 的 `rew_contact` (hdmi.py:1108) 也是 **position-only** (无 orientation 约束):

```python
# hdmi.py:1108-1114 — HDMI contact reward, 纯 position, 无 orientation!
eef_dist = (target_pos - contact_eef).norm(dim=-1)
pos_rew = torch.exp(-eef_dist / rc["eef_pos_sigma"])
contact_rew = pos_rew * force_factor
```

但 HDMI 在 `move_suitcase` 任务上能实现自然手掌接触。**差异不在 CEM 能力, 而在任务特性**:

| | HDMI move_suitcase | CORE4D box025 |
|---|---|---|
| 接触目标 | 固定把手位置 (手自然抓握) | 平坦表面 (手掌/手背都能贴) |
| 手腕自由度 | **零噪声** (run_hdmi.py:112-118 wrist noise=0) | 正常 CEM 采样 |
| 物体形状 | 箱子有凸出把手 → 几何引导手掌方向 | 平面 → 无几何引导 |
| ref 动作 | 人抓把手行走 → 手腕姿态固定 | 人推/搬 → 手腕姿态自由变化 |

**HDMI 的 "秘密" 不是 RL, 而是 (1) wrist 零噪声 + (2) 固定把手的几何引导**。

→ **下一步 E042: 在 CORE4D 中冻结 wrist 关节噪声** (和 HDMI run_hdmi.py 一样), 让手腕保持 ref 姿态不被 CEM 随机扰动, 可能直接解决手背接触问题。

### 4. E041c vs E040 的实际改善

| 维度 | E040 | E041c | 改善 |
|------|------|-------|------|
| box025 Contact<10cm | 64% | **66%** | +2% |
| box025 MPKPE | 1.3cm | **1.4cm** | ≈ |
| box025 Preservation | 88.8% | **88.2%** | ≈ |
| bucket010 Contact<10cm | **66%** | 57% | -9% |
| bucket010 MPKPE | 1.2cm | **1.4cm** | -0.2cm |
| Stability | 100%/100% | 100%/100% | = |

**结论**: E041c 对 box025 有微小改善 (+2%), 对 bucket010 有退化 (-9%)。orientation reward 的边际收益不显著。

## 改动文件

| 文件 | 改动 |
|------|------|
| `spider/config.py` | +`contact_hdmi_ori_weight`, `ori_mode`, `palm_normal_left/right` |
| `spider/simulators/mjwp.py` | contact_hdmi_rew 块 + 3 种 ori 模式 (multiply/additive/near_field) |
| `examples/config/override/core4d_e041{,b,c,d}.yaml` | 4 种变体配置 |
| `workspace/core4d/scripts/run_E041_sweep_remote.sh` | 远程 2-GPU 并行执行脚本 |

## 结果路径

| 产出 | 路径 |
|------|------|
| E041 box025 | `workspace/core4d/results/E041/E041_box025.{npz,mp4}` |
| E041 bucket010 | `workspace/core4d/results/E041/E041_bucket010.{npz,mp4}` |
| E041b box025 | `workspace/core4d/results/E041/E041b_box025.{npz,mp4}` |
| E041b bucket010 | `workspace/core4d/results/E041/E041b_bucket010.{npz,mp4}` |
| E041c box025 | `workspace/core4d/results/E041/E041c_box025.{npz,mp4}` |
| E041c bucket010 | `workspace/core4d/results/E041/E041c_bucket010.{npz,mp4}` |
| E041d box025 | `workspace/core4d/results/E041/E041d_box025.{npz,mp4}` |
| 计划 | `workspace/core4d/plan/50_E041_orientation_reward_plan.md` |
| 远程脚本 | `workspace/core4d/scripts/run_E041_sweep_remote.sh` |
