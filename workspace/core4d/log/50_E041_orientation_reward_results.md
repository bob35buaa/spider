# E041: Hand Orientation Reward — 手掌方向约束

## 状态: 方向部分有效 — additive 模式(E041c)是最佳变体,但未根本解决不自然前倾

## 背景

E040 发现 position-only reward 导致 CEM 用手背接触物体 (满足距离约束但方向错误)。E041 添加 orientation 乘法门控: pos_rew × ori_rew, 要求手掌法向量朝向物体。

## 核心改动

```python
# Palm normal in wrist local frame (从 ref FK 分析得出)
palm_normal_left = [0, -1, 0]   # left wrist -y axis
palm_normal_right = [0, +1, 0]  # right wrist +y axis

# Transform to world, compute alignment with hand→target direction
palm_world = quat_apply(eef_quat, palm_normal_local)
dir_to_target = normalize(target_world - contact_point)
dot = dot_product(palm_world, dir_to_target)
ori_rew = clamp(dot, min=0)  # [0, 1]

# Multiplicative gating
pos_rew = pos_rew * ori_rew  # only reward when both close AND palm faces target
```

## 结果

| Case | E036 (baseline) | E040 (动态target) | **E041 (+ orientation)** | Ref |
|------|------|------|------|------|
| box025 contact<10cm | 56% | 64% | **62%** | 69% |
| bucket010 contact<10cm | 2% | 66% | **56%** | 76% |
| box025 stability | 100% | 100% | **100%** | - |
| bucket010 stability | 100% | 100% | **90%** | - |
| box025 MPKPE | 1.4cm | 1.3cm | **1.6cm** | - |
| bucket010 MPKPE | 1.3cm | 1.2cm | **1.8cm** | - |

## 可视化分析

### box025

| 帧 | ref | sim | 对比 E040 |
|------|-----|-----|----------|
| t=0.8s | 弯腰手搭箱面 | 弯腰手伸向箱面 | 手掌朝向有所改善 |
| t=1.5s | 站箱后手搭面 | 站箱侧弯腰手伸向箱 | 手位置偏但方向似乎更正确 |
| t=2.0s | 站箱侧 | **过度前倾, 头趴箱顶** | ❌ 仍有不自然前倾 |
| t=2.5s | 站箱侧手搭 | 弯腰前倾手在箱面 | 不自然弯腰 |
| t=3.3s | 弯腰趴箱 | 弯腰手在箱面 | 接近 ref |

**问题**: t=2.0-2.5s 仍有过度前倾。orientation reward 改善了手掌方向但没有解决 "身体过度弯曲" 问题 — 这可能是 body tracking 本身的 tradeoff (MPKPE 退化到 1.6cm)。

### bucket010

| 帧 | ref | sim | 对比 E040 |
|------|-----|-----|----------|
| t=0.8s | 弯腰手搭桶顶 | 弯腰手伸向桶顶 | 手掌朝向改善, 更自然地伸手 |
| t=1.7s | 站着手扶桶面 | 站着手在桶面附近 | ✅ 手掌朝向桶面, 比 E040 更自然 |
| t=2.5s | 站着手搭桶顶 | 站着手在桶旁 | 手位置可接受 |

bucket010 的手掌方向比 E040 有明显改善 (t=1.7s 可见), 但 Contact 和 Stability 都退化了。

## Claims 验证

| Claim | 阈值 | 结果 | 状态 |
|-------|------|------|------|
| C1: Contact<10cm ≥60% | ≥60% | 62%/56% | ⚠️ bucket010 未达 |
| C2: MPKPE <3cm | <3cm | 1.6/1.8cm | ✅ 达成 |
| C3: Stability >90% | >90% | 100%/90% | ⚠️ bucket010 边缘 |
| C4: 手掌朝向改善 | 视频验证 | 部分改善, 仍有前倾 | ⚠️ 部分达成 |

## 分析

### 1. Orientation reward 方向正确但 multiplicative gating 过于严格

乘法门控 `pos_rew × ori_rew` 意味着:
- 当 palm 方向错误 (dot ≈ 0): 无论距离多近, reward ≈ 0
- CEM 在 1024 samples 中更难找到 "距离近 AND 方向对" 的解
- 结果: CEM 有时直接放弃接触 (Contact 降低) 或为满足约束做出其他妥协 (Stability 降低)

### 2. "过度前倾" 问题的根因

box025 t=2.0s 的过度前倾不是 orientation reward 导致的 — E040 也有同样问题。这是因为:
- 动态 target 在该帧指向箱顶
- body tracking 要求身体在某个位置
- CEM 的折中: 前倾+伸手 → 同时部分满足 body tracking 和 contact
- 这与 orientation 无关, 是 position reward 引起的身体姿态问题

### 3. 指标对比总结

| 指标 | E036→E040→E041 趋势 | 分析 |
|------|---|---|
| Contact<10cm | 56→64→62 / 2→66→56 | E041 退化, ori gating 太严 |
| Stability | 100→100→100 / 100→100→90 | bucket010 略退化 |
| MPKPE | 1.4→1.3→1.6 / 1.3→1.2→1.8 | E041 body tracking 退化 |
| 手掌方向 | 无约束→无约束→有约束 | E041 有改善但不完全 |

## 下一步方向

1. **降低 ori_weight 或改为加权和**: `0.5 * pos_rew + 0.5 * ori_rew` 替代乘法门控, 更宽容
2. **只在近距离时激活 ori**: 当 dist < 0.15m 时才检查方向, 远距离只看 position
3. **降低 contact gain**: gain=5.0 → 3.0, 让 body tracking 主导, contact 只做轻微引导
4. **根本性思考**: CEM 的 1024 samples × 32 iter 是否足以同时优化 position + orientation? 可能需要接受 CEM 的结构性限制

## 改动文件

| 文件 | 改动 |
|------|------|
| `spider/config.py` | +`contact_hdmi_ori_weight`, `ori_mode`, `palm_normal_left/right` |
| `spider/simulators/mjwp.py` | contact_hdmi_rew 块 + 3 种 ori 模式 (multiply/additive/near_field) |
| `examples/config/override/core4d_e041{,b,c,d}.yaml` | 4 种变体配置 |
| `workspace/core4d/scripts/run_E041_sweep_remote.sh` | 远程并行执行脚本 |

## E041 Sweep 完整结果 (远程 2-GPU 并行)

| 变体 | Mode | Gain | box025 Contact | box025 Stable | box025 MPKPE | bucket010 Contact | bucket010 Stable | bucket010 MPKPE |
|------|------|------|------|------|------|------|------|------|
| E040 (baseline) | 无ori | 5.0 | 64% | 100% | 1.3cm | 66% | 100% | 1.2cm |
| E041 (multiply) | multiply | 5.0 | 62% | 100% | 1.6cm | 56% | 90% | 1.8cm |
| E041b (multiply) | multiply | 3.0 | 52% | 85% | 1.5cm | 40% | 90% | 1.7cm |
| **E041c (additive)** | additive(0.3) | 5.0 | **66%** | **100%** | **1.4cm** | 57% | **100%** | 1.4cm |
| E041d (near_field) | near_field | 5.0 | 58% | 98% | 1.5cm | - | - | - |

### 结论

1. **E041c (additive, w=0.3) 是最佳变体**: Contact=66%/57% + Stability=100%/100% + MPKPE=1.4cm
2. **Multiplicative gating 过于严格** (E041/E041b): CEM 放弃部分接触帧
3. **Near-field** (E041d) 效果中等: 只在近距离才激活 ori → stability 好但 contact 退化
4. **降低 gain** (E041b) 有害: contact 大幅退化 (52%/40%)

### 视觉评估 (E041c box025)

t=0.8s: 手掌方向相比 E040 有所改善, 但 t=2.0s 仍有前倾趴箱的不自然行为。Orientation reward 提供了方向 signal 但不足以完全阻止 CEM 用不自然姿态满足距离约束。

### 根本性结论

**Orientation reward 是正确的方向但对 CEM 效果有限**:
- CEM 的 1024 samples × 32 iterations 优化空间有限
- 同时满足 body tracking (7 terms) + position contact + orientation contact 对 CEM 压力太大
- HDMI 的 RL policy 经数百万步训练自然学会正确行为, CEM 没有这种学习能力
- **这是 CEM 的结构性限制, 不是 reward 设计的问题**

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
