# E043: 原始 OmniRetarget Ref 对比 — Phase 3 (无松弛) vs Phase 4 (松弛)

## 状态: 原始 ref 反而更差 — Phase 4 松弛版的接触质量更高

## 背景

当前 SPIDER 使用的 ref 来自 Phase 4 (约束松弛版 OmniRetarget), 允许少量穿透换取 100% 求解成功率。用户提出: 原始 Phase 3 (无松弛) 的接触更精细, 可能提升 SPIDER 的 contact 质量。

## 数据来源

| 版本 | 路径 | 特点 |
|------|------|------|
| Phase 4 (当前) | `trajectory_kinematic.npz` (124帧, box025) | 约束松弛, Contact Preservation 86.9% |
| Phase 3 (原始) | `*_with_obj_original.npz` | 无松弛, Contact Preservation 78.6% |

使用 `spider/process_datasets/core4d.py` 正确转换原始数据 (mj_differentiatePos + FK contact_pos)。

## 结果 (E040 配置: 动态 target + contact reward)

| Case | E040 (Phase 4 ref) | E043 (Phase 3 原始 ref) | 变化 |
|------|------|------|------|
| box025 Contact<10cm | **64%** | 52% | **-12%** ❌ |
| box025 Preservation | 88.8% | 72.1% | -17% ❌ |
| box025 MPKPE | 1.3cm | 1.5cm | -0.2cm |
| box025 Stability | 100% | 100% | = |
| bucket010 Contact<10cm | **66%** | 57% | **-9%** ❌ |
| bucket010 MPKPE | 1.2cm | 1.2cm | = |
| bucket010 Stability | 100% | 100% | = |
| desk005 Contact<10cm | 4% | 7% | +3% |
| desk005 MPKPE | 1.9cm | **1.0cm** | **+0.9cm** ✅ |
| desk005 Stability | 81% | **100%** | **+19%** ✅ |

## 分析

### 1. box025/bucket010: 原始 ref 接触更差

**意外**: Phase 3 原始版本虽然运动学求解时约束更严 (无穿透松弛), 但 SPIDER 物理重定向后接触反而更差。

可能原因:
- Phase 3 运动学约束过严 → 机器人姿态更僵硬, 手部位置受限
- Phase 4 松弛允许手深入物体表面附近 → 动态 target 更精确指向物体表面
- Phase 3 的 120 帧 vs Phase 4 的 124 帧 → 可能时间对齐不同

### 2. desk005: 原始 ref 表现更好

desk005 是唯一改善的 case: MPKPE 1.9→1.0cm, Stability 81→100%。
- Phase 3 原始版 179 帧 (vs Phase 4 的 116 帧) — 更长的序列
- 原始 ref 的桌子交互动作可能更平稳 (无松弛抖动)

### 3. 结论: Phase 4 ref 整体更优

| 维度 | Phase 4 (松弛) | Phase 3 (原始) |
|------|------|------|
| box025 | **64%** contact | 52% contact |
| bucket010 | **66%** contact | 57% contact |
| desk005 | 4%/81% | **7%/100%** |
| 整体 | **box/bucket 更好** | desk 更好 |

**决定**: 继续使用 Phase 4 ref 作为默认 (box025/bucket010 是主要评估 case)。desk005 可考虑单独用 Phase 3 ref。

## 结果路径

| 产出 | 路径 |
|------|------|
| E043 box025 | `workspace/core4d/results/E043_original_ref/E043_box025.{npz,mp4}` |
| E043 bucket010 | `workspace/core4d/results/E043_original_ref/E043_bucket010.{npz,mp4}` |
| E043 desk005 | `workspace/core4d/results/E043_original_ref/E043_desk005.{npz,mp4}` |
| Phase 4 ref 备份 | `*/trajectory_kinematic_phase4.npz` |
