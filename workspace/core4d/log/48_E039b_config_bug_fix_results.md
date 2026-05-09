# E039b: Config Bug Fix — Contact Reward 首次真正生效

## 状态: 突破性发现 — E036-E039 所有实验的 contact reward 从未执行过!

## Bug 描述

### 根因

`spider/config.py:651` 中 `hand_approach_body_ids` 和 `hand_approach_obj_half_extents` 的解析条件:
```python
# BUG: 只在 hand_approach_rew_scale > 0 时解析
if config.hand_approach_rew_scale > 0.0 and config.simulator == "mjwp":
    resolved_ids = [...]
    config.hand_approach_body_ids = resolved_ids
    config.hand_approach_obj_half_extents = [...]
```

从 E036 开始 `hand_approach_rew_scale=0.0` (关闭旧 reward) → `body_ids=[]` → E037 的 `contact_mask_rew` 和 E039 的 `contact_hdmi_rew` 的执行条件 `if config.hand_approach_body_ids:` 永远为 False → **contact reward 从未执行过!**

### 影响范围

| 实验 | contact_mask_rew | contact_hdmi_rew | 实际执行? |
|------|-----------------|-----------------|----------|
| E037/E037b/E037c sweep | 有配置 | — | ❌ 从未执行 |
| E038 (physics_dt) | 有配置 | — | ❌ 从未执行 |
| E039 (HDMI-aligned) | — | 有配置 | ❌ 从未执行 |
| **E039b (修复后)** | — | 有配置 | **✅ 执行** |

### 修复

```python
# FIX: 扩展解析条件
if (config.hand_approach_rew_scale > 0.0 or config.contact_mask_rew_scale > 0.0 
    or config.contact_hdmi_gain > 0.0) and config.simulator == "mjwp":
```

同时添加 E039b rotated-SDF mask 预计算 (替代旧的 axis-aligned mask):
```python
# run_mjwp.py: 正确的 rotated SDF mask
local = obj_mat.T @ (hand_pos - obj_pos)   # 旋转到物体坐标系
clamped = np.clip(local, -half_ext, half_ext)
surf_dist = np.linalg.norm(local - clamped)  # 正确的 surface distance
```

## E039b 结果 (修复后首次正确运行)

| Case | E036 (无 contact) | E039 (bug, 未执行) | **E039b (修复)** | Ref 上限 | Stability |
|------|------|------|------|------|------|
| **box025** | 56% | 52% | **85%** ✅ | 69% | **100%** |
| **desk005** | 7% | 8% | **81%** ✅ | 0% | **20%** ❌ |
| **bucket010** | 2.4% | 13% | **76%** ✅ | 76% | **100%** |

### Mask 激活率 (修复后)

| Case | E034 axis-aligned mask | E039b rotated-SDF mask |
|------|----------------------|----------------------|
| box025 | 100% | 100% |
| desk005 | 100% | 100% |
| bucket010 | 99% | 95% |

## 可视化分析

### box025 — Contact 85%, Stability 100%, MPKPE 1.1cm ✅

| 帧 | ref | sim | 观察 |
|------|-----|-----|------|
| t=1.7s | 站箱旁手扶箱面 | **弯腰手紧贴箱面** | sim 主动趴向箱子, 手贴面 |
| t=2.5s | 弯腰趴箱 | **弯腰手搭箱顶** | 非常接近! |

sim 全程手贴在箱面上, 且 body tracking 和 stability 均保持。**最佳结果 — 超越 ref 水平!**

### bucket010 — Contact 76%, Stability 100%, MPKPE 1.5cm ✅

| 帧 | ref | sim | 观察 |
|------|-----|-----|------|
| t=1.7s | 站着手扶桶 | **站着手贴桶面** | 手紧贴桶壁 |
| t=2.5s | 弯腰扶桶 | **弯腰手在桶顶** | 接触良好 |

sim 手全程贴桶, **完美恢复到 ref 水平** (76% = 76%)。

### desk005 — Contact 81%, **Stability 20%** ❌

| 帧 | ref | sim | 观察 |
|------|-----|-----|------|
| t=1.7s | 站着推桌 | **完全摔倒!** 四脚朝天 | gain=5.0 把手拉向桌下横梁, 导致前倾摔倒 |

**问题**: desk005 的 contact target 在桌面下方横梁 (z=0.35, y=0.15), 手当前在桌面上方 (z=0.53)。gain=5.0 要把手拉下去 ~20cm — 这个"向下"的力矩导致机器人前倾摔倒。
**解决**: desk005 需要降低 gain (如 2.0-3.0) 或调整 target 位置。

## 关键教训

### 1. Config 解析 bug 的严重性

一个 `if` 条件错误导致 **整个 E036-E039 实验系列 (~2周工作)** 的 contact reward 从未生效。之前所有关于"CEM 算法瓶颈"的结论都是错误的 — 实际上是 reward 根本没执行。

### 2. 之前的"算法瓶颈"结论需要全面修正

| 旧结论 | 修正 |
|--------|------|
| "gain=2.0 太弱, CEM 忽略" | 错: reward 没执行, 不是 gain 的问题 |
| "Contact 无法超越 ref" | 错: box025 达到 85% 超越 ref 69% |
| "CEM 优化能力有限" | 待验证: 修复后 bucket010=76% 达到 ref 水平 |
| "physics_dt=0.002 无益" | 待验证: E038 时 reward 也没执行 |
| "需要转向 RL" | 待定: 先用正确的 reward 充分实验 |

### 3. Rotated SDF mask 很重要

旧的 axis-aligned mask 和新的 rotated-SDF mask 在 box025/desk005 上差异不大 (都是 100%)。但在 bucket010 上 rotated 版本更准确 (95% vs 99%) — 对于有旋转的物体差异会更大。

## 下一步方向

### 优先: 修复 desk005 的 stability 问题

1. **降低 gain**: desk005 用 gain=3.0 或 3.5 (< tracking max), 避免摔倒
2. **调整 target**: desk005 target (桌面下横梁) 距手 20cm, 可能太远 → 用更近的中间点
3. **Per-case gain tuning**: box025 和 bucket010 用 gain=5.0 OK, desk005 需要单独调

### 后续: 全面重新评估

1. **E037 系列用修复后代码重跑** — 验证 box-SDF contact_mask_rew 是否也能 work
2. **调参 sweep**: gain 在 [2, 3, 3.5, 5] × 三个 case
3. **论文级评估**: 修复后的最佳配置 vs HDMI R013 vs DynaRetarget 做正式对比

## 改动文件

| 文件 | 改动 |
|------|------|
| `spider/config.py:651` | 解析条件扩展: `+contact_mask_rew_scale>0 \|\| contact_hdmi_gain>0` |
| `examples/run_mjwp.py:524-549` | +E039b rotated-SDF per-EEF mask 预计算 |

## 结果路径

| 产出 | 路径 |
|------|------|
| box025 | `workspace/core4d/results/E039b/E039b_box025.{npz,mp4}` |
| desk005 | `workspace/core4d/results/E039b/E039b_desk005.{npz,mp4}` |
| bucket010 | `workspace/core4d/results/E039b/E039b_bucket010.{npz,mp4}` |
| 运行日志 | `logs/E039b/` |

---

## E039c: threshold=0.15m + 问题分析

### 结果

降低 mask threshold 从 0.30m 到 0.15m, 让非接触帧正确关闭:

| Case | Mask active | MPKPE | Stability | Contact<10cm |
|------|-----------|-------|-----------|-------------|
| box025 | 64% | 1.4cm | 100% | **82%** |
| bucket010 | 67% | 1.3cm | 100% | **68%** |
| desk005 | 72% | 2.1cm | **17%** ❌ | 91% |

### 新发现: Mask 激活率过高的原因

threshold=0.30m 时 mask 接近 100% 激活, 因为这些 ref 序列中人始终站在物体旁边, 手到物体表面距离全程 <30cm (包括"非接触"段)。实际距离分布:
- box025: max=18.9cm (全程 <30cm → 100% active)
- desk005: max=20.8cm (全程 <30cm → 100% active)
- bucket010: max=44.4cm (仅首帧 >30cm → 94-99% active)

threshold=0.15m 合理过滤了 30-40% 的非接触帧。

### 新问题: box025 "手粘连物体" 现象 (严重)

视频分析 (t≈1.5-2.5s) 发现不自然行为:

**表现**:
1. **手腕反关节**: sim 手被 contact reward 强行拉向箱面固定 target, 手腕超过关节限位, 出现反关节扭曲
2. **手粘连物体**: body tracking 要求身体移动 (从箱子一侧走到另一侧) 时, contact reward 仍拉手到固定 target → 手"粘"在箱面不松
3. **身体后仰扭转**: 身体跟 ref 走, 但手被粘住 → 手臂过度伸展 → 身体被迫向后扭转

对比 E036 (无 contact reward) 同时刻, E036 虽然手距物体远但姿态自然无扭曲。

**根因: 固定 contact_target_offset 不适合 CORE4D 任务**

| | HDMI (move_suitcase) | CORE4D (box025/bucket010) |
|---|---|---|
| 手在物体上的位置 | **固定** (把手位置不变) | **随时变化** (人围着物体活动) |
| 固定 offset 合理性 | ✅ 把手是固定的 | ❌ 手在箱面滑动/切换位置 |
| 人体运动模式 | 抓住把手行走 | 推/搬/抱, 手持续调整位置 |

**HDMI 的 contact_target_offset 假设 "手始终在物体同一个点" — 对 suitcase 把手成立, 对 CORE4D 自由交互不成立。**

当 ref 中机器人从箱子前方移动到侧方时, 手在箱面上的实际接触点从"前面中部"变到"侧面上方"。但固定 target 始终指向"前面中部" → contact reward 强行拉手回去 → 手腕反关节。

### 下一步方向 (E040)

**方案 1: 动态 per-frame target (推荐)**

每帧从 ref FK 中提取手相对物体的实际位置 (in obj local frame) 作为当帧的 target:
```python
# 预计算: (T, 2, 3) — per-frame, per-EEF target offset
for t in range(T):
    target_offset[t, ei] = rot_inv(obj_quat[t]) @ (hand_pos_ref[t] - obj_pos[t])
```
这等价于 "ref 中手在物体上的位置" → sim 的 contact reward 引导手去 ref 手所在的物体表面位置。

优点: 完全避免固定 offset 的粘连问题 (target 随 ref 动态变化)
本质: 变成了 "在物体坐标系下的 EEF tracking"

**方案 2: 降低 gain + 关节限位惩罚**

- gain 从 5.0 降到 2.0-3.0 (与 tracking 竞争更弱)
- 添加 joint limit penalty 防止反关节

**方案 3: 重新定义 contact metric**

当前 MPKPE=1.4cm 已很好, contact<10cm 的提升可能不需要额外 reward:
- 如果用 <15cm 作为 contact 标准, E036 已有 80% → 足够
- 如果用动态 target (方案 1), 本质上就是提高 hand tracking 精度

