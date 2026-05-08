# E037: Contact Mask-Gated Reward — HDMI-Style Proximity

## 状态: Contact 恢复失败 — gain=2.0 信号相对 tracking(3.5) 太弱，即使修复 baseline 也无效

## 背景

E036 MPKPE=1.4cm (突破性 body tracking)，但 Contact<10cm 暴跌至 7% (desk005)。E037 目标：用 HDMI 风格 contact mask 门控 reward 恢复 contact。

## 核心改动

| 改动 | E036 | E037 |
|------|------|------|
| contact_mask_rew_scale | 0.0 | **2.0** (新增) |
| contact_mask_rew_sigma | — | **0.3m** (HDMI default) |
| hand_approach_contact_threshold | 100.0 | **0.3m** (真正 mask 门控) |

代码新增 `contact_mask_rew` (~20行) in mjwp.py，复用 hand_approach 的 box-SDF 距离计算。

## 评估结果

### E037 vs E036 对比

| 指标 | E036 desk005 | **E037 desk005** | E036 box025 | **E037 box025** | E036 bucket010 | **E037 bucket010** |
|------|-------------|------------------|-------------|-----------------|----------------|-------------------|
| **MPKPE (cm)** | 1.36 | **1.38** | 1.43 | **1.31** | 1.28 | **1.40** |
| **Joint (deg)** | 0.72 | **0.70** | 0.85 | **0.78** | 0.72 | **0.80** |
| **Root Pos (cm)** | 1.06 | **1.04** | 1.06 | — | 1.06 | — |
| **Obj Pos (cm)** | 1.02 | **0.98** | 0.87 | **0.79** | 0.80 | **0.87** |
| **Stability >0.60m** | 100% | **97.4%** | 100% | **100%** | 100% | **100%** |
| **Stability >0.70m** | 79.3% | **84.5%** | 99.2% | **100%** | 100% | **97.6%** |
| **Contact <10cm** | 6.9% | **10.3%** | 55.6% | **51.6%** | 2.4% | **4.8%** |
| **Contact <15cm** | 19.8% | **32.8%** | 79.8% | **88.7%** | 29.6% | **45.6%** |
| **Foot Skating** | 14.3% | **17.8%** | 6.9% | **7.3%** | 30.2% | **13.3%** |
| **Smoothness** | 10.4 | **10.1** | 11.8 | **10.9** | 9.8 | **11.1** |

### Contact 改善幅度

| Case | E036 <10cm | E037 <10cm | E036 <15cm | E037 <15cm | 变化 |
|------|-----------|-----------|-----------|-----------|------|
| desk005 | 6.9% | 10.3% | 19.8% | 32.8% | +3% / +13% |
| box025 | 55.6% | 51.6% | 79.8% | 88.7% | -4% / +9% |
| bucket010 | 2.4% | 4.8% | 29.6% | 45.6% | +2% / +16% |

## Claims 验证

| Claim | 预期 | 实际 | 结论 |
|-------|------|------|------|
| **C1**: Contact<10cm ≥ 50% (desk005) | ≥50% | **10.3%** | ❌ **未达成** — 仅 +3% 微改善 |
| **C2**: MPKPE < 5cm | <5cm | **1.3-1.4cm** | ✅ 保持不变 |
| **C3**: Stability >0.60m ≥ 90% | ≥90% | **97-100%** | ✅ 达成 |

## 根因分析: 为什么 contact_mask_rew 无效?

### 1. reward 结构导致 CEM 无法从 contact reward 获益

```
contact_mask_rew = mask * gain * exp(-dist/sigma) + (1 - mask) * gain
```

- **非接触帧 (mask=0)**: reward = gain = 2.0 (constant)
- **接触帧 (mask=1)**: reward = 2.0 * exp(-dist/0.3)
  - dist=0cm → reward = 2.0 (最大值 = non-contact baseline!)
  - dist=10cm → reward = 2.0 * exp(-0.33) = 1.44
  - dist=30cm → reward = 2.0 * exp(-1.0) = 0.74

**问题**: 接触帧的最大 reward (2.0) = 非接触帧的 baseline (2.0)。CEM 在接触帧只能获得 **≤ baseline** 的 reward → 没有正向激励让手靠近物体！CEM 只看到"接触帧 reward 更低" → 实际上是在**惩罚** contact 帧（如果手不够近）。

### 2. HDMI 用 1.0 作为 baseline 而非 gain

HDMI 的公式：`rew = in_range * gain * proximity + (1 - in_range) * 1.0`
- 非接触帧: reward = 1.0
- 接触帧: reward = [0, gain] = [0, 5.0] >> baseline

**HDMI 的接触帧 max (5.0) >> non-contact baseline (1.0)** → CEM 有强激励在接触帧把手伸近。

### 3. 修正方案

应该用 HDMI 的原始 baseline 设计：
```python
contact_mask_rew = mask * gain * exp(-dist/sigma) + (1 - mask) * 1.0
```
这样：
- 非接触帧: reward = 1.0 (constant)
- 接触帧: reward = [0, gain] = [0, 2.0] >> 1.0 baseline

或者更直接：
```python
# 只在接触帧给 reward，非接触帧给 0
contact_mask_rew = mask * gain * exp(-dist/sigma)
```

## 正面发现

1. **Body tracking 完全未退化**: MPKPE 保持 1.3-1.4cm，contact_mask_rew 的加入没有干扰主目标
2. **Stability 轻微退化但可接受**: desk005 100%→97.4% (pelvis_z min=0.585m，略低于 E036 的 0.656m)
3. **<15cm 有一定改善**: desk005 +13%，bucket010 +16% — 说明 reward 确实产生了信号，但太弱
4. **Foot skating 改善**: bucket010 30%→13%，可能因为 contact reward 让 CEM 选择了更平稳的步态

## 下一步 (E037b 或 E038)

### 方向 A: 修复 baseline 设计 (推荐)
```python
# 非接触帧 baseline=0 (无贡献), 接触帧 max=gain
contact_mask_rew = mask * gain * exp(-dist/sigma)
```
预期：接触帧 CEM 有清晰的正向激励 (max=2.0)，非接触帧 reward=0 (无梯度)。

### 方向 B: 大幅提高 gain
gain=2.0 时 max reward 等于 baseline → 无效。如果 baseline 不改，需要 gain >> baseline 才有激励。但这又会回到 E035 的问题 (contact > tracking)。

### 方向 C: 使用 per-hand reward (而非 min over hands)
当前取 min_dist (best hand)。如果两只手都需要接近物体，应该用 mean_dist 或 per-hand reward。

**推荐**: 方向 A — 最小改动，修正 reward 设计缺陷。

## E037b: Baseline 修复结果 — 仍然无效

修复 formula 为 `contact_mask_rew = mask * gain * exp(-dist/sigma)` (non-contact=0)。

### E037b 结果

| Case | MPKPE | Stability>0.6 | Contact<10cm | Contact<15cm |
|------|-------|---------------|-------------|-------------|
| desk005 | 1.3cm | 100% | **6.9%** | 16.4% |
| box025 | 1.4cm | 100% | **49.2%** | 79.0% |
| bucket010 | 1.3cm | 100% | **3.2%** | — |

### 对比 E036→E037→E037b

| Case/Metric | E036 | E037 | E037b |
|-------------|------|------|-------|
| desk005 <10cm | 6.9% | 10.3% | 6.9% |
| box025 <10cm | 55.6% | 51.6% | 49.2% |
| bucket010 <10cm | 2.4% | 4.8% | 3.2% |

**结论**: baseline 修复无效。E037b 甚至略差于 E037 (constant baseline 版本在接触帧提供了一个"惩罚"让 CEM 至少不远离物体)。

### 真正的根因

**gain=2.0 在 sigma=0.3m 下信号太弱**：
- 手距物体 surface 20cm 时: reward = 2.0 * exp(-0.2/0.3) = 2.0 * 0.51 = **1.03**
- 手距物体 surface 30cm 时: reward = 2.0 * exp(-0.3/0.3) = 2.0 * 0.37 = **0.74**
- Body tracking 每帧 max: **3.5**

CEM 在 horizon (24 steps) 内累加 reward。contact mask 只在 ~20-40% 帧激活 → contact 对总 reward 的贡献 ≈ 0.7 * 0.3 * 24 = **5**。而 tracking 贡献 ≈ 3.5 * 24 = **84**。Contact 只占总 reward 的 ~6% → CEM 完全忽略。

### 下一步方向 (E038)

1. **大幅提高 gain 到 3.5** (与 tracking 平齐) + **降低 sigma 到 0.15m** (更陡峭)
2. 或者: **换策略** — 不用 CEM reward 来驱动 contact，而是用 contact_guidance 的 PD 控制器直接驱动手到物体上。类似 object 的 PD override，对 hand 也做 PD tracking。
3. 或者: **在 tracking reward 内部偏置** — 提高 hand/wrist 在 local_frame_rew 中的权重，利用 body tracking 本身把手送到 ref 位置（ref 中手在物体上 → tracking 好的话手自然在物体上）。

## 配置

```yaml
# examples/config/override/core4d_e037.yaml (完整见文件)
contact_mask_rew_scale: 2.0
contact_mask_rew_sigma: 0.3
hand_approach_contact_threshold: 0.3  # 真正 mask (not 100m)
```

## 结果路径

| 产出 | 路径 |
|------|------|
| desk005 | `workspace/core4d/results/E037/E037_desk005.{npz,mp4}` |
| box025 | `workspace/core4d/results/E037/E037_box025.{npz,mp4}` |
| bucket010 | `workspace/core4d/results/E037/E037_bucket010.{npz,mp4}` |
| 配置 | `examples/config/override/core4d_e037.yaml` |
| 代码改动 | `spider/simulators/mjwp.py`, `spider/config.py` |
