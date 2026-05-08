# E037c: S3 Best Config — All Cases Validation

## 状态: S3 confirmed — Contact 改善主要限于 box025 (高 contact-ratio case)

## 实验设计

Sweep 选出 S3 (gain=3.5, sigma=0.15, baseline=0) 为最优。
本轮在 3 个 case 上验证泛化性，同时对比 S3b (baseline=1.0 变体)。

## 结果

### S3 (gain=3.5, sigma=0.15, baseline=0.0)

| Case | MPKPE | Joint | ObjPos | Stable>0.6 | Contact<10cm | Contact<15cm |
|------|-------|-------|--------|-----------|-------------|-------------|
| desk005 | 1.4cm | 0.7° | 1.0cm | 100% | **5.2%** | — |
| box025 | 1.4cm | 0.9° | 0.7cm | 100% | **57.3%** | 79.8% |
| bucket010 | 1.3cm | 0.7° | 0.8cm | 100% | **2.4%** | — |

### S3b (gain=3.5, sigma=0.15, baseline=1.0)

| Case | MPKPE | Joint | ObjPos | Stable>0.6 | Contact<10cm | Contact<15cm |
|------|-------|-------|--------|-----------|-------------|-------------|
| desk005 | 1.3cm | 0.7° | 1.0cm | 100% | **5.2%** | — |
| box025 | 1.4cm | 0.8° | 0.7cm | 100% | **41.1%** | — |
| bucket010 | 1.3cm | 0.7° | 0.8cm | 100% | **3.2%** | — |

### 对比 E036 baseline (无 contact reward)

| Case | E036 <10cm | S3 <10cm | S3b <10cm | 变化 (S3 vs E036) |
|------|-----------|----------|-----------|-------------------|
| desk005 | 6.9% | 5.2% | 5.2% | -1.7% (无改善) |
| box025 | 55.6% | **57.3%** | 41.1% | +1.7% (微改善) |
| bucket010 | 2.4% | 2.4% | 3.2% | ±0% (无改善) |

## 结论

### 1. Contact mask reward 对 desk005 和 bucket010 完全无效

这两个 case 的 ref 中手距物体较远 (approach_mask 激活率低):
- desk005: ref 大部分时间在行走, 手偶尔碰桌面
- bucket010: ref 弯腰搬桶, 但桶被 PD 推远了

**根本原因**: contact_mask_rew 只在 approach_mask=1 的帧激活 (ref 手 <30cm)。对于 ref 中手本身就远离物体的帧, 没有任何激励。

### 2. box025 是唯一受益 case

box025 ref 中 ~80% 帧手在物体 15cm 内 → approach_mask 激活率高 → contact reward 有效。
但改善仅 +1.7% (55.6% → 57.3%)，且 S3b (baseline=1.0) 反而退化到 41%。

### 3. S3b (baseline=1.0) 在 box025 上显著差于 S3

| | S3 (b=0) | S3b (b=1) |
|---|---|---|
| box025 <10cm | 57.3% | 41.1% |

baseline=1.0 给非接触帧一个常数 reward，减少了接触帧的"net advantage" → CEM 不会特别偏好把手伸近。

### 4. Body tracking 和 stability 完全不受影响

所有配置 MPKPE=1.3-1.4cm, Stability=100%。contact_mask_rew 的加入是安全的。

## 核心问题总结

**Contact mask reward 的根本局限**: 它只能在 "ref 中手已经在物体附近" 的帧提供额外激励。但:
1. desk005/bucket010 中这样的帧很少
2. 即使有这样的帧, body tracking (MPKPE=1.4cm) 本身已经很好地把手送到了正确位置
3. **contact<10cm 差的原因不是"手不想去那里", 而是 "ref 中手的正确位置本身就距物体表面 >10cm"**

**Box025 之所以有 57% contact<10cm, 不是因为 reward, 而是因为 ref 动作本身就是贴着箱子的** (比较: E036 也有 56%)。

## 下一步方向

Contact improvement 可能需要完全不同的方法:
1. **physics_dt=0.002** (E038 正在跑) — 更精细的物理可能改善接触精度
2. **放弃 contact reward, 接受 body tracking 本身的 contact 质量** — 如果 MPKPE=1.4cm 已经足够好, contact 就是 ref 本身决定的
3. **检查 eval 标准是否合理** — "手距物体表面 <10cm" 是否是正确的指标? 也许应该看 "手位置 vs ref 手位置" 的误差

## 结果路径

| 产出 | 路径 |
|------|------|
| S3 all cases | `workspace/core4d/results/E037c/S3_{desk005,bucket010}.{npz,mp4}` |
| S3b all cases | `workspace/core4d/results/E037c/S3b_{desk005,box025,bucket010}.{npz,mp4}` |
| S3 box025 (from sweep) | `workspace/core4d/results/E037_sweep/S3_gain3.5_sigma0.15_base0.0_box025.npz` |
| Logs | `logs/E037c/` |
