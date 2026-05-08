# E038: Physics_dt=0.002 (HDMI Alignment) — Higher Fidelity Physics Test

## 状态: 无改善 — 4.3x 计算代价换来零收益，physics_dt 不是 contact 瓶颈

## 背景

HDMI simulator 使用 `physics_dt=0.002` + `implicitfast` integrator (decimation=8)。
progress.md 记录 E032 Round3："接触 OK 但 CEM budget 不足以稳定行走"。
E036 已解决稳定性 (local-frame tracking)，本实验验证: 在稳定状态下，physics_dt=0.002 是否改善 contact。

## 配置

```yaml
# examples/config/override/core4d_e038.yaml
physics_dt: 0.0020833375  # HDMI style
sim_dt: 0.0166667         # → decimation = 8
# 其余同 E037 S3 best (gain=3.5, sigma=0.15, baseline=0)
```

## 结果 (box025)

| 指标 | E036 (dt=default) | S3 (dt=default) | **E038 (dt=0.002)** |
|------|-------------------|-----------------|---------------------|
| **MPKPE** | 1.43cm | 1.43cm | **1.32cm** |
| **Joint Err** | 0.85° | 0.85° | **0.82°** |
| **Obj Pos** | 0.87cm | 0.68cm | **0.77cm** |
| **Stability >0.6m** | 100% | 100% | **100%** |
| **Stability >0.7m** | 99.2% | 98.4% | **100%** |
| **Contact <10cm** | 55.6% | 57.3% | **46.0%** ❌ |
| **Contact <15cm** | 79.8% | 79.8% | **80.6%** |
| **Mean Surf Dist** | 12.5cm | 12.3cm | **14.0cm** |
| **Foot Skating** | 6.9% | 7.7% | **7.3%** |
| **Penetration** | 0% | 0% | **0%** |
| **Total Time** | 371s | 370s | **1609s** (4.3x) |

## 可视化分析

| 帧 | 时间 | E038 观察 | vs E036 |
|------|------|-----------|---------|
| 0 (0s) | 起始 | ref/sim 对齐良好 | 一致 |
| 24 (0.8s) | 弯腰 | sim 弯腰趴箱, 手搭箱面 | 类似, sim 手略高于箱面 |
| 49 (1.7s) | 推箱 | sim 站箱旁, 手伸向箱面 | 类似 |
| 74 (2.5s) | 站立 | sim 站箱侧, 手碰箱角 | 类似 |
| 99 (3.3s) | 弯腰 | sim 弯腰趴箱 | 类似 |
| 123 (4.1s) | 结束 | sim 站立 | 类似 |

**视觉上 E038 与 E036 无明显差异。** robot 行为模式完全相同。

## 分析

### 1. Contact 略微退化 (46% vs 56%)

physics_dt=0.002 让物理更精确，但同时让 CEM 的每次 rollout 中 8x substeps 使得 "一步犯错后恢复" 更难。CEM 可能变得更保守 → 手不敢太靠近物体。

### 2. 其他指标几乎不变

MPKPE 1.43→1.32cm (微改善), stability 不变, foot skating 不变。
physics_dt 的精细化对 body tracking 几乎无影响 — 因为 local-frame reward 在 coarse dt 下已经足够好。

### 3. 计算成本不可接受

4.3x slowdown (6min → 27min/case) 换来零改善。不值得。

## 结论

**physics_dt=0.002 对当前 SPIDER+CORE4D pipeline 无益。** 原因:
1. Body tracking 精度已被 local-frame reward 解决 (MPKPE=1.3-1.4cm)
2. Contact 质量由 ref 中手的位置决定，不由物理精度决定
3. CEM 在 coarse dt 下就能找到足够好的解
4. 更精细的物理让 CEM 更保守，反而降低 contact

**建议**: 保持 `physics_dt: 0` (default)，不对齐 HDMI 的 physics_dt。

## 结果路径

| 产出 | 路径 |
|------|------|
| box025 npz | `workspace/core4d/results/E038/E038_box025.npz` |
| box025 video | `workspace/core4d/results/E038/E038_box025_video.mp4` |
| 配置 | `examples/config/override/core4d_e038.yaml` |
| 运行日志 | `logs/E038/box025_test.log` |
