# E033: desk005 稳定性优化 — 解决 79% stable 瓶颈

## Context

E032a 在 desk005 上取得了 83% contact<10cm（从 E027d2 的 9% 大幅提升），但稳定性只有 79%（base_pos=5）。增加 base_pos=10 反而降到 75%。

**关键事实**:
- Ref 搬运期手在桌面 12-13cm 外（<15cm=100%, <10cm=0%）
- E032a 已让手比 ref 更贴近物体 → 接触"过度达标"
- 不稳定发生在 ~40% 时刻（约 1.9s），机器人前倾但未完全摔倒
- 行走速度 0.60 m/s，关节 range max=0.79rad（正常步态）

**不稳定根因假设**:
1. hand_approach(σ=5) 的指数衰减太陡 → 手离物体 10cm 内时梯度爆发 → 上半身突然前倾
2. CEM 32iter/1024samp 对 0.6m/s 步态切换优化不够（步态 transition 需要更多样本）
3. horizon=0.8s 只看到 ~1.5 步 → 步态规划不够前瞻

## Claims

1. **降低 hand_approach_sigma (5→2)** 可以减少重心突变，提升 stable 到 ≥90%
2. **增加 num_samples (1024→2048)** 或 **max_iterations (32→48)** 可以改善步态稳定性
3. **增加 horizon (0.8→1.2s)** 让 CEM 看到更多步 → 更好的步态规划
4. 以上调整后 contact<15cm（ref 标准）应保持 ≥80%

## 成功标准

| 指标 | E032a baseline | 目标 |
|------|---------------|------|
| stable (pelvis>0.55) | 79% | **≥92%** |
| contact<15cm (ref 标准) | 86% | ≥80% (不退步) |
| contact<10cm | 83% | ≥60% (可适度降低) |
| contact<5cm | 58% | 报告 |
| contact<3cm | ? | 报告 |
| contact<1cm | ? | 报告 |

## 实验矩阵

### E033-a: sigma sweep (手吸引力柔和度)
```
sigma ∈ {1.0, 2.0, 3.0} (baseline=5.0)
# 更低 sigma = 更宽的吸引力分布 = 手不会被"突然拉"向物体
```

### E033-b: CEM budget (samples/iterations)
```
(1) num_samples=2048, max_iterations=32  (2x samples)
(2) num_samples=1024, max_iterations=48  (1.5x iters)
(3) num_samples=2048, max_iterations=48  (2x both, slowest)
```

### E033-c: horizon sweep
```
horizon ∈ {1.0, 1.2, 1.6} (baseline=0.8)
# 注意: horizon 增加会线性增加 rollout 耗时
```

### E033-d: 最佳组合
用 a/b/c 中各自最优参数组合，跑完整评估 + 视频

## 运行时间估算
- E032a baseline (32iter/1024samp/0.8s): ~6min per case
- E033-b3 (48iter/2048samp): ~18min per case
- E033-c horizon=1.6: ~12min per case
- 总估计: ~2h (含分析)

## 运行命令

```bash
# E033-a: sigma sweep
MUJOCO_GL=egl uv run examples/run_mjwp.py +override=core4d_e032a task=desk005_person2 \
    task_body_rew_scale=0.0 base_pos_rew_scale=5.0 base_rot_rew_scale=3.0 \
    hand_approach_rew_scale=3.0 hand_approach_sigma={SIGMA}

# E033-b: CEM budget  
MUJOCO_GL=egl uv run examples/run_mjwp.py +override=core4d_e032a task=desk005_person2 \
    task_body_rew_scale=0.0 base_pos_rew_scale=5.0 base_rot_rew_scale=3.0 \
    hand_approach_rew_scale=3.0 num_samples={N} max_num_iterations={I}

# E033-c: horizon
MUJOCO_GL=egl uv run examples/run_mjwp.py +override=core4d_e032a task=desk005_person2 \
    task_body_rew_scale=0.0 base_pos_rew_scale=5.0 base_rot_rew_scale=3.0 \
    hand_approach_rew_scale=3.0 horizon={H}
```

## 评估脚本

多阈值接触分析 + 稳定性 + 视频:
```python
# Thresholds: 15cm, 10cm, 5cm, 3cm, 1cm
# Stability: pelvis_z > 0.55
# Per-phase: carrying phase only (t=0.3-3.6s for desk005)
```
