# R006 计划：CEM 参数调优 + data_id=1 对齐 HF

## Context

R005 完成了 HDMI 后端重写为 MuJoCo Warp GPU，open-loop reward 匹配 HF (5.14 vs 5.40)，性能 45x 加速。但 CEM 始终早停 (opt_steps=1)，rew_mean=2.65 远低于 HF 的 5.72。

### 根因分析

通过深入研究 CEM 优化器代码和 HF 参考数据，发现三个根本问题：

**1. 噪声尺度过大 (最关键)**
- R005: `joint_noise_scale=0.2` → 所有 29 个 DOF 噪声 = 0.2 弧度 ≈ 11.5°
- HF 等效噪声: normalized action × action_scaling = 0.2 × (0.07~0.55) = **0.014~0.11 弧度**
- R005 的噪声比 HF 大 **2~14 倍**，扰动太大导致所有采样比参考差

**2. 早停阈值过高**
- `improvement_threshold=0.02` + `improvement_check_steps=1`
- HF 参考 data_id=1: mean improvement = 0.011，仅 1.6% 的 timestep improvement > 0.02
- 但 HF 仍有 40.8% timestep 的 opt_steps > 1 → 说明 HF 有足够多的 timestep 能过阈值
- 我们的噪声太大导致 improvement ≈ 0 甚至为负 → 100% 早停

**3. data_id 不匹配**
- R005 用 data_id=0，HF 参考数据是 data_id=1
- HF data_id=1: rew_mean=5.90, tracking=2.19, obj_track=3.71

**4. nu=29 vs HF nu=23 (次要)**
- HF 不控制 6 个腕关节 (left/right × roll/pitch/yaw)
- 我们控制全部 29 个，包括腕关节 → 多 6 个噪声维度但无实质收益
- 暂不修改，先解决噪声/阈值问题

## Claims (可验证的假设)

1. **降低 noise_scale 到 0.05 将使 opt_steps > 1 的比例 > 30%**
   - 理由: 0.05 弧度 ≈ 2.9°，接近 HF 等效噪声范围 (0.014~0.11)
2. **降低 improvement_threshold 到 0.005 将进一步增加 opt_steps**
   - 理由: HF mean improvement = 0.011，阈值 0.005 能让更多改进被接受
3. **组合调优后 rew_mean > 4.0 (data_id=1)**
   - 理由: open-loop 已 ~5.1，CEM 应至少维持不低于 open-loop

## 改动

### 代码修改 (仅配置参数)

| 文件 | 改动 | 原值 | 新值 |
|------|------|------|------|
| `examples/config/hdmi.yaml` | joint_noise_scale | 0.2 | 0.05 |
| `examples/config/hdmi.yaml` | improvement_threshold | 0.02 | 0.005 |
| `examples/config/hdmi.yaml` | improvement_check_steps | 1 | 3 |
| `examples/config/hdmi.yaml` | temperature | 0.1 | 0.3 |

### 运行参数

```bash
uv run python examples/run_hdmi.py \
    task=move_suitcase \
    +data_id=1 \
    viewer=none \
    save_video=true \
    save_info=true \
    max_sim_steps=250 \
    num_samples=1024 \
    max_num_iterations=32 \
    output_dir=workspace/hdmi_reproduce/results/R006
```

## 成功标准

| 指标 | R005 | R006 目标 | HF 参考 (data_id=1) |
|------|------|----------|---------------------|
| rew_mean | 2.65 | **> 4.0** | 5.90 |
| tracking | 0.77 | **> 1.5** | 2.19 |
| obj_track | 1.88 | **> 2.5** | 3.71 |
| opt_steps | 1.00 | **> 1.5** | 3.12 |
| % opt_steps > 1 | 0% | **> 30%** | 40.8% |

## 验证步骤

1. 修改 hdmi.yaml 参数
2. 用 test_hdmi_fast.py (data_id=1, N=64) 快速验证：
   - 确认 open-loop reward 合理 (~5.0+)
   - 确认 step_env 性能正常 (~2ms)
3. 完整运行 (N=1024, 250 steps)
4. 分析 trajectory_hdmi.npz:
   - opt_steps 分布
   - rew_mean 全程 + Q1/Q2/Q3/Q4 分解
   - 与 HF data_id=1 对比
5. 记录 log + 更新 tracker

## 风险

| 风险 | 缓解 |
|------|------|
| noise=0.05 仍然太大 | Phase 4 备选: 试 0.02 |
| data_id=1 轨迹长度不同 | 检查 max_episode_length |
| temperature=0.3 太保守 | 如 opt_steps 不够，试 0.5 |

## 需要修改的文件

| # | 文件 | 改动 |
|---|------|------|
| 1 | `examples/config/hdmi.yaml` | 4 个参数调整 |
| 2 | `workspace/hdmi_reproduce/scripts/test_hdmi_fast.py` | data_id=0 → 1 |
| 3 | `workspace/hdmi_reproduce/plan/R006_plan.md` | 本计划文件 |
