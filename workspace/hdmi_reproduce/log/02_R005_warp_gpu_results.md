# R005 Results: MuJoCo Warp GPU Backend

## Summary
Rewrote HDMI backend from CPU MuJoCo (N MjData + thread pool) to MuJoCo Warp GPU architecture, aligned with MJWP workflow.

## Configuration
- N=1024, max_iter=32, max_sim_steps=250
- device=cuda:0 (RTX 5090)
- ctrl = joint position targets (radians) via XML affine actuators
- scene XML: mjlab scene.xml (nq=43, nv=41, nu=29)

## Results

| 指标 | R005 | R003 (CPU) | HF 参考 | R005 目标 |
|------|------|-----------|---------|----------|
| rew_mean | **2.65** | 1.93 | 5.72 | > 4.0 |
| tracking_mean | **0.77** | 0.59 | 2.11 | > 1.5 |
| obj_track_mean | **1.88** | 1.33 | 3.61 | > 2.0 |
| opt_steps_mean | 1.00 | 1.00 | 2.5 | > 2.0 |
| step perf (N=1024) | **~2ms** | 90ms | — | < 5ms |
| total time (250 steps) | **16.4s** | ~150s est | — | — |

## Performance (达标)
- step_env: ~2ms/step at N=1024 GPU (vs 90ms/step N=1024 CPU) — **45x 加速**
- Total time: 16.4s for 250 steps (vs ~150s estimated for CPU)
- Control tick: ~110ms (32 iterations rollout + overhead)

## Phase breakdown

| Quarter | rew | tracking | obj_track |
|---------|-----|----------|-----------|
| Q1 (0-62) | 3.73 | 0.88 | 2.85 |
| Q2 (63-124) | 3.51 | 0.66 | 2.85 |
| Q3 (125-186) | 2.54 | 0.85 | 1.69 |
| Q4 (187-250) | 0.87 | 0.70 | 0.17 |

## Open-loop verification (Phase 2)

| 指标 | R005 (step 1) | R005 (step 10) | HF 参考 |
|------|--------------|----------------|---------|
| rew | **5.14** | **4.63** | 5.40 |
| tracking | **2.15** | **1.78** | 2.22 |
| obj_track | **3.00** | **2.85** | 2.87 |

Open-loop initial reward matches HF closely — physics simulation is correct.

## Claims 验证

| Claim | 状态 | 结果 |
|-------|------|------|
| GPU Warp 可用 | ✅ 通过 | mujoco_warp 3.7.0 on RTX 5090 sm_120 工作正常 |
| step < 5ms/step | ✅ 通过 | ~2ms/step at N=1024 |
| Open-loop ≈ 5.2 | ✅ 通过 | 5.14 (HF: 5.40) |
| rew_mean > 4.0 | ❌ 未达标 | 2.65 — CEM 仍然早停 |
| tracking > 1.5 | ❌ 未达标 | 0.77 |
| obj_track > 2.0 | ❌ 未达标 | 1.88 |

## 分析

### 进步
1. **GPU pipeline 完整可用**: setup/step/save/load/sync/reward 全部 GPU 化
2. **性能大幅提升**: 45x faster than CPU at N=1024
3. **Open-loop reward 匹配 HF**: 物理仿真正确
4. **Full state save/load**: 30+ fields via _copy_state，比 R003 的 4 fields 完整得多

### 未解决的问题
1. **CEM 始终早停 (opt_steps=1)**: 噪声控制全部比参考差 → improvement=0
2. **obj_track 后半程崩溃**: Q4=0.17，suitcase 跟踪完全失败
3. **与 HF 的 gap**: HF opt_steps=2.5，说明 HF 的优化器确实在工作

### 根因分析
CEM 早停问题的可能原因：
1. **噪声尺度不对**: joint_noise_scale=0.2 对关节位置目标可能太大（这不是 normalized action 了）
2. **horizon/temperature 参数**: 可能需要调整
3. **ctrl_ref 作为基准太优**: 参考轨迹已经接近最优，小扰动只会变差

## 下一步 (R006 候选)
1. **降低 noise_scale**: 试 0.05-0.1（因为 ctrl 现在是弧度不是归一化）
2. **增大 temperature**: 试 0.5-1.0（让优化器更探索）
3. **对比 HF 的 CEM 参数**: 检查 HF 用的 noise/temperature
