# R006 Results: CEM 参数调优 + data_id=1

## Summary
调优 CEM 参数 (noise_scale, threshold, check_steps, temperature) 并切换到 data_id=1 对齐 HF。

## Configuration
- N=1024, max_iter=32, max_sim_steps=250, data_id=1
- **改动**: noise=0.05, threshold=0.005, check_steps=3, temp=0.3

## Results

| 指标 | R005 | R006 | HF (data_id=1) | 目标 |
|------|------|------|-----------------|------|
| rew_mean | 2.65 | **2.71** | 5.90 | > 4.0 ❌ |
| tracking | 0.77 | **0.83** | 2.19 | > 1.5 ❌ |
| obj_track | 1.88 | **1.88** | 3.71 | > 2.5 ❌ |
| opt_steps | 1.00 | **3.58** | 3.12 | > 1.5 ✅ |
| % opt > 1 | 0% | **100%** | 40.8% | > 30% ✅ |
| total time | 16.4s | **47.0s** | — | — |

## Phase breakdown

| Quarter | rew | tracking | obj_track |
|---------|-----|----------|-----------|
| Q1 | 4.07 | 1.22 | 2.85 |
| Q2 | 3.58 | 0.73 | 2.85 |
| Q3 | 2.46 | 0.77 | 1.69 |
| Q4 | 0.78 | 0.61 | 0.17 |

## Claims 验证

| Claim | 状态 | 结果 |
|-------|------|------|
| noise=0.05 使 opt_steps > 1 比例 > 30% | ✅ 通过 | 100%，mean=3.58 |
| threshold=0.005 增加 opt_steps | ✅ 通过 | check_steps=3 是主因 |
| rew_mean > 4.0 | ❌ 未达标 | 2.71 — CEM 迭代更多但 reward 没改善 |

## 分析

### CEM 参数修复有效但不够
- opt_steps 从 1.0 → 3.58 (甚至超过 HF 的 3.12)
- 但 rew_mean 几乎没变 (2.65 → 2.71)
- 说明 CEM 找到的"改进"只是统计噪声，不是真正的优化

### obj_track 崩溃模式不变
- Q3/Q4 的 obj_track 和 R005 完全一致: Q3=1.69, Q4=0.17
- 说明 **suitcase 后半程跟踪失败是物理模拟问题，不是 CEM 参数问题**
- 可能原因: suitcase freejoint 在 MuJoCo Warp 中漂移，缺少物体保持力

### 与 HF 的核心差距
HF rew_mean=5.90 vs R006=2.71，差距主要来自:
1. obj_track 后半程崩溃 (HF Q4=2.66 vs R006 Q4=0.17)
2. tracking 全程偏低 (HF ~2.2 vs R006 ~0.8)

### 根因推测
1. **nu=29 vs HF nu=23**: 我们多控制了 6 个腕关节，CEM 在 29D 空间搜索效率低于 23D
2. **ctrl 语义不同**: 我们直接写 joint pos target (rad)，HF 写 normalized action → PD
3. **suitcase 物理**: 可能需要 object actuator 或 contact guidance 来保持物体

## 下一步
1. **回退到 23 DOF**: 只控制 HF 的 23 个关节，固定腕关节
2. **ctrl 语义对齐**: 考虑是否需要回到 normalized action 空间
3. **调查 suitcase 漂移**: 对比 HF 和 R006 在 Q3/Q4 的 object xpos
