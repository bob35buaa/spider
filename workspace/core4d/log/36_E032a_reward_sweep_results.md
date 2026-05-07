# E032a: Hand Approach + Reward Weight Sweep Results

## 状态: 完成 — 确认最佳配置

## 实验矩阵

### Quick Tests (8iter/256samp) — 用于方向验证
| 配置 | box025 contact<10cm | desk005 contact<10cm | bucket contact<10cm |
|------|-------------------|--------------------|--------------------|
| E027d2 baseline (base=5) | 65%* | 9%* | 36%* |
| +HA=3 +task_body=5 +base=10 | 56% | 30% | 63% |
| +HA=3 no_task base=15 | 82% | 38% | 24% |
| +HA=3 no_task base=25 | 82% | 38% | 24% |

*注: baseline 的 quick-test stable 也只有 18%! quick mode CEM 能力不足以稳定行走。contact 比较仍有参考价值。

**关键发现**:
1. `task_body_rew` **有害** — 强制手/脚跟踪 ref 位置与 balance 冲突
2. `hand_approach_rew` 有效提升接触但**与 stability 严重 tradeoff**
3. `base_pos` 提高对接触改善无帮助（甚至降低接触）

### Full Tests (32iter/1024samp) — 最终结论
| 配置 | box stable | box contact<10cm | desk stable | desk contact<10cm | bucket stable | bucket contact<10cm |
|------|-----------|-----------------|-------------|-------------------|--------------|---------------------|
| E027d2 (base=5, HA=0) | **100%** | 65% | 87% | 9% | **100%** | 36% |
| E032a (base=5, HA=3) | 71% | **81%** | 79% | **83%** | **100%** | **63%** |
| E032a (base=10, HA=3) | **100%** | 57% | 75% | **82%** | - | - |

## 最佳配置

**Per-case 最优**:
- box025: `base_pos=10, HA=3` → 100% stable, 57% contact (vs baseline 65% — 略退步但仍可接受)
- desk005: `base_pos=5, HA=3` → 79% stable, 83% contact (巨大提升! 9%→83%)
- bucket010: `base_pos=5, HA=3` → 100% stable, 63% contact (36%→63%)

**通用最优**: `base_pos=5, hand_approach=3, no task_body`
- 在 desk005 和 bucket010 上大幅改善接触
- box025 stability 从 100%→71% (需要针对性调高 base_pos)

## Round 2 结论 (PD Gain Sweep)
- kp=100: pos_err=0.38 (物体跟踪太差)
- kp=200: pos_err=0.27 (中等)
- kp=500 (default): pos_err=0.13
- **结论**: 降低 PD gains 只会让物体跟踪变差，不推荐

## 下一步
- Round 3: HDMI physics_dt=0.002 测试
- 最终评估: 用 per-case 最优配置跑 3 case + 视频
