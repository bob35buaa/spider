# E032a: Hand Approach + Reward Weight Sweep Results

## 状态: 完成 — 视频分析确认 desk005 改善，box025 退步

## 视频可视化深度分析

### E032a box025 (base_pos=10, HA=3) — 相比 E027d2 退步

- **0% (0s)**: ref/sim 一致，初始姿态良好
- **20% (1s)**: ref 弯腰俯身贴箱面。sim **趴在箱子上方** — 为了靠近物体表面牺牲了身体平衡，hand_approach 将上半身拉向箱子
- **40% (2s)**: ref 站立推箱走。sim **蹲在箱后、头几乎低于箱顶** — 严重偏离 ref 姿态，HA reward 主导了 CEM 优化方向
- **60% (3s)**: ref 正常推。sim **看不到机器人**（被箱子挡住，说明机器人在箱后极低姿态或半跪）
- **80% (4s)**: sim 头部可见在箱后上方，姿态扭曲，脚似乎脱离地面
- **100% (5s)**: sim **完全不可见**（在箱后方，可能倒地或极低蹲姿）

**结论**: E032a 的 hand_approach reward 对 box025 是**有害的**。机器人为了手靠近物体放弃了正常的行走/推动姿态，变成趴/蹲在箱后的畸形姿态。虽然数值上 contact<10cm=57%（base=10时），但**视觉质量远差于 E027d2**。E027d2 虽然 contact 更低(65%)，但机器人姿态正常、像在走路推箱。

**用户判断确认**: "box025变坏了" ✓

### E032a desk005 (base_pos=5, HA=3) — 相比 E027d2 显著改善

- **0% (0s)**: ref/sim 一致
- **20% (0.9s)**: ref 侧身走。sim **弯腰前倾、手伸向桌面** — hand_approach 引导手靠近桌子，身体前倾但还在走
- **40% (1.9s)**: ref 走路中。sim 严重前倾、**一只脚拖地** — 不稳定但没有完全摔倒（E027d2 在此时刻完全摔倒面朝下）
- **60% (2.8s)**: sim **恢复较好姿态，手在桌面附近**，身体虽然前倾但在行走
- **80% (3.7s)**: ref 弯腰看桌面。sim 也弯腰、**手接近桌面** — 姿态与 ref 较匹配
- **100% (4.6s)**: ref/sim 都在桌旁，sim 走路姿态合理

**结论**: 相比 E027d2 (40%完全摔倒面朝下)，E032a 虽然 40% 处也不稳但**没有完全摔倒**，且后半段手持续靠近桌面。手-桌接触 83% 是真实的视觉改善。

**用户判断确认**: "desk005要比之前的E027d2要好" ✓

### E032a bucket010 (base_pos=5, HA=3) — 接触时间增加但方式不同

- **0% (0s)**: ref/sim 在桶旁
- **20% (1s)**: ref 弯腰手伸向桶。sim 也弯腰但**手往下伸得更低**（低于桶底部），HA 引导方向偏了
- **40% (2s)**: ref 手在桶旁。sim 手在桶侧面附近但**身体偏离桶更远**
- **60% (3s)**: ref/sim 都在桶旁走。sim 手偏离桶，身体距离比 ref 大
- **80% (4s)**: ref 弯腰手在桶顶。sim **极度前倾、头几乎碰桶顶** — HA 再次让上半身过度前倾
- **100% (5s)**: sim 在桶旁，姿态还算正常

**结论**: 接触时间确实增加(36→63%)，但机器人姿态在某些时刻过度前倾。整体比 E027d2 略好但也有畸形时刻。

**用户判断确认**: "好像接触时间变多了" ✓ "看不出来变好还是变坏" — 确实是混合结果

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
