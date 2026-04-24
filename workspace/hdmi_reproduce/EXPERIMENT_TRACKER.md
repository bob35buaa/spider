# HDMI Reproduce 实验跟踪

## 目标

复现 HuggingFace 上 HDMI 的 move_suitcase 物理优化结果（机器人站立行走、抓取并搬运行李箱）。

## 关键指标演进

| Run | 日期 | 描述 | Pelvis Z (mean) | Object Tracking Error | Duration | 状态 |
|-----|------|------|-----------------|----------------------|----------|------|
| baseline | 04-24 | HuggingFace 参考 (data_id=0) | 0.70m | ~0.15m | 10s (250步) | 参考 |
| R001 | 04-25 | 首次 Warp batch stepping | 0.008m | 0.89m | 0.2s (5步) | 失败 |
| R002 | 04-25 | 修复 qpos 布局 + sim/ref 对齐 | **0.789m** | 1.81m | 2s (50步) | 部分通过 |

## Logs

- `log/01_R001_first_run.md` — 首次运行结果分析

## 已知问题

1. 机器人初始化后立刻倒地 (pelvis Z: 0.8m → 0.04m)
2. 仅跑了 10 步 (max_sim_steps=10)，参考为 500 步
3. 输出覆盖了 HuggingFace 原始数据（需重新下载）
