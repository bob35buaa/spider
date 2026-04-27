# R012: 修复 PD 增益索引排序 bug + 完整 10s

## Context

R011 达 HF 96% (rew=5.66) 但 pelvis_z t=4s=0.63 < 0.7 — 机器人弯腰后无法站直。
发现 `hdmi.py:558-574` PD 增益索引排序 bug：用 MuJoCo 深度优先索引查 Isaac 广度优先数组，29 个关节中 20 个获得错误增益。

关键错误：ankle 7.5-10x 过高、waist 7.5x 过低、右臂归零。

## Claims

1. pelvis_z t=4s > 0.70 (waist 恢复 7.5x + ankle 恢复柔性)
2. Q4 tracking > 1.5 (R011 Q4=0.98)
3. rew_mean > 5.8 (HF=5.90)
4. 完整 10s 后总指标与 HF 可比

## 改动

| # | 文件 | 改动 |
|---|------|------|
| 1 | `spider/simulators/hdmi.py` L555-574 | `robot.joint_names` 替代 MuJoCo joint list |
| 2 | 命令行 | 去掉 `max_sim_steps=250`，用 yaml 默认 -1 (10s) |

## 训练命令

```bash
uv run examples/run_hdmi.py +data_id=1 \
    output_dir=workspace/hdmi_reproduce/results/R012
```

## 成功标准

| 指标 | R011 (5s) | HF (10s) | R012 目标 |
|------|-----------|----------|-----------|
| rew_mean | 5.66 | 5.90 | > 5.8 |
| tracking | 2.30 | 2.19 | > 2.0 |
| obj_track | 3.36 | 3.71 | > 3.5 |
| pelvis_z t=4s | 0.63 | 0.75 | > 0.70 |
