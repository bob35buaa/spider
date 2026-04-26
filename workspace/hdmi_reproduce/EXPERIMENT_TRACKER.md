# HDMI Reproduce Experiment Tracker

## 实验总览

| Run | 日期 | 描述 | rew_mean | tracking | obj_track | 状态 |
|-----|------|------|----------|----------|-----------|------|
| R002 | 2026-04-24 | feat/hdmi-uv-env, 简化 qpos L2 reward | -0.3 | N/A | N/A | 完成 |
| R003 | 2026-04-25 | CPU MuJoCo + HDMI native reward (N=8) | 1.93 | 0.59 | 1.33 | 完成 |
| R005 | 2026-04-26 | MuJoCo Warp GPU (N=1024, 250 steps) | **2.65** | **0.77** | **1.88** | 完成 |
| HF ref | — | HuggingFace 参考 (data_id=0) | 5.72 | 2.11 | 3.61 | 参考 |

## 关键指标演进

```
rew_mean:  R002(-0.3) → R003(1.93) → R005(2.65) → HF(5.72)
tracking:  R002(N/A)  → R003(0.59) → R005(0.77) → HF(2.11)
obj_track: R002(N/A)  → R003(1.33) → R005(1.88) → HF(3.61)
step_perf: R003(90ms@N=1024) → R005(2ms@N=1024) → 45x 加速
```

## Logs

- R002: `workspace/hdmi_reproduce/results/R002/`
- R003: `workspace/hdmi_reproduce/results/R003/`
- R005: `workspace/hdmi_reproduce/results/R005/`
- HF ref: `example_datasets/processed/hdmi/unitree_g1/humanoid_object/move_suitcase/0/`

## 已知问题

- CEM 优化器始终早停 (opt_steps=1): 噪声控制全部比参考差，需调 noise_scale/temperature
- data_id=1 的 HF trajectory 被 R003 覆盖，需重新下载
- obj_track 后半程崩溃 (Q4=0.17): suitcase 跟踪失败
