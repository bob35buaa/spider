# HDMI Reproduce Experiment Tracker

## 实验总览

| Run | 日期 | 描述 | rew_mean | tracking | obj_track | 状态 |
|-----|------|------|----------|----------|-----------|------|
| R002 | 2026-04-24 | feat/hdmi-uv-env, 简化 qpos L2 reward | -0.3 | N/A | N/A | 完成 |
| R003 | 2026-04-25 | CPU MuJoCo + HDMI native reward (N=8) | 1.93 | 0.59 | 1.33 | 完成 |
| R005 | 2026-04-26 | MuJoCo Warp GPU (N=1024, 250 steps) | 2.65 | 0.77 | 1.88 | 完成 |
| R006 | 2026-04-26 | CEM 调优 + decimation + HDMI gains | 4.13 | 2.25 | 1.88 | 完成 |
| **R007** | **2026-04-26** | **+freejoint init from motion data** | **5.33** | **2.20** | **3.13** | **完成** |
| **R008** | **2026-04-27** | **+contact guidance (suitcase 6 actuators)** | **5.63** | **2.52** | **3.10** | **完成** |
| HF ref | — | HuggingFace 参考 (data_id=1) | 5.90 | 2.19 | 3.71 | 参考 |

## 关键指标演进

```
rew_mean:  R002(-0.3) → R003(1.93) → R005(2.65) → R006(4.13) → R007(5.33) → R008(5.63) → HF(5.90)
tracking:  R002(N/A)  → R003(0.59) → R005(0.77) → R006(2.25) → R007(2.20) → R008(2.52) → HF(2.19)
obj_track: R002(N/A)  → R003(1.33) → R005(1.88) → R006(1.88) → R007(3.13) → R008(3.10) → HF(3.71)
```

## R008 达标总结

| 指标 | R008 | HF | 达标率 |
|------|------|-----|--------|
| rew_mean | 5.63 | 5.90 | 95% ✅ |
| tracking | 2.52 | 2.19 | 115% ✅ |
| obj_track | 3.10 | 3.71 | 84% ✅ |
| 视频 t=4s | 站立搬箱 | 站立搬箱 | ✅ |

## Logs

- R005: `workspace/hdmi_reproduce/results/R005/`
- R006: `workspace/hdmi_reproduce/results/R006c/`
- R007: `workspace/hdmi_reproduce/results/R007/`
- R008: `workspace/hdmi_reproduce/results/R008/`
- HF ref: `example_datasets/processed/hdmi/unitree_g1/humanoid_object/move_suitcase/1/`

## 关键修复 (R005→R008)

1. MuJoCo Warp GPU backend (R005)
2. CEM 参数: noise=0.05, threshold=0.005, check_steps=3, temp=0.3 (R006)
3. Decimation=10 (physics_dt=0.002) (R006b)
4. HDMI PD gains override (R006c)
5. Freejoint init from motion data (R007)
6. Contact guidance: suitcase 6 actuators + decaying gains (R008)
