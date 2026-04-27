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
| R009b | 2026-04-27 | fix output_dir + ref render + wrist noise + threshold=0.005 | 3.75 | 1.60 | 2.15 | 完成 |
| R010b | 2026-04-27 | fix suitcase slide offset + threshold=0.0 | 5.37 | 2.20 | 3.17 | 完成 |
| **R011** | **2026-04-27** | **per-actuator gain + wrist noise 30% + guidance kp=20** | **5.66** | **2.30** | **3.36** | **完成** |
| HF ref | — | HuggingFace 参考 (data_id=1) | 5.90 | 2.19 | 3.71 | 参考 |

## 关键指标演进

```
rew_mean:  R002(-0.3) → R003(1.93) → R005(2.65) → R006(4.13) → R007(5.33) → R008(5.63) → R011(5.66) → HF(5.90)
tracking:  R002(N/A)  → R003(0.59) → R005(0.77) → R006(2.25) → R007(2.20) → R008(2.52) → R011(2.30) → HF(2.19)
obj_track: R002(N/A)  → R003(1.33) → R005(1.88) → R006(1.88) → R007(3.13) → R008(3.10) → R011(3.36) → HF(3.71)
```

## R011 达标总结

| 指标 | R011 | HF | 达标率 |
|------|------|-----|--------|
| rew_mean | 5.66 | 5.90 | 96% ✅ |
| tracking | 2.30 | 2.19 | 105% ✅ |
| obj_track | 3.36 | 3.71 | 91% ✅ |
| 视频 t=3s | 弯腰抓箱子 | 弯腰抓箱子 | ✅ |
| 视频 t=4s | 弯腰抱箱 | 站立搬箱 | 部分 |

## Logs

- R005: `workspace/hdmi_reproduce/results/R005/`
- R006: `workspace/hdmi_reproduce/results/R006c/`
- R007: `workspace/hdmi_reproduce/results/R007/`
- R008: `workspace/hdmi_reproduce/results/R008/`
- R011: `workspace/hdmi_reproduce/results/R011/`
- HF ref: `example_datasets/processed/hdmi/unitree_g1/humanoid_object/move_suitcase/1/`

## 关键修复 (R005→R008)

1. MuJoCo Warp GPU backend (R005)
2. CEM 参数: noise=0.05, threshold=0.005, check_steps=3, temp=0.3 (R006)
3. Decimation=10 (physics_dt=0.002) (R006b)
4. HDMI PD gains override (R006c)
5. Freejoint init from motion data (R007)
6. Contact guidance: suitcase 6 actuators + decaying gains (R008)
7. Fix output_dir, ref render (freejoint model), wrist noise zeroed (R009)
8. Fix suitcase slide joint offset (body_default_pos) (R010)
9. Per-actuator contact guidance gains (pos=20 vs rot=0.3) (R011)
10. Wrist noise 30% instead of zero (R011)
11. Guidance kp=20, decay=0.85 (R011)

## 已知问题

- R010b (suitcase 位置修正后) 指标 5.37 不如 R008 (suitcase 位置偏移 0.4m) 的 5.63
- R008 的 suitcase 初始位置虽然错误 (偏移 0.4m)，但机器人反而能搬起来
- R010b 修正位置后机器人在 t=4s 仍然摔倒 — 可能 CEM 对 suitcase 起始位置敏感
- 腕关节噪声归零可能过度限制了 CEM 的搜索空间
- improvement_threshold=0.0 导致每步跑满 32 iter，运行时间 ~2000s
