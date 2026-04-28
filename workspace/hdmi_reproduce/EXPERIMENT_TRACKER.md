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
| **R012** | **2026-04-27** | **fix PD gain index ordering (Isaac vs MuJoCo) + full 10s** | **6.56** | **2.93** | **3.63** | **完成** |
| **R013** | **2026-04-28** | **wrist jitter fix: noise=0 + dof_damping=5 + skip zero gains** | **6.83** | **3.07** | **3.75** | **完成** |
| R013b | 2026-04-28 | same as R013 but threshold=0.001 (早停) | 6.50 | 2.83 | 3.67 | 完成 |
| HF ref | — | HuggingFace 参考 (data_id=1) | 5.90 | 2.19 | 3.71 | 参考 |

## 关键指标演进

```
rew_mean:  R002(-0.3) → R003(1.93) → R005(2.65) → R006(4.13) → R007(5.33) → R008(5.63) → R011(5.66) → R012(6.56) → R013(6.83) → HF(5.90)
tracking:  R002(N/A)  → R003(0.59) → R005(0.77) → R006(2.25) → R007(2.20) → R008(2.52) → R011(2.30) → R012(2.93) → R013(3.07) → HF(2.19)
obj_track: R002(N/A)  → R003(1.33) → R005(1.88) → R006(1.88) → R007(3.13) → R008(3.10) → R011(3.36) → R012(3.63) → R013(3.75) → HF(3.71)
```

## R012 达标总结

| 指标 | R012 | HF | 达标率 |
|------|------|-----|--------|
| rew_mean | 6.56 | 5.90 | **111%** ✅ |
| tracking | 2.93 | 2.19 | **134%** ✅ |
| obj_track | 3.63 | 3.71 | **98%** ✅ |
| pelvis_z t=5s | 0.785 | 0.761 | ✅ |
| pelvis_z t=9s | 0.785 | 0.728 | ✅ |

## Logs

- R005: `workspace/hdmi_reproduce/results/R005/`
- R006: `workspace/hdmi_reproduce/results/R006c/`
- R007: `workspace/hdmi_reproduce/results/R007/`
- R008: `workspace/hdmi_reproduce/results/R008/`
- R011: `workspace/hdmi_reproduce/results/R011/`
- R012: `workspace/hdmi_reproduce/results/R012/`
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
12. Fix PD gain index ordering: Isaac breadth-first vs MuJoCo depth-first (R012)
13. Full 10s simulation (500 sim steps, was 250) (R012)
14. Wrist MPC noise → 0 (match HDMI/HF: wrists not in action space) (R013)
15. Skip zero-gain override for wrist joints (keep scene XML Kp=14-17) (R013)
16. Add dof_damping=5.0 for wrist joints (ζ=0.20 → ζ≈1.1, critical damping) (R013)

## R013 腕关节抖动修复 (新 SOTA)

| 指标 | R012 | **R013** | R013b | HF | 变化(R013 vs R012) |
|------|------|----------|-------|-----|-------------------|
| rew_mean | 6.56 | **6.83** | 6.50 | 5.90 | +4.1% |
| tracking | 2.93 | **3.07** | 2.83 | 2.19 | +4.8% |
| obj_track | 3.63 | **3.75** | 3.67 | 3.71 | +3.3% |
| opt_steps | 32.0 | 32.0 | 6.56 | 3.12 | — |
| 运行时间 | ~110min | 107min | ~23min | — | — |
| wrist jitter (前2s) | 1.0-4.0° | **0.003-0.012°** | 0.02-0.07° | — | **-99%** |

注: R013b 的 threshold=0.001 导致站立阶段优化不足 (opt=1-3)，产生摔倒倾向

## 已知问题

- improvement_threshold=0.0 导致每步跑满 32 iter，运行时间 ~2000s (R013 验证中)
- R013b 使用 threshold=0.001 可大幅减少运行时间但保持性能
