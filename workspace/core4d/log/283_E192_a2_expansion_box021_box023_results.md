# E192 扩展结果：A2 hand-gate 补 box021/box023 —— 跨物体复现姿态门回退，A2 仍不升级

_Core4D · Phase 61 · 2026-08-13 · 计划 [plan226](../plan/226_E198_g1xa2_factorial_and_E192_a2_expansion_plan.md) · 承接 [E192 log271](271_E192_A2_full_diagnostic_results.md) · evaluator `core4d-e154-physics-contact-v1`_

## TL;DR

- **执行闭合**：A2 单臂 Full CEM 补齐 box021(28)+box023(16)=**44/44**（本地 8 卡 priority 队列 tier P1，与他人 job 叠加共跑，0 失败），用公共 evaluator 打分，0 error。至此 A2 覆盖 box024/004（E192）+ box021/023（本轮）四物体。
- **A2 对 box021/023 的 object tracking 无收益、且略伤**：`track_obj_z_abs_err_cm_mean` box021 `6.237→6.429`(+0.19)、box023 `5.817→5.848`(+0.03)；3D pos 与 in-mask contact 变化都很小（见下表）。
- **跨物体复现 E192 的核心问题——A2 单用损伤姿态/朝向门**：在四物体全 59 例上，A0→A2 迁移显著砸 `hand_ori −16.9pp`(exact p=0.006)、`root_ori −13.6pp`(p=0.021)、`root_pos −6.8pp`、`hand_pos −5.1pp`（详见 [E198 factorial log284](284_E198_g1xa2_factorial_results.md) §3）。
- **判决：维持 E192 结论，A2 不升级为默认 hand-gate 策略。** 本轮把 E192 的 box024/004-only 证据扩展到中箱 box021 与小箱 box023，A2 的“降穿透不稳定 + 姿态代偿”在更大物体集上一致，不是 box024 特异。A2 的 `INCONCLUSIVE_GATE_COLLAPSE` governance 保持。

## 1. 设计

补 E192 从未跑过的 box021/box023 的 A2 单臂（`cem_hand_gate_min_sdf=-0.010 / max_violation=0.05 / hard_floor=-0.015`，no-gravcomp，PRG on，`rubber_hull`，CEM 1024×32 seed 0）。A0 基线复用 box021(E170/E169)、box023(E173/E179) 历史 PRG rollout，由同一 evaluator 重打分（C3 parity：box021/023 A0 重打分 vs 冻结 E194 表 z 差 `0.000000 cm`）。retarget variant 逐 case 保留（box021 v1=25/v2=3，box023 v1=15/v2=1）。

本轮是 [E198 G1×A2 因子](284_E198_g1xa2_factorial_results.md)的 P1 tier，A2 单臂结果既独立成立（本 log），又作为 2×2 的 (G=0,A=1) 单元。

## 2. A2 vs A0（box021/box023，逐物体均值）

| 指标（越低越好，contact 越高越好） | box021 A0 | box021 A2 | Δ | box023 A0 | box023 A2 | Δ |
|---|---:|---:|---:|---:|---:|---:|
| track_obj_z_abs_err_cm_mean | 6.237 | 6.429 | +0.192 | 5.817 | 5.848 | +0.031 |
| track_obj_pos_err_cm_mean | 15.675 | 15.173 | −0.502 | 13.062 | 12.465 | −0.597 |
| hand 3mm penetration | 0.176 | 0.195 | +0.019 | 0.148 | 0.144 | −0.004 |
| leg penetration | 0.120 | 0.085 | −0.035 | 0.031 | 0.039 | +0.008 |
| 3mm in-mask contact | 0.426 | 0.415 | −0.011 | 0.473 | 0.502 | +0.029 |
| obj ori err (deg) | 6.091 | 6.261 | +0.170 | 5.108 | 5.160 | +0.052 |

A2 单用在 box021/023 上：z 无改善（略升）、3D 位置略降、手物穿透基本不变（box021 反升）、接触基本持平。没有出现 E192 曾在 box024 上看到的“均值降穿透”，说明降穿透效应本身既弱又物体特异。

## 3. Claims（本扩展相关）

| Claim | 判定 | 证据 |
|---|---|---|
| scope/provenance | PASS | box021×28+box023×16=44 唯一 case；variant/override/scene/trajectory/contact SHA 全可追溯 |
| 单变量 intervention | PASS | 仅改 3 个 `cem_hand_gate_*` 字段，no-gravcomp，其余同 A0 |
| execution/numeric closure | PASS | 44/44 Full；scored 0 error；non-finite/diverged=0 |
| baseline parity (C3) | PASS | box021/023 A0 重打分 vs 冻结 E194 表 z 差 `0.000000 cm` |
| A2 跨物体有效性 | **FAIL（维持 E192）** | z 无改善；穿透不稳定且不叠加；姿态/朝向门跨物体显著回退 |

## 4. 结论

E192 当时只在 box024/004 上诊断 A2，判 `INCONCLUSIVE_GATE_COLLAPSE / THRESHOLD_POLICY_NOT_EFFECTIVE`。本轮把 A2 扩到 box021/box023 后，同一姿态/朝向门回退在中箱/小箱一致复现，且没有稳定的 object-tracking 或降穿透收益。**A2 不升级**；其价值只有在与 G1 组合时体现为“被 G1 救援后残留的一点姿态门微调”（见 log284 的 A2→G1+A2 迁移），而非独立有效策略。

## 5. 可视化

见 [log284 §6](284_E198_g1xa2_factorial_results.md)：安装 `libosmesa6` 后 osmesa 渲染恢复，已产出 2×2 四单元 MP4（含各 case 的 A2 单元）；`review_player.sh E198` 可 live 回放全部 236 个 arm-case（含 box021/023 的 A2 臂）供逐例复核 A2 姿态回退。

## 6. 产物

- CEM：`workspace/core4d/results/E192/s6_downstream/cem/full_a2_expansion/`（44）
- 快照：`workspace/core4d/results/E192/scene_snapshot/a2_expansion/`
- 打分：并入 `results/E198/s6_downstream/eval/full_factorial/e198_arm_cache.tsv`（arm=A2 的 box021/023 行）与 `e198_factorial_by_object.tsv`

## 7. 下一步

见 [log284 §9](284_E198_g1xa2_factorial_results.md)。不再对 A2 单参数重复 seed。
