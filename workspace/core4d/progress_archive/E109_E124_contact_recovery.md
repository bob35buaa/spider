# E109-E124 Progress Archive — Contact Recovery

## E109 — Spider vs OmniRetarget fair-eval (2026-06-02)

- [x] 检查 `existing_cases.tsv`：1931 行，其中 `cem_status=pass` 12 条。
- [x] 创建计划：`workspace/core4d/plan/118_E109_spider_vs_omniretarget_fair_eval_plan.md`。
- [x] 新增可复用脚本：`workspace/core4d/scripts/eval_omni_vs_spider/`，包含 case bank 构建、多阈值 proxy 指标计算、method/object/threshold summary。
- [x] 运行 `run_E109_fair_eval.sh`：共 129 行 case bank；输出三档阈值 summary：3cm/5cm/8cm。
- [x] 新增完整 OmniRetarget vs Spider 配对对比生成器：12/12 existing CEM pass case 均有 OmniRetarget 轨迹与 Spider CEM summary 覆盖，warnings 为 0。
- [x] 写结果 log：`workspace/core4d/log/139_E109_spider_vs_omniretarget_fair_eval_results.md`。
- [x] 统一 replay 评测补齐：新增 `unified_replay_eval.py`，只使用 12 条 `cem_status=pass` case。历史对齐校验共 87 项、mismatch=0。
- [x] E109 汇报表收口：新增 24-case work 扩展表、20-case 汇报表 xlsx，补齐 `EEF 12/15cm`、`hand_geom_near_*`、`hand_geom_penetration`、`hand_object_physics_contact`。
- [x] E109 接触差异分析：Spider 相对 OmniRetarget 的近场接触下降主要集中在 `eef_near_3/5/8cm` 和 `hand_object_physics_contact`，但 `hand_geom_deep_penetration_2cm` 从 29.4% 降到 3.1%。
- [x] 创建接触提升总控计划：`workspace/core4d/plan/contact_improvement_plan.md`。

## E110 — Contact metric audit (2026-06-02)

- [x] 创建计划：`workspace/core4d/plan/119_E110_contact_metric_audit_plan.md`。
- [x] 评测完成：48 method rows / 24 paired cases。Spider deep penetration `29.4%→3.1%`，physics contact `54.4%→43.1%`，hand12 `67.1%→65.7%`；failure labels 为 `penetration_removed_contact_not_recovered=18`。
- [x] 写结果 log：`workspace/core4d/log/140_E110_contact_metric_audit_results.md`。

## E111 — Contact evidence chain (2026-06-02)

- [x] 创建计划：`workspace/core4d/plan/120_E111_contact_evidence_chain_plan.md`。
- [x] 实现完成：S1 raw contact TSV 新增 contact artifact path/person/label/raw frame/active/run-length/contact target status；NPZ 新增 3cm/5cm world/object-local centroid；S3/S4/S5/registry 传播。
- [x] 验证完成：fixture `rows=1, mean raw-contact F1=0.857143`；chain smoke pass；release audit `63/63`。
- [x] log: `workspace/core4d/log/141_E111_contact_evidence_chain_results.md`。

## E112 — Contact-aware CEM ablation (2026-06-02/03)

- [x] 创建计划：`workspace/core4d/plan/121_E112_contact_aware_cem_ablation_plan.md`。Phase A 为 3 case x 3 variants（baseline_ref_fk/raw_mask_ref_fk/hold_band）。
- [x] Phase A 实现：9 个 override、preflight 通过。修正 `baseline_ref_fk` 继承 `contact_hdmi_gain` 的问题。
- [x] smoke 完成：远端首次 smoke 失败根因为缺 object mesh assets，已修复。
- [x] full CEM 完成：9/9 NPZ + MP4。full 结论：box004/box021 的 raw_mask 与 hold_band 均从 FAIL 提到 WORK，contact 分别提升到 54-78%；box026 有提升但仍 FAIL。
- [x] log: `workspace/core4d/log/142_E112_contact_aware_cem_ablation_results.md`。

## E113 — Contact-aware expanded workset (2026-06-03)

- [x] 创建计划：`workspace/core4d/plan/122_E113_contact_aware_expanded_workset_plan.md`。Phase A 扩展 6 个非 box026 runnable case。
- [x] smoke + full CEM 完成：6/6 variants，无 release candidate。3 个 strict WORK 均未达到 +8pp 接触改善；`box021_029_p2` 物理接触 +20.0pp 但 lower-body interference 8.0%。
- [x] log: `workspace/core4d/log/143_E113_contact_aware_expanded_workset_results.md`。

## E114 — Contact alignment RL handoff gate (2026-06-03)

- [x] E114 gate 结果：6 rows evaluated，`rl_ready_rows=0`，blocked 6。诊断队列：`strict_contact_margin=3`、`lowerbody_repair=2`、`lowerbody_aware_contact=1`。
- [x] log: `workspace/core4d/log/144_E114_contact_alignment_rl_handoff_gate_results.md`。

## E115 — Lowerbody-aware contact diagnostic (2026-06-03)

- [x] 15 variants smoke 完成，0 release candidate；decision counts 为 `lowerbody_fixed_contact_fail=12`、`penetration_fail=1`、`review=2`。主 case `box021_029_p2`：naive leg penalty 的核心 trade-off 为 s2/s4 将 lower-body 8.0%→0.0%，但物理接触 65.3%→12.0%。
- [x] 结论：不启动同配置 full CEM；下一步需要 phase/state-gated lower-body penalty。log: `workspace/core4d/log/145_E115_lowerbody_aware_contact_diagnostic_results.md`。

## E116 — Surface target + upright guard (2026-06-03)

- [x] 15 variants smoke 完成，0 release candidate。main `box021_029_p2` 中 `surface_upright_safety` 把 lower-body 8.0%→0.0%，但 physics contact 仅 46.7%（低于 E113 65.3%）且 deep penetration +4.0pp。
- [x] 结论：不启动 E116 full CEM。log: `workspace/core4d/log/146_E116_surface_target_upright_guard_results.md`。

## E117 — Phase-gated lowerbody penalty (2026-06-03)

- [x] 15 variants smoke 完成，0 release candidate。主 case `box021_029_p2` 三个变体均把 lower-body 8.0%→0.0%，但 physics contact 仅 8.0-12.0% 且 pelvis 大幅回退。
- [x] 结论：不启动 E117 full CEM。log: `workspace/core4d/log/147_E117_phase_gated_lowerbody_contact_results.md`。

## E118 — Carry corridor soft gate (2026-06-03)

- [x] 15 variants smoke 完成，0 release candidate。主 case `box021_029_p2` 中 ref variants 提升 contact 但 lower-body/deep penetration 回归，surface variant 修 lower-body 但 contact 崩到 9.3%。
- [x] 结论：不启动 E118 full CEM。log: `workspace/core4d/log/148_E118_carry_corridor_soft_gate_results.md`。

## E119 — Upright carry support guard (2026-06-03)

- [x] 9 variants smoke 完成，0 release candidate。主 case `box021_029_p2` 中 ref/bodyguard 49.3% contact 但 pelvis fail，bodygate 48.0% 且 pelvis fail，surface 18.7% 且 lower-body 21.3%。
- [x] companion `box021_035_p2/corridor_surface_pose_bodyguard` 有 78.2% contact、lower-body 0.0%、deep +0.0pp，但不是 main gate。
- [x] 结论：不启动 E119 full CEM。log: `workspace/core4d/log/149_E119_upright_carry_support_guard_results.md`。

## E120 — Object support decomposition (2026-06-03)

- [x] core reward 实现：`config.py` 增加 `hand_support_*` 与 `nonhand_support_penalty_*` 字段；`mjwp.py` 加入 hand near-zero support reward、non-hand support penalty。
- [x] 12 variants smoke 完成，0 release candidate。主 case `box021_029_p2` physics contact 仅 29.3/36.0/37.3%，lower-body 18.7/32.0/22.7%，non-hand support 30.7/36.0/36.0%，pelvis 均 fail。
- [x] 结论：不启动 E120 full CEM；保留 support decomposition metric/reward plumbing。log: `workspace/core4d/log/150_E120_object_support_decomposition_results.md`。

## E121 — Terminal carry gate (2026-06-03)

- [x] 新增 terminal carry semantic gate 字段（默认 disabled）并在 `get_terminal_reward` 中计算 pelvis/object-rot/non-hand support/hand-near terminal violation。
- [x] 12 variants smoke 完成，0 release candidate。主 case `box021_029_p2` physics contact 仅 32.0/32.0/28.0%；`terminal_hard_ref` gate valid 0.0%、fallback 92.0%，说明 hard terminal filtering starves。
- [x] 结论：不启动 E121 full CEM。log: `workspace/core4d/log/151_E121_terminal_carry_gate_results.md`。

## E122 — Snap warmstart carry prior (2026-06-03)

- [x] 新增 `warmstart_update_ctrl_from_qpos`，snap warmstart 在 `snap_mask` 帧同时更新 robot `ctrl_ref` 初始均值。
- [x] 12 variants smoke 完成，0 release candidate。主 case `box021_029_p2` physics contact 全部只有 12.0%。
- [x] 结论：不启动 E122 full CEM；下一步应转真正 staged optimizer/curriculum 或 RL hand-support objective。log: `workspace/core4d/log/152_E122_snap_warmstart_carry_prior_results.md`。

## E123 — Two-stage carry curriculum (2026-06-03)

- [x] 创建计划：`workspace/core4d/plan/132_E123_two_stage_carry_curriculum_plan.md`。Stage1→Stage2 bridge 成立（scene-act 42→source/freejoint 43 warmstart 桥接可用）。
- [x] 完整 12/12 Stage1/Stage2 smoke 完成，0 release candidate。全部决策为 `support_decomp_fail`。主 case Stage2 physics contact 只有 16.0% / 6.7%。
- [x] 结论：不启动 E123 full CEM；Stage1→Stage2 bridge 成立，但当前 Stage2 objective 不能保留 hand-supported carry。log: `workspace/core4d/log/153_E123_two_stage_carry_curriculum_results.md`。

## E124 — SBTO carry horizon (2026-06-03)

- [x] SBTO artifact contract 修正：SBTO 内部保存 sim-only `qpos/qvel/ctrl/time`，同时新增 `qpos_ref/qvel_ref/ctrl_ref/time_ref`；train script 合成为标准 evaluator-facing paired format。
- [x] 8 variants smoke 完成，0 release candidate。主 case physics contact 都只有 4.0%，object/pelvis 均 fail。
- [x] 结论：不启动 E124 full；现有 SBTO growing horizon 不足。log: `workspace/core4d/log/154_E124_sbto_carry_horizon_results.md`。
