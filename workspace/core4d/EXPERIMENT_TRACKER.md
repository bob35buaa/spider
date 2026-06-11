# CORE4D 动力学重定向实验跟踪器

## 实验总览

| Run | 日期 | Phase | 描述 | 状态 | Log |
|-----|------|-------|------|------|-----|
| E155 | 2026-06-11 | Phase 27 | hand_support 放手平滑过渡 3case×4 full | ✅ 12/12 complete；四方案 tracking 3/3；`decay` 最优: release_false3/5=0.033/0.054, inmaskC3/5=0.291/0.421 | [196](log/196_E155_release_smooth_transition_results.md) |
| E154 | 2026-06-11 | Phase 27 | 评测方法学修订:真实 3cm contact mask + body tracking(对固定运动学真值),重评 E152/E153 | ✅ 纯评测(不重训)。修复全序列+全1-mask 评测缺陷;新增 tracking 门控 success。**根因**:`core4d.py` one-mode 全1 mask 经 `io.py` 泄漏到 reward→优化器全程被奖励接触,放手普遍失败(连 b1 都不放)。重评:`(−0.010,0.10)` 存活 3/3,`(−0.005,0.05)` 降级 2/3;box004 三 combo 结尾弯腰被判 fail。**取代 192/193 接触结论**;详见 log 194 | [194](log/194_E154_masked_tracking_eval_results.md) |
| E153 | 2026-06-10 | Phase 27 | CEM hand gate 阈值扫（先解耦 max_violation 再扫 min_sdf×max_viol，gateA_b1 3case×3×2=18 grid） | ✅ full complete, found 3/3 strict sweet spot (−0.010,0.10): deep<−5mm −0.172/接触零损失/fallback修复; min_sdf 主导单调权衡, 0 fall；详见 log 193 + results/E153 | [193](log/193_E153_gate_threshold_sweep_results.md) |
| E152 | 2026-06-10 | Phase 27 | 轴1：手-物体物理穿透硬约束（CEM hand safety gate） | ✅ full complete, axis-1 gate cuts deep penetration; gateA_b1 box021 contact↑pen↓；详见 log 192 + results/E152 | [192](log/192_E152_axis1_hand_object_physics_gate_results.md) |
| E151 | 2026-06-10 | Phase 27 | Route-B hand surface contact reward | ❌ full complete, route B contact gain is penetration tradeoff；详见 log 191 + results/E151 | [191](log/191_E151_route_b_hand_surface_contact_reward_results.md) |
| E150 | 2026-06-09 | Phase 27 | Contact anchor `eef_offset` route-A sweep | ❌ full complete, route A not validated；详见 log 190 + results/E150 | [190](log/190_E150_contact_anchor_eef_offset_sweep_results.md) |
| E149 | 2026-06-09 | Phase 27 | E143 clean benchmark rubber hand eval | 🔬 eval-only clean benchmark complete；详见 log 189 + results/E149 | [189](log/189_E149_e143_clean_rubber_benchmark_results.md) |
| E148 | 2026-06-09 | Phase 27 | E143 24-case rubber hand collision extension | ⚠️ full complete, larger-set geometry tradeoff；详见 log 188 + results/E148 | [188](log/188_E148_e143_24case_rubber_hand_collision_results.md) |
| E147 | 2026-06-09 | Phase 27 | Rubber hand collision variant A/B full CEM | ⚠️ full complete, mixed geometry benefit；详见 log 187 + results/E147 | [187](log/187_E147_rubber_hand_collision_full_cem_results.md) |
| E145 | 2026-06-05 | Phase 26 | Full nonbox template release to CEM-ready | ✅ Phase1 CEM-ready complete, no CEM/RL launched；详见 log 184 + results/E145 | [185](log/185_E145_phase2_high_priority_full_cem_results.md) |
| E144 | 2026-06-05 | Phase 25 | Full nonbox raw_mask_ref_fk CEM pipeline | ❌ full CEM complete but no RL-ready rows；详见 log 173/174/175/176 + results/E144 | [183](log/183_E144_nonbox_template_policy_pipeline_update.md) |
| E143 | 2026-06-03 | Phase 24 | raw_mask_ref_fk 24-case OmniRetarget comparison | ❌ full complete: raw_mask_ref_fk 未超过 OmniRetarget；详见 log 172 + results/E143 | [189](log/189_E149_e143_clean_rubber_benchmark_results.md) |
| E142 | 2026-06-03 | Phase 24 | OmniRetarget contact exceedance audit | 🔬 audit complete: contact-aware improves Spider but does not yet exceed OmniRetarget；详见 log 171 + results/E142 | [171](log/171_E142_omniretarget_contact_exceedance_audit_results.md) |
| E140 | 2026-06-03 | Phase 24 | Holosoma semantic ref-mask reward readiness audit | 🔬 readiness audit complete, needs Box021 ref-mask reward config variant；详见 log 170 + results/E140 | [170](log/170_E140_holosoma_semantic_ref_mask_reward_readiness_results.md) |
| E139 | 2026-06-03 | Phase 24 | E138 semantic ref_object_contact env probe | ✅ partner-runtime semantic mask plumbing pass, PPO/RL still not launched；详见 log 169 + results/E139 | [169](log/169_E139_e138_semantic_ref_object_contact_env_probe_results.md) |
| E138 | 2026-06-03 | Phase 24 | E137 semantic contact converter preflight | ✅ converter + MotionLoader preflight complete, env/RL still not launched；详见 log 168 + results/E138 | [169](log/169_E139_e138_semantic_ref_object_contact_env_probe_results.md) |
| E137 | 2026-06-03 | Phase 24 | E107 semantic object-contact export preflight | ✅ qpos-style semantic contact exports complete, runtime/converter preflight required next；详见 log 167 + results/E137 | [168](log/168_E138_e137_semantic_contact_converter_preflight_results.md) |
| E136 | 2026-06-03 | Phase 24 | E135 semantic contact to Holosoma bridge audit | 🔬 bridge audit complete, E107 semantic export preflight allowed next；详见 log 166 + results/E136 | [166](log/166_E136_e135_semantic_contact_holosoma_bridge_results.md) |
| E135 | 2026-06-03 | Phase 24 | Box021 v3 S1 raw-contact remine | ✅ S1 raw-contact remine complete, bridge/export audit required next；详见 log 165 + results/E135 | [166](log/166_E136_e135_semantic_contact_holosoma_bridge_results.md) |
| E134 | 2026-06-03 | Phase 24 | Semantic contact to Holosoma bridge audit | 🔬 bridge audit complete, v3 raw-contact remine required；详见 log 164 + results/E134 | [164](log/164_E134_semantic_contact_holosoma_bridge_audit_results.md) |
| E133 | 2026-06-03 | Phase 24 | Holosoma ref_object_contact env probe | 🔬 env runtime plumbing pass, semantic/RL still blocked；详见 log 163 + results/E133 | [163](log/163_E133_holosoma_ref_object_contact_env_probe_results.md) |
| E132 | 2026-06-03 | Phase 24 | Holosoma MotionLoader object_contact runtime probe | 🔬 MotionLoader contract pass, semantic/RL still blocked；详见 log 162 + results/E132 | [162](log/162_E132_holosoma_motionloader_object_contact_probe_results.md) |
| E131 | 2026-06-03 | Phase 24 | Holosoma object_contact proxy contract | 🔬 proxy contract complete, semantic mask/RL still blocked；详见 log 161 + results/E131 | [161](log/161_E131_holosoma_object_contact_proxy_results.md) |
| E130 | 2026-06-03 | Phase 24 | Holosoma reward-side inspection | 🔬 reward-side inspection complete, RL still blocked；详见 log 160 + results/E130 | [160](log/160_E130_holosoma_reward_side_inspection_results.md) |
| E129 | 2026-06-03 | Phase 24 | Main carry-state constraint audit | 🔬 audit complete, main still blocked；详见 log 159 + results/E129 | [159](log/159_E129_main_carry_state_constraint_audit_results.md) |
| E128 | 2026-06-03 | Phase 24 | Holosoma runtime startup preflight | ✅ runtime startup preflight complete, smoke/training blocked；详见 log 158 + results/E128 | [158](log/158_E128_holosoma_runtime_startup_preflight_results.md) |
| E127 | 2026-06-03 | Phase 24 | Holosoma training-contract preflight | ✅ static contract complete, smoke/training blocked；详见 log 157 + results/E127 | [157](log/157_E127_holosoma_training_contract_preflight_results.md) |
| E126 | 2026-06-03 | Phase 24 | Holosoma fragment adapter preflight | ✅ adapter preflight complete, no training launch；详见 log 156 + results/E126 | [156](log/156_E126_holosoma_fragment_adapter_preflight_results.md) |
| E125 | 2026-06-03 | Phase 24 | RL hand-support preflight/export gate | ✅ preflight complete, no training launch；详见 log 155 + results/E125 | [155](log/155_E125_rl_hand_support_preflight_results.md) |
| E124 | 2026-06-03 | Phase 24 | SBTO carry-horizon diagnostic | 🔬 smoke complete, no full launch；详见 log 154 + results/E124 | [154](log/154_E124_sbto_carry_horizon_results.md) |
| E123 | 2026-06-03 | Phase 24 | Two-stage carry curriculum diagnostic | 🔬 smoke complete, no full launch；详见 log 153 + results/E123 | [153](log/153_E123_two_stage_carry_curriculum_results.md) |
| E122 | 2026-06-03 | Phase 24 | Snap warmstart carry prior diagnostic | 🔬 smoke complete, no full launch；详见 log 152 + results/E122 | [152](log/152_E122_snap_warmstart_carry_prior_results.md) |
| E121 | 2026-06-03 | Phase 24 | Terminal carry gate diagnostic | 🔬 smoke complete, no full launch；详见 log 151 + results/E121 | [151](log/151_E121_terminal_carry_gate_results.md) |
| E120 | 2026-06-03 | Phase 24 | Object-support decomposition diagnostic | 🔬 smoke complete, no full launch；详见 log 150 + results/E120 | [150](log/150_E120_object_support_decomposition_results.md) |
| E119 | 2026-06-03 | Phase 24 | Upright carry support-guard diagnostic | 🔬 smoke complete, no full launch；详见 log 149 + results/E119 | [149](log/149_E119_upright_carry_support_guard_results.md) |
| E118 | 2026-06-03 | Phase 24 | Carry-corridor soft-gate diagnostic | 🔬 smoke complete, no full launch；详见 log 148 + results/E118 | [148](log/148_E118_carry_corridor_soft_gate_results.md) |
| E117 | 2026-06-03 | Phase 24 | Phase/state-gated lower-body contact diagnostic | 🔬 smoke complete, no full launch；详见 log 147 + results/E117 | [147](log/147_E117_phase_gated_lowerbody_contact_results.md) |
| E116 | 2026-06-03 | Phase 24 | Surface target + upright guard diagnostic | 🔬 smoke complete, no full launch；详见 log 146 + results/E116 | [146](log/146_E116_surface_target_upright_guard_results.md) |
| E115 | 2026-06-03 | Phase 24 | Lower-body-aware contact diagnostic | 🔬 smoke complete, no full launch；详见 log 145 + results/E115 | [145](log/145_E115_lowerbody_aware_contact_diagnostic_results.md) |
| E114 | 2026-06-03 | Phase 24 | Contact alignment RL handoff gate | ✅ no RL export by gate；详见 log 144 + results/E114 | [144](log/144_E114_contact_alignment_rl_handoff_gate_results.md) |
| E113 | 2026-06-03 | Phase 24 | Contact-aware expanded workset | ⚠️ full complete, 0 release candidates；详见 log 143 + results/E113 | [143](log/143_E113_contact_aware_expanded_workset_results.md) |
| E112 | 2026-06-03 | Phase 24 | Contact-aware CEM ablation | ✅ Phase A full complete；详见 log 142 + results/E112 | [142](log/142_E112_contact_aware_cem_ablation_results.md) |
| E111 | 2026-06-02 | Phase 24 | data_construction_v3 contact evidence chain | ✅ contact evidence smoke complete；详见 log 141 + results/E111 | [141](log/141_E111_contact_evidence_chain_results.md) |
| E110 | 2026-06-02 | Phase 24 | Contact metric audit after E109 | ✅ contact audit complete；详见 log 140 + results/E110 | [140](log/140_E110_contact_metric_audit_results.md) |
| E109 | 2026-06-02 | Phase 23 | Spider vs OmniRetarget unified replay eval | ✅ unified replay / 24-case / 20-case xlsx complete；详见 log 139 + scripts/eval_omni_vs_spider | [139](log/139_E109_spider_vs_omniretarget_fair_eval_results.md) |
| E108 | 2026-06-02 | Phase 22 | Non-box to RL data pipeline through RL smoke | ✅ completed through RL smoke；详见 log 136/138 + plan 117 | [138](log/138_E108_nonbox_cem_and_rl_handoff.md) |
| E106 | 2026-06-01 | Phase 21 | Box026 30-candidate clean ref-FK batch CEM completed | ✅ completed；详见 log 133 + results/E106 | [133](log/133_E106_box026_30candidate_ref_fk_batch_results.md) |
| E105 | 2026-06-01 | Phase 21 | Box026 clean-scene full CEM rerun after E103 template fix | ⚠️ 4/6 upper-body WORK, 0/6 lower-body strict；详见 log 132 + results/E105 | [132](log/132_E105_box026_clean_scene_full_cem_rerun_results.md) |
| E104 | 2026-05-31 | Phase 21 | D002 multi-threshold raw-contact remine | ✅ data gate fixed; next E105 select typical/diversity；详见 log 131 + results/E104 | [131](log/131_E104_d002_multithreshold_remine_results.md) |
| E103 | 2026-05-31 | Phase 21 | scene rebuild + inertial audit + validity reset (data foundation repair) | ✅ data foundation repaired; no E103 CEM；详见 log 127/128/129/130 + results/E103 | [130](log/130_E103_rebuilt_target_regeneration_results.md) |
| E102 | 2026-05-31 | Phase 21 | failure taxonomy + Box022/data expansion + RL-ready handoff (exp_diagnostic_v2 … | ⚠️/✅ data expansion failed; handoff PARTIAL；详见 log 126 + results/E102 | [126](log/126_E102_failure_taxonomy_box022_remine_handoff_results.md) |
| E101 | 2026-05-31 | Phase 21 | box021 D003 + box004 full CEM with E100 fingertip target (exp_diagnostic_v2 Sta… | ❌/✅ stop-loss；详见 log 125 + results/E101 | [125](log/125_E101_box021_d003_rerun_with_new_target_results.md) |
| E100 | 2026-05-30 | Phase 21 | contact target 重做 + 干净 A/B (exp_diagnostic_v2 Stage 2) | ✅/⚠️ 详见 log 124 + plan 107 | [124](log/124_E100_fingertip_target_and_clean_ab_results.md) |
| E099 | 2026-05-30 | Phase 21 | 接触语义信息流补全（exp_diagnostic_v2 Stage 1） | ✅ 详见 log 123 + plan 106 | [99](log/99_E078_3cm_per_eef_mask_results.md) |
| E098 | 2026-05-30 | Phase 21 | 诊断基础设施（exp_diagnostic_v2 Stage 0） | ✅ 详见 log 122 + plan 105 | [98](log/98_E077_3cm_contact_mask_and_person2_results.md) |
| E097 | 2026-05-29 | Phase 20 | feature-based data_construction_v2 candidate refresh + visual correction | ✅/🔬 详见 log 120/121 + plan 104 | [97](log/97_E076_contact_source_audit.md) |
| E096b | 2026-05-29 | Phase 20 | box004 mask-on full CEM rerun | ✅ 详见 log 119 + plan 103 | [96](log/96_E075_limited_hold_contact_results.md) |
| E096 | 2026-05-29 | Phase 20 | box004 first-batch contact semantics + full CEM | ✅/🔬 详见 log 118 + plan 102 | [96](log/96_E075_limited_hold_contact_results.md) |
| E095 | 2026-05-29 | Phase 20 | Worklike data mining after Box026 failures | ✅/🔬 详见 log 117 + plan 101 | [95](log/95_E074_remote_hold_contact_results.md) |
| E094 | 2026-05-29 | Phase 20 | G1-handbox-aware adaptive-support target projection + full CEM | ⚠️/✅ 详见 log 116 + plan 100 | [94](log/94_E074_preflight_base_palm_normal_analysis.md) |
| E093 | 2026-05-29 | Phase 20 | Contact target geometry audit before CEM/RL | ✅/🔬 详见 log 115 + plan 99 | [93](log/93_E073_contact_target_offset_consistency_results.md) |
| E092 | 2026-05-29 | Phase 20 | Three-case SPIDER dynamic + direct OmniRetarget comparison corrected by full CEM | ⚠️/🔬 log 114 需修订；full 结果在 `workspace/core4d/results/E092/spider_dyn/full/` | [92](log/92_E072_post2_hold_place_diagnosis_results.md) |
| E091 | 2026-05-29 | Phase 20 | Holosoma data_construction_v2 medium-box discovery results | ✅/🔬 详见 log 113 + plan 97 | [91](log/91_E071_scene_act_ctrl_mapping_fix_results.md) |
| E090 | 2026-05-28 | Phase 19 | H2-first fingertip replacement ablation + true pre-IK topface repair + SPIDER s… | ❌/🔬 详见 log 112 + plan 96 | [90](log/90_E070_mjwarp_ref_control_parity_results.md) |
| E089 | 2026-05-28 | Phase 19 | G1-Feasibility gate 验证 (A 路: box021_person1 SPIDER full CEM; B 路: D003 box021 1… | ✅ 详见 log 111 + plan 95 | [89](log/89_E069_first_tick_warmup_results.md) |
| E088 | 2026-05-28 | Phase 18 | Hard safety gate + absolute object clearance | ❌ 详见 log 110；plan 94 | [88](log/88_E068_mjwp_init_drift_results.md) |
| E087 | 2026-05-28 | Phase 18 | Box021 mass + reward breakdown audit | ❌ 详见 log 109；plan 93 | [87](log/87_R4_HDMI_diagnosis_init_pose_bug.md) |
| E086 | 2026-05-28 | Phase 18 | raw-target CEM failure iteration | ❌ 详见 log 108；plan 92 | [86](log/86_E067_results_synthesis_pause.md) |
| E085 | 2026-05-28 | Phase 18 | raw contact target repair + gate | ❌ 详见 log 108；plan 91 | [85](log/85_E066_results_actuator_port.md) |
| E084-audit | 2026-05-28 | Phase 18 | E084 contact target 语义核查 | 🔬 详见 log 107 | [84](log/84_E065_results_task_obj_ablation.md) |
| E084 | 2026-05-28 | Phase 18 | Box021 constraint groups main gate | ❌ 详见 log 106；plan 90 | [84](log/84_E065_results_task_obj_ablation.md) |
| E083 | 2026-05-28 | Phase 18 | upper-body-object collision pairs 验证 | ❌ 详见 log 105；plan 89 | [83](log/83_E065_plan_task_obj_ablation.md) |
| E082 | 2026-05-27 | Phase 18 | D003 Box021 三 case 回到 E081 leg/foot-object collision 路线 + 上半身穿模诊断 | ❌ 详见 log 103/104 | [82](log/82_pre_contact_body_tracking_diagnosis.md) |
| E081 | 2026-05-16 | Phase 18 | leg/foot-object collision 派生 scene 验证 | ⚠️ 详见 log 102 | [81](log/81_E064_tier2_threshold_raise.md) |
| E080 | 2026-05-15 | Phase 18 | box025 大物体边界复查 | ⚠️ 详见 log 101 | [80](log/80_E063_tier1_stability_taskobj.md) |
| E079 | 2026-05-15 | Phase 18 | CORE4D 10+ 高接触质量 case 泛化验证 | ⚠️ 详见 log 100 | [79](log/79_E062_box023_diagnosis_optimization_candidates.md) |
| E075 | 2026-05-14 | Phase 18 | 限时 hold_contact 组合验证 | ⚠️ 详见 log 96 | [96](log/96_E075_limited_hold_contact_results.md) |
| E074 | 2026-05-14 | Phase 18 | post-2s hold/contact 首轮远程并行 | ⚠️ 详见 log 95 | [95](log/95_E074_remote_hold_contact_results.md) |
| E074 preflight | 2026-05-14 | Phase 18 | E074 base/palm normal 前置分析 | 📋 详见 log 94 | [95](log/95_E074_remote_hold_contact_results.md) |
| E073 | 2026-05-14 | Phase 18 | contact target eef_offset 口径修正 | ⚠️ 详见 log 93 | [93](log/93_E073_contact_target_offset_consistency_results.md) |
| E072 | 2026-05-14 | Phase 18 | box023 post-2s hold/place failure 诊断 | ✅ 详见 log 92 | [92](log/92_E072_post2_hold_place_diagnosis_results.md) |
| E070 | 2026-05-14 | Phase 18 | MJWarp ref-control parity 诊断定位根因 | ✅ 详见 log 90 | [90](log/90_E070_mjwarp_ref_control_parity_results.md) |
| E070 plan | 2026-05-14 | Phase 18 | MJWarp ref-control commit parity 诊断计划 | 📋 待确认 | [90](log/90_E070_mjwarp_ref_control_parity_results.md) |
| E069 | 2026-05-14 | Phase 18 | First-tick ref-control warmup 验证失败 | ❌ 详见 log 89 | [89](log/89_E069_first_tick_warmup_results.md) |
| E068 | 2026-05-14 | Phase 18 | MJWP init drift 诊断修正 | 诊断完成 | [88](log/88_E068_mjwp_init_drift_results.md) |
| E001 | 2026-04-30 | Phase 0 | 数据管线 | 通过 | [80](log/80_E063_tier1_stability_taskobj.md) |
| E002 | 2026-04-30 | Phase 1 | SPIDER MJWP 无引导 (Box025 p1) | 完成 | [98](log/98_E077_3cm_contact_mask_and_person2_results.md) |
| E003 | 2026-04-30 | Phase 1 | SPIDER MJWP 有引导 (Box025 p1) | 完成 | [99](log/99_E078_3cm_per_eef_mask_results.md) |
| E004 | 2026-04-30 | Phase 1 | 强增益 kp=100/1000, decay=1 | 完成 | [87](log/87_R4_HDMI_diagnosis_init_pose_bug.md) |
| E005 | 2026-04-30 | Phase 4 | 混合轨迹导出 | 通过 | [04](log/04_E005_holosoma_export_results.md) |
| E006 | 2026-05-01 | Phase 1 | 视频证实箱子未离地 | 虚假突破(更正) | [05](log/05_E006_forearm_contact_results.md) |
| E007 | 2026-05-01 | Phase 1 | 路径Y物理对齐 | 失败 | [06](log/06_E007_path_Y_results.md) |
| E008 | 2026-05-01 | Phase 1 | 视频驱动诊断 | 失败 | [07](log/07_E008_real_lift_diagnosis_results.md) |
| E009 | 2026-05-01 | Phase 1 | 几何错位:双人对夹±x端 | 失败(几何洞察) | [08](log/08_E009_person2_support_results.md) |
| — | 2026-05-02 | Phase 2 | 规划 | 规划完成 |  |
| E010 | 2026-05-02 | Phase 2 | G1运动学可行性确认 | **通过** | [10](log/10_E011_mocap_partner_results.md) |
| E011 | 2026-05-02 | Phase 2 | Mocap Partner协作 | 部分成功 | [11](log/11_E012_export_results.md) |
| E012 | 2026-05-02 | Phase 2 | 导出Holosoma格式+partner数据 | **通过** | [53](log/53_E044_E047_phase12_explore_results.md) |
| — | 2026-05-03 | Phase 3 | 规划 | 规划完成 |  |
| E013 | 2026-05-03 | Phase 3 | 但高方差 | 部分成功 | [13](log/13_E014_larger_partner_results.md) |
| E014 | 2026-05-03 | Phase 3 | 增大Partner碰撞体 | 失败 | [14](log/14_E015_bucket005_results.md) |
| — | 2026-05-03 | Phase 4 | 规划 | 规划完成 |  |
| E015 | 2026-05-03 | Phase 4 | 修复scene_name bug | 完成 | [15](log/15_E016_dual_robot_results.md) |
| E016 | 2026-05-04 | Phase 4 | 双机器人Gibbs CEM | ❌ connect假象 | [16](log/16_E017_dual_connect_results.md) |
| E017 | 2026-05-05 | Phase 4 | 双机器人 Soft 2-Connect | ❌ connect假象 | [17](log/17_E018_taskspace_results.md) |
| E018 | 2026-05-06 | Phase 4 | Task-Space奖励(DynaRetarget)+Interaction(Harmanoid) | ❌ connect假象 | [18](log/18_E020_multicase_diagnosis_results.md) |
| E020 | 2026-05-06 | Phase 5 | 多Case诊断(5物体×2模式) | 诊断完成 | [20](log/20_E022_anchored_objrew_results.md) |
| E021 | 2026-05-06 | Phase 5 | IK可达性+Anchor | **突破** | [21](log/21_E023_fullanchor_results.md) |
| E022 | 2026-05-06 | Phase 5 | Anchored+ObjRew | 部分成功 | [22](log/22_E024_partner_force_results.md) |
| E023 | 2026-05-06 | Phase 5 | Full Anchor(XY+Yaw) | 结构性结论 | [23](log/23_E025_hand_approach_results.md) |
| E024 | 2026-05-07 | Phase 5 | Partner Force(50-90%grav) | 失败(结构性) | [24](log/24_E026_sustained_contact_results.md) |
| E025 | 2026-05-07 | Phase 6 | Hand Approach Reward | 部分成功(技术有效, 目标未达) | [26](log/26_E025_E027_reassessment.md) |
| E026 | 2026-05-07 | Phase 6 | Sustained Contact (4096samp/24iter) | 同上 | [26](log/26_E025_E027_reassessment.md) |
| E027 | 2026-05-07 | Phase 6 | Partner Force Sweep + desk005 | 同上 | [35](log/35_E027d2_bodyframe_fix_results.md) |
| E028 | 2026-05-07 | Phase 7 | 阻尼弹簧 4Case全覆盖 | C6 FAIL (视频不像搬运) | [28](log/28_E029_quasikin_results.md) |
| E029 | 2026-05-07 | Phase 7 | Quasi-Kinematic(kp=100) | FAIL (根本性限制) | [29](log/29_E029_actuator_kinoverride_results.md) |
| E029-act | 2026-05-07 | Phase 7 | PD Actuator+Kin Override | 方向明确(待debug torque) | [29](log/29_E029_actuator_kinoverride_results.md) |
| E030 | 2026-05-07 | Phase 7 | Orientation Debug+Hybrid Export | 质量不足 | [30](log/30_E030_orientation_torque_results.md) |
| E031 | 2026-05-07 | Phase 8 | 双机器人Connect泛化4case | FAIL (结构性) | [31](log/31_E031_dual_generalization_results.md) |
| E027b | 2026-05-07 | Phase 8 | Object PD Override(scene_act+grav_comp+relative_euler) | **desk005成功** | [35](log/35_E027d2_bodyframe_fix_results.md) |
| E027c | 2026-05-07 | Phase 8 | Contact Guidance(OMOMO方案)on CORE4D | FAIL (simulator限制) | [35](log/35_E027d2_bodyframe_fix_results.md) |
| E027d | 2026-05-08 | Phase 8 | HDMI Physics+Debug | 调试中 | [35](log/35_E027d2_bodyframe_fix_results.md) |
| E027d2 | 2026-05-08 | Phase 8 | Body-Frame Fix+Commit Gain Restore | body tracking有效 | [35](log/35_E027d2_bodyframe_fix_results.md) |
| E032a | 2026-05-08 | Phase 9 | Hand Approach+Reward Sweep | 完成 | [37](log/37_E032_case_difficulty_analysis.md) |
| E033 | 2026-05-08 | Phase 9 | desk005 σ sweep+CEM budget | **stable目标达成** | [38](log/38_E033_desk005_stability_results.md) |
| E034 | 2026-05-08 | Phase 10 | HDMI-Style Reward: Stability Penalty | 改善但未解决 | [40](log/40_E034_hdmi_reward_results.md) |
| E035 | 2026-05-08 | Phase 10 | Local-Frame Body Tracking | body tracking突破 | [41](log/41_E035_local_frame_results.md) |
| E036 | 2026-05-08 | Phase 10 | 关闭hand_approach | body tracking突破 | [42](log/42_E036_hdmi_reward_alignment_results.md) |
| E037-E039 | 2026-05-08~09 | Phase 11 | Contact Reward系列 | Bug发现 | [45](log/45_E037c_all_cases_results.md) |
| E039b | 2026-05-09 | Phase 11 | Config Bug Fix+Rotated SDF | 突破+新问题 | [48](log/48_E039b_config_bug_fix_results.md) |
| E040 | 2026-05-09 | Phase 11 | Dynamic Per-Frame Target | ❌ 不自然行为未消除 | [49](log/49_E040_dynamic_target_results.md) |
| E041 | 2026-05-09 | Phase 11 | Orientation Reward(乘法门控) | ⚠️ 方向正确但约束过严 | [77](log/77_E041c_box025_real_issue_arm_reach.md) |
| E041c | 2026-05-09 | Phase 11 | Additive Ori(w=0.3)最佳变体 | ★ CEM最佳(非搬运) | [77](log/77_E041c_box025_real_issue_arm_reach.md) |
| E042 | 2026-05-10 | Phase 11 | Wrist Freeze(零化手腕噪声) | ❌ 有害 | [51](log/51_E042_wrist_freeze_results.md) |
| E043 | 2026-05-10 | Phase 11 | 原始OmniRetarget Ref对比 | Phase4更优 | [52](log/52_E043_original_ref_results.md) |
| E044a | 2026-05-10 | Phase 12 | Wrist Weight=2.0 | ❌ stability退化 | [53](log/53_E044_E047_phase12_explore_results.md) |
| E045 | 2026-05-10 | Phase 12 | Sigma Sweep(0.3/0.15) | ❌ 无效 | [45](log/45_E037c_all_cases_results.md) |
| E047a | 2026-05-10 | Phase 12 | SBTO对齐DynaRetarget | ❌❌ 失败 | [53](log/53_E044_E047_phase12_explore_results.md) |
| E044b | 2026-05-10 | Phase 12 | Wrist Weight=3.0 | ❌ contact退化 | [53](log/53_E044_E047_phase12_explore_results.md) |
| E047b | 2026-05-10 | Phase 12 | SBTO放松参数(α_μ=0.5,σ_min=0.03) | ❌ tracking差 | [53](log/53_E044_E047_phase12_explore_results.md) |
| — | 2026-05-11 | Bug Fix | 碰撞盒模板Bug修复 | 修复完成 |  |
| E048 | 2026-05-11 | Phase 13 | 碰撞盒修复后Baseline+HDMI对比 | ⚠️ 指标不可信 | [61](log/61_E048_E052_visual_reevaluation.md) |
| E049 | 2026-05-11 | Phase 14 | HDMI优化移植失败+eval修正 | ❌ 移植失败 | [57](log/57_E048_E049_eval_correction.md) |
| E050 | 2026-05-11 | Phase 14 | Euler Convention Fix尝试 | ❌ gimbal lock | [58](log/58_E050_hdmi_euler_analysis.md) |
| E051 | 2026-05-11 | Phase 15 | HDMI Scene物理配置全面诊断 | 方向明确 | [59](log/59_E051_hdmi_scene_diagnosis.md) |
| E052a | 2026-05-11 | Phase 15 | Suitcase模板+旧euler | ❌ 单修scene不够 | [61](log/61_E048_E052_visual_reevaluation.md) |
| E052c | 2026-05-12 | Phase 15 | Suitcase模板+正确euler(XZY) | ❌❌ 全矩阵失败 | [61](log/61_E048_E052_visual_reevaluation.md) |
| E053 | 2026-05-12 | Phase 16 | 碰撞盒Margin Sweep(0.90/0.95/1.00×3case) | ⚠️ per-case策略 | [63](log/63_E001_E053_stage_summary.md) |
| — | 2026-05-12 | Phase 17 | 路线图 | 规划完成 |  |
| E054 | 2026-05-12 | Phase 17 | Case Tier + Mocap质量分析(21 case, v3 detector + 视频核实主导手) | **✅ 收口完成** | [64](log/64_E054_case_tier_analysis_results.md) |
| E055 | 2026-05-12 | Phase 17 | box023 Hand-Snap Warmstart (Path B 首验证, 无CEM) | **✅ Path B 几何验证 + 诊断工具** | [65](log/65_E055_box023_hand_snap_results.md) |
| E056 | 2026-05-12 | Phase 17 | 多 case Hand-Face 诊断 (6 B+C case) | **✅ E057 决策完成 (路线 A: bucket005_s2)** | [66](log/66_E056_multi_case_diagnosis_results.md) |
| E057 | 2026-05-13 | Phase 17 | bucket005_s2 Hand-Snap (Path B 第 2 case, 对侧握姿) | **✅ 6/6 通过 (Path B 几何验证 + face guard)** | [67](log/67_E057_bucket005_s2_hand_snap_results.md) |
| E058 | 2026-05-13 | Phase 17 | Path B-CEM 首跑 (bucket005_s2 baseline vs warm) | ❌ 2/6 (流水线 OK 内容失败) | [68](log/68_E058_bucket005_s2_warmstart_cem_results.md) |
| E059 | 2026-05-13 | Phase 17 | Path B-CEM 第 2 case (box023, 区分 E058 失败原因) | ❌ 1.5/6 (与 E058 同情景, baseline 是真正 bug, 详见 audit log 70) | [69](log/69_E059_box023_warmstart_cem_results.md) |
| Audit | 2026-05-13 | Phase 17 | Pre-E060 全面审查 (E041c 数据层 + reward task-specific) | 📋 详见 log 70, **指导 E060 必须先修数据层再做 reward 消融** |  |
| E060.0 | 2026-05-13 | Phase 17 | 数据层修复后 baseline (E041c, no warmstart) on box023 + bucket005_s2 | ❌ 1/6 (流水线 OK, 数据层修复无效) | [74](log/74_box025_3box_regression_and_E060_invalidation.md) |
| E060.1 | 2026-05-13 | Phase 17 | `contact_hdmi_ori_weight=0.0` ablation on box023 + bucket005_s2 | ❌ 0/4 (mixed signal, **结论需 sphere baseline 验证后重新评估**) | [74](log/74_box025_3box_regression_and_E060_invalidation.md) |
| E060.2 | 2026-05-13 | Phase 17 | case-correct palm_normal ablation | ❌❌ 0/3 (catastrophic, **E060 reward ablation 全部暂停**) | [74](log/74_box025_3box_regression_and_E060_invalidation.md) |
| Audit | 2026-05-13 | Phase 17 | 🚨 Box025 3-box hand regression 发现 + E060 phase invalidation | 🚨 详见 log 74, **E060 全部暂停, 必做 sphere 验证** |  |
| E061 | 2026-05-13 | Phase 18 | Sphere baseline verification (D 方案) | ✅ 5/5 通过 (诊断决定性确认) | [75](log/75_E061_sphere_baseline_verify.md) |
| Strategic | 2026-05-13 | Phase 18 | 🚨 E041c reward stack 完全是 box025 sphere 的过拟合 — A/B 决策框架失效, 转向 X1+X2 reward 泛化方向 | 🚨 详见 log 76, **A/B 框架失效, 转 X1+X2** |  |
| Correction | 2026-05-13 | Phase 18 | 🔄 E041c box025 真问题不是反关节而是臂展物理硬限制 — 修正"3-box 解决反关节"的错误论证, X1+X2 改为基于 sphere | 🔄 详见 log 77, **X1+X2 改为基于 sphere, X1 优先** |  |
| E062 | 2026-05-13 | Phase 18 | ⚠️ X1 Auto Palm Normal on Sphere — Mixed result (box025 PASS self-consistency, … | ⚠️ 2.5/5 (box025 ✓, box023 mixed) | [79](log/79_E062_box023_diagnosis_optimization_candidates.md) |
| R4-diag | 2026-05-14 | Phase 18 | 🚨 HDMI vs MJWP diagnosis — INIT POSE BUG smoking gun (sim t=0 pelvis 偏 ref 22° … | 🚨 详见 log 87 |  |
| E067 | 2026-05-14 | Phase 18 | ❌❌ Port HDMI body partition (lower 12→6, upper 17→6) — CATASTROPHIC FAIL, sim 做… | ❌❌ 详见 log 86, **PAUSE 等用户** | [86](log/86_E067_results_synthesis_pause.md) |
| E066 | 2026-05-14 | Phase 18 | ❌ Port HDMI object actuator gains (kp 500→20, kp_rot 50→0.3, decay 1.0→0.85) — … | ❌ 详见 log 85, 进 R3 E067 | [85](log/85_E066_results_actuator_port.md) |
| E065 | 2026-05-14 | Phase 18 | ❌ task_obj_rew form ablation on box023 — FAIL, 但揭示真元凶不是 reward form 而是 actuator… | ❌ 详见 log 84, 进 R2 E066 | [84](log/84_E065_results_task_obj_ablation.md) |
| E065 plan | 2026-05-14 | Phase 18 | 📋 task_obj_rew form ablation 设计 (A=drop / D=HDMI exp form) — 待跑 | 📋 详见 log 83, 待跑 | [84](log/84_E065_results_task_obj_ablation.md) |
| Update | 2026-05-14 | Phase 18 | 🎯 HDMI workflow 对照修正 E065 方向 | 🎯 详见 log 82 §10, **建议优先 E065-A** |  |
| Diagnosis | 2026-05-14 | Phase 18 | 🚨 Pre-contact body tracking failure 诊断 — box023 在 t=0.7s sim 单脚悬空 60cm, box025 … | 🚨 详见 log 82, 等用户决策 |  |
| E064 | 2026-05-14 | Phase 18 | ❌ Tier 2 + threshold raise on box023 (root_σ=0.3 + contact_gain=3.0 + thresh=0.… | ❌ 2/5 box023 + 2/3 box025 (FAIL, 进 E065 X5) | [81](log/81_E064_tier2_threshold_raise.md) |
| E063 | 2026-05-14 | Phase 18 | ⚠️ Tier 1 reward fix on box023 (re-enable stability_penalty=1.0 + reduce task_o… | ⚠️ 2/5 box023 + 3/3 box025 (FAIL 整体, 进 Tier 2) | [80](log/80_E063_tier1_stability_taskobj.md) |
| Diagnosis | 2026-05-14 | Phase 18 | 🔬 E062 box023 深度诊断 + 优化候选 — 真相: sim 完成了 task (final object pos 仅 9.8cm 偏离 ref),… | 📋 详见 log 79, 推荐 E063 = T1-A + T1-B |  |
