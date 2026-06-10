# E125-E145 Progress Archive

## E137 — E107 semantic object-contact export preflight (2026-06-03)

- [x] 按 `experiment-planning-zh` 复核 contact improvement plan、E136 log、tracker/progress；确认 E137 应只写 E107 semantic `object_contact` candidate exports，不启动 CEM/PPO/Holosoma 训练。
- [x] 创建计划：`workspace/core4d/plan/146_E137_e107_semantic_object_contact_export_preflight_plan.md`。
- [x] 新增 E137 exporter 和固定 eval wrapper。
- [x] 运行 E137 export preflight：输出 8 个 isolated qpos-style E107 semantic `object_contact` NPZ；retargeted-untrimmed `4`、trimmed `4`。
- [x] 关键细节：默认 `object_contact` 使用 3cm，另保留 `object_contact_3cm/object_contact_5cm`；所有输出 qpos frame count 与 `(T,2)` bool contact shape 匹配。
- [x] 写结果 log：`workspace/core4d/log/167_E137_e107_semantic_object_contact_export_preflight_results.md`。

## E136 — E135 semantic contact to Holosoma bridge audit (2026-06-03)

- [x] 创建计划：`workspace/core4d/plan/145_E136_e135_semantic_contact_holosoma_bridge_plan.md`。
- [x] 运行 E136 audit：输出 24 行 audit；E107 semantic bridge candidate rows `16`，其中 direct raw-axis `8`、trim-window slice `8`；E126/E131 semantic-ready rows `0`，fragment mapping blocked rows `8`。
- [x] 关键细节：E107 trimmed windows 均通过 raw-to-export slice proof，但 E136 只做 audit，不写 semantic `object_contact`。
- [x] 写结果 log：`workspace/core4d/log/166_E136_e135_semantic_contact_holosoma_bridge_results.md`。

## E135 — Box021 v3 S1 raw-contact remine (2026-06-03)

- [x] 创建计划：`workspace/core4d/plan/144_E135_box021_v3_s1_raw_contact_remine_plan.md`。
- [x] 运行 E135 eval：bounded inventory `4` 行、raw sequences `2` 条；3cm/5cm 均为 `4/4 raw_contact_pass`；proxy NPZ 为 `20231011_035_box021 (182x2x2)` 和 `20231018_029_box021 (134x2x2)`，均含 3cm/5cm masks 与 object-local centroids。
- [x] isolated combined registry `4` 行，四个 raw case-person 的 `raw_contact_3cm_status=pass/raw_contact_5cm_status=pass`。
- [x] 写结果 log：`workspace/core4d/log/165_E135_box021_v3_s1_raw_contact_remine_results.md`。

## E134 — Semantic contact to Holosoma bridge audit (2026-06-03)

- [x] 创建计划：`workspace/core4d/plan/143_E134_semantic_contact_holosoma_bridge_audit_plan.md`。
- [x] 运行 E134 audit：7 行 audit，legacy mask available rows=7，`v3_semantic_contact_ready_rows=0`，`semantic_bridge_candidate_rows=0`。
- [x] 关键细节：E107 trimmed 的 `box021_035_p2` 和 `box021_029_p2` 有 legacy `spider_contact_mask_3cm` exact axis match，但 v3 registry not_run。
- [x] 写结果 log：`workspace/core4d/log/164_E134_semantic_contact_holosoma_bridge_audit_results.md`。

## E133 — Holosoma ref_object_contact env probe (2026-06-03)

- [x] 运行固定入口：输出 summary `status=pass`。2/2 startup pass；两行 `has_object/has_partner/has_object_contact=true`；full motion contact total p1/p2 = `9/160`；p2 runtime `ref_contact_total=74`、`ref_either_active=54`，证明 E131 proxy mask 能进入 `motion_command.ref_object_contact`。
- [x] 写结果 log：`workspace/core4d/log/163_E133_holosoma_ref_object_contact_env_probe_results.md`。

## E132 — Holosoma MotionLoader object_contact runtime probe (2026-06-03)

- [x] 运行 E132：4/4 runtime contract pass；E126 两行 `has_object_contact=false`；E131 两行 `has_object_contact=true` 且 shape `214x2`；`structural_ref_mask_runtime_ready_rows=2`。
- [x] 写结果 log：`workspace/core4d/log/162_E132_holosoma_motionloader_object_contact_probe_results.md`。

## E131 — Holosoma object_contact proxy contract (2026-06-03)

- [x] 运行 E131：2/2 export shape pass，dtype bool。`structural_ref_mask_ready_rows=2`、`semantic_ref_mask_ready_rows=0`、`rl_ready_rows=0`。
- [x] 写结果 log：`workspace/core4d/log/161_E131_holosoma_object_contact_proxy_results.md`。

## E130 — Holosoma reward-side inspection (2026-06-03)

- [x] 运行 E130 inspection：17 个 reward/config rows、8 个 motion proxy rows、2 个 paired exports audited；`ref_mask_reward_allowed_rows=0`、`rl_ready_rows=0`。
- [x] 关键边界：E126 exports 有 object pose 和 partner hand pose，但无 `object_contact`；actor-side 无 Holosoma handbox body，proxy 使用 `left/right_rubber_hand_link` fallback。
- [x] 写结果 log：`workspace/core4d/log/160_E130_holosoma_reward_side_inspection_results.md`。

## E129 — main carry-state constraint audit (2026-06-03)

- [x] E129 本地审计聚合 E113 full + E119-E124 smoke 中 `box021_029_p2` 的 17 个 test rows 和 per-frame timeseries。
- [x] 结果：0 strict gate row、`rl_ready_rows=0`。最佳全局 row 是 E113 `hold_band`：physics contact 65.3%、pelvis 0.646m，但 lower-body 8.0% 且 FAIL。
- [x] 结论：main 仍 blocked；下一步应在"stage-local constrained teacher/smoke"或"Holosoma reward-side inspection without PPO"之间推进。log: `workspace/core4d/log/159_E129_main_carry_state_constraint_audit_results.md`。

## E128 — Holosoma runtime startup preflight (2026-06-03)

- [x] E128 两个 E126 paired fragment export 均 return code 0、无 timeout、无 exception signature；2/2 startup pass、`rl_ready_rows=0`。
- [x] 结论：fragment inspection 的 runtime startup/config gap 关闭，但 source rows 仍是 `FRAGMENT_HOLDOUT_ONLY`。log: `workspace/core4d/log/158_E128_holosoma_runtime_startup_preflight_results.md`。

## E127 — Holosoma training-contract preflight (2026-06-03)

- [x] E127 验证 E126 两个 paired exports：2/2 structural pass、frames 214、fps 50、NaN/nonfinite 0、`rl_smoke_allowed_rows=0`、`rl_ready_rows=0`。
- [x] log: `workspace/core4d/log/157_E127_holosoma_training_contract_preflight_results.md`。

## E126 — Holosoma fragment adapter preflight (2026-06-03)

- [x] adapter 要求 `box021_035_p1/p2` 仍为 `FRAGMENT_HOLDOUT_ONLY` 且 `rl_train_allowed=false`；2/2 adapter rows pass、paired frames 214、output fps 50。
- [x] 结论：可用于 downstream reward inspection/adapter debugging；main `box021_029_p2` 仍 blocked。log: `workspace/core4d/log/156_E126_holosoma_fragment_adapter_preflight_results.md`。

## E125 — RL hand-support preflight (2026-06-03)

- [x] 脚本扫描 E120-E124 recent smoke 的 52 个 test rows；每个 workset case 选 1 个最佳 row 并转成 source/freejoint `qpos43` converter input。
- [x] 结果：4 个 selected rows、4 个 finite `qpos43` npz/json、`rl_ready_rows=0`、training_launched=false。main `box021_029_p2` 最佳仍是 `BLOCK_MAIN_GATE_FAIL`。
- [x] log: `workspace/core4d/log/155_E125_rl_hand_support_preflight_results.md`。

## E138 — E137 semantic contact converter preflight (2026-06-03)

- [x] 完成：4/4 convert pass、4/4 inject pass、4/4 CPU MotionLoader pass；converted frames/shapes 为 117/124/210/220；main `box021_029_p2` either active 0.741935。
- [x] log: `workspace/core4d/log/168_E138_e137_semantic_contact_converter_preflight_results.md`。未启动 CEM/PPO/远程训练，`rl_ready_rows=0`。

## E139 — E138 semantic ref_object_contact env probe (2026-06-03)

- [x] Partner injection 4/4 pass；R135 Box021 partner env bounded no-debug stepping 4/4 startup pass，`has_partner=true` 4/4，`has_object_contact=true` 4/4，`ref_object_contact` positive 4/4；main `box021_029_p2` motion/ref contact totals 177/151，ref either 79。
- [x] log: `workspace/core4d/log/169_E139_e138_semantic_ref_object_contact_env_probe_results.md`。未启动 PPO/CEM/远程任务，`rl_ready_rows=0`。

## E140 — Holosoma semantic ref-mask reward readiness (2026-06-03)

- [x] runtime consumer 有 `RefMaskedHandObjectContactReward`、`RefMaskedTwoHandObjectContactReward`、`LostHandContactTermination`；Box023 R099/R109/R110 已用 ref-mask reward；Box021 R135/R138 均 `missing_ref_mask_reward`。E139 artifacts ready=true，recommendation=`needs_ref_mask_reward_config_variant`。
- [x] log: `workspace/core4d/log/170_E140_holosoma_semantic_ref_mask_reward_readiness_results.md`。未启动 PPO/CEM/远程任务，`rl_ready_rows=0`。

## E141 — Box021 semantic ref-mask reward probe (started, superseded)

- [ ] 2026-06-03 E141 启动：目标是给 Holosoma 增加最小 Box021 semantic ref-mask reward/experiment variant，并跑 bounded no-PPO reward probe。
- [ ] E141 代码审计和配置实现中；被 E142 纠正方向后 superseded。

## E142 — OmniRetarget contact exceedance audit (2026-06-03)

- [x] 结论：12/12 E112/E113 candidate rows 成功 join E110 OmniRetarget；0/12 candidates、0/7 best-by-case 超过 OmniRetarget physics contact。E112/E113 确实相对 E110 Spider 或局部 baseline 改善接触，但还没有达到"比 OmniRetarget 手物 physics contact 更好"的目标。
- [x] log: `workspace/core4d/log/171_E142_omniretarget_contact_exceedance_audit_results.md`。

## E143 — raw_mask_ref_fk 24-case OmniRetarget comparison (2026-06-03)

- [x] manifest/preflight 完成：24 rows，`to_run=21`、`already_done=3`（复用 E112 raw）；split 分布各 7 个待跑。
- [x] full 运行与回收完成：本地 7/7 + 远端 14/14，E143 root NPZ/MP4 各 21 个，另复用 3 个，最终 24/24。
- [x] 方法均值：OmniRetarget hand-object contact 54.4%，ref_fk 43.1%，raw_mask_ref_fk 33.0%；raw_mask_ref_fk 0/24 超过 OmniRetarget。
- [x] log: `workspace/core4d/log/172_E143_raw_mask_ref_fk_24case_omniretarget_comparison_results.md`。

## E144 — Full nonbox raw_mask_ref_fk CEM (2026-06-05)

- [x] Phase 1 全量 nonbox raw-contact/template audit：真实覆盖 132 条 nonbox 候选（bucket 46 / desk 56 / chair 30），5cm pass 82；21 个 required source templates 全部 `manual_review_required`。
- [x] Phase 1 template review：bucket 4 个 approve_clean / 17 个 needs_manual_edit。
- [x] Stage2b/S4/S5 推进到 CEM 前 gate：`stage2b_ready=11`、target gate `pass=11`；S5 `HANDOFF_REVIEW_VISUAL_QC=11`。
- [x] CEM-ready 恢复：S5 `HANDOFF_READY=11`，11 variants，preflight 11/11 True。
- [x] full CEM 完成：本地 4/4 + 远端 7/7，11/11 evaluated，`cem_status=fail` 11（`cem_work_status_fail=6`、`lowerbody_interference=5`），`RL_EXPORT_READY=0`。
- [x] completion audit：1456/1456 nonbox accounting、eval missing=0、S6 `DOWNSTREAM_CEM_FAIL=11`。log: `workspace/core4d/log/176_E144_completion_audit.md`。
- [x] nonbox template draft 补齐：17 个未 release template 生成 review-only draft（bucket `bucket_wall_proxy_aabb`、desk/chair voxel proxy）。
- [x] template mesh/collision 可视化、desk/chair multi-box proxy、desk 坐标轴修正、surface voxel proxy、tight proxy 收紧全部完成。
- [x] non-box template policy 写入正式 v3 pipeline：desk/chair tight surface voxel multi-box proxy 上升到 `data_construction_v3/stages/s2_templates/build_or_audit_templates.py`。log: `workspace/core4d/log/183_E144_nonbox_template_policy_pipeline_update.md`。

## E145 — Full nonbox template release to RL-ready (2026-06-05)

- [x] Phase 1 完成：Stage2b 82 rows 终态为 69 pass / 13 fail；S5 `HANDOFF_READY=69`；CEM-ready 69 variants，三卡 split local-gpu0 23 / remote-gpu0 23 / remote-gpu1 23。log: `workspace/core4d/log/184_E145_phase1_cem_ready_visual_qc_results.md`。
- [x] Phase 2 高优 9 case full CEM 完成：S6 evidence `cem_status=pass` 1、fail 8，`rl_status=not_run` 9；`RL_EXPORT_READY=1`、`SKIP_CEM_FAIL=8`。log: `workspace/core4d/log/185_E145_phase2_high_priority_full_cem_results.md`。
- [x] Partner OmniRetarget 两条补齐：`bucket010_20231003_2_055_p2` 和 `desk023_20231030_019_p1` partner retarget/trim 完成。本次只做 OmniRetarget/trim，未启动 CEM/RL。
