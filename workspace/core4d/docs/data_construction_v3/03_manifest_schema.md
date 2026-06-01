# 03 Manifest 结构

v3 的可复现性依赖 manifest，而不是依赖 results 进 git。每个阶段都必须输出 machine-readable TSV/JSON 和简短 Markdown summary。

## 运行 manifest

`run_manifest.json` 字段：

| 字段 | 说明 |
|---|---|
| `run_id` | 本次运行唯一 id |
| `mode` | `full-from-raw` / `resume-from-summary` |
| `start_time` / `end_time` | ISO timestamp |
| `config_path` | 用户输入配置 |
| `config_hash` | resolved config hash |
| `raw_data_root` | `CORE4D_RAW_ROOT` |
| `output_root` | 当前 run root |
| `spider_git_sha` / `spider_dirty` | spider 状态 |
| `holosoma_git_sha` / `holosoma_dirty` | holosoma 状态 |
| `command_line` | 完整命令 |
| `stage_decisions` | 每阶段 pass/fail/skip |
| `known_warnings` | 可继续但需要注意的问题 |

## 可复现性报告

`verify_reproducibility.py` 输出 run 级自检报告：

| 字段 | 说明 |
|---|---|
| `stage` | 固定为 `verify_reproducibility` |
| `checked_at` | 检查时间 |
| `run_dir` | 被检查的 v3 run 目录 |
| `status` | `pass` / `fail` |
| `errors` | 必须修复的问题 |
| `warnings` | 可继续但需要注意的问题 |
| `summary` | 行数、manifest 数量、config hash、decision 分布等摘要 |

报告路径：

```text
stage_s5_handoff/reproducibility/reproducibility_report.json
stage_s5_handoff/reproducibility/reproducibility_report.md
```

## 导入 manifest

`import_legacy_snapshot.py` 输出 legacy/manual imported snapshot manifest：

| 字段 | 说明 |
|---|---|
| `snapshot_id` | 本次导入快照 id |
| `source_type` | `legacy_import` / `manual_seed` |
| `source_ref` | legacy root、原始文件或人工来源 |
| `original_input` | 原始 TSV 路径 |
| `snapshot_input` | 复制到 run 内的 TSV 路径 |
| `input_sha256` | 导入 TSV checksum |
| `imported_registry_tsv` | 规范化后的 v3 registry TSV |
| `known_risks` | 导入风险说明 |
| `errors` / `warnings` | 导入校验结果 |

导入快照目录：

```text
imported_snapshots/<snapshot_id>/
  imported_case_state_registry.tsv
  imported_case_state_registry.json
  import_manifest.json
  import_manifest.md
```

`resume-from-summary` 还会在当前 run 下生成：

```text
imported_snapshots/resume_inputs/
  resume_input_manifest.json
  case_state_<source registry filename>
  stage2b_<N>_<source manifest filename>
  target_gate_<N>_<source manifest filename>
```

`resume_input_manifest.json` 记录 source/copied path、source/copied sha256 和字节数；run 内后续步骤只使用 copied path。`verify_reproducibility.py` 会重算 copied file sha256，确认 resume run 内部输入没有被后续改动。

Manual seed TSV 可用 `write_manual_seed_template.py` 生成。它使用与 `case_state_registry` 相同的字段，导入时通过 `import_legacy_snapshot.py --source-type manual_seed` 规范化为 `imported_case_state_registry.tsv`。

## case 状态注册表

`case_state_registry` 是跨阶段状态表。S0-S2 可按 case 记录；S3 以后必须按 `(case_id, retarget_variant_id, target_variant_id)` 记录。

核心字段：

| 字段 | 说明 |
|---|---|
| `case_id` | 规范化 case id |
| `object_key` | 例如 `box026` |
| `object_name` | CORE4D / OmniRetarget 使用的物体名 |
| `date` / `seq` / `person` | CORE4D 身份 |
| `person_idx` | `person1=0`, `person2=1` |
| `raw_inventory_status` | `present` / `invalid_raw` / `not_seen` |
| `raw_contact_3cm_status` | `pass` / `reject` / `review` / `not_run` |
| `raw_contact_5cm_status` | `pass` / `reject` / `review` / `not_run` |
| `template_status` | `clean` / `backlog` / `audit_fail` / `manual_review_required` |
| `retarget_variant_id` | S3 之后必填；S0-S2 可用 `shared` |
| `stage2b_status` | `pass` / `omniretarget_infeasible` / `preprocess_fail` / `not_run` |
| `target_variant_id` | `ref_fk` / `adaptive` / `fingertip_aware` / future |
| `target_gate_status` | `pass` / `review` / `reject` / `not_run` |
| `visual_qc_status` | `pass` / `review` / `reject` / `not_run` |
| `cem_status` | `pass` / `fail` / `not_run` / `not_required` |
| `rl_status` | `pass` / `fail` / `not_run` / `not_required` |
| `downstream_decision` | S6 下游证据分类；不参与 S1-S5 数据 gate |
| `downstream_failure_mode` | CEM/RL 失败模式说明 |
| `downstream_evidence_root` | S6 证据目录 |
| `cem_run_id` | CEM run id |
| `cem_result_npz` / `cem_video` / `cem_metrics_ref` | CEM 结果、视频、指标引用 |
| `rl_run_id` | RL run id |
| `rl_checkpoint` / `rl_metrics_ref` / `rl_video` | RL checkpoint、评测指标、视频引用 |
| `current_decision` | 当前综合状态 |
| `evidence_root` | 证据目录 |
| `source_type` | `v3_run` / `manual_seed` / `legacy_import` |
| `source_ref` | run id、legacy path 或人工记录 |
| `schema_version` | registry schema 版本 |
| `updated_at` | 更新时间 |

`fingertip_aware` route 必填字段：

| 字段 | 说明 |
|---|---|
| `route_diagnostic_status` | E099-E101 route diagnostic 总状态；进入 Stage2b 必须为 `pass` |
| `route_diagnostic_ref` | 诊断 manifest 或生成目录引用 |
| `fingertip_vote_status` | raw fingertip vote 是否可用 |
| `palm_vote_status` | palm/FK vote 是否可用 |
| `quat_audit_status` | object quat/world-up audit 是否通过 |
| `face_changed_l` / `face_changed_r` | palm face 与 fingertip face 是否不同 |
| `disable_world_up` | quat audit 是否禁用 world-up projection |
| `target_active_mask_status` | external target active mask 是否存在并被下游读取 |
| `target_npz` | E100 external target NPZ 路径 |
| `target_npz_sha256` | E100 external target NPZ hash |
| `target_gap_status` | E100 target gap 检查状态 |
| `active_L_frac` / `active_R_frac` | E100 active mask 左/右有效比例 |
| `e101_route_evidence_status` | E101 route-level guard / negative-prior evidence 是否通过 |

对默认 `ref_fk` route，上述字段可为空或 diagnostic，不作为 hard gate。

S1 当前脚本输出：

| 文件 | 说明 |
|---|---|
| `inventory.tsv/json` | 从 raw 扫描出的全量 case-person inventory。 |
| `inventory_summary.{json,md}` | inventory 决策、对象、尺寸和 route 统计。 |
| `raw_contact_candidates_3cm.tsv/json` | 3cm 几何 contact proxy 候选表。 |
| `raw_contact_pass_3cm.tsv/json` | 3cm pass 子集。 |
| `raw_contact_candidates_5cm.tsv/json` | 5cm 几何 contact proxy 候选表。 |
| `raw_contact_pass_5cm.tsv/json` | 5cm pass 子集。 |
| `raw_contact_run_summary.json` | 同一 raw-contact run 的汇总。 |
| `per_sequence/*/raw_contact_proxy.npz` | sequence-level 距离、vertex count、`raw_contact_mask_3cm` 和 `raw_contact_mask_5cm`。 |

`config_resolved.json` 的 `stage2b_contact_label` 记录本轮 S2/S1b/S3 继续使用哪一档 pass 子集，允许值是 `3cm` 或 `5cm`。该字段不改变 S1 同时输出两档候选的要求。

`update_case_state_registry.py` 支持从 S1 输出同步 registry：

- `--from-inventory-tsv` 更新 `raw_inventory_status` 和 `template_status`；
- `--from-raw-contact-tsv --raw-contact-label 3cm` 更新 `raw_contact_3cm_status`；
- `--from-raw-contact-tsv --raw-contact-label 5cm` 更新 `raw_contact_5cm_status`。

S2 当前脚本输出：

| 文件 | 说明 |
|---|---|
| `template_backlog.tsv/json` | Required source templates 的综合状态和 recommended action。 |
| `template_audit.tsv/json` | 每个 source scene 的 MuJoCo load、robot inertial、object mesh/collision 审计。 |
| `template_build.tsv/json` | `--apply-build` 或 dry-run 的 build 记录。 |
| `template_visual_review/template_visual_manifest.tsv/json` | source template review MP4/sheet 的生成状态。 |
| `template_visual_review/template_visual_summary.json/md` | source template visual review 分布。 |
| `template_summary.{json,md}` | template 状态分布和非 clean 列表。 |

`update_case_state_registry.py --from-template-backlog-tsv` 会把 matching `(object_key, person)` 的 case rows 更新为：

- `template_status=clean`
- `template_status=clean_reviewed`
- `template_status=backlog`
- `template_status=audit_fail`
- `template_status=manual_review_required`

其中 `manual_review_required` 主要用于非 box 物体，不能自动进入 Stage2b。
`clean_reviewed` 主要用于非 box proxy template 审查通过后的显式放行状态；它和 `clean` 一样可进入 S3，但 provenance 必须来自 review manifest。

非 box review TSV 字段：

| 字段 | 说明 |
|---|---|
| `source_scene_task` | 例如 `bucket007_person1` |
| `object_key` / `person` / `object_category` | source template 身份 |
| `proxy_scene_xml` | 被审查的 proxy scene |
| `review_decision` | `approve_clean` / `reject` / `needs_manual_edit` |
| `reviewer` | `human` / `subagent` / `scripted_audit` |
| `review_notes` | 审查结论 |
| `approved_collision_policy` | 审批后的 collision policy |
| `approved_mass_policy` | 审批后的 mass/inertia policy |
| `evidence_video` / `evidence_sheet` | 可视化证据路径 |

S3 当前脚本输出：

| 文件 | 说明 |
|---|---|
| `cases_stage2b_ready_<retarget_variant>_<target_variant>.tsv` | 可直接喂给 legacy `pipeline.sh` 的 case file。 |
| `stage2b_manifest_<retarget_variant>_<target_variant>.tsv/json` | 每个 case-person 在该 retarget variant + target route 下的 Stage2b queue/status/provenance。 |
| `stage2b_summary_<retarget_variant>_<target_variant>.json/md` | Stage2b ready/template-blocked/variant-blocked 分布。 |
| `run_stage2b_<retarget_variant>_<target_variant>.sh` | 固化执行命令；默认 dry-run，显式 execute 才跑。 |

S3 manifest 必须带：

- `retarget_variant_id`
- `target_variant_id`
- `replace_wrist_with_fingertip`
- `solver_repo_path`
- `solver_git_sha`
- `converter_script`
- `converter_git_sha`
- `params_json`
- `result_root`
- `holosoma_case_root`
- `converted_npz`
- `omniretarget_output_npz`
- `trimmed_npz`
- `spider_task_dir`
- `spider_trajectory`
- `contact_mask_npz`
- `verify_summary`
- `command_line`

`update_case_state_registry.py --from-stage2b-manifest-tsv` 会按 `(case_id, retarget_variant_id, target_variant_id)` 新增或更新 registry row。`retarget_variant_id` 和 `target_variant_id` 是独立轴。默认 `ref_fk` row 只写 `diagnostic_contracts=E098_global`；`fingertip_aware` row 写 `E098_global+E099_E100_E101_fingertip_aware`。

`fingertip_aware` row 还必须写 `route_diagnostic_status=pass`。如果缺失或不通过，registry/S5 统一归为 `REJECT_FINGERTIP_ROUTE_CONTRACT`，不能被提升为 Stage2b-ready。

S4 当前脚本输出：

| 文件 | 说明 |
|---|---|
| `target_gate_manifest.tsv/json` | 每个 Stage2b row 的 target gate 状态、路径、shape、MuJoCo load 和污染审计。 |
| `target_gate_summary.json/md` | gate pass/not_run/reject 分布和失败模式。 |
| `visual_qc_render/visual_qc_render_manifest.tsv/json` | 可视化审查用 replay MP4/keyframe sheet 生成状态。 |
| `visual_qc/visual_qc_manifest.tsv/json` | visual QC 待审/人工审查状态。 |
| `visual_qc/visual_qc_summary.json/md` | visual QC 分布和 review/reject 列表。 |

S5 当前脚本输出：

| 文件 | 说明 |
|---|---|
| `candidate_bank.tsv/json` | 全量 registry rows 的统一 candidate decision。 |
| `handoff_manifest.tsv/json` | 可交接或待运行 rows 的下游入口索引。 |
| `rejected_manifest.tsv/json` | reject rows 与标准失败原因。 |
| `handoff_summary.json/md` | candidate/handoff/reject 分布。 |
| `cem_overrides/cem_override_manifest.tsv/json` | CEM override 导出状态；按 `(case_id, retarget_variant_id, target_variant_id)` 对齐 handoff。 |
| `cem_overrides/cem_override_summary.json/md` | override 成功/失败与 target adapter 分布。 |
| `cem_overrides/overrides/*.yaml` | 下游 CEM 可直接引用的 override 配置。 |

CEM override manifest 常用字段：

- `override_status`: `pass` / `skip` / `fail`
- `target_adapter_status`: `ref_fk` / `external_target` / `external_target_invalid` / `external_target_missing`
- `target_source`: `ref_fk` / `external`
- `override_config`
- `target_npz`
- `target_npz_sha256`
- `target_array_key`
- `target_shape`
- `target_time_axis`
- `contact_hdmi_target_uses_eef_offset`

其中 `ref_fk` 不需要外部 target NPZ；`adaptive` / `fingertip_aware` 必须提供并通过外部 target 校验。

S6 当前脚本输出：

| 文件 | 说明 |
|---|---|
| `downstream_evidence_manifest.tsv/json` | CEM/RL 下游证据；按 `(case_id, retarget_variant_id, target_variant_id)` 对齐 S5 handoff。 |
| `downstream_evidence_summary.json/md` | 下游 pass/fail/not_run 分布。 |

S6 manifest 常用字段：

- `cem_status`
- `rl_status`
- `downstream_decision`
- `downstream_failure_mode`
- `cem_run_id`
- `cem_result_npz`
- `cem_video`
- `cem_metrics_ref`
- `rl_run_id`
- `rl_checkpoint`
- `rl_metrics_ref`
- `rl_video`

这些字段只作为 downstream evidence。`update_case_state_registry.py --from-downstream-evidence-tsv` 合并后不会把 CEM/RL 失败反向写成 `REJECT_RAW_CONTACT`、`REJECT_TEMPLATE_*` 或 `REJECT_TARGET_GATE`。

S4 manifest 关键字段：

- `target_gate_status`: `pass` / `review` / `reject` / `not_run`
- `visual_qc_status`: 当前默认为 `not_run`
- `failure_mode`
- `target_scene`
- `scene_act`
- `trajectory`
- `trimmed_npz`
- `missing_required`
- `qpos_shape`
- `trimmed_qpos_matches_spider_qpos`
- `scene_dims`
- `scene_act_dims`
- `scene_robot_polluted_mass_29_632`
- `scene_act_robot_polluted_mass_29_632`
- `contact_pos_source`
- `pelvis_end_z`
- `pelvis_tilt_end`

`update_case_state_registry.py --from-target-gate-manifest-tsv` 会更新 variant-specific row 的 `target_gate_status` 和 `visual_qc_status`。如果 Stage2b outputs 缺失，则保持 `target_gate_status=not_run`，不会把 case 错误降级为 target gate reject。

`update_case_state_registry.py --from-visual-qc-manifest-tsv` 只更新 variant-specific row 的 `visual_qc_status` 和 visual evidence，不改写 raw/template/Stage2b 事实。`visual_qc_status=pass` 且 machine `target_gate_status=pass` 时，S5 可升为 `PASS/HANDOFF_READY`；`visual_qc_status=reject` 时，S5 归为 `REJECT_VISUAL_QC`。

S5 当前脚本输出：

| 文件 | 说明 |
|---|---|
| `candidate_bank.tsv/json` | 全量 registry rows 的统一 candidate decision。 |
| `handoff_manifest.tsv/json` | 可交接或待运行 rows 的下游入口索引。 |
| `rejected_manifest.tsv/json` | reject rows 与标准失败原因。 |
| `handoff_summary.json/md` | candidate/handoff/reject 分布。 |

`handoff_manifest` 必须包含：

- `case_id`
- `object_key`
- `object_name`
- `date`
- `seq`
- `person`
- `retarget_variant_id`
- `target_variant_id`
- `candidate_decision`
- `handoff_decision`
- `source_scene`
- `target_scene`
- `trajectory`
- `scene_act`
- `contact_mask`
- `raw_contact_threshold_label`
- `stage2b_target_task`
- `target_gate_status`
- `visual_qc_status`
- `cem_status`
- `rl_status`
- `diagnostic_contracts`

S5 不修改 raw/template/Stage2b/gate 事实，只把已有 manifest 归并成下游可消费索引。

## retarget variant 注册表

`retarget_variant_registry` 管理 OmniRetarget/input conversion 版本。

| 字段 | 说明 |
|---|---|
| `retarget_variant_id` | 稳定唯一 id |
| `display_name` | 人读名称 |
| `solver_family` | `omniretarget` / future |
| `solver_version` | 原始 / v1 / future |
| `solver_repo_path` | 实际执行 repo |
| `solver_git_sha` | 执行时 git sha |
| `solver_dirty` | 是否 dirty |
| `converter_script` | conversion 脚本 |
| `converter_git_sha` | converter 所在 repo sha |
| `params_json` | 完整显式参数 |
| `input_rewrite_policy` | `wrist` / `fingertip` / `topface` / future |
| `branch_from_stage` | 通常是 `conversion` |
| `created_at` | 创建时间 |

初始 variant：

- `omnirt_original`
- `omnirt_v1`
- `omnirt_v1_fingertip_replacement`

exact repo/path/commit 必须在实现时确认，不能只靠名字推断。

## 阶段 manifest

每个阶段输出 `stage_manifest_<stage>.tsv/json`。共同字段：

| 字段 | 说明 |
|---|---|
| `run_id` | 所属 run |
| `stage` | `S1` / `S2` / ... |
| `case_id` | case id |
| `retarget_variant_id` | S3 后必填 |
| `target_variant_id` | S4 后必填 |
| `decision` | pass/review/reject |
| `failure_mode` | 标准失败分类 |
| `metrics_json` | 阶段指标 |
| `evidence_paths_json` | 文件证据 |
| `command_line` | 生成该 row 的命令 |
| `schema_version` | schema 版本 |
