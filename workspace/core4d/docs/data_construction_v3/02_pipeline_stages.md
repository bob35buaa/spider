# 02 管线阶段

## S0: 配置与环境检查

输入：

- repo 路径；
- raw data 路径；
- run root；
- SPIDER Python 与 OmniRetarget/hsretargeting Python 环境；
- GPU/MuJoCo/ffmpeg 配置。

输出：

- `environment_check.json/md`
- `config_resolved.json`
- `git_state.json`

失败时必须给出明确原因，不能进入后续阶段。

S0 会同时检查 SPIDER 当前 Python 环境和 Holosoma retargeting 环境。由于外层 shell 可能已经激活了 SPIDER `.venv`，Stage2b wrapper 在 source `HOLOSOMA_REPO/scripts/source_retargeting_setup.sh` 后必须优先使用 `$CONDA_PREFIX/bin/python` 或显式 `RETARGET_PYTHON_BIN`，不能裸调用 `python`。

## S0b: 状态注册表初始化

创建或加载：

- `case_state_registry`
- `retarget_variant_registry`

`resume-from-summary` 只能基于 v3 registry、stage manifest 或显式 imported snapshot。

## S1: 原始清单与 raw contact

输入：

- `CORE4D_RAW_ROOT/human_object_motions`
- `CORE4D_RAW_ROOT/object_models`

输出：

- 全量 case-person inventory；
- object mesh extents / volume；
- raw contact 3cm / 5cm；
- contact proxy summary；
- raw contact 可视化；
- 对 `fingertip_aware` route 必需的 fingertip vote / palm vote / quat audit。

规则：

- 3cm 和 5cm 必须分别输出；
- raw contact 是几何 proxy，不是人工真值；
- source scene missing 不能导致 raw mining 跳过；
- face 相关计算必须使用 E098 后的全 3D 六面 helper。

当前入口：

```bash
workspace/core4d/scripts/data_construction_v3/stages/s1_raw_contact/build_inventory.py \
  --core4d-raw-root "$CORE4D_RAW_ROOT" \
  --out-dir "$RUN_DIR/s1_raw_contact/inventory"

workspace/core4d/scripts/data_construction_v3/stages/s1_raw_contact/run_raw_contact.py \
  --core4d-raw-root "$CORE4D_RAW_ROOT" \
  --inventory-tsv "$RUN_DIR/s1_raw_contact/inventory/inventory.tsv" \
  --out-dir "$RUN_DIR/s1_raw_contact/raw_contact" \
  --queue selected-medium-box \
  --thresholds-m 0.03,0.05
```

`run_raw_contact.py` 同一次 surface-distance 计算会写出两套候选：

- `raw_contact_candidates_3cm.{tsv,json}`
- `raw_contact_pass_3cm.{tsv,json}`
- `raw_contact_candidates_5cm.{tsv,json}`
- `raw_contact_pass_5cm.{tsv,json}`

并在 `per_sequence/*/raw_contact_proxy.npz` 中保存 `raw_contact_mask_3cm` 和 `raw_contact_mask_5cm`。两档阈值是并行候选集，不能用 5cm 覆盖 3cm。

`run_pipeline.py` 默认用 `--stage2b-contact-label 3cm` 选择后续 S2/S1b/S3 使用的 pass 子集；若本轮候选放宽到 5cm，显式传 `--stage2b-contact-label 5cm`。无论选择哪一档，S1 仍同时输出并同步 3cm/5cm 候选状态。

### S1b: fingertip-aware 路线诊断

`fingertip_aware` route 需要 E099-E101 的 route diagnostic manifest。默认 `ref_fk` 不需要该步骤。

当前入口：

```bash
STAGE2B_CONTACT_LABEL=3cm

workspace/core4d/scripts/data_construction_v3/stages/s1_raw_contact/build_fingertip_route_diagnostics.py \
  --input-tsv "$RUN_DIR/s1_raw_contact/raw_contact/raw_contact_pass_${STAGE2B_CONTACT_LABEL}.tsv" \
  --out-dir "$RUN_DIR/s1_raw_contact/fingertip_route_diagnostics"
```

输入：

- E099 `fingertip_face_stats.tsv`
- E099 `palm_face_stats.tsv`
- E099 `quat_audit.tsv`
- E100 `target_gap_summary.tsv`
- E100 `fingertip_targets/*/spider_contact_target_object_local.npz`
- E101 `cem_outcome_matrix.tsv`

输出：

- `fingertip_route_diagnostics.tsv/json`
- `fingertip_route_diagnostics_summary.json/md`

通过条件：

- `fingertip_vote_status=pass`
- `palm_vote_status=pass`
- `quat_audit_status=pass`
- `target_active_mask_status=pass`
- `e101_route_evidence_status=pass`

E101 在这里按 route-level evidence 使用：box004 guard 证明 `fingertip_aware` 不退化；E101 当前负例作为 negative prior。未命中当前负例且 route guard 存在时，`e101_route_evidence_status=pass`。

## S2: source scene template 构建与审计

输入：

- candidate manifest；
- object mesh；
- clean base scene；
- template policy。

输出：

- source template；
- template backlog；
- inertial audit；
- collision extents audit；
- MuJoCo load audit；
- visual sheet/mp4；
- `task_info.json` provenance。

规则：

- box 类物体可按 E103 clean base 流程自动生成；
- 非 box 物体必须人工审查；
- source template 不承担 target pose 责任；
- target pose 必须由 trimmed qpos 第一帧 patch。

当前入口：

```bash
STAGE2B_CONTACT_LABEL=3cm

workspace/core4d/scripts/data_construction_v3/stages/s2_templates/build_or_audit_templates.py \
  --input-tsv "$RUN_DIR/s1_raw_contact/raw_contact/raw_contact_pass_${STAGE2B_CONTACT_LABEL}.tsv" \
  --core4d-raw-root "$CORE4D_RAW_ROOT" \
  --out-dir "$RUN_DIR/s2_templates"
```

默认是只读审计。只有显式传入 `--apply-build` 时才会创建缺失的 box source template：

```bash
workspace/core4d/scripts/data_construction_v3/stages/s2_templates/build_or_audit_templates.py \
  --input-tsv "$RUN_DIR/s1_raw_contact/raw_contact/raw_contact_pass_${STAGE2B_CONTACT_LABEL}.tsv" \
  --core4d-raw-root "$CORE4D_RAW_ROOT" \
  --out-dir "$RUN_DIR/s2_templates" \
  --apply-build
```

`--apply-build` 自动处理 `object_category=box` 的缺失模板。非 box 只允许生成 review 用 proxy template：

- bucket 使用 `bucket_wall_proxy_aabb`；
- board/stick 使用 `mesh_aabb_box_proxy`；
- desk/chair 使用 tight surface voxel multi-box proxy：`desk_surface_voxel_multibox_proxy_draft` / `chair_surface_voxel_multibox_proxy_draft`。

desk/chair proxy 不是语义桌/椅模板；它从 OBJ 表面 voxelization 生成一组局部 AABB boxes，避免把圆面三脚凳、侧板/U 型架、非标准 chair 强行套成标准桌椅。所有非 box proxy 即使 MuJoCo load 成功，也保持 `template_status=manual_review_required`，不能自动进入 Stage2b。

source template 可视化包入口：

```bash
workspace/core4d/scripts/data_construction_v3/stages/s2_templates/render_template_review_package.py \
  --template-backlog-tsv "$RUN_DIR/s2_templates/template_backlog.tsv" \
  --out-dir "$RUN_DIR/s2_templates/template_visual_review"
```

非 box proxy review 还应生成 mesh/collision overlay。通用入口：

```bash
workspace/core4d/scripts/data_construction_v3/stages/s2_templates/render_template_mesh_collision_review_package.py \
  --input-tsv "$RUN_DIR/s2_templates/template_backlog.tsv" \
  --out-dir "$RUN_DIR/s2_templates/template_mesh_collision_review" \
  --render-statuses manual_review_required \
  --object-only \
  --overwrite
```

`--object-only` 会隐藏机器人，只保留 real mesh、collision proxy、mesh+collision 三栏，适合检查 desk/chair surface voxel proxy 是否贴合 mesh、是否明显外扩、是否套错拓扑。

`run_pipeline.py` 会在 S2 audit/build 后自动生成 orbit visual review package 和 `--object-only` mesh/collision review package。缺 scene 的 backlog row 会明确标为 `not_rendered`；已有 scene 但 render 失败会进入 `render_error`，供 release review 和复现性自检定位。

输出：

- `template_backlog.tsv/json`：每个 required source template 的综合状态；
- `template_audit.tsv/json`：MuJoCo load、robot inertial、object collision/mesh audit；
- `template_build.tsv/json`：本次 build/dry-run 记录；
- `template_visual_review/template_visual_manifest.tsv/json`：source template visual sheet/mp4 的生成状态；
- `template_visual_review/template_visual_summary.json/md`：source template visual review 分布；
- `template_mesh_collision_review/template_mesh_collision_review_manifest.tsv/json`：source mesh/collision overlay review；
- `template_summary.{json,md}`：状态分布与非 clean 列表。

非 box 通过审查后，用 review TSV 显式覆盖 registry：

```bash
workspace/core4d/scripts/data_construction_v3/state/update_case_state_registry.py \
  --registry-dir "$RUN_DIR/registries" \
  --from-template-review-tsv "$RUN_DIR/s2_templates/nonbox_template_review.tsv" \
  --evidence-root "$RUN_DIR/s2_templates" \
  --source-ref S2_nonbox_template_review
```

只有 `review_decision=approve_clean` 会写入 `template_status=clean_reviewed`，从而允许进入 S3。

## S3: 按 variant 重定向

S3 开始按 `retarget_variant_id` × `target_variant_id` 双轴分叉。S0-S2 共享，S3 之后不能覆盖其它 variant/route 组合的输出。

标准链路：

```text
convert_core4d_to_omniretarget.py
  -> robot_retarget.py
  -> trim_no_contact.py
  -> generate_core4d_contact_masks.py
  -> create target scene from source template
  -> spider/process_datasets/core4d.py
  -> generate scene_act
  -> verify
```

每条 row 必须记录 solver git sha、converter git sha、参数 JSON、trim policy 和 output path。output path 不能只写最终 SPIDER 输入；必须显式展开 `converted_npz`、`omniretarget_output_npz`、`trimmed_npz`、`spider_task_dir`、`spider_trajectory`、`contact_mask_npz` 和 `verify_summary`，这样后续能直接定位 OmniRetarget 算法输出。

当前入口：

```bash
workspace/core4d/scripts/data_construction_v3/stages/s3_retarget/run_stage2b.py \
  --raw-contact-tsv "$RUN_DIR/s1_raw_contact/raw_contact/raw_contact_pass_${STAGE2B_CONTACT_LABEL}.tsv" \
  --template-backlog-tsv "$RUN_DIR/s2_templates/template_backlog.tsv" \
  --retarget-variant-registry "$RUN_DIR/registries/retarget_variant_registry.tsv" \
  --retarget-variant-id omnirt_v1 \
  --target-variant-id ref_fk \
  --inventory-tsv "$RUN_DIR/s1_raw_contact/inventory/inventory.tsv" \
  --core4d-raw-root "$CORE4D_RAW_ROOT" \
  --smplx-model-dir "$SMPLX_MODEL_DIR" \
  --out-dir "$RUN_DIR/s3_retarget/omnirt_v1/ref_fk"
```

该命令默认只生成：

- `cases_stage2b_ready_<retarget_variant>_<target_variant>.tsv`
- `stage2b_manifest_<retarget_variant>_<target_variant>.tsv/json`
- `stage2b_summary_<retarget_variant>_<target_variant>.json/md`
- `run_stage2b_<retarget_variant>_<target_variant>.sh`

`run_stage2b_<retarget_variant>_<target_variant>.sh` 默认带 `--dry-run`。真正执行时必须由 v3 wrapper 显式传入：

```bash
--execute --allow-legacy-stage2b-wrapper
```

这是有意设计：短期可复用历史 `workspace/core4d/data_preprocess/pipeline.sh`，但不能隐式回到 legacy 流程；每次调用都必须记录 variant、参数、输出路径和命令。

`retarget_variant_id` 和 `target_variant_id` 是两个独立轴：`omnirt_v1` / `omnirt_v1_fingertip_replacement` 只决定 OmniRetarget/input rewrite 参数；`ref_fk` / `adaptive` / `fingertip_aware` 决定 SPIDER target route。默认是 `target_variant_id=ref_fk`。

当前 legacy execute adapter 只支持 `target_variant_id=ref_fk`。非 `ref_fk` route 可以生成 dry-run manifest 和 route diagnostic，但真实执行必须补专门的 Stage2b target adapter；`run_stage2b.py --execute` 会直接拒绝 `adaptive` / `fingertip_aware`，防止把旧 pipeline 的输出误标成外部 target route 输出。

`fingertip_aware` 是可选 route，不是默认路线。启用该 route 时，`run_stage2b.py` 必须额外读取 E099-E101 route diagnostic manifest：

```bash
workspace/core4d/scripts/data_construction_v3/stages/s3_retarget/run_stage2b.py \
  --raw-contact-tsv "$RUN_DIR/s1_raw_contact/raw_contact/raw_contact_pass_${STAGE2B_CONTACT_LABEL}.tsv" \
  --template-backlog-tsv "$RUN_DIR/s2_templates/template_backlog.tsv" \
  --retarget-variant-registry "$RUN_DIR/registries/retarget_variant_registry.tsv" \
  --retarget-variant-id omnirt_v1 \
  --target-variant-id fingertip_aware \
  --route-diagnostic-tsv "$RUN_DIR/s1_raw_contact/fingertip_route_diagnostics.tsv" \
  --inventory-tsv "$RUN_DIR/s1_raw_contact/inventory/inventory.tsv" \
  --core4d-raw-root "$CORE4D_RAW_ROOT" \
  --smplx-model-dir "$SMPLX_MODEL_DIR" \
  --out-dir "$RUN_DIR/s3_retarget/omnirt_v1/fingertip_aware"
```

缺少该 manifest，或 manifest 中 `fingertip_vote_status` / `palm_vote_status` / `quat_audit_status` / `target_active_mask_status` / `e101_route_evidence_status` 任一未 pass 时，S3 输出 `stage2b_route_diagnostics_missing` 或 `stage2b_route_diagnostics_not_pass`，不会执行 Stage2b。

## S4: target gate 与可视化 QC

输入：

- target scene；
- trajectory；
- contact masks；
- source template audit；
- optional external target。

输出：

- gate summary；
- replay metrics；
- MuJoCo replay；
- OmniRetarget visual；
- object-local overlay；
- external target gap summary；
- visual QC manifest。

硬 gate：

- clean scene/source audit；
- qpos/layout 一致；
- object pose patch 正确；
- penetration/inside 不超阈值；
- lower-body object interference 不超阈值；
- replay 不趴箱、不穿箱；
- external target 若启用，必须检查 shape/hash/active-mask/target-gap。

`pelvis_tilt_end` 是诊断项，不单独作为失败归因。

当前入口：

```bash
workspace/core4d/scripts/data_construction_v3/stages/s4_gate_visual_qc/run_target_gate.py \
  --stage2b-manifest-tsv "$RUN_DIR/s3_retarget/omnirt_v1/ref_fk/stage2b_manifest_omnirt_v1_ref_fk.tsv" \
  --out-dir "$RUN_DIR/s4_gate_visual_qc/omnirt_v1/ref_fk"
```

输出：

- `target_gate_manifest.tsv/json`
- `target_gate_summary.json/md`

机器 gate 检查：

- Stage2b expected outputs 是否存在；
- target `scene.xml` / `scene_act.xml` 是否能 MuJoCo load；
- `trajectory_kinematic.npz` 是否有 `qpos/qvel/ctrl/contact` 且帧数一致；
- `trimmed_npz["qpos"]` 是否与 SPIDER trajectory `qpos` 一致；
- scene / scene_act 维度是否符合当前 G1 object task 约定；
- robot inertial 是否残留 `mass=29.632` 污染；
- `contact_pos` 若存在，标注来源为 `fk_palm_site`。

当前 visual QC 状态独立保留为 `visual_qc_status=not_run`。replay MP4/sheet 可由独立入口生成，供人工/LLM release review 使用，但这不会替代机器 gate。

Visual QC replay package 入口：

```bash
workspace/core4d/scripts/data_construction_v3/stages/s4_gate_visual_qc/render_visual_qc_package.py \
  --target-gate-manifest-tsv "$RUN_DIR/s4_gate_visual_qc/omnirt_v1/ref_fk/target_gate_manifest.tsv" \
  --out-dir "$RUN_DIR/s4_gate_visual_qc/omnirt_v1/ref_fk/visual_qc_render"
```

输出：

- `visual_qc_render_manifest.tsv/json`
- `visual_qc_render_summary.json/md`
- 每个可渲染 case 的 replay MP4 和 keyframe sheet。

Visual QC manifest 入口：

```bash
workspace/core4d/scripts/data_construction_v3/stages/s4_gate_visual_qc/make_visual_qc.py \
  --target-gate-manifest-tsv "$RUN_DIR/s4_gate_visual_qc/omnirt_v1/ref_fk/target_gate_manifest.tsv" \
  --out-dir "$RUN_DIR/s4_gate_visual_qc/omnirt_v1/ref_fk/visual_qc"
```

默认行为：`target_gate_status=pass` 的 row 进入 `visual_qc_status=review`，表示待人工/LLM release review；未通过机器 gate 的 row 保持 `not_run`。`run_pipeline.py` 在 S4 target gate 后会自动生成这份默认 visual QC manifest 并同步 registry；人工/LLM 审查 TSV 可以后续再次导入覆盖。

如果已有人工/LLM 审查 TSV，可显式导入：

```bash
workspace/core4d/scripts/data_construction_v3/stages/s4_gate_visual_qc/make_visual_qc.py \
  --target-gate-manifest-tsv "$RUN_DIR/s4_gate_visual_qc/omnirt_v1/ref_fk/target_gate_manifest.tsv" \
  --review-tsv "$RUN_DIR/s4_gate_visual_qc/omnirt_v1/ref_fk/manual_visual_review.tsv" \
  --out-dir "$RUN_DIR/s4_gate_visual_qc/omnirt_v1/ref_fk/visual_qc"

workspace/core4d/scripts/data_construction_v3/state/update_case_state_registry.py \
  --registry-dir "$RUN_DIR/registries" \
  --from-visual-qc-manifest-tsv "$RUN_DIR/s4_gate_visual_qc/omnirt_v1/ref_fk/visual_qc/visual_qc_manifest.tsv" \
  --evidence-root "$RUN_DIR/s4_gate_visual_qc/omnirt_v1/ref_fk/visual_qc"
```

`manual_visual_review.tsv` 最小字段：

- `case_id`
- `retarget_variant_id`
- `target_variant_id`
- `visual_qc_status`: `pass` / `review` / `reject`
- `reviewer`
- `review_notes`
- 可选 `video_path` / `sheet_path`

## S5: 候选库与 handoff

输出：

- candidate bank；
- handoff manifest；
- hand collision sidecar scene manifest；
- CEM override manifest/config；
- updated registry。

S5 只表达数据和 target 是否可进入下游，不把 RL/CEM 失败反向写成 raw data 失败。
机器人手部碰撞体在 S5/CEM 作为独立轴处理，字段为 `hand_collision_variant_id`。默认 `sphere5cm` 保持旧行为；`rubber_hull` 通过 sidecar scene 把 `lh/rh` 从 5cm sphere 换成 rubber hand mesh convex hull。该轴只改变机器人侧碰撞几何，不改变 OmniRetarget 输入、不改变 target route，也不同于物体侧 `collision_policy`。

当前入口：

```bash
workspace/core4d/scripts/data_construction_v3/stages/s5_handoff/export_handoff.py \
  --case-state-registry "$RUN_DIR/registries/case_state_registry.tsv" \
  --stage2b-manifest-tsv "$RUN_DIR/s3_retarget/omnirt_v1/ref_fk/stage2b_manifest_omnirt_v1_ref_fk.tsv" \
  --target-gate-manifest-tsv "$RUN_DIR/s4_gate_visual_qc/omnirt_v1/ref_fk/target_gate_manifest.tsv" \
  --out-dir "$RUN_DIR/s5_handoff"
```

输出：

- `candidate_bank.tsv/json`：所有 registry rows 的统一分类；
- `handoff_manifest.tsv/json`：可交接或待运行的 rows，默认不塞入全部 reject；
- `rejected_manifest.tsv/json`：reject rows 和原因；
- `handoff_summary.json/md`：状态分布。

CEM override 导出入口：

```bash
workspace/core4d/scripts/data_construction_v3/stages/s5_handoff/export_cem_overrides.py \
  --handoff-manifest-tsv "$RUN_DIR/s5_handoff/handoff_manifest.tsv" \
  --out-dir "$RUN_DIR/s5_handoff/cem_overrides"
```

输出：

- `cem_override_manifest.tsv/json`：每个 handoff row 的 CEM 配置导出状态；
- `cem_override_summary.json/md`：`override_status` 与 target adapter 分布；
- `overrides/core4d_dcv3_<retarget>_<target>_<case>.yaml`：CEM 可引用的 override YAML。

`ref_fk` route 使用 `contact_hdmi_target_source=ref_fk`。`adaptive` / `fingertip_aware` 等 external target route 必须提供 `target_npz`，且导出时校验 path、sha256、`spider_contact_target_object_local` 或 `eval_contact_target_object_local` 的 `(T,2,3)` shape 与有限值。校验不通过只会让该 override row 失败，不会反向改写 raw/template/Stage2b/gate 状态。

手部碰撞体 scene adapter 入口：

```bash
workspace/core4d/scripts/data_construction_v3/stages/s5_handoff/patch_hand_collision.py \
  --base-scene-act "$TASK_DIR/scene_act.xml" \
  --hand-collision-variant-id rubber_hull \
  --scene-name scene_act_rubber_hull \
  --install-dir "$TASK_DIR" \
  --out-dir "$RUN_DIR/s5_handoff/hand_collision/<case_id>"
```

adapter 必须写 sidecar scene，不覆盖源 `scene_act.xml`。CEM override 通过 `scene_name=<sidecar basename>` 指向该 scene。

候选分类：

- `PASS`
- `REVIEW`
- `STAGE2B_READY`
- `RAW_CONTACT_READY`
- `INVENTORY_READY`
- `REJECT_RAW_INVENTORY`
- `REJECT_RAW_CONTACT`
- `REJECT_RAW_FINGERTIP_MISMATCH`
- `REJECT_FINGERTIP_ROUTE_CONTRACT`
- `REJECT_TEMPLATE_BACKLOG`
- `REJECT_TEMPLATE_AUDIT`
- `REJECT_OMNIRETARGET`
- `REJECT_TARGET_GATE`
- `REJECT_VISUAL_QC`

handoff 分类：

- `HANDOFF_READY`
- `HANDOFF_REVIEW_VISUAL_QC`
- `HANDOFF_PENDING_STAGE2B`
- `HANDOFF_PENDING_TEMPLATE_OR_VARIANT`
- `NO_HANDOFF_REJECTED`
- `NO_HANDOFF_PENDING`

默认 `handoff_manifest` 只写 `HANDOFF_*` rows；如需要把 rejected rows 也塞进 handoff，可显式加 `--include-rejected-in-handoff`。

## S6: 下游 CEM/RL 证据

输出：

- CEM metrics；
- lower-body strict proxy；
- videos；
- RL train/eval result；
- downstream evidence row。
- RL export input manifest。

S6 只记录下游证据，不反向改变 S1-S5 的数据构建判定。也就是说，CEM/RL 失败可以形成 `DOWNSTREAM_*` 证据，但不能把 raw contact、template、Stage2b 或 target gate 改写成失败。

当前入口：

```bash
workspace/core4d/scripts/data_construction_v3/stages/s6_downstream/record_downstream_evidence.py \
  --handoff-manifest-tsv "$RUN_DIR/s5_handoff/handoff_manifest.tsv" \
  --evidence-tsv "$RUN_DIR/s6_downstream/manual_or_eval_results.tsv" \
  --evidence-root "$RUN_DIR/s6_downstream" \
  --out-dir "$RUN_DIR/s6_downstream"

workspace/core4d/scripts/data_construction_v3/state/update_case_state_registry.py \
  --registry-dir "$RUN_DIR/registries" \
  --from-downstream-evidence-tsv "$RUN_DIR/s6_downstream/downstream_evidence_manifest.tsv" \
  --evidence-root "$RUN_DIR/s6_downstream"
```

RL motion export 不直接消费 S5，也不在脚本里猜 CEM npz。CEM 结果产生后，先生成 S6 join manifest：

```bash
workspace/core4d/scripts/data_construction_v3/stages/s6_downstream/export_rl_inputs.py \
  --handoff-manifest-tsv "$RUN_DIR/s5_handoff/handoff_manifest.tsv" \
  --cem-evidence-tsv "$RUN_DIR/s6_downstream/downstream_evidence_manifest.tsv" \
  --out-dir "$RUN_DIR/s6_downstream/rl_export"
```

输出：

- `rl_export_input.tsv/json`：S5 handoff 与 S6 CEM evidence 的 join 表，包含 `scene_act`、`trajectory`、`contact_mask`、`cem_result_npz`、`cem_status` 和 `rl_export_decision`；
- `rl_export_summary.json/md`：`RL_EXPORT_READY`、`SKIP_CEM_FAIL`、`WAIT_CEM_NOT_RUN` 等分布。

下游 RL 导出只消费 `rl_export_input.tsv` 中 `rl_export_decision=RL_EXPORT_READY` 的 rows。S5 保持 CEM 前 handoff 语义，不写入 CEM 后验结果。

`manual_or_eval_results.tsv` 最小字段：

- `case_id`
- `retarget_variant_id`
- `target_variant_id`
- `cem_status`: `pass` / `fail` / `not_run`
- `rl_status`: `pass` / `fail` / `not_run`
- 可选 `downstream_failure_mode` / `downstream_notes`
- 可选 `cem_result_npz` / `cem_video` / `cem_metrics_ref`
- 可选 `rl_run_id` / `rl_checkpoint` / `rl_metrics_ref` / `rl_video`

输出：

- `downstream_evidence_manifest.tsv/json`
- `downstream_evidence_summary.json/md`

标准下游分类：

- `DOWNSTREAM_CEM_PASS`
- `DOWNSTREAM_RL_PASS`
- `DOWNSTREAM_POSTURE_FAIL`
- `DOWNSTREAM_MOTION_BINDING_FAIL`
- `DOWNSTREAM_CEM_FAIL`
- `DOWNSTREAM_RL_FAIL`
- `DOWNSTREAM_NOT_RUN`

S6 结果必须回写 registry，但保持为 downstream evidence。
