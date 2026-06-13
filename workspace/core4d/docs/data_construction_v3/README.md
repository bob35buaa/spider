# Core4D 数据构建 v3

本目录是 Core4D 数据构建 v3 的规范入口。v3 的目标是：在任意机器上，基于用户提供的 `CORE4D_Real` 原始数据路径，稳定重建从 raw mocap 到 SPIDER/CEM/RL handoff 的数据链路。

v3 不是继续往旧目录追加结果。以下目录只作为 legacy：

- `${HOLOSOMA_REPO}/workspace/v3/data_construction`
- `${HOLOSOMA_REPO}/workspace/v3/data_construction_v2`
- `workspace/core4d/data_preprocess`

新流程默认不写入这些旧目录，也不隐式读取它们。必须使用旧结果时，只能通过显式 import 或显式 legacy path，并写入 run manifest。

## 运行模式

| 模式 | 用途 |
|---|---|
| `full-from-raw` | 从 `CORE4D_Real` 原始 mocap 和 object mesh 开始，完整重建 inventory、raw contact、template、Stage2b、gate、handoff。 |
| `resume-from-summary` | 从 v3 自己的 `case_state_registry`、stage manifest 或显式导入 snapshot 继续，跳过已有且 evidence 有效的阶段。 |

显式导入已有/legacy 状态时，先生成 imported snapshot，再合并 registry：

```bash
workspace/core4d/scripts/data_construction_v3/migration/import_legacy_snapshot.py \
  --input-tsv "$LEGACY_OR_MANUAL_STATE_TSV" \
  --out-dir "$RUN_DIR/imported_snapshots" \
  --snapshot-id <snapshot_id> \
  --legacy-root "$LEGACY_ROOT"

workspace/core4d/scripts/data_construction_v3/state/update_case_state_registry.py \
  --registry-dir "$RUN_DIR/registries" \
  --input-tsv "$RUN_DIR/imported_snapshots/<snapshot_id>/imported_case_state_registry.tsv"
```

## 默认目录

代码和文档放在 spider：

```text
workspace/core4d/docs/data_construction_v3/
workspace/core4d/scripts/data_construction_v3/
```

运行结果放机器本地 run root：

```text
workspace/core4d/results/<run_id>/
```

正式 Core4D 数据构建实验使用 `workspace/core4d/results/E###/`。临时 smoke 可以显式改 `--run-root`，但任务结束前若要保留结果，必须整理回 `workspace/core4d/results/E###/` 的 canonical layout。运行结果、NPZ、MP4、CSV、可视化不进 git。进入 git 的是文档、wrapper、schema、registry 工具和小型配置。

## 阶段总览

```text
S0  environment/config check
S0b case_state_registry init
S1  raw inventory + raw contact 3cm/5cm
S2  source scene template build/audit
S3  OmniRetarget/SPIDER preprocess by retarget_variant_id x target_variant_id
S4  target gate + visual QC
S5  candidate bank + handoff manifest + hand_collision_variant_id scene adapter
S6  CEM/RL downstream evidence
```

S0-S2 在不同 retarget variant / target route 之间共享；S3 之后必须带 `retarget_variant_id` 和 `target_variant_id`。S5/CEM 之后还可带机器人手部碰撞体轴 `hand_collision_variant_id`，默认 `sphere5cm`；不同组合不能互相覆盖输出。

## 默认路线

默认 target route 是 `ref_fk`。`fingertip_aware` 与 `ref_fk`、`adaptive` 平级，不是全局默认。

- E098 是全路线基础 contract。
- E099-E101 是 `fingertip_aware` route 的专属 contract。
- 选择 `fingertip_aware` 时，S3 必须显式提供 E099-E101 route diagnostic manifest；否则该 row 会停在 `REJECT_FINGERTIP_ROUTE_CONTRACT`，不能进入 Stage2b。

## 最小入口

先初始化 run workspace：

```bash
workspace/core4d/scripts/data_construction_v3/orchestration/init_workspace.sh <run_id> \
  --holosoma-repo "$HOLOSOMA_REPO" \
  --run-root "$DATA_CONSTRUCTION_RUN_ROOT" \
  --core4d-raw-root "$CORE4D_RAW_ROOT"
```

初始化会生成：

- `s0_environment/environment_check.{json,md}`
- `registries/retarget_variant_registry.{tsv,json}`
- `registries/case_state_registry.{tsv,json}`

内置 retarget variants 只描述 OmniRetarget/input rewrite 版本；target route 独立选择。默认组合是 `omnirt_v1/ref_fk`，只要求 E098。`fingertip_aware` 是显式 target route，要求 E098-E101。

也可以用一条 orchestration 命令跑 S0-S5 的 dry-run 流程：

```bash
workspace/core4d/scripts/data_construction_v3/orchestration/run_pipeline.py \
  --mode full-from-raw \
  --run-id <run_id> \
  --run-root "$DATA_CONSTRUCTION_RUN_ROOT" \
  --core4d-raw-root "$CORE4D_RAW_ROOT" \
  --smplx-model-dir "$SMPLX_MODEL_DIR"
```

从已有 v3 registry 或 imported snapshot 继续时，使用：

```bash
workspace/core4d/scripts/data_construction_v3/orchestration/run_pipeline.py \
  --mode resume-from-summary \
  --run-id <run_id> \
  --run-root "$DATA_CONSTRUCTION_RUN_ROOT" \
  --resume-registry "$RUN_DIR/imported_snapshots/<snapshot_id>/imported_case_state_registry.tsv"
```

`resume-from-summary` 会先做 strict validation：已完成状态必须有存在的 `evidence_root`，schema/version 必须兼容，Stage2b manifest 必须包含 OmniRetarget 原始输出路径字段。通过后才会把输入 registry 和可选 stage manifests 复制到当前 run 的 `imported_snapshots/resume_inputs/`，并在 `resume_input_manifest.json` 记录 source/copied path 与 sha256，再生成本 run 自己的 registry、handoff 和 run manifest。它不会回写原始 snapshot，也不会隐式读取 legacy 目录。

如果要导入人工确认的历史状态，先生成 seed 模板再导入 snapshot：

```bash
workspace/core4d/scripts/data_construction_v3/state/write_manual_seed_template.py \
  --out-tsv "$RUN_DIR/inputs/manual_seed_template.tsv" \
  --write-help-md

workspace/core4d/scripts/data_construction_v3/migration/import_legacy_snapshot.py \
  --input-tsv "$RUN_DIR/inputs/manual_seed.tsv" \
  --out-dir "$RUN_DIR/imported_snapshots" \
  --snapshot-id <snapshot_id> \
  --source-type manual_seed \
  --source-ref <source_note>
```

默认只跑 `omnirt_v1/ref_fk`。如果要启用 `fingertip_aware`，必须额外传入该 route 的 diagnostic manifest，或显式让 pipeline 构建该 manifest：

```bash
workspace/core4d/scripts/data_construction_v3/orchestration/run_pipeline.py \
  --mode full-from-raw \
  --run-id <run_id> \
  --run-root "$DATA_CONSTRUCTION_RUN_ROOT" \
  --core4d-raw-root "$CORE4D_RAW_ROOT" \
  --smplx-model-dir "$SMPLX_MODEL_DIR" \
  --stage2b-contact-label 3cm \
  --retarget-variant-id omnirt_v1 \
  --target-variant-id ref_fk \
  --target-variant-id fingertip_aware \
  --route-diagnostic-tsv "$RUN_DIR/s1_raw_contact/fingertip_route_diagnostics.tsv"
```

基于已有 E099/E100/E101 产物构建 route diagnostic manifest 的入口是：

```bash
STAGE2B_CONTACT_LABEL=3cm

workspace/core4d/scripts/data_construction_v3/stages/s1_raw_contact/build_fingertip_route_diagnostics.py \
  --input-tsv "$RUN_DIR/s1_raw_contact/raw_contact/raw_contact_pass_${STAGE2B_CONTACT_LABEL}.tsv" \
  --out-dir "$RUN_DIR/s1_raw_contact/fingertip_route_diagnostics"
```

也可以在 `run_pipeline.py` 中显式开启：

```bash
workspace/core4d/scripts/data_construction_v3/orchestration/run_pipeline.py \
  --mode full-from-raw \
  --run-id <run_id> \
  --run-root "$DATA_CONSTRUCTION_RUN_ROOT" \
  --core4d-raw-root "$CORE4D_RAW_ROOT" \
  --smplx-model-dir "$SMPLX_MODEL_DIR" \
  --stage2b-contact-label 3cm \
  --retarget-variant-id omnirt_v1 \
  --target-variant-id ref_fk \
  --target-variant-id fingertip_aware \
  --build-fingertip-route-diagnostics
```

S1 从 raw 重建 inventory 和 3cm/5cm raw-contact 候选：

```bash
RUN_DIR="${DATA_CONSTRUCTION_RUN_ROOT}/<run_id>"

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

`run_pipeline.py` 默认用 `--stage2b-contact-label 3cm` 选择 `raw_contact_pass_3cm.tsv` 进入 S2 template audit/build、S1b `fingertip_aware` diagnostic 和 S3 Stage2b 队列。若本轮实验要放宽到 5cm，显式传 `--stage2b-contact-label 5cm`；两档候选仍都会写出并同步 registry，只有被选中的那一档继续进入 S2/S3。

然后同步 registry：

```bash
workspace/core4d/scripts/data_construction_v3/state/update_case_state_registry.py \
  --registry-dir "$RUN_DIR/registries" \
  --from-inventory-tsv "$RUN_DIR/s1_raw_contact/inventory/inventory.tsv" \
  --evidence-root "$RUN_DIR/s1_raw_contact/inventory"

workspace/core4d/scripts/data_construction_v3/state/update_case_state_registry.py \
  --registry-dir "$RUN_DIR/registries" \
  --from-raw-contact-tsv "$RUN_DIR/s1_raw_contact/raw_contact/raw_contact_candidates_3cm.tsv" \
  --raw-contact-label 3cm \
  --evidence-root "$RUN_DIR/s1_raw_contact/raw_contact"

workspace/core4d/scripts/data_construction_v3/state/update_case_state_registry.py \
  --registry-dir "$RUN_DIR/registries" \
  --from-raw-contact-tsv "$RUN_DIR/s1_raw_contact/raw_contact/raw_contact_candidates_5cm.tsv" \
  --raw-contact-label 5cm \
  --evidence-root "$RUN_DIR/s1_raw_contact/raw_contact"
```

S2 审计 source template backlog：

```bash
STAGE2B_CONTACT_LABEL=3cm

workspace/core4d/scripts/data_construction_v3/stages/s2_templates/build_or_audit_templates.py \
  --input-tsv "$RUN_DIR/s1_raw_contact/raw_contact/raw_contact_pass_${STAGE2B_CONTACT_LABEL}.tsv" \
  --core4d-raw-root "$CORE4D_RAW_ROOT" \
  --out-dir "$RUN_DIR/s2_templates"
```

默认只审计，不会写 scene。确认需要自动补 box template 时显式加 `--apply-build`；非 box 永远只进入人工审查。
`run_pipeline.py` 会在 S2 后自动生成 `s2_templates/template_visual_review/`，里面包含 source template 的 MP4/sheet 和 render manifest；手动跑阶段时也可以单独执行：

```bash
workspace/core4d/scripts/data_construction_v3/stages/s2_templates/render_template_review_package.py \
  --template-backlog-tsv "$RUN_DIR/s2_templates/template_backlog.tsv" \
  --out-dir "$RUN_DIR/s2_templates/template_visual_review"
```

```bash
workspace/core4d/scripts/data_construction_v3/state/update_case_state_registry.py \
  --registry-dir "$RUN_DIR/registries" \
  --from-template-backlog-tsv "$RUN_DIR/s2_templates/template_backlog.tsv" \
  --evidence-root "$RUN_DIR/s2_templates"
```

S3 生成按 retarget variant × target route 分叉的 Stage2b 队列。默认只生成 manifest 和 dry-run 脚本，不执行旧 pipeline：

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

`stage2b_manifest` 会显式记录每条 row 的 `converted_npz`、`omniretarget_output_npz`、`trimmed_npz`、`spider_trajectory`、`contact_mask_npz` 和 `verify_summary`。其中 `omniretarget_output_npz` 是 OmniRetarget 算法输出，不是 SPIDER 转换后的 trajectory。

如果短期需要复用 legacy Stage2b 执行链路，必须显式加：

```bash
--execute --allow-legacy-stage2b-wrapper
```

当前 legacy execute adapter 只支持 `target_variant_id=ref_fk`。`adaptive` 与 `fingertip_aware` 可以先生成 manifest、route diagnostics 和 gate 输入，但真实执行必须补专门的 Stage2b target adapter；`run_stage2b.py --execute` 会拒绝非 `ref_fk` route，避免旧 pipeline 假装执行了外部 target route。

然后同步 registry：

```bash
workspace/core4d/scripts/data_construction_v3/state/update_case_state_registry.py \
  --registry-dir "$RUN_DIR/registries" \
  --from-stage2b-manifest-tsv "$RUN_DIR/s3_retarget/omnirt_v1/ref_fk/stage2b_manifest_omnirt_v1_ref_fk.tsv" \
  --evidence-root "$RUN_DIR/s3_retarget/omnirt_v1/ref_fk"
```

S4 运行机器 target gate，并保留 visual QC 为独立状态：

```bash
workspace/core4d/scripts/data_construction_v3/stages/s4_gate_visual_qc/run_target_gate.py \
  --stage2b-manifest-tsv "$RUN_DIR/s3_retarget/omnirt_v1/ref_fk/stage2b_manifest_omnirt_v1_ref_fk.tsv" \
  --out-dir "$RUN_DIR/s4_gate_visual_qc/omnirt_v1/ref_fk"

workspace/core4d/scripts/data_construction_v3/state/update_case_state_registry.py \
  --registry-dir "$RUN_DIR/registries" \
  --from-target-gate-manifest-tsv "$RUN_DIR/s4_gate_visual_qc/omnirt_v1/ref_fk/target_gate_manifest.tsv" \
  --evidence-root "$RUN_DIR/s4_gate_visual_qc/omnirt_v1/ref_fk"
```

如果 Stage2b 还没有实际执行，S4 会明确输出 `target_gate_status=not_run` 和 `failure_mode=stage2b_outputs_missing`，不会伪造 pass。

`run_pipeline.py` 会在 target gate 后自动生成默认 visual QC manifest：`target_gate_status=pass` 的 row 进入 `visual_qc_status=review`，等待人工/LLM release review；未通过机器 gate 的 row 保持 `not_run`。如果手动执行阶段命令，或已有人工/LLM 审查 TSV，可单独生成/导入 visual QC：

```bash
workspace/core4d/scripts/data_construction_v3/stages/s4_gate_visual_qc/render_visual_qc_package.py \
  --target-gate-manifest-tsv "$RUN_DIR/s4_gate_visual_qc/omnirt_v1/ref_fk/target_gate_manifest.tsv" \
  --out-dir "$RUN_DIR/s4_gate_visual_qc/omnirt_v1/ref_fk/visual_qc_render"

workspace/core4d/scripts/data_construction_v3/stages/s4_gate_visual_qc/make_visual_qc.py \
  --target-gate-manifest-tsv "$RUN_DIR/s4_gate_visual_qc/omnirt_v1/ref_fk/target_gate_manifest.tsv" \
  --review-tsv "$RUN_DIR/s4_gate_visual_qc/omnirt_v1/ref_fk/manual_visual_review.tsv" \
  --out-dir "$RUN_DIR/s4_gate_visual_qc/omnirt_v1/ref_fk/visual_qc"

workspace/core4d/scripts/data_construction_v3/state/update_case_state_registry.py \
  --registry-dir "$RUN_DIR/registries" \
  --from-visual-qc-manifest-tsv "$RUN_DIR/s4_gate_visual_qc/omnirt_v1/ref_fk/visual_qc/visual_qc_manifest.tsv" \
  --evidence-root "$RUN_DIR/s4_gate_visual_qc/omnirt_v1/ref_fk/visual_qc"
```

S5 导出 candidate bank 和 handoff manifest：

```bash
workspace/core4d/scripts/data_construction_v3/stages/s5_handoff/export_handoff.py \
  --case-state-registry "$RUN_DIR/registries/case_state_registry.tsv" \
  --stage2b-manifest-tsv "$RUN_DIR/s3_retarget/omnirt_v1/ref_fk/stage2b_manifest_omnirt_v1_ref_fk.tsv" \
  --target-gate-manifest-tsv "$RUN_DIR/s4_gate_visual_qc/omnirt_v1/ref_fk/target_gate_manifest.tsv" \
  --out-dir "$RUN_DIR/s5_handoff"
```

输出 `candidate_bank`、`handoff_manifest`、`rejected_manifest` 和 summary。`handoff_manifest` 默认只包含可交接或待运行的 rows；reject rows 单独进 `rejected_manifest`。

S5 同时可以导出 CEM override 配置：

```bash
workspace/core4d/scripts/data_construction_v3/stages/s5_handoff/export_cem_overrides.py \
  --handoff-manifest-tsv "$RUN_DIR/s5_handoff/handoff_manifest.tsv" \
  --out-dir "$RUN_DIR/s5_handoff/cem_overrides"
```

`ref_fk` route 写成 `contact_hdmi_target_source=ref_fk`；`adaptive` / `fingertip_aware` route 必须带外部 target NPZ，并在导出时校验 target 文件、sha256、shape 和有限值。这个步骤只生成下游 CEM 可消费配置，不改变 S1-S5 的数据构建判定。

S6 记录 CEM/RL 下游证据：

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

S6 只更新 `cem_status`、`rl_status` 和 `downstream_*` 证据字段，不改变 S5 的数据构建 pass/reject 语义。

RL motion export 先生成 S5/S6 join manifest：

```bash
workspace/core4d/scripts/data_construction_v3/stages/s6_downstream/export_rl_inputs.py \
  --handoff-manifest-tsv "$RUN_DIR/s5_handoff/handoff_manifest.tsv" \
  --cem-evidence-tsv "$RUN_DIR/s6_downstream/downstream_evidence_manifest.tsv" \
  --out-dir "$RUN_DIR/s6_downstream/rl_export"
```

下游 RL 导出只读：

```text
$RUN_DIR/s6_downstream/rl_export/rl_export_input.tsv
```

并只消费 `rl_export_decision=RL_EXPORT_READY` 的 rows。这个 TSV 同时包含 `scene_act`、`trajectory`、`contact_mask`、`cem_result_npz`、`source_exp_id` 和 `spider_method_id`，避免 RL 脚本重新猜路径或混淆 CEM/SPIDER 方法版本。

`target_variant_id` 只表示 target route，例如默认 `ref_fk`；CEM/reward/selection 方法版本写入 `spider_method_id`，来源实验编号写入 `source_exp_id`。

S5 后建议跑一次 run 级可复现性检查：

```bash
workspace/core4d/scripts/data_construction_v3/qa/verify_reproducibility.py \
  --run-dir "$RUN_DIR"
```

该检查会验证 `config_hash`、`run_manifest` 命令返回码、registry evidence、S1-S5 关键 manifest 和 schema version。若 run 内存在 visual QC render manifest，还会检查 `render_error`、pass gate 未渲染、以及 pass row 的 MP4/keyframe sheet 是否存在且非空。输出：

- `s5_handoff/reproducibility/reproducibility_report.json`
- `s5_handoff/reproducibility/reproducibility_report.md`

代码库级 release audit 不依赖 CORE4D raw data，可先在新机器上检查 v3 入口、文档、legacy 隔离、默认 route、fingertip contract 和 CEM override adapter 是否齐全：

```bash
workspace/core4d/scripts/data_construction_v3/qa/audit_pipeline_release.py \
  --out-dir /tmp/core4d_dcv3_release_audit
```

推荐的新机器 release check 入口：

```bash
workspace/core4d/scripts/data_construction_v3/orchestration/run_release_checks.sh \
  --run-root /tmp/core4d_dcv3_release_checks
```

如果已配置 `CORE4D_RAW_ROOT` 和 `SMPLX_MODEL_DIR`，该脚本会自动追加 compact smoke suite；也可以显式传入：

```bash
workspace/core4d/scripts/data_construction_v3/orchestration/run_release_checks.sh \
  --run-root /tmp/core4d_dcv3_release_checks \
  --core4d-raw-root "$CORE4D_RAW_ROOT" \
  --smplx-model-dir "$SMPLX_MODEL_DIR" \
  --with-smoke
```

新机器上可以先跑 compact smoke suite：

```bash
workspace/core4d/scripts/data_construction_v3/qa/run_smoke_suite.py \
  --run-root /tmp/core4d_dcv3_smoke_suite \
  --core4d-raw-root "$CORE4D_RAW_ROOT" \
  --smplx-model-dir "$SMPLX_MODEL_DIR"
```

它会验证脚本编译、代码库级 release audit、geometry helper、扩展接口 contract、manual seed、registry 状态矩阵、坏 registry resume 拒绝、visual render、CEM override handoff、小样本 `full-from-raw` dry-run、`resume-from-summary` 和 run 级自检。

## 文档索引

| 文件 | 内容 |
|---|---|
| [00_environment.md](00_environment.md) | 环境、路径、依赖检查 |
| [01_data_layout.md](01_data_layout.md) | raw、repo、run root、结果目录布局 |
| [02_pipeline_stages.md](02_pipeline_stages.md) | S0-S6 阶段输入、输出和硬规则 |
| [03_manifest_schema.md](03_manifest_schema.md) | registry、variant、stage manifest 字段 |
| [04_scene_template_policy.md](04_scene_template_policy.md) | source template 与 target scene 责任边界 |
| [05_reproducibility.md](05_reproducibility.md) | git sha、config hash、resume/from-raw 复现策略 |
| [06_failure_taxonomy.md](06_failure_taxonomy.md) | 失败分类和 diagnostic/hard gate 区别 |
| [07_troubleshooting.md](07_troubleshooting.md) | 常见错误和处理 |
| [08_retarget_variants.md](08_retarget_variants.md) | OmniRetarget/input conversion variant 管理 |
| [09_extension_interfaces.md](09_extension_interfaces.md) | 可复用过滤器、算法、gate、visualizer 接口 |
| [10_diagnostic_contracts.md](10_diagnostic_contracts.md) | E098 全局 contract 与 E099-E101 route contract |
| [11_legacy_migration.md](11_legacy_migration.md) | 旧目录和旧脚本迁移边界 |
| [12_completion_audit.md](12_completion_audit.md) | 当前实现对照计划必做项的完成度审查和剩余风险 |
| [13_requirements_traceability.md](13_requirements_traceability.md) | 用户需求到实现与验证证据的追踪表 |
| [14_release_readiness.md](14_release_readiness.md) | 当前可交接状态、推荐入口、最新验证和计划内边界 |
| [15_hand_collision_variants.md](15_hand_collision_variants.md) | 机器人手部碰撞体 variant 轴、sidecar scene patch 和 CEM A/B 约定 |
