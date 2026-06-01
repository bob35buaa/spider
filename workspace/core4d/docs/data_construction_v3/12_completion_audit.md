# 12 完成度审查

审查时间：2026-06-01

本文件对照 `workspace/core4d/plan/116_data_construction_v3_reproducible_pipeline_plan.md` 的 4.1 必做项，记录当前 v3 数据构建管线的落地证据、验证命令和剩余风险。

## 总体结论

当前 v3 已具备可复现数据构建的主干能力：

- 支持 `full-from-raw` 与 `resume-from-summary` 两种模式；
- 默认不读写 legacy `workspace/v3/data_construction*`；
- S1 同时输出 3cm/5cm raw-contact 候选，并用 `stage2b_contact_label` 显式选择进入 S2/S3 的档位；
- source scene missing 进入 template backlog，不再被当作数据失败直接跳过；
- retarget variant 与 target route 是两个独立轴；
- 默认 target route 是 `ref_fk`，只绑定 E098 基础 contract；
- `fingertip_aware` 是可选 target route，显式选择后必须通过 E099-E101 route diagnostics；
- case 状态由 registry 管理，S3 之后按 `(case_id, retarget_variant_id, target_variant_id)` 分叉；
- downstream CEM/RL 作为 S6 evidence，不反向定义 S1-S5 数据构建成败。

## 计划 4.1 对照

| # | 必做项 | 当前状态 | 证据 |
|---|---|---|---|
| 1 | 任意机器可通过本地配置指定 repo、原始数据和结果目录 | 已实现 | `run_pipeline.py`、`check_environment.py`、`init_workspace.sh` 均支持显式 `--spider-repo`、`--holosoma-repo`、`--core4d-raw-root`、`--run-root`、`--smplx-model-dir`；文档见 `00_environment.md`、`01_data_layout.md` |
| 2 | 从原始 CORE4D mocap 重新生成 inventory、raw-contact、template、Stage2b、target gate 和 handoff | 主干已实现，真实 Stage2b 已做 box004 与 box026/ref_fk execute 验证，批量覆盖和非 ref_fk execute 仍需后续扩展 | `run_pipeline.py --mode full-from-raw` 覆盖 S0-S5；compact smoke 和 5cm multi-object smoke 均通过；`smoke_execute_stage2b_box004_r3` 与 `smoke_execute_stage2b_box026_ref_fk_5cm` 均真实执行了 convert、OmniRetarget、trim、contact mask、SPIDER preprocess、verify、target gate 和 visual render |
| 3 | 所有阶段都有明确输入、输出、manifest schema、失败分类和可重跑命令 | 已实现 | `README.md`、`02_pipeline_stages.md`、`03_manifest_schema.md`、`06_failure_taxonomy.md`、`07_troubleshooting.md`；每个阶段脚本输出 TSV/JSON/summary |
| 4 | template 缺失进入 backlog 并补 template，不再当作数据失败 | 已实现 | `build_or_audit_templates.py`、`render_template_review_package.py`；`04_scene_template_policy.md` 明确 source scene missing -> backlog/build；非 box 为 manual review |
| 5 | E103 robot inertial / scene 污染类问题进入硬审计 | 已实现 | `build_or_audit_templates.py` 审计 robot inertial、object collision/mesh AABB、MuJoCo load；`run_target_gate.py` 复查 target scene 与 robot inertial pollution；文档见 `04_scene_template_policy.md` |
| 6 | `results/` 不进 git，靠 config、manifest、git sha、环境检查和命令记录重建 | 已实现 | `.gitignore` 忽略 `workspace/core4d/results/` 和 v3 run root；`run_pipeline.py` 写 `config_resolved.json`、`run_manifest.json`、`git_state.json`；`verify_reproducibility.py` 复查 config hash、schema、manifest、evidence |
| 7 | 数据状态管理，记录 raw contact、template、Stage2b、target gate、CEM、RL；支持 skip/resume 和 from-scratch | 已实现 | `update_case_state_registry.py`、`export_handoff.py`、`record_downstream_evidence.py`、`import_legacy_snapshot.py`；compact smoke 覆盖 state matrix、manual seed/import、坏 resume 拒绝、resume-from-summary |
| 8 | retarget algorithm/variant 管理：solver 版本、输入改写、trim、SPIDER 参数写入 manifest | 已实现；`omnirt_original` 已 pin 到 initial public release，但执行需单独 checkout/adapter | `register_retarget_variant.py` 注册 `omnirt_original`、`omnirt_v1`、`omnirt_v1_fingertip_replacement`；`omnirt_original` 指向 Holosoma git `9c238cf80f531c0e65d818348c3c1a5cc2764f5b` 并带 `requires_solver_checkout=true`；`run_stage2b.py` 输出 variant 参数、`replace_wrist_with_fingertip`、OmniRetarget 输出路径、trimmed NPZ、SPIDER trajectory 等字段；文档见 `08_retarget_variants.md` |
| 9 | 预留过滤器接口和算法接口，后续新增规则/版本不重造 pipeline | 已实现 | `interfaces.py` 定义 `CandidateFilter`、`RetargetAdapter`、`TemplateBuilder`、`TargetGate`、`Visualizer` 等 Protocol 和统一 result contract；文档见 `09_extension_interfaces.md` |
| 10 | E098 固化为全 route gate/schema，E099-E101 固化为 `fingertip_aware` 可选 contract | 已实现 | `geometry.py` 固化 E098 3D face/contact source helper；`run_stage2b.py` 只对 `target_variant_id=fingertip_aware` 强制 route diagnostics；`build_fingertip_route_diagnostics.py` 汇总 E099-E101；文档见 `10_diagnostic_contracts.md` |

## 已验证命令

### 编译

```bash
find workspace/core4d/scripts/data_construction_v3 -name '*.py' -print0 | \
  xargs -0 python3 -m py_compile
```

结果：通过。

### 默认 compact smoke suite

```bash
workspace/core4d/scripts/data_construction_v3/qa/run_smoke_suite.py \
  --run-root /tmp/core4d_dcv3_smoke_suite_release_audit \
  --core4d-raw-root /mnt/a0ccc676-9496-49f8-a861-f8a1797dec52/mocap_data/CORE4D/CORE4D_Real \
  --smplx-model-dir /mnt/a0ccc676-9496-49f8-a861-f8a1797dec52/mocap_data/smplx
```

结果：

- `smoke_suite_report.json`: `status=pass`；
- warnings 为空；
- 覆盖 py_compile、E098 geometry self-test、extension interface self-test、manual seed/import、registry/downstream state matrix、坏 resume 拒绝、source template visual render、visual QC render、默认 3cm full-from-raw dry-run、resume-from-summary；
- 新增覆盖代码库级 release audit：检查 v3 必要脚本/文档、`.gitignore`、默认 `omnirt_v1/ref_fk`、legacy Stage2b 非 `ref_fk` execute guard、E098/E099-E101 contract、双阈值 raw contact、template backlog、E103 inertial audit 和 CEM override adapter；
- 新增 `run_release_checks.sh` 作为新机器统一检查入口：无 raw data 时跑 py_compile、legacy wrapper syntax check 和 codebase release audit；有 raw data/SMPLX 时可追加 compact smoke suite；
- 新增 `13_requirements_traceability.md`，把用户约束逐条映射到实现和验证证据；
- 新增 `14_release_readiness.md`，集中记录当前可交接状态、推荐入口、最新验证和计划内边界；
- 新增覆盖 default retarget variant registry 断言：`omnirt_original` 必须有 exact source reference 且标记 `requires_solver_checkout=true`；
- 新增覆盖 `run_stage2b.py --execute` 的非 `ref_fk` target route expected-fail，避免 legacy execute adapter 被误用于 `adaptive/fingertip_aware`；
- 新增覆盖 S5 CEM override handoff adapter：`ref_fk` route 生成 `contact_hdmi_target_source=ref_fk`，`fingertip_aware` route 生成 external target override 并校验 target NPZ；
- `smoke_full_from_raw` 和 `smoke_resume_from_summary` 的 `verify_reproducibility.py` 均为 `status=pass`，errors/warnings 为空。

### 脚本结构重组验证

2026-06-02 已将 `workspace/core4d/scripts/data_construction_v3/` 的实现文件按功能拆分：

- `lib/`：公共 helper、schema、几何与扩展接口；
- `orchestration/`：一键 pipeline 与 release checks；
- `stages/s0_environment/` 到 `stages/s6_downstream/`：各阶段入口；
- `state/`：case registry 与 handoff；
- `migration/`：legacy/manual seed 导入；
- `qa/`：smoke、audit、render 等验证工具。

根目录已删除旧的平铺 `.py/.sh` 入口和 symlink，只保留 `README.md` 与功能子目录；历史命令必须改为新的子目录路径。新的结构说明见 `workspace/core4d/scripts/data_construction_v3/README.md` 和 `01_data_layout.md`。

### release checks 统一入口

无 raw-data 模式：

```bash
workspace/core4d/scripts/data_construction_v3/orchestration/run_release_checks.sh \
  --run-root /tmp/core4d_dcv3_release_checks_existing_seed \
  --no-smoke
```

结果：通过；release audit `status=pass`，62/62 checks pass；compact smoke 按预期跳过。新增检查会确认脚本目录结构已经文档化、根目录没有平铺 `.py/.sh` 脚本或 symlink，且 git-tracked `existing_cases.tsv` seed 存在。

带 compact smoke 模式：

```bash
workspace/core4d/scripts/data_construction_v3/orchestration/run_release_checks.sh \
  --run-root /tmp/core4d_dcv3_release_checks_existing_seed_with_smoke \
  --core4d-raw-root /mnt/a0ccc676-9496-49f8-a861-f8a1797dec52/mocap_data/CORE4D/CORE4D_Real \
  --smplx-model-dir /mnt/a0ccc676-9496-49f8-a861-f8a1797dec52/mocap_data/smplx \
  --with-smoke
```

结果：通过；release audit `62/62`；`smoke_suite/smoke_suite_report.json` 为 `status=pass`，20 个步骤全 pass，warnings 为空。

### original variant dry-run smoke

```bash
workspace/core4d/scripts/data_construction_v3/orchestration/run_pipeline.py \
  --mode full-from-raw \
  --run-id smoke_original_variant_requires_checkout \
  --run-root /tmp/core4d_dcv3_original_variant_smoke \
  --core4d-raw-root /mnt/a0ccc676-9496-49f8-a861-f8a1797dec52/mocap_data/CORE4D/CORE4D_Real \
  --smplx-model-dir /mnt/a0ccc676-9496-49f8-a861-f8a1797dec52/mocap_data/smplx \
  --queue object-key \
  --object-keys box004 \
  --max-sequences 1 \
  --sample-count 100 \
  --retarget-variant-id omnirt_original \
  --target-variant-id ref_fk
```

随后运行：

```bash
workspace/core4d/scripts/data_construction_v3/qa/verify_reproducibility.py \
  --run-dir /tmp/core4d_dcv3_original_variant_smoke/smoke_original_variant_requires_checkout
```

结果：

- `retarget_variant_registry.tsv` 中 `omnirt_original` 指向 `9c238cf80f531c0e65d818348c3c1a5cc2764f5b`，`requires_solver_checkout=true`；
- S3 `decision_counts={'stage2b_variant_requires_solver_checkout': 1}`，`pipeline_ready=0`；
- registry summary 中 `current_decision_counts` 包含 `PENDING_VARIANT_ADAPTER=1`；
- S5 `candidate_decision_counts` 包含 `VARIANT_PENDING=1`，handoff 为 `HANDOFF_PENDING_TEMPLATE_OR_VARIANT`；
- `verify_reproducibility.py` 为 `status=pass`，errors/warnings 为空。

### 5cm 多物体 smoke

```bash
workspace/core4d/scripts/data_construction_v3/orchestration/run_pipeline.py \
  --mode full-from-raw \
  --run-id smoke_phase_d_box004_021_026_5cm \
  --run-root /tmp/core4d_dcv3_phase_d_multiobject_5cm_smoke \
  --core4d-raw-root /mnt/a0ccc676-9496-49f8-a861-f8a1797dec52/mocap_data/CORE4D/CORE4D_Real \
  --smplx-model-dir /mnt/a0ccc676-9496-49f8-a861-f8a1797dec52/mocap_data/smplx \
  --queue object-key \
  --object-keys box004,box021,box026 \
  --max-case-persons 18 \
  --sample-count 100 \
  --stage2b-contact-label 5cm \
  --retarget-variant-id omnirt_v1 \
  --target-variant-id ref_fk
```

随后运行：

```bash
workspace/core4d/scripts/data_construction_v3/qa/verify_reproducibility.py \
  --run-dir /tmp/core4d_dcv3_phase_d_multiobject_5cm_smoke/smoke_phase_d_box004_021_026_5cm
```

结果：

- `config_resolved.json` 记录 `stage2b_contact_label=5cm`；
- `raw_contact_pass_3cm.tsv`: 5 rows，object 分布为 box004 3 / box021 2 / box026 0；
- `raw_contact_pass_5cm.tsv`: 14 rows，object 分布为 box004 4 / box021 9 / box026 1；
- S2 `template_backlog.tsv`: 5 rows，包含 box026 1；
- S3 `stage2b_manifest_omnirt_v1_ref_fk.tsv`: 14 rows，包含 box026 1；
- `verify_reproducibility.py`: `status=pass`，errors/warnings 为空。

这证明 `--stage2b-contact-label 5cm` 确实使 5cm pass 子集进入 S2/S3，而不是仍然写死使用 3cm。

### 真实 Stage2b execute smoke

```bash
workspace/core4d/scripts/data_construction_v3/orchestration/run_pipeline.py \
  --mode full-from-raw \
  --run-id smoke_execute_stage2b_box004_r3 \
  --run-root /tmp/core4d_dcv3_execute_stage2b_smoke \
  --core4d-raw-root /mnt/a0ccc676-9496-49f8-a861-f8a1797dec52/mocap_data/CORE4D/CORE4D_Real \
  --smplx-model-dir /mnt/a0ccc676-9496-49f8-a861-f8a1797dec52/mocap_data/smplx \
  --queue object-key \
  --object-keys box004 \
  --max-sequences 1 \
  --sample-count 300 \
  --stage2b-contact-label 3cm \
  --retarget-variant-id omnirt_v1 \
  --target-variant-id ref_fk \
  --execute-stage2b
```

随后生成 visual QC render 并复查：

```bash
workspace/core4d/scripts/data_construction_v3/stages/s4_gate_visual_qc/render_visual_qc_package.py \
  --target-gate-manifest-tsv /tmp/core4d_dcv3_execute_stage2b_smoke/smoke_execute_stage2b_box004_r3/stage_s4_gate_visual_qc/omnirt_v1/ref_fk/target_gate_manifest.tsv \
  --out-dir /tmp/core4d_dcv3_execute_stage2b_smoke/smoke_execute_stage2b_box004_r3/stage_s4_gate_visual_qc/omnirt_v1/ref_fk/visual_qc_render \
  --max-video-frames 120 \
  --overwrite

workspace/core4d/scripts/data_construction_v3/qa/verify_reproducibility.py \
  --run-dir /tmp/core4d_dcv3_execute_stage2b_smoke/smoke_execute_stage2b_box004_r3
```

结果：

- S0 `retarget_python_imports=pass`，使用 `/home/ubuntu/.holosoma_deps/miniconda3/envs/hsretargeting/bin/python`；
- S3 `stage2b_manifest_omnirt_v1_ref_fk.tsv` 为 1 row，`stage2b_status=pass`；
- `converted_npz`、`omniretarget_output_npz`、`trimmed_npz`、`spider_trajectory`、`contact_mask_npz`、`verify_summary` 均存在且非空；
- S4 `target_gate_status=pass`；
- S4 visual render `render_status_counts={'pass': 1}`；
- S5 handoff 为 `HANDOFF_REVIEW_VISUAL_QC`；
- `verify_reproducibility.py` 为 `status=pass`，errors/warnings 为空。

该 smoke 同时暴露并修复了一个环境漏洞：外层 SPIDER `.venv` 会抢占裸 `python`。现在 `pipeline.sh` 在 source Holosoma retargeting env 后优先使用 `$CONDA_PREFIX/bin/python` 或显式 `RETARGET_PYTHON_BIN`，S0 也会检查 retargeting Python 可导入 `smplx` 与 `holosoma_retargeting`。

### box026 5cm 真实 Stage2b execute smoke

```bash
workspace/core4d/scripts/data_construction_v3/orchestration/run_pipeline.py \
  --mode full-from-raw \
  --run-id smoke_execute_stage2b_box026_ref_fk_5cm \
  --run-root /tmp/core4d_dcv3_execute_stage2b_box026_smoke \
  --core4d-raw-root /mnt/a0ccc676-9496-49f8-a861-f8a1797dec52/mocap_data/CORE4D/CORE4D_Real \
  --smplx-model-dir /mnt/a0ccc676-9496-49f8-a861-f8a1797dec52/mocap_data/smplx \
  --queue object-key \
  --object-keys box026 \
  --max-case-persons 2 \
  --sample-count 300 \
  --stage2b-contact-label 5cm \
  --retarget-variant-id omnirt_v1 \
  --target-variant-id ref_fk \
  --execute-stage2b
```

随后生成 visual QC render 并复查：

```bash
workspace/core4d/scripts/data_construction_v3/stages/s4_gate_visual_qc/render_visual_qc_package.py \
  --target-gate-manifest-tsv /tmp/core4d_dcv3_execute_stage2b_box026_smoke/smoke_execute_stage2b_box026_ref_fk_5cm/stage_s4_gate_visual_qc/omnirt_v1/ref_fk/target_gate_manifest.tsv \
  --out-dir /tmp/core4d_dcv3_execute_stage2b_box026_smoke/smoke_execute_stage2b_box026_ref_fk_5cm/stage_s4_gate_visual_qc/omnirt_v1/ref_fk/visual_qc_render \
  --max-video-frames 120 \
  --overwrite

workspace/core4d/scripts/data_construction_v3/qa/verify_reproducibility.py \
  --run-dir /tmp/core4d_dcv3_execute_stage2b_box026_smoke/smoke_execute_stage2b_box026_ref_fk_5cm
```

结果：

- S1 5cm 选择出 2 个 box026 case-person；3cm 为 1 pass / 1 review，5cm 为 2 pass；
- S2 两个 source templates 均为 clean，template visual render `pass=2`；
- S3 `stage2b_manifest_omnirt_v1_ref_fk.tsv` 两行均为 `stage2b_status=pass`：
  - `box026_20231018_039_p1` -> `dcv3_omnirt_v1_ref_fk_box026_20231018_039_p1`；
  - `box026_20231018_039_p2` -> `dcv3_omnirt_v1_ref_fk_box026_20231018_039_p2`；
- 两行均有 `converted_npz`、`omniretarget_output_npz`、`trimmed_npz`、`spider_trajectory`、`contact_mask_npz`、`verify_summary`；
- S4 target gate 两行均为 `target_gate_status=pass`；
- S4 visual render `render_status_counts={'pass': 2}`；
- `verify_reproducibility.py` 为 `status=pass`，errors/warnings 为空，summary 中 `visual_qc_render_status_counts={'pass': 2}`。

该 smoke 证明 E103 template 修复后的 box026 clean source 可以通过 v3 的 `omnirt_v1/ref_fk/5cm` 路线完成真实 Stage2b 和机器 gate。当前不把它外推到 `adaptive` 或 `fingertip_aware`，因为这两个 route 的真实 execute adapter 尚未实现。

## 剩余风险

1. `omnirt_original` 的 exact source 已 pin 到 Holosoma initial public release，但该历史提交没有当前 CORE4D converter；严格 original 对照需要后续单独 checkout 并补 original Stage2b adapter。
2. 当前真实 Stage2b execute 已覆盖 `box004_20231003_2_082_p1 / omnirt_v1 / ref_fk / 3cm` 和 `box026_20231018_039_p1,p2 / omnirt_v1 / ref_fk / 5cm`；更大批量构建仍需后续逐批验证。
3. 当前 legacy execute adapter 只支持 `target_variant_id=ref_fk`。`adaptive` 和 `fingertip_aware` 可以生成 manifest 和 route diagnostics，但真实 execute 需要专门 Stage2b target adapter。
4. 当前自动 template 构建策略只面向 box 类物体；非 box 仍为 `manual_review_required`，这是计划内边界。
5. downstream CEM/RL 的 evidence schema 已固化，但 v3 不把 CEM/RL 成败作为 S1-S5 数据构建的唯一成败定义；完整 RL-ready positive 仍需额外 S6 证据。
