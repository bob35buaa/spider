# 05 可复现性

v3 不要求运行结果进 git。可复现性来自配置、manifest、git sha、环境检查和 deterministic stage outputs。

## 必须记录

每个 run 至少记录：

- resolved config；
- command line；
- spider git sha/dirty；
- holosoma git sha/dirty；
- retarget variant registry；
- case state registry；
- raw root path；
- stage manifest；
- evidence path；
- schema version。

## `full-from-raw`

从 raw 数据开始重建。成功证据：

- inventory 从 `CORE4D_RAW_ROOT` 生成；
- raw contact 3cm/5cm 重新计算；
- `config_resolved.json` 记录 `stage2b_contact_label=3cm|5cm`，用于说明 S2/S1b/S3 消费的是哪一档 `raw_contact_pass_<label>.tsv`；
- source template 重新 build/audit 或从 v3 clean template registry 读取；
- Stage2b 按 retarget variant × target route 执行；
- S4/S5 manifest 完整；
- S6 若已有 CEM 结果，必须生成 `s6_downstream/rl_export/rl_export_input.tsv` 作为 RL 导出的路径索引；
- registry 更新。

## `resume-from-summary`

只能从 v3 registry、stage manifest 或显式 imported snapshot 继续。

入口：

```bash
workspace/core4d/scripts/data_construction_v3/orchestration/run_pipeline.py \
  --mode resume-from-summary \
  --run-id <run_id> \
  --run-root "$DATA_CONSTRUCTION_RUN_ROOT" \
  --resume-registry "$RUN_DIR/imported_snapshots/<snapshot_id>/imported_case_state_registry.tsv"
```

可选追加：

```bash
--resume-stage2b-manifest <stage2b_manifest.tsv>
--resume-target-gate-manifest <target_gate_manifest.tsv>
--resume-retarget-variant-registry <retarget_variant_registry.tsv>
```

resume 输入会被复制到当前 run 的 `imported_snapshots/resume_inputs/`，并写入 `resume_input_manifest.json`。manifest 会同时记录 source/copied path、source/copied sha256 和字节数。当前 run 只读取复制后的文件，保证后续复查不依赖原路径继续变化。

`run_pipeline.py --mode resume-from-summary` 会在复制前做 strict validation：已完成状态必须有存在的 `evidence_root`；registry、Stage2b manifest、target gate manifest 和 retarget variant registry 的 `schema_version` 必须兼容；Stage2b manifest 还必须包含 OmniRetarget 原始输出路径字段。校验失败时直接停止，不生成可交接 handoff。

允许跳过阶段的条件：

- evidence path 存在；
- schema version 兼容；
- config hash 或关键参数一致；
- output hash 或 manifest hash 一致；
- previous decision 不是 stale legacy hard label。

## legacy 导入

必须显式执行：

```text
legacy path -> imported snapshot -> v3 registry
```

入口：

```bash
workspace/core4d/scripts/data_construction_v3/migration/import_legacy_snapshot.py \
  --input-tsv "$LEGACY_OR_MANUAL_STATE_TSV" \
  --out-dir "$RUN_DIR/imported_snapshots" \
  --snapshot-id <snapshot_id> \
  --legacy-root "$LEGACY_ROOT"
```

导入后记录：

- `source_type=legacy_import`
- `source_ref=<legacy path>`
- import time；
- import script；
- import decision；
- 任何 known risk。

导入脚本只产生 snapshot，不直接改 registry。合并必须再显式执行：

```bash
workspace/core4d/scripts/data_construction_v3/state/update_case_state_registry.py \
  --registry-dir "$RUN_DIR/registries" \
  --input-tsv "$RUN_DIR/imported_snapshots/<snapshot_id>/imported_case_state_registry.tsv"
```

不允许隐式读取 legacy 目录。

## 手工 seed

已人工确认可信的历史状态可以作为 `manual_seed` 导入，但必须走同一套 snapshot 机制，不能直接手改当前 registry。

先生成标准 TSV 模板：

```bash
workspace/core4d/scripts/data_construction_v3/state/write_manual_seed_template.py \
  --out-tsv "$RUN_DIR/inputs/manual_seed_template.tsv" \
  --write-help-md
```

填写后导入：

```bash
workspace/core4d/scripts/data_construction_v3/migration/import_legacy_snapshot.py \
  --input-tsv "$RUN_DIR/inputs/manual_seed.tsv" \
  --out-dir "$RUN_DIR/imported_snapshots" \
  --snapshot-id <snapshot_id> \
  --source-type manual_seed \
  --source-ref <source_note> \
  --require-existing-evidence

workspace/core4d/scripts/data_construction_v3/state/update_case_state_registry.py \
  --registry-dir "$RUN_DIR/registries" \
  --input-tsv "$RUN_DIR/imported_snapshots/<snapshot_id>/imported_case_state_registry.tsv"
```

规则：

- 任何非 `not_run` 状态必须有 `evidence_root`；
- `current_decision` 可以留空，导入时会重算；
- `source_ref` 要写清楚人工状态来自哪个 log、eval summary 或审查表；
- E103 前被污染 template 影响的结果不能作为 clean hard seed。

## 结果不进 git

不进 git：

- `results/`
- run root；
- NPZ；
- MP4；
- large CSV；
- visualizations。

进 git：

- docs；
- scripts；
- small configs；
- schema；
- tests；
- example fixtures。

## 完成证明

一个 case 进入 RL-ready positive 前，必须有：

- clean template evidence；
- Stage2b evidence；
- target gate evidence；
- visual QC evidence；
- CEM/RL downstream evidence；
- registry row；
- route-specific contract evidence。

## run 级自检

每次 S5 输出后，建议运行：

```bash
workspace/core4d/scripts/data_construction_v3/qa/verify_reproducibility.py \
  --run-dir "$RUN_DIR"
```

该脚本检查：

- `config_resolved.json` 的 `config_hash` 是否可重算一致；
- `run_manifest.json` 的 `output_root`、`schema_version` 和命令返回码；
- `case_state_registry.tsv` 中非 `not_run` 状态是否有存在的 `evidence_root`；
- `retarget_variant_registry.tsv` 的 id 唯一性和 `params_json` 可解析性；
- S1 inventory、3cm/5cm raw contact、S2 template、S3 Stage2b manifest、S4 target gate manifest、S5 candidate bank 是否存在且 schema 兼容；
- 如果存在 S6 downstream evidence manifest，则检查其 schema 和 CEM/RL 关键字段。

输出：

```text
s5_handoff/reproducibility/reproducibility_report.json
s5_handoff/reproducibility/reproducibility_report.md
```

`status=pass` 才说明该 run 具备基本的 resume/复查条件。这个检查不证明 CEM/RL 已成功；S6 存在时，只证明下游证据被规范记录且能和 registry/handoff 对齐。若 run 内存在 `visual_qc_render_manifest.tsv`，自检还会检查 render error 以及 pass row 的 MP4/sheet 是否存在且非空。

若 run 内存在 `s6_downstream/rl_export/rl_export_input.tsv`，自检会额外检查 `RL_EXPORT_READY` rows 的 `scene_act`、`trajectory` 和 `cem_result_npz` 是否存在且非空。RL 端不得通过扫描 CEM 目录自行决定输入。

## 已有真实历史结果 seed

可信历史结果用一个小型 git-tracked seed 固化：

```text
workspace/core4d/data_construction_v3/existing_cases.tsv
```

该 TSV 使用 `case_state_registry` 同一 schema，只保存状态、指标摘要和证据路径，不复制 CEM/视频/NPZ 大文件。当前纳入范围：

- Box004 已确认正例：E092/E094/E096b 中仍可信的 clean Box004 结果；
- Box023/Box025 历史 CEM cache：E079/E080 的 legacy numeric/case-window 结果，以及 E081 的 leg-object strict proxy；同一 person2 以 E081 覆盖 E079/E080；
- E105 clean Box026 非重复 route；
- E106 clean Box026 30-candidate batch；
- E107 clean Box021 gate 和 selected-4 full CEM；
- E103 之前被 template/inertial bug 污染的 Box021/Box026 旧结论不纳入。

可用下面命令重新生成：

```bash
workspace/core4d/scripts/data_construction_v3/migration/build_existing_cases_seed.py
```

导入到某个 run 的 registry：

```bash
workspace/core4d/scripts/data_construction_v3/migration/import_legacy_snapshot.py \
  --input-tsv workspace/core4d/data_construction_v3/existing_cases.tsv \
  --out-dir "$RUN_DIR/imported_snapshots" \
  --snapshot-id existing_cases_seed \
  --source-type manual_seed \
  --require-existing-evidence

workspace/core4d/scripts/data_construction_v3/state/update_case_state_registry.py \
  --registry-dir "$RUN_DIR/registries" \
  --input-tsv "$RUN_DIR/imported_snapshots/existing_cases_seed/imported_case_state_registry.tsv"
```

## smoke suite

新机器上建议先跑 compact smoke suite：

```bash
workspace/core4d/scripts/data_construction_v3/qa/run_smoke_suite.py \
  --run-root /tmp/core4d_dcv3_smoke_suite \
  --core4d-raw-root "$CORE4D_RAW_ROOT" \
  --smplx-model-dir "$SMPLX_MODEL_DIR"
```

该脚本会验证：

- 所有 v3 Python 脚本可编译；
- E098 公共 geometry helper；
- 扩展接口 contract self-test；
- manual seed 模板与导入；
- registry 状态矩阵：raw contact fail、Stage2b pass/CEM 未跑、CEM pass/RL 未跑、CEM+RL pass；
- `resume-from-summary` 会拒绝 evidence 缺失的坏 registry；
- source template visual review package；
- visual QC render package；
- visual QC render summary 至少有 1 条 pass；
- 如果 raw root 和 SMPL-X 可用，则跑一个小样本 `full-from-raw` dry-run、`resume-from-summary` 和两次 `verify_reproducibility.py`。

输出：

```text
smoke_suite_report.json
smoke_suite_report.md
logs/*.log
```

smoke suite 只证明管线关键入口能在当前机器跑通，不等同完整数据集构建完成。
