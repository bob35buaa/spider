# 13 需求追踪

本文件把本轮固定 Core4D 数据构建 v3 时确认过的用户需求，映射到当前实现、文档和验证证据。它不是新的设计入口；真正运行仍以 `README.md`、`02_pipeline_stages.md` 和脚本为准。

## 追踪表

| 需求 | 当前实现 | 验证证据 |
|---|---|---|
| 新流程先放在 SPIDER repo，不继续写入 Holosoma 旧 `workspace/v3/data_construction*` | 代码在 `workspace/core4d/scripts/data_construction_v3/`；文档在 `workspace/core4d/docs/data_construction_v3/`；默认 run root 为 `${HOLOSOMA_REPO}/workspace/v3/data_construction_v3_runs` | `audit_pipeline_release.py` 检查 v3 脚本/文档和 run root ignore；`run_release_checks.sh --no-smoke` 通过 |
| v3 脚本不能继续平铺堆在一个目录里，要按功能组织 | 实现文件已拆到 `lib/`、`orchestration/`、`stages/`、`state/`、`migration/`、`qa/`；根目录只保留 `README.md` 和功能子目录，不保留平铺脚本或 symlink | `workspace/core4d/scripts/data_construction_v3/README.md`；release audit `script_structure_documented`；reorg 后 `orchestration/run_release_checks.sh --no-smoke/--with-smoke` 均通过 |
| 支持 `full-from-raw` 和 `resume-from-summary` 两种模式 | `run_pipeline.py --mode full-from-raw|resume-from-summary`；resume 输入会复制到 `imported_snapshots/resume_inputs` 并记录 sha256 | compact smoke 覆盖 `pipeline_full_from_raw`、`verify_full_from_raw`、`pipeline_resume_from_summary`、`verify_resume_from_summary` |
| 用户自己提供 `CORE4D_Real` 路径 | `check_environment.py`、`run_pipeline.py`、`run_smoke_suite.py`、`run_release_checks.sh` 均支持 `--core4d-raw-root` / `CORE4D_RAW_ROOT` | `run_release_checks.sh --with-smoke` 使用显式 raw root 通过 |
| SPIDER 与 OmniRetarget/hsretargeting 环境分离 | S0 检查 SPIDER Python 和 retargeting Python；legacy Stage2b wrapper source Holosoma 环境后优先用 `$RETARGET_PYTHON_BIN` 或 `$CONDA_PREFIX/bin/python` | `check_environment.py` 的 `retarget_python_imports`；真实 Stage2b smoke 中 retarget Python import 通过；`audit_pipeline_release.py` 检查 legacy Python guard |
| 代码和文档进 git；运行 results 不进 git | `.gitignore` 忽略 `results/` 与 `workspace/v3/data_construction_v3_runs/`；脚本输出默认写 run root | `audit_pipeline_release.py` 检查 `.gitignore`；`git status` 中 v3 run 输出未进入工作区 |
| 不隐式读取/写入 legacy `workspace/v3/data_construction` 和 `data_construction_v2` | `11_legacy_migration.md` 明确 legacy 边界；导入 legacy 必须显式用 `import_legacy_snapshot.py` | compact smoke 覆盖 manual seed/import；release audit 检查必要 legacy 迁移文档存在 |
| source scene missing 不能跳过，必须进入 template backlog 或补 template | `build_or_audit_templates.py` 将 missing scene 写为 `template_status=backlog` 和 `recommended_action=build_box_source_template`；`--apply-build` 可自动构建 box source template | `audit_pipeline_release.py` 检查 template backlog 逻辑；S2 template visual render smoke 通过 |
| box template 自动化；非 box 必须人工审查 | `build_or_audit_templates.py` 只自动处理 `object_category=box`；非 box 输出 `manual_review_required` | `04_scene_template_policy.md` 和 `03_manifest_schema.md` 记录边界；release audit 检查必要脚本/文档 |
| E103 robot inertial / scene 污染进入硬审计 | S2 审计 source template 的 robot inertial、object collision/mesh AABB、MuJoCo load；S4 复查 target scene 与 `scene_act` 的 pollution | `build_or_audit_templates.py`、`run_target_gate.py`；release audit 检查 S2/S4 inertial audit；真实 Stage2b smoke 的 S4 gate pass |
| raw contact 支持 3cm/5cm 两档，输出两份候选 | `run_raw_contact.py --thresholds-m 0.03,0.05` 同时写 `raw_contact_candidates_3cm/5cm` 与 `raw_contact_pass_3cm/5cm` | 5cm 多物体 smoke 证明 `--stage2b-contact-label 5cm` 进入 S2/S3；release audit 检查双阈值输出 |
| `stage2b_contact_label` 决定后续用 3cm 还是 5cm，不得写死 3cm | `run_pipeline.py --stage2b-contact-label 3cm|5cm` 选择 S2/S1b/S3 输入；S1 仍同时输出两档 | 5cm 多物体 smoke 中 S3 manifest 14 rows，包含 box026；`verify_reproducibility.py` 通过 |
| retarget algorithm/variant 必须版本化，参数显式暴露 | `register_retarget_variant.py` 注册 `omnirt_original`、`omnirt_v1`、`omnirt_v1_fingertip_replacement`；S3 manifest 写 solver/converter git sha 与 `params_json` | compact smoke 的 `retarget_variant_registry` step；original variant dry-run smoke 显示 `PENDING_VARIANT_ADAPTER` |
| `REPLACE_WRIST_WITH_FINGERTIP` 是 input rewrite 参数，不是 solver 内部隐式默认 | `omnirt_v1_fingertip_replacement` 独立 variant；`omnirt_v1` 默认 `replace_wrist_with_fingertip=false` | `08_retarget_variants.md`；release audit 检查默认 `omnirt_v1/ref_fk` |
| `ref_fk` 是默认 target route；`adaptive` 和 `fingertip_aware` 是平级可选 route | `run_pipeline.py` 默认 `target_variant_id=["ref_fk"]`；S3 后按 `(retarget_variant_id, target_variant_id)` 分叉 | release audit 检查默认 route；registry/S5 按双轴 key 归并 |
| E098 是全 route 基础 contract | `geometry.py` 固化 3D face/contact source helper；S4 gate 记录 replay metrics；文档 `10_diagnostic_contracts.md` | release audit 检查 E098/E099-E101 文档 contract；completion audit 第 10 项 |
| E099-E101 只作为 `fingertip_aware` route 的 hard contract | `build_fingertip_route_diagnostics.py` 汇总 E099-E101；`run_stage2b.py` 只在 `target_variant_id=fingertip_aware` 时硬检查 route diagnostics | compact smoke expected-fail 覆盖非 `ref_fk` execute guard；release audit 检查 fingertip contract |
| 当前 legacy Stage2b execute 不能假装执行 `adaptive/fingertip_aware` | `run_stage2b.py --execute` 对非 `ref_fk` 直接拒绝 | compact smoke 的 `execute_rejects_non_ref_fk_target_route` pass；release audit 检查该 guard |
| S5 要导出下游 CEM 可用配置，external target route 必须校验 target NPZ | `export_cem_overrides.py` 生成 `cem_override_manifest` 和 YAML；`ref_fk` 写 `contact_hdmi_target_source=ref_fk`，external route 校验 target path、sha256、shape、finite 值 | compact smoke 的 `cem_override_handoff` pass；release audit 检查 CEM override adapter |
| 数据状态管理要区分 raw contact、template、Stage2b、target gate、CEM、RL | `case_state_registry.tsv` 字段覆盖各阶段；S3 后按 `(case_id, retarget_variant_id, target_variant_id)` 区分 | compact smoke 覆盖 state matrix、handoff、downstream merge；`verify_reproducibility.py` 检查 registry evidence |
| 已有真实历史结果要纳入，可用 git 追踪的小型 TSV 缓存状态 | `workspace/core4d/data_construction_v3/existing_cases.tsv` 使用 `case_state_registry` schema，纳入 Box004 trusted positives、E105/E106/E107 clean 结果，以及 E079-E081 的 Box023/Box025 legacy CEM cache；大文件仍只通过路径引用 | `build_existing_cases_seed.py` 可重建；`import_legacy_snapshot.py --require-existing-evidence` 验证通过；release audit 检查 seed 文件存在 |
| 下游 CEM/RL 失败不能反向改写 raw/template/target gate 失败 | `record_downstream_evidence.py` 只写 S6 evidence；`update_case_state_registry.py` 合并 downstream 字段但不改写上游事实 | compact smoke state matrix 中 CEM/RL pass 状态不改变 S1-S5 判定；`06_failure_taxonomy.md` 明确边界 |
| 结果可复现要依赖 config、manifest、git sha、环境检查和命令记录 | `run_pipeline.py` 写 `config_resolved.json`、`run_manifest.json`、`git_state.json`；`verify_reproducibility.py` 重算 config hash 并检查 stage manifest/evidence | release checks 与多个 smoke 的 `verify_reproducibility.py` 均通过 |
| 新机器要有统一检查入口 | `run_release_checks.sh` 默认跑 py_compile、legacy wrapper syntax check、release audit；有 raw data/SMPLX 时可追加 compact smoke | `run_release_checks.sh --no-smoke` 和 `--with-smoke` 均通过 |
| 文档使用中文说明 | v3 docs 标题和主要说明为中文，保留 schema/命令/接口名等技术标识 | 2026-06-02 中文标题收口；release checks 通过 |

## 当前计划内边界

- `omnirt_original` 已 pin exact commit，但严格 original execute 还需要单独 checkout/adapter。
- `adaptive` / `fingertip_aware` 目前可生成 manifest、route diagnostics 和 CEM handoff adapter；真实 Stage2b execute adapter 仍需后续实现。
- 自动 source template 构建只覆盖 box 类物体；非 box 仍需人工审查。
- v3 固化的是 S1-S5 数据构建和 S6 evidence schema；完整 RL-ready positive 仍需要具体 CEM/RL 结果补证。
