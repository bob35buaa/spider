# 01 数据布局

v3 把代码、原始数据、运行输出和 legacy 目录分开管理。

## 代码目录

```text
workspace/core4d/docs/data_construction_v3/
workspace/core4d/scripts/data_construction_v3/
```

这两个目录进入 git。

`scripts/data_construction_v3/` 内部按功能分层：

```text
lib/                  公共工具
orchestration/        端到端编排
stages/s0_environment/
stages/s1_raw_contact/
stages/s2_templates/
stages/s3_retarget/
stages/s4_gate_visual_qc/
stages/s5_handoff/
stages/s6_downstream/
state/                状态注册表
migration/            legacy/manual 导入
qa/                   release audit、smoke、复现性检查
```

根目录只保留 `README.md` 和功能子目录，不再保留旧的平铺 `.py/.sh` 入口或 symlink。运行命令必须使用对应子目录路径，例如 `orchestration/run_pipeline.py`、`qa/run_smoke_suite.py`、`stages/s1_raw_contact/run_raw_contact.py`。详细说明见 `workspace/core4d/scripts/data_construction_v3/README.md`。

## 原始数据

用户通过 `CORE4D_RAW_ROOT` 指向本机原始数据：

```text
${CORE4D_RAW_ROOT}/
  human_object_motions/<date>/<seq>/
  object_models/<category>/<object>_m.obj
```

v3 不负责同步 raw data，也不假设 raw data 存在于固定挂载点。

## 运行输出

默认：

```text
workspace/core4d/results/<run_id>/
  config_resolved.json
  run_manifest.json
  environment_check.json
  git_state.json
  registries/
  inputs/
  s0_environment/
  s1_raw_contact/
  s2_templates/
  s3_retarget/
  s4_gate_visual_qc/
  s5_handoff/
  s6_downstream/
  archive_legacy/        可选；只放整理前的旧路径，不作为新脚本默认输入
```

`<run_id>` 应包含日期、模式和简短任务名，例如：

```text
20260601_full_from_raw_medium_boxes
20260601_resume_box026_ref_fk_batch
```

正式实验建议使用 `E###` 作为 `<run_id>`，例如 `workspace/core4d/results/E108/`。临时 smoke 可以使用其它 run root，但需要保留的结果必须整理回同一个 `E###` 目录，并放入 `s0_environment/` 到 `s6_downstream/` 对应阶段目录。

## Registry 位置

```text
registries/
  case_state_registry.tsv
  case_state_registry.json
  retarget_variant_registry.tsv
  retarget_variant_registry.json
```

registry 是 v3 状态索引。运行结果不进 git，但 registry 必须足够指向 evidence。

## Legacy 目录

| 目录 | 规则 |
|---|---|
| `${HOLOSOMA_REPO}/workspace/v3/data_construction` | 不删除、不改写；默认不读取。 |
| `${HOLOSOMA_REPO}/workspace/v3/data_construction_v2` | 不删除、不改写；默认不读取。 |
| `workspace/core4d/data_preprocess` | legacy Stage2b 入口；不作为 v3 canonical 入口。 |

必须使用 legacy 文件时，执行显式 import，复制到当前 run 的 imported snapshot，并在 manifest 记录原路径。
