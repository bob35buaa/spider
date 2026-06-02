# Core4D 数据构建 v3 脚本结构

本目录按功能分层组织实现文件。根目录只保留本说明和功能子目录；旧的平铺 `.py/.sh` 入口已移除，后续命令必须使用下方的子目录路径。

## 目录分层

| 目录 | 内容 |
|---|---|
| `lib/` | 公共工具：`common.py`、`geometry.py`、`interfaces.py` |
| `orchestration/` | 端到端编排入口：`run_pipeline.py`、`init_workspace.sh`、`run_release_checks.sh` |
| `stages/s0_environment/` | 环境检查 |
| `stages/s1_raw_contact/` | inventory、raw contact、fingertip-aware route diagnostics |
| `stages/s2_templates/` | source template 构建、审计和可视化审查包 |
| `stages/s3_retarget/` | retarget variant registry 和 Stage2b wrapper |
| `stages/s4_gate_visual_qc/` | target gate、visual QC manifest、visual render |
| `stages/s5_handoff/` | candidate handoff 与 CEM override 导出 |
| `stages/s6_downstream/` | CEM/RL downstream evidence，以及 RL export input join manifest |
| `state/` | case state registry 和 manual seed template |
| `migration/` | legacy/manual snapshot 导入、历史 seed 生成 |
| `qa/` | release audit、smoke suite、run 级复现性检查 |

## 入口约定

常用入口：

```bash
workspace/core4d/scripts/data_construction_v3/orchestration/run_pipeline.py ...
workspace/core4d/scripts/data_construction_v3/orchestration/run_release_checks.sh ...
workspace/core4d/scripts/data_construction_v3/qa/run_smoke_suite.py ...
workspace/core4d/scripts/data_construction_v3/migration/build_existing_cases_seed.py
```

新增脚本必须放到对应功能目录；不要再在根目录添加平铺脚本或 symlink。若编排脚本需要按短名调用新脚本，需要同步更新 `lib/common.py` 中的 `SCRIPT_RELATIVE_PATHS`。

运行输出也必须按阶段目录写入当前 run root：

```text
s0_environment/
s1_raw_contact/
s2_templates/
s3_retarget/
s4_gate_visual_qc/
s5_handoff/
s6_downstream/
registries/
```

实验特定的 CEM/RL 结果属于 `s6_downstream/`；不要在 `workspace/core4d/results/E###/` 根目录下新增 `cem/`、`handoff_*`、`registry_*`、`stage2b_*` 这类平铺目录。整理前的历史目录只能放入 `archive_legacy/`。

RL motion export 的输入 TSV 由 `stages/s6_downstream/export_rl_inputs.py` 生成，默认路径为 `s6_downstream/rl_export/rl_export_input.tsv`。RL 端只消费 `rl_export_decision=RL_EXPORT_READY` 的 rows。
