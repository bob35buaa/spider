# E168 Box021 人工冻结与配对 RL export

日期：2026-07-17

状态：Box021 全 28 条人工结论已冻结，13 条可用 source 及其相反 person OmniRetarget partner 已完成 RL export。

## 人工结论

- 全量：`reviewed=28`，`USE=13`，`DO_NOT_USE=15`。
- 质量标签：`NO_ISSUE=10`，`MINOR_ACCEPTABLE=3`，`MAJOR_ISSUE=15`。
- 本轮新增可用：`box021_20231020_019_p2`、`022_p1`、`022_p2`、`023_p1`、`023_p2`。
- 本轮其余 7 条均为 `DO_NOT_USE/MAJOR_ISSUE`。
- canonical 快照：`results/E168/s6_downstream/cem/eval/manual_review/e168_user_visual_review.tsv`。

人工结论是 source 是否进入 RL export 的最终依据；数值门控告警保留在指标和 notes 中，但不覆盖用户的逐条视频结论。

## Excel

| 范围 | reviewed | USE | DO_NOT_USE | formulas | errors |
|---|---:|---:|---:|---:|---:|
| remaining12 | 12 | 5 | 7 | 396 | 0 |
| 022 pair | 2 | 2 | 0 | 96 | 0 |
| all28 | 28 | 13 | 15 | 874 | 0 |

三张表分别位于：

- `results/E168/s6_downstream/cem/eval/box021_remaining12_available/E168_box021_remaining12_available_case_metrics.xlsx`
- `results/E168/s6_downstream/cem/eval/box021_20231020_022_pair/E168_box021_20231020_022_pair_case_metrics.xlsx`
- `results/E168/s6_downstream/cem/eval/box021_all28_reviewed/E168_box021_all28_reviewed_case_metrics.xlsx`

## RL export

导出入口：

```bash
.venv/bin/python workspace/core4d/scripts/experiments/E168/export_box021_user_approved_rl.py
```

主目录：`results/E168/s6_downstream/rl_export/`。

- 标准 S6 source：`13/13 RL_EXPORT_READY`。
- downstream evidence：`13/13 DOWNSTREAM_CEM_PASS`。
- partner：`13/13 pass`，全部复用现有 E168 Stage2b 结果，无需重跑 OmniRetarget。
- paired export：`13/13 PAIR_COMPLETE + RL_EXPORT_READY`。
- partner variant：`omnirt_v1=11`，`omnirt_v2=2`；v2 为 `034_p2` 和 `019_p1` 两个既有 rescue。
- 人工拒绝项进入 ready export：`0/15`。

主要入口文件：

- `results/E168/s6_downstream/rl_export/rl_export_input.tsv`
- `results/E168/s6_downstream/rl_export/paired_rl_export_input.tsv`
- `results/E168/s6_downstream/rl_export/partner_omnirt/rl_partner_omnirt_manifest.tsv`

partner manifest 对齐 E167 的字段前缀，并追加 Stage2b provenance、pair status 和 SHA。每条 source 均绑定同一 sequence 的相反 person；converted、OmniRetarget output、retargeted、trimmed、trim-window 共 65 个 SHA 已从磁盘重新计算并与 manifest 一致。

## 验证

- exporter `py_compile` 通过。
- 三张 xlsx LibreOffice 重算均为零公式错误。
- source、evidence、partner、paired 四个表均为 13 个唯一 source，且与 canonical `USE` 集合精确相等。
- 所有 source 必要文件和 partner Stage2b/verify 文件存在且非空。
- `git diff --check` 通过。
