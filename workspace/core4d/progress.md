# CORE4D 当前进度

> 完整备份：
> [E167–E179 初始恢复](progress_archive/E167_E179_20260724_full_backup.md) ·
> [E179 执行过程](progress_archive/E179_20260724_execution_full_backup.md) ·
> [E179 最终收尾](progress_archive/E179_20260725_closeout_full_backup.md)
>
> 本文件只保留最近完成实验的可靠结论与下一会话入口。

## E179：box023 / E167A no-PRG / Full CEM

### 最终状态

- 计划：
  `plan/197_E179_box023_e167a_no_prg_full_cem_plan.md`。
- 日志：
  `log/242_E179_box023_e167a_no_prg_vs_e173_results.md`。
- E173 box023 完整 CEM-eligible authority=`16`；E179 使用同一 `16`
  条、同 target/retarget/contact mask/rubber hand，paired denominator
  始终为 `16`。
- Full budget=`seed 0, 1024 samples × 32 opt steps`。本地 RTX 5090 GPU0
  完成 `4/4`；A100 GPU `2/3/6/7` 各完成 `3/3`；合计 `16/16`，
  terminal failure=`0`。
- E167A profile parity=`16/16`，no-PRG audit=`16/16`，
  unexpected PRG runtime diagnostics=`0`。
- 最终 completion audit=`11/11 PASS`：paired metrics=`16/16`、
  gate cells=`192`、E173/E179/paired videos=`16/16/16`、
  visual review=`16/16`。

### 最终结果

| 口径 | E173 PRG | E179 no-PRG |
|---|---:|---:|
| Physics 六门 | `13/16` | `9/16` |
| Physics + tracking 十二门 | `7/16` | `4/16` |

- 十二门迁移=`PASS_TO_PASS 3 / PASS_TO_FAIL 4 / FAIL_TO_PASS 1 /
  FAIL_TO_FAIL 8`；McNemar exact `p=0.375`。
- 唯一救回=`021_p1`；新退化=`041_p1` (hand_ori)、
  `021_p2` (lower_body)、`040_p2` (lower_body+hand_ori)、
  `041_p2` (lower_body)。
- lower-body=`14→10/16`，leg penetration paired mean
  `Δ=+0.07363`，bootstrap 95% CI=`[+0.03704,+0.11427]`。
- Object position error paired mean `Δ=-0.752cm`，
  bootstrap 95% CI=`[-1.540,-0.108]cm`；其它主要 tracking CI 多数跨 `0`。
- 16 条 paired 视频均抽取 `10/30/50/70/90%` 五帧并实际复核。
  `021_p2/040_p2/041_p2` 的 no-PRG 下肢更贴近或跨在箱体上方，
  支持 lower-body 退化。
- 数值结论=`PRG_BETTER`，视觉结论=`SUPPORTS_PRG_BETTER`。
  box023 保留 E170 PRG，不晋级 no-PRG。

### Canonical evidence

- 评测：
  `results/E179/s6_downstream/eval/full/E179_vs_E173_report.md`。
- Summary：
  `results/E179/s6_downstream/eval/full/e179_eval_summary.json`。
- 视觉 authority：
  `results/E179/s6_downstream/render/full/visual_review.tsv`。
- Completion：
  `results/E179/completion_audit/completion_audit.json`。
- Full manifest：
  `results/E179/s6_downstream/manifests/cem_full_manifest.tsv`。
- Runtime logs：`logs/E179/cem/`。

### 已解决问题

- Inactive PRG diagnostics 错误序列化：修复 runtime，fresh canary `3/3`。
- OSMesa 初始化失败：切换本机已验证的 `egl`。
- Remote sync inventory 漏传：补传并复用同一 remote run root。
- `rsync` 连续超时：切换 `scp`，remote runtime `12/12`、merge `16/16`。
- ffprobe 初始 glob 错误：按 manifest canonical video path 重检，failures=`0`。
- scp probe 临时 NPZ 与 canonical SHA 一致后已精确删除，正式结果保留。

### 收尾

- Log 242、Tracker、log INDEX 已更新；Tracker 描述长度 `43≤80`。
- Progress 完整执行记录已归档，活跃页由 `314` 行压缩为 `81` 行。
- Python compile、shell `bash -n`、`git diff --check` 与最终 completion
  audit 均 PASS。
- 提交范围已隔离为 E179 scene/config/runtime/scripts/docs；E172/E173 partner
  补齐改动与 `finalize_reused_partner_rl.py` 保持 unstaged。scene snapshot
  本地为 `145` files / `3.0MB`，但 `workspace/core4d/results` 位于仓库 symlink
  之后，Git 报 `pathspec ... is beyond a symbolic link`，不能从该路径 force-add；
  active 16-case scene/config 的 `64` 个复现文件已直接 staged。
- C0–C6 全部通过；实验计算、评测、视频复核和文档均已闭合。
- 下一会话无需重跑 E179；如继续研究，只做 P/R/G 子组件消融。

### 最终提交前复核

- `01:11` method/no-PRG audit 再次 `16/16 PASS`，completion audit 再次
  `11/11 PASS`。
- 首轮 staged `git diff --check` 报告三个新增文件 EOF 多一个空行；需删除
  `audit_completion.py`、`render_paired_results.py`、`run_E179_local.sh`
  的冗余 EOF blank line 后以 `set -e` 重跑。
- 上述三个 EOF blank line 已删除。
- 以 `set -euo pipefail` 严格重跑 staged whitespace、Python compile、
  shell syntax、method audit、completion audit、Tracker/INDEX/progress
  断言，结果 `STRICT_FINAL_VALIDATION=PASS`。
- 最终 staged scope=`89` files，仅含 E179 的 `64` 个 active scene/config、
  runtime fix、E179 scripts、plan/log/Tracker/INDEX/progress archives；
  E172/E173 partner 补齐与 `finalize_reused_partner_rl.py` 未进入 staged diff。
- 技能自带 `check-complete.sh` 在当前 GNU grep 多文件输出上把两个 `0`
  拼成 `0\n0`，触发整数表达式错误；这是技能检查脚本问题。Tracker/progress
  已由独立断言确认更新，不把该辅助脚本输出当作实验失败。
