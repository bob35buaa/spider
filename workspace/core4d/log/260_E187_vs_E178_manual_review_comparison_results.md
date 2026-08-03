# E187 vs E178 人工终审配对比较结果

日期：2026-08-03
实验：E187
阶段：S6 manual review comparison
结论：双方均终裁的15条中E187人工USE净`+2`且无USE→DNU；但E187仍有3条PENDING，numeric/物理安全明显退化，整体结论为`MIXED`

## Context

用户已在E187的22条paired视频上填写
`user_manual_review_filled.tsv`。本阶段执行[plan 210](../plan/210_E187_vs_E178_manual_review_comparison_plan.md)，
只读比较E187与E178人工annotation authority，并关联冻结的numeric结果；未修改两份人工表、
Full artifact、reward/grid/P/G或C9治理状态。

## Authority 与覆盖

| 项目 | E178 keep22 | E187 keep22 |
|---|---:|---:|
| authority cases | 22 | 22 |
| annotation rows present | 18 | 22 |
| review status=`reviewed` | 18 | 22 |
| 最终decision | 18 | 19 |
| USE | 8 | 11 |
| DO_NOT_USE | 10 | 8 |
| PENDING/缺失 | 4 | 3 |
| annotation SHA256 | `2d2ef9c8...e223a` | `0b3de9ea...f0161a` |

E178缺失的四条全部是bucket004；它们按未决处理，不能推断为DNU。E187虽然22行均被
viewer保存为`reviewed`，但三行decision仍为`PENDING`，因此“22条看完”不等于
“22条终裁完成”。正式状态保持`PENDING_USER_REVIEW`。

## 人工配对结果

### 双方均有最终裁决的15条

| E178 → E187 | 数量 |
|---|---:|
| USE → USE | 7 |
| **DO_NOT_USE → USE** | **2** |
| USE → DO_NOT_USE | **0** |
| DO_NOT_USE → DO_NOT_USE | 6 |

人工USE从E178的`7/15`增至E187的`9/15`，净`+2`、`+13.3pp`。discordant pair为
2个恢复、0个回归；双侧exact McNemar `p=0.5`。方向是正向的，但样本只有两个
discordant pair，不能宣称统计显著。

按object：

| Object | common decisive | E178 USE | E187 USE | delta | 迁移 |
|---|---:|---:|---:|---:|---|
| bucket003 | 5 | 2 | 3 | +1 | DNU→USE 1、U→U 2、D→D 2 |
| bucket004 | 0 | — | — | — | E178四条均未决；E187为USE2/DNU2 |
| bucket007 | 10 | 5 | 6 | +1 | DNU→USE 1、U→U 5、D→D 4 |

### 两条人工恢复

| Case | 人工迁移 | E187观察 | 关键numeric变化 |
|---|---|---|---|
| `bucket003_20231018_003_p1` | UNUSABLE→MINOR_ACCEPTABLE | “勉强用，中间有几帧没有接触上” | contact `0.526→0.753`、root误差`42.0→14.5cm`、hand误差`31.6→13.4cm`；但leg penetration `0→0.103`，仍FAIL |
| `bucket007_20231003_1_021_p1` | UNUSABLE→CLEAN | 用户直接判CLEAN | hand penetration `0.342→0.132`，tracking略好；但leg penetration `0→0.263`，仍FAIL |

这两条都说明continuation/CoACD路线能改善肉眼动作或手部/跟踪表现，但改善同时伴随
lower-body物理回归。人工恢复不等于12门恢复。

### Quality变化

在15条common-decisive上，按
`UNUSABLE < MAJOR_DEFECT < MINOR_ACCEPTABLE < CLEAN`作描述性排序：

| 方向 | 数量 | 说明 |
|---|---:|---|
| 改善 | 5 | 含2条DNU→USE、2条UNUSABLE→MAJOR_DEFECT但仍DNU、1条MINOR→CLEAN |
| 持平 | 9 | decision/quality均未改变 |
| 回归 | 1 | `bucket007_20231023_075_p2` CLEAN→MINOR_ACCEPTABLE，但仍USE |

不能把UNUSABLE→MAJOR_DEFECT写成“可用恢复”；两者都仍是DO_NOT_USE。

## 未决项与边界

E187仍未终裁：

| Case | E178 | E187 note | E187 numeric |
|---|---|---|---|
| `bucket007_20231003_2_021_p1` | USE | “这个接触不好啊” | PASS |
| `bucket007_20231018_019_p1` | DO_NOT_USE | “不太好说” | FAIL |
| `bucket007_20231018_021_p2` | DO_NOT_USE | 空 | FAIL |

在E178已裁决的18条上，E178共有8条USE；E187当前已经有9条USE。即使三条PENDING最终
全部判DNU，E187仍至少为`9/18`，净`+1`；若其中两条原E178-DNU均恢复，最多净`+4`。
因此未决项会改变改善幅度和是否出现一条USE→DNU，但不会抹掉当前E187 USE数量至少
高于E178一条的事实。由于裁决尚未填写，正式报告仍不得把这个敏感性边界冒充最终迁移表。

另外，E178 bucket004四条没有人工baseline。E187新增的USE2/DNU2是新信息，不是可定义
的E178→E187迁移。

## Numeric 与人工的一致性

| 指标 | E178 | E187 |
|---|---:|---:|
| 已终裁行 | 18 | 19 |
| numeric对人工USE recall | `6/8 = 75.0%` | **`4/11 = 36.4%`** |
| numeric对人工DNU rejection | `10/10 = 100%` | `7/8 = 87.5%` |
| numeric误拒人工USE | 2 | **7** |
| numeric误收人工DNU | 0 | **1** |

E187的7条人工USE/numeric FAIL中有6条包含`lower_body` failure；剩余一条是
`bucket003_20231018_005_p2`，用户标为CLEAN，但numeric因release、hand penetration和
hand orientation拒绝。唯一numeric PASS/人工DNU是
`bucket004_20231003_1_012_p2`，用户理由为“接触很差”。

因此numeric `6/22 vs 8/22`不能直接解释为人工可用度下降。更准确的解释是：

- E187在人工可见动作、接触或tracking上存在净改善；
- 冻结12门对E187的lower-body风险非常敏感，并捕获了人工可能接受或视频中不明显的物理
  穿透；
- numeric gate也漏掉一条用户明确认为接触很差的轨迹；
- 人工和numeric衡量的是不同风险，当前两者都不能覆盖对方。

## 用户DNU原因

E187的8条DNU主要来自三类自由文本：

- 接触差：bucket004两条、bucket007两条（其中一条同时tracking差）；
- tracking差：bucket003两条、bucket007两条（可与接触/摔倒重叠）；
- 摔倒或险些摔倒：bucket007两条。

人工USE中也不是全部CLEAN：3条为MINOR_ACCEPTABLE，备注包括整体较差、短时失接触、
动作不自然和脚滑。人工USE应理解为“可用”，不能等价为无缺陷。

## 结论

E187相对E178的最终科学判断应为`MIXED`：

1. **人工可用度方向改善**：common15净恢复2条，无人工USE→DNU；即使对三条pending做
   最保守敏感性分析，E187在E178已裁18条上的USE数仍至少净+1。
2. **物理安全与效率退化**：numeric 12门净`-2`，lower-body penetration显著恶化，且
   C9 plan-time仍是technical FAIL、仅由用户waiver放行。
3. **不能发布为整体更优**：人工改善尚不统计显著、3条未决、bucket004无E178人工
   baseline，并且manual/numeric错配明显。

这支持后续新实验把“保留continuation带来的人工动作收益，同时修复lower-body穿透与
foot slip”作为目标；不支持回调E187已冻结参数或直接把全部人工USE导出为RL成功。

## 可视化 → 实际观察

本阶段没有生成新视频。人工标签来自[log 259](259_E187_vs_E178_paired_video_review_results.md)
已验证的22/22 paired MP4；用户的逐条备注被原样保留在comparison TSV中。本阶段只做
annotation表统计，因此不重复中帧spot-check，也不修改历史视觉证据。

## Claims

| Claim | 状态 | 证据 |
|---|---|---|
| M1 authority | PASS | 两侧SHA冻结；执行前后不变；E187 22 unique |
| M2 pending-safe | PASS | E178缺失4和E187 PENDING3均保留未决 |
| M3 paired transition | PASS | TSV 22行；common15；2恢复/0回归 |
| M4 quality transition | PASS | 改善5/持平9/回归1，decision边界未混淆 |
| M5 numeric alignment | PASS | 两侧USE recall/DNU rejection与逐case错配完整 |
| M6 governance | PASS | C9仍`FAIL/USER_WAIVED`；status=`PENDING_USER_REVIEW` |

## 结果路径与SHA256

| 内容 | 路径 | SHA256 |
|---|---|---|
| E187 annotation authority | `workspace/core4d/results/E187/s6_downstream/eval/full/user_manual_review_filled.tsv` | `0b3de9ea88e8aa25238a92626b815fa64c9bdff2ec6c2faed392400716f0161a` |
| E178 annotation authority | `workspace/core4d/results/E178/s6_downstream/eval/full/user_manual_review_filled.tsv` | `2d2ef9c854bcad50e39e1a0a2e3f2dd27296fa34c58af9df61c692041b2e223a` |
| 22-row comparison | `workspace/core4d/results/E187/s6_downstream/eval/full/e187_vs_e178_manual_review_comparison.tsv` | `c88d8d482f5b8d5c828d7767bae6b679a7fe79c8314fa47e000e04a7747bf01f` |
| machine summary | `workspace/core4d/results/E187/s6_downstream/eval/full/manual_review_comparison_summary.json` | `2ae268992d720fbf70951c945cd2003e9d8c1bf893f134671d1c39100902facf` |
| readable summary | `workspace/core4d/results/E187/s6_downstream/eval/full/manual_review_comparison_summary.md` | `0975b21bfe97de18749cd7d7f6635f6fad57a8037e4e57bdcc3ef6dd9cb50894` |

## 下一步

1. 用户若愿意，可补齐三条E187 PENDING；补齐后只需重跑canonical wrapper，不能改历史
   E178标签；
2. 在三条补齐前，E187保持`PENDING_USER_REVIEW`；
3. 不自动生成RL export。若要选择下游case，需另行确定人工USE与lower-body safety冲突的
   release policy；
4. 科学迭代应新开实验修lower-body/foot-slip trade-off，不在E187 Full后调参。
