# E168 Box021 失败机制分析

日期：2026-07-17

状态：28 条人工审查结果完成离线机制分析；未重跑 CEM，未影响 E168 生产任务。

## Scope

- Box021 人工标签：`USE=13`，`DO_NOT_USE=15`。
- 证据：E167A 对齐指标、15 条失败视频 timeline、5 条代表性 ref/sim timeline、CEM root NPZ gate health、28 条 effective config。
- 人工结论作为质量真值；numeric gate 只用于诊断，不覆盖用户标签。

## Artifacts

- 报告：`report/E168/E168_box021_failure_analysis.md`
- 分析脚本：`scripts/experiments/E168/analyze_box021_failures.py`
- 机器可读资产：`report/E168/assets/box021_failure_analysis/`
- 核心表：`group_comparison.tsv`、`cem_health.tsv`、`cem_health_comparison.tsv`、`manual_failure_taxonomy.tsv`。

## Main Findings

1. Box021 人工失败率为 `15/28=53.6%`；`20231018` cohort 为 `12/12 DO_NOT_USE`，失败明显按动作批次集中。
2. 主失败是 lower-body/object collision 与 invalid support。失败中 `14/15` 命中 lower-body numeric failure；人工 taxonomy 中 `9/15` 以此为首要类型，`14/15` 至少包含该症状。
3. 最强人工失败区分指标为 trackbody jerk 与 ankle jerk，AUC 均为 `0.918`；随后是 EEF position error `0.892`、EEF orientation error `0.882`、ankle acceleration `0.877`、root position error `0.872`。
4. 失败组更短更快：时长中位数 `2.93s vs 3.67s`，object speed max `2.05m/s vs 1.36m/s`。`20231018` 是 hard reference cohort，但 date 与动作内容仍有混杂。
5. CEM fallback 不是主失败分类器：fallback mean AUC 仅 `0.549`；9/15 失败 case fallback mean `<=0.10`。`033_p1/p2` 等明显站箱动作几乎无 fallback，证明当前 gate 把任务非法动作判为可行。
6. 28 条 config 的关键缺口完全一致：leg-object、nonhand-support、stability、foot-slip、foot-ground、smooth 均关闭；safety gate 排除全部 lower body；posture/body-z 只约束 z，无法阻止脚交叉、踩箱、借箱支撑和 root roll/pitch 失真。
7. 当前 numeric gate 抓住 `15/15` 人工失败，但误伤 `6/13` 人工可用；lower-body 单项召回 `14/15`，hand penetration 单项只抓 `1/15` 且误伤 3 条可用。

## Decision

算法优先级：

1. P0：lower-body/object 与 non-hand support hard feasibility。
2. P0：stance-foot XY/yaw/ground 与 root/support/terminal-upright gate。
3. P1：可行性成立后叠加 contact-preserving CEM smooth/B2 postprocess。
4. P1：对短时、高速、极端 lateral reach 做 reference retiming、projection 或 adaptive horizon。
5. P2：release gate 增加 invalid support、jerk、ankle acceleration、root/EEF tracking；阈值需跨 date/sequence/object 验证，不能在当前 28 条上过拟合。

推荐最小消融集：`030_p2`、`033_p1`、`028_p2`、`034_p2`、`019_p1`，加 `023_p1/p2` 两条人工可用 control。先因子化验证 lower-body/nonhand gate 与 stance/root gate，再加 smooth/reference arm。

## Verification

- `analyze_box021_failures.py`：`py_compile` 与真实运行通过。
- 输入/人工标签断言：`28 rows`，`15 failed / 13 usable`。
- failure taxonomy 精确覆盖 15 条 rejected case，无多无少。
- 报告 21 个相对链接全部存在。
- `git diff --check` 通过。

## Analysis Errors Fixed

| 错误 | 尝试次数 | 解决方案 |
|---|---:|---|
| `csv.DictWriter` 因源 row 含额外字段拒绝写表 | 1 | 显式使用 `extrasaction="ignore"`，输出字段仍由固定 schema 控制 |
| 批量 ffmpeg 抽帧时进程读取 shell loop stdin，下一条 case ID 首字符丢失 | 1 | ffmpeg 增加 `-nostdin`，15 条 timeline 重新生成并逐条核对 |
