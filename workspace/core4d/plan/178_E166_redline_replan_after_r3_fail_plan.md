# E166-R3b 修订计划：C-R3 未过后的分病因红线重审

日期：2026-06-18
前置结果：`log/214_E166_redline_predictive_results.md`
状态：**不启动 Phase 1/2/3 GPU**，先修正 Phase0 假设。

## Context

E166 原计划把 `jerk/acc` 作为进入脚约束/平滑 CEM 的硬门：若平滑度与 downstream staggered 的相关性强于接触且 `|Spearman rho|>=0.6`，再进入 GPU 消融。

实测 E166-R3：
- 8/8 handoff 已算 Tier-1 指标；7/8 有 downstream staggered 标签（`box026_139_p1` 未训练）。
- contact 几乎不预测 downstream：`isaac_both_contact_frac` / `isaac_contact_iou` 的 `|rho|=0.027`。
- 平滑/速度指标确实强于 contact，但最高 `|rho|` 只有 `ankle_acc_max=0.505`，未达 `0.6`。
- `box023_person2` 是平滑但失败的例外；它失败主因不是抖动，而是低接触/手髋自碰撞类 embodiment pathology。
- 去掉 box023 的敏感性检查中，`trackbody_jerk_p95 rho=-0.725`、`ankle_acc_max rho=-0.783`、`obj_speed_max rho=-0.667`，说明“平滑预测力”可能只在 contact/self-collision 已过筛的子集内成立。

因此原 E166 的因果链需要改成 **分病因红线**：

1. 先识别并隔离 “非平滑主因” case（box023 类 contact/self-collision/pathology）。
2. 再在剩余 contact-feasible case 上验证脚/平滑预测力。
3. 只有分层后红线通过，才恢复 A/B CEM 与下游 RL 消融。

## Claims

| # | Claim | 最低证据 |
|---|---|---|
| C-R3b-1 | 下游失败至少有两类主因：平滑/脚不可执行 vs contact/self-collision pathology | box023 被 contact/self-collision 指标判为 pathology；在该类被隔离后，平滑/速度指标对剩余 case 的 `|rho|>=0.6` |
| C-R3b-2 | foot redline 需要比当前 `foot_ground_dev` 更有效 | 新 foot metric 与 staggered 或 ee_body 脚失败占比的 `|rho|>=0.5`，且不再近常数 |
| C-R3b-3 | Tier-1 redline 必须可解释，不能只靠 n=7 阈值过拟合 | 输出每条触发原因；阈值不能只由单个极值 case 决定；precision/recall 仍 `>=0.75` |

## 改动

1. 扩展 `eval_E166_redline_predictive.py`：
   - 输出 subgroup correlations：`all_labeled`、`exclude_contact_pathology`、`exclude_box023_sensitivity`。
   - 引入 pathology flags：
     - low Isaac contact / low IoU。
     - E165 E1 on-rails `init_net_force` / self-collision proxy（若可用）。
     - strict eval 中 hand/object contact regression 指标。
   - 输出 `redline_pathology_split.tsv` 和 `redline_subgroup_correlations.tsv`。
2. 改进 foot metrics：
   - stance segment 内 foot XY drift 改为“短窗口/每秒漂移”而非整段累计位移，避免长 clip 天然更大。
   - foot height consistency 改为相对地面 percentile + stance variance/P95，不再用 `max(abs(z-ground_z))` 这种近常数指标。
   - 可选读取 SUGAR `failed_windows.csv`，把 `ee_body_pos` 脚失败占比作为 supervised diagnostic target。
3. 更新 `redline_summary.md`：
   - 明确 all-case C-R3 不过。
   - 明确分层后是否恢复 E166 GPU。

## 成功标准

继续 GPU 的硬门改为同时满足：

| 条件 | 门槛 |
|---|---|
| all-case 报告不隐藏 box023 | 必须保留，且仍标注原 C-R3 未过 |
| pathology split 可解释 | box023 被非平滑 pathology 捕获；触发原因写入表格 |
| contact-feasible 子集 smoothness/velocity 预测力 | 至少一个脚/平滑指标 `|rho|>=0.6`，且强于 contact |
| foot metric 有效 | 至少一个 foot metric `|rho|>=0.5` 或能解释 `ee_body_pos` 脚失败占比 |
| Tier-1 PR | precision/recall 均 `>=0.75`，且阈值触发原因非单一极值过拟合 |

若上述失败：E166 不进入 CEM/RL，转向数据/pathology 分流和 SUGAR 侧 curriculum/contact 方案。

## 命令

```bash
bash workspace/core4d/scripts/eval/wrappers/eval_E166_redline.sh
```

修订后仍是 training-free、本机 CPU 分析，不使用 GPU，不触碰远程。

## 决策

当前状态：**等待 R3b 修订分析**。在 R3b 通过前，不运行 `run_E166_remote.sh`，不启动 9 条 CEM 或 12 条 SUGAR RL。
