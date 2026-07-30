# E180 分析计划：38 条 RL 标签的指标线性可分性审计

_CORE4D Phase 43 · 2026-07-26 · eval-only，不启动 CEM/RL_

---

## 📋 背景与问题

现有下游 authority 包含 38 条已验证 case：32 条 RL 成功、6 条 RL 失败，
覆盖 Box001/004/021/023/024。用户希望判断单指标或多个指标的线性组合能否：

1. 分开 32 条 RL 成功与 6 条 RL 失败；
2. 同时拒绝未进入 RL 候选的人工/数值失败 case；
3. 形成可解释、可复核、不过度拟合的新 gate 候选。

历史 E166 已出现“7 条训练内完美、leave-one 仅 `3/7`”的过拟合，因此训练内
可分性与泛化可分性必须分开裁决。

## 🎯 Claims

| Claim | 判定标准 |
|---|---|
| C0 authority | 38 条 RL case 标签、输入 variant 与特征来源逐例可追溯；缺失特征不静默填补 |
| C1 single metric | 穷举每个连续指标与方向，报告最佳阈值、混淆矩阵和 margin |
| C2 sparse linear | 对 1–5 个特征的标准化线性模型搜索训练内 hard-margin 可分性 |
| C3 robustness | 报告 LOOCV、按 object leave-one-group-out、bootstrap coefficient stability |
| C4 proxy negatives | 人工 `DO_NOT_USE` / pre-RL numeric fail 单列为 proxy negatives，不冒充 RL fail |
| C5 decision | 只有训练内、LOOCV、group-CV 和 proxy-negative stress test 同时稳健才建议 hard gate |

## 🔬 数据与标签口径

| 集合 | 正例 | 负例 | 用途 |
|---|---|---|---|
| RL-observed | 32 RL success | 6 RL fail | 主分析 |
| RL-observed standardized | 有完整同口径特征的 RL case | 同上 | 模型拟合 |
| Pre-RL stress | RL success | 人工 DNU 或明确 pre-RL numeric fail | 压力测试，不估计 RL 混淆矩阵 |

特征优先取各实验 frozen case metrics 的共同连续字段：physics/contact、
tracking、dynamics、foot slip、object motion、pre-Omni SE(3)。实验 ID、
case ID、对象名、日期、人工标签和既有 pass/fail gate 不作为连续模型输入，
避免身份记忆与标签泄漏。若输入 variant 与 RL 实际 variant 不同，该行只进入
敏感性分析。

## ⚙️ 方法

1. 建立 38-case frozen label manifest 与特征 lineage。
2. 取共同非缺失特征，删除常数、近常数和重复列。
3. 单指标穷举所有相邻值中点及两个方向。
4. 对标准化连续特征拟合：
   - hard/soft-margin linear SVM；
   - L1/L2 logistic regression；
   - 1–5 特征的稀疏组合或递归选择。
5. 对所有候选报告训练集、LOOCV、leave-one-object-out 和 permutation
   baseline；类别不平衡时同时报告 balanced accuracy、failure recall、
   success precision/recall。
6. 将 frozen 模型应用到 pre-RL proxy negatives，报告 coverage 与冲突案例。

任何使用全数据选择特征/阈值后的分数只算 apparent fit；交叉验证必须在每个
fold 内重新做标准化、特征选择和拟合。

## 📊 成功标准

| 等级 | 条件 | 决策 |
|---|---|---|
| A：候选 hard gate | LOOCV 与 object-group CV 均无 RL false negative，failure recall 高，proxy negatives 有稳定拒绝率 | 可进入独立 held-out 验证 |
| B：仅 soft score | 训练内可分或高分，但 CV/系数不稳定 | 只作风险排序 |
| C：不可分 | 稀疏线性模型仍有重叠，或依赖身份/variant 泄漏 | 转向 phase probe/非线性或补特征 |

即使达到 A，也不能在同一 38 条上宣称已获得生产 gate；最终需要新对象/新序列
的独立 held-out RL 标签。

## 📦 产物

| 产物 | 路径 |
|---|---|
| 评估脚本 | `scripts/eval/runners/eval_E180_rl_metric_separability.py` |
| 单元测试 | `scripts/eval/runners/test_eval_E180_rl_metric_separability.py` |
| Frozen labels/features | `results/E180/rl_metric_separability/` |
| 独立报告 | `analysis/RL38_metric_linear_separability_audit_20260726.md` |
| 结果日志 | `log/243_E180_rl38_metric_linear_separability_results.md` |

## 🛡️ 边界

- 不启动新 CEM、RL 或远程任务。
- 不修改 E168–E179 历史 metrics、人工标签、registry 或日志。
- 不把人工 DNU 当作已观测 RL failure。
- 不用 case/object/date one-hot 获得伪线性可分。
- 不以训练集完美分割替代泛化证据。
