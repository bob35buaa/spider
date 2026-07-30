# E180 结果：38 条 RL case 的指标线性可分性审计

_2026-07-26 · eval-only · 不启动 CEM/RL_

## 🧾 结论

E180 完成，最终裁决为 **B：只保留探索性 soft risk score，不建立 hard
gate**。

- 38 条 authority 为 `32 success / 6 fail`。
- 单指标 exact separator 为 `0`。
- Reference 67 指标中，3 指标 SVM 训练内 `38/38`；LOOCV failure recall
  `4/6`，object-held-out `5/6`，但误拒 success `10/32`。
- Standardized 111 指标中，2 指标 SVM 训练内 `36/36`；LOOCV failure
  recall `5/6`，object-held-out 仅 `1/6`。
- Frozen standardized sparse separator 只拒绝 proxy `4/42=9.5%`。
- 将 42 proxy 加入后，5 指标模型仍接受 `6/48` failure-like；111 指标虽
  训练内 `78/78`，object-held-out failure recall 仅 `22/48`。

训练内线性可分成立；跨 object、真实 failure 与 pre-RL proxy 的联合 hard
gate 不成立。独立报告见
[RL38 audit](../analysis/RL38_metric_linear_separability_audit_20260726.md)。

## 📊 Claim 验证

| Claim | 结果 | 证据 |
|---|---|---|
| C0 authority 可追溯 | ✅ | 38 labels、release lineage、36 standard rows 全冻结 |
| C1 单指标穷举 | ✅ 完成；❌ 无 exact | 三面板 exact count 均为 0 |
| C2 1–5 指标 sparse linear | ⚠️ RL-only apparent exact | reference 3 指标、standard 2 指标 |
| C3 LOOCV/object robustness | ❌ | object shift 下 failure recall/误拒不可接受 |
| C4 proxy negatives 单列 | ✅ | 42 proxy 与 RL failure 分开记录 |
| C5 hard-gate decision | ❌ | frozen proxy reject 4/42；联合 sparse 仍错 6 |

## 🔬 关键公式

Failure 判定为 `score >= 0`。

```text
score_ref =
  -87.50178614
  + 0.004147183827 * joint_jerk_l2_p95
  + 36.13989131  * object_lin_speed_p95
  - 0.005249266208 * object_lin_jerk_p95

score_std =
  -6.153407590
  - 2784.555096 * track_root_quat_err_terminal
  + 15.94531470 * trackbody_speed_max
```

两式只记录训练内 apparent separator，不写入生产 gate。

## 🧪 稳健性

| 面板/模型 | 验证 | Balanced acc. | Failure recall | Success recall |
|---|---|---:|---:|---:|
| Reference sparse | LOOCV | 0.771 | 4/6 | 28/32 |
| Reference sparse | Object | 0.760 | 5/6 | 22/32 |
| Standard sparse | LOOCV | 0.867 | 5/6 | 27/30 |
| Standard sparse | Object | 0.583 | 1/6 | 30/30 |
| Joint sparse | 5-fold | 0.775 | 36/48 | 24/30 |
| Joint sparse | Object | 0.708 | 36/48 | 20/30 |
| Joint all-111 | Object | 0.663 | 22/48 | 26/30 |

100 次 bootstrap 中，reference 最终三特征选择频率为 `40%/4%/3%`；
standardized 两特征为 `44%/15%`。公式不稳定。

200 次 permutation 下，L1 5-fold balanced accuracy：

- reference：actual `0.745`，permutation mean `0.503`，`p=0.0398`；
- standardized：actual `0.733`，permutation mean `0.492`，`p=0.0448`。

这支持 soft signal 高于随机，不支持零容错 gate。

## 📦 结果路径

| 类型 | 路径 |
|---|---|
| Plan | `workspace/core4d/plan/198_E180_rl38_metric_linear_separability_audit_plan.md` |
| Runner | `workspace/core4d/scripts/eval/runners/eval_E180_rl_metric_separability.py` |
| Tests | `workspace/core4d/scripts/eval/runners/test_eval_E180_rl_metric_separability.py` |
| Artifacts | `workspace/core4d/results/E180/rl_metric_separability/` |
| Report | `workspace/core4d/analysis/RL38_metric_linear_separability_audit_20260726.md` |

## ✅ 验证

- `.venv/bin/python -m py_compile`：通过。
- 4 个直接导入执行的 `test_*`：通过。
- pytest：环境未安装，已显式记录，未冒充 pytest pass。
- 正式评估：`bootstrap=100`、`permutations=200`，完成。

## 🛠️ 下一步

1. 不部署 E180 hard gate。
2. 冻结 3 指标 reference audit score，只用于新 case 风险排序。
3. 补 10–20 条新 date/object RL label 做真正 held-out。
4. 将 proxy 按 failure taxonomy 拆分，优先做 phase-stratified closed-loop
   probe。
