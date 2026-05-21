# E031 P0 9case 保守汇总

Case 列表（9 个）：box023_p1, box023_p2, box025_p1, box025_p2, bucket001_p2, bucket005_s2_p1, bucket005_s2_p2, bucket007_p1, bucket007_p2

| 方法 | 角色 | N | 缺失 case | 物体位置误差 cm ↓ | 5cm 接触/proxy ↑ | 深穿透 % ↓ | 摔倒数 ↓ | 严格成功 ↑ |
|---|---|---:|---|---:|---:|---:|---:|---:|
| omniretarget_kinematic | 基线 | 9/9 | - | 0.00 | - | - | 0 | 0/9 |
| spider_E081_full_rerun | 基线 | 9/9 | - | 21.27 | 42.17 | 16.25 | 1 | 4/9 |
| spider_E018b | 保守正向候选 | 9/9 | - | 5.14 | 63.61 | 35.83 | 1 | 1/9 |
| spider_best_conservative_E018b_E022_E025 | 保守正向候选 | 9/9 | - | 4.97 | 65.81 | 30.52 | 0 | 1/9 |
| spider_E028 | 诊断/拒绝 | 5/9 | box023_p1, box023_p2, box025_p1, bucket007_p2 | 4.95 | 40.18 | 36.35 | 1 | 0/5 |
| spider_E029 | 诊断/拒绝 | 2/9 | box023_p1, box023_p2, box025_p1, bucket005_s2_p1, bucket005_s2_p2, bucket007_p1, bucket007_p2 | 4.43 | 98.09 | 34.71 | 0 | 1/2 |
| spider_E030 | 诊断/拒绝 | 6/9 | bucket001_p2, bucket005_s2_p2, bucket007_p1 | 5.17 | 32.89 | 17.32 | 2 | 0/6 |

## 数据质量 caveat

| Case | 质量标签 | P0 | 是否从成功分母剔除 | 判定依据 |
|---|---|---:|---:|---|
| box023_p1 | `usable_with_caveat`（可用但需 caveat） | 是 | 否 | 可用但需 caveat: 失败证据类别数=1; 从 P0 剔除=否; 从成功分母剔除=否 |
| box023_p2 | `usable_with_caveat`（可用但需 caveat） | 是 | 否 | 可用但需 caveat: 失败证据类别数=1; 从 P0 剔除=否; 从成功分母剔除=否 |
| box025_p1 | `retarget_questionable`（前置重定向可疑） | 是 | 否 | 前置重定向可疑: 失败证据类别数=1; 从 P0 剔除=否; 从成功分母剔除=否 |
| box025_p2 | `usable_algorithmic_failure`（可用，主要是算法失败） | 是 | 否 | 可用，主要是算法失败: 失败证据类别数=0; 从 P0 剔除=否; 从成功分母剔除=否 |
| bucket001_p2 | `usable_with_caveat`（可用但需 caveat） | 是 | 否 | 可用但需 caveat: 失败证据类别数=1; 从 P0 剔除=否; 从成功分母剔除=否 |
| bucket005_s2_p1 | `usable_algorithmic_failure`（可用，主要是算法失败） | 是 | 否 | 可用，主要是算法失败: 失败证据类别数=0; 从 P0 剔除=否; 从成功分母剔除=否 |
| bucket005_s2_p2 | `usable_algorithmic_failure`（可用，主要是算法失败） | 是 | 否 | 可用，主要是算法失败: 失败证据类别数=0; 从 P0 剔除=否; 从成功分母剔除=否 |
| bucket007_p1 | `usable_algorithmic_failure`（可用，主要是算法失败） | 是 | 否 | 可用，主要是算法失败: 失败证据类别数=0; 从 P0 剔除=否; 从成功分母剔除=否 |
| bucket007_p2 | `retarget_questionable`（前置重定向可疑） | 是 | 否 | 前置重定向可疑: 失败证据类别数=1; 从 P0 剔除=否; 从成功分母剔除=否 |

说明：
- E028-E030 只作为诊断基线展示，不参与 `best-positive` 选择。
- `desk021_p1` 保留在 P1 caveat 表中，但它是 E027 唯一的 success-denominator discard。
- 如果 conservative best 的 strict 数量没有变化，E031 只是账本结果，不是新的优化收益。
