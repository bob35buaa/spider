# E031 P1 13case 保守汇总

Case 列表（13 个）：box021_p1, box021_p2, box023_p1, box023_p2, box025_p1, box025_p2, bucket001_p1, bucket001_p2, bucket005_s2_p1, bucket005_s2_p2, bucket007_p1, bucket007_p2, desk021_p1

| 方法 | 角色 | N | 缺失 case | 物体位置误差 cm ↓ | 5cm 接触/proxy ↑ | 深穿透 % ↓ | 摔倒数 ↓ | 严格成功 ↑ |
|---|---|---:|---|---:|---:|---:|---:|---:|
| omniretarget_kinematic | 基线 | 12/13 | desk021_p1 | 0.00 | - | - | 0 | 0/12 |
| spider_E081_full_rerun | 基线 | 13/13 | - | 27.10 | 36.23 | 12.39 | 4 | 4/13 |
| spider_E018b | 保守正向候选 | 13/13 | - | 5.45 | 54.35 | 30.04 | 4 | 1/13 |
| spider_best_conservative_E018b_E022_E025 | 保守正向候选 | 13/13 | - | 5.33 | 55.87 | 26.37 | 3 | 1/13 |
| spider_E028 | 诊断/拒绝 | 5/13 | box021_p1, box021_p2, box023_p1, box023_p2, box025_p1, bucket001_p1, bucket007_p2, desk021_p1 | 4.95 | 40.18 | 36.35 | 1 | 0/5 |
| spider_E029 | 诊断/拒绝 | 3/13 | box021_p1, box021_p2, box023_p1, box023_p2, box025_p1, bucket005_s2_p1, bucket005_s2_p2, bucket007_p1, bucket007_p2, desk021_p1 | 4.06 | 65.39 | 23.14 | 0 | 1/3 |
| spider_E030 | 诊断/拒绝 | 6/13 | box021_p1, box021_p2, bucket001_p1, bucket001_p2, bucket005_s2_p2, bucket007_p1, desk021_p1 | 5.17 | 32.89 | 17.32 | 2 | 0/6 |

## 数据质量 caveat

| Case | 质量标签 | P0 | 是否从成功分母剔除 | 判定依据 |
|---|---|---:|---:|---|
| box021_p1 | `usable_algorithmic_failure`（可用，主要是算法失败） | 否 | 否 | 可用，主要是算法失败: 失败证据类别数=0; 从 P0 剔除=是; 从成功分母剔除=否 |
| box021_p2 | `usable_algorithmic_failure`（可用，主要是算法失败） | 否 | 否 | 可用，主要是算法失败: 失败证据类别数=0; 从 P0 剔除=是; 从成功分母剔除=否 |
| box023_p1 | `usable_with_caveat`（可用但需 caveat） | 是 | 否 | 可用但需 caveat: 失败证据类别数=1; 从 P0 剔除=否; 从成功分母剔除=否 |
| box023_p2 | `usable_with_caveat`（可用但需 caveat） | 是 | 否 | 可用但需 caveat: 失败证据类别数=1; 从 P0 剔除=否; 从成功分母剔除=否 |
| box025_p1 | `retarget_questionable`（前置重定向可疑） | 是 | 否 | 前置重定向可疑: 失败证据类别数=1; 从 P0 剔除=否; 从成功分母剔除=否 |
| box025_p2 | `usable_algorithmic_failure`（可用，主要是算法失败） | 是 | 否 | 可用，主要是算法失败: 失败证据类别数=0; 从 P0 剔除=否; 从成功分母剔除=否 |
| bucket001_p1 | `usable_with_caveat`（可用但需 caveat） | 否 | 否 | 可用但需 caveat: 失败证据类别数=1; 从 P0 剔除=是; 从成功分母剔除=否 |
| bucket001_p2 | `usable_with_caveat`（可用但需 caveat） | 是 | 否 | 可用但需 caveat: 失败证据类别数=1; 从 P0 剔除=否; 从成功分母剔除=否 |
| bucket005_s2_p1 | `usable_algorithmic_failure`（可用，主要是算法失败） | 是 | 否 | 可用，主要是算法失败: 失败证据类别数=0; 从 P0 剔除=否; 从成功分母剔除=否 |
| bucket005_s2_p2 | `usable_algorithmic_failure`（可用，主要是算法失败） | 是 | 否 | 可用，主要是算法失败: 失败证据类别数=0; 从 P0 剔除=否; 从成功分母剔除=否 |
| bucket007_p1 | `usable_algorithmic_failure`（可用，主要是算法失败） | 是 | 否 | 可用，主要是算法失败: 失败证据类别数=0; 从 P0 剔除=否; 从成功分母剔除=否 |
| bucket007_p2 | `retarget_questionable`（前置重定向可疑） | 是 | 否 | 前置重定向可疑: 失败证据类别数=1; 从 P0 剔除=否; 从成功分母剔除=否 |
| desk021_p1 | `discard_from_success_denominator`（从成功分母剔除） | 否 | 是 | 从成功分母剔除: 失败证据类别数=3; 从 P0 剔除=是; 从成功分母剔除=是 |

说明：
- E028-E030 只作为诊断基线展示，不参与 `best-positive` 选择。
- `desk021_p1` 保留在 P1 caveat 表中，但它是 E027 唯一的 success-denominator discard。
- 如果 conservative best 的 strict 数量没有变化，E031 只是账本结果，不是新的优化收益。
