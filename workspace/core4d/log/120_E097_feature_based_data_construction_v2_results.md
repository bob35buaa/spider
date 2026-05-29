# E097 结果：feature-based data_construction_v2 candidate refresh

日期：2026-05-29

对应计划：`workspace/core4d/plan/104_E097_feature_based_data_construction_v2_plan.md`

> 2026-05-29 correction: 后续重新可视化时发现本日志原先列出的 6 条 enabled Box021 rows 并不是干净的未验证新候选；它们已有 legacy D003/D004 visual-QC 或 infeasible 记录。E097 miner 已修正，当前 pipeline enabled rows 为 `0`。详见 `workspace/core4d/log/121_E097_visual_review_and_candidate_correction.md`。

## 1. 目标和结论

用户要求把 E095/E096/E096b 的结论反映回 `data_construction_v2`：后续找数据不能再按 object key 或 source scene readiness 加权，而应按 `raw contact / reach / inside / support / CEM posture gate` 这条 feature route 排队，并排除已经验证过的 case。

本轮新增 E097 miner：

- `workspace/core4d/scripts/E097/mine_feature_based_candidates.py`

输出了新的 feature candidate bank、verified exclusion 表和下一批 pipeline TSV：

- `workspace/core4d/results/E097/feature_candidate_mining/summary.md`
- `/home/ubuntu/Workspace/holosoma/workspace/v3/data_construction_v2/results/e097_feature_candidates/summary.md`
- `/home/ubuntu/Workspace/holosoma/workspace/v3/data_construction_v2/inputs/cases_e097_feature_candidate_pipeline.tsv`

结论：

- 已验证 case 共 `14` 条，全部从新队列排除。
- 未验证且可进入下一关的 candidate 有 `8` 条，默认 enabled 前 `6` 条。
- 当前新候选全部是 `box021`，但标签是 `candidate_target_posture_gate`，不是直接 positive。它们必须先跑 Stage2b/OmniRetarget，再跑 inside/support target gate，过 gate 后才允许 full CEM。
- Box026 即使 raw contact 强，也仍保持 `large_reach_holdout`；Box022 因 raw contact 未跑，只进入 `raw_contact_preflight_disabled`。

## 2. Scoring / Routing 改动

E097 score 只使用：

- D002 raw-contact: raw score、target both-active、L/R balance、longest run、partner-any active。
- geometry/reach proxy: volume ratio、max extent、aspect。
- known outcome exclusion: 已经验证过的 positive/reject/review 不再进入新执行队列。

明确不使用：

- object key bonus / penalty；
- source scene exists bonus；
- template readiness bonus；
- 已有 WBT / 已有 scene 作为质量加分。

source scene 只作为 route metadata：缺 scene 说明工程上还要补模板，不说明数据质量差。

## 3. 输出统计

| route | count | meaning |
|---|---:|---|
| `excluded_verified` | 14 | 已经验证/跑过，本轮不重复 |
| `candidate_target_posture_gate` | 8 | raw contact 强，但必须先做 target/posture gate |
| `raw_contact_preflight_disabled` | 6 | raw contact 未跑，先补 D002/preflight |
| `hold_raw_contact_failed` | 6 | D002 raw contact fail，不进下一轮 |
| `large_reach_holdout` | 4 | Box026-style large reach 风险，需单独 repair route |

## 4. 下一批 enabled candidates

这些是 `cases_e097_feature_candidate_pipeline.tsv` 里默认 enabled 的 6 条：

| target | route | score | raw | both | L/R | longest | note |
|---|---|---:|---:|---:|---|---:|---|
| `e091_box021_20231018_028_p1` | `candidate_target_posture_gate` | 118.990 | 100.000 | 1.000 | 1.000/1.000 | 1.000 | strong raw contact; run preprocess + inside/support gate |
| `e091_box021_20231018_028_p2` | `candidate_target_posture_gate` | 118.990 | 100.000 | 1.000 | 1.000/1.000 | 1.000 | strong raw contact; run preprocess + inside/support gate |
| `e091_box021_20231020_020_p2` | `candidate_target_posture_gate` | 118.990 | 100.000 | 1.000 | 1.000/1.000 | 1.000 | strong raw contact; run preprocess + inside/support gate |
| `e091_box021_20231011_035_p1` | `candidate_target_posture_gate` | 113.846 | 95.933 | 0.987 | 0.987/1.000 | 0.707 | strong raw contact; run preprocess + inside/support gate |
| `e091_box021_20231018_030_p2` | `candidate_target_posture_gate` | 113.636 | 95.417 | 0.972 | 1.000/0.972 | 0.778 | strong raw contact; run preprocess + inside/support gate |
| `e091_box021_20231020_019_p2` | `candidate_target_posture_gate` | 112.426 | 94.079 | 0.947 | 1.000/0.947 | 0.816 | strong raw contact; run preprocess + inside/support gate |

Disabled but still in candidate bank:

- `e091_box021_20231018_030_p1`
- `e091_box021_20231018_029_p1`

这两条同样是 unverified candidate，只是默认先不扩大 batch。

## 5. 已排除的 verified cases

本轮排除的主要 case：

- box004 positives: `083_p2`, `083_p1`, `082_p1`
- box004 preprocess reject: `082_p2`
- Box026 verified reject/fail: `039_p2`, `040_p2`, `135_p2`
- Box021 已跑过的代表 case: `029_p2`, `035_p2`, `019_p1`, `031_p2`, `020_p1`, `034_p1`, `034_p2`

完整表：

- `workspace/core4d/results/E097/feature_candidate_mining/excluded_verified_cases.tsv`

## 6. Holdouts

### Box026

未验证的 Box026 行仍不进入下一批：

- `e091_box026_20231020_134_p1`
- `e091_box026_20231020_134_p2`
- `e091_box026_20231020_135_p1`
- `e091_box026_20231018_039_p1`

原因不是 object-key 降权，而是 feature-level large reach risk：max extent `0.629m`、volume ratio `3.23`，且 E092/E094 已经显示同尺寸/同物体 pattern 会诱导低髋/趴箱/倒地局部解。它们需要单独 repair route，不应该混进“找 worklike 数据”的下一批。

### Box022

Box022 有 `6` 条进入 `raw_contact_preflight_disabled`：

- `125_p1/p2`
- `126_p1/p2`
- `127_p1/p2`

原因：D002 raw contact 未跑，且 long edge `0.667m`，不能在没有 raw-contact/contact-target preflight 的情况下直接 Stage2b/CEM。

## 7. 结果路径

| 内容 | 路径 |
|---|---|
| E097 script | `workspace/core4d/scripts/E097/mine_feature_based_candidates.py` |
| SPIDER result root | `workspace/core4d/results/E097/feature_candidate_mining/` |
| Holosoma v2 result root | `/home/ubuntu/Workspace/holosoma/workspace/v3/data_construction_v2/results/e097_feature_candidates/` |
| Holosoma v2 input TSV | `/home/ubuntu/Workspace/holosoma/workspace/v3/data_construction_v2/inputs/cases_e097_feature_candidate_pipeline.tsv` |
| candidate bank | `workspace/core4d/results/E097/feature_candidate_mining/feature_candidate_bank.tsv` |
| verified exclusion | `workspace/core4d/results/E097/feature_candidate_mining/excluded_verified_cases.tsv` |
| summary | `workspace/core4d/results/E097/feature_candidate_mining/summary.md` |

## 8. Claims 验证

| Claim | 结果 | 证据 |
|---|---|---|
| C1: 新候选排序不按 object key 加权/降权 | PASS | score 只使用 raw contact + reach proxy + known exclusion |
| C2: 已验证 case 不进入新执行队列 | PASS | `excluded_verified=14`，pipeline enabled 无 box004/verified Box026/verified Box021 |
| C3: source scene/template readiness 不加分 | PASS | `source_scene_exists` 仅输出为 metadata |
| C4: 能输出下一批可执行候选 | PASS | 6 条 enabled `candidate_target_posture_gate` |
| C5: 结果可复现 | PASS | 脚本、summary、bank、excluded list 均保存 |

## 9. 下一步

建议下一步不要直接跑 full CEM 或 RL，而是按 feature route 执行：

1. 对 6 条 enabled box021 candidate 跑 Stage2b/OmniRetarget。
2. 对 preprocess pass 的 case 跑 E096-style contact semantics / inside-support gate。
3. 只对 gate pass 的 case 跑 SPIDER full CEM。
4. 只有 full CEM `WORK` 的序列进入 Holosoma RL positive set。

当前最值得先跑的小 batch 是：

```text
e091_box021_20231018_028_p1
e091_box021_20231018_028_p2
e091_box021_20231020_020_p2
e091_box021_20231011_035_p1
e091_box021_20231018_030_p2
e091_box021_20231020_019_p2
```
