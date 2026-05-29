# E097 Plan: feature-based data_construction_v2 candidate refresh

日期：2026-05-29

## Context

E095/E096/E096b 说明 box004 能 work 不是因为 object key 本身，而是因为：

- raw hand-object contact 强；
- OmniRetarget / SPIDER preprocess 可解；
- G1 target 没有明显 inside；
- support/contact target 语义风险较低；
- full CEM posture gate 没有低髋、趴箱、头/上身/手撑地。

用户要求把这个理解反映回 `data_construction_v2`，继续寻找更多可能 work 的 case，并排除已经验证过的 case。

## Claims

| Claim | 验证方式 |
|---|---|
| C1: 新候选排序不按 object key 加权/降权 | scoring function 只使用 raw contact、reach proxy、inside/support status、geometry risk、known outcome exclusion |
| C2: 已验证 case 不进入新执行队列 | 输出 `excluded_verified_cases.tsv`，pipeline TSV 默认不包含这些 target |
| C3: source scene/template readiness 不加分 | 仅作为 route metadata；缺 scene 不扣质量分，只影响 readiness/action note |
| C4: 能输出下一批可执行候选 | 生成 `cases_e097_feature_candidate_pipeline.tsv`，按 feature route 排序 |
| C5: 结果可复现 | 脚本、summary、candidate bank、excluded list 写入 repo 和 Holosoma v2 results |

## Feature Route

按以下顺序解释候选：

1. raw contact: 直接来自 D002 `target_both_active_frac_3cm`、balanced L/R、longest run、raw score。
2. reach proxy: 物体最大边长、体积比、aspect，以及已知 OmniRetarget preprocess outcome。
3. inside/support: 若已有 D005b / E096 contact-semantics 证据，写入 outcome；未跑则标 `pending_gate`，不伪造通过。
4. CEM posture gate: 若已有 full CEM outcome，直接 exclude；未跑则标 `pending_full_cem`。

## Already Verified / Excluded

本轮默认排除：

- box004 positive set: `083_p2`, `083_p1`, `082_p1`
- box004 preprocess reject: `082_p2`
- Box026 verified reject/fail: `039_p2`, `040_p2`, `135_p2`
- D003/Box021 已进入过 SPIDER/CEM 或 topface/smoke 的代表 case: `029_p2`, `035_p2`, `019_p1`, `031_p2`, `020_p1`, `034_p1`, `034_p2`

排除不等于永久否定；它只表示“这轮找新数据不要重复拿它们当新发现”。

## Outputs

| 内容 | 路径 |
|---|---|
| script | `workspace/core4d/scripts/E097/mine_feature_based_candidates.py` |
| SPIDER result root | `workspace/core4d/results/E097/feature_candidate_mining/` |
| Holosoma v2 result root | `/home/ubuntu/Workspace/holosoma/workspace/v3/data_construction_v2/results/e097_feature_candidates/` |
| next pipeline TSV | `/home/ubuntu/Workspace/holosoma/workspace/v3/data_construction_v2/inputs/cases_e097_feature_candidate_pipeline.tsv` |

## Success Criteria

- `feature_candidate_bank.tsv/json` generated.
- `excluded_verified_cases.tsv` generated.
- `cases_e097_feature_candidate_pipeline.tsv` contains only unverified candidates.
- Summary lists top candidates and why each is chosen or held.
- Static checks pass.
