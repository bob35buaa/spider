# E102 Results: failure taxonomy + Box022/data expansion + RL-ready handoff

日期：2026-05-31
对应 plan：`workspace/core4d/plan/109_E102_data_expansion_and_rl_ready_plan.md`
上游：E098 replay gate / E099 fingertip audit / E100 fingertip target / E101 stop-loss

## TL;DR

E102 已按计划执行完，结论是 **data expansion 没有产出新的 executable CEM candidate**，因此 Phase 3 full-CEM 按 stop-loss 跳过；handoff 只包含已有 3 条 box004 WORK，状态 `PARTIAL`。

- Phase 0：E101 failure taxonomy 完成，4 个 `current_negative` + 2 个 `positive_guard`，6 条 rollout 均有 3 张 overlay。
- Phase 1：Box022 当前 v2 selected 4 条 raw 可读，但 raw fingertip close-contact frames 全为 0，全部 `REJECT`；且仍缺 `box022_person*/scene.xml`。
- Phase 2：medium-box inventory 80 行 re-mine 后，`v2_candidates_with_fingertip.tsv` 为 0 条 executable candidate。
- Phase 3：候选数 0 < 2，未启动本地/远程 full-CEM。
- Phase 4：`rl_ready_set.tsv` 3 条 READY（box004 083_p1、082_p1、083_p2），`holosoma_handoff.md` 标为 `PARTIAL`。

## 1. 产物

| artifact | path |
|---|---|
| E101 taxonomy | `workspace/core4d/results/E102/e101_failure_taxonomy.tsv` |
| negative registry | `workspace/core4d/results/E102/negative_case_registry.tsv` |
| E101 visual review | `workspace/core4d/results/E102/visuals/e101_failure_review/REVIEW.md` |
| Box022 inventory | `workspace/core4d/results/E102/box022_inventory.tsv` |
| Box022 preflight | `workspace/core4d/results/E102/box022_preflight.tsv` |
| Box022 review | `workspace/core4d/results/E102/visuals/box022_preflight/REVIEW.md` |
| mined candidates | `workspace/core4d/results/E102/v2_candidates_with_fingertip.tsv` |
| rejected/held candidates | `workspace/core4d/results/E102/v2_candidates_rejected.tsv` |
| candidate audit | `workspace/core4d/results/E102/visuals/candidate_audit/REVIEW.md` |
| RL-ready set | `workspace/core4d/results/E102/rl_ready_set.tsv` |
| do-not-retry list | `workspace/core4d/results/E102/dont_try_this_list.tsv` |
| Holosoma handoff | `workspace/core4d/results/E102/holosoma_handoff.md` |

## 2. Phase 0 — E101 taxonomy

| group | count | meaning |
|---|---:|---|
| `current_negative` | 4 | E098-E100 修复后 E101 仍失败，可硬排除 |
| `positive_guard` | 2 | box004 guard WORK，不进负例 |
| `legacy_failure_prior` | 8 | 历史失败只作机制先验，未升级 |

分类：
- `box021_030_p1` seed0/1：`tilted_no_transport + object_miss + motion_level_H2_binding`
- `box021_11035/035_p2`：`pelvis_collapse_residual + lie_on_box`
- `box021_18029_p2`：`reward_hacking_residual + tilted_no_transport`；视觉子型为 upperbody lean
- `box004_083_p2` seed0/1：`positive_guard`

视觉 sidecar 独立审查确认上述标签；`18029_p2` 的 reward hacking 解释来自 gate/上下文，视觉本身只证明 upperbody lean/tilt。

## 3. Phase 1 — Box022 preflight

初始检查后修正 raw root：旧 `/mnt/ali-sh-1/...` 当前不可用，实际可访问 root 为：

`/mnt/a0ccc676-9496-49f8-a861-f8a1797dec52/mocap_data/CORE4D/CORE4D_Real/human_object_motions`

E102 scoped Box022 从 6 条改为当前 v2 `review_stress_test_selected` 的 4 条：

| result | count |
|---|---:|
| raw readable | 4/4 |
| source scene template exists | 0/4 |
| fingertip close-contact frames | 0/4 cases |
| `PREFLIGHT_PASS` | 0 |
| `REJECT` | 4 |

结论：Box022 不进入 CEM。当前问题不只是 template 缺失；raw fingertip audit 也没有支持双手 close-contact。

## 4. Phase 2 — data_construction_v2 re-mine

`medium_box_manifest.tsv` 80 行全部审计。可执行候选为 0。

| route | count |
|---|---:|
| `reject_source_scene_missing` | 34 |
| `reject_raw_contact_not_pass` | 30 |
| `reject_box022_preflight_not_pass` | 8 |
| `existing_positive_not_new` | 3 |
| `reject_preprocess_infeasible` | 2 |
| `reject_verified_legacy` | 2 |
| `reject_box026_large_reach_holdout` | 1 |

解释：
- 已有 box004 WORK 只是 positive set，不是新增候选；
- Box022 raw preflight 没过；
- Box026 仍是 large-reach/posture holdout；
- 其它 rows 主要卡在 source scene 缺失或 raw-contact 不过。

## 5. Phase 3 — full CEM

未启动。原因：Phase 2 executable candidates = 0，低于计划中 Phase 3 的最低条件（至少 2 个 candidate）。

因此本轮没有本地/远程 GPU 任务，也没有需要回收的远程 CEM 结果。

## 6. Phase 4 — handoff

`rl_ready_set.tsv` 有 3 条 READY：

| case | provenance | status |
|---|---|---|
| `e091_box004_20231003_2_083_p1` | E096b | READY |
| `e091_box004_20231003_2_082_p1` | E096b | READY |
| `e091_box004_20231003_2_083_p2` | E101 guard with E100 target | READY |

`holosoma_handoff.md` 状态为 `PARTIAL`：RL-ready set 只有 3 条，未达到计划的 ≥5 条。

## 7. Claims

| Claim | 判定 | 证据 |
|---|---|---|
| C1 E101 failure taxonomy | PASS | `e101_failure_taxonomy.tsv` + 18 overlay jpg + REVIEW |
| C2 Box022 raw-contact preflight | PASS/negative | 4 scoped selected rows完成 preflight；0 pass，4 reject |
| C3 re-mine with E099/E101 constraints | FAIL to expand | 0 executable candidates；拒绝原因已分桶 |
| C4 new candidate full CEM | SKIP | candidate 数 0 < 2，按 stop-loss 不跑 |
| C5 RL-ready handoff | PARTIAL | 3 READY existing positives，未达 ≥5 |

## 8. 下一步

不要在当前 inventory 上直接继续 full-CEM。下一步若要扩数据，需要先做其中一个：

1. 补新的 source scene/template 和 raw-contact proxy 后扩 inventory；
2. 单独开 posture/upright/valid-carry 方向尝试救特定 box021/box026；
3. 放弃单 G1 的这批 medium/large cases，转 dual-G1 或 Holosoma RL 只吃当前 3 条 box004 baseline。
