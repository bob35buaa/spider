# E095 结果：从 data_construction_v2 反挖 box004-like work 候选

日期：2026-05-29

对应计划：`workspace/core4d/plan/101_E095_worklike_data_mining_plan.md`

## 1. 目标和结论

E095 的目标不是继续修 Box026，而是把 E092/E094 的失败经验带回 `data_construction_v2`，找更多接近 `box004_083_p2` 这种可行 pattern 的候选。

本轮结论：

- 候选库已重建：`32` 条候选，分为 box004 priority、box021 review、Box022 needs raw-contact、Box026 deprioritized。
- 第一批只启用 `3` 条 box004 priority case，不启用 Box026。
- Stage2b / OmniRetarget / SPIDER preprocess 通过 `2/3`：`083_p1` 和 `082_p1` 可进入下一步 full CEM。
- `082_p2` 在 OmniRetarget CVXPY solve 阶段 infeasible，作为 preprocess reject 保留，不进入 CEM/RL。
- D005b gate 通过 `2/2` passed-only rows，visual QC `6/6` PNG 非空；两条通过 case 都是新 `CEM-candidate`。

## 2. Candidate Mining

新增脚本：

`workspace/core4d/scripts/E095/mine_worklike_candidates.py`

输入来自旧 `data_construction` 的 D001/D002 inventory 与 raw-contact summary，并显式加入最近失败经验：

| 规则 | 处理 |
|---|---|
| box004 / box023-like 尺寸 | 优先；接近 E092/E094 已 WORK 的 small/medium box pattern |
| D002 raw-contact pass | 加分；尤其 both-hand active 和 longest-run |
| source scene template 已存在 | 加分；缺失时只在可安全补齐时启用 |
| Box026 | 降权到 `tier4_box026_deprioritized`；E092/E094 已证明 raw-contact pass 不能预测 dynamics work |
| box021 | 保留为 `tier2_box021_review_after_target_gate`；历史 D003/Box021 多次 CEM 失败，不能直接混入第一批 |
| Box022 | `tier3_box022_needs_raw_contact`；长边更大且缺 D002 raw-contact evidence |

候选库输出：

| 内容 | 路径 |
|---|---|
| repo candidate bank | `workspace/core4d/results/E095/worklike_candidate_mining/worklike_candidate_bank.tsv` |
| repo summary | `workspace/core4d/results/E095/worklike_candidate_mining/summary.md` |
| v2 copy | `/home/ubuntu/Workspace/holosoma/workspace/v3/data_construction_v2/results/e095_worklike_candidates/` |
| first-batch case file | `/home/ubuntu/Workspace/holosoma/workspace/v3/data_construction_v2/inputs/cases_e095_box004_priority_pipeline.tsv` |

Tier summary：

| tier | count |
|---|---:|
| `tier0_known_work` | 1 |
| `tier1_box004_priority` | 3 |
| `tier2_box021_review_after_target_gate` | 15 |
| `tier3_box022_needs_raw_contact` | 6 |
| `tier4_box026_deprioritized` | 7 |

第一批：

| rank | target | raw score | source scene | status |
|---:|---|---:|---|---|
| 1 | `e091_box004_20231003_2_083_p2` | 100.0 | `box004_person2` | known WORK control |
| 2 | `e091_box004_20231003_2_083_p1` | 100.0 | `box004_person1` | new priority |
| 3 | `e091_box004_20231003_2_082_p1` | 91.932 | `box004_person1` | new priority |
| 4 | `e091_box004_20231003_2_082_p2` | 80.682 | `box004_person2` | new priority, later preprocess reject |

## 3. Source Scene 和快照

本轮补齐 `box004_person1` source scene template：

| 内容 | 路径 |
|---|---|
| source scene | `example_datasets/processed/core4d/unitree_g1/humanoid_object/box004_person1/scene.xml` |
| task info | `example_datasets/processed/core4d/unitree_g1/humanoid_object/box004_person1/task_info.json` |

MuJoCo load 校验：`scene.xml nq=43 nv=41 nu=29`。

Scene snapshot 已写入：

`workspace/core4d/results/E095/scene_snapshot/`

快照包含：

- `box004_person1`
- `e091_box004_20231003_2_083_p1`
- `e091_box004_20231003_2_082_p1`

## 4. Stage2b / OmniRetarget 结果

运行设置：

```bash
CASE_FILE_REL=../holosoma/workspace/v3/data_construction_v2/inputs/cases_e095_box004_priority_pipeline.tsv \
RESULT_ROOT_REL=../holosoma/workspace/v3/data_construction_v2/results/stage2b_medium/results \
REPLACE_WRIST_WITH_FINGERTIP=0 \
bash workspace/core4d/scripts/E091/run_stage2b_medium_boxes.sh
```

日志：

`/home/ubuntu/Workspace/holosoma/workspace/v3/data_construction_v2/logs/stage2b_medium_20260529_154526.log`

结果：

| task | OmniRetarget | trimmed frames | SPIDER verify | decision |
|---|---|---:|---|---|
| `e091_box004_20231003_2_083_p1` | PASS | 102 | `trimmed_qpos_matches_spider_qpos=true`, scene `43/41/29`, scene_act `42/41/35` | keep |
| `e091_box004_20231003_2_082_p1` | PASS | 109 | `trimmed_qpos_matches_spider_qpos=true`, scene `43/41/29`, scene_act `42/41/35` | keep |
| `e091_box004_20231003_2_082_p2` | FAIL | 0 | missing retargeted/trimmed output | reject |

`082_p2` 失败原因：

```text
RuntimeError: CVXPY solve failed: infeasible
```

发生在 OmniRetarget `robot_retarget.py` 的 solve 阶段，约 frame `81/139`；转换后的 CORE4D NPZ 存在，但没有 retargeted / trimmed / SPIDER trajectory。因此这不是 CEM 失败，而是上游 retarget infeasible。

Verify summary：

| task | verify JSON |
|---|---|
| `083_p1` | `/home/ubuntu/Workspace/holosoma/workspace/v3/data_construction_v2/results/stage2b_medium/results/e091_box004_20231003_2_083_p1_verify_summary.json` |
| `082_p1` | `/home/ubuntu/Workspace/holosoma/workspace/v3/data_construction_v2/results/stage2b_medium/results/e091_box004_20231003_2_082_p1_verify_summary.json` |

## 5. 可视化

OmniRetarget 可视化：

| 内容 | 路径 |
|---|---|
| summary | `/home/ubuntu/Workspace/holosoma/workspace/v3/data_construction_v2/results/omniretarget_visuals/summary.md` |
| videos | `/home/ubuntu/Workspace/holosoma/workspace/v3/data_construction_v2/visualizations/omniretarget/` |

汇总：`7` cases scanned，`5` OK visualizations，`2` missing retargeted NPZ，`15/15` PNG 非空，`5/5` MP4 存在。新通过的两条 box004 case：

| task | retargeted frames | trimmed frames | video |
|---|---:|---:|---|
| `e091_box004_20231003_2_083_p1` | 121 | 102 | `.../visualizations/omniretarget/e091_box004_20231003_2_083_p1_retargeted.mp4` |
| `e091_box004_20231003_2_082_p1` | 139 | 109 | `.../visualizations/omniretarget/e091_box004_20231003_2_082_p1_retargeted.mp4` |

Raw-contact 可视化：

| 内容 | 路径 |
|---|---|
| summary | `/home/ubuntu/Workspace/holosoma/workspace/v3/data_construction_v2/results/raw_contact_visuals/summary.md` |
| visualizations | `/home/ubuntu/Workspace/holosoma/workspace/v3/data_construction_v2/visualizations/` |

汇总：`11` raw-contact candidate rows，`11/11` PNG 非空。

D005b visual QC：

| 内容 | 路径 |
|---|---|
| original gate JSON | `workspace/core4d/results/E095/worklike_candidate_mining/d005b_box004_priority.json` |
| passed-only gate JSON | `workspace/core4d/results/E095/worklike_candidate_mining/d005b_box004_passed_only.json` |
| visual QC root | `/home/ubuntu/Workspace/holosoma/workspace/v3/data_construction_v2/e095_worklike_d005b/` |
| visual QC summary | `/home/ubuntu/Workspace/holosoma/workspace/v3/data_construction_v2/e095_worklike_d005b/results/visual_qc/summary.md` |
| D005b TSV | `/home/ubuntu/Workspace/holosoma/workspace/v3/data_construction_v2/e095_worklike_d005b/results/d005b_g1_feasibility/d005b_summary.tsv` |

D005b visual QC：`2` rows，`2` pass，`6/6` PNG 非空。

## 6. D005b Gate 结果

| task | T | L/R inside | signed dist L/R | support either | pelvis min | decision |
|---|---:|---:|---:|---:|---:|---|
| `e091_box004_20231003_2_083_p1` | 102 | `0.0/0.0%` | `0.170/0.165m` | `61.8%` | `0.664m` | PASS |
| `e091_box004_20231003_2_082_p1` | 109 | `0.0/0.0%` | `0.195/0.171m` | `47.7%` | `0.651m` | PASS |
| `e091_box004_20231003_2_082_p2` | - | - | - | - | - | reject: missing scene/npz after OmniRetarget infeasible |

## 7. 为什么 box004 pattern 更像 work

E092/E094 的正反例给出的是组合条件，不是单一 raw-contact 分数：

- box004 尺寸接近 box023：box004 约 `0.348 x 0.264 x 0.447m`，box023 约 `0.306 x 0.314 x 0.353m`。
- Box026 明显更大：约 `0.629 x 0.394 x 0.469m`，体积约 box004 `2.8x`。
- box004 的 D005b support/inside 更干净：本轮两个新 case 都 inside `0%`，support either `47.7-61.8%`，pelvis min `0.651-0.664m`。
- Box026 即使 raw-contact 和 object tracking 很强，E092/E094 full CEM 仍会走向低髋/趴箱/倒地局部解；这说明它的问题已经不是单纯 contact target，而是 reach + posture feasibility。

因此，`data_construction_v2` 能把 Box026 挑出来，是因为 Stage0/Stage1 只在做几何/接触候选筛选；E095 新规则把下游 dynamics 失败经验补回去，用 `tier4_box026_deprioritized` 阻止它继续占第一批资源。

## 8. Claims 验证

| Claim | 结果 | 证据 |
|---|---|---|
| C1: `data_construction_v2` 里还有未跑的 box004-like 候选 | PASS | `083_p1`、`082_p1`、`082_p2` 三条 box004 priority 被挖出 |
| C2: Box026 失败经验能转成筛选规则 | PASS | Box026 保留 `7` 条 traceability rows，但全部降到 `tier4_box026_deprioritized` |
| C3: 至少生成一批可直接进入 Stage2b 的 box004 priority case | PASS | case file 写入 v2 inputs；Stage2b 通过 `2/3` |
| C4: 备选候选需要分层而不是混跑 | PASS | box021 / Box022 / Box026 分层输出，未混入第一批 |

## 9. 决策和下一步

本轮新增可进入 full CEM 的候选：

1. `e091_box004_20231003_2_083_p1`
2. `e091_box004_20231003_2_082_p1`

保留 control：

- `e091_box004_20231003_2_083_p2`：E092/E094 known WORK。

不进入下一步：

- `e091_box004_20231003_2_082_p2`：OmniRetarget CVXPY infeasible。
- Box026 tier：已由 E092/E094 full CEM 证明不应继续盲跑。
- box021 tier：先做 target/posture gate 或专门路线，不和 box004 worklike mining 混跑。
- Box022 tier：先补 raw-contact/reach review。

建议下一轮 E096：

1. 对 `083_p1` 和 `082_p1` 跑 SPIDER full CEM，`083_p2` 作为 known-WORK control 同跑或复用 E092/E094 control。
2. 只有 full CEM 达到 `WORK` 的序列，才进入 Holosoma RL。
3. Holosoma RL 的输入分两路：`RL-from-SPIDER-WORK` 与 `RL-from-OmniRetarget`，但不再把 preprocess reject 或 Box026 fail case 混进去。
