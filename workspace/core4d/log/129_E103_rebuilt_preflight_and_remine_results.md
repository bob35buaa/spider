# E103 Phase 4 — rebuilt source preflight + v2 re-mine results

日期：2026-05-31
计划：`workspace/core4d/plan/110_E103_core4d_scene_rebuild_and_inertial_audit_plan.md`
上游：log 127 Phase 0 audit；log 128 Phase 1/2 source template rebuild

## 目标

在 canonical source templates 重建后，重新运行 template/preflight/mining 数据门，验证 E102 中 `source_scene_missing` 不再作为候选 hard reject，同时确认 Box022/Box026 是否可以进入后续 target regeneration 或 CEM。

本阶段仍不跑 CEM。

## 执行内容

新增/修改脚本：

- `workspace/core4d/scripts/E103/build_rebuilt_template_preflight.py`
- `workspace/core4d/scripts/E103/render_rebuilt_box022_raw_contact_3d.py`
- `workspace/core4d/scripts/E102/mine_v2_with_fingertip.py`
  - source scene readiness 改为 live filesystem check。
  - 若 source scene 仍缺失，不再输出 `reject_source_scene_missing`，而是 `needs_source_scene_template` / `needs_source_scene_template_then_preflight`。
  - 读取 E103 pre-rebuild invalidated registry；旧污染 scene 的 `verified_legacy` dynamics label 不再作为 hard reject。
  - E103 重跑中所有 source scene live check 均为 True。
- `workspace/core4d/scripts/E102/render_candidate_audit.py`
  - REVIEW 表改显示 `source_scene_live`。

产物：

- `workspace/core4d/results/E103/rebuilt_template_preflight.tsv`
- `workspace/core4d/results/E103/rebuilt_box022_inventory.tsv`
- `workspace/core4d/results/E103/rebuilt_box022_missing_sources.tsv`
- `workspace/core4d/results/E103/rebuilt_box022_preflight.tsv`
- `workspace/core4d/results/E103/rebuilt_v2_candidates_with_fingertip.tsv`
- `workspace/core4d/results/E103/rebuilt_v2_candidates_rejected.tsv`
- `workspace/core4d/results/E103/rebuilt_v2_candidate_mining_summary.md`
- `workspace/core4d/results/E103/visuals/rebuilt_box022_preflight/`
- `workspace/core4d/results/E103/visuals/rebuilt_box022_raw_contact_3d/`
- `workspace/core4d/results/E103/visuals/rebuilt_candidate_audit/`

## 结果

### Source template preflight

| source_scene_task | preflight | inertial | geometry | robot unique pairs | pelvis mass | object mass |
|---|---|---|---|---:|---:|---:|
| `box021_person1` | PASS | clean | clean | 18 | 3.813 | 5.000 |
| `box021_person2` | PASS | clean | clean | 18 | 3.813 | 5.000 |
| `box022_person1` | PASS | clean | clean | 18 | 3.813 | 5.000 |
| `box022_person2` | PASS | clean | clean | 18 | 3.813 | 5.000 |
| `box026_person1` | PASS | clean | clean | 18 | 3.813 | 5.000 |
| `box026_person2` | PASS | clean | clean | 18 | 3.813 | 5.000 |

结论：source template readiness 已修复，不再是 Box022/Box026 阻塞因素。

### Box022 rebuilt preflight

E102 scoped Box022 selected rows：4。

| target_task | raw_ok | source_scene_xml_exists | decision | L/R contact |
|---|---|---|---|---:|
| `e091_box022_20231023_125_p1` | True | True | REJECT | 0/0 |
| `e091_box022_20231023_125_p2` | True | True | REJECT | 0/0 |
| `e091_box022_20231023_126_p1` | True | True | REJECT | 0/0 |
| `e091_box022_20231023_126_p2` | True | True | REJECT | 0/0 |

结论：Box022 的 source scene missing 已解除，但 raw fingertip close-contact evidence 仍为 0，按 E103 decision rule 不能进入 CEM。

### v2 re-mine

| metric | count |
|---|---:|
| source rows | 80 |
| executable candidates | 0 |
| rejected/held rows | 80 |
| `reject_source_scene_missing` | 0 |
| `source_scene_exists_live=True` | 80 |

Route counts：

| route | count |
|---|---:|
| `reject_raw_contact_not_pass` | 61 |
| `reject_box022_preflight_not_pass` | 8 |
| `reject_box026_large_reach_holdout` | 6 |
| `existing_positive_not_new` | 3 |
| `reject_preprocess_infeasible` | 2 |

附加审计：

- `reject_verified_legacy=0`。
- `legacy_label_invalidated_by_e103=True` 为 2 行：`e091_box026_20231018_039_p2` 和 `e091_box026_20231020_135_p2`。

结论：E102 的 source-scene missing bug 已在 E103 重跑中消除；旧 polluted-scene dynamics label 也不再硬继承为 negative。当前没有 executable candidate，原因回到 raw/contact/preprocess/Box026 reach-support holdout，而不是模板缺失或旧污染 label。

## 可视化审查

可视化产物：

- Box022 preflight panel PNG：4 张，`1120x520`。
- Box022 raw-contact 3D 四视图：4 张，`1540x1320`。
- Box022 raw-contact 3D turntable MP4：4 个，`720x640`，`36 frames`，`12fps`，`3.0s`。
- Candidate audit REVIEW：`workspace/core4d/results/E103/visuals/rebuilt_candidate_audit/REVIEW.md`

High subagent 只读审查结论：PASS。

- 所有 PNG/MP4 存在、非零、可读/可解码。
- MP4 抽样帧有视角变化。
- preflight PNG 明确显示左右手 close-contact frames 均为 0。
- 3D 四视图标注 `L/R no contact`，未见靠近盒面的指尖接触证据。
- v2 re-mine 表、summary、REVIEW 互相一致：0 executable，80 rejected/held，无 `reject_source_scene_missing` / `reject_verified_legacy`，`source_scene_exists_live=True` 为 80/80。

## Claims 验证

| claim | 状态 | 说明 |
|---|---|---|
| C1 污染范围可复现、可审计 | PASS | log 127/128 已覆盖 |
| C2 canonical source templates 重建 | PASS | 6/6 template preflight PASS |
| C3 source template 与 target case 责任分离 | PARTIAL | source clean 已完成；target regeneration 尚未执行 |
| C4 E091/E102 mining 口径修正 | PASS for mining/preflight | source missing 不再 hard reject；live check 后 80/80 source scene available |
| C5 历史结果重新标注有效性 | PASS | validity reset 后 `reject_verified_legacy=0`；旧污染 label 不再硬排除 |

## 决策

- Box022 不进入 CEM：source template 已 clean，但 raw fingertip contact 仍 0/0。
- v2 data expansion 不进入 CEM：executable candidates = 0 < 2。
- E103 下一步若继续执行 Phase 3，只能做 selected target regeneration/verification；这些结果不能继承旧 polluted dynamics label。
