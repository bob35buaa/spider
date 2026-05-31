# E103 Phase 3 — selected target regeneration results

日期：2026-05-31
计划：`workspace/core4d/plan/110_E103_core4d_scene_rebuild_and_inertial_audit_plan.md`
上游：log 127 Phase0 audit；log 128 source template rebuild；log 129 rebuilt preflight/re-mine

## 目标

只对后续仍可能使用的 selected target 做 clean-template regeneration/verification，不批量覆盖历史 derived scenes，不继承旧 polluted dynamics label。

## 产物

| 类型 | 路径 |
|---|---|
| Phase3 target status | `workspace/core4d/results/E103/rebuilt_target_regeneration_status.tsv` |
| Phase3 target status summary | `workspace/core4d/results/E103/rebuilt_target_regeneration_status.md` |
| Verify summaries | `workspace/core4d/results/E103/verify/` |
| Replay visuals/videos | `workspace/core4d/results/E103/visuals/rebuilt_target_replay/` |
| Pre-regeneration snapshot | `workspace/core4d/results/E103/pre_regen_target_snapshot/` |
| Full audit registry after target regen | `workspace/core4d/results/E103/affected_scene_registry.tsv` |
| Validity reset after target regen | `workspace/core4d/results/E103/result_validity_reset.md` |

新增脚本：

- `workspace/core4d/scripts/E103/render_rebuilt_target_replay.py`
- `workspace/core4d/scripts/E103/build_target_regeneration_status.py`

## 执行内容

### Regenerated targets

两条 Box026 selected target 有 clean source template 与 existing trimmed qpos，因此重新生成 target `scene.xml` / `trajectory_kinematic.npz` / `scene_act.xml` 并校验：

| target | source scene | verify | qpos shape | scene | scene_act |
|---|---|---|---|---|---|
| `e091_box026_20231018_039_p2` | `box026_person2/scene.xml` | PASS | `[123, 43]` | `nq=43,nv=41,nu=29` | `nq=42,nv=41,nu=35` |
| `e091_box026_20231020_135_p2` | `box026_person2/scene.xml` | PASS | `[82, 43]` | `nq=43,nv=41,nu=29` | `nq=42,nv=41,nu=35` |

两条 verify summary 均为：

- `trimmed_qpos_matches_spider_qpos=True`
- target scene load OK
- target `scene_act.xml` load OK

### Skipped by data gate

| target | status | reason |
|---|---|---|
| `box021_person1` old guard equivalent | source-template-only | canonical source template 已重建 clean；旧 `scene_act/0/trajectory` runtime artifact 已移出 current source template 目录，不复用旧 dynamics label/trajectory 作为 target evidence |
| `e091_box022_20231023_125_p1` | skipped | source scene exists and clean，但 raw fingertip close-contact `L/R=0/0` |
| `e091_box022_20231023_125_p2` | skipped | source scene exists and clean，但 raw fingertip close-contact `L/R=0/0` |
| `e091_box022_20231023_126_p1` | skipped | source scene exists and clean，但 raw fingertip close-contact `L/R=0/0` |
| `e091_box022_20231023_126_p2` | skipped | source scene exists and clean，但 raw fingertip close-contact `L/R=0/0` |

## Audit after regeneration

Post-regeneration full registry:

| metric | value |
|---|---:|
| registry rows | 201 |
| `keep_clean` | 21 |
| `quarantine_invalidated_by_scene_inertial_bug` | 82 |
| `review_before_use` | 98 |
| inertial clean rows | 108 |

Key rows:

| task | action | inertial | geometry | robot unique | pelvis mass | object mass |
|---|---|---|---|---:|---:|---:|
| `box021_person1` | `keep_clean` | `clean` | `clean` | 18 | 3.813 | 5.000 |
| `box022_person1` | `keep_clean` | `clean` | `clean` | 18 | 3.813 | 5.000 |
| `box022_person2` | `keep_clean` | `clean` | `clean` | 18 | 3.813 | 5.000 |
| `box026_person2` | `keep_clean` | `clean` | `clean` | 18 | 3.813 | 5.000 |
| `e091_box026_20231018_039_p2` | `keep_clean` | `clean` | `clean` | 18 | 3.813 | 5.000 |
| `e091_box026_20231020_135_p2` | `keep_clean` | `clean` | `clean` | 18 | 3.813 | 5.000 |

相邻历史派生目录如 `_e092_dyn/_e092_omni` 仍保持 quarantine；本次没有批量覆盖它们。

## 可视化

`workspace/core4d/results/E103/visuals/rebuilt_target_replay/` 包含：

- 2 张 replay sheet PNG。
- 8 张 MuJoCo keyframe PNG。
- 2 个 kinematic replay MP4：
  - `e091_box026_20231018_039_p2_kinematic_replay.mp4`：123 frames。
  - `e091_box026_20231020_135_p2_kinematic_replay.mp4`：82 frames。

实际观察：

- 两条 replay sheet/keyframe 都能看到 clean rebuilt scene 中的机器人与 Box026 物体姿态序列。
- 视频是 kinematic/data validation replay，不是 CEM/dynamics WORK 证据。
- 视频用于确认 target scene、trajectory、scene_act、object pose patch 和 MuJoCo 渲染链路可读、非空、非静态。

High subagent 只读审查：PASS。

- 两条 target 在 registry 中均为 `keep_clean / clean / clean`。
- Verify summary 均 `trimmed_qpos_matches_spider_qpos=True`，scene/scene_act dims 正确。
- PNG/MP4 存在、非零、可读；MP4 可解码且采样帧有明显变化。
- Phase4 mining 表/summary/REVIEW 同步复审 PASS：0 executable、80 rejected、`reject_source_scene_missing=0`、`reject_verified_legacy=0`、`legacy_label_invalidated_by_e103=True` 正好 2。
- 风险：旧 manifest 列 `source_scene_exists` 仍保留历史 False，后续消费脚本必须使用 `source_scene_exists_live` 作为最终 source gate。

## Claims 验证

| claim | 状态 | 说明 |
|---|---|---|
| C1 污染范围可复现、可审计 | PASS | target regen 后 registry/audit 已更新；clean 108、quarantine 82 |
| C2 canonical source templates 重建 | PASS | log 128 已覆盖；Phase3 使用 clean `box026_person2` |
| C3 source template 与 target case 责任分离 | PASS for selected batch | Box026 target 从 trimmed qpos patch；Box022 因 raw-contact 0/0 不生成 target；box021 old guard 不复用旧 runtime/dynamics label |
| C4 E091/E102 mining 口径修正 | PASS | log 129 已修正：source missing 与 polluted legacy label 都不再 hard reject |
| C5 历史结果重新标注有效性 | PASS | validity reset 更新为 82 个 polluted derived quarantine；旧 dynamics label 不作为硬正/负例 |

## 决策

- E103 不启动 CEM：Phase4 `rebuilt_v2_candidates_with_fingertip.tsv` 仍为 0 executable。
- Box022 不进入 target regeneration/CEM：4 条 selected row raw fingertip close-contact 仍为 `0/0`。
- 两条 rebuilt Box026 target 只作为 clean data/regeneration evidence；仍因 Box026 large-reach holdout 进入后续 H2/reach-support review，不在 E103 内做 WORK 结论。
