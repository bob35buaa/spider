# E103 Phase 1/2 — canonical source template rebuild results

日期：2026-05-31
计划：`workspace/core4d/plan/110_E103_core4d_scene_rebuild_and_inertial_audit_plan.md`
上游：log 127 Phase 0 audit

## 目标

在不批量覆盖历史派生 scene 的前提下，重建 Box021/Box022/Box026 的 canonical source templates，消除 `box021_person1` 污染 base 造成的 robot inertial 错误，并为后续 target regeneration 提供干净 source scene。

## 执行内容

新增 generator：

- `workspace/core4d/scripts/E103/rebuild_core4d_box_source_templates.py`

重建前已保存 snapshot：

- `workspace/core4d/results/E103/pre_rebuild_scene_snapshot/box021_person1/`
- `workspace/core4d/results/E103/pre_rebuild_scene_snapshot/box021_person2/`
- `workspace/core4d/results/E103/pre_rebuild_scene_snapshot/box022_person1/`
- `workspace/core4d/results/E103/pre_rebuild_scene_snapshot/box022_person2/`
- `workspace/core4d/results/E103/pre_rebuild_scene_snapshot/box026_person1/`
- `workspace/core4d/results/E103/pre_rebuild_scene_snapshot/box026_person2/`

Phase 0 pre-rebuild audit 表已冻结：

- `workspace/core4d/results/E103/scene_inertial_audit_phase0_pre_rebuild.tsv`
- `workspace/core4d/results/E103/scene_geometry_audit_phase0_pre_rebuild.tsv`
- `workspace/core4d/results/E103/affected_scene_registry_phase0_pre_rebuild.tsv`
- `workspace/core4d/results/E103/summary_phase0_pre_rebuild.md`
- `workspace/core4d/results/E103/result_validity_reset_phase0_pre_rebuild.md`

重建策略：

- base scene 固定为 clean `box023_person1/scene.xml`。
- 只复制 robot/world/contact skeleton，不使用旧 `box021_person1` 作为 base。
- object mesh 替换为对应 CORE4D asset：
  - `box021_m.obj`
  - `box022_m.obj`
  - `box026_m.obj`
- `box022_m.obj` 从 raw CORE4D object model 复制进 repo asset。
- object collision half-extents = mesh AABB half-extents。
- object mass 统一为 `5.0kg`，写入 `task_info.json` 为 E103 建模假设。
- object inertia 使用 box inertia 公式。
- source template 的 object pose 使用 `box023_person1` neutral placeholder；后续 target scene 必须由 trimmed qpos 第一帧 patch，不能把 source template pose 当 target 数据。
- canonical source template 目录只保留 `scene.xml` 与 `task_info.json`；旧 `scene_act.xml` / `scene_act_meta.json` / `0/trajectory_kinematic.npz` 属于旧 runtime artifact，已移出当前 source template 使用路径。

## 重建结果

| task | action | inertial | geometry | robot unique pairs | pelvis mass | object mass | collision max rel err |
|---|---|---|---|---:|---:|---:|---:|
| `box021_person1` | keep_clean | clean | clean | 18 | 3.813 | 5.000 | <=0.000003 |
| `box021_person2` | keep_clean | clean | clean | 18 | 3.813 | 5.000 | <=0.000003 |
| `box022_person1` | keep_clean | clean | clean | 18 | 3.813 | 5.000 | <=0.000003 |
| `box022_person2` | keep_clean | clean | clean | 18 | 3.813 | 5.000 | <=0.000003 |
| `box026_person1` | keep_clean | clean | clean | 18 | 3.813 | 5.000 | <=0.000003 |
| `box026_person2` | keep_clean | clean | clean | 18 | 3.813 | 5.000 | <=0.000003 |

Immediate MuJoCo load validation:

- 6/6 rebuilt scene load OK.
- 6/6 `nq=43,nv=41,nu=29`。
- 6/6 hand contact site ids remain `11,15`。
- 6/6 robot inertials exactly match clean `box023_person1` robot inertial skeleton。

Stale runtime artifact cleanup:

- `box021_person2/scene_act.xml` 审计发现仍带污染 robot inertial（30 处 `mass=29.632`），已从 current source template 目录删除。
- `box021_person1/scene_act.xml` 虽未污染 robot inertial，但仍是旧 actuator artifact，object inertial/pose 不代表 E103 clean source policy，也已删除。
- `box021_person1/0/`、`box021_person2/0/` 已移至 `workspace/core4d/results/E103/removed_stale_source_runtime_artifacts/`；pre-rebuild snapshot 中也保留原始副本。
- 后续 target case 生成 `trajectory_kinematic.npz` 后，必须重新生成 `scene_act.xml`。

## Post-rebuild full audit

当前无后缀 audit 表是 post-rebuild 状态：

- `workspace/core4d/results/E103/scene_inertial_audit.tsv`
- `workspace/core4d/results/E103/scene_geometry_audit.tsv`
- `workspace/core4d/results/E103/affected_scene_registry.tsv`
- `workspace/core4d/results/E103/summary.md`
- `workspace/core4d/results/E103/result_validity_reset.md`

关键变化：

| metric | pre-rebuild | post-rebuild |
|---|---:|---:|
| existing scene audit rows | 198 | 201 |
| inertial clean | 100 | 106 |
| polluted robot inertial | 87 | 84 |
| canonical templates requiring rebuild | 6 | 0 |
| `keep_clean` rows | 13 | 19 |
| `quarantine_invalidated_by_scene_inertial_bug` rows | 84 | 84 |

解释：3 个缺失 canonical source templates 已创建，3 个污染 canonical templates 已重建；84 个历史派生污染 scene 未批量覆盖，继续 quarantine。

## 可视化审查

Post-rebuild MuJoCo 3D evidence：

- `workspace/core4d/results/E103/visuals/source_templates_post_rebuild/REVIEW.md`

渲染对象：

- 6 个 rebuilt canonical templates：`box021_person1/2`、`box022_person1/2`、`box026_person1/2`
- 2 个 guard：`box023_person1`、`box004_person1`

High subagent 只读审查结论：PASS。

- 8 个 audit sheet 和 8 个 turntable MP4 全部存在、非零。
- PNG 均为 `2100x1350`，sheet 中机器人/物体可见，metric panel 可读。
- MP4 均为 `960x720`、`72 frames`、`24fps`、`3.0s`，可完整解码且有视角变化。
- `REVIEW.md` 与 post-rebuild registry/audit 表一致。
- 6 个 rebuilt canonical templates 均为 `keep_clean / clean / clean`；未检出 `29.632`、污染惯量、`object_mass_policy_review` 或 `geometry_review` 残留。

## Claims 验证

| claim | 状态 | 说明 |
|---|---|---|
| C1 污染范围可复现、可审计 | PASS | pre/post audit 表均保留 |
| C2 canonical source templates 重建 | PASS | 6 个模板已重建并通过 MuJoCo/audit/visual check |
| C3 source template 与 target case 责任分离 | PARTIAL PASS | source template 已采用 neutral placeholder，并在 metadata 写明 target 必须从 trimmed qpos patch；target regeneration 未开始 |
| C4 E091/E102 mining 口径修正 | PENDING | 后续需修改 mining script，把 source scene missing 变为 backlog |
| C5 历史结果重新标注有效性 | PASS for source-template stage | current validity reset 保留 84 个历史派生 quarantine；旧 polluted dynamics label 仍不可硬用 |

## 结论

E103 Phase 1/2 达到目标：Box021/Box022/Box026 canonical source templates 已从干净 base 重建，数据基础不再依赖污染的 `box021_person1`。

下一步不应直接跑 CEM。应进入 Phase 3/4：从 clean source templates 重新生成 selected target cases，然后重跑 raw/contact/quat/target/preflight audits；只有 clean executable candidate 足够时，才单独开 E104/E105 做 dynamics/CEM。
