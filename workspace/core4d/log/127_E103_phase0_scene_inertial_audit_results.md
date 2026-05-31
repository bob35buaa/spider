# E103 Phase 0 — scene inertial audit + quarantine registry

日期：2026-05-31
计划：`workspace/core4d/plan/110_E103_core4d_scene_rebuild_and_inertial_audit_plan.md`

## 目标

E103 先处理数据层可信度，不继续做 reward/CEM 优化。Phase 0 的目标是把 `humanoid_object` 下现有 scene 的 robot inertial 污染范围、object geometry policy、以及历史结果有效性边界落成可复现表格和可视化证据。

## 执行内容

新增脚本：

- `workspace/core4d/scripts/E103/audit_scene_inertials.py`
- `workspace/core4d/scripts/E103/audit_scene_geometry.py`
- `workspace/core4d/scripts/E103/build_affected_scene_registry.py`
- `workspace/core4d/scripts/E103/build_result_validity_reset.py`
- `workspace/core4d/scripts/E103/render_scene_audit_visuals.py`

产物：

- `workspace/core4d/results/E103/scene_inertial_audit.tsv`
- `workspace/core4d/results/E103/scene_geometry_audit.tsv`
- `workspace/core4d/results/E103/affected_scene_registry.tsv`
- `workspace/core4d/results/E103/summary.md`
- `workspace/core4d/results/E103/result_validity_reset.md`
- `workspace/core4d/results/E103/visuals/scene_audit/REVIEW.md`

## 关键结果

| 项 | 结果 |
|---|---:|
| 现有 `scene.xml` 审计行 | 198 |
| 缺失 canonical template 占位 | 3 |
| registry 总行数 | 201 |
| inertial clean | 100 |
| polluted robot inertial | 87 |
| quarantine invalidated rows | 84 |
| canonical templates 需要重建 | 6 |
| scene-level keep clean | 13 |
| review before use | 98 |

需要重建的 canonical source templates：

- `box021_person1`
- `box021_person2`
- `box022_person1`
- `box022_person2`
- `box026_person1`
- `box026_person2`

说明：87 个 polluted existing scene 中，3 个是 canonical source template，进入 `rebuild_source_template`；其余 84 个历史派生目录进入 `quarantine_invalidated_by_scene_inertial_bug`。

## 可视化审查

MuJoCo 3D 证据已生成 6 个代表 case 的 audit sheet 和 turntable video：

- `box023_person1`
- `box004_person1`
- `box021_person1`
- `box026_person2`
- `d003_box021_20231018_029_p2`
- `e091_box026_20231018_039_p2`

High subagent 只读审查结论：PASS。

- 12 个文件均存在且非零。
- audit sheet 均为 `2100x1350`，机器人、物体、metric panel 可见可读。
- video 均为 `960x720`、`72` 帧、`24fps`，可完整解码且有视角变化。
- `REVIEW.md` 状态与 sheet panel 一致。
- 视觉证据支持当前结论：污染 case 的外观可能正常，主要证据来自 `robot unique pairs: 1`、pelvis/hip mass 异常等 metric。

## Claims 验证

| claim | Phase 0 状态 | 说明 |
|---|---|---|
| C1 污染范围可复现、可审计 | PASS | inertial/geometry/registry 三表已生成，统计口径已修正为 198 existing + 3 missing canonical |
| C2 canonical source templates 重建 | PENDING | Phase 1/2 才实现 generator 和重建；本阶段未改 scene XML |
| C3 source template 与 target case 责任分离 | PENDING | 重建后才能重新生成 target case |
| C4 E091/E102 mining 口径修正 | PENDING | 需后续修改 mining script，把 source scene missing 变为 backlog |
| C5 历史结果重新标注有效性 | PARTIAL PASS | `result_validity_reset.md` 已生成；tracker 后续条目需沿用新口径 |

## 结论

E103 Phase 0 支持如下决策：

- `box021_person1` 也必须重建；不能作为 Box022/Box026 的 base。
- `box023_person1` robot inertial clean，可作为重建 robot/world/contact skeleton 的候选 base；其 object collision policy 仍需单独标注，不影响用它复制 robot inertial skeleton。
- 旧 Box021/Box026 polluted scene 上的 dynamics/CEM 成败不能再作为硬正/负标签，只能作为 geometry/semantic prior。
- 下一步进入 Phase 1/2：先实现 source template rebuild generator，保存旧 snapshot，再只重建 6 个 canonical source templates；不批量覆盖 84 个历史派生目录。
