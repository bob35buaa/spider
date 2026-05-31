# E103 — CORE4D scene rebuild + inertial audit + data validity reset

日期：2026-05-31
上游：
- E091/E095 data_construction_v2 medium-box pipeline
- E098-E102 exp_diagnostic_v2
- `workspace/core4d/data_preprocess/SCENE_TEMPLATE_GUIDE.md`

## Context

E102 后续审查发现，`source_scene_missing` 不能直接作为候选 reject；缺模板的 case 应进入 `needs_source_scene_template` backlog 并新建模板。

但进一步审查发现更基础的数据问题：`box021_person1/scene.xml` 的 robot link inertial 被污染为同一组 Box021 object mass/inertia：

```text
mass=29.632
diaginertia=0.844374 0.995016 0.604342
```

这组惯量来自 Box021 mesh AABB + 29.632kg 的长方体惯量公式，本应只属于 object，却出现在 pelvis/hip/knee/torso 等所有 robot link 上。当前只读扫描结果：

- `example_datasets/processed/core4d/unitree_g1/humanoid_object` 共 `198` 个 `scene.xml`
- `87` 个 scene 有同类 robot inertial 污染
- 受影响前缀：
  - `box021`: 11
  - `d003_box021`: 68
  - `box026`: 1
  - `e091_box026`: 6
  - `dc_box021`: 1
- `box023_person1` / `box004_person1` / `box004_person2` 当前 robot inertial 正常；box004 已有 WORK 结果仍可作为 positive guard。

因此 E103 先处理数据层，不做新的 reward / CEM 算法优化。数据错误未清零前，后续算法结论都必须降级为不可靠或条件性结论。

## Claims

### C1 — 污染范围可复现、可审计

成功标准：
- 新增 inertial/geometry audit 脚本，输出全量表：
  - `workspace/core4d/results/E103/scene_inertial_audit.tsv`
  - `workspace/core4d/results/E103/scene_geometry_audit.tsv`
  - `workspace/core4d/results/E103/affected_scene_registry.tsv`
- 每个 scene 至少记录：
  - robot inertial unique count；
  - pelvis/hip sample mass；
  - object mass/inertia；
  - object mesh path；
  - object collision half-extents；
  - expected mesh AABB half-extents；
  - `mujoco_load_ok`；
  - `status={clean, polluted_robot_inertial, object_mass_policy_review, geometry_review, missing_asset}`。

### C2 — canonical source templates 重建，不直接污染历史实验目录

成功标准：
- 先只重建 canonical source templates，避免批量覆盖历史 experiment-derived scene：
  - `box021_person1`
  - `box021_person2`
  - `box026_person1`
  - `box026_person2`
  - `box022_person1`
  - `box022_person2`
- 重建 base：优先从干净 `box023_person1/scene.xml` 复制 robot/world/contact/pair skeleton，不再使用 `box021_person1` 作为任何新模板的 base。
- 每个重建模板必须：
  - robot link inertial 与 `box023_person1` 保持一致；
  - object mesh 替换为对应 CORE4D object mesh；
  - object collision half-extents = mesh AABB half-extents（或明确记录 margin，默认不加 margin）；
  - object mass 默认统一 `5.0kg`，若未来使用真实质量，必须单独记录 source；
  - object inertia 用 `mass/12 * box_extents^2` 计算；
  - `scene.xml` MuJoCo load 通过，`nq=43,nv=41,nu=29`。

### C3 — source template 与 target case 责任分离

成功标准：
- source template 只负责干净物体/robot scene skeleton；具体 target case 的 object initial `pos/quat` 仍由 `create_spider_scene_from_template.py` 用 trimmed qpos 第一帧 patch。
- 不直接把旧 `box021_person1` 的 object `pos/quat` 作为真实新数据证据。
- 若重建后要重新评价 old `box021_person1` 或 D003 Box021，必须从对应 raw/trimmed qpos 重新生成 target scene、`trajectory_kinematic.npz`、`scene_act.xml`、contact target 和 gate metrics。

### C4 — E091/E102 mining 口径修正

成功标准：
- E102 mining 不再输出 `reject_source_scene_missing` 作为最终拒绝；改为：
  - `needs_source_scene_template`
  - `needs_source_scene_template_then_preflight`
  - 或在 raw/source mocap 缺失时才是 `blocked_raw_source_missing`
- E091/E102 template preflight 改为 live filesystem check，不使用过期 manifest flag 作为最终判断。
- Box022/Box026 只有在干净 source templates + raw-contact/preflight 通过后才进入 CEM queue。

### C5 — 历史结果重新标注有效性

成功标准：
- 生成 `workspace/core4d/results/E103/result_validity_reset.md`：
  - 明确哪些历史结论仍可信：例如 box004/box023/box025 positive/negative 中不依赖污染 scene 的部分；
  - 哪些结论降级为 `invalidated_by_scene_inertial_bug`：所有 polluted scene 上的 dynamics/CEM 成败；
  - 哪些结论仍可作为 geometry/semantic prior，但不能作为 dynamics label。
- `EXPERIMENT_TRACKER.md` 后续新条目必须标注 E103 之后的数据口径；不回滚旧日志，但在 E103 log 写清 invalidation 范围。

## Non-goals

- 不在 E103 内优化 reward、CEM、posture/upright。
- 不一次性重跑 87 个历史 polluted derived scenes。
- 不把旧 contaminated CEM 结果当作新的 negative/positive label。
- 不在没有 raw/trimmed qpos 的情况下伪造 target trajectory。

## Phases

### Phase 0 — Full audit and quarantine registry

新增脚本：
- `workspace/core4d/scripts/E103/audit_scene_inertials.py`
- `workspace/core4d/scripts/E103/audit_scene_geometry.py`
- `workspace/core4d/scripts/E103/build_affected_scene_registry.py`

输出：
- `workspace/core4d/results/E103/scene_inertial_audit.tsv`
- `workspace/core4d/results/E103/scene_geometry_audit.tsv`
- `workspace/core4d/results/E103/affected_scene_registry.tsv`
- `workspace/core4d/results/E103/summary.md`

审查规则：
- robot inertial unique count 必须大于 5；若所有 robot link mass/inertia 相同，直接 fail。
- pelvis mass 应接近干净 G1 (`~3.8kg`)；若为 `29.632kg`，直接 fail。
- object mass/inertia 单独审查，不与 robot link 混淆。
- collision half-extents 与 mesh AABB half-extents 相对误差默认 <= 1%；若有历史 margin，需要在 `geometry_policy` 字段说明。

### Phase 1 — Rebuild source template generator

修改或新增：
- `workspace/core4d/scripts/E103/rebuild_core4d_box_source_templates.py`
- 可复用 E091 的 object mesh copy / box inertia 逻辑，但禁止使用 polluted base。

设计约束：
- base scene 固定为 clean `box023_person1/scene.xml`，除非 audit 证明另一个 base 更合适。
- object metadata 显式列出：
  - `box021`: `box/box021_m.obj`
  - `box022`: `box/box022_m.obj`
  - `box026`: `box/box026_m.obj`
  - `box004`: 只保留 audit，不默认重建已 WORK template
- object name 大小写必须同时兼容 Holosoma retarget model：
  - SPIDER scene mesh key 用 lower slug (`box021`, `box022`, `box026`)
  - Holosoma object_name 仍按 source case TSV (`Box021`, `Box022`, `Box026`) 传入
- 写 `task_info.json`，包含：
  - source base；
  - mesh sha256；
  - extents/half extents；
  - mass policy；
  - inertia formula；
  - clean robot inertial source；
  - `e103_rebuilt=true`。

### Phase 2 — Rebuild canonical source templates only

目标模板：
- `box021_person1`
- `box021_person2`
- `box026_person1`
- `box026_person2`
- `box022_person1`
- `box022_person2`

操作规则：
- 写入前先保存旧文件快照到：
  - `workspace/core4d/results/E103/pre_rebuild_scene_snapshot/<task>/`
- 重建后立即运行 audit；任何 task 不通过则停止。
- 重建后 `git add -f` 仅对 canonical source templates 和必要 metadata；不 force-add 派生历史目录。

### Phase 3 — Regenerate selected target cases from clean templates

只重建后续会继续用的 selected target，不批量修历史：

第一批建议：
- E102/E091 source-template validation：
  - `box021_person1` old guard equivalent（重新从已有 trajectory/trimmed 或 raw source生成；不能复用旧 dynamics label）
  - `e091_box026_20231018_039_p2`
  - `e091_box026_20231020_135_p2`
  - `e091_box022_20231023_125_p1`
  - `e091_box022_20231023_125_p2`
  - `e091_box022_20231023_126_p1`
  - `e091_box022_20231023_126_p2`

重建流程：
1. 使用 clean source scene 跑 Holosoma/OmniRetarget/trim。
2. `create_spider_scene_from_template.py` patch target `pos/quat`。
3. `spider/process_datasets/core4d.py` 重新生成 trajectory。
4. `generate_scene_act.py` 重新生成 `scene_act.xml`。
5. `verify_processed_case.py` 校验 qpos shape、scene/scene_act load、trimmed qpos match。
6. 重新跑 E099 fingertip raw-contact / quat audit / E100 target builder。

### Phase 4 — Re-run data mining and preflight only

先不跑 CEM。重建后先重新生成：
- `workspace/core4d/results/E103/rebuilt_template_preflight.tsv`
- `workspace/core4d/results/E103/rebuilt_box022_preflight.tsv`
- `workspace/core4d/results/E103/rebuilt_v2_candidates_with_fingertip.tsv`
- `workspace/core4d/results/E103/rebuilt_v2_candidates_rejected.tsv`

进入 CEM 的条件：
- source template clean；
- target scene clean；
- raw fingertip contact evidence 非空；
- D005b/G1 feasibility pass 或至少进入 explicit review queue；
- 不在 `invalidated_by_scene_inertial_bug` 旧 label 上做硬负例继承。

### Phase 5 — Minimal dynamics sanity only after data gates pass

如果 Phase 4 给出至少 2 个 clean executable candidate，单独开 E104/E105 做 dynamics/CEM；E103 只允许做 very small smoke：
- 1 个 clean positive guard：box004 existing clean case，用来确认 audit 不误杀；
- 1 个 rebuilt box021/box026/box022 candidate，只验证 pipeline 能跑，不做 WORK 结论。

## Required tests / checks

每次重建后必须运行：

```bash
.venv/bin/python workspace/core4d/scripts/E103/audit_scene_inertials.py \
  --root example_datasets/processed/core4d/unitree_g1/humanoid_object \
  --out workspace/core4d/results/E103/scene_inertial_audit.tsv

.venv/bin/python workspace/core4d/scripts/E103/audit_scene_geometry.py \
  --root example_datasets/processed/core4d/unitree_g1/humanoid_object \
  --core4d-root /mnt/a0ccc676-9496-49f8-a861-f8a1797dec52/mocap_data/CORE4D/CORE4D_Real \
  --out workspace/core4d/results/E103/scene_geometry_audit.tsv
```

对每个 rebuilt target case 必须运行：

```bash
.venv/bin/python workspace/core4d/data_preprocess/verify_processed_case.py \
  --task <target_task> \
  --source-scene example_datasets/processed/core4d/unitree_g1/humanoid_object/<source_scene_task>/scene.xml \
  --trimmed <trimmed_npz> \
  --out workspace/core4d/results/E103/verify/<target_task>_verify_summary.json
```

## Decision Rules

- 如果 `box023_person1` audit 失败：停止，先找真正 clean base。
- 如果 rebuilt source template robot inertial 与 clean base 不一致：停止。
- 如果 object collision 与 mesh AABB 不一致且无明确 policy：停止。
- 如果 `box021_person1` clean rebuild 后 E089A old WORK 不再成立，旧 WORK 降级为 invalidated，不反推算法失败。
- 如果 Box022 raw fingertip close-contact 仍为 0，即使 template clean，也不能进 CEM。
- 如果 Box026 clean template 后 D005b 仍 near-reject，保留为 H2/reach-support review，不扩大同配置扫描。

## Expected Outcome

E103 的交付物不是更多 WORK case，而是恢复数据可信度：

- 干净 canonical source templates；
- 明确的 polluted scene registry；
- E102 mining 修正为 backlog/template-first；
- 旧 box021/box026 dynamics 结论的 validity reset；
- 给后续 E104/E105 提供可复现、可审计的数据基础。
