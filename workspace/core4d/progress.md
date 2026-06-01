# E101 Progress — 2026-05-31

## 状态：E101 Phase 1 DONE，触发 stop-loss，不启动 Phase 2

- [x] 读取 E098-E100 log、E101 plan、现有 Phase 1 产物
- [x] 确认现有结果：box004_083_p2 seed0/1 已跑；box021_18029_p2 seed0、box021_11035_p2 seed0 已跑
- [x] 修 `scripts/E101/replay_gate_cem_results.py`：支持 `qpos` shape `(T,2,nq)`，取 sim channel `[:,0,:]`
- [x] 更新 `scripts/train/train_E101_phase1.sh`：已有产物自动 skip，并补跑 face_changed 的 `030_p1` seed0/1
- [x] 运行 E101 Phase 1 缺失 run（`030_p1` seed0/1 完成）
- [x] 生成 `phase1_gate_summary.tsv`
- [x] 视觉复核关键视频帧
- [x] 写 E101 log + 更新 EXPERIMENT_TRACKER

## 当前已知指标（只读临时计算）

| variant | gate | 关键指标 |
|---|---|---|
| box004_083_p2 seed0 | PASS | pelvis_min 0.656, pelvis_end 0.781, tilt_end 43.0, lie 0.000 |
| box004_083_p2 seed1 | PASS | pelvis_min 0.660, pelvis_end 0.781, tilt_end 43.4, lie 0.000 |
| box021_11035_p2 seed0 | FAIL | pelvis_min 0.154, pelvis_end 0.159, lie 0.541 |
| box021_18029_p2 seed0 | FAIL | pelvis_min 0.521, pelvis_end 0.561, tilt_end 96.4 |
| box021_030_p1 seed0 | FAIL | pelvis_min 0.690, pelvis_end 0.698, tilt_end 78.1 |
| box021_030_p1 seed1 | FAIL | pelvis_min 0.690, pelvis_end 0.698, tilt_end 78.1 |

## 下一步

E101 按 plan stop-loss 收尾：box021 D003 0/4 WORK，Phase 2 不启动。

## E102 planning

- [x] 创建 `workspace/core4d/plan/109_E102_data_expansion_and_rl_ready_plan.md`
- [x] 修订 E102 计划：区分 `current_negative` / `legacy_failure_prior` / `legacy_replayed_negative` / `positive_guard`，历史失败只作为机制先验，不能直接硬排除候选
- [x] E102 Phase 0：E101 failure taxonomy + visualization
  - 新增脚本：`scripts/E102/classify_e101_failures.py`、`build_negative_case_registry.py`、`render_e101_failure_review.py`
  - 输出：`results/E102/e101_failure_taxonomy.tsv`（4 current_negative + 2 positive_guard）
  - 输出：`results/E102/negative_case_registry.tsv`（case-level registry，current negative 会归一化去掉 `_upperobj_e083/_e092_dyn` 派生后缀）
  - 输出：`results/E102/legacy_failure_replay.tsv`（历史失败保留为 prior，未升级）
  - 输出：`results/E102/visuals/e101_failure_review/`（6 rollout × 3 overlay = 18 jpg + REVIEW.md）
  - 视觉 sidecar 审查确认：box004 guard WORK；030_p1 = tilted_no_transport + object_miss；11035/035_p2 = pelvis_collapse + lie_on_box；18029_p2 = upperbody_lean_tilt，作为 reward_hacking_residual 子型记录
- [x] E102 Phase 1：Box022 source recovery + raw-contact preflight
  - 新增脚本：`scripts/E102/recover_box022_inventory.py`、`run_box022_preflight.py`、`render_box022_preflight.py`
  - 修正 raw root：旧 `/mnt/ali-sh-1/...` 当前不可用，实际 raw 在 `/mnt/a0ccc676-9496-49f8-a861-f8a1797dec52/...`
  - 输出：`results/E102/box022_inventory.tsv`（10 discovered；当前 data_construction_v2 selected scoped 4，reserve/strike/historical 只保留审计）
  - 输出：`results/E102/box022_missing_sources.tsv`（主要缺 `source_scene_xml`；historical 20231022 raw 仍缺）
  - 输出：`results/E102/box022_preflight.tsv`（4/4 `REJECT`：raw 可读但 L/R fingertip close-contact frames 均为 0；同时缺 box022 source scene template）
  - 输出：`results/E102/visuals/box022_preflight/REVIEW.md`
  - 结论：Box022 不进入 E102 CEM；Phase 2 继续挖其它 medium-box inventory
- [x] E102 Phase 2：data_construction_v2 re-mine with fingertip + E101 negative priors
  - 新增脚本：`scripts/E102/mine_v2_with_fingertip.py`、`render_candidate_audit.py`
  - 输出：`results/E102/v2_candidates_with_fingertip.tsv`（0 executable candidates）
  - 输出：`results/E102/v2_candidates_rejected.tsv`（80 held/rejected rows）
  - 输出：`results/E102/v2_candidate_mining_summary.md`、`visuals/candidate_audit/REVIEW.md`
  - 主要原因：source_scene_missing 34、raw_contact_not_pass 30、Box022 preflight_not_pass 8、existing_positive_not_new 3、preprocess_infeasible 2、verified_legacy 2、Box026 large-reach holdout 1
- [x] E102 Phase 3：conditional typical-2 full CEM
  - stop-loss：Phase 2 executable candidates = 0 < 2，按计划跳过 typical-2 full CEM；未启动本地/远程 CEM
- [x] E102 Phase 4：RL-ready set + Holosoma handoff
  - 新增脚本：`scripts/E102/build_rl_ready_set.py`、`render_rl_ready_turntables.py`
  - 输出：`results/E102/rl_ready_set.tsv`（3 READY：box004 083_p1、082_p1、083_p2）
  - 输出：`results/E102/dont_try_this_list.tsv`（16 rows）
  - 输出：`results/E102/holosoma_handoff.md`（PARTIAL：3/5，data expansion 未产生新 WORK）
  - 输出：`results/E102/visuals/rl_ready/`（3 个正样 mp4 copy + REVIEW.md）

## E102 final status

- [x] Phase 0 taxonomy/visualization done
- [x] Phase 1 Box022 preflight done：selected 4 条 raw 可读但 fingertip close-contact=0，REJECT；scene template 仍缺
- [x] Phase 2 re-mine done：0 executable candidates
- [x] Phase 3 full-CEM skipped by stop-loss；没有启动本地/远程 CEM
- [x] Phase 4 handoff done：RL-ready PARTIAL，3 existing box004 positives only

## E102 follow-up audit — source scene template policy

- [x] 复核历史规则：E095/E097 计划和日志都明确 source scene template readiness 只影响 pipeline readiness，缺失时应补齐；不应作为 work-likelihood 负证据。
- [x] 复核历史模板链路：`SCENE_TEMPLATE_GUIDE.md` 要求复制同类 CORE4D scene，替换 object mesh/material/collision/mass/inertia，MuJoCo load 后再由 retarget/trim 第一帧 patch object pos/quat。
- [x] 发现 E102 mining bug：`scripts/E102/mine_v2_with_fingertip.py` 把 `source_scene_exists=False` 直接归为 `reject_source_scene_missing`，应改为 `needs_source_scene_template` backlog 并新建模板后重跑。
- [x] 发现模板风险：`box021_person1/scene.xml` 自 2026-05-13 tracked snapshot 起 robot link inertial 全部被污染为 `mass=29.632` / 同一惯量；任何以 `box021_person1` 为 base 的新模板（如 box026/box022）都会继承这个错误，不能直接复用。
- [x] 2026-05-31 追加全量只读审计：`humanoid_object` 198 个 scene 中 87 个出现同类 robot inertial 污染，集中在 `box021` / `d003_box021` / `box026` / `e091_box026` 派生目录；`box023_person1`、`box004_person1/2` 当前 robot inertial 正常。
- [x] 创建 E103 数据重建计划：`workspace/core4d/plan/110_E103_core4d_scene_rebuild_and_inertial_audit_plan.md`。核心策略：先审计并 quarantine，重建 canonical source templates（box021/022/026 person1/2），不直接批量覆盖 87 个历史派生目录；旧 polluted dynamics 结果做 validity reset。

## E103 Phase 0 — scene inertial audit + quarantine registry

- [x] 新增 E103 审计脚本：
  - `workspace/core4d/scripts/E103/audit_scene_inertials.py`
  - `workspace/core4d/scripts/E103/audit_scene_geometry.py`
  - `workspace/core4d/scripts/E103/build_affected_scene_registry.py`
  - `workspace/core4d/scripts/E103/build_result_validity_reset.py`
  - `workspace/core4d/scripts/E103/render_scene_audit_visuals.py`
- [x] 运行全量只读 audit：
  - `scene_inertial_audit.tsv`：198 个现有 scene，100 clean，87 polluted robot inertial。
  - `scene_geometry_audit.tsv`：198 个现有 scene，geometry/mass policy 需 review 的主要原因是 collision box 与 mesh AABB policy 不完全一致。
  - `affected_scene_registry.tsv`：198 existing rows + 3 missing canonical placeholders = 201 registry rows。
- [x] 生成 validity reset：`workspace/core4d/results/E103/result_validity_reset.md`
  - 84 个历史派生 scene 标记为 `quarantine_invalidated_by_scene_inertial_bug`。
  - 旧 polluted scene 上的 dynamics/CEM 成败不能作为硬正/负标签。
  - 6 个 canonical source templates 必须重建：`box021_person1/2`、`box022_person1/2`、`box026_person1/2`。
- [x] 生成 MuJoCo 3D/视频审计证据：`workspace/core4d/results/E103/visuals/scene_audit/`
  - 6 个代表 case：`box023_person1`、`box004_person1`、`box021_person1`、`box026_person2`、`d003_box021_20231018_029_p2`、`e091_box026_20231018_039_p2`。
  - 每个 case 输出 4-view PNG、audit sheet、72-frame turntable MP4。
- [x] high subagent 可视化审查：PASS。12 个文件均存在且非零；sheet 面板可读；视频可解码且有视角变化；`REVIEW.md` 状态与 panel 一致。
- [x] 写 E103 Phase 0 log：`workspace/core4d/log/127_E103_phase0_scene_inertial_audit_results.md`

## E103 当前结论 / 下一步

- `box021_person1` 已确认不能保留，也不能作为 Box022/Box026 base；它本身必须重建。
- `box023_person1` robot inertial clean，可作为重建 robot/world/contact skeleton 的候选 base；它的 object collision policy 另行 review，不影响复制 robot inertial skeleton。
- 下一步进入 E103 Phase 1/2：实现 source template rebuild generator，先保存旧 snapshot，再只重建 6 个 canonical source templates；不批量覆盖 84 个历史派生目录。

## E103 Phase 1/2 — canonical source template rebuild

- [x] 新增 generator：`workspace/core4d/scripts/E103/rebuild_core4d_box_source_templates.py`
- [x] 冻结 Phase 0 pre-rebuild audit 表：
  - `results/E103/scene_inertial_audit_phase0_pre_rebuild.tsv`
  - `results/E103/scene_geometry_audit_phase0_pre_rebuild.tsv`
  - `results/E103/affected_scene_registry_phase0_pre_rebuild.tsv`
  - `results/E103/summary_phase0_pre_rebuild.md`
  - `results/E103/result_validity_reset_phase0_pre_rebuild.md`
- [x] 保存 pre-rebuild snapshot：`results/E103/pre_rebuild_scene_snapshot/<task>/`
  - 旧存在：`box021_person1/2`、`box026_person2`
  - 旧缺失：`box022_person1/2`、`box026_person1` 写 `MISSING_SOURCE_TEMPLATE.txt`
- [x] 重建 6 个 canonical source templates：
  - `box021_person1`
  - `box021_person2`
  - `box022_person1`
  - `box022_person2`
  - `box026_person1`
  - `box026_person2`
- [x] 生成/更新 `task_info.json`：记录 `e103_rebuilt=true`、base=`box023_person1`、mesh sha256、AABB half-extents、5kg mass policy、box inertia formula、target pose patch policy。
- [x] 清理 stale runtime artifacts：
  - `box021_person2/scene_act.xml` 审计发现仍有污染 robot inertial（30 处 `29.632`），已从 current source template 目录删除。
  - `box021_person1/scene_act.xml` 虽未污染 robot inertial，但属于旧 actuator artifact，已删除；后续 target 生成 trajectory 后必须重新生成。
  - 旧 `box021_person1/0/`、`box021_person2/0/` 已移到 `results/E103/removed_stale_source_runtime_artifacts/`，pre-rebuild snapshot 也保留副本。
- [x] `git add -f` 纳入 canonical scene XML / task_info / `box022_m.obj`，满足 scene XML 快照规则。
- [x] immediate validation：6/6 MuJoCo load OK；`nq=43,nv=41,nu=29`；hand contact site ids `11,15`；robot inertials 与 `box023_person1` 完全一致。
- [x] post-rebuild full audit：
  - 当前 `results/E103/scene_inertial_audit.tsv`：201 existing rows，106 clean，84 polluted historical derived remain。
  - 当前 `results/E103/affected_scene_registry.tsv`：6 个 rebuilt canonical templates 全部 `keep_clean`；canonical templates requiring rebuild = 0。
  - `quarantine_invalidated_by_scene_inertial_bug` 仍为 84；这些历史派生目录没有批量覆盖，旧 dynamics label 仍不可硬用。
- [x] post-rebuild MuJoCo 3D/视频：`results/E103/visuals/source_templates_post_rebuild/`
  - 8 个 case：6 个 rebuilt canonical + `box023_person1`/`box004_person1` guard。
  - high subagent 审查 PASS：8 sheet + 8 mp4 全部可读；6 个 rebuilt canonical 均 `keep_clean / clean / clean`，无 `29.632` 或污染惯量残留。
- [x] 写 E103 Phase 1/2 log：`workspace/core4d/log/128_E103_source_template_rebuild_results.md`

## E103 下一步

- Phase 3：只重新生成 selected target cases，不批量修历史 derived scene。
- Phase 4：重跑 raw/contact/quat/target/preflight audit，修 E091/E102 mining 口径，把 source scene missing 改为 template backlog。
- 暂不直接跑 CEM；clean executable candidate 达标后，另开 E104/E105 做 dynamics/CEM。

## E103 Phase 4 — rebuilt preflight + v2 re-mine

- [x] 修改 `workspace/core4d/scripts/E102/mine_v2_with_fingertip.py`：
  - source scene readiness 改为 live filesystem check。
  - source 缺失时不再输出 `reject_source_scene_missing`，改为 `needs_source_scene_template` / `needs_source_scene_template_then_preflight`。
  - 输出新增 `source_scene_exists_live` / `source_scene_xml_live`。
- [x] 新增 `workspace/core4d/scripts/E103/build_rebuilt_template_preflight.py`
  - 输出：`results/E103/rebuilt_template_preflight.tsv`
  - 结果：6 个 rebuilt canonical source templates 全部 PASS。
- [x] 重跑 Box022 inventory/preflight 到 E103：
  - `results/E103/rebuilt_box022_inventory.tsv`
  - `results/E103/rebuilt_box022_missing_sources.tsv`
  - `results/E103/rebuilt_box022_preflight.tsv`
  - 结果：E102 scoped 4 rows 均 `raw_ok=True`、`source_scene_xml_exists=True`，但 `decision=REJECT`，L/R contact = `0/0`。
- [x] 重跑 v2 mining 到 E103：
  - `results/E103/rebuilt_v2_candidates_with_fingertip.tsv`：0 executable。
  - `results/E103/rebuilt_v2_candidates_rejected.tsv`：80 rejected/held。
  - 初次 route counts：raw_contact_not_pass 61、Box022 preflight_not_pass 8、Box026 large_reach_holdout 4、existing_positive_not_new 3、preprocess_infeasible 2、verified_legacy 2。
  - `reject_source_scene_missing` 已消失；`source_scene_exists_live=True` 为 80/80。
  - 后续 continuation 已用 E103 invalidated registry 修正旧 legacy hard reject；最终口径见下方 `E103 continuation` 小节。
- [x] 可视化：
  - `results/E103/visuals/rebuilt_box022_preflight/`：4 张 preflight panel PNG。
  - `results/E103/visuals/rebuilt_box022_raw_contact_3d/`：4 张 object-local 3D 四视图 + 4 个 turntable MP4。
  - `results/E103/visuals/rebuilt_candidate_audit/REVIEW.md`。
- [x] high subagent 可视化/表格审查：PASS。PNG/MP4 均存在、非零、可读/可解码；视频有视角变化；3D 图支持 Box022 selected rows 无 close fingertip contact；v2 re-mine 表/summary/REVIEW 一致。
- [x] 写 E103 Phase 4 log：`workspace/core4d/log/129_E103_rebuilt_preflight_and_remine_results.md`

## E103 当前状态

- C1/C2/C4/C5 已完成或对当前阶段 PASS。
- C3 仍未完全完成：selected target cases 尚未从 clean source templates 重新生成/验证。
- 当前数据门结论：source template 缺失已不是阻塞；Box022 被 raw fingertip preflight 拒绝；v2 expansion 仍 0 executable，所以不应在 E103 内启动 CEM。

## E103 continuation — invalidated legacy label口径修正

- [x] 用 `affected_scene_registry_phase0_pre_rebuild.tsv` 作为 `--invalidated-registry` 重跑 E103 v2 mining。
- [x] 更新产物：
  - `results/E103/rebuilt_v2_candidates_with_fingertip.tsv`：0 executable。
  - `results/E103/rebuilt_v2_candidates_rejected.tsv`：80 rejected/held。
  - `results/E103/rebuilt_v2_candidate_mining_summary.md`。
  - `results/E103/visuals/rebuilt_candidate_audit/REVIEW.md`。
- [x] 修正后 route counts：raw_contact_not_pass 61、Box022 preflight_not_pass 8、Box026 large_reach_holdout 6、existing_positive_not_new 3、preprocess_infeasible 2。
- [x] `reject_verified_legacy=0`、`reject_source_scene_missing=0`、`source_scene_exists_live=True` 为 80/80；2 条旧污染 legacy label 被标记 `legacy_label_invalidated_by_e103=True`，不再作为 hard reject。

## E103 Phase 3 — selected target regeneration

- [x] 新增 `workspace/core4d/scripts/E103/build_target_regeneration_status.py`，生成第一批 selected target 处理状态表：
  - `results/E103/rebuilt_target_regeneration_status.tsv`
  - `results/E103/rebuilt_target_regeneration_status.md`
- [x] 两条 Box026 selected target 已从 clean `box026_person2/scene.xml` 重新生成并验证：
  - `e091_box026_20231018_039_p2`：`trimmed_qpos_matches_spider_qpos=True`，qpos `[123,43]`，scene `nq=43,nv=41,nu=29`，scene_act `nq=42,nv=41,nu=35`。
  - `e091_box026_20231020_135_p2`：`trimmed_qpos_matches_spider_qpos=True`，qpos `[82,43]`，scene `nq=43,nv=41,nu=29`，scene_act `nq=42,nv=41,nu=35`。
- [x] Target replay 可视化已生成：`results/E103/visuals/rebuilt_target_replay/`，含 2 张 sheet、8 张 keyframe PNG、2 个 MP4。
- [x] 第一批中其余 case 状态：
  - `box021_person1` old guard equivalent：source template 已 clean rebuild；旧 `scene_act/0/trajectory` runtime artifact 已移出，不复用旧 dynamics label/trajectory。
  - Box022 四条 selected target：source template clean/source scene exists，但 raw fingertip close-contact `L/R=0/0`，按 data gate 跳过 target generation/CEM。
- [x] Target regen 后 registry：201 rows；`keep_clean=21`，`quarantine_invalidated_by_scene_inertial_bug=82`，`review_before_use=98`，inertial clean rows=108。
- [x] 写 E103 Phase 3 log：`workspace/core4d/log/130_E103_rebuilt_target_regeneration_results.md`
- [x] high subagent 复审 Phase3/Phase4 更新：PASS。
  - 两个 rebuilt Box026 target verify/registry/replay 视频均自洽。
  - Box022 四条 selected target 正确归为 `box022_raw_contact_reject`，不是 source missing。
  - 最新 mining 与 summary/REVIEW 一致：0 executable、80 rejected、`reject_source_scene_missing=0`、`reject_verified_legacy=0`、`legacy_label_invalidated_by_e103=True` 正好 2。
  - 风险记录：旧 manifest 列 `source_scene_exists` 仍有历史 False，后续消费脚本必须用 `source_scene_exists_live` 作为最终 gate。

## E104 planning — D002 multi-threshold remine

- [x] 用户指出：`reject_raw_contact_not_pass` 中 `not_run=56` 应补跑 D002，不能直接当失败；D002 应改为多阈值模式，同时输出 3cm/5cm 两套候选。
- [x] 创建计划：`workspace/core4d/plan/111_E104_d002_multithreshold_remine_plan.md`
- [x] 修改 D002 raw-contact 脚本：`/home/ubuntu/Workspace/holosoma/workspace/v3/data_construction/scripts/score_raw_contact_candidates_v2.py`
  - 默认 stage0 输入改为现有 D001 JSON。
  - 新增 `stage1-queue=selected-medium-box`，覆盖 `box004/box022/box026` 80 个 case-person。
  - 新增 `--decision-thresholds-m 0.03,0.05`，同一次 raw proxy 输出 3cm/5cm summary 和 timeline。
  - `python -m py_compile` PASS。
- [x] 对 medium-box 80 case-person 全量重跑 D002，包括已跑过的 11 条。
  - 输出目录：`workspace/core4d/results/E104/d002_medium_box_multithreshold/`
  - 3cm：80 rows；46 pass / 5 review / 29 fail。
  - 5cm：80 rows；48 pass / 5 review / 27 fail。
- [x] 新增 `workspace/core4d/scripts/E104/build_threshold_medium_manifests.py`，生成：
  - `results/E104/medium_box_manifest_3cm.tsv/json`
  - `results/E104/medium_box_manifest_5cm.tsv/json`
- [x] 基于两套 manifest 跑 E103-style mining：
  - 3cm：`v2_candidates_3cm_with_fingertip.tsv` 11 executable（全 box004）；69 rejected。
  - 5cm：`v2_candidates_5cm_with_fingertip.tsv` 13 executable（全 box004）；67 rejected。
  - 5cm 相比 3cm 新增 2 条 executable：`e091_box004_20231002_048_p2`、`e091_box004_20231003_2_089_p2`。
- [x] 生成候选对比：`workspace/core4d/results/E104/d002_multithreshold_candidate_comparison.md`
- [x] high subagent 复审 PASS 后修复两个风险：
  - `render_candidate_audit.py` 旧 boilerplate 已改为根据候选数输出；3cm/5cm REVIEW 不再错误写 “<2 candidates skip CEM”。
  - D002 / manifest / mining 字段改为 generic threshold fields：`target_both_active_frac` 表示当前阈值，`target_both_active_frac_3cm` / `_5cm` 分别保留真实 3cm/5cm 值，避免 5cm 文件里 `*_3cm` 承载 5cm 数值。
- [x] 修复后重跑 D002、manifest、3cm/5cm mining、candidate REVIEW、comparison；数量不变：
  - 3cm：46 pass / 5 review / 29 fail；11 executable（全 box004）。
  - 5cm：48 pass / 5 review / 27 fail；13 executable（全 box004）。

## E104 threshold-preflight alignment

- [x] 明确 Box022 fingertip preflight 原默认值：距离阈值来自 `vote_case(... contact_thresh=0.02)`，即 2cm；`20` 是每手最少 contact frame 数，不是距离。
- [x] 已参数化 `run_box022_preflight.py`：新增 `--contact-threshold-m/--contact-threshold-label/--min-contact-frames`，输出 threshold/min-frame/contact fraction 字段，并按阈值写独立 JSON。
- [x] 已更新 `render_box022_preflight.py`：可视化标题显示 threshold/min_frames，footer 不再写死 “never pass”。

- [x] 重跑 E104 Box022 preflight：3cm/5cm 各 10 rows，均为 8 REJECT + 2 SOURCE_BLOCKED；max L/R close-contact frames = 0/0。
- [x] 用阈值对齐后的 Box022 preflight 重跑 E104 mining：3cm 仍 11 executable / 69 rejected；5cm 仍 13 executable / 67 rejected；`reject_box022_preflight_not_pass=8` 不变。
- [x] 更新 E104 comparison、log 131、tracker，明确 `20` 是每手最少 contact frame 数，不是距离阈值。

## E104 policy update — remove Box026 volume-ratio hard holdout

- [x] 用户确认：不需要考虑 `size_vs_box023_volume_ratio > 3.0`，尺寸 gate 只按 D001 `target_medium_between_box023_and_box025`。
- [x] 修改 `workspace/core4d/scripts/E102/mine_v2_with_fingertip.py`：删除 Box026 `size_vs_box023_volume_ratio > 3.0` hard reject 分支。
- [x] 重跑 E104 3cm/5cm mining + candidate REVIEW + comparison。
- [x] 新结果：3cm 41 candidates（39 executable + 2 legacy-risk review；box004=11, box026=30）/39 rejected；5cm 43 candidates（41 executable + 2 legacy-risk review；box004=13, box026=30）/37 rejected。
- [x] 更新 `log/131_E104_d002_multithreshold_remine_results.md`、`results/E104/d002_multithreshold_candidate_comparison.md`、`EXPERIMENT_TRACKER.md`。

- [x] 修正 mining summary 文案：`candidate_legacy_risk_needs_visual` 是候选但不是 executable，summary 改为 `Candidate rows` 后重跑 3cm/5cm mining 和 candidate REVIEW。

## E105 planning — Box026 clean-scene full CEM rerun

- [x] 用户确认：E103 template/inertial bug 使旧 Box026 full CEM 结论不可信，下一步先重跑历史已跑过的 Box026 case，而不是直接扩到 E104 新候选。
- [x] 创建计划：`workspace/core4d/plan/112_E105_box026_clean_scene_full_cem_rerun_plan.md`
- [x] 对齐历史 primary full-CEM rerun matrix：
  - `E092D2_box026_039_p2_dyn` -> `E105R1_box026_039_p2_ref_fk_clean`
  - `E092D3_box026_135_p2_dyn` -> `E105R2_box026_135_p2_ref_fk_clean`
  - `E094P2_box026_039_p2_hbproj` -> `E105A1_box026_039_p2_adaptive_clean`
  - `E094P3_box026_135_p2_hbproj` -> `E105A2_box026_135_p2_adaptive_clean`
- [x] 计划明确不复用旧 `_e092_dyn` / `_e092_omni` derived task；预检查显示这些旧派生 task 仍有 polluted robot inertial。E105 将从 E103 clean target 新建：
  - `e091_box026_20231018_039_p2_e105_clean`
  - `e091_box026_20231020_135_p2_e105_clean`
- [x] 计划要求 CEM 前先过 clean-scene gate：source template、source target、E105 derived task inertial audit clean，qpos 与 E103 verified qpos 对齐，`scene_act` MuJoCo load 维度正确。
- [x] 计划的并行执行拆分：本地先跑 `039_p2 adaptive`，远程 GPU0/GPU1 并行跑两个 ref-fk，wave2 补 `135_p2 adaptive`。
- [x] 计划输出：数值对比表 `results/E105/comparison/box026_clean_vs_old_comparison.{csv,md}`，以及 old-vs-new keyframes/timeline/REVIEW 可视化；`results/E105/` 仍不纳入 git。

## E105 planning update — add E101-style fingertip ablation + pre-CEM visual gate

- [x] 根据用户反馈更新 `workspace/core4d/plan/112_E105_box026_clean_scene_full_cem_rerun_plan.md`：E105 从 4 个 historical primary rerun 扩展为 6 个实验。
- [x] 新增 E101-style secondary ablation：
  - `E105F1_box026_039_p2_fingertip_clean`
  - `E105F2_box026_135_p2_fingertip_clean`
- [x] 明确三条路线区别：
  - `ref_fk_clean` 对齐 E092；
  - `adaptive_clean` 对齐 E094；
  - `fingertip_clean` 对齐 E100/E101 fingertip-aware external target，但历史 E101 没有 Box026 full CEM，所以作为 secondary ablation。
- [x] 并行计划改为 3 卡两轮：
  - wave1：local GPU0 跑 `E105A1`，remote GPU0/1 跑 `E105R1/R2`；
  - wave2：local GPU0 跑 `E105F1`，remote GPU0/1 跑 `E105A2/F2`。
- [x] 新增 CEM 前硬 gate：每个 variant 先渲染 MuJoCo clean derived task replay + target overlay/keyframe sheet，再用 medium subagent 写 `results/E105/pre_cem_visual_review/{variant}/REVIEW.md`；只有 `PASS/PASS_WITH_NOTES` 才允许启动对应 full CEM。
- [x] 计划中新增 `build_e105_fingertip_targets.py`、`render_pre_cem_mujoco_replays.py`、`pre_cem_visual_gate.tsv` 等 E105 产物路径。

## E105 implementation start

- [x] 新增 E105 执行脚本骨架：
  - `workspace/core4d/scripts/E105/e105_common.py`
  - `workspace/core4d/scripts/E105/build_box026_historical_manifest.py`
  - `workspace/core4d/scripts/E105/build_box026_clean_tasks.py`
  - `workspace/core4d/scripts/E105/build_e105_adaptive_targets.py`
  - `workspace/core4d/scripts/E105/build_e105_fingertip_targets.py`
  - `workspace/core4d/scripts/E105/render_pre_cem_mujoco_replays.py`
  - `workspace/core4d/scripts/eval/eval_E105_box026_clean_cem.py`
  - `workspace/core4d/scripts/train/train_E105_box026_clean_full.sh`
  - `workspace/core4d/scripts/run_E105_remote.sh`
  - `workspace/core4d/scripts/pull_E105_remote_results.sh`
- [x] `py_compile` 已通过；训练/远程/拉取脚本已 chmod executable。
- [x] 执行 E105 Phase 0/1/2 non-CEM steps：
  - historical manifest: `workspace/core4d/results/E105/box026_historical_full_cem_manifest.tsv`，4 条 old Box026 full-CEM 对齐记录。
  - clean tasks: `e091_box026_20231018_039_p2_e105_clean`、`e091_box026_20231020_135_p2_e105_clean`，qpos shape 分别 `[123,43]` / `[82,43]`。
  - clean preflight: `workspace/core4d/results/E105/clean_scene_preflight.tsv`；E105 derived/source tasks PASS，旧 `_e092_dyn/_e092_omni` 明确 `INVALIDATED_OLD_DERIVED`。
  - variants/overrides: `workspace/core4d/scripts/E105/variants.tsv` 和 6 个 `examples/config/override/core4d_E105*.yaml`。
  - adaptive targets: `workspace/core4d/results/E105/adaptive_targets/`，2 cases / 4 hand rows / 4 nonblank PNG。
  - fingertip targets: `workspace/core4d/results/E105/fingertip_targets/`，2 rows summary，target metadata 含 source scene/trajectory/vote SHA256。
  - pre-CEM visuals: `workspace/core4d/results/E105/pre_cem_visual_gate.tsv`，6/6 rows 生成 kinematic replay MP4、kinematic sheet、target sheet，当前等待 medium subagent review。
- [x] medium subagent pre-CEM review 完成：6/6 variants 均为 `PASS_WITH_NOTES`，`pre_cem_visual_gate.tsv` 已更新为 `PASS_WITH_NOTES`。
- [x] 已按远程执行规则将 E105 必需代码、overrides、E105 clean derived tasks、E105 adaptive/fingertip targets、pre-CEM review 文件同步到 `spider-remote:/home/xiayb/pHRI_workspace/spider`；未依赖远程 git pull，避免覆盖远程已有 dirty worktree。
- [x] wave1 已按“已有 GPU 程序不 kill，直接叠加跑”的用户要求启动：
  - local GPU0: `E105A1_box026_039_p2_adaptive_clean`，PID `1059161`，log `logs/E105/launch/local_wave1_20260601_013512.log`。
  - remote tmux retry2: `e105_wave1_retry2_20260601_013712`，remote GPU0/1 跑 `E105R1/R2`。
- [x] 远程启动前两次暴露数据同步问题并已修复：
  - 缺 `core4d_E089A_box021_person1_upperobj` override 依赖链 -> 已同步 E089/E088/E087/E085/E084/e074/e073/e071/e062/e041 configs。
  - 缺 `example_datasets/processed/core4d/assets/objects/box026/box026_m.obj` -> 已同步 Box026 asset。
- [ ] wave1 运行中状态：
  - local `E105A1_box026_039_p2_adaptive_clean`: 正常推进到约 `226/246` sim steps，尚未落最终 NPZ/MP4。
  - remote `E105R1_box026_039_p2_ref_fk_clean`: 正常推进到约 `166/246` sim steps。
  - remote `E105R2_box026_135_p2_ref_fk_clean`: 已完成，已落 `trajectory_mjwp_act.npz`、variant `.npz`、MP4、keyframes；final object tracking error `pos=0.1626, quat=0.0707`。
- [x] 修正 E105 remote runner：远程只负责 CEM 产物，不再自动跑 E105 eval（`E105_SKIP_EVAL=1`）；最终 6 个结果拉回本地后统一 eval。原因：远程旧环境缺 `workspace/core4d/scripts/eval/eval_E090.py`，导致 R2 完成后 wrapper eval traceback，但 CEM 输出本身已落盘。
- [x] 本地 wave1 `E105A1_box026_039_p2_adaptive_clean` 完成，已落 root `.npz`、outdir `trajectory_mjwp_act.npz`、MP4、keyframes；final object tracking error `pos=0.2807, quat=0.1508`。
- [x] A1 本地 eval 已写出，单项 `work_status=WORK`：contact `78.0%`，obj mean/max `0.010/0.045m`，pelvis min `0.703m`，head/upper/floor safety `0%`。这只是 interim 结果，最终会在 6/6 拉齐后统一重跑 eval。
- [x] 新增 `workspace/core4d/scripts/E105/render_box026_clean_comparison.py`，用于最终生成 E105 old-vs-new sheets、per-variant contact/object/pelvis timeline PNG、`visuals/box026_clean_rerun/REVIEW.md`；`py_compile` 已通过。
- [x] remote wave1 `E105R1_box026_039_p2_ref_fk_clean` 已完成；`E105R2` 已完成。
- [x] 为提高 GPU 利用率，E105 调度改为 sliding window（最多 3 个 E105 CEM 并行）：R1 继续跑 remote GPU0，同时启动 wave2 的 local F1 和 remote GPU1 F2。
  - local F1: PID `1147975`, launch log `logs/E105/launch/local_wave2_F1_20260601_015535.log`
  - remote F2: tmux `e105_wave2_gpu1_F2_20260601_015535`, wrapper log `logs/E105/remote/remote_gpu1_wave2_F2_full.log`
  - remote GPU0 的 A2 等 R1 完成后启动。
- [x] F1 第一次本地 `nohup` 启动异常早退且 CEM log 为空；已用 `single` 模式前台重启，当前正常加载 fingertip external target 并推进（初始检查到 `12/246` sim steps）。
- [x] remote wave1 `E105R1_box026_039_p2_ref_fk_clean` 完成并落盘；final object tracking error `pos=0.1798, quat=0.1173`。
- [x] remote GPU0 剩余 `E105A2_box026_135_p2_adaptive_clean` 已完成：tmux `e105_wave2_gpu0_A2_20260601_020042`，wrapper log `logs/E105/remote/remote_gpu0_wave2_A2_full.log`。
- [x] local F1、remote F2、remote A2 均已完成并拉回本地。
- [x] 6/6 E105 CEM 全部完成并拉回本地；每个 variant 均有 root NPZ、outdir `trajectory_mjwp_act.npz`、MP4、10 张 keyframes、config。
- [x] 统一 full eval 完成并补入 E098 replay gate：4/6 `WORK` under strict gate。关键结果：
  - `E105R1/R2` ref-fk clean：WORK；contact `76.4/62.2%`，pelvis min `0.704/0.655m`，replay gate PASS。
  - `E105A1` adaptive clean：WORK；contact `78.0%`，pelvis min `0.703m`，lie `11.4%`。
  - `E105A2` adaptive clean：FAIL only by replay body-on-box gate；contact `30.5%`，pelvis min `0.672m`，lie `30.5% >= 30%`。旧 E094P3 的 low pelvis/RH-floor 模式已消失。
  - `E105F1` fingertip clean：FAIL only by replay body-on-box gate；contact `82.1%`，pelvis min `0.709m`，lie `30.1% >= 30%`。
  - `E105F2` fingertip clean：WORK；contact `61.0%`，pelvis min `0.656m`，replay gate PASS。
- [x] 生成并校验 E105 visual package：`workspace/core4d/results/E105/visuals/box026_clean_rerun/REVIEW.md`，6 张 old-vs-new sheet + 6 张 timeline PNG 均非空；REVIEW 已反映 A2/F1 严格 FAIL。

## E105 metric supplement — E026/E081 lower-body proxy

- [x] 用户指出 E105 指标缺腿部接触/干涉项；已参考 `workspace/core4d_collab_retarget/log/26_E026_full_eval_results.md` 和 `workspace/core4d/scripts/eval/eval_E081.py`，把 E026/E081 leg-box SDF proxy 接入 `workspace/core4d/scripts/eval/eval_E105_box026_clean_cem.py`。
- [x] 新增输出：`legobj_timeseries_{variant}.csv`、`leg_box_interference_frac/pct`、`leg_object_contact_frac`、`leg_box_sdf_min_m`、`leg_box_sdf_argmin`、`lowerbody_strict_pass`、`work_status_lowerbody_strict`。
- [x] 重跑 E105 full eval + visual REVIEW。结果：upper-body/replay 口径仍为 4/6 WORK，但 E026/E081 lower-body strict proxy 为 0/6；leg interference 分别为 R1/R2/A1/A2/F1/F2 = `25.2/15.9/60.2/9.8/27.6/18.3%`，全部高于 5% 阈值。
- [x] 更新 log 132、EXPERIMENT_TRACKER、visual REVIEW 和 comparison。结论修正：E105 可推翻旧 polluted scene 上的 Box026 失败解释，但不能作为 RL-ready positive；后续 Box026 候选必须把 lower-body/object interference 纳入硬指标。

## E106 planning start — Box026 30-candidate 3-card batch

- [x] 已按 `experiment-planning-zh` 恢复 E105/E104 上下文，并确认 E104 3cm Box026 candidate 共 30 条：2 条 `candidate_legacy_risk_needs_visual` + 28 条 `candidate_executable`。
- [x] 当前数据 readiness 检查发现：30 条中只有 4 条已有 SPIDER task，只有 2 条已有 Holosoma/OmniRetarget `retargeted/trimmed` 输出；因此 E106 不能直接启动 30 条 CEM，必须先补 data_construction_v2 pipeline。
- [x] 新增计划 `workspace/core4d/plan/113_E106_box026_30candidate_ref_fk_batch_plan.md`：E106 primary route 固定为 E105R-style `ref_fk_clean`，单路线 30 run；本地 GPU0 跑 ranks 1-10，远程 GPU0 跑 11-20，远程 GPU1 跑 21-30；先跑 CEM 不评测，全部回收后统一 eval。
- [x] 新增 E106 脚本：
  - `workspace/core4d/scripts/E106/e106_common.py`
  - `workspace/core4d/scripts/E106/build_e106_box026_manifest.py`
  - `workspace/core4d/scripts/E106/build_e106_clean_tasks.py`
  - `workspace/core4d/scripts/E106/render_pre_cem_replays.py`
  - `workspace/core4d/scripts/run_E106_data_preprocess.sh`
  - `workspace/core4d/scripts/train/train_E106_box026_candidate_batch.sh`
  - `workspace/core4d/scripts/run_E106_remote.sh`
  - `workspace/core4d/scripts/pull_E106_remote_results.sh`
  - `workspace/core4d/scripts/eval/eval_E106_box026_candidate_batch.py`
- [x] `py_compile` 和 `bash -n` 已通过；E106 scripts 已 chmod executable。
- [x] 执行 `build_e106_box026_manifest.py`，输出：
  - frozen candidates: `workspace/core4d/scripts/E106/candidates.tsv`，30 rows；
  - readiness: `workspace/core4d/results/E106/data_readiness.tsv`；
  - pipeline case file: `/home/ubuntu/Workspace/holosoma/workspace/v3/data_construction_v2/inputs/cases_e106_box026_30candidate_pipeline.tsv`；
  - summary: `workspace/core4d/results/E106/manifest_summary.md`。
- [x] readiness 收紧为 CEM 必须同时有 source scene + Holosoma retargeted + Holosoma trimmed + SPIDER qpos；当前 `ready_for_cem=2/30`，`ready_for_clean_task=4/30`，pipeline enabled rows `28`。
- [x] 试跑 clean-task builder：当前可建 4 条 `_e106_clean` derived task 和 override，另外 26 条因缺 SPIDER trajectory 被跳过；训练脚本默认要求 `variants.tsv` 满 30 行，否则拒绝启动 CEM。
- [x] 更新 `EXPERIMENT_TRACKER.md` 添加 E106 planning/Phase0 row。当前尚未启动 data preprocess 或 CEM。
- [x] 已启动 E106 Phase 0 data preprocess tmux：`e106_data_preprocess_20260601_053937`。
  - log: `/home/ubuntu/Workspace/holosoma/workspace/v3/data_construction_v2/logs/stage2b_medium_20260601_053937.log`
  - 命令入口：`bash workspace/core4d/scripts/run_E106_data_preprocess.sh`
  - case file: `/home/ubuntu/Workspace/holosoma/workspace/v3/data_construction_v2/inputs/cases_e106_box026_30candidate_pipeline.tsv`
  - `e091_box026_20231018_041_p2` 已完成到 SPIDER verify，`trimmed_qpos_matches_spider_qpos=true`，qpos shape `[60,43]`。
  - `e091_box026_20231018_044_p2` 已完成到 SPIDER verify，`trimmed_qpos_matches_spider_qpos=true`，qpos shape `[109,43]`。
  - 当前正在跑第三条 `e091_box026_20231020_134_p1` OmniRetarget；没有 kill 任何已有 RL/CEM 进程。
  - 当前粗略计数：30 pipeline rows / 28 enabled；Holosoma retargeted+trimmed `4/30`；SPIDER trajectory `6/30`。

## E106 Phase0 preprocess failure/resume

- [x] Phase0 第一轮在 `e091_box026_20231020_137_p2` 失败并停止；日志 traceback：`RuntimeError: CVXPY solve failed: infeasible`，发生在 OmniRetarget 第一帧。
- [x] 已记录失败到 `workspace/core4d/results/E106/preprocess_failures.tsv`，不静默删除候选；后续 CEM 只对 ready variants 启动，该 row 留作 preprocess failure。
- [x] 更新 `build_e106_box026_manifest.py`：读取 `preprocess_failures.tsv`，readiness 标记 `preprocess_failed=True`，pipeline case file 对失败 row 设置 `enabled=0`。
- [x] 更新 `train_E106_box026_candidate_batch.sh`：CEM 前 variants 数量要求改为 `30 - preprocess_failures`，避免因真实 preprocess infeasible 永久阻塞可运行批次。
- [x] 重新生成 manifest：当前 `ready_for_cem=7/30`，`preprocess_failures=1`，pipeline enabled rows `22`。
- [x] 已启动 Phase0 resume tmux：`e106_data_preprocess_resume_20260601_054330`，继续补剩余 22 条；不会重复已完成 rows，不会重跑失败 row。
- [x] resume 继续推进：当前 ready all-inputs 约 `10/30`，已新增完成 `138_p1`、`138_p2`、`139_p1`；正在跑 `139_p2`。失败仍只有 `137_p2` 一条。
- [x] resume 继续推进：当前 ready all-inputs `14/30`，已新增完成 `139_p2`、`20231023_138_p1`、`20231023_139_p1`、`20231023_137_p1`；失败仍只有 `137_p2` 一条。
- [x] Phase0 第二个 preprocess failure：`e091_box026_20231018_043_p2` 在 OmniRetarget 约 `80/95` 帧处 `RuntimeError: CVXPY solve failed: infeasible`。
- [x] 已追加到 `workspace/core4d/results/E106/preprocess_failures.tsv`，重新生成 manifest 后：`ready_for_cem=14/30`，`preprocess_failures=2`，剩余 pipeline enabled rows `14`，下一条从 `e091_box026_20231018_042_p2` 开始。
- [x] 已启动 Phase0 第二次 resume tmux：`e106_data_preprocess_resume2_20260601_055054`。
- [x] 第二次 resume 继续推进：当前 ready all-inputs `17/30`，新增完成 `042_p2`、`038_p2`、`141_p2`；正在处理 `20231023_139_p2`。失败仍为 `137_p2`、`043_p2` 两条。
- [x] 第二次 resume 继续推进：当前 ready all-inputs `23/30`，新增完成 `20231023_139_p2`、`133_p2`、`135_p1`、`133_p1`、`042_p1`、`039_p1`；当前正在处理长序列 `20231023_141_p2`。失败仍为两条。

## E106 execution update — Phase0 complete and clean-task gate fix

- [x] 2026-06-01 checked active preprocess sessions: no `e106_data_preprocess*` tmux session remains running; did not kill or interrupt any existing RL/CEM process.
- [x] Phase0 data readiness now has `28/30` candidates with Holosoma/OmniRetarget `retargeted + trimmed` output and SPIDER `trajectory_kinematic.npz`.
- [x] Two rows remain explicit preprocess rejects, both already recorded in `results/E106/preprocess_failures.tsv`: `e091_box026_20231020_137_p2` and `e091_box026_20231018_043_p2`, both `RuntimeError: CVXPY solve failed: infeasible` in OmniRetarget.
- [x] Patched `workspace/core4d/scripts/E106/build_e106_clean_tasks.py` so `--require-all-ready` expects `30 - preprocess_failures` runnable tasks and treats registered preprocess failures as intentional skips, while still failing on any unexpected missing source task.

## E106 execution update — 28 runnable clean tasks built

- [x] Re-ran `build_e106_box026_manifest.py` after Phase0: manifest now reports `ready_for_clean_task=28/30`, `ready_for_cem=28/30`, `pipeline enabled rows=0`, `preprocess failures=2`.
- [x] Rebuilt clean derived tasks with `build_e106_clean_tasks.py --force --require-all-ready`; produced `workspace/core4d/scripts/E106/variants.tsv` with `28` runnable variants.
- [x] Split after excluding failed preprocess rows: `local-gpu0=9`, `remote-gpu0=9`, `remote-gpu1=10`. This preserves the intended 3-card batch layout with roughly 10 serial cases per GPU.
- [x] Clean-task validation passed for all 28 built rows: source qpos copied exactly, `scene.xml` is `nq=43,nv=41,nu=29`, `scene_act.xml` is `nq=42,nv=41,nu=35`, robot inertial pollution flag is false, and required leg/upper-body object collision pairs are present.

## E106 execution update — pre-CEM visuals and snapshot guard

- [x] Rendered E106 pre-CEM MuJoCo replay package for all `28` runnable variants: `workspace/core4d/results/E106/pre_cem_visual_gate.tsv`, with 28 non-empty MP4 files and 28 non-empty kinematic sheets.
- [x] Started a medium worker subagent to inspect the 28 E106 visual packages and update `workspace/core4d/results/E106/pre_cem_visual_review/{variant}/REVIEW.md`; CEM remains blocked until each launched variant has PASS/PASS_WITH_NOTES.
- [x] Patched `workspace/core4d/scripts/train/train_E106_box026_candidate_batch.sh` so `single`, `local`, `remote-gpu0`, and `remote-gpu1` modes snapshot their derived clean task scenes via `snapshot_scenes.sh E106 ...` before running CEM.

- [x] Added `workspace/core4d/scripts/sync_E106_remote.sh` to rsync E106 scripts, overrides, Box026 asset, 28 clean derived tasks, manifest/readiness/failure tables, and pre-CEM review files to `spider-remote` without git pull.

- [x] Medium worker completed E106 visual gate: `28/28 PASS_WITH_NOTES`, `0 FAIL_PRE_CEM_VISUAL`; runnable variants are all `E106B01`-`E106B30` except preprocess-failed `E106B08` and `E106B16`.
- [x] Removed the noisy final `done` line from E106 train script `list` mode so split lists are machine-consumable.

## E106 execution update — remote sync correction

- [x] Pre-launch remote gate check caught sync-path bug: `train_E106_box026_candidate_batch.sh` had been copied to `workspace/core4d/scripts/` root while `run_E106_remote.sh` expects `workspace/core4d/scripts/train/`.
- [x] Patched `sync_E106_remote.sh` to sync the train script into `workspace/core4d/scripts/train/`; CEM had not been launched yet, so no run was wasted.

## E106 execution update — full CEM launched

- [x] Launched local full CEM queue in tmux `e106_local_full_20260601_060745`: `local-gpu0` split, 9 variants, first variant `E106B01_box026_20231018_039_p2_ref_fk_clean` started on local GPU0.
- [x] Launched remote full CEM queue in tmux `e106_remote_full_20260601_060745`: remote wrapper started `remote-gpu0` (9 variants, first `E106B11`) and `remote-gpu1` (10 variants, first `E106B21`).
- [x] Confirmed run scripts executed `snapshot_scenes.sh E106 ...` before first CEM command on local and remote. Existing RL/GPU processes were not killed; E106 is stacked on current GPU usage as requested.

## E106 eval implementation update

- [x] Replaced E106 eval placeholder with actual post-run evaluator: it now blocks until selected root NPZ/MP4/outdir trajectory outputs exist, then computes E090 object/safety metrics, E098 replay gate, and E105/E026/E081 lower-body leg-object strict proxy.
- [x] Eval outputs will include `full_eval_summary.{json,csv,md}`, per-variant `legobj_timeseries_*.csv`, and `preprocess_rejects.csv` for the two OmniRetarget failures.

- [x] Re-ran local `snapshot_scenes.sh E106` for all 28 runnable derived tasks after noticing remote GPU0/GPU1 snapshot concurrently write the same remote manifest; local `results/E106/scene_snapshot` now contains 28 case directories and a fresh manifest.

- [x] Patched `run_E106_remote.sh` for future reruns: remote wrapper now snapshots both remote splits once before launching GPU0/GPU1 children, then sets `E106_SKIP_SCENE_SNAPSHOT=1` in child train processes to avoid concurrent writes to the same snapshot manifest.

## E106 monitoring — first wave in progress

- [x] 2026-06-01 06:11 checked three queues: local tmux `e106_local_full_20260601_060745` running `E106B01` (~38/246 sim steps); remote wrapper `e106_remote_full_20260601_060745` running GPU0 `E106B11` (~32/162) and GPU1 `E106B21` (~30/196).
- [x] No root NPZ/MP4 outputs yet, expected because all three are still on first full-CEM case. GPU usage remains stacked with existing processes: local ~16GB/32GB, remote ~15GB/49GB per GPU.

- [x] 2026-06-01 06:15 monitor: local first case `E106B01` progressed to ~58/246 sim steps; still no outputs, expected mid-run. One remote SSH monitor attempt reset during key exchange, treated as monitor failure only; no process was killed.

- [x] Updated E106 tracker row from Phase0 pending to full-CEM running: 28 runnable / 2 preprocess rejects, 28/28 pre-CEM visual PASS_WITH_NOTES, three serial queues launched (9/9/10) with eval deferred until all outputs are collected.

- [x] 2026-06-01 06:26 monitor: three queues still healthy. Local `E106B01` ~126/246; remote GPU0 `E106B11` ~106/162; remote GPU1 `E106B21` ~102/196. No NPZ/MP4 outputs yet because all queues remain on first case.

- [x] 2026-06-01 06:41 monitor: remote GPU0 completed first case `E106B11` and wrote root NPZ + MP4; final object tracking error `pos=0.1677, quat=0.2976`. Remote GPU0 automatically started next serial case `E106B12`, confirming split serialization works. Remote GPU1 `E106B21` ~184/196 and local `E106B01` ~222/246 are still running.

- [x] 2026-06-01 06:52 monitor: serialization confirmed on all three queues. Local completed `E106B01` (final obj error pos=0.1781, quat=0.1302) and started `E106B02`; remote GPU0 completed `E106B11` and is running `E106B12`; remote GPU1 completed `E106B21` (final obj error pos=0.0447, quat=0.4328) and is running `E106B22`. Current completed root outputs: local 1, remote 2.

## E106 automation update — wait/pull/eval guard

- [x] Added `workspace/core4d/scripts/wait_pull_eval_E106.sh`: waits until local and remote E106 tmux queues are both gone, pulls remote outputs, checks expected NPZ/MP4 count against `variants.tsv`, then runs the unified E106 eval exactly once.
- [x] The wait script does not evaluate while CEM queues are still running, preserving the requested “先都跑一下，不评测；等都跑完了，再回收结果评测” workflow.

- [x] Launched local monitor tmux `e106_wait_pull_eval_20260601_065413`; initial guard log shows `local_running=1 remote_running=1 local_npz=1 remote_npz=2`, so it is waiting and will not eval until both queues finish.

- [x] 2026-06-01 07:09 monitor: remote GPU0 completed `E106B12` (final obj error pos=0.2506, quat=0.1865) and started `E106B13`; remote count now 3 root outputs. Local `E106B02` ~154/164 and remote GPU1 `E106B22` ~154/164 are still running. Wait/pull/eval monitor still reports both queues active, so no eval has run.

- [x] 2026-06-01 07:15 monitor: local completed `E106B02` (final obj error pos=0.1667, quat=0.0726) and started `E106B03`; remote GPU1 completed `E106B22` (final obj error pos=0.1577, quat=0.1392) and started `E106B23`. Current root outputs: local 2, remote 4, total observed 6/28 before pull.

- [x] 2026-06-01 07:16 monitoring hardening: patched `wait_pull_eval_E106.sh` to treat transient remote SSH failures as `remote_running=1` and require two consecutive reachable all-done checks before pulling/eval. This prevents accidental early pull/eval if remote SSH resets while CEM is still running.

- [x] Restarted only the E106 wait/pull/eval monitor tmux with hardened remote-check logic: killed old monitor `e106_wait_pull_eval_20260601_065413`, started `e106_wait_pull_eval_20260601_071633`. The first check saw a transient remote SSH reset and correctly kept `remote_running=1`; no pull/eval was triggered.

- [x] 2026-06-01 07:18 monitor: local queue still healthy on `E106B03` (~44/120). Remote manual SSH status checks reset during key exchange twice; this is treated as monitor connectivity failure only, not CEM failure. Hardened wait monitor keeps `remote_running=1` on remote SSH failure and will not pull/eval early.

- [x] 2026-06-01 07:21 monitor retry: remote SSH recovered; remote tmux still running. Local `E106B03` ~68/120; remote GPU0 `E106B13` ~96/258; remote GPU1 `E106B23` ~58/192. Completed outputs remain local 2 + remote 4 = 6/28.

- [x] 2026-06-01 07:36 monitor: local completed `E106B03` (final obj error pos=0.2880, quat=0.3126) and started `E106B04`; local outputs now 3. Remote GPU0 `E106B13` ~176/258 and remote GPU1 `E106B23` ~160/192; remote outputs remain 4. Total observed completed before pull: 7/28. Hardened wait monitor is still waiting and has not pulled/evaled.

- [x] 2026-06-01 07:47 monitor: remote GPU1 completed `E106B23` (final obj error pos=0.0726, quat=0.3556) and started `E106B24`; remote outputs now 5. Local `E106B04` ~118/218; remote GPU0 `E106B13` ~224/258. Total observed completed before pull: local 3 + remote 5 = 8/28. Wait monitor remains active and has not pulled/evaled.

- [x] 2026-06-01 07:50 monitor: local tmux `e106_local_full_20260601_060745`, remote tmux `e106_remote_full_20260601_060745`, and wait monitor `e106_wait_pull_eval_20260601_071633` are all still running. Current root outputs remain local 3 NPZ/MP4 and remote 5 NPZ/MP4. Active cases are local `E106B04` (~136/218), remote GPU0 `E106B13` (~246/258), and remote GPU1 `E106B24` (~46/118). No eval outputs exist yet, as intended.

- [x] 2026-06-01 08:02 monitor: local and remote E106 queues still running; wait monitor still active and has not pulled/evaled. Remote GPU0 completed `E106B13` (final obj error pos=0.2011, quat=0.2461) and started `E106B14`; remote outputs now 6. Local `E106B04` is ~202/218 and remote GPU1 `E106B24` is ~102/118, both close to finishing. Total observed completed before pull: local 3 + remote 6 = 9/28.

- [x] 2026-06-01 08:12 monitor: local completed `E106B04` (final obj error pos=0.1856, quat=1.3160) and started `E106B05` (~56/204); remote GPU1 completed `E106B24` (final obj error pos=0.2700, quat=0.2036) and started `E106B25`. Remote GPU0 is on `E106B14`; `B14/B25` logs are newly created and not yet at `sim_steps` in the sampled tail. Current root outputs: local 4 + remote 7 = 11/28. Wait monitor remains active; no eval outputs yet.

- [x] 2026-06-01 08:23 monitor: no new root outputs since 08:12; counts remain local 4 + remote 7 = 11/28. Local `E106B05` is running (~116/204). Remote `E106B14` and `E106B25` are confirmed healthy after inspecting logs and processes: `B14` ~164/284 on remote GPU0, `B25` ~116/236 on remote GPU1. Earlier `startup/no sim_steps` was a monitor parser issue from SSH heredoc regex escaping, not an experiment stall. Existing remote RL Python processes remain untouched.

- [x] 2026-06-01 08:34 monitor: still no new root outputs; counts remain local 4 + remote 7 = 11/28. Progress is healthy: local `E106B05` ~174/204, remote GPU0 `E106B14` ~222/284, remote GPU1 `E106B25` ~166/236. Wait monitor remains active and has not pulled/evaled.

- [x] 2026-06-01 08:45 monitor: local completed `E106B05` (final obj error pos=0.1335, quat=0.0621) and started `E106B06` (~46/214); local outputs now 5. Remote outputs remain 7, but remote GPU0 `E106B14` is ~274/284 and remote GPU1 `E106B25` is ~234/236, both nearly complete. Total observed completed before pull: 12/28. Wait monitor remains active and has not pulled/evaled.

- [x] 2026-06-01 08:50 monitor: remote completed `E106B14` (final obj error pos=0.1242, quat=0.1760) and `E106B25` (final obj error pos=0.1669, quat=0.1556); remote outputs now 9. Remote GPU0 started `E106B15` (~38/180) and remote GPU1 started `E106B26` (~38/216). Local `E106B06` is ~82/214. Total observed completed before pull: local 5 + remote 9 = 14/28. Wait monitor remains active and has not pulled/evaled.

- [x] 2026-06-01 09:02 monitor: no new root outputs since 08:50; counts remain local 5 + remote 9 = 14/28. Progress remains healthy: local `E106B06` ~152/214, remote GPU0 `E106B15` ~98/180, remote GPU1 `E106B26` ~96/216. Remote E106 wrapper and both Python CEM processes are still running. Wait monitor remains active and has not pulled/evaled.

- [x] 2026-06-01 09:13 monitor: still no new root outputs; counts remain local 5 + remote 9 = 14/28. Local `E106B06` is at ~210/214 and near completion. Remote GPU0 `E106B15` is ~152/180; remote GPU1 `E106B26` is ~148/216. Wait monitor remains active and has not pulled/evaled.

- [x] 2026-06-01 09:17 monitor: local completed `E106B06` (final obj error pos=0.0616, quat=0.1216) and started `E106B07` (~28/140); local outputs now 6. Remote outputs remain 9, with GPU0 `E106B15` near completion (~172/180) and GPU1 `E106B26` ~174/216. Total observed completed before pull: local 6 + remote 9 = 15/28. Wait monitor remains active and has not pulled/evaled.

- [x] 2026-06-01 09:23 monitor: remote completed `E106B15` (final obj error pos=0.1357, quat=0.0876) and `E106B26` (final obj error pos=0.1046, quat=0.2166); remote outputs now verified as 11 NPZ/11 MP4. Remote GPU0 started `E106B17` (~30/150; `E106B16` is the registered preprocess failure and skipped), and remote GPU1 started `E106B27` (~12/170). Local `E106B07` is ~58/140. Total observed completed before pull: local 6 + remote 11 = 17/28. Wait monitor remains active and has not pulled/evaled.

- [x] 2026-06-01 09:33 monitor: no new outputs since 09:23; counts remain local 6 + remote 11 = 17/28. Local `E106B07` is ~116/140, remote GPU0 `E106B17` is ~86/150, and remote GPU1 `E106B27` is ~70/170. Wait monitor remains active and has not pulled/evaled.

- [x] 2026-06-01 09:44 monitor: local completed `E106B07` (final obj error pos=0.1258, quat=0.0532) and started `E106B09` (~44/168; `E106B08` is registered preprocess failure and skipped); local outputs now 7. Remote outputs remain 11, with GPU0 `E106B17` ~136/150 and GPU1 `E106B27` ~120/170. Total observed completed before pull: local 7 + remote 11 = 18/28. Wait monitor remains active and has not pulled/evaled.

- [x] 2026-06-01 09:49 monitor: remote GPU0 completed `E106B17` (final obj error pos=0.1344, quat=0.0964) and started `E106B18` (~24/226); remote outputs now 12. Remote GPU1 `E106B27` is ~148/170. Local `E106B09` is ~80/168. Total observed completed before pull: local 7 + remote 12 = 19/28. Wait monitor remains active and has not pulled/evaled.

- [x] 2026-06-01 09:55 monitor: remote GPU1 completed `E106B27` (final obj error pos=0.1114, quat=0.0826) and started `E106B28` (~18/208); remote outputs now 13. Remote GPU0 `E106B18` is ~50/226. Local `E106B09` is ~110/168. Total observed completed before pull: local 7 + remote 13 = 20/28. Wait monitor remains active and has not pulled/evaled.

- [x] 2026-06-01 10:06 monitor: local completed `E106B09` (final obj error pos=0.1816, quat=0.3981) and started local final case `E106B10` (~16/176); local outputs now 8. Remote outputs remain 13, with GPU0 `E106B18` ~104/226 and GPU1 `E106B28` ~74/208. Total observed completed before pull: local 8 + remote 13 = 21/28. Wait monitor remains active and has not pulled/evaled.

- [x] 2026-06-01 10:16 monitor: no new outputs since 10:06; counts remain local 8 + remote 13 = 21/28. Active cases continue normally: local final `E106B10` ~88/176, remote GPU0 `E106B18` ~158/226, remote GPU1 `E106B28` ~124/208. Wait monitor remains active and has not pulled/evaled.

- [x] 2026-06-01 10:27 monitor: no new outputs since 10:06; counts remain local 8 + remote 13 = 21/28. Active cases are near completion: local final `E106B10` ~146/176, remote GPU0 `E106B18` ~220/226, remote GPU1 `E106B28` ~176/208. Wait monitor remains active and has not pulled/evaled.

- [x] 2026-06-01 10:31 monitor: local final case `E106B10` completed (final obj error pos=0.0680, quat=0.2163); local split is now complete with 9 NPZ/9 MP4 and local CEM tmux exited naturally. Remote GPU0 completed `E106B18` (final obj error pos=0.1137, quat=1.4529) and started `E106B19`; remote outputs now 14. Remote GPU1 remains on `E106B28`. Total observed completed before pull: local 9 + remote 14 = 23/28. Wait monitor remains active and has not pulled/evaled.

- [x] 2026-06-01 10:42 monitor: local remains complete with 9 NPZ/9 MP4. Remote GPU1 completed `E106B28` (final obj error pos=0.1374, quat=1.6823) and started `E106B29` (~74/162); remote outputs now 15. Remote GPU0 is running `E106B19` (~86/168). Total observed completed before pull: local 9 + remote 15 = 24/28. Wait monitor correctly reports `local_running=0 remote_running=1` and has not pulled/evaled.

- [x] 2026-06-01 10:53 monitor: no new remote outputs since 10:42; counts remain local 9 + remote 15 = 24/28. Remote active cases are healthy and near completion: GPU0 `E106B19` ~144/168, GPU1 `E106B29` ~130/162. Remote wrapper and both E106 Python processes are still running. Wait monitor remains active and has not pulled/evaled.

- [x] 2026-06-01 10:59 monitor: remote GPU0 completed `E106B19` (final obj error pos=0.1070, quat=0.0760) and started `E106B20` (~18/282); remote outputs now 16. Remote GPU1 `E106B29` is near completion (~158/162). Total observed completed before pull: local 9 + remote 16 = 25/28. Wait monitor remains active and has not pulled/evaled.

- [x] 2026-06-01 11:02 monitor: remote GPU1 completed `E106B29` (final obj error pos=0.2775, quat=0.2429) and started remote final case `E106B30` (~28/192); remote outputs now 17. Remote GPU0 final case `E106B20` is ~36/282. Total observed completed before pull: local 9 + remote 17 = 26/28. Wait monitor remains active and has not pulled/evaled.

- [x] 2026-06-01 11:14 monitor: no new outputs since 11:02; counts remain local 9 + remote 17 = 26/28. Remote final cases are healthy: GPU0 `E106B20` ~90/282, GPU1 `E106B30` ~80/192. Remote wrapper and both E106 Python processes are still running. Wait monitor remains active and has not pulled/evaled.

- [x] 2026-06-01 11:24 monitor: no new outputs since 11:02; counts remain local 9 + remote 17 = 26/28. Remote final cases are still progressing normally: GPU0 `E106B20` ~152/282, GPU1 `E106B30` ~150/192. Remote wrapper remains running; wait monitor remains active and has not pulled/evaled.

- [x] 2026-06-01 11:35 monitor: remote GPU1 final case `E106B30` completed (final obj error pos=0.2295, quat=1.5114); remote outputs now 18 NPZ/18 MP4. Only `E106B20` remains active on remote GPU0, progressing at ~214/282. Local remains complete with 9 NPZ/9 MP4. Wait monitor remains active and has not pulled/evaled.

- [x] 2026-06-01 11:45 monitor: only `E106B20` remains active; remote outputs still 18 NPZ/18 MP4. `E106B20` progressed to ~268/282 and remote tmux is still running, so wait monitor correctly has not pulled/evaled.

- [x] 2026-06-01 11:52 monitor: remote final case `E106B20` completed at 11:48 (final obj error pos=0.1738, quat=0.2750); remote outputs now 19 NPZ/19 MP4 and remote GPU0 wrapper exited. All 28 runnable CEM cases are now complete before pull/eval: local 9 + remote 19. Waiting for hardened wait monitor's consecutive all-done checks before automatic pull/eval.

- [x] 2026-06-01 12:07 E106 auto pull/eval completed: hardened wait monitor pulled remote results after all CEM finished, verified `28/28` root NPZ and `28/28` MP4, then ran unified eval. Outputs written: `full_eval_summary.{json,csv,md}`, `preprocess_rejects.csv`, and 28 `legobj_timeseries_*.csv`.
- [x] 2026-06-01 12:15 E106 result interpretation written to `workspace/core4d/log/133_E106_box026_30candidate_ref_fk_batch_results.md` and tracker updated. Summary: `28/30` runnable, `2/30` OmniRetarget infeasible preprocess rejects, upper-body WORK `15/28`, lower-body strict pass `7/28`, final RL strict positives `4/28` (`E106B05`, `E106B15`, `E106B22`, `E106B27`).

## E107 planning — Box021 clean reconstruction gate

- [x] 创建计划：`workspace/core4d/plan/114_E107_box021_clean_reconstruction_gate_plan.md`
- [x] 确认 E103 后 Box021 状态：`box021_person1/2` source template clean，但 D003/E101 Box021 target 与 CEM 结果均在 template bug 修复前，不能当 hard label。
- [x] 确认 D003 Box021 输入池：15 个 case-person；13 个 D003 preprocess pass，2 个 OmniRetarget infeasible。E107 只重建 13 个 pass rows，2 个失败 rows 保留为 `preprocess_infeasible`。
- [x] 实现 `workspace/core4d/scripts/E107/build_box021_clean_gate.py`
- [x] 运行 clean reconstruction + 3cm/5cm gate：15 rows；13 `cem_ready`，2 `preprocess_infeasible`。
- [x] 13/13 rebuilt targets 校验通过：qpos match、scene `nq=43,nv=41,nu=29`、scene_act `nq=42,nv=41,nu=35`、无 `29.632` robot inertial、collision pairs 完整。
- [x] 生成 replay 可视化：`workspace/core4d/results/E107/visuals/box021_clean_replay/REVIEW.md`，13 sheet + 13 MP4；抽查 `030_p2` 视频 `960x720@24fps`、75 frames。
- [x] 写 E107 log：`workspace/core4d/log/134_E107_box021_clean_reconstruction_gate_results.md`

## E107 Phase 2 — selected-4 full CEM

- [x] 读取 `workspace/core4d/results/E107/selected_case_to_cem.json`：4 个 selected id，均可映射到已存在的 `{id_without_e107}_e107_clean` task。
- [x] 创建计划：`workspace/core4d/plan/115_E107_box021_selected4_full_cem_plan.md`
- [x] 生成 selected-4 variants/overrides/preflight：
  - `workspace/core4d/scripts/E107/selected4_variants.tsv`
  - `workspace/core4d/results/E107/selected4_clean_task_preflight.tsv`
  - 4 个 `examples/config/override/core4d_E107C*.yaml`
  - 4/4 validation_ok；scene/scene_act clean；leg/upper collision pairs complete。
- [x] 生成 pre-CEM replay，并用 medium subagent 审查：4/4 `PASS_WITH_NOTES`；MP4 帧数匹配 `143/129/133/75`，无 `_e092_` 路径污染。
- [x] 首次尝试启动本地 C01 与远程 GPU1 C03；均在启动阶段失败，错误为 `KeyError: 'qvel is not a file in the archive'`。根因：E107 clean reconstruction 只保存了 `qpos`，没有复制旧 D003 trajectory 的 `qvel/ctrl/contact/contact_pos`；这是执行链路/数据构造 bug，不是 CEM 失败。
- [x] 修复 `workspace/core4d/scripts/E107/build_box021_clean_gate.py`：重建 clean target 时复制旧 D003 trajectory NPZ 的全部 arrays，而不是只保存 `qpos`。
- [x] 本地 1 卡 + 远程 2 卡启动 full CEM；远程 GPU0 被已有 R134 训练占用后，改为本地 GPU0 跑 C01→C02、远程 GPU1 跑 C03→C04，停止 remote-gpu0 wait 队列避免重复。
- [x] 回收结果、评估、写 E107 Phase 2 log：4/4 full CEM 完成，4 NPZ + 4 MP4 + eval summary 完整；strict positive 仅 `E107C02_box021_20231011_035_p1_ref_fk_clean`。结果见 `workspace/core4d/log/135_E107_box021_selected4_full_cem_results.md`
- [x] 按用户要求修订 E107 eval 口径：`pelvis_tilt_end` 只作为 diagnostic，不参与 replay pass/fail；同时 E107 `work_status` 不再使用 source duration gate。重跑 eval 后 `C04` 从 replay/upper FAIL 修正为 upper/replay WORK，但 lower-body strict 仍 FAIL；strict positive 仍仅 `C02`。
