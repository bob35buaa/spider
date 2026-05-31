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
