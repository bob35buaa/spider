# E098-E108 Progress Archive — Data Foundation

## E108 — 非 box bucket004 进入 RL smoke (2026-06-02)

- [x] `bucket004_person1` template 经 high subagent review 后置为 `clean_reviewed`，4 条 5cm pass case 完成 Stage2b + target gate。
- [x] 3 条 visual QC pass case 完成 full CEM；`012_p1`、`022_p1` 经 bucket-aware visual review 判定可作为 RL smoke 候选，`021_p1` 因 lower-body/bucket interference 拒绝。
- [x] Holosoma 侧新增 bucket004 motion export、bucket004 handbox reward/config 与固定训练脚本。
- [x] `bucket004_20231003_1_012_p1` 完成 no-partner RL smoke：`2` iterations, `64` envs, total timesteps `3072`，保存 checkpoint `model_00001.pt`。
- [x] S6 downstream evidence 已更新：`012_p1` 为 `DOWNSTREAM_RL_PASS`，`022_p1` 保持 `DOWNSTREAM_CEM_PASS`，`021_p1` 为 `DOWNSTREAM_CEM_FAIL`，`013_p1` 为 visual QC reject。
- [x] 已纳入 verified seed：`existing_cases.tsv` 从 57 行更新到 1931 行。

## E101 — Fingertip-aware target + CEM (2026-05-31)

### Data construction v3 固化补充

- [x] 明确 E098/E099-E101 的纳入口径：E098 是全路线基础 contract；E099-E101 是 `target_variant_id=fingertip_aware` 的 route-specific contract。
- [x] 补充 S5 CEM override handoff 文档。
- [x] 新增代码库级 release audit：`audit_pipeline_release.py`，52/52 checks pass。
- [x] 新增新机器统一检查入口：`run_release_checks.sh`。

### Data construction v3 文档收口

- [x] 将 v3 docs 英文标题改为中文口径；release audit 53/53 checks pass。

### Data construction v3 需求追踪

- [x] 新增 `13_requirements_traceability.md`；release audit 54/54 checks pass。

### Data construction v3 发布就绪

- [x] 新增 `14_release_readiness.md`；release audit 55/55 checks pass。

### E101 Phase 1 结果

| variant | gate | 关键指标 |
|---|---|---|
| box004_083_p2 seed0 | PASS | pelvis_min 0.656, pelvis_end 0.781, tilt_end 43.0, lie 0.000 |
| box004_083_p2 seed1 | PASS | pelvis_min 0.660, pelvis_end 0.781, tilt_end 43.4, lie 0.000 |
| box021_11035_p2 seed0 | FAIL | pelvis_min 0.154, pelvis_end 0.159, lie 0.541 |
| box021_18029_p2 seed0 | FAIL | pelvis_min 0.521, pelvis_end 0.561, tilt_end 96.4 |
| box021_030_p1 seed0 | FAIL | pelvis_min 0.690, pelvis_end 0.698, tilt_end 78.1 |
| box021_030_p1 seed1 | FAIL | pelvis_min 0.690, pelvis_end 0.698, tilt_end 78.1 |

E101 按 plan stop-loss 收尾：box021 D003 0/4 WORK，Phase 2 不启动。

## E102 — Data expansion and RL-ready (2026-05-31)

- [x] Phase 0 taxonomy/visualization done
- [x] Phase 1 Box022 preflight done：selected 4 条 raw 可读但 fingertip close-contact=0，REJECT
- [x] Phase 2 re-mine done：0 executable candidates
- [x] Phase 3 full-CEM skipped by stop-loss
- [x] Phase 4 handoff done：RL-ready PARTIAL，3 existing box004 positives only

### E102 follow-up — source scene template policy

- [x] 发现 E102 mining bug：`source_scene_exists=False` 直接归为 reject，应改为 backlog。
- [x] 发现模板风险：`box021_person1/scene.xml` robot link inertial 全部被污染为 `mass=29.632`；198 个 scene 中 87 个出现同类 robot inertial 污染。

## E103 — Core4D scene rebuild and inertial audit

### Phase 0 — scene inertial audit + quarantine registry

- [x] `scene_inertial_audit.tsv`：198 个现有 scene，100 clean，87 polluted robot inertial。
- [x] 84 个历史派生 scene 标记为 `quarantine_invalidated_by_scene_inertial_bug`。
- [x] 6 个 canonical source templates 必须重建：`box021_person1/2`、`box022_person1/2`、`box026_person1/2`。
- [x] log: `workspace/core4d/log/127_E103_phase0_scene_inertial_audit_results.md`

### Phase 1/2 — canonical source template rebuild

- [x] 重建 6 个 canonical source templates，从 `box023_person1` clean base。
- [x] 6/6 MuJoCo load OK；`nq=43,nv=41,nu=29`；robot inertials 与 `box023_person1` 完全一致。
- [x] post-rebuild full audit：201 existing rows，106 clean，84 polluted historical derived remain。
- [x] log: `workspace/core4d/log/128_E103_source_template_rebuild_results.md`

### Phase 3 — selected target regeneration

- [x] 两条 Box026 selected target 已重新生成并验证：`e091_box026_20231018_039_p2` 和 `e091_box026_20231020_135_p2`。
- [x] log: `workspace/core4d/log/130_E103_rebuilt_target_regeneration_results.md`

### Phase 4 — rebuilt preflight + v2 re-mine

- [x] Box022 preflight：8 REJECT + 2 SOURCE_BLOCKED；max L/R close-contact frames = 0/0。
- [x] v2 mining：0 executable candidates after invalidated legacy label 修正。
- [x] log: `workspace/core4d/log/129_E103_rebuilt_preflight_and_remine_results.md`

### E103 continuation — invalidated legacy label 口径修正

- [x] 2 条旧污染 legacy label 被标记 `legacy_label_invalidated_by_e103=True`，不再作为 hard reject。

## E104 — D002 multi-threshold remine

- [x] 修改 D002 raw-contact 脚本：新增 `--decision-thresholds-m 0.03,0.05`，同次输出 3cm/5cm。
- [x] 结果：3cm 46 pass / 5 review / 29 fail；5cm 48 pass / 5 review / 27 fail。
- [x] 3cm 11 executable（全 box004）；5cm 13 executable（全 box004）。
- [x] 删除 Box026 volume-ratio hard holdout 后：3cm 39 executable（box004=11, box026=28+2 review）；5cm 41 executable（box004=13, box026=28+2 review）。
- [x] log: `workspace/core4d/log/131_E104_d002_multithreshold_remine_results.md`

## E105 — Box026 clean-scene full CEM rerun

- [x] 对齐历史 primary full-CEM rerun matrix：4 historical + 2 fingertip ablation = 6 variants。
- [x] 6/6 E105 CEM 全部完成并拉回本地。
- [x] 统一 full eval：4/6 `WORK` under strict gate。E105R1/R2 ref-fk clean WORK，E105A1 adaptive WORK，E105A2/E105F1 FAIL by replay body-on-box gate，E105F2 WORK。
- [x] lower-body strict proxy 为 0/6；leg interference 全部高于 5% 阈值。
- [x] 结论：E105 可推翻旧 polluted scene 上的 Box026 失败解释，但不能作为 RL-ready positive。
- [x] log: `workspace/core4d/log/132_E105_box026_clean_scene_full_cem_rerun_results.md`

## E106 — Box026 30-candidate ref_fk batch (2026-06-01)

- [x] manifest：30 candidates，28 runnable（2 OmniRetarget infeasible preprocess rejects）。
- [x] Phase0 data preprocess：28/30 with Holosoma/OmniRetarget retargeted+trimmed + SPIDER trajectory。
- [x] 28 clean derived tasks built，pre-CEM visuals `28/28 PASS_WITH_NOTES`。
- [x] full CEM launched and completed：local 9 + remote GPU0 9 + remote GPU1 10 = 28。
- [x] auto pull/eval completed：`28/28` root NPZ and `28/28` MP4。
- [x] 结果：upper-body WORK `15/28`, lower-body strict pass `7/28`, final RL strict positives `4/28` (`E106B05`, `E106B15`, `E106B22`, `E106B27`)。
- [x] log: `workspace/core4d/log/133_E106_box026_30candidate_ref_fk_batch_results.md`

## E107 — Box021 clean reconstruction gate

- [x] D003 Box021 输入池：15 个 case-person；13 个 D003 preprocess pass，2 个 OmniRetarget infeasible。
- [x] 13/13 rebuilt targets 校验通过。
- [x] log: `workspace/core4d/log/134_E107_box021_clean_reconstruction_gate_results.md`

### E107 Phase 2 — selected-4 full CEM

- [x] 4/4 full CEM 完成；strict positive 仅 `E107C02_box021_20231011_035_p1_ref_fk_clean`。
- [x] 按用户要求修订 eval 口径后 `C04` 从 FAIL 修正为 upper/replay WORK，但 lower-body strict 仍 FAIL。
- [x] log: `workspace/core4d/log/135_E107_box021_selected4_full_cem_results.md`

## Data construction v3 — reproducibility pipeline (2026-06-01/02)

- [x] 落地 Phase A 文档骨架：`workspace/core4d/docs/data_construction_v3/`
- [x] 落地 Phase B 最小工具脚本：`workspace/core4d/scripts/data_construction_v3/`
- [x] Phase C/S1-S6 全部落地并通过 smoke 验证
- [x] 扩展接口、manual seed、legacy import、visual QC、downstream evidence、可复现性自检全部完成
- [x] 固化 compact smoke suite：`run_smoke_suite.py`
- [x] 脚本目录重组完成：release audit `58/58`，smoke `20/20 pass`
- [x] 纳入可信真实历史结果 seed：`existing_cases.tsv`（57 rows → 1931 rows after E108）
- [x] 提交：commit `af5ab7c Add reproducible Core4D data construction v3 pipeline`；release audit `62/62`
