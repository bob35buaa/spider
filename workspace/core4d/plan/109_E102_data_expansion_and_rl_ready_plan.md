# E102 — Stage 4: E101 failure taxonomy + Box022/data expansion + RL-ready handoff

日期：2026-05-31
分支：`exp/core4d-collab-retarget`
上游：
- `tmp/diagnostic_v2-plan-overview.md`
- E098: `face_utils.py` + replay gate
- E099: fingertip face vote + quat audit + raw-contact 3D renderer
- E100: fingertip-aware target builder + target NPZ
- E101: Phase 1 stop-loss (`box004` 2/2 WORK, `box021 D003` 0/4 WORK)

下游：Holosoma RL handoff / 后续 posture-upright 或 dual-G1 方向

## Context

E101 Phase 1 已验证：

- `box004_083_p2` 在 E100 fingertip target 下 2/2 WORK；
- `box021 D003` 在 E100 fingertip target 下 0/4 WORK；
- `030_p1` 是真正 face_changed 的 D003 case，但 seed0/1 都 FAIL，最终物体误差约 `0.835m`；
- `18029_p2` / `11035(035)_p2` 作为 H1 control 仍分别出现 tilt / pelvis+lie-on-box 失败。

因此 E101 Phase 2 不启动。按照全局计划，E102 原本是 Stage 4 数据扩张；但由于 E101 stop-loss，**E102 必须先补 E101 Phase 3 失败模式分类 + 可视化**，再进入 Box022 / re-mine / RL-ready set。

注意：E098-E100 之前的历史失败没有经过当前的 face/gate/fingertip target 修复，不能和 E101 后修复失败混成同一类“最终负例”。E102 可以参考历史失败模式来定义 taxonomy 和 risk prior，但 mining exclusion 必须区分证据等级。

本实验不动 OmniRetarget IK 算法，不做新的 reward 大改；只做诊断、数据筛选、候选生成、小规模 full CEM 验证、handoff 整理。

## Claims

### C1 — E101 failure taxonomy 完整闭环

**判据**：
- E101 所有 FAIL rollout（至少 `030_p1` seed0/1、`18029_p2` seed0、`11035/035_p2` seed0）都有：
  - gate 指标；
  - 失败标签；
  - mp4；
  - 1fps sheet；
  - 至少 3 张关键帧 overlay（early/contact/end）。
- 标签必须落在 `{motion_level_H2_binding, pelvis_collapse_residual, lie_on_box, reward_hacking_residual, tilted_no_transport, object_miss, other}`；不允许只写 unknown。
- 输出 `workspace/core4d/results/E102/e101_failure_taxonomy.tsv` 和 `workspace/core4d/results/E102/visuals/e101_failure_review/REVIEW.md`。
- `negative_case_registry.tsv` 必须包含 `evidence_level`：
  - `current_negative`: E098-E100 修复后仍失败的 E101 rollout，可作为硬负例；
  - `legacy_failure_prior`: E098-E100 前的失败，只能作为机制/风险先验，不能单独硬排除；
  - `legacy_replayed_negative`: 历史失败用 E098 replay gate/当前 target 口径复核后仍失败，才可升级为负例；
  - `positive_guard`: 已知 WORK guard，不进入负例排除。

### C2 — Box022 raw-contact preflight

**判据**：
- 从 historical manifest / data_construction_v2 inventory 恢复 Box022 候选全集（目标 6 case；若 raw 路径缺失，必须写出 missing-source 表）。
- 每个可解析 Box022 case 跑 raw-contact + fingertip face audit + quat audit + reach/support preflight。
- 输出 `workspace/core4d/results/E102/box022_preflight.tsv`，每行 PASS/REJECT + 原因 + 视频路径。
- 每个 REJECT 必须有可视化证据，不允许仅用数值拒绝。

### C3 — data_construction_v2 re-mine with E099/E101 constraints

**判据**：
- 新 mining 脚本必须合入：
  - E099 fingertip-vote face；
  - E099 quat `disable_world_up`；
  - E097 legacy outcome exclusion，但只排除 source/preprocess 硬不可行或已验证非目标的 case；
  - E101 `current_negative` list（box021 D003 target-fix failed，不再作为 worklike 候选）；
  - historical failure priors（E098-E100 前失败只参与风险打分/标签，不直接硬排除）；
  - box004 known-positive pattern as lower-bound positive reference；
  - object size prior: prioritize boxes in the **box023-to-box025 medium range**, not only box004/box023-near cases。
- 输出 `workspace/core4d/results/E102/v2_candidates_with_fingertip.tsv`。
- 至少给出 5 个非空候选；如果不足 5 个，必须区分是 inventory 真空、raw 缺失、preprocess infeasible，还是 legacy exclusion 导致。

### C4 — new candidate full CEM stop-loss

**判据**：
- 只在 C2/C3 给出可执行候选后启动。
- 先选 2 个 typical candidate 双卡 full CEM：
  - 1 个 best-score medium-range candidate（尺寸在 box023 到 box025 之间，且 fingertip/reach/support 最像已 WORK pattern）；
  - 1 个 high-risk/high-reward（优先 Box022 或非 box004 medium box）。
- 使用 E100 target builder 生成 external target；full CEM 32 iter；每 case 1-2 seed。
- 验证条件：至少 1 个达成 replay gate PASS + visual WORK，才扩到最多 5 个候选。
- 若 2 个 typical 全 FAIL，停止扩展，只写失败归因。

### C5 — RL-ready handoff

**判据**：
- 整理所有 SPIDER-WORK case 到 `workspace/core4d/results/E102/rl_ready_set.tsv`。
- 每行包含：`case_name / source_npz / scene_xml / contact_target_npz / gate_metrics / video_path / provenance / status`。
- baseline 必须包含已有 box004 positive set（E094/E096b/E101 可复用）；若 E102 新候选成功，追加到同一表。
- PASS 标准：RL-ready set ≥ 5 case；若只有已有 box004 3 case，则标为 PARTIAL，并明确 “data expansion did not yield enough new WORK case”。
- 输出 `workspace/core4d/results/E102/holosoma_handoff.md`，Holosoma side 不需要再回 spider 手动找文件。

## Phases

### Phase 0 — E101 failure taxonomy + visualization

**目标**：补全全局计划中 E101 Phase 3，但把产出归档到 E102，作为 data expansion 的 evidence-aware failure prior。

新增脚本：
- `workspace/core4d/scripts/E102/classify_e101_failures.py`
- `workspace/core4d/scripts/E102/render_e101_failure_review.py`
- `workspace/core4d/scripts/E102/build_negative_case_registry.py`

输入：
- `workspace/core4d/results/E101/phase1_gate_summary.tsv`
- `workspace/core4d/results/E101/cem_outcome_matrix.tsv`
- `workspace/core4d/results/E101/phase1/*.npz`
- `workspace/core4d/results/E101/phase1/*.mp4`
- E082/E088/E090/E094 historical baselines where available（只作为 `legacy_failure_prior`，除非用 E098 gate/current target 口径重放复核）

输出：
- `workspace/core4d/results/E102/e101_failure_taxonomy.tsv`
- `workspace/core4d/results/E102/negative_case_registry.tsv`
- `workspace/core4d/results/E102/legacy_failure_replay.tsv`
- `workspace/core4d/results/E102/visuals/e101_failure_review/*.jpg`
- `workspace/core4d/results/E102/visuals/e101_failure_review/REVIEW.md`

预期分类：
- `030_p1`: `tilted_no_transport + object_miss`
- `18029_p2`: `reward_hacking_residual / upperbody_lean_tilt`
- `11035/035_p2`: `pelvis_collapse_residual + lie_on_box`
- `box004_083_p2`: positive guard, not negative

历史失败用法：
- E082-E094 的失败模式可用于命名 taxonomy 和解释 E101 失败，例如 pelvis collapse、lie-on-box、torso tilt、reward hacking；
- 但这些失败默认标为 `legacy_failure_prior`，因为它们没有同时包含 E098 face/gate、E099 fingertip audit、E100 target 修复；
- 只有能用当前 replay gate/target 口径复核的历史 case，才允许升级到 `legacy_replayed_negative`；
- Phase 2 mining 不得因为某 case 在旧 CEM 配置里失败就直接排除它，除非失败原因是 source missing、preprocess infeasible、visual impossible 或 current/replayed negative。

### Phase 1 — Box022 source recovery + raw-contact preflight

**目标**：补 v1 §6 + E091 一直没做的 Box022 raw-contact preflight。

新增脚本：
- `workspace/core4d/scripts/E102/recover_box022_inventory.py`
- `workspace/core4d/scripts/E102/run_box022_preflight.py`
- `workspace/core4d/scripts/E102/render_box022_preflight.py`

输入：
- `workspace/core4d/scripts/E098/historical_case_manifest.tsv`
- Holosoma `data_construction_v2` inventory / cases files
- E099 `fingertip_face_vote.py`
- E099 `quat_identity_audit.py`
- E098 `face_utils.py`

输出：
- `workspace/core4d/results/E102/box022_inventory.tsv`
- `workspace/core4d/results/E102/box022_missing_sources.tsv`
- `workspace/core4d/results/E102/box022_preflight.tsv`
- `workspace/core4d/results/E102/visuals/box022_preflight/*.mp4`
- `workspace/core4d/results/E102/visuals/box022_preflight/REVIEW.md`

判定维度：
- raw source exists / missing；
- fingertip contact strength；
- fingertip-vote face stability；
- quat `disable_world_up`；
- reach/support proxy；
- Stage2b/D005b status if available；
- visual PASS/REJECT。

### Phase 2 — Re-mine data_construction_v2 with fingertip + negative priors

**目标**：避免 E097 phantom candidates，重新挖 box023-to-box025 medium-size range 内的可执行候选，并补 Box022/medium-box 路线。

新增脚本：
- `workspace/core4d/scripts/E102/mine_v2_with_fingertip.py`
- `workspace/core4d/scripts/E102/render_candidate_audit.py`

输入：
- E095/E097 mining banks
- E099 fingertip stats + quat audit
- E101/E102 negative registry（按 `evidence_level` 使用）
- Box022 preflight
- known-positive box004 set（作为正样下界/行为参考，不作为尺寸硬模板）

输出：
- `workspace/core4d/results/E102/v2_candidates_with_fingertip.tsv`
- `workspace/core4d/results/E102/v2_candidates_rejected.tsv`
- `workspace/core4d/results/E102/visuals/candidate_audit/*.mp4`
- `workspace/core4d/results/E102/visuals/candidate_audit/REVIEW.md`

候选排序原则：
- 优先尺寸落在 box023 到 box025 之间的 medium-box 候选；box004/box023 只作为 lower-bound positive reference，不把候选限制成“接近 box004/box023”；
- 在 medium 尺寸范围内，再按 fingertip contact strength、support、reach、quat/world-up 风险排序；
- 硬排除 `current_negative`（例如 box021 D003 target-fix negative）和 `legacy_replayed_negative`；
- `legacy_failure_prior` 不硬排除，只降低排序或要求人工 visual audit；
- Box026 默认 large-reach holdout，除非 fingertip/reach 指标明显改善；
- Box022 只有 preflight PASS 后才能进入 CEM 队列。

### Phase 3 — New candidate target generation + typical-2 full CEM

**条件**：Phase 2 输出至少 2 个 executable candidate。

新增脚本：
- `workspace/core4d/scripts/E102/build_e102_targets.py`
- `workspace/core4d/scripts/train/train_E102_typical2.sh`
- `workspace/core4d/scripts/E102/replay_gate_e102_results.py`

流程：
1. 对候选复用 E100 `build_fingertip_aware_target.py` 生成 external target。
2. 先跑 2 个 typical:
   - best-score medium-range candidate（box023-to-box025 尺寸范围内的最高分）；
   - high-risk/high-reward（Box022/medium-box 第一名）。
3. full CEM 32 iter，双卡并行，1-2 seed。
4. replay E098/E101 gate，输出 mp4 + outcome matrix。
5. 若 typical ≥1 WORK，扩到最多 5 个候选；否则 stop-loss。

输出：
- `workspace/core4d/results/E102/fingertip_targets/{case}/spider_contact_target_object_local.npz`
- `workspace/core4d/results/E102/cem_typical2_outcome.tsv`
- `workspace/core4d/results/E102/cem_expanded_outcome.tsv`（条件性）
- `workspace/core4d/results/E102/visuals/cem_review/*.mp4`
- `workspace/core4d/results/E102/visuals/cem_review/REVIEW.md`

### Phase 4 — RL-ready set + Holosoma handoff

**目标**：把可交付给 Holosoma RL 的正样和不要再试的负样都整理清楚。

新增脚本：
- `workspace/core4d/scripts/E102/build_rl_ready_set.py`
- `workspace/core4d/scripts/E102/render_rl_ready_turntables.py`

输入：
- E094 box004_083_p2 WORK
- E096b box004_083_p1 / box004_082_p1 WORK
- E101 box004_083_p2 guard WORK
- E102 new WORK cases（如果有）
- E101/E102 negative registry

输出：
- `workspace/core4d/results/E102/rl_ready_set.tsv`
- `workspace/core4d/results/E102/dont_try_this_list.tsv`
- `workspace/core4d/results/E102/visuals/rl_ready/*.mp4`
- `workspace/core4d/results/E102/holosoma_handoff.md`

## Validation commands

```bash
# Phase 0: E101 failure taxonomy
.venv/bin/python workspace/core4d/scripts/E102/classify_e101_failures.py \
  --e101-summary workspace/core4d/results/E101/phase1_gate_summary.tsv \
  --e101-matrix workspace/core4d/results/E101/cem_outcome_matrix.tsv \
  --out workspace/core4d/results/E102/e101_failure_taxonomy.tsv

.venv/bin/python workspace/core4d/scripts/E102/render_e101_failure_review.py \
  --phase1-dir workspace/core4d/results/E101/phase1 \
  --taxonomy workspace/core4d/results/E102/e101_failure_taxonomy.tsv \
  --out-dir workspace/core4d/results/E102/visuals/e101_failure_review

.venv/bin/python workspace/core4d/scripts/E102/build_negative_case_registry.py \
  --e101-taxonomy workspace/core4d/results/E102/e101_failure_taxonomy.tsv \
  --legacy-manifest workspace/core4d/scripts/E098/historical_case_manifest.tsv \
  --out workspace/core4d/results/E102/negative_case_registry.tsv

# Phase 1: Box022 preflight
.venv/bin/python workspace/core4d/scripts/E102/recover_box022_inventory.py \
  --manifest workspace/core4d/scripts/E098/historical_case_manifest.tsv \
  --out workspace/core4d/results/E102/box022_inventory.tsv

.venv/bin/python workspace/core4d/scripts/E102/run_box022_preflight.py \
  --inventory workspace/core4d/results/E102/box022_inventory.tsv \
  --out workspace/core4d/results/E102/box022_preflight.tsv

# Phase 2: mine candidates
.venv/bin/python workspace/core4d/scripts/E102/mine_v2_with_fingertip.py \
  --negative-registry workspace/core4d/results/E102/negative_case_registry.tsv \
  --out workspace/core4d/results/E102/v2_candidates_with_fingertip.tsv

# Phase 3: conditional typical full CEM
bash workspace/core4d/scripts/train/train_E102_typical2.sh

# Phase 4: handoff
.venv/bin/python workspace/core4d/scripts/E102/build_rl_ready_set.py \
  --out workspace/core4d/results/E102/rl_ready_set.tsv
```

## Stop-loss rules

1. If Phase 0 cannot assign non-unknown labels to all E101 FAIL cases, do not proceed to mining; fix taxonomy first.
2. If `negative_case_registry.tsv` lacks `evidence_level`, do not use it for exclusion.
3. If Box022 raw sources cannot be recovered, mark Box022 as source-blocked and continue re-mine on other inventory; do not fabricate Box022 labels.
4. If Phase 2 yields <2 executable candidates, skip Phase 3 CEM and still produce RL-ready handoff from existing positives.
5. If typical-2 CEM yields 0 WORK, do not expand to 5 cases; write negative analysis and handoff existing positives only.
6. Do not add box021 D003 to RL-ready set unless a future separate posture/upright or dual-G1 plan turns it WORK.

## Risks

| 风险 | 应对 |
|---|---|
| Box022 raw path still missing | 输出 `box022_missing_sources.tsv`；E102 不阻塞在 Box022 上 |
| mining 候选数不足 | 明确是 inventory shortage 还是 exclusion 造成；RL-ready set 标 PARTIAL |
| full CEM GPU 成本高 | typical-2 stop-loss；不一次性扩全量 |
| box004 positive set 重复计数 | `rl_ready_set.tsv` 用 unique source case + provenance，E094/E101 同 case 只保留最佳视频/metrics |
| E101 failure taxonomy 过粗 | 每个标签必须绑定 keyframe/sheet 证据；不接受纯数值标签 |

## Expected result

E102 的最小可接受完成形态：
- E101 failure taxonomy 完整；
- Box022 有明确 source/preflight 状态；
- re-mine 给出候选或明确说明候选为空原因；
- known-positive box004 RL-ready set 可交付；
- 不再把 box021 D003 当作 contact-target 修复路径继续烧 GPU。

理想完成形态：
- Box022 或其它 medium-box 新增 ≥2 个 WORK；
- RL-ready set ≥5 case；
- Holosoma handoff 文档完整，RL 可以直接接管。
