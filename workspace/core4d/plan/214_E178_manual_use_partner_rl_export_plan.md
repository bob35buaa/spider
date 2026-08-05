# Plan 214 — E178 final manual-USE partner-aware RL asset export

_CORE4D Phase 41 S6 extension · 2026-08-05 · status: COMPLETED_

## 1. Objective

以用户指定的
`workspace/core4d/results/E178/s6_downstream/eval/full/user_manual_review_filled.tsv`
作为唯一选择 authority，将全部人工 `USE` 的 E178 CEM 结果导出为 partner-complete RL
输入与 Holosoma motion assets。任何 partner 缺失、歧义、不可读或对齐失败都 fail closed；
禁止退化成 source-only export。

本阶段只做 S6 输入资产构建与 pre-train readiness 验证，不启动 RL policy training，也不把
`RL_EXPORT_READY` 宣称为 RL 成功。

## 2. Frozen authority

- Review SHA256：`d430a8ef125117027bc54e0b7bd9bbeafe5c7a0a2e2a25f17a6cea0fd57c6c9f`
- Final decisions：`USE=13`、`DO_NOT_USE=14`、`PENDING=0`
- Quality：`CLEAN=8`、`MINOR_ACCEPTABLE=5`、`UNUSABLE=14`
- USE object split：bucket003=5、bucket004=1、bucket007=7
- Source metrics：`results/E178/s6_downstream/eval/full/e178_case_metrics.tsv`
- Source/evaluation authority：同目录 `evaluated_manifest_snapshot.tsv`
- Selection：全部且仅 `manual_use_decision=USE`
- Numeric gate 不移除人工 USE；numeric status/failure modes 必须作为风险 metadata 保留

旧 log241 记录的是早期 `23 reviewed / USE=12 / 4 pending` 状态，已被用户指定的当前 filled
TSV 覆盖。本轮不回写或篡改旧 log，只在新日志中记录 authority 演进。

## 3. Partner contract

复用已通过 E187 验证的 canonical adapter：

```text
workspace/core4d/scripts/data_construction_v3/stages/s6_downstream/
finalize_reused_partner_rl.py
```

每个 source case 的 partner 为相同 `(object,date,seq)` 的另一 person，必须从 E174 passing
Stage2b `omnirt_v1/ref_fk` 或必要时 `omnirt_v2/ref_fk` 中唯一解析。partner manifest 至少记录：

- source/partner case/person；
- partner trimmed NPZ、OmniRetarget output、trim window 与 SHA；
- source/partner raw window、frame counts、crop offsets 与 alignment policy；
- partner Stage2b manifest provenance。

输出只有在13/13 source、13/13 partner、13/13 common-window alignment全部通过时才能标记
`RL_EXPORT_READY`。

## 4. Source metadata contract

每条 source row 必须包含并验证：

- `scene_act`、`trajectory`、`contact_mask`、`cem_result_npz`、`cem_video`；
- `source_exp_id=E178`；
- 实际 `spider_method_id`、`hand_collision_variant_id`、retarget/target variant；
- manual decision/quality/reviewer/review SHA；
- numeric release、12门状态与 failure modes；
- material artifact SHA256 与 frame counts。

路径使用可从 repo root 解析的逻辑路径；历史 `/mnt/.../spider_workdirs/...` 通过 canonical
adapter 映射，不能把远端绝对路径直接发布给 consumer。

## 5. Holosoma asset export

在 source/partner manifests 通过后，使用 Holosoma fixed exporter：

1. `registry_gate.py pre-export --target-source both`；
2. 分 object 正式转换 `cem + trajectory` 两个 target source；
3. 预期 `13 case × 2 = 26` 个 `_mj_w_obj_w_partner.npz`；
4. 每个 motion 必须含真实 `partner_hand_pos_w`、`partner_hand_quat_w` 与 contact mask；
5. NPZ finite、frame/manifest一致、motion ID唯一；
6. 注册 manifests、重建 registry draft；
7. `post-export 26/26` 与 `pre-train 26/26`。

Holosoma canonical output root：

```text
/home/ubuntu/Workspace/holosoma/workspace/v3/data/E178_manual_use_partner_rl/
```

bucket003/004/007 使用已在 E187 验证的原始 mesh half-extents；bucket007 object name 必须规范化
为 Holosoma 现有的 `Bucket007` asset 名称。

## 6. Implementation

### Phase A — Exporter and static gates

- 新增 `scripts/experiments/E178/export_manual_use_partner_rl.py`；
- 新增 canonical wrapper `scripts/launch/active/run_E178_manual_use_partner_rl_export.sh`；
- 冻结 authority SHA/counts；
- dry-run / py_compile / shell syntax / targeted contract test / diff-check。

### Phase B — Source + partner publication

产出到：

```text
workspace/core4d/results/E178/s6_downstream/rl_export/
  rl_export_input.tsv
  partner_omnirt/rl_partner_omnirt_manifest.tsv
  paired_rl_export_input.tsv
  partner_resolution_audit.tsv
  manual_review_snapshot.tsv
  rl_export_summary.json
```

发布必须 staging→atomic replace；重复运行应 deterministic/resume-safe，不覆盖历史 CEM/eval。

### Phase C — Holosoma conversion and registry gates

- pre-export gate；
- dry-run；
- object-wise full conversion；
- independent NPZ audit；
- registry registration/rebuild；
- post-export/pre-train gates；
- validation JSON 回写 E178 S6 evidence。

## 7. Claims / acceptance criteria

| Claim | Criterion |
|---|---|
| C0 Manual authority | SHA精确匹配；13 USE / 14 DNU / 0 pending；集合唯一 |
| C1 Source completeness | 13/13 scene/trajectory/contact/CEM/video存在、可读并hash |
| C2 Partner completeness | 13/13唯一 opposite-person passing Stage2b partner |
| C3 Alignment | 13/13 common raw window有效，source/partner/CEM/contact帧可解释 |
| C4 Canonical manifests | source/partner/paired/alignment case集合一致，全部RL_EXPORT_READY |
| C5 Holosoma assets | 26/26 partner-enabled motion NPZ finite且schema/frame audit通过 |
| C6 Registry readiness | post-export 26/26、pre-train 26/26 |
| C7 Governance | 不改E178 S1–S5/CEM/eval历史；不启动训练、不夸大为RL成功 |

## 8. Commands

本地 source/partner export：

```bash
bash workspace/core4d/scripts/launch/active/run_E178_manual_use_partner_rl_export.sh export
```

Holosoma 的确切命令由 wrapper 固化；先 dry-run，再按 object 执行，禁止直接在日志里保留
不可复现的裸命令。

## 9. Failure policy

- authority drift：停止，要求重新冻结用户 authority；
- partner missing/ambiguous/invalid/alignment：输出明确 blocked audit，不发布 ready manifest；
- Holosoma asset/schema issue：保留已通过的 Core4D manifests，修复 consumer-side最小问题后重试；
- 不以 source-only、零值 partner、复制 source actor 或最短长度静默裁切作为 fallback。
