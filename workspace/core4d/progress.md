# CORE4D 当前进度

## 归档索引

- [E187 Full至人工比较完整备份](progress_archive/E187_full_manual_comparison_20260803_full_backup.md)
- [E186 production P/R/G完整备份](progress_archive/E186_production_prg_20260802_full_backup.md)
- 更早阶段见 `progress_archive/`。

## 当前活跃状态：E187 RL输入验证完成

### 2026-08-03

- 最终人工authority SHA=`42dd0ef...431b`；`USE=14`、`DNU=8`、`PENDING=0`。
- 用户批准全部14条人工USE进入下游；numeric仅`5/14 PASS`，其余9条风险作为metadata保留。
- Source/partner/paired/alignment均`14/14 RL_EXPORT_READY`；partner为同sequence另一person的E174 passing OmniRetarget motion。
- Holosoma正式生成14个case × `(CEM, trajectory)`=`28`条paired motion，全部含contact mask和真实partner手轨迹。
- 三份manifest均通过post-export；registry中28个E187 motion ID全部通过pre-train gate。
- C9仍为technical `FAIL`、progression authority=`USER_WAIVED`；当前状态是`DOWNSTREAM_RL_INPUT_VALIDATION_PASS`，不是RL policy成功。
- 正式记录：[plan211](plan/211_E187_manual_use_partner_rl_export_plan.md)、[log261](log/261_E187_manual_use_partner_rl_export_results.md)。

### 当前未决

- 若要继续完整RL实验，需另行冻结run ID、训练arm、seed/GPU、E178 baseline、成功指标和9条numeric FAIL人工USE的安全停止条件。
- 本阶段不启动完整RL训练。

### 验证

- exporter py_compile、targeted diff-check、14-row independent audit全部PASS；
- Holosoma pre-export `14/14`、NPZ结构/NaN audit `28/28`、post-export三manifest、pre-train `28/28`全部PASS；
- Tracker描述45字符，log INDEX含261；未修改E178/shared S6语义或E187 S1–S5/Full artifact。
# E187 latest manual authority / RL-export handoff (2026-08-03)

- Latest filled review: `results/E187/s6_downstream/eval/full/user_manual_review_filled.tsv`
- SHA256: `42dd0ef48cb47dde6cd35d88db0a4e46932e89bbe123227eae2b98f5252f431b`
- Final manual counts: `USE=14`, `DO_NOT_USE=8`, `PENDING=0`; quality: `CLEAN=7`, `MINOR_ACCEPTABLE=7`, `MAJOR_DEFECT=2`, `UNUSABLE=6`.
- User authority: export all 14 manually approved `USE` source cases for downstream RL validation; manual authority overrides numeric filtering for selection, while numeric status/failure modes remain required metadata.
- Partner information is a hard readiness gate: no source-only silent export; missing, ambiguous, or misaligned partner data must fail closed with an explicit blocked status.
- Governance: E187 C9 remains technical `FAIL`, progression authority remains `USER_WAIVED`; RL export/readiness is S6 downstream evidence and must not be reported as RL success.
- Plan created: `plan/211_E187_manual_use_partner_rl_export_plan.md`; next action is read-only discovery of the canonical RL consumer/export partner contract before implementation.
- Contract discovery: canonical S6 pairing is implemented by `finalize_reused_partner_rl.py`. It keeps `rl_export_input.tsv` as the source/CEM authority, infers the opposite CORE4D person in the same object/date/sequence, reuses a uniquely passing `ref_fk` Stage2b motion (variant preference `omnirt_v1`, then `omnirt_v2`), and emits a one-row-per-source paired manifest with hash-pinned partner artifacts. E187 should reuse this adapter rather than invent a new pair schema.
- E187 input facts confirmed: evaluation manifest has 22 unique source rows with `person`, `retarget_variant_id`, `scene_act`, `trajectory`, `contact_mask`, CEM result and frozen method/collision IDs; final manual file has 22 rows and is the selection authority.
- Manual `USE` set resolved to 14 cases across `bucket003`, `bucket004`, and `bucket007`. All partner case IDs are defined as the opposite person in the same sequence. E174 exposes canonical passing Stage2b manifests for `omnirt_v1/ref_fk` and `omnirt_v2/ref_fk`.
- Discovery caveat: Stage2b manifests retain historical remote absolute artifact paths under `/mnt/.../workspace/core4d/...`; the shared adapter intentionally remaps these to the local repository by the `/workspace/core4d/` marker. Ad-hoc direct path loading fails without this canonical remapping, so validation/export must use the adapter's resolver.
- Downstream contract confirmed from Holosoma v3 planning/registry docs: a new SPIDER handoff must first be ingested into the Holosoma upstream registry, then use the fixed export entry `rl_export_input.tsv + partner_omnirt_manifest.tsv -> manifest.tsv`; consumers must not scan CEM directories or use a one-off source-only path. The paired TSV remains useful audit evidence, while the fixed exporter consumes the separate source and partner manifests.
- E187 numeric risk source is `e187_case_metrics.tsv`; it contains `numeric_release_pass`, `numeric_failure_modes` and per-gate fields. These will be copied as metadata while human `USE` controls export selection.
- Holosoma fixed exporter contract inspected. It accepts all ready cases, requires a unique passing partner with existing `trimmed_npz`, aligns target/partner by common raw window (fallback min-frame crop), converts CEM `qpos[:,0,:]` plus the real opposite-person partner into `_mj_w_obj_w_partner.npz`, and supports `--dry-run`. The formal downstream sequence is `registry_gate.py pre-export` → exporter → register manifest → rebuild registry → post-export/pre-train gates.
- For this handoff, target-source validation should include both CEM and trajectory, and the final motion export should include the contact mask. No full RL training has been authorized by this input-packaging step.
- Implemented E187-specific S6 exporter at `scripts/experiments/E187/export_manual_use_partner_rl.py`. It freezes the final annotation SHA, selects exactly 14 `USE` rows across three objects, retains manual/numeric/C9 metadata, resolves same-sequence opposite-person partners through the shared E174 Stage2b adapter, hashes source/partner artifacts, and audits source/CEM/contact/partner frame counts plus common-raw-window alignment before publishing.
- Implementation intentionally leaves shared S6/E178 code unchanged; compatibility is by reuse/import of the canonical partner adapter rather than modifying legacy semantics.
- E187 exporter execution passed at `2026-08-03T18:54:06+08:00`: source `14/14`, partner `14/14`, pair complete `14/14`, all alignment audits use a valid `common_raw_window`, and all partners resolve to passing E174 `omnirt_v1/ref_fk` artifacts. Object split is bucket003=3, bucket004=2, bucket007=9.
- Exported source risk distribution is preserved: numeric release `5 true / 9 false`; manual quality `7 CLEAN / 7 MINOR_ACCEPTABLE`. Published status is `RL_EXPORT_READY`, not RL success; C9 remains `FAIL / USER_WAIVED`.
- Independent manifest audit passed: source/partner/paired/alignment tables are each 14 rows, case sets are identical, recorded artifact hashes match, and common raw overlap ranges from 76 to 201 source frames.
- Holosoma canonical `registry_gate.py pre-export` passed `14/14` for `target_source=both` with a unique existing partner per source. The subsequent fixed-exporter dry run did not start because the shell's default Python 3.13 lacks `mujoco`; this is an environment-selection issue after the input gate, not a data/partner failure. Next action is to use the repository's declared Holosoma runtime.
- Re-ran the fixed Holosoma exporter dry-run with its declared `hsretargeting` Python. It passed and selected exactly 14 source-partner pairs, planning both `cem` and `trajectory` targets (28 output motions total) with contact-mask inclusion. No files were written by the dry-run.
- Full conversion preflight found a downstream diagnostic-contract gap: Holosoma's fixed exporter has object AABB half-extents for bucket004 but not bucket003/bucket007. E177 mesh evidence gives full extents of approximately `0.541×0.763×0.466 m` and `0.547×0.574×0.570 m` respectively. These extents affect only exported partner-hand surface-distance diagnostics, not pair resolution or motion alignment; exact mesh bounds will be computed before actual conversion.
- Exact original-mesh half-extents computed: bucket003=`0.270393755,0.381407245,0.23287703`; bucket004=`0.161566625,0.231058755,0.15230013`; bucket007=`0.273275865,0.2869532,0.284911265` meters.
- First bucket003 conversion attempt exposed a downstream path bug: a relative `--out-dir` is written relative to the Holosoma root but passed unchanged to a converter whose CWD is deeper, so the converter cannot find its generated input. No manifest was published. Retry will use an explicit absolute output directory; source/partner data remain valid.
- Bucket003 formal Holosoma conversion has been restarted with an absolute output directory and is currently running; this exercises the actual converter rather than dry-run selection.
- Bucket003 formal conversion passed: 3 cases produced 6 paired motions (`cem` + `trajectory`) with output frames `334/209/204` per case and embedded contact masks. Output: `/home/ubuntu/Workspace/holosoma/workspace/v3/data/E187_manual_use_partner_rl/bucket003/`.
- Bucket004 formal conversion is running for 2 cases / 4 motions using the exact original-mesh half-extents.
- Bucket004 formal conversion passed: 2 cases produced 4 paired motions with output frames `237/207` per case and embedded contact masks. Output: `/home/ubuntu/Workspace/holosoma/workspace/v3/data/E187_manual_use_partner_rl/bucket004/`.
- Bucket007 formal conversion is blocked inside Holosoma's generic converter before manifest publication: its object model references `../bucket007/bucket007.obj`, which is absent from the converter's asset tree. The E187 scene and source/partner alignment loaded correctly up to conversion (`76` aligned input frames for the first case). This is a downstream object-asset packaging gap, not an E187 collision/partner gate failure.
- Root cause confirmed as case-sensitive legacy asset naming: Holosoma contains `models/Bucket007/Bucket007.obj` and `g1_29dof_w_Bucket007.xml`, while some E174 rows spell `object_name=bucket007` and select a non-existent lowercase directory. The E187 wrapper now normalizes all bucket007 downstream `object_name` values to canonical `Bucket007`; no mesh or motion data are changed.
- E187 source/partner manifests were regenerated after object-name normalization and again passed 14/14 source, partner, pairing and alignment checks. Bucket007 formal conversion has restarted against the canonical existing `Bucket007` Holosoma asset.
- Bucket007 converter is still running after successfully passing the previous missing-asset point; no new error has been emitted.
- Bucket007 18-motion conversion remains active; converter output is buffered, so completion will be audited from its final manifest and NPZ contents rather than inferred from silence.
- Bucket007 conversion remains active on the 9-case queue; no failure output is present.
- Bucket007 formal conversion passed: 9 cases produced 18 paired motions, with per-case output frames `125,125,189,145,155,154,222,174,174`; every case has both CEM and trajectory outputs with embedded contact masks. Output: `/home/ubuntu/Workspace/holosoma/workspace/v3/data/E187_manual_use_partner_rl/bucket007/`.
- Formal Holosoma conversion total is now 14 cases / 28 paired motion artifacts across bucket003/004/007. Next actions: independent NPZ/manifest audit, register the three manifests, rebuild registry, and run post-export/pre-train gates.
- Independent downstream artifact audit passed: all 28 NPZs have finite numeric arrays, `partner_hand_pos_w=(T,2,3)`, `partner_hand_quat_w=(T,2,4)`, embedded `object_contact=(T,2)`, manifest/output frame agreement, and unique `(case_id,target_source)` rows. Manifest SHA256: bucket003=`8c36db32...a212ce`, bucket004=`ea4d119a...60d5fc`, bucket007=`50688804...9f1b5`.
- Holosoma registry index was checked before mutation and is clean for the target registry/data paths; the three new manifests are not yet registered.
- Registered all three E187 manifests in Holosoma `workspace/v3/registry/motion_manifests.tsv` under owner scope `E187_manual_use_partner_rl`, then rebuilt `workspace/v3/registry_draft` successfully (`upstream=115`, `motions=73`).
- Holosoma `post-export` gate passed for all three manifests (`6+4+18=28` rows). Registry discovery found exactly 28 E187 motion IDs, and `pre-train` gate passed `28/28` for both target types (`spider` CEM and `omnirt` trajectory) with real OmniRetarget partners.
- Added machine-readable downstream evidence at `results/E187/s6_downstream/rl_export/holosoma_downstream_validation.json`; final status is `DOWNSTREAM_RL_INPUT_VALIDATION_PASS`, explicitly bounded below RL policy/training success.
- Formal results recorded in `log/261_E187_manual_use_partner_rl_export_results.md`, including authority, partner/alignment audit, 28-motion Holosoma conversion, registry gates, artifact SHAs, resolved downstream contract issues, governance boundary and the variables that must be frozen before any full RL training comparison.
- Tracker and log INDEX updated to log261/plan211. Final checks pass: exporter compiles, targeted `git diff --check` is clean, source/partner/paired/alignment tables remain 14/14, downstream evidence remains 28/28, bucket007 canonical asset name is frozen, and no shared data-construction/E178 file was modified by this stage.
