# Plan 211 — E187 manual-USE partner-aware RL export and downstream validation

## 1. Objective

Use the final human review authority in `results/E187/s6_downstream/eval/full/user_manual_review_filled.tsv` to export every E187 source case marked `USE` into the canonical downstream RL input format, with complete and aligned partner information. Then run deterministic structural and loader-level validation of the exported package.

This plan covers RL-input construction and validation only. It does not claim RL policy success and does not alter S1–S5 construction facts.

## 2. Frozen authority and selection rule

- Human review SHA256: `42dd0ef48cb47dde6cd35d88db0a4e46932e89bbe123227eae2b98f5252f431b`.
- Expected decisions: `USE=14`, `DO_NOT_USE=8`, `PENDING=0`.
- Export selection: all and only rows whose final human decision is `USE`.
- Numeric/full-CEM gates do not remove a human-approved row. Their status, scores and failure modes remain attached as risk metadata.
- `target_variant_id=ref_fk` remains frozen.
- `source_exp_id=E187` is mandatory.

## 3. Partner hard gate

For each selected source case, resolve the interacting partner from canonical upstream evidence, not from filename guessing alone. Record at minimum:

- source and partner person/actor identifiers;
- source and partner trajectory references;
- sequence/case identity and scene/target identity;
- frame interval, sample count, time base/FPS and alignment rule;
- source and partner artifact SHA256 values;
- partner provenance and resolution status.

Readiness is fail-closed:

- `RL_EXPORT_READY` only if all 14 selected source cases have one unambiguous, loadable and frame-aligned partner;
- missing partner → `RL_EXPORT_BLOCKED_PARTNER_MISSING`;
- multiple unresolved partners → `RL_EXPORT_BLOCKED_PARTNER_AMBIGUOUS`;
- length/frame/time mismatch → `RL_EXPORT_BLOCKED_PARTNER_ALIGNMENT`;
- unreadable/schema-invalid partner → `RL_EXPORT_BLOCKED_PARTNER_INVALID`.

The exporter must never silently emit a source-only substitute. Whether the canonical contract uses one paired row per case or two actor rows per case will be inherited from the existing RL consumer schema and stated explicitly in the manifest.

## 4. Required source metadata

Each exported case/pair must preserve or reference:

- `scene_act`, `trajectory`, `contact_mask`;
- `cem_result_npz`, `cem_status`;
- `source_exp_id`, `spider_method_id`, `hand_collision_variant_id`;
- `target_variant_id`;
- final human decision and quality tier;
- numeric gate status, scores and failure modes;
- partner identity, trajectory, timing/alignment and provenance fields.

All relative paths must be resolvable from the manifest or carry an explicit root contract. Material artifacts must be hashed.

## 5. Implementation stages

### S1 — Contract discovery (read-only)

1. Inspect the canonical S6 exporter/evidence recorder and prior user-approved exports (E170/E172/E173/E178 where available).
2. Inspect the downstream RL loader/config to determine the exact partner representation and required keys.
3. Inspect representative NPZ/manifests without mutating results.
4. Write a compact contract note into the experiment log before implementing deviations or extensions.

### S2 — Authority and partner resolution audit

1. Parse and validate the frozen manual file and its SHA.
2. Assert exactly 14 `USE` rows and no pending rows.
3. Resolve partner data for each selected case.
4. Produce a 14-row partner audit table containing resolution, alignment, paths and hashes.
5. Stop export readiness if any partner hard gate fails; still retain the audit artifact for diagnosis.

### S3 — Canonical export implementation

1. Add an E187 wrapper around the shared exporter when the shared contract is sufficient.
2. If shared code lacks partner fields, extend it backward-compatibly: default behavior and legacy E178 invocation/output must remain unchanged.
3. Make selection deterministic and sorted; embed authority SHA and exporter git SHA/state.
4. Write to a staging directory and publish the final manifest only after every validation passes.
5. Output under `results/E187/s6_downstream/rl_export/`; results remain untracked by git.

### S4 — Downstream validation

Run, in order:

1. authority/count/schema/path/hash validation;
2. load every source and partner trajectory/contact/scene artifact;
3. assert pairwise sequence, frame interval, length and time-base alignment;
4. invoke the actual downstream RL dataset loader or its canonical dry-run/smoke entry point;
5. retrieve at least one batch and verify finite tensors, expected shapes, source/partner separation and metadata round-trip;
6. run a legacy E178 compatibility smoke using its frozen inputs/config, without overwriting historical results.

No full RL training is started unless the repository's established “RL validation” contract explicitly defines it as part of the smoke stage or the user separately approves the compute run.

## 6. Acceptance criteria

Export is accepted only when all conditions hold:

- manual authority SHA matches the frozen SHA;
- selection equals 14 human `USE` cases and excludes all 8 `DO_NOT_USE` cases;
- 14/14 cases pass partner resolution and alignment;
- every required source, CEM, contact and partner field is present;
- all referenced files exist, load and match recorded hashes;
- downstream loader consumes the complete export and returns a valid batch;
- E178 legacy compatibility smoke passes with unchanged legacy semantics;
- summary clearly distinguishes `RL_EXPORT_READY` / loader smoke from RL outcome.

## 7. Outputs

Expected output family:

- `results/E187/s6_downstream/rl_export/rl_export_manifest.tsv` (or canonical existing filename);
- `results/E187/s6_downstream/rl_export/partner_resolution_audit.tsv`;
- `results/E187/s6_downstream/rl_export/rl_export_summary.json`;
- `results/E187/s6_downstream/rl_export/validation_report.json`;
- source/partner payloads or immutable references, according to the discovered consumer contract;
- S6 downstream evidence/registry record;
- `log/261_E187_manual_use_partner_rl_export.md`;
- tracker and progress updates.

Exact filenames may follow an already-established canonical exporter, but the artifact map and SHA256 values must be recorded in log 261.

## 8. Reproducibility and compatibility

- Record command lines, config, Python/runtime details, git commit plus dirty-state note, inputs and output hashes.
- Do not modify or overwrite historical E178/E187 artifacts, including plan 210/log 260 comparison outputs.
- Do not reset, clean or discard unrelated worktree changes.
- Shared-code changes require a focused regression test proving old invocations remain valid.
- Results under `workspace/core4d/results/` are evidence, not git-tracked source.

## 9. Decision rules after validation

- If all gates pass: mark `RL_EXPORT_READY`, record loader-smoke evidence, and hand the package to the existing downstream RL evaluation entry point.
- If a partner gate fails: mark the precise `RL_EXPORT_BLOCKED_PARTNER_*` status and report affected cases; do not downgrade to source-only export.
- If loader compatibility fails: mark `RL_EXPORT_BLOCKED_CONSUMER_CONTRACT`, preserve the staged audit, and fix the smallest backward-compatible contract mismatch before retrying.
