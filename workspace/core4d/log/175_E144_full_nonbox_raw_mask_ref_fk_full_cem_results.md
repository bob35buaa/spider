# E144 Full Nonbox raw_mask_ref_fk Full-CEM Results

## Context

- Plan: `workspace/core4d/plan/153_E144_full_nonbox_raw_mask_ref_fk_cem_plan.md`
- Run root: `workspace/core4d/results/E144/E144_full_nonbox_raw_contact`
- CEM output root: `workspace/core4d/results/E144/cem/full`
- Eval output root: `workspace/core4d/results/E144/eval/full`
- Route: `retarget_variant_id=omnirt_v1`, `target_variant_id=ref_fk`, CEM ablation `raw_mask_ref_fk`
- RL: not launched.

## Data Construction Accounting

- Full nonbox inventory accounting: 1456 rows.
- Flow nonbox candidates: 132 rows (`bucket=46`, `desk=56`, `chair=30`).
- 5cm raw-contact pass: 82 rows.
- Required source templates: 21.
- Template review result: 4 `approve_clean` bucket wall proxies, 17 `needs_manual_edit`.
- Stage2b ready/executed: 11 rows.
- Stage2b blocked by template/manual review: 71 rows.
- S4 target gate: 11 pass, 71 not_run.
- Visual QC: 11 pass after local render sheet review, 71 not_run.
- S5 handoff: 11 `HANDOFF_READY`; additional shared/pending rows remain non-CEM-ready.

## Full CEM Execution

Generated CEM variants:

- `workspace/core4d/scripts/E144/variants.tsv`
- rows: 11
- split: local-gpu0 4, remote-gpu0 4, remote-gpu1 3

Execution:

- local-gpu0 completed 4/4.
- spider-remote completed remote-gpu0 and remote-gpu1 rows.
- A stale remote output for `E144_bucket004_20231003_1_012_p1_raw_mask_ref_fk` was deleted and rerun as a single GPU1 job before final pull.
- `run_E144_remote.sh` was fixed to sync processed object assets under `example_datasets/processed/core4d/assets/objects/<object_key>/`; without this, remote bucket jobs failed on missing `bucket003_m.obj`.
- Several logs end with MuJoCo/EGL cleanup exceptions after core files were written. Final artifact audit was therefore based on required files, not log tail text.

Artifact audit:

- expected variants: 11
- root npz present: 11/11
- full mp4 present: 11/11
- outdir `trajectory_mjwp_act.npz` present: 11/11
- missing outputs: 0

## Evaluation

Command:

```bash
bash workspace/core4d/scripts/eval/eval_E144_raw_mask_ref_fk_full_cem.sh full
```

Summary:

- manifest rows: 11
- evaluated rows: 11
- missing rows: 0
- `cem_status_counts`: `fail=11`
- `downstream_decision_counts`: `DOWNSTREAM_CEM_FAIL=11`
- `rl_export_decision_counts`: `SKIP_CEM_FAIL=11`

Failure modes:

- `cem_work_status_fail`: 6
- `lowerbody_interference`: 5

Aggregate diagnostics:

- mean hand-object physics contact: 36.7%
- mean hand near 5cm: 64.8%

Primary files:

- Metrics: `workspace/core4d/results/E144/eval/full/e144_raw_mask_ref_fk_full_cem_metrics.tsv`
- Eval summary: `workspace/core4d/results/E144/eval/full/e144_summary.json`
- Eval evidence input: `workspace/core4d/results/E144/eval/full/e144_downstream_evidence_input.tsv`
- Eval RL candidates: `workspace/core4d/results/E144/eval/full/e144_rl_export_candidates.tsv`

## S6 Evidence And RL Handoff

S6 downstream evidence was recorded with:

- `workspace/core4d/results/E144/E144_full_nonbox_raw_contact/s6_downstream/downstream_evidence_manifest.tsv`
- `workspace/core4d/results/E144/E144_full_nonbox_raw_contact/s6_downstream/downstream_evidence_summary.json`

S6 evidence rows:

- total rows: 22
- `DOWNSTREAM_CEM_FAIL`: 11 (`omnirt_v1/ref_fk` CEM rows)
- `DOWNSTREAM_NOT_RUN`: 11 (`shared/ref_fk` pending handoff rows)
- `rl_status`: all `not_run`

RL export input was generated, but no RL was launched:

- `workspace/core4d/results/E144/E144_full_nonbox_raw_contact/s6_downstream/rl_export/rl_export_input.tsv`
- `rl_export_decision_counts`: `SKIP_CEM_FAIL=11`, `SKIP_NOT_HANDOFF_READY=11`
- `RL_EXPORT_READY=0`

## Conclusion

E144 now matches the requested full nonbox flow through full CEM for all currently reviewed CEM-ready nonbox rows. The all-nonbox raw-contact/template pipeline is not limited to E108. However, only 11 bucket rows were safely released through template + visual QC, and all 11 full-CEM runs failed downstream gates. There are no RL-ready rows, and RL was not run.

Remaining blockers:

- 17 nonbox source templates still require manual edit/review before their 71 raw-contact-pass rows can enter Stage2b/CEM.
- Current bucket `raw_mask_ref_fk` full CEM does not pass downstream metrics; failures are split between lower-body interference and CEM work-status failure.
