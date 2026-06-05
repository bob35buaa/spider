# E144 Completion Audit

Date: 2026-06-05

Scope: verify `workspace/core4d/plan/153_E144_full_nonbox_raw_mask_ref_fk_cem_plan.md` against current artifacts.

## Requirement Audit

| Requirement | Evidence | Status |
|---|---|---|
| Full nonbox inventory rows have accounting decisions | `workspace/core4d/results/E144/nonbox_inventory_accounting.tsv`: 1456 rows; inventory nonbox rows also 1456; no missing case ids; all rows have decision/status/reason columns populated | PASS |
| desk/chair are not automatically proxy released | `nonbox_template_review.tsv`, `stage2b_manifest_omnirt_v1_ref_fk.tsv`, and `variants.tsv`: 0 desk/chair `approve_clean`, 0 desk/chair Stage2b pass, 0 desk/chair CEM variants | PASS |
| Unreviewed templates do not enter Stage2b/CEM-ready | Registry + Stage2b + variants: Stage2b pass rows = 11; all CEM variants = 11; all have `template_status` in `clean/clean_reviewed` | PASS |
| Each CEM-ready row has `scene_act`, `trajectory`, `contact_mask`, override YAML, and split | `workspace/core4d/results/E144/cem_ready/raw_mask_ref_fk_cem_ready.tsv`: 11 rows; `raw_mask_ref_fk_cem_ready_preflight.tsv`: 11/11 `preflight_ok=True`; split counts local-gpu0=4, remote-gpu0=4, remote-gpu1=3 | PASS |
| Full CEM completed or has explicit missing/failure log for every row | `workspace/core4d/results/E144/cem/full/`: 11/11 root NPZ, 11/11 `_full.mp4`, 11/11 outdir `trajectory_mjwp_act.npz`; eval `missing_rows=0` | PASS |
| S6 evidence distinguishes CEM pass/fail/not_run | `workspace/core4d/results/E144/E144_full_nonbox_raw_contact/s6_downstream/downstream_evidence_summary.json`: `DOWNSTREAM_CEM_FAIL=11`, `DOWNSTREAM_NOT_RUN=11`; no pass rows | PASS |
| Do not start PPO/Holosoma RL or generate checkpoints | `rl_status_counts={'not_run': 22}`; `rl_export_decision_counts={'SKIP_CEM_FAIL': 11, 'SKIP_NOT_HANDOFF_READY': 11}`; no `.pt/.pth/.ckpt` under `workspace/core4d/results/E144` | PASS |

## Outcome

E144 satisfies the plan scope through full CEM for all currently CEM-ready nonbox rows. The full nonbox inventory is accounted for, template gates prevent unsafe desk/chair/unreviewed template release, and full CEM/eval/S6 evidence are complete.

The experimental outcome is negative for downstream use: 11/11 CEM-ready rows evaluated as `cem_status=fail`, so `RL_EXPORT_READY=0`. RL was not run.
