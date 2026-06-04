# E142 OmniRetarget Contact Exceedance Audit Plan

## Context

The corrected objective is not Holosoma/RL readiness. The immediate objective is to determine whether the existing contact-aware retarget/CEM outputs beat OmniRetarget on hand-object contact metrics under a paired, same-case evaluation.

E110 already provides paired OmniRetarget vs Spider CEM contact metrics for the E109 workset. E112/E113 provide contact-aware CEM variants (`raw_mask_ref_fk`, `hold_band`) for selected cases. E142 joins those artifacts into one explicit exceedance table.

## Claims

1. E110 OmniRetarget rows can be joined to E112/E113 contact-aware outputs by normalized source case id.
2. For each E112/E113 candidate, E142 can compute contact deltas vs OmniRetarget using `hand_object_contact_physics_frac` as the primary metric.
3. The audit can identify which cases already exceed OmniRetarget contact and which cases remain below OmniRetarget without launching CEM, RL, or Holosoma.

## Scope

- Read E110:
  - `workspace/core4d/results/E110/contact_metric_audit/contact_band_metrics.tsv`
- Read E112:
  - `workspace/core4d/results/E112/cem/full/full_eval_summary.csv`
- Read E113:
  - `workspace/core4d/results/E113/cem/full/full_method_metrics.csv`
- Write E142 audit outputs under:
  - `workspace/core4d/results/E142/omniretarget_contact_exceedance_audit/`

## Primary Metric

Primary pass:

`candidate hand_object_contact_physics_frac > OmniRetarget hand_object_physics_contact_frac`

Secondary diagnostics:

- candidate 8cm hand-object near/contact fraction;
- OmniRetarget EEF 8cm and hand 12cm near-band references;
- lower-body strict status;
- leg-box interference;
- object tracking mean error;
- pelvis minimum height.

## Implementation

1. Add `workspace/core4d/scripts/E142/audit_omniretarget_contact_exceedance.py`.
2. Normalize case ids using `source_task`/`e109_case_id`/`derived_task`.
3. Emit:
   - all candidate rows with OmniRetarget deltas;
   - best candidate per case by primary physics-contact metric;
   - summary JSON/MD.
4. Add fixed eval entry:
   `workspace/core4d/scripts/eval/eval_E142_omniretarget_contact_exceedance_audit.sh`.

## Success Criteria

- Every E112/E113 candidate either joins to an E110 OmniRetarget row or is explicitly marked `missing_omni_pair`.
- The audit reports per-case pass/fail for contact exceedance vs OmniRetarget.
- No CEM, RL, remote job, or Holosoma command is launched.

## Fixed Command

```bash
bash workspace/core4d/scripts/eval/eval_E142_omniretarget_contact_exceedance_audit.sh
```

## Expected Outputs

- `workspace/core4d/results/E142/omniretarget_contact_exceedance_audit/e142_contact_exceedance_rows.tsv`
- `workspace/core4d/results/E142/omniretarget_contact_exceedance_audit/e142_best_by_case.tsv`
- `workspace/core4d/results/E142/omniretarget_contact_exceedance_audit/e142_contact_exceedance_summary.json`
- `workspace/core4d/results/E142/omniretarget_contact_exceedance_audit/e142_contact_exceedance_summary.md`
