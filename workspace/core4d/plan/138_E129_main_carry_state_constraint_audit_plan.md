# E129 Main Carry-State Constraint Audit Plan

Date: 2026-06-03

## Context

E113 proved that `hold_band` can recover main `box021_029_p2` hand-object
physics contact, but it still failed the strict release gate because lower-body
support remained present. E119-E124 then tested posture/bodyguard, support
decomposition, terminal carry gates, snap warmstarts, two-stage curriculum, and
SBTO. None produced an RL-ready main row.

E125-E128 moved fragment holdouts through Holosoma adapter/static/runtime startup
preflights, but those rows remain `FRAGMENT_HOLDOUT_ONLY`. They do not solve the
main `box021_029_p2` gate.

E129 is therefore a no-GPU audit before the next optimizer change. It analyzes the
main-case per-frame evidence already produced by E113 and E119-E124 to answer:
has any trajectory produced a sustained window that simultaneously has hand
contact/near-contact, no lower-body contact, no non-hand support, no object-floor
support, low penetration, and acceptable global metrics?

## Claims

| claim | success evidence |
|---|---|
| C1: recent main-case rows are exhaustively summarized | E129 table covers `box021_029_p2` rows from E113 full and E119-E124 smoke |
| C2: clean carry feasibility is measured per frame where evidence exists | E129 computes frame-level clean-carry fractions and longest runs from leg/object and support-decomposition time series |
| C3: next experiment choice is evidence-driven | E129 summary identifies whether the blocker is contact loss, lower-body support, non-hand support, object-floor support, or global posture/object error |
| C4: no training is launched | summary reports `training_launched=false`, `cem_launched=false`, `rl_ready_rows=0` |

## Inputs

- `workspace/core4d/results/E113/cem/full/full_method_metrics.csv`
- `workspace/core4d/results/E119/cem/smoke/smoke_method_metrics.csv`
- `workspace/core4d/results/E120/cem/smoke/smoke_method_metrics.csv`
- `workspace/core4d/results/E121/cem/smoke/smoke_method_metrics.csv`
- `workspace/core4d/results/E122/cem/smoke/smoke_method_metrics.csv`
- `workspace/core4d/results/E123/cem/smoke/smoke_method_metrics.csv`
- `workspace/core4d/results/E124/cem/smoke/smoke_method_metrics.csv`
- referenced `legobj_timeseries_csv` and `support_decomp_timeseries_csv`

## Outputs

```text
workspace/core4d/results/E129/main_carry_state_constraint_audit/
  main_candidate_rows.tsv
  frame_constraint_summary.tsv
  e129_main_carry_state_summary.json
  e129_main_carry_state_summary.md
```

## Success Criteria

- Output includes every `box021_029_p2` test row from E113/E119-E124 where metrics
  are present.
- For rows with time-series evidence, output frame-level:
  - hand contact / hand near-zero fraction
  - lower-body-clean fraction
  - non-hand-clean fraction
  - object-floor-clean fraction
  - clean-carry frame fraction
  - longest clean-carry run
- Summary preserves the hard boundary:
  `cem_launched=false`, `training_launched=false`, `rl_ready_rows=0`.
- Summary recommends the next route without declaring fragment rows or audited
  main rows release/RL-ready.

## Command

```bash
bash workspace/core4d/scripts/eval/eval_E129_main_carry_state_audit.sh
```

## Non-Goals

- Do not run full CEM or smoke CEM.
- Do not launch Holosoma RL.
- Do not edit CEM reward code in E129.
- Do not claim that `FRAGMENT_HOLDOUT_ONLY` rows solve the main release gate.

