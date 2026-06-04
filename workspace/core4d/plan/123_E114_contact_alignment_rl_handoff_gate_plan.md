# E114 Contact Alignment RL Handoff Gate Plan

Date: 2026-06-03

## Context

E114 is the final step of `workspace/core4d/plan/contact_improvement_plan.md`: only cases that are strict, contact-aligned, low-penetration, and lower-body safe should enter RL handoff.

E113 completed 6-case full CEM but produced `0` release candidates. Therefore E114 must not export any RL-ready motion. Its role is to make the no-handoff decision reproducible and to split failed candidates into the next diagnostic queues.

## Inputs

- `workspace/core4d/results/E113/cem/full/pareto_decisions.tsv`
- `workspace/core4d/results/E113/cem/full/release_candidates.tsv`
- `workspace/core4d/results/E113/cem/full/full_method_metrics.csv`
- E113 videos/NPZ in `workspace/core4d/results/E113/cem/full/`

## Handoff Criteria

A row can enter E114 RL handoff only if all are true:

- `pareto_decision == release_candidate`
- `hold_strict_status == WORK`
- `contact_ok == True`
- `penetration_ok == True`
- `lowerbody_ok == True`
- `object_ok == True`
- `phase_scope == phaseA_release_candidate`

Rows that fail the gate are not silently dropped. They must be written to one of:

- `blocked_candidates.tsv`
- `diagnostic_queues.tsv`
- `no_handoff_manifest.tsv`

## Diagnostic Queue Rules

| queue | condition | meaning |
|---|---|---|
| `lowerbody_aware_contact` | contact improves but lower body fails | next CEM variant should include lower-body-aware contact objective/safety schedule |
| `strict_contact_margin` | strict WORK but contact does not improve enough | next variant should increase contact margin without relaxing strict physics |
| `lowerbody_repair` | lower body fails and contact is not enough | repair posture/lower-body first |
| `contact_target_repair` | contact decreases or remains below gate | inspect target/mask/approach corridor |

## Outputs

```text
workspace/core4d/results/E114/rl_handoff_gate/
  rl_export_list.tsv
  blocked_candidates.tsv
  diagnostic_queues.tsv
  no_handoff_manifest.tsv
  rl_handoff_summary.json
  rl_handoff_summary.md
```

Expected outcome for current E113: `rl_export_list.tsv` has 0 rows and `no_handoff_manifest.tsv` records all 6 rows with reasons.

## Success Criteria

- E114 script exits 0 and writes all outputs.
- Summary reports `rl_ready_rows=0`.
- The strongest contact-improving row (`box021_029_p2`) is not exported to RL and is routed to `lowerbody_aware_contact`.
- Strict WORK rows with insufficient contact improvement are routed to `strict_contact_margin`.
- Tracker/progress/log record that E114 is complete with no RL export.
