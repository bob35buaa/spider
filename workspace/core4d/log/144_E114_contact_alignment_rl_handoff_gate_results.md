# E114 Contact Alignment RL Handoff Gate Results

Date: 2026-06-03

## Goal

E114 is the RL handoff gate for the contact-improvement sequence. A case should enter RL only if it is strict WORK, contact-aligned, low-penetration, lower-body safe, object-safe, and marked as an E113 release candidate.

Because E113 produced zero release candidates, E114 intentionally does not launch RL. The goal is to make the no-handoff decision reproducible and route each blocked case to the next diagnostic queue.

## Implementation

New artifacts:

- Plan: `workspace/core4d/plan/123_E114_contact_alignment_rl_handoff_gate_plan.md`
- Gate script: `workspace/core4d/scripts/E114/build_rl_handoff_gate.py`
- Fixed entrypoint: `workspace/core4d/scripts/eval/eval_E114_rl_handoff_gate.sh`

Input:

- `workspace/core4d/results/E113/cem/full/pareto_decisions.tsv`

Outputs:

- `workspace/core4d/results/E114/rl_handoff_gate/rl_export_list.tsv`
- `workspace/core4d/results/E114/rl_handoff_gate/blocked_candidates.tsv`
- `workspace/core4d/results/E114/rl_handoff_gate/diagnostic_queues.tsv`
- `workspace/core4d/results/E114/rl_handoff_gate/no_handoff_manifest.tsv`
- `workspace/core4d/results/E114/rl_handoff_gate/rl_handoff_summary.json`
- `workspace/core4d/results/E114/rl_handoff_gate/rl_handoff_summary.md`

## Result

`bash workspace/core4d/scripts/eval/eval_E114_rl_handoff_gate.sh` completed successfully.

Summary:

- evaluated rows: 6
- RL-ready rows: 0
- blocked rows: 6
- strict WORK rows: 3
- strict FAIL rows: 3

Diagnostic queue counts:

| queue | count | interpretation |
|---|---:|---|
| `strict_contact_margin` | 3 | strict WORK, but contact gain is below E114 handoff threshold |
| `lowerbody_repair` | 2 | lower-body strict gate failed before RL handoff |
| `lowerbody_aware_contact` | 1 | contact improved, but lower-body strict gate failed |

Blocked rows:

| case | strict | contact delta | physics delta | leg hold | queue |
|---|---|---:|---:|---:|---|
| `box021_035_p1` | WORK | +0.0% | +2.3% | 0.8% | `strict_contact_margin` |
| `box021_035_p2` | FAIL | +4.5% | +6.0% | 9.8% | `lowerbody_repair` |
| `box021_029_p2` | FAIL | +4.0% | +20.0% | 8.0% | `lowerbody_aware_contact` |
| `box004_082_p1` | WORK | +0.0% | -5.5% | 0.9% | `strict_contact_margin` |
| `box004_083_p1` | FAIL | +0.0% | +2.9% | 10.8% | `lowerbody_repair` |
| `box004_083_p2` | WORK | -4.8% | +6.7% | 2.9% | `strict_contact_margin` |

## Interpretation

No E113 row should be exported to RL. The result is not an RL pipeline failure; it is the correct outcome of the stricter contact-alignment gate.

The best next diagnostic is `box021_029_p2`: it gains +20.0pp physics contact without increasing deep penetration, but lower-body interference is 8.0%. This is the clearest evidence that contact and penetration can be improved together, while the next bottleneck is lower-body/object interference.

The strict WORK rows (`box004_082_p1`, `box004_083_p2`, `box021_035_p1`) are stable enough to serve as guard cases, but they should not enter RL because contact did not improve enough over the cached baseline.

## Next Direction

Start the next contact-improvement branch from diagnostics rather than RL:

1. `lowerbody_aware_contact`: use `box021_029_p2` to add lower-body-aware contact objective or stricter lower-body safety scheduling.
2. `strict_contact_margin`: use strict WORK rows as guards while increasing contact margin.
3. Box026 remains a separate diagnostic branch for surface target / approach corridor / posture schedule.
