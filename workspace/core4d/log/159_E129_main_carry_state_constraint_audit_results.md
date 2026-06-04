# E129 Main Carry-State Constraint Audit Results

Date: 2026-06-03

## Scope

E129 followed `workspace/core4d/plan/138_E129_main_carry_state_constraint_audit_plan.md`.
It did not launch CEM, smoke CEM, PPO, Holosoma RL, or checkpoint creation. It
audited existing main `box021_029_p2` evidence from E113 full and E119-E124 smoke.

Subagent `Dirac` performed a read-only route audit and agreed that the current
CEM/SBTO branch should not be repeated as another knob sweep. The subagent
recommended Holosoma reward-side inspection as the next branch, while preserving
the boundary that fragment rows are not main release evidence.

## Outputs

```text
workspace/core4d/results/E129/main_carry_state_constraint_audit/
  main_candidate_rows.tsv
  frame_constraint_summary.tsv
  e129_main_carry_state_summary.json
  e129_main_carry_state_summary.md
```

## Result

| metric | value |
|---|---:|
| main candidate rows | 17 |
| frame summary rows | 17 |
| strict gate rows | 0 |
| RL-ready rows | 0 |
| CEM launched | false |
| training launched | false |
| status | pass |

Best global/ranked main row:

| variant | decision | physics contact | lower-body | non-hand | pelvis |
|---|---|---:|---:|---:|---:|
| `E113_box021_029_p2_hold_band` | `LOWERBODY_SUPPORT_BLOCK` | 65.3% | 8.0% | 0.0%* | 0.646m |

`*` E113 lacks support-decomposition time series, so non-hand frame evidence is
not complete for that row. It is still blocked by lower-body contact and
`work_status=FAIL`.

Best full frame-level clean-carry row:

| variant | clean-carry frame fraction | longest clean run |
|---|---:|---:|
| `E121_box021_029_p2_terminal_hard_surface` | 37.3% | 10 frames |

Decision counts:

| decision | count |
|---|---:|
| `LOWERBODY_SUPPORT_BLOCK` | 1 |
| `CONTACT_BELOW_RL_GATE` | 16 |

Primary frame blockers:

| blocker | count |
|---|---:|
| `hand_contact_or_near_missing` | 10 |
| `missing_nonhand_support_timeseries` | 4 |
| `object_floor_not_clean` | 2 |
| `nonhand_not_clean` | 1 |

## Interpretation

E129 confirms that no existing main row is release/RL-ready. The best high-contact
main row is still E113 `hold_band`, but it is blocked by lower-body support. The
later attempts that reduce support shortcuts generally lose hand contact or object
carry quality. The best per-frame clean-carry evidence is only a short local run,
not a sustained trajectory or global pass.

The next aligned branch should not be full CEM or PPO from current rows. Two valid
next steps are:

1. Build a stage-local constrained teacher/smoke that preserves E113/E119 contact
   while hard-filtering lower-body, non-hand, object-floor, and pelvis violations
   inside the raw-contact window.
2. Run Holosoma reward-side inspection on the E126/E128 fragment exports to verify
   hand-support reward wiring before any PPO launch, keeping the fragments labeled
   `FRAGMENT_HOLDOUT_ONLY`.

## Verification

- `.venv/bin/python -m py_compile workspace/core4d/scripts/E129/audit_main_carry_state.py`
- `bash -n workspace/core4d/scripts/eval/eval_E129_main_carry_state_audit.sh`
- `bash workspace/core4d/scripts/eval/eval_E129_main_carry_state_audit.sh`
- `git diff --check` on E129 plan/script/eval entrypoint

