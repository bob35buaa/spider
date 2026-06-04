# E113 Contact-Aware Expanded Workset Results

Date: 2026-06-03

## Goal

E113 follows `workspace/core4d/plan/contact_improvement_plan.md` and E112. The goal was to test whether the E112 `hold_band` contact-aware CEM recipe scales from the 3-case ablation to a larger non-box026 workset, without mixing in Box026 diagnostic failures.

Phase A used 6 runnable cases:

- Box004 release-candidate cases: `box004_082_p1`, `box004_083_p1`, `box004_083_p2`
- Box021 cases: `box021_035_p1` as release candidate, `box021_035_p2` and `box021_029_p2` as lower-body-risk diagnostics

Box026 remains a separate diagnostic holdout for surface target / approach corridor / posture schedule work.

## Implementation

New E113 artifacts:

- Plan: `workspace/core4d/plan/122_E113_contact_aware_expanded_workset_plan.md`
- Manifest builder: `workspace/core4d/scripts/E113/build_expanded_contact_manifest.py`
- Variants: `workspace/core4d/scripts/E113/variants.tsv`
- Preflight: `workspace/core4d/results/E113/preflight/phaseA_preflight.tsv`
- Overrides: `examples/config/override/core4d_E113_*_hold_band.yaml`
- CEM runner: `workspace/core4d/scripts/train/train_E113_contact_aware_expanded.sh`
- Remote runner/pull: `workspace/core4d/scripts/run_E113_remote.sh`, `workspace/core4d/scripts/pull_E113_remote_results.sh`
- Evaluator: `workspace/core4d/scripts/eval/eval_E113_contact_aware_expanded.py`, `.sh`

The manifest builder is policy-driven from E109's 24-case comparison TSV, then attaches existing target/baseline/mask assets. It no longer copies E112 hard-coded cases. `box004_083_p2` uses an E104 raw-contact proxy converted to a MJWP-compatible mask at:

`workspace/core4d/results/E113/contact_masks/box004_083_p2/raw_contact_mask_3cm.npz`

Evaluator note: E105 lower-body replay exposes `hand_box_sdf_min_m` and `hand_object_contact_physics_frac`, but not a deep hand-penetration fraction. E113 therefore computes `hand_geom_deep_penetration_2cm` from the hand-box SDF timeseries and aliases `hand_object_physics_contact` to E105's physics-contact field, keeping the decision table schema stable.

## Execution

Preflight and static checks:

- `python3 -m py_compile workspace/core4d/scripts/E113/build_expanded_contact_manifest.py workspace/core4d/scripts/eval/eval_E113_contact_aware_expanded.py`
- `bash -n` for train/remote/pull/eval scripts
- `git diff --check`
- split-list checks for local / remote-gpu0 / remote-gpu1

Smoke:

- Local GPU0: 3 Box004 variants
- Remote `spider-remote` GPU0/GPU1: 3 Box021 variants
- Result: 6/6 NPZ + MP4 complete
- Eval: `workspace/core4d/results/E113/cem/smoke/smoke_eval_summary.md`

Full CEM:

- Initial split: local GPU0 = 3 Box004, remote GPU0 = 2 Box021, remote GPU1 = 1 Box021
- To avoid leaving remote GPU1 idle, `E113_box004_083_p2_hold_band` was additionally run as a remote single job on GPU1. After local `box004_083_p1` completed, the local E113 queue was stopped before duplicating `box004_083_p2`.
- Result: 6/6 NPZ + MP4 + `trajectory_mjwp_act.npz` complete
- Full eval: `workspace/core4d/results/E113/cem/full/full_eval_summary.md`

## Full Results

| case | phase | contact base -> hold | physics delta | deep pen delta | pelvis delta | leg hold | strict | decision |
|---|---|---:|---:|---:|---:|---:|---|---|
| `box004_082_p1` | release | 54.1% -> 54.1% | -5.5% | +0.9% | +0.000m | 0.9% | WORK | contact_not_improved |
| `box004_083_p1` | release | 55.9% -> 55.9% | +2.9% | +0.0% | -0.018m | 10.8% | FAIL | contact_not_improved |
| `box004_083_p2` | release | 64.8% -> 60.0% | +6.7% | +0.0% | +0.009m | 2.9% | WORK | contact_not_improved |
| `box021_035_p1` | release | 77.5% -> 77.5% | +2.3% | +0.0% | +0.002m | 0.8% | WORK | contact_not_improved |
| `box021_035_p2` | lower-body risk | 75.9% -> 80.5% | +6.0% | +0.8% | -0.002m | 9.8% | FAIL | contact_not_improved |
| `box021_029_p2` | lower-body risk | 69.3% -> 73.3% | +20.0% | +0.0% | -0.017m | 8.0% | FAIL | contact_good_lowerbody_fail |

Decision-table outputs:

- `workspace/core4d/results/E113/cem/full/pareto_decisions.tsv`
- `workspace/core4d/results/E113/cem/full/release_candidates.tsv`
- `workspace/core4d/results/E113/cem/full/contact_good_lowerbody_fail.tsv`
- `workspace/core4d/results/E113/cem/full/full_method_metrics.csv`

## Interpretation

No E113 Phase A release candidate passed the current Pareto gate. The full run does not reproduce E112's large contact recovery on the expanded workset:

- 3 rows are strict WORK: `box004_082_p1`, `box004_083_p2`, `box021_035_p1`.
- None of the strict WORK rows reaches the configured contact improvement threshold (`contact_frac_either` or physics contact delta >= 8pp).
- The strongest contact gain is `box021_029_p2`: physics contact +20.0pp and no deep penetration increase, but lower-body interference is 8.0%, so it is classified as `contact_good_lowerbody_fail`.
- `box021_035_p2` has contact gains (+4.5pp geom / +6.0pp physics) but also lower-body interference 9.8%, below the E113 contact threshold and above lower-body strict threshold.

This suggests E112's contact-aware reward is useful but not sufficient as a direct release recipe on a broader set. The current hold-band formulation can preserve or modestly increase contact without deep penetration, but it does not reliably create additional contact margin, and lower-body interference remains the main blocker for some Box021 cases.

## Next Direction

E114 should not hand off E113 Phase A as RL-ready positives. Recommended next steps:

1. Add a lower-body-aware contact objective or safety schedule for Box021 risk cases, using `box021_029_p2` as the main diagnostic because it has contact improvement but lower-body failure.
2. Sweep contact acceptance thresholds around physics-contact gain 4-8pp and strict WORK, but keep the current release gate unchanged until visual review supports relaxing it.
3. Compare E112 anchor cases against E113 expanded cases for why `hold_band` recovers contact strongly on E112 but mostly preserves contact on E113.
4. Keep Box026 in a separate E114 diagnostic branch for surface target / approach corridor / posture schedule rather than mixing it into the E113 release set.
