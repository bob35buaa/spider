# E117 Phase/State-Gated Lower-Body Contact Results

Date: 2026-06-03

## Goal

E117 follows the E115/E116 negative diagnostics. The question was whether lower-body/object avoidance can be enabled only during meaningful contact phases, instead of applying a uniform leg/object penalty that competes with hand-object contact.

The tested method adds a gated `leg_object_penalty` and evaluates 5 cases x 3 variants:

- main case: `box021_029_p2`
- companion lower-body repair case: `box021_035_p2`
- strict/contact-margin guards: `box021_035_p1`, `box004_082_p1`, `box004_083_p2`
- variants: `mask_gate_s2`, `mask_time_gate_s2`, `handtarget_gate_s2`

## Implementation

Code changes:

- `spider/config.py`
  - added `leg_object_penalty_gate_source`
  - added `leg_object_penalty_start_eval_time`
  - added `leg_object_penalty_end_eval_time`
  - added `leg_object_penalty_hand_target_threshold_m`
- `spider/simulators/mjwp.py`
  - added phase/state gating for `leg_object_penalty`
  - supported `always`, `contact_mask`, `time_window`, `contact_mask_time_window`, `hand_target`, and `contact_mask_and_hand_target`
  - emitted `leg_object_penalty_gate` in reward info

Defaults preserve the previous E115 behavior: `leg_object_penalty_gate_source=always`.

New artifacts:

- Plan: `workspace/core4d/plan/126_E117_phase_gated_lowerbody_contact_plan.md`
- Manifest/preflight builder: `workspace/core4d/scripts/E117/build_phase_gated_lowerbody_manifest.py`
- Manifest: `workspace/core4d/scripts/E117/variants.tsv`
- Preflight: `workspace/core4d/results/E117/preflight/phaseA_preflight.tsv`
- Train script: `workspace/core4d/scripts/train/train_E117_phase_gated_lowerbody.sh`
- Remote runner: `workspace/core4d/scripts/run_E117_remote.sh`
- Pull script: `workspace/core4d/scripts/pull_E117_remote_results.sh`
- Evaluator: `workspace/core4d/scripts/eval/eval_E117_phase_gated_lowerbody.py`
- Eval entrypoint: `workspace/core4d/scripts/eval/eval_E117_phase_gated_lowerbody.sh`
- Overrides: `examples/config/override/core4d_E117_*.yaml`

Preflight generated 15 variants. All required task, mask, baseline, and E113 hold-result checks passed. Split assignment was local 6, remote GPU0 3, remote GPU1 6.

Static checks passed:

```bash
python -m py_compile spider/config.py spider/simulators/mjwp.py workspace/core4d/scripts/E117/build_phase_gated_lowerbody_manifest.py workspace/core4d/scripts/eval/eval_E117_phase_gated_lowerbody.py
bash -n workspace/core4d/scripts/train/train_E117_phase_gated_lowerbody.sh workspace/core4d/scripts/run_E117_remote.sh workspace/core4d/scripts/pull_E117_remote_results.sh workspace/core4d/scripts/eval/eval_E117_phase_gated_lowerbody.sh
git diff --check -- spider/config.py spider/simulators/mjwp.py workspace/core4d/scripts/E117 workspace/core4d/scripts/train/train_E117_phase_gated_lowerbody.sh workspace/core4d/scripts/run_E117_remote.sh workspace/core4d/scripts/pull_E117_remote_results.sh workspace/core4d/scripts/eval/eval_E117_phase_gated_lowerbody.py workspace/core4d/scripts/eval/eval_E117_phase_gated_lowerbody.sh
```

## Execution

Smoke CEM used `SMOKE_MAX_NUM_ITERATIONS=4`.

Execution layout:

| split | machine | rows |
|---|---|---:|
| `local-gpu0` | local GPU0 | 6 |
| `remote-gpu0` | `spider-remote` GPU0 | 3 |
| `remote-gpu1` | `spider-remote` GPU1 | 6 |

Remote execution used `tmux` session `E117_smoke_072858`. No unrelated GPU processes were stopped. GPU0 completed the main `box021_029_p2` rows; GPU1 completed the companion/guard rows. Results were pulled back with:

```bash
bash workspace/core4d/scripts/pull_E117_remote_results.sh smoke
```

Smoke outputs after local + remote pull:

| artifact | count |
|---|---:|
| root NPZ | 15 |
| MP4 | 15 |
| outdir `trajectory_mjwp_act.npz` | 15 |
| missing variants | 0 |

No local or remote E117 training processes remained after completion.

## Quantitative Result

Evaluation command:

```bash
bash workspace/core4d/scripts/eval/eval_E117_phase_gated_lowerbody.sh smoke
```

Evaluator output:

- `workspace/core4d/results/E117/cem/smoke/smoke_eval_summary.md`
- `workspace/core4d/results/E117/cem/smoke/pareto_decisions.tsv`
- `workspace/core4d/results/E117/cem/smoke/release_candidates.tsv`
- `workspace/core4d/results/E117/cem/smoke/guard_failures.tsv`

Summary:

- evaluated variants: 15
- method rows: 45
- missing variants: 0
- release candidates: 0

Decision counts:

| decision | count |
|---|---:|
| `lowerbody_fixed_contact_fail` | 10 |
| `penetration_fail` | 3 |
| `contact_good_lowerbody_fail` | 1 |
| `review` | 1 |

Main case:

| case | variant | physics contact E107 -> test | lower-body E113 -> test | deep penetration delta | pelvis ok | decision |
|---|---|---:|---:|---:|---|---|
| `box021_029_p2` | `mask_gate_s2` | 45.3% -> 8.0% | 8.0% -> 0.0% | +0.0pp | no | `lowerbody_fixed_contact_fail` |
| `box021_029_p2` | `mask_time_gate_s2` | 45.3% -> 9.3% | 8.0% -> 0.0% | +0.0pp | no | `lowerbody_fixed_contact_fail` |
| `box021_029_p2` | `handtarget_gate_s2` | 45.3% -> 12.0% | 8.0% -> 0.0% | +0.0pp | no | `lowerbody_fixed_contact_fail` |

Companion and guard observations:

| case | best observed signal | blocker |
|---|---|---|
| `box021_035_p2` | `mask_gate_s2` gives 75.9% physics contact and lowers leg interference 9.8% -> 3.8% | deep penetration +10.5pp |
| `box021_035_p2` | `handtarget_gate_s2` keeps deep penetration small (+1.5pp) | lower-body remains 11.3%, strict FAIL |
| `box021_035_p1` | `mask_gate_s2`/`mask_time_gate_s2` reach 79-80% physics contact | deep penetration +5.4pp / +10.9pp |
| `box004_082_p1` | `handtarget_gate_s2` improves physics contact 37.6% -> 44.0% and keeps lower-body 0.0% | pelvis regression; strict FAIL |
| `box004_083_p2` | `handtarget_gate_s2` retains some contact at 34.3% | deep penetration +5.7pp; strict FAIL |

## Visual Check

Representative keyframes:

- `workspace/core4d/results/E117/cem/smoke/keyframes/E117_box021_029_p2_handtarget_gate_s2/f70.jpg`
- `workspace/core4d/results/E117/cem/smoke/keyframes/E117_box021_035_p2_mask_gate_s2/f70.jpg`
- `workspace/core4d/results/E117/cem/smoke/keyframes/E117_box004_082_p1_handtarget_gate_s2/f70.jpg`

Visual observations:

- `box021_029_p2 / handtarget_gate_s2`: lower-body support is removed, but the hands abandon the box and the robot collapses into a seated/squatting pose. This matches physics contact 12.0% and `pelvis_ok=False`.
- `box021_035_p2 / mask_gate_s2`: contact is high, but the lower leg/foot is still visually involved near the box, and metrics show a deep-penetration regression.
- `box004_082_p1 / handtarget_gate_s2`: contact improves relative to the other Box004 E117 rows, but posture drops substantially; this matches the large pelvis regression.

## Decision

Do not launch E117 full CEM.

Reason:

- Smoke has 0 release candidates.
- The main case has no Pareto-improving row: all three variants fix lower-body interference but collapse physics contact from the E113 level and regress pelvis stability.
- Guard cases show the same pattern as E115/E116 in a different form: rows that preserve or raise contact often introduce penetration, lower-body support, or posture failure.

## Interpretation

E117 proves that binary phase/state gating alone is still too brittle. It gates the lower-body penalty later or more selectively, but once active it still lets CEM solve the conflict by moving the body/object state rather than maintaining a semantic hand-support configuration.

The next method should change the objective structure, not increase E117 iteration count:

1. Replace the hard gate with a soft state-dependent penalty or curriculum tied to achieved hand contact and object pose.
2. Add an explicit carry/object-support corridor that couples hand contact, object pose, pelvis/torso posture, and lower-body clearance.
3. Add anti-tip/object-orientation and pose-prior terms so contact cannot be recovered through object tilt, crouch, or seated support.
4. Keep E111/S6 raw-contact and object-local target evidence as the evaluation contract for future variants.
