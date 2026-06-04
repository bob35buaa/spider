# E115 Lower-Body-Aware Contact Diagnostic Results

Date: 2026-06-03

## Goal

E115 follows E114's no-RL handoff. The main question is whether the best E113 blocked case, `box021_029_p2`, can keep its contact gain while repairing lower-body/object interference.

The experiment tests a minimal CEM-side lower-body-aware contact diagnostic:

- main case: `box021_029_p2`
- companion lower-body repair case: `box021_035_p2`
- strict/contact-margin guards: `box021_035_p1`, `box004_082_p1`, `box004_083_p2`
- variants: `leg_penalty_s2`, `leg_penalty_s4`, `leg_penalty_s2_contact_gain8`

## Implementation

New artifacts:

- Plan: `workspace/core4d/plan/124_E115_lowerbody_aware_contact_diagnostic_plan.md`
- Manifest/preflight builder: `workspace/core4d/scripts/E115/build_lowerbody_contact_manifest.py`
- Manifest: `workspace/core4d/scripts/E115/variants.tsv`
- Preflight: `workspace/core4d/results/E115/preflight/phaseA_preflight.tsv`
- Train script: `workspace/core4d/scripts/train/train_E115_lowerbody_contact.sh`
- Remote runner: `workspace/core4d/scripts/run_E115_remote.sh`
- Pull script: `workspace/core4d/scripts/pull_E115_remote_results.sh`
- Evaluator: `workspace/core4d/scripts/eval/eval_E115_lowerbody_contact.py`
- Eval entrypoint: `workspace/core4d/scripts/eval/eval_E115_lowerbody_contact.sh`
- Overrides: `examples/config/override/core4d_E115_*.yaml`

Preflight generated 15 variants: 5 cases x 3 variants. All rows passed task, mask, baseline, and E113 hold-result checks. Split assignment was local 6, remote GPU0 3, remote GPU1 6.

Static checks passed:

- `python3 -m py_compile workspace/core4d/scripts/E115/build_lowerbody_contact_manifest.py workspace/core4d/scripts/eval/eval_E115_lowerbody_contact.py`
- `bash -n` for E115 train/remote/pull/eval shell scripts
- split-list checks for local, remote-gpu0, and remote-gpu1
- `bash workspace/core4d/scripts/eval/eval_E115_lowerbody_contact.sh smoke --allow-missing`

## Execution

Smoke CEM used `SMOKE_MAX_NUM_ITERATIONS=4`.

Execution layout:

| split | machine | rows |
|---|---|---:|
| `local-gpu0` | local RTX 5090 | 6 |
| `remote-gpu0` | `spider-remote` RTX 6000 Ada GPU0 | 3 |
| `remote-gpu1` | `spider-remote` RTX 6000 Ada GPU1 | 6 |

The remote sync initially placed one script copy under `workspace/core4d/scripts/` instead of the canonical `scripts/train/` and `scripts/eval/` locations. This was corrected before smoke. No unrelated GPU processes were stopped.

Smoke outputs:

| artifact | count |
|---|---:|
| root NPZ | 15 |
| MP4 | 15 |
| outdir `trajectory_mjwp_act.npz` | 15 |

Remote results were pulled back with `workspace/core4d/scripts/pull_E115_remote_results.sh smoke`.

## Quantitative Result

`bash workspace/core4d/scripts/eval/eval_E115_lowerbody_contact.sh smoke` completed successfully.

Summary:

- evaluated variants: 15
- method rows: 45
- missing variants: 0
- release candidates: 0

Decision counts:

| decision | count |
|---|---:|
| `lowerbody_fixed_contact_fail` | 12 |
| `penetration_fail` | 1 |
| `review` | 2 |

Main case:

| case | variant | physics contact E113 -> test | lower-body interference E113 -> test | deep penetration delta | strict | decision |
|---|---|---:|---:|---:|---|---|
| `box021_029_p2` | `leg_penalty_s2` | 65.3% -> 12.0% | 8.0% -> 0.0% | +0.0pp | FAIL | `lowerbody_fixed_contact_fail` |
| `box021_029_p2` | `leg_penalty_s4` | 65.3% -> 12.0% | 8.0% -> 0.0% | +0.0pp | FAIL | `lowerbody_fixed_contact_fail` |
| `box021_029_p2` | `leg_penalty_s2_contact_gain8` | 65.3% -> 52.0% | 8.0% -> 29.3% | +13.3pp | FAIL | `penetration_fail` |

Companion and guard observations:

| case | best observed signal |
|---|---|
| `box021_035_p2` | `leg_penalty_s4` reduces lower-body interference 9.8% -> 4.5% while keeping physics contact 73.7% -> 70.7%, but strict remains FAIL and contact is still below E113. |
| `box021_035_p1` | all variants remove lower-body interference, but physics contact drops from 72.9% to 59.7-63.6%; strict remains FAIL. |
| `box004_082_p1` | lower-body can be removed, but deep penetration increases for s2/s4 or pelvis/contact worsens for contact_gain8. |
| `box004_083_p2` | lower-body is removed, but physics contact collapses from 52.4% to 28.6-32.4%; contact_gain8 also fails WORK. |

## Visual Check

Keyframes were extracted with `video-frames` / ffmpeg:

- `workspace/core4d/results/E115/cem/smoke/keyframe_review/box021_029_p2_s2_t2.jpg`
- `workspace/core4d/results/E115/cem/smoke/keyframe_review/box021_029_p2_gain8_t2.jpg`
- `workspace/core4d/results/E115/cem/smoke/keyframe_review/box021_035_p2_s4_t2.jpg`

Visual observations:

- `box021_029_p2 / leg_penalty_s2`: lower-body contact is avoided, but the robot and object separate and the pose is far from the reference carry; this matches the 65.3% -> 12.0% physics-contact collapse.
- `box021_029_p2 / leg_penalty_s2_contact_gain8`: the object is pulled back toward the body, but lower-body/object interference and penetration reappear; this matches 29.3% lower-body interference and +13.3pp deep penetration.
- `box021_035_p2 / leg_penalty_s4`: the object remains near the hands, but lower-body proximity is still visually suspicious; metrics mark lower-body improved but not fully strict/release-safe.

## Interpretation

E115 is a negative but useful diagnostic. A naive lower-body penalty can remove leg/object interference, but in the main case it does so by letting the hands abandon the object. Raising contact gain restores some contact, but then lower-body interference and deep penetration return.

This means the current bottleneck is not simply "add a larger leg/object penalty." The penalty competes with the same optimizer degrees of freedom used to maintain hand-object contact. The result supports a more structured next step:

- lower-body avoidance must be phase-gated or state-gated, not a uniform penalty over the whole contact window;
- the contact objective should be tied to raw-contact hand/object surface targets and release timing, not only aggregate near/contact gain;
- for `box021_029_p2`, use a corridor/pose constraint that prevents crouch-under-object solutions while preserving hand-side contact;
- keep `box021_035_p1`, `box004_082_p1`, and `box004_083_p2` as guards because naive penalties regress them.

## Decision

Do not launch E115 full CEM for the same 15 variants. The smoke did validate the infrastructure, but the method signal is negative: 0 release candidates, 0 guard passes, and the main case shows a clear contact/lower-body/penetration trade-off under the tested formulation.

Next experiment should change the method, not only increase iteration count:

1. Add a phase-gated lower-body penalty active after hand-contact acquisition or only when hand contact is already above a threshold.
2. Add a carry-corridor / pelvis-upright constraint for `box021_029_p2` to prevent crouch-under-object compensation.
3. Compare against a contact-side target refinement rather than only stronger penalties: surface/corridor target for main hands, with strict guard rows preserved.
