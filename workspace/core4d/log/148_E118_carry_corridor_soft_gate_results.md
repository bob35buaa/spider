# E118 Carry Corridor Soft-Gate Results

Date: 2026-06-03

## Goal

E118 follows the E115-E117 negative diagnostics. E115 showed that a uniform lower-body penalty trades hand-object contact for lower-body avoidance. E117 showed that binary phase/state gating still collapses contact or posture. E118 therefore tests a softer objective structure: a coherent carry-state reward that couples hand target proximity, object clearance/orientation, pelvis height, and leg clearance.

The workset stays fixed at 5 cases x 3 variants:

- main case: `box021_029_p2`
- companion lower-body repair case: `box021_035_p2`
- strict/contact-margin guards: `box021_035_p1`, `box004_082_p1`, `box004_083_p2`
- variants: `corridor_ref_soft`, `corridor_ref_leglight`, `corridor_surface_soft`

## Implementation

Code changes:

- `spider/config.py`
  - added `carry_corridor_*` reward/config fields
  - added processing for hand body/object geometry dependencies
  - resolved `carry_corridor_leg_geom_names` into geom ids
- `spider/simulators/mjwp.py`
  - added `carry_corridor_rew`
  - supported `contact_mask`, `time_window`, and `contact_mask_time_window` gates
  - combined hand target, object clearance band, pelvis min-height, object rotation, and leg clearance into the reward
  - emitted carry-corridor component scores in reward info

New artifacts:

- Plan: `workspace/core4d/plan/127_E118_carry_corridor_soft_gate_plan.md`
- Manifest/preflight builder: `workspace/core4d/scripts/E118/build_carry_corridor_manifest.py`
- Manifest: `workspace/core4d/scripts/E118/variants.tsv`
- Preflight: `workspace/core4d/results/E118/preflight/phaseA_preflight.tsv`
- Train script: `workspace/core4d/scripts/train/train_E118_carry_corridor.sh`
- Remote runner: `workspace/core4d/scripts/run_E118_remote.sh`
- Pull script: `workspace/core4d/scripts/pull_E118_remote_results.sh`
- Evaluator: `workspace/core4d/scripts/eval/eval_E118_carry_corridor.py`
- Eval entrypoint: `workspace/core4d/scripts/eval/eval_E118_carry_corridor.sh`
- Overrides: `examples/config/override/core4d_E118_*.yaml`

Preflight generated 15 variants. All required task, mask, baseline, and E113 hold-result checks passed. Split assignment was local 6, remote GPU0 3, remote GPU1 6.

Static checks passed:

```bash
python -m py_compile spider/config.py spider/simulators/mjwp.py workspace/core4d/scripts/E118/build_carry_corridor_manifest.py workspace/core4d/scripts/eval/eval_E118_carry_corridor.py
bash -n workspace/core4d/scripts/train/train_E118_carry_corridor.sh workspace/core4d/scripts/run_E118_remote.sh workspace/core4d/scripts/pull_E118_remote_results.sh workspace/core4d/scripts/eval/eval_E118_carry_corridor.sh
git diff --check -- spider/config.py spider/simulators/mjwp.py workspace/core4d/scripts/E118 workspace/core4d/scripts/train/train_E118_carry_corridor.sh workspace/core4d/scripts/run_E118_remote.sh workspace/core4d/scripts/pull_E118_remote_results.sh workspace/core4d/scripts/eval/eval_E118_carry_corridor.py workspace/core4d/scripts/eval/eval_E118_carry_corridor.sh
```

## Execution

Smoke CEM used `SMOKE_MAX_NUM_ITERATIONS=4`.

Execution layout:

| split | machine | rows |
|---|---|---:|
| `local-gpu0` | local GPU0 | 6 |
| `remote-gpu0` | `spider-remote` GPU0 | 3 |
| `remote-gpu1` | `spider-remote` GPU1 | 6 |

Remote execution used `tmux` session `E118_smoke_082608`. GPU0 completed the main `box021_029_p2` rows. GPU1 was slow on the companion/guard rows, so two already-queued `box021_035_p1` variants were safely backfilled on idle remote GPU0 via `single` mode after confirming GPU0 free memory and avoiding duplicate concurrent writes:

- `E118_box021_035_p1_corridor_ref_leglight`
- `E118_box021_035_p1_corridor_surface_soft`

The original GPU1 queue then skipped those completed variants. No unrelated GPU processes were stopped.

Results were pulled back with:

```bash
bash workspace/core4d/scripts/pull_E118_remote_results.sh smoke
```

Smoke outputs after local + remote pull:

| artifact | count |
|---|---:|
| root NPZ | 15 |
| MP4 | 15 |
| outdir `trajectory_mjwp_act.npz` | 15 |
| missing variants | 0 |

Remote `tmux` and E118 training processes were not present after completion.

## Quantitative Result

Evaluation command:

```bash
bash workspace/core4d/scripts/eval/eval_E118_carry_corridor.sh smoke
```

Evaluator output:

- `workspace/core4d/results/E118/cem/smoke/smoke_eval_summary.md`
- `workspace/core4d/results/E118/cem/smoke/pareto_decisions.tsv`
- `workspace/core4d/results/E118/cem/smoke/release_candidates.tsv`
- `workspace/core4d/results/E118/cem/smoke/guard_failures.tsv`

Summary:

- evaluated variants: 15
- method rows: 45
- missing variants: 0
- release candidates: 0

Decision counts:

| decision | count |
|---|---:|
| `lowerbody_fixed_contact_fail` | 8 |
| `penetration_fail` | 3 |
| `contact_good_lowerbody_fail` | 2 |
| `review` | 2 |

Main case:

| case | variant | physics contact E107 -> test | lower-body E113 -> test | deep penetration delta | decision |
|---|---|---:|---:|---:|---|
| `box021_029_p2` | `corridor_ref_soft` | 45.3% -> 50.7% | 8.0% -> 25.3% | +13.3pp | `penetration_fail` |
| `box021_029_p2` | `corridor_ref_leglight` | 45.3% -> 56.0% | 8.0% -> 33.3% | +10.7pp | `contact_good_lowerbody_fail` |
| `box021_029_p2` | `corridor_surface_soft` | 45.3% -> 9.3% | 8.0% -> 0.0% | +0.0pp | `lowerbody_fixed_contact_fail` |

Companion and guard observations:

| case | best observed signal | blocker |
|---|---|---|
| `box021_035_p2` | `corridor_ref_soft/ref_leglight` reduce lower-body 9.8% -> 0.0% while preserving 67-68% physics contact | strict still FAIL; contact does not improve relative to E113; small penetration regression |
| `box021_035_p2` | `corridor_surface_soft` raises physics contact 67.7% -> 82.7% | lower-body remains 10.5%, classified `contact_good_lowerbody_fail` |
| `box021_035_p1` | `corridor_surface_soft` raises physics contact 70.5% -> 78.3% and lower-body 0.0% | pelvis/posture regression, strict FAIL, classified `review` |
| `box004_082_p1` | `corridor_ref_soft/surface_soft` raise contact strongly | deep penetration +5.5pp / +25.7pp |
| `box004_083_p2` | all variants remove lower-body contact | contact collapses or penetration/posture fails |

## Visual Check

Representative keyframes:

- `workspace/core4d/results/E118/cem/smoke/keyframes/E118_box021_029_p2_corridor_ref_soft/f100.jpg`
- `workspace/core4d/results/E118/cem/smoke/keyframes/E118_box021_029_p2_corridor_ref_leglight/f100.jpg`
- `workspace/core4d/results/E118/cem/smoke/keyframes/E118_box021_029_p2_corridor_surface_soft/f100.jpg`
- `workspace/core4d/results/E118/cem/smoke/keyframes/E118_box021_035_p1_corridor_surface_soft/f100.jpg`

Visual observations:

- `box021_029_p2 / corridor_ref_soft`: robot falls backward with the box above/against the body; this matches high lower-body interference and deep penetration regression.
- `box021_029_p2 / corridor_ref_leglight`: similar collapse, with contact recovered through a non-carry posture and lower-body/object involvement.
- `box021_029_p2 / corridor_surface_soft`: lower-body interference is removed, but the hands abandon the box and physics contact collapses.
- `box021_035_p1 / corridor_surface_soft`: apparent hand/object contact improves, but posture bends downward; strict remains FAIL and this is not a release row.

## Decision

Do not launch E118 full CEM.

Reason:

- Smoke has 0 release candidates.
- The main case has no Pareto-improving row. The ref-target variants improve contact modestly but worsen lower-body interference and deep penetration. The surface-target variant removes lower-body interference but loses hand-object contact.
- Companion/guard rows confirm the same trade-off: contact can be increased in some rows, but not while preserving lower-body, penetration, and posture gates.

## Interpretation

E118 is useful as a negative diagnostic: a scalar soft carry corridor is better structured than E117's binary gate, but it still does not force a physically valid hand-supported carry state. CEM continues to satisfy parts of the objective through collapse, object/body contact, or hand abandonment.

The next method should not simply increase E118 iterations. It should make the carry state less gameable, for example:

1. Add a pose prior or torso/upright constraint that is active only during the carry window.
2. Make object support explicit, with separate penalties for object-on-body and lower-body support instead of only leg clearance.
3. Add anti-tip/object-orientation constraints with stronger coupling to hand contact.
4. Consider a staged/curriculum objective: first preserve standing hand support, then tighten lower-body/object clearance.
