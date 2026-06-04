# E119 Upright Carry Support-Guard Results

Date: 2026-06-03

## Goal

E119 follows the E115-E118 negative diagnostics and tests whether existing knobs can combine the partial positives from:

- E116 posture/body safety guard
- E118 carry/contact corridor

Phase A intentionally does not change core reward code. It tests a 3-case Box021 workset with 3 variants:

- `corridor_ref_pose_bodyguard`
- `corridor_ref_pose_bodygate`
- `corridor_surface_pose_bodyguard`

## Implementation

New artifacts:

- Plan: `workspace/core4d/plan/128_E119_upright_carry_support_guard_plan.md`
- Manifest/preflight builder: `workspace/core4d/scripts/E119/build_upright_carry_manifest.py`
- Manifest: `workspace/core4d/scripts/E119/variants.tsv`
- Preflight: `workspace/core4d/results/E119/preflight/phaseA_preflight.tsv`
- Train script: `workspace/core4d/scripts/train/train_E119_upright_carry.sh`
- Remote runner: `workspace/core4d/scripts/run_E119_remote.sh`
- Pull script: `workspace/core4d/scripts/pull_E119_remote_results.sh`
- Evaluator: `workspace/core4d/scripts/eval/eval_E119_upright_carry.py`
- Eval entrypoint: `workspace/core4d/scripts/eval/eval_E119_upright_carry.sh`
- Overrides: `examples/config/override/core4d_E119_*.yaml`

Preflight generated 9 variants. All required task, mask, baseline, E113 hold-result, and E100 surface-target checks passed. Split assignment was local 3, remote GPU0 3, remote GPU1 3.

Static checks passed:

```bash
python -m py_compile workspace/core4d/scripts/E119/build_upright_carry_manifest.py workspace/core4d/scripts/eval/eval_E119_upright_carry.py
bash -n workspace/core4d/scripts/train/train_E119_upright_carry.sh workspace/core4d/scripts/run_E119_remote.sh workspace/core4d/scripts/pull_E119_remote_results.sh workspace/core4d/scripts/eval/eval_E119_upright_carry.sh
bash workspace/core4d/scripts/eval/eval_E119_upright_carry.sh smoke --allow-missing
git diff --check -- workspace/core4d/plan/128_E119_upright_carry_support_guard_plan.md workspace/core4d/scripts/E119/build_upright_carry_manifest.py workspace/core4d/scripts/train/train_E119_upright_carry.sh workspace/core4d/scripts/run_E119_remote.sh workspace/core4d/scripts/pull_E119_remote_results.sh workspace/core4d/scripts/eval/eval_E119_upright_carry.py workspace/core4d/scripts/eval/eval_E119_upright_carry.sh workspace/core4d/progress.md examples/config/override/core4d_E119_*.yaml
```

Remote static eval initially exposed missing helper files on `spider-remote`; the E090/E105/E115 eval helpers and E098/E105 import helpers were synced, then remote `eval_E119_upright_carry.sh smoke --allow-missing` passed.

## Execution

Smoke CEM used `SMOKE_MAX_NUM_ITERATIONS=4`.

Execution layout:

| split | machine | rows |
|---|---|---:|
| `local-gpu0` | local GPU0 | 3 |
| `remote-gpu0` | `spider-remote` GPU0 | 3 |
| `remote-gpu1` | `spider-remote` GPU1 | 3 |

Remote execution used `tmux` session `E119_smoke_090739`. Local GPU0 completed the main `box021_029_p2` split, then backfilled three pending companion variants while the remote workers continued:

- `E119_box021_035_p1_corridor_ref_pose_bodygate`
- `E119_box021_035_p1_corridor_surface_pose_bodyguard`
- `E119_box021_035_p2_corridor_surface_pose_bodyguard`

This produced a complete local 9-row smoke set after pulling completed remote artifacts. No unrelated GPU processes were stopped.

Results were pulled back with:

```bash
bash workspace/core4d/scripts/pull_E119_remote_results.sh smoke
```

Smoke outputs after local + remote pull:

| artifact | count |
|---|---:|
| root NPZ | 9 |
| MP4 | 9 |
| outdir `trajectory_mjwp_act.npz` | 9 |
| missing variants | 0 |

## Quantitative Result

Evaluation command:

```bash
bash workspace/core4d/scripts/eval/eval_E119_upright_carry.sh smoke
```

Evaluator output:

- `workspace/core4d/results/E119/cem/smoke/smoke_eval_summary.md`
- `workspace/core4d/results/E119/cem/smoke/pareto_decisions.tsv`
- `workspace/core4d/results/E119/cem/smoke/release_candidates.tsv`
- `workspace/core4d/results/E119/cem/smoke/guard_failures.tsv`

Summary:

- evaluated variants: 9
- method rows: 27
- missing variants: 0
- release candidates: 0

Decision counts:

| decision | count |
|---|---:|
| `lowerbody_fixed_contact_fail` | 7 |
| `review` | 2 |

Main case:

| case | variant | physics contact E107 -> test | lower-body E113 -> test | deep penetration delta | pelvis ok | decision |
|---|---|---:|---:|---:|---|---|
| `box021_029_p2` | `corridor_ref_pose_bodyguard` | 45.3% -> 49.3% | 8.0% -> 0.0% | +0.0pp | false | `lowerbody_fixed_contact_fail` |
| `box021_029_p2` | `corridor_ref_pose_bodygate` | 45.3% -> 48.0% | 8.0% -> 1.3% | +0.0pp | false | `lowerbody_fixed_contact_fail` |
| `box021_029_p2` | `corridor_surface_pose_bodyguard` | 45.3% -> 18.7% | 8.0% -> 21.3% | +0.0pp | false | `review` |

Companion and guard observations:

| case | best observed signal | blocker |
|---|---|---|
| `box021_035_p2` | `corridor_surface_pose_bodyguard` reaches 78.2% physics contact, lower-body 0.0%, deep penetration +0.0pp | strict still FAIL; this is a companion row, not the main gate |
| `box021_035_p2` | ref/bodyguard variants remove lower-body interference | contact falls below E113 and strict remains FAIL |
| `box021_035_p1` | all variants remove lower-body interference; surface target reaches 72.9% physics contact | strict remains FAIL; ref variants lose contact and have small penetration regression |

## Visual Check

Representative keyframes:

- `workspace/core4d/results/E119/cem/smoke/keyframes/E119_box021_029_p2_corridor_ref_pose_bodyguard/f100.jpg`
- `workspace/core4d/results/E119/cem/smoke/keyframes/E119_box021_029_p2_corridor_surface_pose_bodyguard/f100.jpg`
- `workspace/core4d/results/E119/cem/smoke/keyframes/E119_box021_035_p2_corridor_surface_pose_bodyguard/f100.jpg`
- `workspace/core4d/results/E119/cem/smoke/keyframes/E119_box021_035_p1_corridor_surface_pose_bodyguard/f100.jpg`

Visual observations:

- `box021_029_p2 / corridor_ref_pose_bodyguard`: object contact is present, but the sim posture bends/collapses relative to the reference; this matches `pelvis_ok=False`.
- `box021_029_p2 / corridor_surface_pose_bodyguard`: hands/posture lose the valid carry state, and lower-body/object involvement is visible.
- `box021_035_p2 / corridor_surface_pose_bodyguard`: strong companion signal with object contact and no lower-body interference, but it does not clear the main-case release gate.
- `box021_035_p1 / corridor_surface_pose_bodyguard`: contact is present, but posture remains strict-fail.

## Decision

Do not launch E119 full CEM.

Reason:

- Smoke has 0 release candidates.
- The main `box021_029_p2` case has no valid Pareto row. Ref-target variants reduce lower-body interference and avoid deep penetration, but they remain pelvis/posture failures and only reach 48.0-49.3% physics contact. The surface-target variant collapses contact to 18.7% and worsens lower-body interference to 21.3%.
- The best companion row, `box021_035_p2 / corridor_surface_pose_bodyguard`, is useful evidence that surface targets can work in an easier posture, but it does not satisfy the Phase A success gate.

## Interpretation

E119 is a negative diagnostic for the current composition of existing knobs. Combining posture guard, body-support penalty, deep-hand guard, and carry corridor is not enough to make the main Box021 carry state non-gameable.

The next method should add a new constraint or curriculum rather than another scalar sweep. The strongest next direction is an explicit object support decomposition:

1. Reward hand-supported object contact during raw-contact windows.
2. Penalize object support from torso/pelvis/upper arms/lower body separately from hand-object contact.
3. Preserve upright carry posture with a stage schedule instead of applying all objectives at the same strength from the start.
