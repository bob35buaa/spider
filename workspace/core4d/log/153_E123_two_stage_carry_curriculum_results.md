# E123 Two-Stage Carry Curriculum Results

Date: 2026-06-03

## Goal

E123 tests whether a true two-stage CEM curriculum can repair the remaining main Box021 carry failure.

E115-E122 ruled out several one-stage variants around `box021_029_p2`: lower-body penalties, carry corridors, support decomposition, terminal carry gates, and one-shot arm snap warmstarts. E123 therefore changes the optimizer structure:

1. Stage 1 runs a pose/upright/contact seed.
2. Stage 1 output is converted from scene-act qpos back to source/freejoint qpos.
3. Stage 2 loads that converted trajectory as `warmstart_qpos_path` and runs the E120/E121 support/terminal objectives.

The full CEM gate required main `box021_029_p2` Stage 2 smoke to improve over E120/E121/E122, especially physics contact >37.3% and preferably >=57%, lower-body/non-hand support trending toward <=5%, acceptable pelvis posture, and no gate starvation.

## Implementation

New E123 artifacts:

- Plan: `workspace/core4d/plan/132_E123_two_stage_carry_curriculum_plan.md`
- Builder/converter: `workspace/core4d/scripts/E123/build_two_stage_curriculum_manifest.py`
- Stage 1 manifest: `workspace/core4d/scripts/E123/stage1.tsv`
- Stage 2 manifest: `workspace/core4d/scripts/E123/variants.tsv`
- Preflight: `workspace/core4d/results/E123/preflight/phaseA_preflight.tsv`
- Warmstarts: `workspace/core4d/results/E123/warmstarts/*_warmstart_qpos.npz`
- Train script: `workspace/core4d/scripts/train/train_E123_two_stage_curriculum.sh`
- Remote runner: `workspace/core4d/scripts/run_E123_remote.sh`
- Pull script: `workspace/core4d/scripts/pull_E123_remote_results.sh`
- Evaluator: `workspace/core4d/scripts/eval/eval_E123_two_stage_curriculum.py`
- Eval entrypoint: `workspace/core4d/scripts/eval/eval_E123_two_stage_curriculum.sh`
- Overrides: `examples/config/override/core4d_E123_*.yaml`

Preflight generated 4 Stage 1 rows and 8 Stage 2 rows. The three Box021 cases use `pose_bodyguard_seed` from E119 `corridor_ref_pose_bodyguard`; Box004 uses `guard_terminal_soft_seed` from E121 because Box004 was not in the E119 workset.

The converter bridge was validated on the main case before the full smoke: Stage 1 scene-act qpos `(75, 42)` was converted to source/freejoint warmstart `(75, 43)`, with `snap_mask_window=8-70` and 84.0% mask coverage. Stage 2 logs confirmed the warmstart was loaded and replaced 126/200 frames of `qpos_ref`.

Static checks passed locally and on `spider-remote`:

```bash
python -m py_compile workspace/core4d/scripts/E123/build_two_stage_curriculum_manifest.py workspace/core4d/scripts/eval/eval_E123_two_stage_curriculum.py
bash -n workspace/core4d/scripts/train/train_E123_two_stage_curriculum.sh workspace/core4d/scripts/run_E123_remote.sh workspace/core4d/scripts/pull_E123_remote_results.sh workspace/core4d/scripts/eval/eval_E123_two_stage_curriculum.sh
bash workspace/core4d/scripts/eval/eval_E123_two_stage_curriculum.sh smoke --allow-missing
```

## Execution

Smoke CEM used `SMOKE_MAX_NUM_ITERATIONS=4`.

Execution layout:

| split | machine | Stage 1 rows | Stage 2 rows |
|---|---|---:|---:|
| `local-gpu0` | local GPU0 | 2 | 4 |
| `remote-gpu0` | `spider-remote` GPU0 | 1 | 2 |
| `remote-gpu1` | `spider-remote` GPU1 | 1 | 2 |

Remote execution used tmux session `E123_smoke_111742`. GPU checks showed enough free memory before launch, so no unrelated processes were killed.

Smoke outputs after local + remote pull:

| artifact | count |
|---|---:|
| root NPZ | 12 |
| MP4 | 12 |
| outdir `trajectory_mjwp_act.npz` | 12 |
| converted Stage 1 warmstarts | 4 |
| evaluator keyframes | 117 |
| contact sheets | 4 |
| missing variants | 0 |

## Quantitative Result

Evaluation command:

```bash
bash workspace/core4d/scripts/eval/eval_E123_two_stage_curriculum.sh smoke
```

Evaluator output:

- `workspace/core4d/results/E123/cem/smoke/smoke_eval_summary.md`
- `workspace/core4d/results/E123/cem/smoke/pareto_decisions.tsv`
- `workspace/core4d/results/E123/cem/smoke/release_candidates.tsv`
- `workspace/core4d/results/E123/cem/smoke/guard_failures.tsv`
- `workspace/core4d/results/E123/cem/smoke/smoke_method_metrics.csv`

Summary:

- evaluated variants: 8
- method rows: 24
- missing variants: 0
- release candidates: 0
- all Stage 2 variants: `support_decomp_fail`

Stage 2 decision table:

| case | variant | physics contact | lower-body | non-hand support | hand near-zero | deep pen | gate valid | fallback | pelvis ok | decision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| `box021_029_p2` | `stage2_support_surface` | 16.0% | 12.0% | 25.3% | 21.3% | 0.0% | 60.0% | 0.0% | false | `support_decomp_fail` |
| `box021_029_p2` | `stage2_terminal_soft` | 6.7% | 17.3% | 37.3% | 17.3% | 0.0% | 60.0% | 0.0% | false | `support_decomp_fail` |
| `box021_035_p2` | `stage2_support_surface` | 52.6% | 38.3% | 67.7% | 63.9% | 2.3% | 35.5% | 39.1% | false | `support_decomp_fail` |
| `box021_035_p2` | `stage2_terminal_soft` | 57.1% | 35.3% | 58.6% | 71.4% | 0.0% | 47.7% | 20.3% | false | `support_decomp_fail` |
| `box021_035_p1` | `stage2_support_surface` | 43.4% | 7.8% | 62.0% | 51.2% | 0.0% | 15.5% | 61.2% | false | `support_decomp_fail` |
| `box021_035_p1` | `stage2_terminal_soft` | 40.3% | 7.8% | 63.6% | 54.3% | 0.0% | 19.0% | 56.6% | false | `support_decomp_fail` |
| `box004_083_p2` | `stage2_support_surface` | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 70.8% | 0.0% | false | `support_decomp_fail` |
| `box004_083_p2` | `stage2_terminal_soft` | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 71.4% | 0.0% | false | `support_decomp_fail` |

Main case result:

- `stage2_support_surface`: physics contact 16.0%, below E120 best 37.3%, E121 best 32.0%, and E119 seed 49.3%.
- `stage2_terminal_soft`: physics contact 6.7%, with worse lower-body and non-hand support than support-surface.
- Both main variants fail pelvis posture and support decomposition.

## Visual Check

Contact sheets:

- `workspace/core4d/results/E123/cem/smoke/keyframes/contact_sheets/E123_main_box021_029_p2_sheet.jpg`
- `workspace/core4d/results/E123/cem/smoke/keyframes/contact_sheets/E123_box021_holdouts_sheet.jpg`
- `workspace/core4d/results/E123/cem/smoke/keyframes/contact_sheets/E123_box004_sheet.jpg`
- `workspace/core4d/results/E123/cem/smoke/keyframes/contact_sheets/E123_stage2_overview_sheet.jpg`

Representative high-scale frames:

- `workspace/core4d/results/E123/cem/smoke/keyframes/E123_box021_029_p2_stage2_support_surface/f100.jpg`
- `workspace/core4d/results/E123/cem/smoke/keyframes/E123_box021_029_p2_stage2_terminal_soft/f100.jpg`
- `workspace/core4d/results/E123/cem/smoke/keyframes/E123_box021_035_p2_stage2_terminal_soft/f100.jpg`

Visual observations:

- Main `box021_029_p2` Stage 2 rows collapse beside or onto the object instead of forming sustained hand-supported carry.
- `box021_035_p2` keeps higher hand proximity, but relies visibly on torso/leg/object support, matching 35.3-38.3% lower-body and 58.6-67.7% non-hand support.
- Box004 loses carry contact entirely.

## Decision

Do not launch E123 full CEM.

Reason:

- Smoke has 0 release candidates and 8/8 Stage 2 rows fail as `support_decomp_fail`.
- Main `box021_029_p2` fails the explicit full gate by a large margin: 16.0% or 6.7% physics contact vs required >37.3% and preferred >=57%.
- The Stage 1 trajectory bridge is mechanically valid, but Stage 2 does not preserve the useful E119 seed signal.
- Holdout cases show the same support-decomposition problem: higher apparent contact is still body/object support, not clean hand-supported carry.

## Interpretation

E123 rejects the hypothesis that a simple sequential CEM curriculum, implemented as Stage 1 trajectory warmstart plus Stage 2 support/terminal objectives, is enough to repair the main Box021 carry failure.

This is different from E122: the bridge is not a one-shot IK snap, and it does condition Stage 2 on an optimized Stage 1 trajectory. The negative result suggests the current Stage 2 objective still lacks a state/horizon-local carry mode that can keep hands engaged while preventing pelvis/body support.

Next work should stop the current E115-E123 CEM diagnostic line unless it makes a deeper optimizer change, such as explicit stage-local horizons, through-window anti-tip/object-orientation coupling, or an RL hand-support objective initialized from a visually verified positive seed.
