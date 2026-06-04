# E124 SBTO Carry Horizon Results

Date: 2026-06-03

## Goal

E124 tests whether the existing SBTO growing-horizon optimizer can repair the remaining main Box021 carry failure when paired with the E120/E121 support and terminal carry objectives.

This is a structural optimizer test, not a new scalar reward sweep. E123 showed that a two-stage warmstart bridge is mechanically valid but does not preserve the E119 carry seed. E124 therefore uses the existing `use_sbto` path in `examples/run_mjwp.py` to optimize an increasing trajectory horizon.

The full gate required main `box021_029_p2` to improve over E120-E123, especially physics contact >37.3%, lower-body/non-hand support trending toward <=5%, acceptable pelvis posture, low deep penetration, and no gate starvation.

## Implementation

New E124 artifacts:

- Plan: `workspace/core4d/plan/133_E124_sbto_carry_horizon_plan.md`
- Builder: `workspace/core4d/scripts/E124/build_sbto_carry_horizon_manifest.py`
- Manifest: `workspace/core4d/scripts/E124/variants.tsv`
- Preflight: `workspace/core4d/results/E124/preflight/phaseA_preflight.tsv`
- Train script: `workspace/core4d/scripts/train/train_E124_sbto_carry_horizon.sh`
- Remote runner: `workspace/core4d/scripts/run_E124_remote.sh`
- Pull script: `workspace/core4d/scripts/pull_E124_remote_results.sh`
- Evaluator: `workspace/core4d/scripts/eval/eval_E124_sbto_carry_horizon.py`
- Eval entrypoint: `workspace/core4d/scripts/eval/eval_E124_sbto_carry_horizon.sh`
- Overrides: `examples/config/override/core4d_E124_*.yaml`

Core compatibility fix:

- `examples/run_mjwp.py`: SBTO replay now records sim arrays plus explicit `qpos_ref/qvel_ref/ctrl_ref/time_ref` channels.
- `train_E124_sbto_carry_horizon.sh`: when copying SBTO outputs, it preserves raw SBTO output as `trajectory_mjwp_act_sbto_raw.npz` and writes evaluator-facing paired `qpos/qvel/ctrl/time` arrays with standard `(T,2,*)` shape.

This keeps SBTO internal object-error computation sim-only while making E090/E105 replay metrics compatible with the standard MJWP artifact contract.

Static checks passed locally and on `spider-remote`:

```bash
python -m py_compile examples/run_mjwp.py workspace/core4d/scripts/E124/build_sbto_carry_horizon_manifest.py workspace/core4d/scripts/eval/eval_E124_sbto_carry_horizon.py
bash -n workspace/core4d/scripts/train/train_E124_sbto_carry_horizon.sh workspace/core4d/scripts/run_E124_remote.sh workspace/core4d/scripts/pull_E124_remote_results.sh workspace/core4d/scripts/eval/eval_E124_sbto_carry_horizon.sh
bash workspace/core4d/scripts/eval/eval_E124_sbto_carry_horizon.sh smoke --allow-missing
```

## Execution

Smoke SBTO used:

```bash
SMOKE_SBTO_MAX_ITER_PER_KNOT=1
SMOKE_NUM_SAMPLES=256
SMOKE_SBTO_KNOT_DT=0.35
```

Execution layout:

| split | machine | rows |
|---|---|---:|
| `local-gpu0` | local GPU0 | 4 |
| `remote-gpu0` | `spider-remote` GPU0 | 2 |
| `remote-gpu1` | `spider-remote` GPU1 | 2 |

Remote execution used tmux session `E124_smoke_114705`. GPU checks showed enough free memory before launch, so no unrelated processes were killed.

Smoke outputs after local + remote pull:

| artifact | count |
|---|---:|
| root NPZ | 8 |
| MP4 | 8 |
| outdir `trajectory_mjwp_act.npz` | 8 |
| raw SBTO backup `trajectory_mjwp_act_sbto_raw.npz` | 8 |
| evaluator keyframes | 78 |
| contact sheets | 4 |
| missing variants | 0 |

## Quantitative Result

Evaluation command:

```bash
bash workspace/core4d/scripts/eval/eval_E124_sbto_carry_horizon.sh smoke
```

Evaluator output:

- `workspace/core4d/results/E124/cem/smoke/smoke_eval_summary.md`
- `workspace/core4d/results/E124/cem/smoke/pareto_decisions.tsv`
- `workspace/core4d/results/E124/cem/smoke/release_candidates.tsv`
- `workspace/core4d/results/E124/cem/smoke/guard_failures.tsv`
- `workspace/core4d/results/E124/cem/smoke/smoke_method_metrics.csv`

Summary:

- evaluated variants: 8
- method rows: 24
- missing variants: 0
- release candidates: 0
- all variants: `support_decomp_fail`

Decision table:

| case | variant | physics contact | lower-body | non-hand support | hand near-zero | deep pen | object ok | pelvis ok | decision |
|---|---|---:|---:|---:|---:|---:|---|---|---|
| `box021_029_p2` | `sbto_support_surface` | 4.0% | 0.0% | 0.0% | 2.0% | 1.3% | false | false | `support_decomp_fail` |
| `box021_029_p2` | `sbto_terminal_soft` | 4.0% | 0.0% | 0.0% | 2.0% | 1.3% | false | false | `support_decomp_fail` |
| `box021_035_p2` | `sbto_support_surface` | 21.8% | 15.0% | 26.7% | 17.3% | 1.1% | false | false | `support_decomp_fail` |
| `box021_035_p2` | `sbto_terminal_soft` | 22.2% | 13.9% | 26.3% | 17.3% | 1.1% | false | false | `support_decomp_fail` |
| `box021_035_p1` | `sbto_support_surface` | 24.0% | 0.0% | 8.5% | 24.8% | 0.0% | false | false | `support_decomp_fail` |
| `box021_035_p1` | `sbto_terminal_soft` | 25.2% | 0.0% | 15.1% | 26.0% | 0.0% | false | false | `support_decomp_fail` |
| `box004_083_p2` | `sbto_support_surface` | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | false | false | `support_decomp_fail` |
| `box004_083_p2` | `sbto_terminal_soft` | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | false | false | `support_decomp_fail` |

Main case result:

- Both main variants collapse to 4.0% physics contact, far below E120 best 37.3%, E121 best 32.0%, E123 best 16.0%, and E119 seed 49.3%.
- Lower-body and non-hand support go to 0.0%, but only because the carry/contact mode is lost.
- `object_ok=False` and `pelvis_ok=False` on every E124 row.

## Visual Check

Contact sheets:

- `workspace/core4d/results/E124/cem/smoke/keyframes/contact_sheets/E124_main_box021_029_p2_sheet.jpg`
- `workspace/core4d/results/E124/cem/smoke/keyframes/contact_sheets/E124_box021_holdouts_sheet.jpg`
- `workspace/core4d/results/E124/cem/smoke/keyframes/contact_sheets/E124_box004_sheet.jpg`
- `workspace/core4d/results/E124/cem/smoke/keyframes/contact_sheets/E124_all_stage_sheet.jpg`

Visual observations:

- Main `box021_029_p2` separates from the box through the contact window and does not form hand-supported carry.
- Box021 holdouts show unstable posture or body-proximity support and do not preserve the prior clean-contact signals.
- Box004 loses carry contact entirely.

## Decision

Do not launch E124 full SBTO/CEM.

Reason:

- Smoke has 0 release candidates and 8/8 rows fail as `support_decomp_fail`.
- Main physics contact is only 4.0%, far below the full gate.
- The growing-horizon SBTO mechanism removes some support shortcuts by losing contact and object/pelvis quality, not by finding a cleaner carry trajectory.

## Interpretation

E124 rejects the hypothesis that the existing SBTO growing-horizon optimizer, without deeper state constraints, is enough to repair the main Box021 carry failure.

The E115-E124 sequence now rules out the current CEM/SBTO reward-and-initialization branch for `box021_029_p2`. The next aligned step should move to an RL hand-support objective or a deeper optimizer that explicitly constrains stage-local carry state rather than relying on CEM reward terms to discover it.
