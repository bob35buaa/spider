# E122 Snap Warmstart Carry Prior Results

Date: 2026-06-03

## Goal

E122 tests whether changing the early CEM reference manifold helps the remaining main Box021 carry failure.

E120/E121 showed that support decomposition and terminal carry gates are useful diagnostics, but they do not move the hard main case `box021_029_p2` into a stable hand-supported carry. E122 therefore uses the smallest existing initialization hook:

- generate per-case arm snap IK warmstarts to the object surface;
- set `warmstart_qpos_path` for every variant;
- enable a new `warmstart_update_ctrl_from_qpos` hook so snapped robot qpos also updates the initial robot `ctrl_ref` mean on `snap_mask` frames;
- keep E120/E121 support and terminal-gate metrics as the smoke decision gate.

The full CEM gate required main `box021_029_p2` to beat E120/E121 smoke, especially physics contact >37.3% and preferably >=57%, while reducing lower-body and non-hand object support toward <=5% with acceptable pelvis/object posture.

## Implementation

Core addition:

- `spider/config.py`: added disabled-by-default `warmstart_update_ctrl_from_qpos`.
- `examples/run_mjwp.py`: when `warmstart_qpos_path` is loaded and the new flag is true, E122 updates robot `ctrl_ref` from snapped qpos on `snap_mask` frames before scene-act object padding.

New E122 artifacts:

- Plan: `workspace/core4d/plan/131_E122_snap_warmstart_carry_prior_plan.md`
- Builder: `workspace/core4d/scripts/E122/build_snap_warmstart_manifest.py`
- Manifest: `workspace/core4d/scripts/E122/variants.tsv`
- Preflight: `workspace/core4d/results/E122/preflight/phaseA_preflight.tsv`
- Warmstarts: `workspace/core4d/results/E122/warmstarts/*_snap_carry_warmstart_qpos.npz`
- Train script: `workspace/core4d/scripts/train/train_E122_snap_warmstart.sh`
- Remote runner: `workspace/core4d/scripts/run_E122_remote.sh`
- Pull script: `workspace/core4d/scripts/pull_E122_remote_results.sh`
- Evaluator: `workspace/core4d/scripts/eval/eval_E122_snap_warmstart.py`
- Eval entrypoint: `workspace/core4d/scripts/eval/eval_E122_snap_warmstart.sh`
- Overrides: `examples/config/override/core4d_E122_*.yaml`

Preflight generated 12 variants across the E120/E121 4-case workset. All required task, mask, baseline, surface, and warmstart checks passed. Warmstart snap masks covered 66.7-84.0% of source frames; final hand-to-surface mean distance was 1.0-1.4cm.

Static checks passed locally and on `spider-remote`:

```bash
python -m py_compile spider/config.py examples/run_mjwp.py workspace/core4d/scripts/E122/build_snap_warmstart_manifest.py workspace/core4d/scripts/eval/eval_E122_snap_warmstart.py
bash -n workspace/core4d/scripts/train/train_E122_snap_warmstart.sh workspace/core4d/scripts/run_E122_remote.sh workspace/core4d/scripts/pull_E122_remote_results.sh workspace/core4d/scripts/eval/eval_E122_snap_warmstart.sh
bash workspace/core4d/scripts/eval/eval_E122_snap_warmstart.sh smoke --allow-missing
```

The evaluator was fixed to parse the E122 commented TSV schema directly, avoiding the older E115 fixed-field parser misclassifying missing outputs.

## Execution

Smoke CEM used `SMOKE_MAX_NUM_ITERATIONS=4`.

Execution layout:

| split | machine | rows |
|---|---|---:|
| `local-gpu0` | local GPU0 | 6 |
| `remote-gpu0` | `spider-remote` GPU0 | 3 |
| `remote-gpu1` | `spider-remote` GPU1 | 3 |

Remote execution used tmux session `E122_smoke_104901`. GPU checks showed enough free memory before launch, so no unrelated processes were killed.

Smoke outputs after local + remote pull:

| artifact | count |
|---|---:|
| root NPZ | 12 |
| MP4 | 12 |
| outdir `trajectory_mjwp_act.npz` | 12 |
| evaluator keyframes | 117 |
| missing variants | 0 |

## Quantitative Result

Evaluation command:

```bash
bash workspace/core4d/scripts/eval/eval_E122_snap_warmstart.sh smoke
```

Evaluator output:

- `workspace/core4d/results/E122/cem/smoke/smoke_eval_summary.md`
- `workspace/core4d/results/E122/cem/smoke/pareto_decisions.tsv`
- `workspace/core4d/results/E122/cem/smoke/release_candidates.tsv`
- `workspace/core4d/results/E122/cem/smoke/guard_failures.tsv`
- `workspace/core4d/results/E122/cem/smoke/smoke_method_metrics.csv`

Summary:

- evaluated variants: 12
- method rows: 36
- missing variants: 0
- release candidates: 0

Main case:

| case | variant | physics contact | lower-body | non-hand support | hand near-zero | deep pen | gate valid | fallback | pelvis ok | decision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| `box021_029_p2` | `snap_support` | 12.0% | 4.0% | 12.0% | 6.7% | 1.3% | 60.0% | 0.0% | false | `support_decomp_fail` |
| `box021_029_p2` | `snap_terminal_soft` | 12.0% | 2.7% | 9.3% | 6.7% | 2.7% | 60.0% | 0.0% | false | `support_decomp_fail` |
| `box021_029_p2` | `snap_hard_surface` | 12.0% | 6.7% | 17.3% | 5.3% | 1.3% | 0.0% | 60.0% | false | `terminal_gate_starved` |

Companion and guards:

| case | best variant by contact | physics contact | lower-body | non-hand support | hand near-zero | pelvis ok | decision |
|---|---|---:|---:|---:|---:|---|---|
| `box021_035_p2` | `snap_terminal_soft` | 46.6% | 10.5% | 32.3% | 46.6% | false | `support_decomp_fail` |
| `box021_035_p1` | `snap_terminal_soft` | 70.5% | 0.0% | 0.0% | 76.7% | true | `lowerbody_fixed_contact_fail` |
| `box004_083_p2` | all variants | 0.0% | 0.0% | 0.0% | 0.0% | false | `support_decomp_fail` / `terminal_gate_starved` |

Decision summary from the evaluator:

- Main `box021_029_p2`: all three variants fail; physics contact is only 12.0%, below E120 best 37.3% and E121 best 32.0%.
- `snap_terminal_soft` improves lower-body and non-hand support relative to E120/E121, but it does so by losing useful hand-object physics contact and pelvis posture.
- `snap_hard_surface` still starves the hard gate: 0.0% valid, 60.0% fallback.
- The companion positive signal from E120/E121 does not survive this warmstart setup for `box021_035_p2`; contact drops to 44.4-46.6% with high non-hand support.

## Visual Check

Contact sheets:

- `workspace/core4d/results/E122/cem/smoke/visual_qc/E122_smoke_box021_029_p2_contact_sheet.jpg`
- `workspace/core4d/results/E122/cem/smoke/visual_qc/E122_smoke_box021_035_p2_contact_sheet.jpg`
- `workspace/core4d/results/E122/cem/smoke/visual_qc/E122_smoke_box021_035_p1_contact_sheet.jpg`
- `workspace/core4d/results/E122/cem/smoke/visual_qc/E122_smoke_box004_083_p2_contact_sheet.jpg`

Visual observations:

- Main `box021_029_p2` rows remain low/unstable and do not form a sustained hand-supported carry. This matches 12.0% physics contact and `pelvis_ok=False`.
- `box021_035_p2` shows object movement and intermittent contact, but the posture and support decomposition are worse than the earlier E120/E121 companion signal.
- `box021_035_p1` remains visually plausible and numerically high-contact, but it is the strict guard rather than the main release case.
- Box004 loses carry contact entirely.

## Decision

Do not launch E122 full CEM.

Reason:

- Smoke has 0 release candidates.
- Main `box021_029_p2` fails the explicit full gate by a large margin: 12.0% physics contact vs required >37.3% and preferred >=57%.
- Although lower-body/non-hand metrics improve in `snap_support` and `snap_terminal_soft`, the improvement is a collapse of carry contact, not a better hand-supported carry.
- Hard terminal gating still starves when combined with the snap warmstart.

## Interpretation

E122 rejects the hypothesis that a one-shot arm snap warmstart plus existing support/terminal gates is enough to repair the main Box021 carry failure.

The negative result is useful: CEM is not simply missing a near-object arm seed. The next iteration should stop adding terminal/penalty variants around the same one-stage optimizer and move to a real staged optimizer/curriculum, for example:

1. a two-stage CEM where stage 1 optimizes upright pose and object-relative hand support, then stage 2 optimizes carry dynamics;
2. a through-window anti-tip/object-orientation constraint active across the contact interval rather than terminal-only;
3. an RL hand-support objective gated by E120-E122 metrics, once an initialization/curriculum can preserve contact without posture collapse.
