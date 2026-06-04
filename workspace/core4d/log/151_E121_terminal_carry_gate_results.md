# E121 Terminal Carry Gate Results

Date: 2026-06-03

## Goal

E121 tests whether a semantic terminal carry gate can fix the remaining Box021 object-support failure by affecting CEM elite selection, rather than adding another scalar reward sweep.

The experiment follows E120's negative result:

- E120 main `box021_029_p2` still had only 29.3-37.3% physics contact, 18.7-32.0% lower-body interference, 30.7-36.0% non-hand support, and failed pelvis posture.
- E120 companion `box021_035_p2/support_surface_direct` showed the positive direction: 72.9% physics contact, 0.0% lower-body, 0.0% non-hand support, and 78.2% hand near-zero.

E121 keeps the E120 workset and compares:

- `terminal_soft_surface`
- `terminal_hard_surface`
- `terminal_hard_ref`

The smoke gate for launching full CEM required the main `box021_029_p2` row to improve over E120's best main row, especially physics contact >37.3% and preferably >=57%, while reducing lower-body and non-hand support.

## Implementation

Core additions:

- `spider/config.py`: added disabled-by-default `terminal_carry_gate_*` config fields and geometry resolution for terminal hand/non-hand support checks.
- `spider/simulators/mjwp.py`: added terminal pelvis/object-rotation/non-hand-support/hand-near checks and exported `info` metrics.
- `hard` and `hard_soft` modes fold terminal violation into the existing `cem_gate_*` elite filter, so E121 does not add a new optimizer scheduler.

New E121 artifacts:

- Plan: `workspace/core4d/plan/130_E121_terminal_carry_gate_plan.md`
- Builder: `workspace/core4d/scripts/E121/build_terminal_carry_gate_manifest.py`
- Manifest: `workspace/core4d/scripts/E121/variants.tsv`
- Preflight: `workspace/core4d/results/E121/preflight/phaseA_preflight.tsv`
- Train script: `workspace/core4d/scripts/train/train_E121_terminal_carry_gate.sh`
- Remote runner: `workspace/core4d/scripts/run_E121_remote.sh`
- Pull script: `workspace/core4d/scripts/pull_E121_remote_results.sh`
- Evaluator: `workspace/core4d/scripts/eval/eval_E121_terminal_carry_gate.py`
- Eval entrypoint: `workspace/core4d/scripts/eval/eval_E121_terminal_carry_gate.sh`
- Overrides: `examples/config/override/core4d_E121_*.yaml`

Preflight generated 12 variants. All required task, mask, baseline, E113 hold-result, E100 surface-target, and support-geometry checks passed. Split assignment was local 6, remote GPU0 3, remote GPU1 3.

Static checks passed:

```bash
python -m py_compile spider/config.py spider/simulators/mjwp.py workspace/core4d/scripts/E121/build_terminal_carry_gate_manifest.py workspace/core4d/scripts/eval/eval_E121_terminal_carry_gate.py
bash -n workspace/core4d/scripts/train/train_E121_terminal_carry_gate.sh workspace/core4d/scripts/run_E121_remote.sh workspace/core4d/scripts/pull_E121_remote_results.sh workspace/core4d/scripts/eval/eval_E121_terminal_carry_gate.sh
bash workspace/core4d/scripts/eval/eval_E121_terminal_carry_gate.sh smoke --allow-missing
git diff --check -- spider/config.py spider/simulators/mjwp.py workspace/core4d/plan/130_E121_terminal_carry_gate_plan.md workspace/core4d/scripts/E121/build_terminal_carry_gate_manifest.py workspace/core4d/scripts/train/train_E121_terminal_carry_gate.sh workspace/core4d/scripts/run_E121_remote.sh workspace/core4d/scripts/pull_E121_remote_results.sh workspace/core4d/scripts/eval/eval_E121_terminal_carry_gate.py workspace/core4d/scripts/eval/eval_E121_terminal_carry_gate.sh workspace/core4d/progress.md examples/config/override/core4d_E121_*.yaml
```

Remote static checks also passed after rsyncing E121 core code, scripts, overrides, and preflight to `spider-remote:/home/xiayb/pHRI_workspace/spider`. During remote setup, the train script was fixed to avoid a bare `python` dependency and now prefers `PYTHON_BIN`, `.venv/bin/python`, then `python3`.

## Execution

Smoke CEM used `SMOKE_MAX_NUM_ITERATIONS=4`.

Execution layout:

| split | machine | rows |
|---|---|---:|
| `local-gpu0` | local GPU0 | 6 |
| `remote-gpu0` | `spider-remote` GPU0 | 3 |
| `remote-gpu1` | `spider-remote` GPU1 | 3 |

Remote execution used tmux session `E121_smoke_101917`. No unrelated GPU processes were stopped.

Results were pulled back with:

```bash
bash workspace/core4d/scripts/pull_E121_remote_results.sh smoke
```

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
bash workspace/core4d/scripts/eval/eval_E121_terminal_carry_gate.sh smoke
```

Evaluator output:

- `workspace/core4d/results/E121/cem/smoke/smoke_eval_summary.md`
- `workspace/core4d/results/E121/cem/smoke/pareto_decisions.tsv`
- `workspace/core4d/results/E121/cem/smoke/release_candidates.tsv`
- `workspace/core4d/results/E121/cem/smoke/guard_failures.tsv`
- `workspace/core4d/results/E121/cem/smoke/smoke_method_metrics.csv`

Summary:

- evaluated variants: 12
- method rows: 36
- missing variants: 0
- release candidates: 0

Decision counts:

| decision | count |
|---|---:|
| `lowerbody_fixed_contact_fail` | 8 |
| `support_decomp_fail` | 2 |
| `terminal_gate_starved` | 1 |
| `review` | 1 |

Main case:

| case | variant | physics contact | lower-body | non-hand support | hand near-zero | CEM gate valid | fallback | pelvis ok | decision |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| `box021_029_p2` | `terminal_soft_surface` | 32.0% | 21.3% | 29.3% | 66.7% | 92.0% | 0.0% | false | `support_decomp_fail` |
| `box021_029_p2` | `terminal_hard_surface` | 32.0% | 16.0% | 21.3% | 66.7% | 2.9% | 87.3% | false | `support_decomp_fail` |
| `box021_029_p2` | `terminal_hard_ref` | 28.0% | 18.7% | 28.0% | 30.7% | 0.0% | 92.0% | false | `terminal_gate_starved` |

Companion signal:

| case | variant | physics contact | lower-body | non-hand support | hand near-zero | CEM gate valid | fallback | pelvis ok | decision |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| `box021_035_p2` | `terminal_soft_surface` | 78.2% | 0.0% | 0.0% | 79.7% | 95.0% | 0.0% | true | `review` |
| `box021_035_p2` | `terminal_hard_surface` | 75.2% | 0.0% | 0.0% | 82.0% | 67.4% | 10.9% | true | `lowerbody_fixed_contact_fail` |
| `box021_035_p2` | `terminal_hard_ref` | 57.1% | 0.0% | 0.0% | 72.9% | 60.8% | 23.7% | true | `lowerbody_fixed_contact_fail` |

## Visual Check

Representative contact sheets:

- `workspace/core4d/results/E121/cem/smoke/e121_main_visual_qc_contact_sheet.jpg`
- `workspace/core4d/results/E121/cem/smoke/e121_companion_guard_visual_qc_contact_sheet.jpg`

Visual observations:

- Main `box021_029_p2` rows remain low/tilted and do not reach a stable upright carry posture. This matches `pelvis_ok=False`, 28.0-32.0% physics contact, and 21.3-29.3% non-hand support.
- `terminal_hard_ref` visibly reduces useful hand contact and the CEM gate starves, matching 0.0% gate-valid fraction and 92.0% fallback.
- Companion `box021_035_p2` rows look much cleaner and maintain the positive hand-supported carry shape, matching 57.1-78.2% physics contact with 0.0% lower-body and non-hand support. This does not transfer to the hard main case.
- Box004 guard remains low-contact visually and numerically; it is not a release signal.

## Decision

Do not launch E121 full CEM.

Reason:

- Smoke has 0 release candidates.
- The main `box021_029_p2` gate requires physics contact > E120 best main value 37.3%, preferably >=57%, with lower-body and non-hand support reduced toward <=5%.
- All main E121 rows fail contact and pelvis posture. The best main physics contact is only 32.0%, below E120's 37.3% best smoke row.
- `terminal_hard_surface` does reduce lower-body/non-hand support relative to E120 but only to 16.0%/21.3%, still far above the <=5% target, and it relies on heavy CEM fallback.
- `terminal_hard_ref` confirms hard filtering can starve valid samples instead of discovering a better carry mode.

## Interpretation

E121 rejects the hypothesis that a terminal elite-selection gate alone is enough to repair the main Box021 carry failure under the current CEM setup.

The useful signal remains case-dependent: companion `box021_035_p2` can achieve clean hand-supported carry with the same plumbing, but `box021_029_p2` needs a mechanism that changes the trajectory manifold earlier, not just terminal selection. The next iteration should avoid another scalar reward sweep and move toward one of:

1. pose-conditioned initialization or a staged carry curriculum that anchors upright posture before contact optimization
2. object orientation / anti-tip coupling active throughout the carry window, not only terminally
3. RL hand-support policy objective using E120/E121 metrics as hard evaluation gates, once a better initialization/curriculum exists
4. continue requiring `nonhand_object_support_frac` and terminal gate/fallback metrics in any future Box021 release decision
