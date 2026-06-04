# E124 SBTO Carry Horizon Plan

Date: 2026-06-03

## Context

`contact_improvement_plan.md` targets high true contact, low deep penetration, stable posture, and lower-body/object strict pass.

E115-E123 produced useful negative evidence on `box021_029_p2`:

- one-stage lower-body/contact reward variants can trade contact for posture, but do not satisfy all gates;
- support decomposition and terminal carry gates expose the body-support shortcut but do not fix it;
- one-shot IK warmstart and a two-stage warmstart bridge both fail to preserve stable hand-supported carry.

The next aligned experiment should not add another scalar reward variant around standard receding-horizon CEM. The code already contains an existing SBTO path (`use_sbto`) that optimizes a growing full trajectory horizon. E124 tests whether that optimizer structure, combined with the existing through-window carry/support terms, can find a cleaner carry mode.

## Claims

### C1: SBTO artifacts are compatible with the existing evaluation pipeline

Each E124 smoke row must produce:

- root NPZ;
- MP4;
- outdir `trajectory_mjwp_act.npz`;
- keyframes;
- E121/E122-style lower-body/contact/support/terminal metrics.

### C2: SBTO changes optimizer structure, not reward semantics

E124 variants inherit existing E120/E121 objective definitions and only add:

- `use_sbto=true`;
- SBTO convergence/iteration defaults;
- smoke-time low-cost SBTO overrides through the train script.

This tests growing-horizon optimization instead of another one-stage penalty sweep.

### C3: smoke gate determines full CEM/SBTO

Full SBTO is allowed only if main `box021_029_p2` improves over E120-E123:

- physics contact >37.3%, preferably >=57%;
- lower-body/object interference <= E119/E113 level plus tolerance, preferably <=5%;
- non-hand object support <21.3%, preferably <=5%;
- `pelvis_ok=True` or clear visual posture improvement;
- deep penetration does not regress by more than +3pp;
- gate fallback is not the dominant path.

If main smoke does not pass this gate, do not launch E124 full.

## Workset

Reuse the E120-E123 4-case workset:

| case | purpose | split |
|---|---|---|
| `box021_029_p2` | main upright-carry gate | `local-gpu0` |
| `box021_035_p2` | companion lower-body repair | `remote-gpu0` |
| `box021_035_p1` | strict guard | `remote-gpu1` |
| `box004_083_p2` | Box004 contact guard | `local-gpu0` |

## Variants

| variant | base override | purpose |
|---|---|---|
| `sbto_support_surface` | E120 `support_surface_direct` | support decomposition objective under growing-horizon optimization |
| `sbto_terminal_soft` | E121 `terminal_soft_surface` | soft terminal carry gate under growing-horizon optimization |

Do not include hard terminal variants in the first E124 smoke; E121/E122 showed hard gates can starve under standard CEM.

## Artifacts

- builder: `workspace/core4d/scripts/E124/build_sbto_carry_horizon_manifest.py`
- variants: `workspace/core4d/scripts/E124/variants.tsv`
- preflight: `workspace/core4d/results/E124/preflight/phaseA_preflight.tsv`
- train: `workspace/core4d/scripts/train/train_E124_sbto_carry_horizon.sh`
- remote: `workspace/core4d/scripts/run_E124_remote.sh`
- pull: `workspace/core4d/scripts/pull_E124_remote_results.sh`
- eval: `workspace/core4d/scripts/eval/eval_E124_sbto_carry_horizon.py`
- eval shell: `workspace/core4d/scripts/eval/eval_E124_sbto_carry_horizon.sh`
- overrides: `examples/config/override/core4d_E124_*.yaml`

## Execution

Phase A smoke:

```bash
python workspace/core4d/scripts/E124/build_sbto_carry_horizon_manifest.py
SMOKE_SBTO_MAX_ITER_PER_KNOT=1 SMOKE_NUM_SAMPLES=512 bash workspace/core4d/scripts/train/train_E124_sbto_carry_horizon.sh local smoke 0
ssh spider-remote 'cd /home/xiayb/pHRI_workspace/spider && SMOKE_SBTO_MAX_ITER_PER_KNOT=1 SMOKE_NUM_SAMPLES=512 bash workspace/core4d/scripts/run_E124_remote.sh smoke'
bash workspace/core4d/scripts/pull_E124_remote_results.sh smoke
bash workspace/core4d/scripts/eval/eval_E124_sbto_carry_horizon.sh smoke
```

The train script uses the same local 1 GPU + remote 2 GPU split as recent CEM smoke runs. Do not kill unrelated GPU processes; only verify available memory before launch.

## Static Checks

- `python -m py_compile` for the E124 builder/evaluator.
- `bash -n` for train/remote/pull/eval shell scripts.
- `eval_E124_sbto_carry_horizon.sh smoke --allow-missing` before SBTO should report missing outputs without fake metrics.
- `git diff --check` on E124 files and generated overrides.

## Stop Rule

If E124 smoke does not improve main `box021_029_p2` over E120-E123, stop this CEM/SBTO diagnostic branch and move to an RL hand-support objective or a deeper optimizer with explicit stage-local state constraints.
