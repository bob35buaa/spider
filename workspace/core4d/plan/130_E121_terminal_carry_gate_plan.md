# E121 Terminal Carry Gate Plan

Date: 2026-06-03

## Context

`contact_improvement_plan.md` requires high true contact, low deep penetration, stable posture, and lower-body/object strict pass before any RL handoff. E115-E120 have made the remaining Box021 blocker sharper:

- E115/E117 lower-body penalties can remove lower-body/object interference, but contact collapses and posture degrades.
- E118/E119 can recover some contact only through body/lower-body support or invalid posture.
- E120 added explicit hand-support/non-hand-support reward and evaluator metrics, but the main `box021_029_p2` smoke still failed: physics contact only 29.3-37.3%, lower-body 18.7-32.0%, non-hand support 30.7-36.0%, and `pelvis_ok=False`.
- E120 companion `box021_035_p2/support_surface_direct` is the strongest positive signal: 72.9% physics contact, 0.0% lower-body, 0.0% non-hand support, 78.2% hand near-zero.

Subagent Heisenberg and local code audit agree that current code does not support a true per-iteration CEM reward curriculum. Existing "staged" variants are time-window reward gates only. E121 should therefore test a different mechanism: a semantic terminal carry gate that affects CEM elite selection, not another reward scalar sweep.

## Claims

### Claim A: terminal semantic filtering prevents collapsed carry solutions

A terminal gate on the CEM horizon should reject samples whose future carry state ends with low pelvis, excessive object rotation, or non-hand object support. This should prevent the main `box021_029_p2` solution from satisfying contact/object terms through a collapsed posture.

Measured by main `box021_029_p2`:

- `pelvis_ok=True` or visibly better terminal posture than E120;
- non-hand support below E120's 30.7-36.0%, ideally <= 5%;
- lower-body/object interference below E120's 18.7-32.0%, ideally <= 5%.

### Claim B: hard gate is materially different from soft reward

E121 compares soft terminal carry shaping with hard elite filtering. If hard filtering helps while soft-only does not, the failure mode is sample selection, not reward magnitude.

Measured by:

- `terminal_hard_surface` improves support/posture metrics over `terminal_soft_surface`;
- `cem_gate_valid_frac`, fallback usage, and terminal gate info show whether the hard gate is feasible or starving samples.

### Claim C: surface target remains the best contact route, but must be paired with posture support

E120 showed surface target improves hand near-zero on the main case and is clean on the companion case. E121 keeps a surface-target hard-gate variant and a ref-target hard-gate variant to separate target-route benefit from terminal-gate benefit.

Measured by:

- `terminal_hard_surface` versus `terminal_hard_ref` on both `box021_029_p2` and `box021_035_p2`;
- contact, lower-body, non-hand support, and visual keyframes.

## Implementation

Add minimal terminal carry gate plumbing:

- `spider/config.py`
  - `terminal_carry_gate_enabled`
  - `terminal_carry_gate_mode`: `soft`, `hard`, or `hard_soft`
  - `terminal_carry_gate_soft_scale`
  - `terminal_carry_gate_pelvis_min_m`
  - `terminal_carry_gate_obj_rot_max_rad`
  - `terminal_carry_gate_nonhand_margin_m`
  - `terminal_carry_gate_hand_near_margin_m`
  - `terminal_carry_gate_hand_min_near_frac`
  - resolved hand/non-hand geoms reuse E120 support geom lists

- `spider/simulators/mjwp.py`
  - compute terminal pelvis error, object rotation error, non-hand support violation, and hand-near violation inside `get_terminal_reward`;
  - soft mode adds a terminal penalty/score to `terminal_rew`;
  - hard/hard_soft mode folds terminal semantic violation into existing `cem_gate_*` info so `spider/optimizers/sampling.py` can reuse the established elite filtering path.

No true multi-stage CEM scheduler is added in E121. That would be a larger optimizer change and is not needed to test the immediate hypothesis.

Expected experiment artifacts:

- builder: `workspace/core4d/scripts/E121/build_terminal_carry_gate_manifest.py`
- variants: `workspace/core4d/scripts/E121/variants.tsv`
- preflight: `workspace/core4d/results/E121/preflight/phaseA_preflight.tsv`
- train: `workspace/core4d/scripts/train/train_E121_terminal_carry_gate.sh`
- remote: `workspace/core4d/scripts/run_E121_remote.sh`
- pull: `workspace/core4d/scripts/pull_E121_remote_results.sh`
- eval: `workspace/core4d/scripts/eval/eval_E121_terminal_carry_gate.py`
- eval shell: `workspace/core4d/scripts/eval/eval_E121_terminal_carry_gate.sh`
- overrides: `examples/config/override/core4d_E121_*.yaml`

## Stage A Workset

Reuse the E120 workset, because it already isolates the main failure and includes one positive companion:

| case | role | split |
|---|---|---|
| `box021_029_p2` | main lower-body-aware upright carry gate | `local-gpu0` |
| `box021_035_p2` | positive companion from E120 surface route | `remote-gpu0` |
| `box021_035_p1` | strict/contact-margin guard | `remote-gpu1` |
| `box004_083_p2` | Box004 contact guard | `local-gpu0` |

## Variants

| ablation | target | terminal gate | intent |
|---|---|---|---|
| `terminal_soft_surface` | E100 external surface | soft | Test terminal carry shaping without elite filtering. |
| `terminal_hard_surface` | E100 external surface | hard_soft | Test whether semantic elite filtering can preserve upright hand-supported carry. |
| `terminal_hard_ref` | `ref_fk` | hard_soft | Separate terminal gate effect from surface-target effect. |

All variants keep E120 support decomposition and E119 posture/bodyguard ingredients. E121 is not a scalar sweep: the primary independent variable is terminal gate semantics.

Initial gate settings:

- `terminal_carry_gate_pelvis_min_m=0.60`
- `terminal_carry_gate_obj_rot_max_rad=0.55`
- `terminal_carry_gate_nonhand_margin_m=0.02`
- `terminal_carry_gate_hand_near_margin_m=0.02`
- `terminal_carry_gate_hand_min_near_frac=0.5`
- hard variants keep `cem_safety_gate_min_valid_frac=0.02` and fallback enabled through existing least-violation behavior.

## Success Gate

E121 Phase A is smoke first (`SMOKE_MAX_NUM_ITERATIONS=4`).

Launch full CEM only if smoke has one of:

1. `box021_029_p2` release candidate; or
2. `box021_029_p2` clear mechanism-level improvement:
   - physics contact > E120 best main value (37.3%) and preferably >= 57%;
   - lower-body/object interference < E120 best main value (18.7%) and preferably <= 5%;
   - non-hand object support < E120 best main value (30.7%) and preferably <= 5%;
   - `pelvis_ok=True` or visual posture clearly better than E120 main rows;
   - deep penetration delta <= +3pp;
   - object tracking remains pass;
   - terminal gate does not rely entirely on fallback (`cem_gate_valid_frac` is not near zero across the useful windows).

Do not launch full if smoke repeats:

- E115/E117: lower-body fixed but contact collapses;
- E118/E119: contact recovered through body/lower-body support or invalid posture;
- E120: hand near-zero present but non-hand support and pelvis failure remain high;
- hard gate starvation: all useful variants require fallback with no metric improvement.

## Verification Before Training

- `python -m py_compile` for touched core files and E121 builder/evaluator.
- `bash -n` for train/remote/pull/eval shell scripts.
- preflight `all_preflight_ok=True`.
- split list returns 6 local, 3 remote-gpu0, 3 remote-gpu1 variants.
- local and remote `eval_E121_terminal_carry_gate.sh smoke --allow-missing` must run without fake metrics.
- `git diff --check` on E121 files and touched core/docs.

## Expected Decision

E121 is still diagnostic. A positive result would justify a full CEM and possibly a future true CEM curriculum. A negative result means terminal filtering is insufficient and the next step should move toward explicit pose-conditioned initialization/curriculum or RL hand-support policy objectives rather than more CEM reward terms.
