# E123 Two-Stage Carry Curriculum Plan

Date: 2026-06-03

## Context

`contact_improvement_plan.md` requires high true contact, low deep penetration, stable posture, and lower-body/object strict pass before RL handoff.

E115-E122 have ruled out the one-stage variants around the hard main case `box021_029_p2`:

- lower-body penalties remove leg contact but also remove hand contact;
- carry corridor and posture guards can improve one metric but not all gates together;
- support decomposition and terminal carry gates produce useful metrics but do not find the main carry mode;
- one-shot arm snap warmstart collapses main contact to 12.0%.

The next hypothesis must change optimizer structure, not another scalar reward/terminal-gate sweep. E123 tests a true two-stage CEM curriculum:

1. Stage 1 optimizes a pose/upright/contact seed using the best prior low-lowerbody main signal.
2. Stage 2 converts the Stage 1 CEM trajectory into a source-axis warmstart and runs the E120/E121 support/carry objective from that seed.

## Claims

### C1: Stage 1 produces reusable trajectory seeds

For each case, E123 Stage 1 must produce a CEM output trajectory and a converted warmstart file with:

- root Stage 1 NPZ, MP4, and outdir `trajectory_mjwp_act.npz`;
- warmstart `qpos_snap` in source/freejoint qpos convention, compatible with the current `warmstart_qpos_path` loader;
- non-empty `snap_mask` covering the raw-contact/carry window;
- conversion metadata that records source shape, stage1 scene-act shape, output shape, and mask window.

### C2: Stage 2 is genuinely conditioned on Stage 1

Every Stage 2 override must set:

- `warmstart_qpos_path` to the Stage 1 converted warmstart;
- `warmstart_update_ctrl_from_qpos=true`;
- the support/terminal metrics from E120/E121 remain enabled.

Stage 2 must not use the E122 arm-only snap warmstart. The independent variable is the Stage 1 optimized trajectory.

### C3: smoke gate determines full CEM

Full CEM is allowed only if main `box021_029_p2` Stage 2 smoke improves over E120/E121/E122:

- physics contact >37.3%, preferably >=57%;
- lower-body/object interference <= E119 stage1 seed level plus tolerance, preferably <=5%;
- non-hand object support <21.3%, preferably <=5%;
- `pelvis_ok=True` or visual posture clearly improves over E119-E122;
- deep penetration does not regress by more than +3pp;
- hard/terminal-gated variants must not rely on near-total fallback.

If main Stage 2 smoke does not pass this gate, do not launch E123 full CEM.

## Mechanism

E123 avoids a large optimizer rewrite. It uses two sequential invocations of the existing CEM runner with a new artifact bridge:

- Stage 1 override inherits E119 `corridor_ref_pose_bodyguard` or `corridor_ref_pose_bodygate`.
- Stage 1 output is `trajectory_mjwp_act.npz`, whose `qpos` is scene-act format `(ticks, ctrl_steps, 42)`.
- A new E123 converter takes one qpos per control tick, converts scene-act object slide/euler back to source freejoint object pos/quaternion, and writes `qpos_snap` with source qpos shape `(T, 43)`.
- Stage 2 overrides inherit E120/E121 support/terminal configs and set the generated Stage 1 warmstart path.

This is a staged optimizer because Stage 2's initial reference/control mean comes from the result of an earlier CEM objective, not from a static IK snap or another reward scalar.

## Workset

Reuse the E120-E122 workset:

| case | purpose | split |
|---|---|---|
| `box021_029_p2` | main upright-carry gate | `local-gpu0` |
| `box021_035_p2` | companion lower-body repair | `remote-gpu0` |
| `box021_035_p1` | strict guard | `remote-gpu1` |
| `box004_083_p2` | Box004 contact guard | `local-gpu0` |

Stage 1 runs one seed per case. Stage 2 runs two variants per case, for 4 Stage 1 rows + 8 Stage 2 rows. The three Box021 cases use `pose_bodyguard_seed`; the Box004 guard uses `guard_terminal_soft_seed` because Box004 was not in the E119 workset.

## Variants

### Stage 1 seeds

| seed | base override | reason |
|---|---|---|
| `pose_bodyguard_seed` | E119 `corridor_ref_pose_bodyguard` | main reached 49.3% contact, 0.0% lower-body, 0 deep, but pelvis failed |
| `guard_terminal_soft_seed` | E121 `terminal_soft_surface` | Box004 guard was not in E119; this keeps Box004 in the two-stage smoke without inventing a new pose-bodyguard recipe |

### Stage 2 objectives

| variant | base override | purpose |
|---|---|---|
| `stage2_support_surface` | E120 `support_surface_direct` | test support decomposition after pose/contact seed |
| `stage2_terminal_soft` | E121 `terminal_soft_surface` | test soft terminal carry gate after pose/contact seed |

Do not include a hard terminal variant in the first E123 smoke. E121/E122 already showed hard variants can starve; hard gating should only be revisited if soft Stage 2 improves main contact/posture.

## Artifacts

- builder/converter: `workspace/core4d/scripts/E123/build_two_stage_curriculum_manifest.py`
- variants: `workspace/core4d/scripts/E123/variants.tsv`
- preflight: `workspace/core4d/results/E123/preflight/phaseA_preflight.tsv`
- stage1 warmstarts: `workspace/core4d/results/E123/warmstarts/*_warmstart_qpos.npz`
- train: `workspace/core4d/scripts/train/train_E123_two_stage_curriculum.sh`
- remote: `workspace/core4d/scripts/run_E123_remote.sh`
- pull: `workspace/core4d/scripts/pull_E123_remote_results.sh`
- eval: `workspace/core4d/scripts/eval/eval_E123_two_stage_curriculum.py`
- eval shell: `workspace/core4d/scripts/eval/eval_E123_two_stage_curriculum.sh`
- overrides: `examples/config/override/core4d_E123_*.yaml`

## Execution

Phase A is smoke first:

```bash
python workspace/core4d/scripts/E123/build_two_stage_curriculum_manifest.py
SMOKE_MAX_NUM_ITERATIONS=4 bash workspace/core4d/scripts/train/train_E123_two_stage_curriculum.sh local smoke 0
ssh spider-remote 'cd /home/xiayb/pHRI_workspace/spider && SMOKE_MAX_NUM_ITERATIONS=4 bash workspace/core4d/scripts/run_E123_remote.sh smoke'
bash workspace/core4d/scripts/pull_E123_remote_results.sh smoke
bash workspace/core4d/scripts/eval/eval_E123_two_stage_curriculum.sh smoke
```

The train script must run Stage 1 before Stage 2 for each split and must rebuild missing warmstarts after Stage 1 completes. Full CEM uses the same local 1 GPU + remote 2 GPU layout only if smoke passes C3.

## Static Checks

- `python -m py_compile` for the E123 builder/converter/evaluator.
- `bash -n` for train/remote/pull/eval shell scripts.
- `eval_E123_two_stage_curriculum.sh smoke --allow-missing` before CEM should report Stage 2 missing outputs without fake metrics.
- `git diff --check` on E123 files and generated overrides.

## Stop Rule

If E123 Stage 2 does not improve main `box021_029_p2` over E120/E121/E122 smoke, stop the one-stage and two-stage CEM line. The next step should be either a deeper optimizer change with explicit stage-local horizons or an RL hand-support objective initialized from a verified positive seed.
