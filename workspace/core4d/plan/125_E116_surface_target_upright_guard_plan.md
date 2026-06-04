# E116 Surface Target + Upright Guard Plan

Date: 2026-06-03

## Context

E115 showed that a naive lower-body/object penalty is not enough. On the main diagnostic case `box021_029_p2`, `leg_penalty_s2/s4` removed lower-body interference but collapsed hand-object physics contact from 65.3% to 12.0%. Increasing contact gain restored some contact, but reintroduced lower-body interference and deep penetration.

This means E116 must change the contact formulation rather than repeat stronger leg penalties. The most conservative next step is to reuse existing infrastructure:

- E113 hold-band contact mask and near-field objective;
- E100 external fingertip/surface target in object-local coordinates;
- E084B-style upright / body tracking / ctrl-ref guard;
- explicit neutralization of inherited safety/upright knobs before each ablation opens only the intended controls.

## Claim

`ref_fk + hold_band` is not enough for E113 release candidates because it often preserves near contact without semantic surface alignment. Replacing the dynamic target with E100 fingertip/object-local surface targets, while adding an upright/ctrl guard, should reduce the tendency to solve lower-body safety by abandoning the object.

## Cases

Use the same diagnostic/guard set as E115 so results are directly comparable:

| case | role | reason |
|---|---|---|
| `box021_029_p2` | main | E113 gained +20.0pp physics contact but failed lower-body strict; E115 showed the contact/lower-body trade-off clearly. |
| `box021_035_p2` | companion | lower-body repair case; E115 `leg_penalty_s4` reduced lower-body interference to 4.5% but did not release. |
| `box021_035_p1` | strict guard | E113 strict WORK but contact margin was not enough; E115 penalties regressed physics contact. |
| `box004_082_p1` | strict guard | E113 strict WORK; protects against penetration/contact regressions. |
| `box004_083_p2` | strict guard | E113 strict WORK; E115 penalties collapsed physics contact. |

All five cases have E100 external target artifacts:

```text
workspace/core4d/results/E100/fingertip_targets/<case>/spider_contact_target_object_local.npz
```

## Variants

Each case gets three variants:

| variant | change | purpose |
|---|---|---|
| `surface_light` | E113 hold-band + E100 external surface target, `contact_hdmi_gain=3.0` | Test whether semantic contact target with lower gain improves contact/lower-body balance. |
| `surface_light_safety` | `surface_light` + light upper-body/object and hand deep-penetration penalties | Keep surface contact from becoming press-through contact, without adding lower-body penalty. |
| `surface_upright_safety` | `surface_light_safety` + E084B-style `ctrl_ref_guard`, `task_body_rew`, `stability_penalty`, weaker object tracking | Prevent crouch-under-object or pose-collapse compensation while keeping contact target. |

No variant uses E115's uniform `leg_object_penalty_scale=2/4`. E116 intentionally avoids lower-body penalty sweeps because E115 already showed that uniform lower-body penalty can fix interference by abandoning contact.

Each generated override writes inherited `robot_object_penalty_*`, `hand_object_deep_penalty_*`, `hand_floor_penalty_*`, `cem_safety_gate_*`, `ctrl_ref_guard_scale`, `stability_penalty_scale`, `task_body_rew_scale`, and `task_obj_use_exp` exactly once, using neutral values unless the ablation explicitly enables that group. This is required because several E113 source overrides inherit earlier safety gates, and otherwise `surface_light` is not a clean surface-target ablation.

## Implementation

Create:

- `workspace/core4d/scripts/E116/build_surface_target_upright_manifest.py`
- `workspace/core4d/scripts/E116/variants.tsv`
- `workspace/core4d/results/E116/preflight/phaseA_preflight.tsv`
- `examples/config/override/core4d_E116_*.yaml`
- `workspace/core4d/scripts/train/train_E116_surface_target_upright.sh`
- `workspace/core4d/scripts/run_E116_remote.sh`
- `workspace/core4d/scripts/pull_E116_remote_results.sh`
- `workspace/core4d/scripts/eval/eval_E116_surface_target_upright.py`
- `workspace/core4d/scripts/eval/eval_E116_surface_target_upright.sh`

The evaluator can reuse the E115 comparison structure: baseline vs E113 hold vs E116 test, with release decision gated by strict/contact/penetration/lower-body/object/pelvis metrics.

## Split

Use the same split shape as E115:

| split | rows |
|---|---:|
| local-gpu0 | 6 (`box004_082_p1`, `box004_083_p2`) |
| remote-gpu0 | 3 (`box021_029_p2`) |
| remote-gpu1 | 6 (`box021_035_p2`, `box021_035_p1`) |

## Success Criteria

Smoke success:

- 15/15 root NPZ, MP4, and outdir trajectories;
- evaluator reports 0 missing variants;
- no infrastructure errors or invalid external-target path issues.

Method success for the main case:

- `box021_029_p2` physics contact does not collapse: test physics contact >= E113 - 10pp;
- lower-body interference improves vs E113 by at least 3pp, or passes strict <= 5%;
- deep penetration increase <= 3pp;
- pelvis/object gates remain acceptable.

Guard success:

- strict guard rows must not show systematic contact collapse;
- no guard should introduce large deep penetration or pelvis collapse;
- any candidate full CEM variant must pass at least one strict guard and not regress both Box004 guards.

Full CEM rule:

Do not launch full CEM merely because smoke runs. Full CEM is warranted only if smoke produces at least one main/companion candidate with a better E113/E115 Pareto trade-off and no severe guard regression.

## Commands

Build:

```bash
python3 workspace/core4d/scripts/E116/build_surface_target_upright_manifest.py
```

Static checks:

```bash
python3 -m py_compile workspace/core4d/scripts/E116/build_surface_target_upright_manifest.py workspace/core4d/scripts/eval/eval_E116_surface_target_upright.py
bash -n workspace/core4d/scripts/train/train_E116_surface_target_upright.sh
bash -n workspace/core4d/scripts/run_E116_remote.sh
bash -n workspace/core4d/scripts/pull_E116_remote_results.sh
bash -n workspace/core4d/scripts/eval/eval_E116_surface_target_upright.sh
```

Smoke:

```bash
SMOKE_MAX_NUM_ITERATIONS=4 bash workspace/core4d/scripts/train/train_E116_surface_target_upright.sh local smoke 0
ssh spider-remote 'cd /home/xiayb/pHRI_workspace/spider && SMOKE_MAX_NUM_ITERATIONS=4 bash workspace/core4d/scripts/run_E116_remote.sh smoke'
bash workspace/core4d/scripts/pull_E116_remote_results.sh smoke
bash workspace/core4d/scripts/eval/eval_E116_surface_target_upright.sh smoke
```
