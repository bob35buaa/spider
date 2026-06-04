# E116 Surface Target + Upright Guard Results

Date: 2026-06-03

## Goal

E115 showed that uniform `leg_object_penalty` can remove lower-body/object interference by abandoning hand-object contact. E116 tested whether the existing, already-supported mechanisms can recover a better Pareto point:

- E100 external fingertip/object-local surface targets;
- E113 hold-band contact mask;
- light upper-body/object and hand deep-penetration penalties;
- E084B-style upright/body/ctrl guard.

E116 intentionally did not repeat E115's uniform lower-body penalty sweep because that mechanism was already a negative diagnostic.

## Implementation

Created:

- `workspace/core4d/plan/125_E116_surface_target_upright_guard_plan.md`
- `workspace/core4d/scripts/E116/build_surface_target_upright_manifest.py`
- `workspace/core4d/scripts/E116/variants.tsv`
- `workspace/core4d/results/E116/preflight/phaseA_preflight.tsv`
- `examples/config/override/core4d_E116_*.yaml`
- `workspace/core4d/scripts/train/train_E116_surface_target_upright.sh`
- `workspace/core4d/scripts/run_E116_remote.sh`
- `workspace/core4d/scripts/pull_E116_remote_results.sh`
- `workspace/core4d/scripts/eval/eval_E116_surface_target_upright.py`
- `workspace/core4d/scripts/eval/eval_E116_surface_target_upright.sh`

Cases and split:

| split | cases | variants |
|---|---|---:|
| local-gpu0 | `box004_082_p1`, `box004_083_p2` | 6 |
| remote-gpu0 | `box021_029_p2` | 3 |
| remote-gpu1 | `box021_035_p2`, `box021_035_p1` | 6 |

Variants:

| ablation | intended control |
|---|---|
| `surface_light` | E113 hold-band + E100 external target, `contact_hdmi_gain=3.0` |
| `surface_light_safety` | `surface_light` + light upper-body/object, hand deep-penetration, hand-floor penalties |
| `surface_upright_safety` | `surface_light_safety` + ctrl-ref guard, body tracking, stability, exp object tracking |

During pre-smoke audit, `surface_light` was found to inherit earlier safety/upright knobs from E113 source overrides. The builder now writes every inherited safety/upright field exactly once, with neutral values unless that ablation explicitly enables it. This keeps the three ablations interpretable and avoids Hydra/OmegaConf duplicate-key failures.

The train script also now skips a variant when root NPZ, MP4, and outdir trajectory are all present, so interrupted smoke runs can resume without redoing completed rows.

The shared E115/E116 evaluator now rewrites `*_missing_outputs.csv` even when there are no missing rows, so stale `--allow-missing` tables cannot be mistaken for current missing outputs.

## Verification

Static checks:

```bash
python3 -m py_compile workspace/core4d/scripts/E116/build_surface_target_upright_manifest.py workspace/core4d/scripts/eval/eval_E116_surface_target_upright.py
bash -n workspace/core4d/scripts/train/train_E116_surface_target_upright.sh
bash -n workspace/core4d/scripts/run_E116_remote.sh
bash -n workspace/core4d/scripts/pull_E116_remote_results.sh
bash -n workspace/core4d/scripts/eval/eval_E116_surface_target_upright.sh
bash workspace/core4d/scripts/eval/eval_E116_surface_target_upright.sh smoke --allow-missing
```

Smoke execution:

```bash
SMOKE_MAX_NUM_ITERATIONS=4 bash workspace/core4d/scripts/train/train_E116_surface_target_upright.sh local smoke 0
ssh spider-remote 'cd /home/xiayb/pHRI_workspace/spider && SMOKE_MAX_NUM_ITERATIONS=4 bash workspace/core4d/scripts/run_E116_remote.sh smoke'
bash workspace/core4d/scripts/pull_E116_remote_results.sh smoke
bash workspace/core4d/scripts/eval/eval_E116_surface_target_upright.sh smoke
```

Artifacts after local + remote pull:

| artifact | count |
|---|---:|
| root NPZ | 15 |
| MP4 | 15 |
| outdir `trajectory_mjwp_act.npz` | 15 |
| missing variants | 0 |

Runtime config checks confirmed:

- `surface_light`: external target loaded, inherited safety/upright knobs neutralized.
- `surface_light_safety`: safety penalties enabled, upright/ctrl guard disabled, CEM safety gate disabled.
- `surface_upright_safety`: safety penalties plus `ctrl_ref_guard_scale=0.8`, `stability_penalty_scale=1.0`, `task_body_rew_scale=1.5`, `task_obj_use_exp=true`.

No local or remote E116 training processes remained after completion.

## Results

Evaluator output:

- `workspace/core4d/results/E116/cem/smoke/smoke_eval_summary.md`
- `workspace/core4d/results/E116/cem/smoke/pareto_decisions.tsv`
- `workspace/core4d/results/E116/cem/smoke/release_candidates.tsv`

Summary:

| case | best useful signal | blocker |
|---|---|---|
| `box021_029_p2` | `surface_upright_safety` fixes lower-body 8.0% -> 0.0% and keeps physics contact at 46.7% vs baseline 45.3% | still below E113 65.3%, deep penetration +4.0pp, strict fail |
| `box021_035_p2` | `surface_light`/`surface_light_safety` reach 80-81% physics contact | lower-body remains 6.8% > 5% |
| `box021_035_p1` | all three improve contact to 79-85% | penetration or lower-body/strict guard fail |
| `box004_082_p1` | `surface_upright_safety` gives 55.0% physics contact with small deep-pen increase | strict still fail; light/safety either penetration or lower-body regression |
| `box004_083_p2` | `surface_upright_safety` improves physics contact to 54.3% and keeps lower-body 0.0% | strict still fail; light collapses contact |

Decision counts from summary rows:

- `lowerbody_fixed_contact_fail`: 6
- `contact_good_lowerbody_fail`: 4
- `penetration_fail`: 2
- `review`: 3
- release candidates: 0

Representative keyframes:

- `workspace/core4d/results/E116/cem/smoke/keyframes/E116_box021_029_p2_surface_upright_safety/f70.jpg`: main case, lower-body interference visually reduced, but hands are not maintaining the object contact.
- `workspace/core4d/results/E116/cem/smoke/keyframes/E116_box021_035_p2_surface_light/f70.jpg`: high hand contact but lower-body/foot involvement remains visible.
- `workspace/core4d/results/E116/cem/smoke/keyframes/E116_box004_083_p2_surface_upright_safety/f70.jpg`: more stable posture/contact, but not enough to satisfy release gates.

## Decision

Do not launch E116 full CEM with the current configuration.

Reason:

- Smoke is infrastructurally clean, but no row is a release candidate.
- The main case `box021_029_p2` shows the core trade-off remains: upright/safety can restore posture and remove lower-body interference, but still loses too much E113 hand contact and adds deep penetration.
- Companion/guard cases show that external surface target can raise contact, but often by accepting lower-body interference, penetration, or strict guard failure.

## Interpretation

E116 is a useful diagnostic, not a final recipe.

The surface target is not sufficient by itself because the optimizer can still satisfy target proximity through non-semantic posture choices: hands may approach/press the object, while body pose or object support remains physically invalid. The upright guard helps posture but can also pull hands away from sustained contact on the main case.

This suggests the next method direction should add phase/state structure rather than another scalar penalty sweep:

- phase-gated lower-body/object penalty active only when the object is in a carry/support phase;
- explicit carry corridor or object-support corridor tying object pose, pelvis/torso pose, and hand contact together;
- anti-tip / object orientation gate so contact cannot be recovered by tilting or dumping the object;
- contact target acceptance that distinguishes hand-surface contact from lower-body-assisted object support;
- full-body pose prior or learned policy prior for crouch/carry phases if pure CEM keeps exploiting posture gaps.

Data-side follow-up remains the same as the contact improvement plan: keep E100/E111-style object-local targets, raw contact masks, active-frame/run-length evidence, and S6 contact-alignment metrics propagated through manifests so future phase-aware rewards can be trained/evaluated against the same evidence.
