# E120 Object Support Decomposition Results

Date: 2026-06-03

## Goal

E120 follows the E119 negative diagnostic and tests an explicit object-support decomposition instead of another scalar sweep.

The new objective separates:

- desired hand-supported object contact during raw-contact windows
- undesired non-hand object support from torso, pelvis, upper arms, and lower body

Stage A used 4 cases x 3 variants:

- `box021_029_p2`: main upright-carry gate
- `box021_035_p2`: companion lower-body repair signal
- `box021_035_p1`: strict guard
- `box004_083_p2`: Box004 contact guard

Variants:

- `support_ref_direct`
- `support_ref_staged`
- `support_surface_direct`

## Implementation

Core reward additions:

- `spider/config.py`: added `hand_support_*` and `nonhand_support_penalty_*` config fields, geometry resolution, and logging.
- `spider/simulators/mjwp.py`: added hand near-zero support reward, non-hand support penalty, support gates, and `info` outputs.

The new scales default to `0.0`, so historical configs are unchanged unless the override enables the terms.

New E120 artifacts:

- Plan: `workspace/core4d/plan/129_E120_object_support_decomposition_plan.md`
- Builder: `workspace/core4d/scripts/E120/build_object_support_decomposition_manifest.py`
- Manifest: `workspace/core4d/scripts/E120/variants.tsv`
- Preflight: `workspace/core4d/results/E120/preflight/phaseA_preflight.tsv`
- Train script: `workspace/core4d/scripts/train/train_E120_object_support_decomp.sh`
- Remote runner: `workspace/core4d/scripts/run_E120_remote.sh`
- Pull script: `workspace/core4d/scripts/pull_E120_remote_results.sh`
- Evaluator: `workspace/core4d/scripts/eval/eval_E120_object_support_decomp.py`
- Eval entrypoint: `workspace/core4d/scripts/eval/eval_E120_object_support_decomp.sh`
- Overrides: `examples/config/override/core4d_E120_*.yaml`

Preflight generated 12 variants. All required task, mask, baseline, E113 hold-result, E100 surface-target, and support-geometry checks passed. Split assignment was local 6, remote GPU0 3, remote GPU1 3.

Static checks passed:

```bash
python -m py_compile spider/config.py spider/simulators/mjwp.py workspace/core4d/scripts/E120/build_object_support_decomposition_manifest.py workspace/core4d/scripts/eval/eval_E120_object_support_decomp.py
bash -n workspace/core4d/scripts/train/train_E120_object_support_decomp.sh workspace/core4d/scripts/run_E120_remote.sh workspace/core4d/scripts/pull_E120_remote_results.sh workspace/core4d/scripts/eval/eval_E120_object_support_decomp.sh
bash workspace/core4d/scripts/eval/eval_E120_object_support_decomp.sh smoke --allow-missing
git diff --check -- spider/config.py spider/simulators/mjwp.py workspace/core4d/plan/129_E120_object_support_decomposition_plan.md workspace/core4d/scripts/E120/build_object_support_decomposition_manifest.py workspace/core4d/scripts/train/train_E120_object_support_decomp.sh workspace/core4d/scripts/run_E120_remote.sh workspace/core4d/scripts/pull_E120_remote_results.sh workspace/core4d/scripts/eval/eval_E120_object_support_decomp.py workspace/core4d/scripts/eval/eval_E120_object_support_decomp.sh workspace/core4d/progress.md examples/config/override/core4d_E120_*.yaml
```

Remote static checks also passed after rsyncing E120 core code, scripts, overrides, preflight, and evaluator helpers to `spider-remote:/home/xiayb/pHRI_workspace/spider`.

## Execution

Smoke CEM used `SMOKE_MAX_NUM_ITERATIONS=4`.

Execution layout:

| split | machine | rows |
|---|---|---:|
| `local-gpu0` | local GPU0 | 6 |
| `remote-gpu0` | `spider-remote` GPU0 | 3 |
| `remote-gpu1` | `spider-remote` GPU1 | 3 |

Remote execution used tmux session `E120_smoke_094325`. No unrelated GPU processes were stopped. The remote GPU0 `box021_035_p2/support_surface_direct` row was slow but healthy, so it was allowed to finish instead of duplicate-running the active variant.

Results were pulled back with:

```bash
bash workspace/core4d/scripts/pull_E120_remote_results.sh smoke
```

Smoke outputs after local + remote pull:

| artifact | count |
|---|---:|
| root NPZ | 12 |
| MP4 | 12 |
| outdir `trajectory_mjwp_act.npz` | 12 |
| missing variants | 0 |

## Quantitative Result

Evaluation command:

```bash
bash workspace/core4d/scripts/eval/eval_E120_object_support_decomp.sh smoke
```

Evaluator output:

- `workspace/core4d/results/E120/cem/smoke/smoke_eval_summary.md`
- `workspace/core4d/results/E120/cem/smoke/pareto_decisions.tsv`
- `workspace/core4d/results/E120/cem/smoke/release_candidates.tsv`
- `workspace/core4d/results/E120/cem/smoke/guard_failures.tsv`
- `workspace/core4d/results/E120/cem/smoke/support_decomp_timeseries_*.csv`

Summary:

- evaluated variants: 12
- method rows: 36
- missing variants: 0
- release candidates: 0

Decision counts:

| decision | count |
|---|---:|
| `lowerbody_fixed_contact_fail` | 8 |
| `support_decomp_fail` | 4 |

Main case:

| case | variant | physics contact | lower-body | non-hand support | non-hand interference | hand near-zero | deep delta | pelvis ok | decision |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| `box021_029_p2` | `support_ref_direct` | 29.3% | 18.7% | 30.7% | 18.7% | 41.3% | +0.0pp | false | `support_decomp_fail` |
| `box021_029_p2` | `support_ref_staged` | 36.0% | 32.0% | 36.0% | 32.0% | 41.3% | +0.0pp | false | `support_decomp_fail` |
| `box021_029_p2` | `support_surface_direct` | 37.3% | 22.7% | 36.0% | 22.7% | 68.0% | +0.0pp | false | `support_decomp_fail` |

Companion and guard observations:

| case | best observed signal | blocker |
|---|---|---|
| `box021_035_p2` | `support_surface_direct` reaches 72.9% physics contact, lower-body 0.0%, non-hand support 0.0%, hand near-zero 78.2% | legacy strict/contact gate still reports `lowerbody_fixed_contact_fail`; this is not the main case |
| `box021_035_p1` | `support_surface_direct` reaches 69.0% physics contact with non-hand support 3.1% | still strict-fail, with small non-hand interference 1.6% |
| `box004_083_p2` | all variants keep lower-body and non-hand support at 0.0% | contact is not improved over the E113 guard baseline; all remain `lowerbody_fixed_contact_fail` |

## Visual Check

Representative keyframes and contact sheet:

- `workspace/core4d/results/E120/cem/smoke/e120_visual_qc_contact_sheet.jpg`
- `workspace/core4d/results/E120/cem/smoke/keyframes/E120_box021_029_p2_support_ref_staged/f100.jpg`
- `workspace/core4d/results/E120/cem/smoke/keyframes/E120_box021_029_p2_support_surface_direct/f70.jpg`
- `workspace/core4d/results/E120/cem/smoke/keyframes/E120_box021_035_p2_support_surface_direct/f100.jpg`

Visual observations:

- Main `box021_029_p2` rows still collapse toward lower-body or near-body object support; this matches the 30.7-36.0% non-hand support and `pelvis_ok=False`.
- `support_surface_direct` improves hand near-zero on the main case, but the object remains supported by non-hand geometry too often and contact remains far below the full gate.
- Companion `box021_035_p2/support_surface_direct` is visually cleaner and matches the 72.9% contact / 0.0% non-hand support signal, but it is not enough to release the main carry case.

## Decision

Do not launch E120 full CEM.

Reason:

- Smoke has 0 release candidates.
- The main `box021_029_p2` gate requires physics contact >=57%, lower-body <=5%, non-hand support <=5%, deep delta <=+3pp, and posture/object pass.
- All main rows fail contact, lower-body, non-hand support, and pelvis posture. The best main physics contact is only 37.3%, and non-hand support remains 30.7-36.0%.
- Deep penetration did not regress, so the failure is not from hand-object penetration; it is specifically an object-support/posture gaming failure.

## Interpretation

E120 validates the new evaluator and reward plumbing, but the current penalty/reward formulation is still not strong enough on the hard `box021_029_p2` carry state. The positive companion signal suggests the support decomposition is directionally useful on easier Box021 posture, but main-case CEM still finds a lower-body/near-body support strategy instead of a stable upright carry.

Next iteration should avoid simply increasing the non-hand penalty. More promising changes are:

1. staged carry curriculum with a posture/pose anchor before contact optimization
2. stronger object orientation/anti-tip coupling so body support cannot satisfy contact while the object tips or drifts
3. separate hard replay gate or terminal gate for pelvis/upright carry posture on `box021_029_p2`
4. keep `nonhand_object_support_frac` as a mandatory release metric for future Box021 experiments
