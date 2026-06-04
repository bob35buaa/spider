# E133 Holosoma Ref-Object-Contact Env Probe Results

Date: 2026-06-03

## Scope

E133 followed `workspace/core4d/plan/142_E133_holosoma_ref_object_contact_env_probe_plan.md`.
It reuses the E128 no-debug Holosoma/IsaacSim startup pattern and the E131
geometry-proxy `object_contact` exports to check whether the env runtime
`motion_command.ref_object_contact` view can observe those masks during bounded
stepping.

E133 did not launch CEM, PPO, Holosoma training, checkpoint creation, or remote
jobs. The user-requested local+remote multi-GPU rule is therefore not triggered
by this bounded probe.

## Outputs

```text
workspace/core4d/results/E133/holosoma_ref_object_contact_env_probe/
  e133_ref_object_contact_manifest.tsv
  e133_ref_object_contact_summary.json
  e133_ref_object_contact_summary.md
  logs/box021_035_p1_ref_object_contact.log
  logs/box021_035_p2_ref_object_contact.log
```

## Result

| metric | value |
|---|---:|
| probe rows | 2 |
| startup pass rows | 2 |
| ref-positive rows | 1 |
| p2 ref positive | true |
| structural ref-mask runtime-ready rows | 2 |
| semantic ref-mask ready rows | 0 |
| RL-ready rows | 0 |
| training launched | false |
| CEM launched | false |
| remote jobs launched | false |
| status | pass |

Runtime rows:

| case | startup | `has_object_contact` | full motion total | runtime ref total | runtime ref either | observed step max |
|---|---|---|---:|---:|---:|---:|
| `box021_035_p1` | pass | true | 9 | 0 | 0 | 204 |
| `box021_035_p2` | pass | true | 160 | 74 | 54 | 204 |

## Claims Verification

1. E131 proxy exports load with object, partner, and `has_object_contact=True`:
   verified for both rows.
2. Full-motion `object_contact` remains nonzero: verified for both rows
   (`9` and `160` total active hand-channel samples).
3. Bounded stepping can expose nonzero `motion_command.ref_object_contact`:
   verified on the p2 positive guard (`ref_contact_total=74`,
   `ref_either_active=54`). The p1 row remains `zero_ref_contact`, consistent
   with its weak proxy signal and is diagnostic rather than a release claim.
4. E131 masks remain geometry proxies: verified by provenance. E133 does not
   prove raw/trimmed semantic contact, simulator contact forces, reward value
   quality, PPO readiness, or main `box021_029_p2` release readiness.

## Visualization

No video artifact was produced. This run is a headless IsaacSim tensor plumbing
probe; evidence comes from the fixed eval script, the manifest, and the
`E133_PROBE_LOADED` / `E133_PROBE_SUMMARY` / `E133_PROBE_DONE` log markers.

## Verification

- `.venv/bin/python -m py_compile workspace/core4d/scripts/E133/holosoma_ref_object_contact_no_debug_probe.py`
- `bash -n workspace/core4d/scripts/eval/eval_E133_holosoma_ref_object_contact_env_probe.sh`
- `bash workspace/core4d/scripts/eval/eval_E133_holosoma_ref_object_contact_env_probe.sh`

## Interpretation

E133 closes the structural runtime command-plumbing gap between E132's
`MotionLoader` contract and Holosoma env stepping: E131 proxy masks can reach
`motion_command.ref_object_contact` in the bounded no-debug runtime path.

This is still not enough to launch PPO or claim main-case readiness. The next
valid branch is to recover or construct semantic raw/trimmed per-hand contact
labels for the relevant source windows, then repeat the loader and env runtime
contract checks before any reward-side RL run.
