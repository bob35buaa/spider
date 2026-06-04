# E132 Holosoma MotionLoader Object-Contact Probe Results

Date: 2026-06-03

## Scope

E132 followed `workspace/core4d/plan/141_E132_holosoma_motionloader_object_contact_probe_plan.md`.
It did not launch IsaacSim stepping, CEM, PPO, Holosoma training, checkpoint
creation, or remote jobs.

E132 ran Holosoma's real `MotionLoader` under the same `hssim` dependency setup
used by E128, but on CPU only. It compared the original E126 paired exports
against the E131 proxy-contact exports.

Subagent `Anscombe` recommended a stronger future E128-style no-debug env probe
for `motion_command.ref_object_contact`. E132 intentionally stops one layer
earlier: it proves the `MotionLoader` contract before spending IsaacSim startup
time.

## Outputs

```text
workspace/core4d/results/E132/holosoma_motionloader_object_contact_probe/
  e132_motionloader_object_contact_manifest.tsv
  e132_motionloader_object_contact_summary.json
  e132_motionloader_object_contact_summary.md
```

## Result

| metric | value |
|---|---:|
| probe rows | 4 |
| runtime contract pass rows | 4 |
| E126 missing-mask rows | 2 |
| E131 object-contact rows | 2 |
| structural ref-mask runtime-ready rows | 2 |
| semantic ref-mask ready rows | 0 |
| RL-ready rows | 0 |
| training launched | false |
| CEM launched | false |
| status | pass |

Loader contrast:

| source | case | `has_object_contact` | shape | left active | right active | both active |
|---|---|---|---|---:|---:|---:|
| E126 | `box021_035_p1` | false | 214x2 | 0.0% | 0.0% | 0.0% |
| E126 | `box021_035_p2` | false | 214x2 | 0.0% | 0.0% | 0.0% |
| E131 | `box021_035_p1` | true | 214x2 | 0.0% | 4.2% | 0.0% |
| E131 | `box021_035_p2` | true | 214x2 | 27.6% | 47.2% | 22.9% |

## Interpretation

E132 proves the structural loader distinction:

- E126 paired exports are still missing-mask baselines. `MotionLoader` sets
  `has_object_contact=false` and exposes an all-false `(T,2)` tensor.
- E131 proxy exports pass the runtime loader contract. `MotionLoader` sets
  `has_object_contact=true` and reads the saved nonzero `(214,2)` bool masks.

This is not semantic contact evidence. E131 masks remain geometry proxies from
actor rubber-hand distance to the Box021 OBB. E132 does not prove raw/trimmed
contact labels, simulator contact forces, policy quality, or main
`box021_029_p2` release readiness.

The next valid bounded branch is E133: reuse the E128 no-debug replay probe with
E131 proxy exports and log `motion_command.ref_object_contact` stats after env
startup. That would prove the command/reward reference-mask plumbing layer while
still avoiding PPO.

## Verification

- `.venv/bin/python -m py_compile workspace/core4d/scripts/E132/probe_holosoma_motionloader_object_contact.py`
- `bash -n workspace/core4d/scripts/eval/eval_E132_holosoma_motionloader_object_contact_probe.sh`
- `bash workspace/core4d/scripts/eval/eval_E132_holosoma_motionloader_object_contact_probe.sh`
