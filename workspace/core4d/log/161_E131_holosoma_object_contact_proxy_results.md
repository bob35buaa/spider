# E131 Holosoma Object-Contact Proxy Results

Date: 2026-06-03

## Scope

E131 followed `workspace/core4d/plan/140_E131_holosoma_object_contact_proxy_plan.md`.
It did not launch CEM, PPO, Holosoma training, checkpoint creation, or remote jobs.

E131 addresses the E130 blocker that E126 paired exports lack `object_contact`.
The goal was structural contract closure for bounded loader/ref-mask debugging,
not raw-contact labeling, release, or RL readiness.

Subagent `Euler` independently audited the E126/E125/E111 chain and found no
propagated real raw/trimmed per-hand contact mask for `box021_035_p1/p2`.
E111 has inventory rows for `box021_20231011_035_p1/p2`, but raw contact is
`not_run`, contact-mask fields are empty, and the only E111 raw-contact proxy
artifact is for a different Box004 sequence. E131 therefore labels its masks as
geometry proxies.

## Outputs

```text
workspace/core4d/results/E131/holosoma_object_contact_proxy/
  exports/E126_box021_035_p1_with_partner_box021_035_p2_object_contact_proxy5cm.npz
  exports/E126_box021_035_p2_with_partner_box021_035_p1_object_contact_proxy5cm.npz
  e131_object_contact_proxy_manifest.tsv
  e131_object_contact_proxy_summary.json
  e131_object_contact_proxy_summary.md
```

## Result

| metric | value |
|---|---:|
| proxy export rows | 2 |
| `object_contact` shape pass | 2/2 |
| structural ref-mask ready rows | 2 |
| semantic ref-mask ready rows | 0 |
| RL-ready rows | 0 |
| training launched | false |
| CEM launched | false |
| status | pass |

Mask provenance:

- `mask_source=actor_rubber_hand_box021_surface_proxy_5cm`
- actor hand bodies: `left_rubber_hand_link`, `right_rubber_hand_link`
- object geometry: Box021 OBB half extents `[0.195495, 0.15155, 0.25004]`
- threshold: signed surface distance `<= 0.05m`
- E126 source exports were not modified.

Proxy contact summary:

| source export | left active | right active | both active | left run | right run | both run |
|---|---:|---:|---:|---:|---:|---:|
| p1 with p2 partner | 0.0% | 4.2% | 0.0% | 0 | 9 | 0 |
| p2 with p1 partner | 27.6% | 47.2% | 22.9% | 27 | 39 | 26 |

## Interpretation

E131 creates Holosoma-compatible `object_contact (T,2)` artifacts for the two
E126 fragment exports. This closes the loader shape contract and allows a future
bounded no-PPO ref-mask wiring smoke to distinguish "mask loaded" from the E130
all-false missing-mask case.

It does not make the fragments semantically ref-mask-ready. The masks are
method-side geometry proxies from actor rubber-hand distance to the object, not
raw-contact labels, trimmed contact masks, human annotation, or simulator force
evidence. The p1 export also has almost no actor-side proxy contact, so even as
debug data it is weak for two-hand ref-mask behavior.

The next valid step is a bounded no-PPO Holosoma replay/loader probe using the
E131 exports to confirm `MotionLoader.has_object_contact=True` and reward terms
read nonzero masks. Do not start PPO or use E131 as main `box021_029_p2` release
evidence.

## Verification

- `.venv/bin/python -m py_compile workspace/core4d/scripts/E131/add_holosoma_object_contact_proxy.py`
- `bash -n workspace/core4d/scripts/eval/eval_E131_holosoma_object_contact_proxy.sh`
- `bash workspace/core4d/scripts/eval/eval_E131_holosoma_object_contact_proxy.sh`
- direct NPZ inspection confirmed `object_contact.shape == (214, 2)` and dtype `bool`
