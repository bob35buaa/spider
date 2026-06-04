# E137 E107 Semantic Object-Contact Export Preflight Results

Date: 2026-06-03

## Scope

E137 followed
`workspace/core4d/plan/146_E137_e107_semantic_object_contact_export_preflight_plan.md`.
It writes isolated E107 qpos-style NPZ copies with semantic `object_contact`
arrays derived from E135 v3 raw-contact masks and E136-proven mappings.

E137 did not write into the original E107 source directories. It did not write
E126/E131 semantic masks. It did not run Holosoma `MotionLoader`, IsaacSim,
CEM, PPO, training, checkpoints, or remote jobs.

## Outputs

```text
workspace/core4d/results/E137/e107_semantic_object_contact_export/
  e137_e107_semantic_object_contact_manifest.tsv
  e137_e107_semantic_object_contact_summary.json
  e137_e107_semantic_object_contact_summary.md
  exports/E107_retargeted_untrimmed/*/*_e135_semantic_object_contact_3cm.npz
  exports/E107_trimmed/*/*_e135_semantic_object_contact_3cm.npz
```

## Result

| metric | value |
|---|---:|
| semantic export NPZ files | 8 |
| retargeted-untrimmed exports | 4 |
| trimmed exports | 4 |
| E126/E131 exports written | 0 |
| default `object_contact` label | 3cm |
| MotionLoader runtime probe launched | false |
| training launched | false |
| CEM launched | false |
| remote jobs launched | false |
| status | pass |

Export table:

| case | source | frames | mapping | object contact | 3cm L/R/both | 5cm L/R/both | status |
|---|---|---:|---|---|---|---|---|
| `box021_029_p1` | `E107_retargeted_untrimmed` | 134 | `direct_raw_axis` | `134x2` | 0.358209/0.328358/0.328358 | 0.358209/0.335821/0.335821 | `pass` |
| `box021_029_p1` | `E107_trimmed` | 71 | `trim_window_slice` | `71x2` | 0.676056/0.619718/0.619718 | 0.676056/0.633803/0.633803 | `pass` |
| `box021_029_p2` | `E107_retargeted_untrimmed` | 134 | `direct_raw_axis` | `134x2` | 0.380597/0.410448/0.380597 | 0.380597/0.410448/0.380597 | `pass` |
| `box021_029_p2` | `E107_trimmed` | 75 | `trim_window_slice` | `75x2` | 0.680000/0.733333/0.680000 | 0.680000/0.733333/0.680000 | `pass` |
| `box021_035_p1` | `E107_retargeted_untrimmed` | 182 | `direct_raw_axis` | `182x2` | 0.532967/0.549451/0.532967 | 0.543956/0.554945/0.543956 | `pass` |
| `box021_035_p1` | `E107_trimmed` | 127 | `trim_window_slice` | `127x2` | 0.763780/0.787402/0.763780 | 0.779528/0.795276/0.779528 | `pass` |
| `box021_035_p2` | `E107_retargeted_untrimmed` | 182 | `direct_raw_axis` | `182x2` | 0.549451/0.554945/0.549451 | 0.560440/0.560440/0.560440 | `pass` |
| `box021_035_p2` | `E107_trimmed` | 133 | `trim_window_slice` | `133x2` | 0.751880/0.759398/0.751880 | 0.766917/0.766917/0.766917 | `pass` |

## Interpretation

E137 converts E136's safe bridge candidates into concrete qpos-style semantic
contact artifacts. Each output keeps the original qpos export content and adds:

- `object_contact`: 3cm semantic mask, bool `(T,2)`;
- `object_contact_3cm`: 3cm semantic mask, bool `(T,2)`;
- `object_contact_5cm`: 5cm diagnostic semantic mask, bool `(T,2)`;
- metadata including `object_contact_source`, mapping type/proof, raw case ID,
  person index, and trim bounds.

These files are not equivalent to E126/E131 paired Holosoma `joint_pos/body_*`
exports. E137 deliberately does not claim `MotionLoader` or runtime readiness
for the qpos-style files. The next bounded step is to either convert these
semantic qpos-style exports into the newer Holosoma `joint_pos/body_*` format or
write a qpos-style loader contract probe if Holosoma has one.

## Claims Verification

| Claim | Result |
|---|---|
| C1: E107 candidates are copied into isolated outputs with `object_contact (T,2)` | pass |
| C2: written contact arrays match E136-proven source frame counts | pass |
| C3: metadata records E135 source, threshold, mapping type/proof, raw case, person, and trim window | pass |
| C4: E126/E131 exports are not written | pass |
| C5: no MotionLoader/runtime readiness is claimed | pass |
| C6: no CEM/PPO/training/remote launched | pass |

## Verification

- `.venv/bin/python -m py_compile workspace/core4d/scripts/E137/export_e107_semantic_object_contact.py`
- `bash -n workspace/core4d/scripts/eval/eval_E137_e107_semantic_object_contact_export.sh`
- `bash workspace/core4d/scripts/eval/eval_E137_e107_semantic_object_contact_export.sh`
