# E135 Box021 v3 S1 Raw-Contact Remine Results

Date: 2026-06-03

## Scope

E135 followed
`workspace/core4d/plan/144_E135_box021_v3_s1_raw_contact_remine_plan.md`.
It is the bounded v3 S1 raw-contact follow-up to E134. E135 rebuilt the S1
inventory from raw CORE4D data, filtered it to two Box021 raw sequences, mined
raw-contact proxy masks at 3cm and 5cm, and synced evidence into isolated E135
registry directories only.

E135 did not update `workspace/core4d/data_construction_v3/existing_cases.tsv`.
It did not launch CEM, PPO, Holosoma training, checkpoints, or remote jobs.

## Outputs

```text
workspace/core4d/results/E135/box021_v3_s1_raw_contact_remine/
  s1_raw_contact/inventory/inventory.tsv
  s1_raw_contact/inventory/inventory_box021_bounded.tsv
  s1_raw_contact/raw_contact/raw_contact_candidates_3cm.tsv
  s1_raw_contact/raw_contact/raw_contact_candidates_5cm.tsv
  s1_raw_contact/raw_contact/raw_contact_pass_3cm.tsv
  s1_raw_contact/raw_contact/raw_contact_pass_5cm.tsv
  s1_raw_contact/raw_contact/per_sequence/20231011_035_box021/raw_contact_proxy.npz
  s1_raw_contact/raw_contact/per_sequence/20231018_029_box021/raw_contact_proxy.npz
  registries_combined_5cm_then_3cm/case_state_registry.tsv
  registries_3cm_only/case_state_registry.tsv
  registries_5cm_only/case_state_registry.tsv
  summary/e135_box021_v3_s1_raw_contact_manifest.tsv
  summary/e135_box021_v3_s1_raw_contact_summary.json
  summary/e135_box021_v3_s1_raw_contact_summary.md
```

## Result

| metric | value |
|---|---:|
| full inventory rows | 1840 |
| bounded inventory rows | 4 |
| unique raw sequences | 2 |
| 3cm candidate rows | 4 |
| 3cm pass rows | 4 |
| 5cm candidate rows | 4 |
| 5cm pass rows | 4 |
| per-sequence proxy NPZ files | 2 |
| isolated combined registry rows | 4 |
| global registry updated | false |
| training launched | false |
| CEM launched | false |
| remote jobs launched | false |
| status | pass |

Candidate table:

| threshold | case | decision | score | L/R/both active | partner any | registry 3cm/5cm |
|---|---|---|---:|---|---:|---|
| `3cm` | `box021_20231011_035_p1` | `raw_contact_pass` | 96.867 | 0.9867/1.0000/0.9867 | 0.9733 | `pass`/`pass` |
| `3cm` | `box021_20231011_035_p2` | `raw_contact_pass` | 95.133 | 0.9600/0.9733/0.9600 | 1.0000 | `pass`/`pass` |
| `3cm` | `box021_20231018_029_p1` | `raw_contact_pass` | 90.625 | 1.0000/0.8750/0.8750 | 1.0000 | `pass`/`pass` |
| `3cm` | `box021_20231018_029_p2` | `raw_contact_pass` | 90.625 | 0.8750/1.0000/0.8750 | 1.0000 | `pass`/`pass` |
| `5cm` | `box021_20231011_035_p1` | `raw_contact_pass` | 99.600 | 1.0000/1.0000/1.0000 | 0.9733 | `pass`/`pass` |
| `5cm` | `box021_20231011_035_p2` | `raw_contact_pass` | 96.000 | 0.9733/0.9733/0.9733 | 1.0000 | `pass`/`pass` |
| `5cm` | `box021_20231018_029_p1` | `raw_contact_pass` | 92.656 | 1.0000/0.9062/0.9062 | 1.0000 | `pass`/`pass` |
| `5cm` | `box021_20231018_029_p2` | `raw_contact_pass` | 90.625 | 0.8750/1.0000/0.8750 | 1.0000 | `pass`/`pass` |

Per-sequence artifacts:

| artifact | 3cm shape | 5cm shape |
|---|---|---|
| `workspace/core4d/results/E135/box021_v3_s1_raw_contact_remine/s1_raw_contact/raw_contact/per_sequence/20231011_035_box021/raw_contact_proxy.npz` | `182x2x2` | `182x2x2` |
| `workspace/core4d/results/E135/box021_v3_s1_raw_contact_remine/s1_raw_contact/raw_contact/per_sequence/20231018_029_box021/raw_contact_proxy.npz` | `134x2x2` | `134x2x2` |

## Interpretation

E135 closes the specific E134 v3 S1 gap for the bounded Box021 raw sequences:
the target rows now have v3 raw-contact proxy artifacts at both thresholds in
isolated E135 outputs, and both threshold-specific registry fields are populated
as `pass`.

The combined isolated registry was synced 5cm first and 3cm last. Therefore the
generic `contact_mask_npz/contact_mask_label/raw_contact_artifact_npz` fields
represent 3cm, while `contact_mask_3cm_npz`, `contact_mask_5cm_npz`,
`raw_contact_3cm_artifact_npz`, and `raw_contact_5cm_artifact_npz` are the
threshold-specific authority fields.

The remaining blocker is not S1 raw-contact existence. The next bounded step is
a bridge/export audit that checks whether these raw-axis masks can safely map to
the Holosoma E126/E131/E133 export time axes without resizing or silently
changing semantics. E135 itself does not write Holosoma `object_contact`.

## Claims Verification

| Claim | Result |
|---|---|
| C1: fixed eval entry runs locally | pass |
| C2: bounded inventory contains exactly four target rows | pass |
| C3: 3cm and 5cm raw-contact candidate/pass outputs are written | pass |
| C4: per-sequence NPZ files contain both 3cm and 5cm masks | pass |
| C5: isolated registry rows expose both threshold statuses | pass |
| C6: no global registry/CEM/PPO/training/remote launched | pass |

## Verification

- `.venv/bin/python -m py_compile workspace/core4d/scripts/E135/summarize_box021_v3_s1_raw_contact_remine.py`
- `bash -n workspace/core4d/scripts/eval/eval_E135_box021_v3_s1_raw_contact_remine.sh`
- `bash workspace/core4d/scripts/eval/eval_E135_box021_v3_s1_raw_contact_remine.sh`
