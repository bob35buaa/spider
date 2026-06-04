# E134 Semantic Contact to Holosoma Bridge Audit Results

Date: 2026-06-03

## Scope

E134 followed `workspace/core4d/plan/143_E134_semantic_contact_holosoma_bridge_audit_plan.md`.
It is a read-only bridge audit after E133. The goal is to check whether existing
Spider/MJWP semantic contact artifacts can safely become Holosoma
`object_contact` masks for the E126/E131/E133 Box021 fragment exports.

E134 did not launch CEM, PPO, Holosoma training, checkpoint creation, or remote
jobs. It also did not write any Holosoma semantic `object_contact` export.

## Outputs

```text
workspace/core4d/results/E134/semantic_contact_holosoma_bridge_audit/
  e134_semantic_contact_holosoma_bridge_manifest.tsv
  e134_semantic_contact_holosoma_bridge_summary.json
  e134_semantic_contact_holosoma_bridge_summary.md
```

## Result

| metric | value |
|---|---:|
| audit rows | 7 |
| legacy semantic mask available rows | 7 |
| v3 semantic contact ready rows | 0 |
| timeline exact match rows | 2 |
| semantic bridge candidate rows | 0 |
| E126/E131 semantic-ready rows | 0 |
| structural proxy ready rows | 2 |
| RL-ready rows | 0 |
| training launched | false |
| CEM launched | false |
| remote jobs launched | false |
| status | pass |

Bridge table:

| case | export kind | v3 3cm | matching mask key | semantic candidate | existing object_contact | failure |
|---|---|---|---|---|---|---|
| `box021_035_p1` | E126 fragment | not_run | none | false | false | `v3_semantic_contact_not_ready` |
| `box021_035_p1` | E131 geometry proxy | not_run | none | false | true | `v3_semantic_contact_not_ready` |
| `box021_035_p1` | E107 trimmed | not_run | none | false | false | `v3_semantic_contact_not_ready` |
| `box021_035_p2` | E126 fragment | not_run | none | false | false | `v3_semantic_contact_not_ready` |
| `box021_035_p2` | E131 geometry proxy | not_run | none | false | true | `v3_semantic_contact_not_ready` |
| `box021_035_p2` | E107 trimmed | not_run | `spider_contact_mask_3cm` | false | false | `v3_semantic_contact_not_ready` |
| `box021_029_p2` | E107 trimmed | not_run | `spider_contact_mask_3cm` | false | false | `v3_semantic_contact_not_ready` |

## Interpretation

E134 separates three facts that were easy to conflate:

1. Legacy masks exist for the target Box021 cases, including E079/E082/E084
   `raw_contact_mask_3cm.npz` artifacts.
2. The current v3 registry still marks the corresponding raw case IDs
   `raw_contact_3cm_status=not_run` and `raw_contact_5cm_status=not_run`, with
   no v3 `raw_contact_artifact_npz`.
3. E131 has `object_contact`, but its source is
   `actor_rubber_hand_box021_surface_proxy_5cm`, so it remains structural proxy
   evidence, not semantic contact evidence.

Two diagnostic rows have an exact legacy `spider_contact_mask_3cm` axis match
with E107 trimmed exports (`box021_035_p2` and main `box021_029_p2`), but they
still do not become semantic-ready because the v3 S1 contact chain is not
populated for those raw case IDs. E126/E131 fragment exports have no exact
semantic time-axis bridge in this audit.

## Claims Verification

| Claim | Result |
|---|---|
| C1: existing Box021 legacy masks can be located | pass |
| C2: exact time-axis compatibility is distinguished from mask existence | pass |
| C3: Holosoma semantic export is allowed only for v3-ready exact match | pass; zero allowed rows |
| C4: mismatch/not-ready states are blockers, not silent conversions | pass |
| C5: no CEM/PPO/training/remote launched | pass |

## Verification

- `.venv/bin/python -m py_compile workspace/core4d/scripts/E134/audit_semantic_contact_holosoma_bridge.py`
- `bash -n workspace/core4d/scripts/eval/eval_E134_semantic_contact_holosoma_bridge_audit.sh`
- `bash workspace/core4d/scripts/eval/eval_E134_semantic_contact_holosoma_bridge_audit.sh`

## Next Step

The next valid bounded step is to run a v3 S1 raw-contact remine/audit for the
target Box021 raw sequences only, using `run_raw_contact.py` with `box021` scope
and both 3cm/5cm thresholds, then sync the resulting raw-contact rows into the
registry without overwriting the separate thresholds. Only after that should a
new bridge/export step consider writing semantic Holosoma `object_contact`.
