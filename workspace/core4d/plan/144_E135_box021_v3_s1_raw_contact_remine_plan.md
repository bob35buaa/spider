# E135 Box021 v3 S1 Raw-Contact Remine Plan

Date: 2026-06-03

## Context

E134 proved that the current Holosoma Box021 branch is blocked on semantic
contact readiness, not on runtime plumbing. Legacy E079/E082/E084 masks exist,
and E131 structural `object_contact` proxies can flow through Holosoma, but the
v3 registry still marks the target raw Box021 cases as
`raw_contact_3cm_status=not_run/raw_contact_5cm_status=not_run`.

E135 is the bounded S1 follow-up requested by E134. It mines raw-contact proxy
artifacts for the target Box021 raw sequences only, at both 3cm and 5cm, then
syncs evidence into isolated E135 registries. It does not mutate the global v3
registry and does not launch CEM, PPO, Holosoma training, checkpoints, or remote
jobs.

## Scope

Target raw case-person rows:

- `box021_20231011_035_p1`
- `box021_20231011_035_p2`
- `box021_20231018_029_p1`
- `box021_20231018_029_p2`

The extra `box021_20231018_029_p1` row is included because S1 raw contact is
sequence-grouped and the `029` sequence has both person rows in the v3 inventory.
The downstream semantic bridge remains focused on E134's target p2 row unless a
later audit explicitly promotes p1.

## Implementation

Add:

```text
workspace/core4d/scripts/E135/summarize_box021_v3_s1_raw_contact_remine.py
workspace/core4d/scripts/eval/eval_E135_box021_v3_s1_raw_contact_remine.sh
```

The eval wrapper will:

- rebuild S1 inventory from the mounted CORE4D raw root;
- filter the inventory to the bounded Box021 target rows;
- run `run_raw_contact.py` with `--queue object-key --object-keys box021` and
  `--thresholds-m 0.03,0.05`;
- sync raw-contact evidence into isolated E135 registry directories, with 5cm
  first and 3cm last in the combined registry so common `contact_mask_*` fields
  represent 3cm while threshold-specific fields remain authoritative;
- write a summary manifest, JSON, and Markdown from the resulting artifacts.

## Success Criteria

- Fixed local eval entry runs end to end.
- Inventory filter contains exactly the four target case-person rows.
- S1 writes 3cm and 5cm candidate/pass TSV+JSON files.
- Per-sequence `raw_contact_proxy.npz` files contain both
  `raw_contact_mask_3cm` and `raw_contact_mask_5cm`.
- E135 isolated registry rows expose separate `raw_contact_3cm_status` and
  `raw_contact_5cm_status` fields.
- `global_registry_updated=false`, `training_launched=false`,
  `cem_launched=false`, and `remote_jobs_launched=false`.

## Command

```bash
bash workspace/core4d/scripts/eval/eval_E135_box021_v3_s1_raw_contact_remine.sh
```

## No-Go Rules

- Do not use `run_pipeline.py`; it continues beyond S1 and has no stop-after-S1
  switch.
- Do not overwrite `workspace/core4d/data_construction_v3/existing_cases.tsv` in
  E135.
- Do not treat raw-contact geometric proxy evidence as Holosoma semantic
  `object_contact` until a later bridge/export audit proves time-axis and schema
  compatibility.
- Do not launch CEM, PPO, Holosoma training, checkpoints, or remote jobs.
