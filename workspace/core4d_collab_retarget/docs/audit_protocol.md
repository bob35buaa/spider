# Failure Attribution Audit Protocol

This document records the E020 audit protocol for CORE4D collaborative retargeting failures. It extends the E076 contact-source audit from one case into a reusable six-step pipeline for all E018b cases.

## Scope

Input data:

- E018b rollout NPZ, online MP4, `comparison.csv`, timeseries CSV, and leg/object timeseries CSV.
- E018b `contact_masks/*/raw_contact_mask_3cm.npz`, which contains raw CORE4D SMPL-X 3cm contact proxy masks and eval/ref frame mappings.
- E018b `scene_snapshot/*/0/trajectory_kinematic.npz`, which contains the kinematic reference and current processed `contact` field.
- E017 anchor audit tables, used as weak evidence for selected-person versus counterpart-person contact-face disagreement.

Output data:

- `workspace/core4d_collab_retarget/results/E020_audit/root_cause_attribution.csv`
- `workspace/core4d_collab_retarget/results/E020_audit/attribution_summary.md`
- `workspace/core4d_collab_retarget/results/E020_audit/per_case/<variant>/attribution_panel.png`

`box025_p2` is a pass row rather than a failure row; E020 records `root_cause=pass` for that case and keeps the failure-root-cause enum for the other 12 cases.

## S1 Anchor vs Raw Contact

Script: `workspace/core4d_collab_retarget/scripts/E020_audit/audit_anchor_vs_raw.py`

For each case, S1 recomputes raw SMPL-X hand/object contact centroids from the CORE4D raw sequence referenced by the existing 3cm contact-mask summary. It samples the object mesh surface, maps it through `smooth_objposes.npy`, queries broad SMPL-X hand vertices against the object surface, and stores object-local nearest surface points.

The E018b canonical support anchor is compared to selected-person and partner-person raw contact centroids after axis-wise scaling from MuJoCo object half-size to raw mesh half-size. The pass gate is intentionally strict: partner raw contact should be within 8cm of the anchor, or on the same dominant face with bounded centroid distance.

## S2 Kinematic Reference Physics

Script: `workspace/core4d_collab_retarget/scripts/E020_audit/audit_ref_physics.py`

S2 reads the saved kinematic reference and E018b ref geometry metrics. It checks:

- pelvis minimum height,
- object bottom proxy,
- ref hand/object contact percentage,
- ref leg/object interference percentage.

Cases with high ref leg/object interference are attributed to `retarget_kinematic` when the rollout failure is contact/artifact related.

## S3 Contact Mask vs Raw

Script: `workspace/core4d_collab_retarget/scripts/E020_audit/audit_mask_vs_raw.py`

S3 compares the current processed `trajectory_kinematic.contact[:, :2]` field with the raw 3cm per-person/per-hand contact proxy aligned to the same 30Hz reference horizon. In E018b this exposes a systemic issue: the processed contact field is effectively all-on for every case, while raw 3cm contact is intermittent or hand-asymmetric.

This does not automatically make every case a contact-mask root cause; it is a required evidence column used by S5 together with the rollout diagnostic class.

## S4 Sim vs Ref Overlay

Script: `workspace/core4d_collab_retarget/scripts/E020_audit/overlay_sim_ref_curves.py`

S4 overlays rollout and reference metrics:

- object position error,
- sim/ref pelvis height,
- sim/ref hand-object SDF,
- penetration heatmap proxy from leg/object and hand/object SDF,
- robot joint error heatmap.

The S4 pass gate reuses E018b strict booleans: object success, transport success, no robot fall, contact-preservation ok, deep-penetration ok, and floor/leg ok.

## S5 Root-Cause Decision

Script: `workspace/core4d_collab_retarget/scripts/E020_audit/decide_root_cause.py`

Decision classes:

- `raw_data`
- `retarget_kinematic`
- `contact_mask`
- `algo_tracking`
- `algo_contact`
- `algo_stability`
- `pass`

Decision priority:

1. Pass rows remain `pass`.
2. Robot-fall visual failures are `algo_stability`.
3. GT-anchor cases with object success but contact gap are `algo_contact`.
4. Known all-on contact-mask mismatch with contact gap is `contact_mask`.
5. High ref leg/object interference is `retarget_kinematic`.
6. Contact/artifact failures after object success are `algo_contact`.
7. Object-tracking failures would be `algo_tracking`.

## S6 Visual Validation

Scripts:

- `workspace/core4d_collab_retarget/scripts/E020_audit/render_attribution_keyframes.py`
- `workspace/core4d_collab_retarget/scripts/E020_audit/plot_attribution_panel.py`

S6 extracts grasp/mid/release keyframes from E018b online videos and builds a per-case attribution panel containing:

- S1 anchor/raw scatter,
- S2 ref physics timeline,
- S3 mask/raw timeline,
- S4 sim/ref overlay,
- penetration heatmap,
- joint error heatmap.

## Reproduction

Run:

```bash
.venv/bin/python workspace/core4d_collab_retarget/scripts/E020_audit/run_all.py
```

The raw contact centroid cache is stored under:

```text
workspace/core4d_collab_retarget/results/E020_audit/raw_contact_cache/
```

Scene/data XML snapshots are stored under:

```text
workspace/core4d_collab_retarget/results/E020_audit/scene_snapshot/
```
