# CORE4D Evaluation Metric Standard

Standard ID: `core4d-e154-physics-contact-v1`

This is the fixed metric baseline for E154 and later CORE4D retargeting/CEM
experiments. New evaluators should import field groups from
`workspace/core4d/scripts/eval/lib/core_metrics.py` instead of redefining metric
lists locally.

## Required Entry Point

Use `lib.core_metrics.evaluate_sequence(...)` for per-case MuJoCo kinematic
replay metrics. Write raw per-case TSVs with `METRIC_FIELDS`.

For method summaries and deltas, import:

- `STANDARD_SUMMARY_METRICS`
- `STANDARD_DELTA_METRICS`
- `STANDARD_TRACK_DIAG`
- `STANDARD_MASK_DELTA_METRICS`
- `STANDARD_TABLE_METRIC_FIELDS`
- `STANDARD_TABLE_METRIC_ORDER`
- `STANDARD_TABLE_METRIC_DIRECTIONS`

Experiment-specific health fields, such as CEM gate valid/fallback rates, may be
appended after the standard fields.

## Physical Contact Rule

Raw MuJoCo hand-object contact is still recorded as
`hand_object_physics_contact_frac`, but it is diagnostic only.

The standard table uses two clean contact thresholds:

- 3mm: `contact.dist < -0.003` is physical penetration.
- 5mm: `contact.dist < -0.005` is physical penetration.

For each threshold, a frame with any hand-object MuJoCo contact is classified
into exactly one bucket:

- contact: frame min `contact.dist >= -threshold`
- penetration: frame min `contact.dist < -threshold`

Therefore, for the same threshold:

`hand_object_physics_contact_{3,5}mm_frac + hand_object_physics_penetration_{3,5}mm_frame_frac == hand_object_physics_contact_frac`

up to formatting precision.

## Masked Contact

Real 3cm contact-mask diagnostics use the same thresholded contact definition:

- `hand_object_physics_contact_3mm_in_mask_frac`
- `hand_object_physics_contact_5mm_in_mask_frac`
- `hand_object_release_false_contact_3mm_frac`
- `hand_object_release_false_contact_5mm_frac`

The old raw-mask fields remain available for backward diagnostics, but E154+
tables should use the thresholded 3mm/5mm fields.

## Table Metrics

The canonical visible comparison table is:

- `hand_geom_near_5cm_frac`
- `hand_geom_penetration_2mm_frac`
- `hand_geom_penetration_5mm_frac`
- `hand_object_physics_contact_3mm_frac`
- `hand_object_physics_penetration_3mm_frame_frac`
- `hand_object_physics_contact_5mm_frac`
- `hand_object_physics_penetration_5mm_frame_frac`
- `leg_penetration_frac`

Use `STANDARD_TABLE_METRIC_ORDER` and `STANDARD_TABLE_METRIC_DIRECTIONS` for
ranking. Higher is better only for near/contact metrics; lower is better for
penetration, leg penetration, errors, and false-contact diagnostics.
