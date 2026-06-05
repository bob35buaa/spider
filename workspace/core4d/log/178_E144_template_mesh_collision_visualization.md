# E144 Template Mesh/Collision Visualization

Date: 2026-06-05

## Purpose

The previous template sheets rendered the scene but did not make the collision proxy explicit enough. This pass renders every E144 nonbox source template with three synchronized views:

- `real mesh`
- `collision proxy`
- `mesh + collision`

This makes the difference between true visual mesh and simplified collision geometry visible for review.

## Script

- `workspace/core4d/scripts/E144/render_template_mesh_collision_review.py`

Command:

```bash
python3 workspace/core4d/scripts/E144/render_template_mesh_collision_review.py \
  --overwrite \
  --frames 18 \
  --width 360 \
  --height 300 \
  --fps 12
```

## Outputs

Root:

- `workspace/core4d/results/E144/E144_full_nonbox_raw_contact/s2_templates_mesh_collision_review/`

Manifest:

- `workspace/core4d/results/E144/E144_full_nonbox_raw_contact/s2_templates_mesh_collision_review/mesh_collision_review_manifest.tsv`

Summary:

- rows: 21
- render pass: 21/21
- release decisions: 4 `approve_clean`, 17 `not_released`
- categories: 9 bucket, 8 desk, 4 chair

Each template has:

- `<task>_mesh_collision_review.mp4`
- `<task>_mesh_collision_review_sheet.png`

Example paths:

- `workspace/core4d/results/E144/E144_full_nonbox_raw_contact/s2_templates_mesh_collision_review/templates/bucket007_person1/bucket007_person1_mesh_collision_review.mp4`
- `workspace/core4d/results/E144/E144_full_nonbox_raw_contact/s2_templates_mesh_collision_review/templates/desk020_person1/desk020_person1_mesh_collision_review.mp4`
- `workspace/core4d/results/E144/E144_full_nonbox_raw_contact/s2_templates_mesh_collision_review/templates/chair021_person1/chair021_person1_mesh_collision_review.mp4`

## Observations

- Bucket wall proxy videos show real mesh separately from the bucket collision proxy. From some angles the translucent wall proxy can look like a solid box, so review should use the `collision proxy` and `overlay` panels together.
- Desk/chair draft videos clearly show that the current collision proxy is a coarse AABB box around complex geometry. These templates remain `not_released` and are not suitable for Stage2b/CEM without explicit review or a better multi-box collision design.

No Stage2b, CEM, or RL run was launched in this visualization pass.
