# E144 desk proxy axis fix results

Date: 2026-06-05 15:58 CST

## Issue

Manual visual review correctly flagged that the desk multi-box proxy looked coordinate-wrong. The earlier proxy assumed the desk tabletop normal / local up axis was MuJoCo local Z. The rendered mesh/collision overlay showed this was wrong for the E144 desk meshes.

## Diagnosis

The visualization was not the root problem: real mesh and collision proxy were rendered from the same MuJoCo scene and object frame.

The desk OBJ meshes have their dominant face-normal area on local Y:

- `desk007`: dominant normal axis Y
- `desk020`: dominant normal axis Y
- `desk021`: dominant normal axis Y
- `desk023`: dominant normal axis Y

Example `desk020` processed mesh bounds:

- extent: approximately `0.492 x 0.330 x 0.492`
- AABB center: near zero
- dominant face-normal axis: Y

So the mesh was centered correctly, but the proxy generator used the wrong semantic axis for the tabletop slab and legs.

## Fix

Updated `workspace/core4d/scripts/E144/build_nonbox_template_drafts.py`:

- parse OBJ vertices and faces directly
- compute mesh bounds and dominant face-normal axis
- for desk proxies, build top/legs/crossbars around that detected axis instead of hard-coding Z
- keep chair policy unchanged for now

The regenerated desk policy is now:

```text
desk_multibox_yup_top_legs_crossbars_draft
```

Example `desk020_person1` scene now has the tabletop slab normal along Y:

```xml
<geom name="object_collision" type="box" pos="0 0.153553 0" size="0.246043 0.0115578 0.246004" ... />
```

## Regenerated evidence

Re-ran:

```bash
python3 workspace/core4d/scripts/E144/build_nonbox_template_drafts.py \
  --core4d-raw-root /mnt/a0ccc676-9496-49f8-a861-f8a1797dec52/mocap_data/CORE4D/CORE4D_Real \
  --apply --overwrite-existing

python3 workspace/core4d/scripts/data_construction_v3/stages/s2_templates/build_or_audit_templates.py \
  --input-tsv workspace/core4d/results/E144/E144_full_nonbox_raw_contact/s1_raw_contact/raw_contact/raw_contact_pass_5cm.tsv \
  --core4d-raw-root /mnt/a0ccc676-9496-49f8-a861-f8a1797dec52/mocap_data/CORE4D/CORE4D_Real \
  --out-dir workspace/core4d/results/E144/E144_full_nonbox_raw_contact/s2_templates_after_draft_audit

python3 workspace/core4d/scripts/data_construction_v3/stages/s2_templates/render_template_review_package.py \
  --template-backlog-tsv workspace/core4d/results/E144/E144_full_nonbox_raw_contact/s2_templates_after_draft_audit/template_backlog.tsv \
  --out-dir workspace/core4d/results/E144/E144_full_nonbox_raw_contact/s2_templates_after_draft_visual_review \
  --render-statuses manual_review_required \
  --width 640 --height 480 --frames 12 --fps 12 --overwrite

python3 workspace/core4d/scripts/E144/render_template_mesh_collision_review.py \
  --overwrite --frames 18 --width 360 --height 300 --fps 12
```

## Results

Draft manifest:

- bucket: 5 `bucket_wall_proxy_aabb`
- chair: 4 `chair_multibox_seat_back_legs_rails_draft`
- desk: 8 `desk_multibox_yup_top_legs_crossbars_draft`

Review queue:

- 17/17 `release_decision=not_released`
- 17/17 standard render pass
- 17/17 mesh/collision render pass

Mesh/collision overlay:

- 21/21 render pass
- proxy policy counts: 9 bucket wall proxy, 4 chair multi-box, 8 desk Y-up multi-box

Representative updated sheets:

- `workspace/core4d/results/E144/E144_full_nonbox_raw_contact/s2_templates_mesh_collision_review/templates/desk020_person1/desk020_person1_mesh_collision_review_sheet.png`
- `workspace/core4d/results/E144/E144_full_nonbox_raw_contact/s2_templates_mesh_collision_review/templates/desk007_person1/desk007_person1_mesh_collision_review_sheet.png`
- `workspace/core4d/results/E144/E144_full_nonbox_raw_contact/s2_templates_mesh_collision_review/templates/desk021_person1/desk021_person1_mesh_collision_review_sheet.png`

## Decision

The previous desk proxy was axis-wrong. The current Y-up proxy is better aligned with the raw mesh local frame, but it remains a review-only draft. No desk template was promoted to `approve_clean`, no Stage2b/CEM-ready rows were added, and no CEM/RL was started.
