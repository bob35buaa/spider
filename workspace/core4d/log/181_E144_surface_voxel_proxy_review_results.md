# E144 desk/chair surface voxel proxy review results

Date: 2026-06-05 16:22 CST

## Issue

Manual review found the previous desk/chair semantic multi-box proxies were still wrong:

- `desk007`: leg/frame proxy did not match the actual side-panel/tube-frame object.
- `desk020`: not the same object family as `desk021`; it is closer to a round-top three-leg stool/small table.
- `desk021` and `desk023`: side panel and U-frame structure should be connected, not represented as disconnected four-leg desk proxies.
- `chair005` and `chair021`: the raw meshes do not match the previous generic seat/back/legs chair proxy.

## Fix

Updated `workspace/core4d/scripts/E144/build_nonbox_template_drafts.py` to stop using semantic desk/chair proxy templates for these draft rows.

For `desk` and `chair`, the draft builder now:

1. Loads the OBJ mesh with `trimesh`.
2. Voxelizes the raw mesh surface in object-local coordinates.
3. Greedily merges occupied surface voxels into local AABB box geoms.
4. Emits all boxes as `object_collision*` geoms.

This produces a surface-following local multi-box proxy instead of assuming a canonical desk/chair topology.

Current policies:

- desk: `desk_surface_voxel_multibox_proxy_draft`
- chair: `chair_surface_voxel_multibox_proxy_draft`
- bucket: unchanged `bucket_wall_proxy_aabb`

## Commands

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

- 5 bucket: `bucket_wall_proxy_aabb`
- 8 desk: `desk_surface_voxel_multibox_proxy_draft`
- 4 chair: `chair_surface_voxel_multibox_proxy_draft`
- 17/17 remain `draft_needs_review`

Review queue:

- 17/17 `release_decision=not_released`
- 17/17 standard render pass
- 17/17 mesh/collision render pass

Mesh/collision overlay package:

- 21/21 render pass
- proxy policy counts: 9 bucket wall proxy, 8 desk surface voxel proxy, 4 chair surface voxel proxy
- release decisions unchanged: 4 `approve_clean`, 17 `not_released`

Object-only review sheets:

- `workspace/core4d/results/E144/E144_full_nonbox_raw_contact/s2_templates_object_only_collision_review/desk007_person1_object_only_mesh_collision_sheet.png`
- `workspace/core4d/results/E144/E144_full_nonbox_raw_contact/s2_templates_object_only_collision_review/desk020_person1_object_only_mesh_collision_sheet.png`
- `workspace/core4d/results/E144/E144_full_nonbox_raw_contact/s2_templates_object_only_collision_review/desk021_person1_object_only_mesh_collision_sheet.png`
- `workspace/core4d/results/E144/E144_full_nonbox_raw_contact/s2_templates_object_only_collision_review/desk023_person1_object_only_mesh_collision_sheet.png`
- `workspace/core4d/results/E144/E144_full_nonbox_raw_contact/s2_templates_object_only_collision_review/chair005_person1_object_only_mesh_collision_sheet.png`
- `workspace/core4d/results/E144/E144_full_nonbox_raw_contact/s2_templates_object_only_collision_review/chair021_person1_object_only_mesh_collision_sheet.png`

Collision geom counts in representative person1 templates:

| source template | `object_collision*` geoms |
|---|---:|
| `desk007_person1` | 31 |
| `desk020_person1` | 32 |
| `desk021_person1` | 33 |
| `desk023_person1` | 17 |
| `chair005_person1` | 23 |
| `chair021_person1` | 62 |

## Visual observation

- `desk020` no longer uses a desk/table semantic proxy; the surface boxes follow the round tabletop and its three support rods.
- `desk021` and `desk023` now follow the connected side-panel/U-frame silhouette instead of disconnected four-leg desk proxies.
- `desk007` now follows the side panel and tube frame more closely, though it remains a blocky surface proxy.
- `chair005` and `chair021` no longer use a generic seat/back/legs proxy; they now follow the actual mesh silhouette with blocky surface boxes.

## Decision

This is a better review draft than the previous semantic multi-box proxies, but still not an automatic release. All desk/chair rows remain `not_released`; no Stage2b/CEM-ready rows were added, and no CEM/RL was started.
