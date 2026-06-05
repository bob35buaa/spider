# E144 tight surface voxel proxy results

Date: 2026-06-05 17:08 CST

## Issue

Manual review noted that the surface voxel collision proxies looked larger than the real mesh. This was expected from the previous surface voxel method:

- `trimesh.voxelized()` returns occupied surface voxels.
- The previous proxy converted each occupied voxel into a full box.
- Therefore each surface point was inflated by roughly half a voxel pitch before rendering.

This is acceptable for a conservative collision draft, but it was visually too loose for template review.

## Change

Updated `workspace/core4d/scripts/E144/build_nonbox_template_drafts.py` for desk/chair surface voxel proxies:

- increased surface voxel resolution from `target_cells=18` to `target_cells=26`
- increased `max_boxes` from `128` to `180`
- shrank each emitted box inward by a small amount after greedy merging

The policy names remain:

- `desk_surface_voxel_multibox_proxy_draft`
- `chair_surface_voxel_multibox_proxy_draft`

## Results

Draft manifest:

- bucket: 5 `bucket_wall_proxy_aabb`
- desk: 8 `desk_surface_voxel_multibox_proxy_draft`
- chair: 4 `chair_surface_voxel_multibox_proxy_draft`
- 17/17 remain `draft_needs_review`

Review queue:

- 17/17 `release_decision=not_released`
- 17/17 standard render pass
- 17/17 mesh/collision render pass

Mesh/collision overlay:

- 21/21 render pass
- release decisions unchanged: 4 `approve_clean`, 17 `not_released`

Representative person1 collision geom counts after tightening:

| source template | `object_collision*` geoms |
|---|---:|
| `desk007_person1` | 41 |
| `desk020_person1` | 69 |
| `desk021_person1` | 73 |
| `desk023_person1` | 37 |
| `chair005_person1` | 64 |
| `chair021_person1` | 127 |

Updated object-only review sheets:

- `workspace/core4d/results/E144/E144_full_nonbox_raw_contact/s2_templates_object_only_collision_review/desk020_person1_object_only_mesh_collision_sheet.png`
- `workspace/core4d/results/E144/E144_full_nonbox_raw_contact/s2_templates_object_only_collision_review/chair021_person1_object_only_mesh_collision_sheet.png`
- `workspace/core4d/results/E144/E144_full_nonbox_raw_contact/s2_templates_object_only_collision_review/desk021_person1_object_only_mesh_collision_sheet.png`

## Visual observation

The proxy is now visibly tighter around `desk020`'s round top and support rods, and around `chair021`'s irregular panels/rails. A small red outline remains around the mesh because this is still a box-based collision approximation, not a true mesh collider.

## Decision

Keep all desk/chair templates as review-only drafts. No `approve_clean` state was added, no Stage2b/CEM-ready rows were released, and no CEM/RL was started.
