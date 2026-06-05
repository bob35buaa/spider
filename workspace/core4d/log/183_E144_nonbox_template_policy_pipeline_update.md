# E144 non-box template policy pipeline update

Date: 2026-06-05 17:20 CST

## Scope

The E144 desk/chair template draft process has been promoted from an experiment-local script into the data-construction v3 S2 template policy.

This update standardizes the accepted non-box review proxy scheme:

- bucket: `bucket_wall_proxy_aabb`
- board/stick: `mesh_aabb_box_proxy`
- desk/chair: tight surface voxel multi-box proxy
  - `desk_surface_voxel_multibox_proxy_draft`
  - `chair_surface_voxel_multibox_proxy_draft`

Desk/chair remain review-only. MuJoCo load success and render pass do not release them.

## Code Changes

Updated `workspace/core4d/scripts/data_construction_v3/stages/s2_templates/build_or_audit_templates.py`:

- added `desk` and `chair` to non-box proxy build categories
- added `SURFACE_VOXEL_PROXY_CATEGORIES`
- added tight surface voxel proxy builder:
  - load OBJ with `trimesh`
  - voxelize object-local surface with `target_cells=26`
  - greedily merge occupied surface voxels into local AABB box geoms
  - shrink boxes slightly inward to reduce visible over-expansion
  - emit `object_collision` + `object_collision_voxel_*`
- `template_adapter` now reports `nonbox_surface_voxel_review` for desk/chair
- generated desk/chair templates keep `proxy_template=True`, `manual_review_required=True`, and `build_status=review_required`

Added generic review renderer:

- `workspace/core4d/scripts/data_construction_v3/stages/s2_templates/render_template_mesh_collision_review_package.py`

It reads `template_backlog.tsv` or any compatible TSV with `scene_xml`, then renders:

- `real mesh`
- `collision proxy`
- `mesh + collision`

with optional `--object-only` mode for desk/chair proxy review.

Updated `workspace/core4d/scripts/data_construction_v3/orchestration/run_pipeline.py`:

- after S2 orbit visual review, it now automatically generates:
  - `s2_templates/template_mesh_collision_review/`
  - only for `template_status=manual_review_required`
  - with `--object-only`

## Documentation / Skill Updates

Updated:

- `workspace/core4d/docs/data_construction_v3/02_pipeline_stages.md`
- `workspace/core4d/docs/data_construction_v3/04_scene_template_policy.md`
- `.codex/skills/data-construction-v3-zh/SKILL.md`

The docs now state that desk/chair should not use canonical semantic desk/chair templates. They should use tight surface voxel multi-box review proxies and require mesh/collision overlay evidence before any explicit release.

## Pipeline Policy Check

Ran the updated S2 builder on E144 5cm non-box raw-contact pass in an isolated results directory:

```bash
python3 workspace/core4d/scripts/data_construction_v3/stages/s2_templates/build_or_audit_templates.py \
  --input-tsv workspace/core4d/results/E144/E144_full_nonbox_raw_contact/s1_raw_contact/raw_contact/raw_contact_pass_5cm.tsv \
  --core4d-raw-root /mnt/a0ccc676-9496-49f8-a861-f8a1797dec52/mocap_data/CORE4D/CORE4D_Real \
  --scene-root workspace/core4d/results/E144/E144_full_nonbox_raw_contact/s2_templates_pipeline_policy_check/scene_root \
  --asset-root workspace/core4d/results/E144/E144_full_nonbox_raw_contact/s2_templates_pipeline_policy_check/assets \
  --base-scene example_datasets/processed/core4d/unitree_g1/humanoid_object/box023_person1/scene.xml \
  --out-dir workspace/core4d/results/E144/E144_full_nonbox_raw_contact/s2_templates_pipeline_policy_check/audit \
  --apply-build --overwrite-existing
```

Policy check result:

- 21 required templates
- 21/21 `build_status=review_required`
- 21/21 `template_status=manual_review_required`
- template adapters:
  - 9 `nonbox_proxy_aabb_review`
  - 12 `nonbox_surface_voxel_review`
- collision policies:
  - 9 `bucket_wall_proxy_aabb`
  - 8 `desk_surface_voxel_multibox_proxy_draft`
  - 4 `chair_surface_voxel_multibox_proxy_draft`

Ran the generic mesh/collision renderer:

```bash
python3 workspace/core4d/scripts/data_construction_v3/stages/s2_templates/render_template_mesh_collision_review_package.py \
  --input-tsv workspace/core4d/results/E144/E144_full_nonbox_raw_contact/s2_templates_pipeline_policy_check/audit/template_backlog.tsv \
  --out-dir workspace/core4d/results/E144/E144_full_nonbox_raw_contact/s2_templates_pipeline_policy_check/template_mesh_collision_review \
  --render-statuses manual_review_required \
  --object-only \
  --width 300 --height 260 --frames 8 --fps 8 --overwrite
```

Renderer result:

- 21/21 render pass
- object categories: bucket 9 / desk 8 / chair 4
- all rendered rows remain `manual_review_required`

Representative policy-check sheet:

- `workspace/core4d/results/E144/E144_full_nonbox_raw_contact/s2_templates_pipeline_policy_check/template_mesh_collision_review/templates/desk020_person1/desk020_person1_object_only_mesh_collision_review_sheet.png`

## Validation

```bash
find workspace/core4d/scripts/data_construction_v3 -name '*.py' -print0 | xargs -0 python3 -m py_compile
git diff --check
```

Both passed.

## Decision

This template creation scheme is now part of the v3 data pipeline. It is suitable for continuing downstream after explicit non-box template review. No new Stage2b/CEM-ready rows were released by this update, and no CEM/RL was started.
