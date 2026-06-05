# E144 desk/chair multi-box proxy review results

Date: 2026-06-05 15:40 CST

## Scope

Responding to the template review finding that desk/chair should not remain as one coarse AABB, I regenerated the E144 review-only draft templates with multi-box collision proxies:

- desk: `desk_multibox_top_legs_crossbars_draft`
- chair: `chair_multibox_seat_back_legs_rails_draft`
- bucket drafts remain `bucket_wall_proxy_aabb`

This is still a draft review package only. No official `approve_clean` / `clean_reviewed` decision was changed, no new Stage2b/CEM-ready rows were released, and no CEM/RL was started.

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

| category | rows | draft policy |
|---|---:|---|
| bucket | 5 | `bucket_wall_proxy_aabb` |
| chair | 4 | `chair_multibox_seat_back_legs_rails_draft` |
| desk | 8 | `desk_multibox_top_legs_crossbars_draft` |

Review queue:

- 17/17 draft rows remain `draft_needs_review`
- 17/17 draft rows remain `release_decision=not_released`
- 17/17 standard orbit renders pass
- 17/17 mesh/collision overlay renders pass

Mesh/collision overlay package:

- Output root: `workspace/core4d/results/E144/E144_full_nonbox_raw_contact/s2_templates_mesh_collision_review/`
- Manifest: `mesh_collision_review_manifest.tsv`
- Rows: 21 total source templates
- Render status: 21/21 pass
- Release decisions: 4 `approve_clean`, 17 `not_released`
- Proxy policy counts: 9 `bucket_wall_proxy_aabb`, 4 `chair_multibox_seat_back_legs_rails_draft`, 8 `desk_multibox_top_legs_crossbars_draft`

Representative sheets:

- `workspace/core4d/results/E144/E144_full_nonbox_raw_contact/s2_templates_mesh_collision_review/templates/desk020_person1/desk020_person1_mesh_collision_review_sheet.png`
- `workspace/core4d/results/E144/E144_full_nonbox_raw_contact/s2_templates_mesh_collision_review/templates/chair021_person1/chair021_person1_mesh_collision_review_sheet.png`
- `workspace/core4d/results/E144/E144_full_nonbox_raw_contact/s2_templates_mesh_collision_review/templates/desk007_person1/desk007_person1_mesh_collision_review_sheet.png`

Visual observation:

- desk proxy now shows a top slab, four legs, and crossbars instead of a single enclosing AABB.
- chair proxy now shows a seat, back, legs, and side rails instead of a single enclosing AABB.
- The overlay title now reflects the current multi-box policy, avoiding the stale `mesh_aabb_box_proxy_draft` label from the earlier draft.

## Decision

Do not promote these templates automatically. The generated proxies are usable as first-pass manual review evidence, but desk/chair geometry still needs human approval or further per-object adjustment before any additional Stage2b/CEM-ready release.
