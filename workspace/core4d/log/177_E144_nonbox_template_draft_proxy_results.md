# E144 Nonbox Template Draft Proxy Results

Date: 2026-06-05

## What "wall proxy" means

For bucket objects, `bucket_wall_proxy_aabb` keeps the real object mesh for visualization, but replaces physical collision with five simple box geoms:

- one bottom slab: `object_collision`
- four side-wall slabs: `object_collision_bucket_xneg`, `object_collision_bucket_xpos`, `object_collision_bucket_yneg`, `object_collision_bucket_ypos`

This is a reviewable approximation of a hollow bucket. It is less wrong than a single solid AABB box, because the bucket interior is not treated as one solid block.

Desk/chair drafts are different: they are currently coarse `mesh_aabb_box_proxy_draft` templates. They also keep the real mesh for visualization, but collision is a single AABB box. These are intentionally rough and should not be treated as clean templates without manual visual/physical review.

## Why only 11 cases entered CEM before

The previous E144 release gate only allowed templates already reviewed as clean. At that point:

- 4 bucket source templates had bottom + four-wall proxy collision and passed local review.
- 5 bucket source templates existed but lacked the four wall collision geoms.
- 8 desk and 4 chair source templates were complex/manual or missing scenes.

Therefore only the 4 approved bucket templates produced 11 Stage2b/CEM-ready rows. Desk/chair were not auto-released by policy.

## New draft generation

Added:

- `workspace/core4d/scripts/E144/build_nonbox_template_drafts.py`

Command:

```bash
python3 workspace/core4d/scripts/E144/build_nonbox_template_drafts.py \
  --core4d-raw-root /mnt/a0ccc676-9496-49f8-a861-f8a1797dec52/mocap_data/CORE4D/CORE4D_Real \
  --apply \
  --overwrite-existing
```

Results:

- draft rows: 17
- bucket wall proxy drafts: 5
- desk/chair AABB drafts: 12
- MuJoCo load status: 17/17 loadable
- draft status: 17/17 `draft_needs_review`
- release status: 17/17 `not_released`

Primary files:

- Draft manifest: `workspace/core4d/results/E144/E144_full_nonbox_raw_contact/s2_templates/draft_proxy/draft_proxy_manifest.tsv`
- Review queue: `workspace/core4d/results/E144/E144_full_nonbox_raw_contact/s2_templates/draft_proxy/draft_proxy_review_queue.tsv`
- Draft audit: `workspace/core4d/results/E144/E144_full_nonbox_raw_contact/s2_templates_after_draft_audit/template_backlog.tsv`
- Draft visual review: `workspace/core4d/results/E144/E144_full_nonbox_raw_contact/s2_templates_after_draft_visual_review/template_visual_manifest.tsv`
- Backups for overwritten existing scenes: `workspace/core4d/results/E144/E144_full_nonbox_raw_contact/s2_templates/draft_backups/`

Visual package:

- rendered rows: 21/21 pass
- review queue rows: 17/17 pass render
- sampled sheets inspected: `bucket007_person1`, `desk020_person1`, `chair021_person1`

## Release Policy

This work creates rough templates only. It does not update the official `nonbox_template_review.tsv` to `approve_clean`, does not rerun Stage2b, does not generate new CEM-ready variants, and does not launch CEM/RL.

To release more rows, the next step is explicit review:

- bucket drafts can be promoted after confirming wall proxy geometry and render sheets.
- desk/chair drafts require stronger scrutiny because single AABB collision is likely too coarse for legs, table tops, and chair backs.
