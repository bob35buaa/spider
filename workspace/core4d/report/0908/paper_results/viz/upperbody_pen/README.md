# Upper-body × object penetration examples

Frames where the **robot's upper body (arm: forearm/elbow)** clips into the
manipulated box. The paper metrics (`pen_3mm/10mm`, `pen_max_mm`) only measure the
**hand**–object penetration, so these were found separately by replaying each case's
`scene_act` qpos and evaluating the analytic box signed-distance of every
upper-body visual-mesh vertex (torso / waist / shoulder / elbow; wrist+hand
excluded). Scan: `code/scan_upperbody_pen.py` → `tmp/upperbody_pen_both_box.tsv`.

Finding: across all 31 box cases (both methods), the **torso/waist never
penetrates** — penetration is entirely the **forearm/elbow** wrapping into the box
during the lift/carry. Most cases are clean (0 mm); the deep ones are below.

| depth | case | method | frame | link |
|------:|------|--------|------:|------|
| 41.9 mm | box023_20231020_040_p2 | SPIDER-CEM   | 72 | left_elbow (forearm) |
| 16.8 mm | box001_20231023_108_p2 | SPIDER-CEM   | 50 | right_elbow (forearm) |
| 13.5 mm | box001_20231023_108_p2 | OmniRetarget | 18 | right_elbow (forearm) |

`box001_20231023_108_p2` penetrates under **both** methods → usable as an A/B pair.

## Files (`*_solid.png` / `*_xray.png`)
- `solid`: opaque box — the arm is seen vanishing into the box face.
- `xray` : box at 40 % alpha, penetrating link painted **red** — the embedded
  forearm shows through.

Peak frames rendered (± neighbours also present): `box023_040_p2_spider_f072`,
`box001_108_p2_spider_f050`, `box001_108_p2_omni_f018`.
Summary montage: `_contact_sheet_upperbody_pen.png`.

Render: `code/render_upperbody_pen.py --scene … --qpos … --frames … --highlight elbow,wrist`.
