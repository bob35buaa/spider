#!/usr/bin/env python3
"""E206: drop every load-bearing proxy box onto the floor plane (G10).

A desk/chair rests on 3-4 legs.  When the proxy's leg boxes end at different
heights the simulated object stands on whichever leg is lowest and rocks on the
others — a tilt invented entirely by the proxy.  The mirror failure is a box
reaching below the mesh, which floats the whole object above the ground.

Measured on the 2026-09-03 hand-edited proxies, before this pass:

    chair020  legs spread 11.7 mm   chair022  16.0 mm (+2 legs 185 mm short)
    desk020   3 boxes 22-30 mm BELOW the mesh floor
    desk023   all 4 legs 5.3 mm below   chair005  whole body 8.8 mm above

No existing gate catches this: G3/G4/G5 are aggregate distances and a 12 mm leg
error does not move a p90.

This tool holds each box's TOP face fixed and moves its BOTTOM onto the floor —
a leg is attached to the seat, so a short leg must grow downwards rather than
slide down and detach.  Only boxes already within `--band` of the floor are
touched; a seat or an armrest is never moved.

    # look first
    .venv/bin/python .../align_supports.py
    # then write
    .venv/bin/python .../align_supports.py --apply
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e206_common as C  # noqa: E402
import lowgeom_proxy_v2 as L  # noqa: E402
import manual_boxes as MB  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write manual_boxes.json (default: dry run)")
    ap.add_argument("--object-keys", default="", help="comma-separated; default = every manual record")
    ap.add_argument("--band", type=float, default=L.SUPPORT_BAND_M)
    ap.add_argument("--editor", default="align_supports")
    args = ap.parse_args()

    records = MB.load_manual()
    keys = (
        [k.strip() for k in args.object_keys.split(",") if k.strip()]
        if args.object_keys
        else sorted(records)
    )
    missing = [k for k in keys if k not in records]
    if missing:
        raise SystemExit(f"no manual record for: {', '.join(missing)}")
    if not keys:
        raise SystemExit("no manual records to align")

    if args.apply:
        backup = MB.MANUAL_PATH.with_suffix(".json.bak-align")
        shutil.copy2(MB.MANUAL_PATH, backup)
        print(f"[E206] backup -> {backup}")

    for key in keys:
        boxes, labels = MB.manual_boxes_for(key, records=records)
        mesh_path = C.object_mesh_path(key)
        before = L.support_contact_metrics(mesh_path, boxes, labels, band=args.band)
        aligned, changes = L.align_support_boxes(mesh_path, boxes, band=args.band)
        after = L.support_contact_metrics(mesh_path, aligned, labels, band=args.band)

        print(
            f"\n=== {key}  floor_y={before['support_floor_y_m']:+.4f}  "
            f"支撑 box {before['support_box_count']}/{len(boxes)}"
        )
        for ch in changes:
            lab = labels[ch["index"]] if ch["index"] < len(labels) else str(ch["index"])
            print(
                f"    {lab:8s} bottom {ch['bottom_before_m']:+.4f} -> "
                f"{ch['bottom_after_m']:+.4f}   ({ch['delta_mm']:+6.1f} mm)"
            )
        if not changes:
            print("    （无需改动）")
        print(
            f"    离地 spread {before['support_bottom_spread_m']*1000:5.1f} -> "
            f"{after['support_bottom_spread_m']*1000:.1f} mm   "
            f"worst |offset| {before['support_bottom_worst_abs_m']*1000:5.1f} -> "
            f"{after['support_bottom_worst_abs_m']*1000:.1f} mm   "
            f"G10 {'PASS' if after['support_bottom_worst_abs_m'] <= L.SUPPORT_TOL_M else 'FAIL'}"
        )
        if before["floating_leg_labels"]:
            print(
                f"    ⚠ 标签为 leg 但离地 >{args.band*1000:.0f}mm，未参与对齐: "
                f"{', '.join(before['floating_leg_labels'])}"
            )

        if args.apply and changes:
            notes = records[key].get("notes", "")
            records = MB.record_manual(
                key,
                aligned,
                labels,
                seed=records[key].get("seed", {}),
                editor=f"{records[key].get('editor', '')}+{args.editor}".strip("+"),
                notes=notes,
                records=records,
                save=False,
            )

    if args.apply:
        MB.save_manual(records)
        print(f"\n[E206] wrote {MB.MANUAL_PATH}")
    else:
        print("\n[E206] dry run — 加 --apply 才写盘")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
