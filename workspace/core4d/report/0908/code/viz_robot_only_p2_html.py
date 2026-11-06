"""Interactive, viewpoint-adjustable HTML viewer for the robot_only_p2 clip (box001 etc.).

Browser twin of viz_robot_only_p2.py: exports the SPIDER-CEM G1 (person2) + its
manipulated object, for every frame of the clip, into a self-contained Three.js page
(shared mesh topology, per-frame transforms only -- a whole 107-frame clip adds little
over a single frame). Scrub to any frame (frame_{k:04d}.png), orbit freely, toggle
background / floor / left-arm, download a square PNG at any viewpoint, and copy the
viz_robot_only_p2.py CLI to reproduce an angle offline.

Reuses the viewer template + payload builder from viz_methods_compare_html.py; only the
data loading differs (single CEM npz, box001-specific scene, camera from the mixed
render's camera_align.json -- identical to viz_robot_only_p2.py).

Run (headless), mirroring render_box001.sh's robot step:
    PYOPENGL_PLATFORM=osmesa MUJOCO_GL=osmesa .venv/bin/python \
        workspace/core4d/report/0908/code/viz_robot_only_p2_html.py \
        --case box001_20231023_108 --primary p2 --exp E198 \
        --cem-dir workspace/core4d/results/E198/s6_downstream/cem/full_g1a2_box001 \
        --cem-suffix G1A2 --scene-prefix dcv3_omnirt_v1_ref_fk --window 0,107
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")

import argparse  # noqa: E402

import numpy as np  # noqa: E402
import mujoco  # noqa: E402

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from viz_methods_compare_html import DEFAULT_EL, HTML, build_data_js  # noqa: E402
from viz_mixed_robot_smplx import mean_facing_yaw  # noqa: E402

DEFAULT_PROCESSED_ROOT = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
ROBOT_PAD = 1.15                 # viz_robot_only_p2.py framing pad (vs 1.10 for methods_compare)
DEFAULT_AZ_OFFSET = 45.0         # fallback az = facing + offset when no camera_align.json


def main():
    ap = argparse.ArgumentParser(description="Interactive robot_only_p2 clip (Three.js HTML).")
    ap.add_argument("--case", default="box001_20231023_108")
    ap.add_argument("--primary", default="p2")
    ap.add_argument("--exp", default="E198")
    ap.add_argument("--cem-dir", default=None, help="CEM results dir (default derived from --exp)")
    ap.add_argument("--cem-suffix", default="G1A2")
    ap.add_argument("--processed-root", default=str(DEFAULT_PROCESSED_ROOT))
    ap.add_argument("--scene-prefix", default="dcv3_omnirt_v1_ref_fk")
    ap.add_argument("--window", default="0,107", help="cem frame window 'lo,hi' (default: 0,107)")
    ap.add_argument("--file-frame", type=int, default=0,
                    help="frame the viewer OPENS on (all frames are selectable in-browser)")
    ap.add_argument("--cam-az-offset", type=float, default=DEFAULT_AZ_OFFSET)
    ap.add_argument("--cam-el", type=float, default=DEFAULT_EL)
    ap.add_argument("--out", default=None, help="viz dir (default: .../viz/{case}_{primary})")
    args = ap.parse_args()

    cem_dir = Path(args.cem_dir) if args.cem_dir else \
        REPO / f"workspace/core4d/results/{args.exp}/s6_downstream/cem/full"
    scene = (Path(args.processed_root) / f"{args.scene_prefix}_{args.case}_{args.primary}"
             / f"scene_act_{args.exp}_lowerbody_physics.xml")
    out_dir = Path(args.out) if args.out else \
        REPO / f"workspace/core4d/report/0908/paper_results/viz/{args.case}_{args.primary}"

    q = np.load(cem_dir / f"{args.exp}_{args.case}_{args.primary}_{args.cem_suffix}.npz")["qpos"]
    q = q[:, 0, :] if q.ndim == 3 else q
    lo, hi = (int(x) for x in args.window.split(","))
    hi = min(hi, len(q))
    n_frames = hi - lo
    assert 0 <= args.file_frame < n_frames, \
        f"file-frame {args.file_frame} outside [0,{n_frames})"

    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    facing = mean_facing_yaw(q[lo:hi])

    # Camera: reuse the mixed render's exact viewpoint (absolute cem az/el) from its
    # camera_align.json if present -- identical to viz_robot_only_p2.py -- expressed here
    # as an az-OFFSET from this robot's facing yaw so the viewer convention is uniform.
    align_path = out_dir / "camera_align.json"
    if align_path.exists():
        j = json.loads(align_path.read_text())
        default_az = float(j["cem_az_deg"]) - facing
        default_el = float(j["cem_el_deg"])
        print(f"[cam] aligned via {align_path}: az={j['cem_az_deg']:.1f} el={j['cem_el_deg']:.1f} "
              f"(facing={facing:.1f} -> az_offset={default_az:.1f})")
    else:
        default_az, default_el = args.cam_az_offset, args.cam_el
        print(f"[cam] no align json; az_offset={default_az:.1f} el={default_el:.1f}")

    cli_fixed = (f"--case {args.case} --primary {args.primary} --exp {args.exp} "
                 f"--cem-dir {args.cem_dir or cem_dir} --cem-suffix {args.cem_suffix} "
                 f"--scene-prefix {args.scene_prefix} --window {lo},{hi}")
    data_js = build_data_js(
        model, data, [q[f] for f in range(lo, hi)], facing=facing,
        default_az=round(default_az, 2), default_el=default_el, pad=ROBOT_PAD, T=np.eye(4),
        case=args.case, method=f"robot_only_{args.primary}", start_frame=args.file_frame,
        cli_tool="workspace/core4d/report/0908/code/viz_robot_only_p2.py",
        cli_fixed=cli_fixed, cli_az="--cam-az-offset", cli_el="--cam-el")

    out = out_dir / "robot_frame_viewer.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(HTML.replace("__DATA__", json.dumps(data_js)))
    print(f"[ok] {out}  ({n_frames} frames, opens on f{args.file_frame})")
    print("open in a browser; scrub frames, orbit, toggle floor/bg, download a PNG at "
          "any viewpoint, and copy the viz_robot_only_p2.py CLI for a high-res render.")


if __name__ == "__main__":
    main()
