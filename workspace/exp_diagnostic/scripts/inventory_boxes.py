"""Inventory all available CORE4D box objects with bounding-box dimensions.
Compare against box023 and box025 to find size-band candidates the user wants.

Looks at:
- example_datasets/processed/core4d/assets/objects/box*/box*_m.obj
- For each, read mesh, compute axis-aligned bounding box, half-extents, volume,
  longest side, ratio against box023/box025.
"""
from pathlib import Path
import numpy as np
import json
import sys

ROOT = Path("/mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/spider")
ASSET = ROOT / "example_datasets/processed/core4d/assets/objects"

def read_obj_vertices(p):
    pts = []
    with open(p, "r") as f:
        for line in f:
            if line.startswith("v "):
                xs = line.split()
                pts.append([float(xs[1]), float(xs[2]), float(xs[3])])
    return np.array(pts)


def summarize(p):
    v = read_obj_vertices(p)
    mn = v.min(axis=0); mx = v.max(axis=0)
    half = (mx - mn) / 2.0
    extents = mx - mn
    vol = float(extents[0] * extents[1] * extents[2])
    return {
        "name": p.parent.name,
        "mesh_pts": int(v.shape[0]),
        "extents_m": [round(float(x), 3) for x in extents],
        "half_extents_m": [round(float(x), 3) for x in half],
        "volume_m3": round(vol, 4),
        "longest_side_m": round(float(extents.max()), 3),
        "shortest_side_m": round(float(extents.min()), 3),
    }


rows = []
for d in sorted(ASSET.glob("box*")):
    mesh = d / f"{d.name}_m.obj"
    if not mesh.exists():
        mesh = next(d.glob("*.obj"), None)
    if mesh is None:
        continue
    rows.append(summarize(mesh))

# Print
ref_023 = next(r for r in rows if r["name"] == "box023")
ref_025 = next(r for r in rows if r["name"] == "box025")
v23 = ref_023["volume_m3"]; v25 = ref_025["volume_m3"]
l23 = ref_023["longest_side_m"]; l25 = ref_025["longest_side_m"]

print(f"Reference: box023 vol={v23} m^3, longest={l23} m")
print(f"           box025 vol={v25} m^3, longest={l25} m\n")
print(f"{'name':<10} {'extents':<22} {'vol':<8} {'long':<6} {'short':<6}  in_band?")
print("-" * 80)
for r in rows:
    in_band = (r["volume_m3"] > v23) and (r["volume_m3"] < v25) and (r["longest_side_m"] > l23) and (r["longest_side_m"] < l25)
    flag = "YES" if in_band else ""
    if r["name"] in ("box023", "box025"):
        flag = "(ref)"
    print(f"{r['name']:<10} {str(r['extents_m']):<22} {r['volume_m3']:<8} {r['longest_side_m']:<6} {r['shortest_side_m']:<6}  {flag}")

with open(ROOT / "workspace/exp_diagnostic/findings/05_box_inventory.json", "w") as f:
    json.dump(rows, f, indent=2)
print(f"\nsaved findings/05_box_inventory.json")
