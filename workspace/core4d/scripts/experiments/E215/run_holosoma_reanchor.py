#!/usr/bin/env python3
"""E215-export step 2: run the Holosoma partner re-anchor exporter on the paired input.

The spider-side export (export_selected_arm_aug_rl.py) only produces the paired RL
INPUT; the actual two-person re-anchor (partner wrist -> partner object frame ->
target object pose, default ON) is done by the Holosoma exporter
``holosoma/workspace/v3/scripts/data/export_rl_motion_from_spider_tsv.py`` -- the
same downstream step E213 ran. This drives it per object (its half-extents dict
lacks box001/box024/bucket003/bucket007, so those are passed explicitly; values
computed from each object mesh bbox, cross-checked against E202 for buckets), with
an ABSOLUTE out-dir (the converter subprocess runs in a deeper cwd, so a relative
out-dir breaks its input path).

Outputs (mirroring E213): holosoma/workspace/v3/data/E215_rot_aug_partner_rl/<object>/
  exports/<uid>_{spider,omnirt}_partner_omnirt_mj_w_obj_w_partner.npz + manifest/summary.

Usage:
    .venv/bin/python .../E215/run_holosoma_reanchor.py --dry-run
    ... --objects box021,box023 --timeout-min 60
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

SP = Path("/mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/spider")
HOLO = Path("/mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/holosoma")
HS_PYTHON = Path("/mnt/ali-sh-1/dataset/zeus/xiayb/.holosoma_deps/miniconda3/envs/hsretargeting/bin/python")
EXPORTER = HOLO / "workspace/v3/scripts/data/export_rl_motion_from_spider_tsv.py"
EXPORT_DIR = SP / "workspace/core4d/results/E215/s6_downstream/export"
INPUT_TSV = EXPORT_DIR / "rl_export_input.tsv"
PARTNER_TSV = EXPORT_DIR / "partner_omnirt/rl_partner_omnirt_manifest.tsv"
OUT_ROOT = HOLO / "workspace/v3/data/E215_rot_aug_partner_rl"

# objects absent from the exporter's built-in OBJECT_HALF_EXTENTS (surface-distance
# diagnostics only). Values = object-mesh bbox half-extents (buckets match E202).
EXTENTS = {
    "box001": (0.2511235, 0.31321025, 0.40630706),
    "box024": (0.254485995, 0.25261463, 0.490977035),
    "bucket003": (0.270393755, 0.381407245, 0.23287703),
    "bucket007": (0.273275865, 0.2869532, 0.284911265),
}


def read_units() -> dict[str, list[str]]:
    # object_name casing varies across task_info (Box001 vs box001); normalize to
    # lowercase so one object is one group (the exporter lowercases internally too).
    by_obj: dict[str, list[str]] = defaultdict(list)
    with INPUT_TSV.open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if r.get("rl_export_decision") == "RL_EXPORT_READY":
                by_obj[r["object_name"].lower()].append(r["case_id"])
    return dict(sorted(by_obj.items()))


def run_object(obj: str, uids: list[str], timeout_min: int, dry: bool) -> dict:
    out_dir = OUT_ROOT / obj
    cmd = [str(HS_PYTHON), str(EXPORTER),
           "--input-tsv", str(INPUT_TSV), "--partner-omnirt-tsv", str(PARTNER_TSV),
           "--target-source", "both", "--partner-source", "omnirt",
           "--out-dir", str(out_dir), "--force"]
    for u in uids:
        cmd += ["--case-id", u]
    if obj in EXTENTS:
        cmd += ["--object-half-extents", *[repr(x) for x in EXTENTS[obj]]]
    if dry:
        return {"object": obj, "n_units": len(uids), "cmd": " ".join(cmd)}
    log = out_dir / "driver.log"
    out_dir.mkdir(parents=True, exist_ok=True)
    env = {"SPIDER_REPO": str(SP), "RETARGET_PYTHON": str(HS_PYTHON), "PATH": "/usr/bin:/bin"}
    import os
    env = {**os.environ, **env}
    with log.open("w", encoding="utf-8") as stream:
        stream.write(f"# object={obj} units={len(uids)}\n# {' '.join(cmd)}\n\n")
        stream.flush()
        try:
            proc = subprocess.run(cmd, cwd=HOLO, env=env, stdout=stream,
                                  stderr=subprocess.STDOUT, timeout=timeout_min * 60, check=False)
            status = "ok" if proc.returncode == 0 else f"rc{proc.returncode}"
        except subprocess.TimeoutExpired:
            status = "timeout"
    n_exp = len(list((out_dir / "exports").glob("*_mj_w_obj_w_partner.npz"))) if (out_dir / "exports").is_dir() else 0
    return {"object": obj, "n_units": len(uids), "status": status, "n_exports": n_exp}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", default="", help="comma-separated object filter")
    ap.add_argument("--timeout-min", type=int, default=90)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    by_obj = read_units()
    if args.objects:
        keep = {o.strip() for o in args.objects.split(",") if o.strip()}
        by_obj = {o: u for o, u in by_obj.items() if o in keep}
    if not by_obj:
        raise SystemExit("no objects/units selected")

    print(f"E215 re-anchor: {sum(len(v) for v in by_obj.values())} units across {len(by_obj)} objects")
    recs = []
    for obj, uids in by_obj.items():
        rec = run_object(obj, uids, args.timeout_min, args.dry_run)
        recs.append(rec)
        if args.dry_run:
            print(f"  {obj:12s} n={rec['n_units']}\n    {rec['cmd']}")
        else:
            print(f"  [{rec['status']:8s}] {obj:12s} units={rec['n_units']} exports={rec['n_exports']}", flush=True)
    if not args.dry_run:
        total = sum(r["n_exports"] for r in recs)
        oks = sum(1 for r in recs if r["status"] == "ok")
        print(f"\n{oks}/{len(recs)} objects ok; {total} exported motions -> {OUT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
