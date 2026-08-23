"""Materialize a v1-layout raw root from CORE4D human_object_motions v2.

CORE4D v2 (`CORE4D_Real_human_object_motions_v2`) stores per-person SMPL-X fits
under `<date>/<seq>/person_{1,2}/result.npz` (same key layout as v1's
`person{N}_poses.npz["arr_0"]`), but carries NO object poses. Object poses and
`object_metadata.json` are reused frame-for-frame from v1 (same captures, same
frame count -- asserted per case).

This script builds a raw root that mimics the v1 layout expected by the
data_construction_v3 pipeline (`build_inventory.py`, `run_raw_contact.py`,
`data_preprocess/pipeline.sh`), so the pipeline runs unchanged on v2 motion:

    <out_root>/
      human_object_motions/<date>/<seq>/
        person1_poses.npz        # repackaged from v2 result.npz -> arr_0 dict
        person2_poses.npz
        smooth_objposes.npy      # symlink -> v1
        object_metadata.json     # symlink -> v1
        aligned_frame_ids.txt    # symlink -> v1 (if present)
      object_models/             # symlink -> v1

Only in-scope objects (E203: box001/004/021/023/024/026 + bucket except
006/008) are materialized. A manifest records source paths + sha256.

Usage:
    uv run workspace/core4d/scripts/data_construction_v3/migration/materialize_v2_raw_root.py \
        --v1-root /.../CORE4D/CORE4D_Real \
        --v2-motions /.../CORE4D/CORE4D_Real_human_object_motions_v2 \
        --out-root /.../CORE4D/CORE4D_Real_v2_materialized
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

# E203 in-scope object set (case-insensitive object_name).
DEFAULT_BOX_OBJECTS = {"box001", "box004", "box021", "box023", "box024", "box026"}
DEFAULT_INCLUDE_BUCKET = True
DEFAULT_EXCLUDE_BUCKET = {"bucket006", "bucket008"}

# v2 result.npz keys expected (must match v1 arr_0 dict keys).
REQUIRED_KEYS = [
    "vertices", "joints", "betas", "expression",
    "global_orient", "transl", "body_pose",
    "left_hand_pose", "right_hand_pose",
]

PERSON_MAP = {"person1": "person_1", "person2": "person_2"}


def object_category(obj_name: str) -> str:
    return "".join(c for c in obj_name if not c.isdigit()).lower()


def is_in_scope(obj_name: str, box_objects: set[str],
                include_bucket: bool, exclude_bucket: set[str]) -> bool:
    o = obj_name.lower()
    if o in box_objects:
        return True
    if include_bucket and object_category(obj_name) == "bucket" and o not in exclude_bucket:
        return True
    return False


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def repackage_person(v2_result: Path, out_npz: Path) -> int:
    """Load v2 result.npz, save as v1-style {arr_0: dict}. Returns frame count T."""
    z = np.load(v2_result, allow_pickle=True)
    missing = [k for k in REQUIRED_KEYS if k not in z.files]
    if missing:
        raise ValueError(f"missing keys {missing} in {v2_result}")
    d = {k: z[k] for k in z.files}
    T = int(d["transl"].shape[0])
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, arr_0=np.array(d, dtype=object))
    return T


def symlink_force(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1-root", required=True,
                    help="CORE4D_Real root (has human_object_motions/ + object_models/)")
    ap.add_argument("--v2-motions", required=True,
                    help="CORE4D_Real_human_object_motions_v2 root")
    ap.add_argument("--out-root", required=True, help="materialized v1-layout root")
    ap.add_argument("--box-objects", default=",".join(sorted(DEFAULT_BOX_OBJECTS)))
    ap.add_argument("--exclude-bucket", default=",".join(sorted(DEFAULT_EXCLUDE_BUCKET)))
    ap.add_argument("--no-bucket", action="store_true", help="exclude all buckets")
    ap.add_argument("--dry-run", action="store_true",
                    help="only enumerate + write inscope list, do not materialize")
    args = ap.parse_args()

    v1_root = Path(args.v1_root)
    v2_motions = Path(args.v2_motions)
    out_root = Path(args.out_root)
    v1_motions = v1_root / "human_object_motions"
    box_objects = {s.strip().lower() for s in args.box_objects.split(",") if s.strip()}
    exclude_bucket = {s.strip().lower() for s in args.exclude_bucket.split(",") if s.strip()}
    include_bucket = not args.no_bucket

    assert v1_motions.is_dir(), f"missing {v1_motions}"
    assert v2_motions.is_dir(), f"missing {v2_motions}"

    out_motions = out_root / "human_object_motions"
    manifest_rows: list[dict] = []
    skipped: list[dict] = []
    n_person = 0
    inscope_cases = 0

    for date_dir in sorted(v1_motions.iterdir()):
        if not date_dir.is_dir():
            continue
        date = date_dir.name
        for seq_dir in sorted(date_dir.iterdir()):
            if not seq_dir.is_dir():
                continue
            seq = seq_dir.name
            meta_path = seq_dir / "object_metadata.json"
            obj_path = seq_dir / "smooth_objposes.npy"
            if not meta_path.exists() or not obj_path.exists():
                continue
            obj_name = json.load(open(meta_path))["obj_name"]
            if not is_in_scope(obj_name, box_objects, include_bucket, exclude_bucket):
                continue

            # v2 presence
            v2_seq = v2_motions / date / seq
            persons = []
            for p_v1, p_v2 in PERSON_MAP.items():
                if (v2_seq / p_v2 / "result.npz").exists():
                    persons.append((p_v1, p_v2))
            if not persons:
                continue
            inscope_cases += 1

            T_obj = int(np.load(obj_path).shape[0])
            out_seq = out_motions / date / seq

            for p_v1, p_v2 in persons:
                v2_res = v2_seq / p_v2 / "result.npz"
                out_npz = out_seq / f"{p_v1}_poses.npz"
                # frame alignment check (dry-run only reads T from v2)
                try:
                    z = np.load(v2_res, allow_pickle=True)
                    T_v2 = int(z["transl"].shape[0])
                except Exception as e:  # noqa: BLE001
                    skipped.append({"date": date, "seq": seq, "person": p_v1,
                                    "reason": f"v2_load_error:{e}"})
                    continue
                if T_v2 != T_obj:
                    skipped.append({"date": date, "seq": seq, "person": p_v1,
                                    "reason": f"frame_mismatch T_v2={T_v2} T_obj={T_obj}"})
                    continue
                if not args.dry_run:
                    T_written = repackage_person(v2_res, out_npz)
                    assert T_written == T_v2
                manifest_rows.append({
                    "date": date, "seq": seq, "person": p_v1,
                    "object_name": obj_name, "T": T_v2,
                    "v2_result": str(v2_res),
                    "out_person_npz": str(out_npz),
                })
                n_person += 1

            if not args.dry_run:
                # object poses + metadata reused from v1 (symlink)
                symlink_force(obj_path.resolve(), out_seq / "smooth_objposes.npy")
                symlink_force(meta_path.resolve(), out_seq / "object_metadata.json")
                afi = seq_dir / "aligned_frame_ids.txt"
                if afi.exists():
                    symlink_force(afi.resolve(), out_seq / "aligned_frame_ids.txt")

    if not args.dry_run:
        # object_models symlink at root
        symlink_force((v1_root / "object_models").resolve(), out_root / "object_models")

    out_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "v1_root": str(v1_root), "v2_motions": str(v2_motions),
        "out_root": str(out_root),
        "box_objects": sorted(box_objects), "exclude_bucket": sorted(exclude_bucket),
        "include_bucket": include_bucket,
        "inscope_cases": inscope_cases,
        "materialized_person_seqs": n_person,
        "skipped": skipped,
        "rows": manifest_rows,
    }
    man_path = out_root / "materialize_manifest.json"
    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # also a compact inscope TSV for downstream queue selection
    tsv_path = out_root / "inscope_cases.tsv"
    with open(tsv_path, "w") as f:
        f.write("date\tseq\tperson\tobject_name\tT\n")
        for r in manifest_rows:
            f.write(f"{r['date']}\t{r['seq']}\t{r['person']}\t{r['object_name']}\t{r['T']}\n")

    print(f"in-scope case-dirs: {inscope_cases}")
    print(f"materialized person-seqs: {n_person}")
    print(f"skipped: {len(skipped)}")
    for s in skipped[:20]:
        print("  SKIP", s)
    print(f"manifest -> {man_path}")
    print(f"inscope tsv -> {tsv_path}")


if __name__ == "__main__":
    main()
