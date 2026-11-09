#!/usr/bin/env python3
"""E216 Phase-2: generate missing partner augmented holosoma motions from raw.

The keep-best paired export needs the opposite-person (partner) augmented
holosoma motion for 17 cases whose partner was never preprocessed. Those
partners have no base retarget / warm-start, so ``run_upstream_retarget.py``
cannot drive them. This driver builds a ``pipeline.sh`` case-file directly and
runs the full convert -> retarget(original + trans) -> trim-original pass
(``RETARGET_AUGMENTATION=1``, omnirt_v2 env, 0.1 m trans-only), writing into the
E216 ``data_preprocess`` holosoma tree. The aug trans_* trims are then produced
by ``build_augmented_tasks.py --trim-only``.

Downstream-only: produces kinematic partner reference motion; no CEM, no scene,
no registry authority change (registry rows are a transient generation input).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "workspace/core4d/scripts/experiments/E216")
import e216_common as C  # noqa: E402


def _base_task(case_id: str) -> str:
    return f"dcv3_omnirt_v1_ref_fk_{case_id}"


def _holo_dir(case_id):
    return C.DP / f"holosoma_{_base_task(case_id)}"

CASE_FIELDS = [
    "# enabled", "date", "seq", "person", "object_name", "object_model_rel",
    "source_scene_task", "target_task", "trim_start", "trim_frames", "data_id", "mask_slug",
]
PERSON = {"1": "person1", "2": "person2"}


def parse_case(case_id: str, object_key: str) -> tuple[str, str, str]:
    rest = case_id[len(object_key) + 1:]
    m = re.match(r"(.+)_p(\d+)$", rest)
    date_seq, pn = m.group(1), m.group(2)
    date, seq = date_seq.rsplit("_", 1)
    return date, seq, PERSON[pn]


def resolve_object_name(case_id, object_key, date, seq, person, obj_meta):
    """Per-sequence object_name (CORE4D casing varies, e.g. box001 vs Box001).

    Prefer the already-converted filename (holosoma convert auto-detects the raw
    casing), then the partner's source task_info.json, then the per-object map.
    """
    import glob as _g
    import os as _o
    conv = _holo_dir(case_id) / "converted"
    hit = _g.glob(str(conv / f"{date}-{seq}-{person}-*_with_obj.npz"))
    if hit:
        base = _o.path.basename(hit[0])[:-len("_with_obj.npz")]
        return base.split("-")[-1]
    opp = {"person1": "person2", "person2": "person1"}[person]
    short = {"person1": "p1", "person2": "p2"}[opp]
    src = f"{case_id[:-3]}_{short}"
    ti = Path("example_datasets/processed/core4d/unitree_g1/humanoid_object") / \
        f"dcv3_omnirt_v1_ref_fk_{src}" / "task_info.json"
    if ti.is_file():
        try:
            name = json.loads(ti.read_text()).get("object_name")
            if name:
                return name
        except Exception:  # noqa: BLE001
            pass
    return obj_meta[object_key][0]


def pipeline_env() -> dict[str, str]:
    import os
    env = dict(os.environ)
    env.update({
        "REPO": str(C.REPO),
        "HOLOSOMA_DIR": str(C.HOLOSOMA_REPO),
        "CORE4D_REAL_ROOT": str(C.CORE4D_RAW_ROOT),
        "SMPLX_MODEL_DIR": str(C.SMPLX_MODEL_DIR),
        "RESULT_ROOT": C.rel(C.data_root()),
        "PYTHON_BIN": str(C.SPIDER_PYTHON_BIN),
        "RETARGET_PYTHON_BIN": str(C.RETARGET_PYTHON_BIN),
        "KEEP_GOING": "1",  # one hard case must not abort the batch
        "RETARGET_AUGMENTATION": "1",
        "RETARGET_MAX_WORKERS": os.environ.get("E216_RETARGET_MAX_WORKERS", "6"),
        "OMNIRT_TRANS_OFFSET_M": C.OMNIRT_TRANS_OFFSET_M,
        "OMNIRT_TRANS_ONLY": "1",
    })
    env.update(C.PIPELINE_DATASET_ENV)
    env.update(C.OMNIRT_V2_ENV)
    return env


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partners-json", type=Path, default=Path("/tmp/missing_partners.json"))
    ap.add_argument("--obj-meta-json", type=Path, default=Path("/tmp/obj_meta.json"))
    ap.add_argument("--cases", default="", help="comma filter (smoke test subset)")
    ap.add_argument("--skip-retarget", action="store_true", help="only run the aug trim step")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    miss = json.load(args.partners_json.open())
    obj_meta = json.load(args.obj_meta_json.open())
    partners = sorted({m["partner"]: m["object"] for m in miss}.items())
    if args.cases:
        keep = {c.strip() for c in args.cases.split(",")}
        partners = [p for p in partners if p[0] in keep]

    rows = []
    for case_id, object_key in partners:
        date, seq, person = parse_case(case_id, object_key)
        object_name = resolve_object_name(case_id, object_key, date, seq, person, obj_meta)
        _, object_model_rel = obj_meta[object_key]
        target_task = _base_task(case_id)  # dcv3_omnirt_v1_ref_fk_<case>
        rows.append(["1", date, seq, person, object_name, object_model_rel,
                     target_task, target_task, "auto", "auto", "0", target_task])

    case_file = C.DP_CASE_FILES / "cases_e216_partner_aug.tsv"
    case_file.parent.mkdir(parents=True, exist_ok=True)
    case_file.write_text(
        "\t".join(CASE_FIELDS) + "\n" + "\n".join("\t".join(r) for r in rows) + "\n",
        encoding="utf-8")
    print(f"case-file: {case_file} ({len(rows)} cases)")

    if not args.skip_retarget:
        cmd = ["bash", "workspace/core4d/data_preprocess/pipeline.sh",
               "--case-file", C.rel(case_file), "--skip-contact", "--skip-spider"]
        print("retarget cmd:", " ".join(cmd))
        if not args.dry_run:
            rc = subprocess.run(cmd, cwd=C.REPO, env=pipeline_env()).returncode
            if rc != 0:
                print(f"WARN pipeline.sh returncode={rc} (KEEP_GOING=1; check per-case below)")

    # aug trim (trans_*) for the generated cases: fixed-window trim at the same
    # trim_start as _original (ported from build_augmented_tasks.fixed_window_trim,
    # bypassing its frozen feasibility-TSV gate which omits these transient cases).
    print("trim: fixed-window trim per case (trans_0/1/2)")
    if not args.dry_run:
        import numpy as np
        holo_names = ["trans_0", "trans_1", "trans_2"]
        for case_id, object_key in partners:
            date, seq, person = parse_case(case_id, object_key)
            object_name = resolve_object_name(case_id, object_key, date, seq, person, obj_meta)
            holo = f"{date}-{seq}-{person}-{object_name}_with_obj"
            root = _holo_dir(case_id)
            r_orig = root / "retargeted" / f"{holo}_original.npz"
            t_orig_npz = root / "trimmed" / f"{holo}_original.npz"
            if not (r_orig.is_file() and t_orig_npz.is_file()):
                print(f"  [trim-skip] {case_id}: missing _original")
                continue
            with np.load(r_orig, allow_pickle=True) as d:
                t_full = int(d["qpos"].shape[0])
            with np.load(t_orig_npz, allow_pickle=True) as d:
                t_trim = int(d["qpos"].shape[0])
            trim_start = t_full - t_trim
            for hn in holo_names:
                src = root / "retargeted" / f"{holo}_{hn}.npz"
                dst = root / "trimmed" / f"{holo}_{hn}.npz"
                if not src.is_file() or dst.is_file():
                    continue
                with np.load(src, allow_pickle=True) as d:
                    saved = {k: (d[k][trim_start:] if hasattr(d[k], "shape")
                                 and d[k].ndim >= 1 and d[k].shape[0] == t_full else d[k])
                             for k in d.files}
                if int(saved["qpos"].shape[0]) != t_trim:
                    print(f"  [trim-fail] {case_id}/{hn}: {saved['qpos'].shape[0]}!={t_trim}")
                    continue
                np.savez(str(dst), **saved)
            print(f"  [trim] {case_id}: trim_start={trim_start} ({t_full}->{t_trim})")

    # verify
    import glob
    vmap = {"trans0": "trans_0", "trans1": "trans_1", "trans2": "trans_2"}
    done, pend = [], []
    for m in miss:
        if args.cases and m["partner"] not in keep:
            continue
        d = _holo_dir(m["partner"]) / "trimmed"
        fv = vmap[m["variant"]]
        ok = bool(glob.glob(str(d / f"*_{fv}.npz"))) and (_holo_dir(m["partner"]) / "trim_window.json").is_file()
        (done if ok else pend).append((m["partner"], m["variant"]))
    print(f"\nVERIFY: {len(done)} ready, {len(pend)} pending")
    for p in pend:
        print("  PENDING", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
