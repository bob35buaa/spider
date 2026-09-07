#!/usr/bin/env python3
"""E213 Phase B: augmented kinematic retargets for the 11 partner-only cases.

The paired export's partner side is a *kinematic* trajectory that the Holosoma
exporter re-anchors to the source object ([[two-person-aug-partner-reanchor]]);
it is never independently CEM'd.  So Phase B stops at the trimmed aug retarget
npz -- no SPIDER task, no scene, no CEM.

Per partner (11 total; the other partners are themselves source rows already
augmented in E208):

1. **seed** its OmniRetarget SMPL-X ``converted/`` + the ``_original`` retarget
   (+ ``trim_window.json``) from wherever it was retargeted (E206 s3_retarget for
   10; the E206 ``partner_omnirt_direct_v2`` tree for the temp case) into an E213
   partner tree, so ``pipeline.sh`` skips convert and warm-starts from ``_original``.
2. **aug retarget** via ``pipeline.sh --skip-spider --skip-contact
   RETARGET_AUGMENTATION=1`` under the partner's own variant (omnirt_v1 first,
   omnirt_v2 rescue for a variant CVXPY declared infeasible; partners already v2
   run v2 directly).  Rotation runs too (holosoma 9e544b1 fix).
3. **fixed-window trim** every produced variant at ``_original``'s trim_start.

Usage:
    .venv/bin/python .../E213/build_partner_aug.py --dry-run
    ... --partners <case_id,...> --timeout-min 180
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import e213_common as C  # noqa: E402

RETARGET_VARIANTS = ("omnirt_v1", "omnirt_v2")
OMNIRT_ENV = {"omnirt_v1": C.E208.OMNIRT_V1_ENV, "omnirt_v2": C.E208.OMNIRT_V2_ENV}
ALL_VARIANTS = [("trans0", "trans_0"), ("trans1", "trans_1"), ("trans2", "trans_2"),
                ("rot0", "rot_0"), ("rot1", "rot_1")]
SKIP_MARKER = re.compile(r"\[skip\] augmentation (?P<name>\S+) infeasible/failed")
ROTATION_BUG = re.compile(r"Expected last dimension of .angles. to match number of sequence axes")

CASE_FIELDS = ["# enabled", "date", "seq", "person", "object_name", "object_model_rel",
               "source_scene_task", "target_task", "trim_start", "trim_frames", "data_id", "mask_slug"]

OBJ_POS = slice(36, 39)


def partner_meta(gap: dict[str, str]) -> dict[str, str]:
    """date/seq/person/object_name/object_model_rel + holo task name for a partner.

    Prefer the partner's dcv3 task_info.json; the direct_omnirt_partner_temp case
    has no dcv3 dir, so derive from the seeded trimmed npz filename + borrow the
    object model path from its paired source case (same object).
    """
    base = gap["base_target_task"]
    info = C.TASK_ROOT / base / "task_info.json"
    holo_task = Path(gap["partner_trimmed_npz"]).name.replace("_original.npz", "")
    if info.is_file():
        meta = C.E208.load_case_meta(base)
        meta["holosoma_task"] = holo_task
        return meta
    # constructed meta (temp partner): parse "<date>-<seq>-<person>-<object>_with_obj"
    stem = holo_task.replace("_with_obj", "")
    date, seq, person, object_name = stem.split("-", 3)
    src = next(c for c in C.load_cases() if c["case_id"] == gap["of_source_case"])
    src_meta = C.E208.load_case_meta(src["base_target_task"])
    return {
        "date": date, "seq": seq, "person": person, "object_name": object_name,
        "object_model_rel": src_meta["object_model_rel"],
        "source_scene_task": base, "holosoma_task": holo_task,
    }


def dest_holo(base: str, variant: str) -> Path:
    return C.PARTNER_DP / variant / f"holosoma_{base}"


def seed(gap: dict[str, str], meta: dict[str, str], variant: str) -> None:
    """Hardlink (fallback copy) converted + _original + trim_window into the E213 tree."""
    import shutil
    src_holo = C.repo_path(gap["partner_trimmed_npz"]).parent.parent
    dst = dest_holo(gap["base_target_task"], variant)
    holo = meta["holosoma_task"]
    pairs = [
        (src_holo / "converted", dst / "converted", None),  # whole dir
        (src_holo / "retargeted" / f"{holo}_original.npz", dst / "retargeted" / f"{holo}_original.npz", "file"),
        (src_holo / "trimmed" / f"{holo}_original.npz", dst / "trimmed" / f"{holo}_original.npz", "file"),
        (src_holo / "trim_window.json", dst / "trim_window.json", "file"),
    ]
    for src, out, kind in pairs:
        if kind is None:
            out.mkdir(parents=True, exist_ok=True)
            for f in sorted(src.glob("*.npz")):
                tgt = out / f.name
                if tgt.exists():
                    continue
                try:
                    os.link(f, tgt)
                except OSError:
                    shutil.copy2(f, tgt)
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            if out.exists():
                continue
            try:
                os.link(src, out)
            except OSError:
                shutil.copy2(src, out)


def write_case_file(gap: dict[str, str], meta: dict[str, str]) -> Path:
    path = C.PARTNER_DP / "case_files" / f"partner_{meta['object_name']}_{meta['seq']}_{meta['person']}.tsv"
    path.parent.mkdir(parents=True, exist_ok=True)
    row = ["1", meta["date"], meta["seq"], meta["person"], meta["object_name"],
           meta["object_model_rel"], meta["source_scene_task"], gap["base_target_task"],
           "auto", "auto", "0", gap["base_target_task"]]
    path.write_text("\t".join(CASE_FIELDS) + "\n" + "\t".join(row) + "\n", encoding="utf-8")
    return path


def pipeline_env(variant: str) -> dict[str, str]:
    env = dict(os.environ)
    env.update({
        "REPO": str(C.REPO), "HOLOSOMA_DIR": str(C.E208.HOLOSOMA_REPO),
        "CORE4D_REAL_ROOT": str(C.E208.env_path("CORE4D_RAW_ROOT")),
        "SMPLX_MODEL_DIR": str(C.E208.env_path("SMPLX_MODEL_DIR")),
        "RESULT_ROOT": C.rel(C.PARTNER_DP / variant),
        "PYTHON_BIN": str(C.SPIDER_PYTHON_BIN),
        "RETARGET_PYTHON_BIN": str(C.E208.retarget_python_bin()),
        "KEEP_GOING": "0", "RETARGET_AUGMENTATION": "1",
        "RETARGET_MAX_WORKERS": os.environ.get("E213_RETARGET_MAX_WORKERS", "6"),
        "SPIDER_DATASET": "core4d", "SPIDER_SOURCE_DATASET": "core4d",
    })
    env.update(OMNIRT_ENV[variant])
    return env


def produced_variants(base: str, meta: dict[str, str], variant: str) -> set[str]:
    root = dest_holo(base, variant) / "retargeted"
    holo = meta["holosoma_task"]
    return {h for _s, h in ALL_VARIANTS if (root / f"{holo}_{h}.npz").is_file()}


def run_pipeline(gap: dict[str, str], meta: dict[str, str], variant: str, timeout_min: int) -> str:
    case_file = write_case_file(gap, meta)
    cmd = ["bash", "workspace/core4d/data_preprocess/pipeline.sh",
           "--case-file", C.rel(case_file), "--skip-contact", "--skip-spider"]
    log = C.PARTNER_DP / "logs" / variant / f"{gap['base_target_task']}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        stream.write(f"# partner={gap['partner_case_id']} variant={variant}\n# cmd={' '.join(cmd)}\n\n")
        stream.flush()
        try:
            proc = subprocess.run(cmd, cwd=C.REPO, env=pipeline_env(variant), stdout=stream,
                                  stderr=subprocess.STDOUT, timeout=timeout_min * 60, check=False)
            return "ok" if proc.returncode == 0 else f"rc{proc.returncode}"
        except subprocess.TimeoutExpired:
            stream.write(f"\n# TIMEOUT after {timeout_min} min\n")
            return "timeout"


def trim(base: str, meta: dict[str, str], variant: str) -> list[str]:
    root = dest_holo(base, variant)
    holo = meta["holosoma_task"]
    with np.load(root / "retargeted" / f"{holo}_original.npz", allow_pickle=True) as d:
        t_orig = int(d["qpos"].shape[0])
    with np.load(root / "trimmed" / f"{holo}_original.npz", allow_pickle=True) as d:
        t_trim = int(d["qpos"].shape[0])
    trim_start = t_orig - t_trim
    (root / "trimmed").mkdir(parents=True, exist_ok=True)
    done: list[str] = []
    for _s, holo_name in ALL_VARIANTS:
        src = root / "retargeted" / f"{holo}_{holo_name}.npz"
        dst = root / "trimmed" / f"{holo}_{holo_name}.npz"
        if not src.is_file():
            continue
        if not dst.is_file():
            with np.load(src, allow_pickle=True) as data:
                n = int(data["qpos"].shape[0])
                out = {k: (data[k][trim_start:] if getattr(data[k], "ndim", 0) >= 1
                           and data[k].shape and data[k].shape[0] == n else data[k])
                       for k in data.files}
            if int(out["qpos"].shape[0]) != t_trim:
                raise ValueError(f"{base}/{holo_name}: trimmed {out['qpos'].shape[0]} != {t_trim}")
            np.savez(str(dst), **out)
        done.append(holo_name)
    return done


def process(gap: dict[str, str], timeout_min: int, dry_run: bool) -> dict[str, Any]:
    meta = partner_meta(gap)
    base = gap["base_target_task"]
    v1_variant = gap["retarget_variant"]
    rec: dict[str, Any] = {"partner_case_id": gap["partner_case_id"], "object_key": gap["object_key"],
                           "base_target_task": base, "primary_variant": v1_variant}
    if dry_run:
        rec["status"] = "dry_run"
        return rec

    # pass 1: partner's own variant
    seed(gap, meta, v1_variant)
    rec["pass1_status"] = run_pipeline(gap, meta, v1_variant, timeout_min)
    got1 = produced_variants(base, meta, v1_variant)

    # pass 2 rescue: only if partner is v1 and something is missing
    got2: set[str] = set()
    missing = {h for _s, h in ALL_VARIANTS} - got1
    if v1_variant == "omnirt_v1" and missing:
        seed(gap, meta, "omnirt_v2")
        rec["pass2_status"] = run_pipeline(gap, meta, "omnirt_v2", timeout_min)
        got2 = produced_variants(base, meta, "omnirt_v2")

    # trim whichever tree produced each variant; record provenance
    trimmed: dict[str, str] = {}
    for variant, got in ((v1_variant, got1), ("omnirt_v2", got2)):
        if not got:
            continue
        for h in trim(base, meta, variant):
            # pass1 wins; rescue only fills gaps
            if h not in trimmed or variant == v1_variant:
                trimmed[h] = variant
    rec["produced"] = ",".join(sorted(trimmed))
    rec["n_produced"] = len(trimmed)
    rec["provenance"] = ";".join(f"{h}={v}" for h, v in sorted(trimmed.items()))
    rec["status"] = "ok" if trimmed else "no_variants"
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partners", default="", help="comma-separated partner case_ids")
    ap.add_argument("--timeout-min", type=int, default=180)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    gap = C.partner_gap_cases()
    if args.partners:
        keep = {p.strip() for p in args.partners.split(",") if p.strip()}
        gap = [g for g in gap if g["partner_case_id"] in keep]
    if not gap:
        raise SystemExit("no partner cases selected")

    if not args.dry_run:
        lock = C.RESULTS / ".locks/build_partner_aug.lock"
        C.SingleInstance(lock).__enter__()

    print(f"E213 Phase B: {len(gap)} partner cases")
    records: list[dict[str, Any]] = []
    # serial within object (pipeline races on per-object files); objects are few
    for g in sorted(gap, key=lambda x: (x["object_key"], x["partner_case_id"])):
        rec = process(g, args.timeout_min, args.dry_run)
        records.append(rec)
        print(f"  [{rec.get('status','?'):10s}] {g['partner_case_id']:36s} "
              f"produced={rec.get('n_produced','-')} {rec.get('provenance','')}", flush=True)

    if not args.dry_run and records:
        C.PARTNER_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        fields: list[str] = []
        for r in records:
            r.setdefault("updated_at", C.now())
            for k in r:
                if k not in fields:
                    fields.append(k)
        C.write_tsv(C.PARTNER_MANIFEST, records, fields)
        print(f"\n-> {C.rel(C.PARTNER_MANIFEST)}")
    total = sum(r.get("n_produced", 0) for r in records)
    print(f"partner aug variants produced: {total} across {len(records)} partners")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
