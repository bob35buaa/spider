#!/usr/bin/env python3
"""E215 partner phase: rot kinematic retargets for the gap partners of the export set.

The paired RL export's partner side is a *kinematic* trajectory the Holosoma
exporter re-anchors to the source object ([[two-person-aug-partner-reanchor]]);
it is never CEM'd. So this stops at the trimmed rot retarget npz -- no SPIDER
task, no scene, no CEM (mirrors E213 Phase B).

For each gap partner (a partner NOT itself a delivered E215 source case; the
others already have rot trimmed npz in the E215 source tree):

1. **seed** its OmniRetarget ``converted/`` + the ``_original`` retarget
   (+ ``trim_window.json``) from where it was retargeted for E199/E202 (box ->
   E199_DP, bucket -> E202_DP) into an E215 partner tree, so ``pipeline.sh``
   skips convert and warm-starts from ``_original``. Existing ``trans_*``
   retargets are hardlinked too so the upstream only computes ``rot_*``.
2. **aug retarget** via ``pipeline.sh --skip-spider --skip-contact
   RETARGET_AUGMENTATION=1`` under the seed's own variant (omnirt_v1, with an
   omnirt_v2 rescue if a rot variant is declared infeasible). Rotation runs
   (holosoma 9e544b1 fix, same as the E215 source retarget).
3. **fixed-window trim** each produced rot variant at ``_original``'s trim_start.

Usage:
    .venv/bin/python .../E215/build_partner_aug.py --dry-run
    ... --partners <case_id,...> --timeout-min 180
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent / "E208"))

import e208_common as E208  # noqa: E402  (OMNIRT_V1/V2 env authority)
import e215_common as C  # noqa: E402
import e215_export_common as X  # noqa: E402

ROT_VARIANTS = [("rot0", "rot_0"), ("rot1", "rot_1")]
TRANS_HOLO = ["trans_0", "trans_1", "trans_2"]  # seeded to skip; only rot is computed
CASE_FIELDS = ["# enabled", "date", "seq", "person", "object_name", "object_model_rel",
               "source_scene_task", "target_task", "trim_start", "trim_frames", "data_id", "mask_slug"]


def partner_meta(gap: dict[str, str]) -> dict[str, str]:
    """date/seq/person/object_name/object_model_rel + on-disk holo task for a partner.

    Prefer the partner's own dcv3 task_info; 2 gap partners have only the E199_DP
    retarget seed (no dcv3 task dir), so derive date/seq/person/object from the
    seeded ``_original`` npz filename and borrow object_model_rel from the paired
    source case's task_info (same object)."""
    seed_root = Path(gap["seed_holosoma_root"])
    orig = sorted((seed_root / "retargeted").glob("*_original.npz"))
    if not orig:
        raise SystemExit(f"{gap['partner_case_id']}: no _original retarget in seed {seed_root}")
    holo = orig[0].name.replace("_original.npz", "")   # authoritative on-disk holo task
    base = gap["base_target_task"]
    if (C.TASK_ROOT / base / "task_info.json").is_file():
        meta = dict(C.load_case_meta(base))
    else:
        stem = holo[: -len("_with_obj")] if holo.endswith("_with_obj") else holo
        date, seq, person, object_name = stem.split("-", 3)
        src_root = X._e215_source_holo(gap["of_source_case"])
        if src_root is None:
            raise SystemExit(f"{gap['partner_case_id']}: no source retarget for object_model_rel")
        src_meta = C.load_case_meta(src_root.name[len("holosoma_"):])
        meta = {"date": date, "seq": seq, "person": person, "object_name": object_name,
                "object_model_rel": src_meta["object_model_rel"], "source_scene_task": base}
    meta["holosoma_task"] = holo
    return meta


def dest_holo(base: str, variant: str) -> Path:
    return X.PARTNER_AUG_ROOT / variant / f"holosoma_{base}"


def seed(gap: dict[str, str], meta: dict[str, str], variant: str) -> None:
    """Hardlink (fallback copy) converted + _original (+ trans_*) + trim_window."""
    src_holo = Path(gap["seed_holosoma_root"])
    dst = dest_holo(gap["base_target_task"], variant)
    holo = meta["holosoma_task"]
    # whole converted dir
    (dst / "converted").mkdir(parents=True, exist_ok=True)
    for f in sorted((src_holo / "converted").glob("*.npz")):
        tgt = dst / "converted" / f.name
        if not tgt.exists():
            _link(f, tgt)
    # _original retarget + trimmed + trim_window, plus existing trans_* retargets
    files = [
        (src_holo / "retargeted" / f"{holo}_original.npz", dst / "retargeted" / f"{holo}_original.npz"),
        (src_holo / "trimmed" / f"{holo}_original.npz", dst / "trimmed" / f"{holo}_original.npz"),
        (src_holo / "trim_window.json", dst / "trim_window.json"),
    ]
    for h in TRANS_HOLO:
        src = src_holo / "retargeted" / f"{holo}_{h}.npz"
        if src.is_file():
            files.append((src, dst / "retargeted" / f"{holo}_{h}.npz"))
    for src, out in files:
        out.parent.mkdir(parents=True, exist_ok=True)
        if src.is_file() and not out.exists():
            _link(src, out)


def _link(src: Path, out: Path) -> None:
    try:
        os.link(src, out)
    except OSError:
        shutil.copy2(src, out)


def write_case_file(gap: dict[str, str], meta: dict[str, str]) -> Path:
    path = C.DP / "case_files" / f"partner_{meta['object_name']}_{meta['seq']}_{meta['person']}.tsv"
    path.parent.mkdir(parents=True, exist_ok=True)
    row = ["1", meta["date"], meta["seq"], meta["person"], meta["object_name"],
           meta["object_model_rel"], meta["source_scene_task"], gap["base_target_task"],
           "auto", "auto", "0", gap["base_target_task"]]
    path.write_text("\t".join(CASE_FIELDS) + "\n" + "\t".join(row) + "\n", encoding="utf-8")
    return path


def pipeline_env(variant: str) -> dict[str, str]:
    env = dict(os.environ)
    env.update({
        "REPO": str(C.REPO), "HOLOSOMA_DIR": str(C.HOLOSOMA_REPO),
        "CORE4D_REAL_ROOT": str(C.CORE4D_RAW_ROOT),
        "SMPLX_MODEL_DIR": str(C.SMPLX_MODEL_DIR),
        "RESULT_ROOT": C.rel(X.PARTNER_AUG_ROOT / variant),
        "PYTHON_BIN": str(C.SPIDER_PYTHON_BIN),
        "RETARGET_PYTHON_BIN": str(C.RETARGET_PYTHON_BIN),
        "KEEP_GOING": "0", "RETARGET_AUGMENTATION": "1",
        "RETARGET_MAX_WORKERS": os.environ.get("E215_RETARGET_MAX_WORKERS", "6"),
        "SPIDER_DATASET": "core4d", "SPIDER_SOURCE_DATASET": "core4d",
    })
    env.update(E208.OMNIRT_ENV_BY_VARIANT[variant])
    return env


def produced_rot(base: str, meta: dict[str, str], variant: str) -> set[str]:
    root = dest_holo(base, variant) / "retargeted"
    holo = meta["holosoma_task"]
    return {h for _s, h in ROT_VARIANTS if (root / f"{holo}_{h}.npz").is_file()}


def run_pipeline(gap: dict[str, str], meta: dict[str, str], variant: str, timeout_min: int) -> str:
    case_file = write_case_file(gap, meta)
    cmd = ["bash", C.rel(C.PIPELINE_SH), "--case-file", C.rel(case_file),
           "--skip-contact", "--skip-spider"]
    log = C.DP / "logs" / "partner_aug" / variant / f"{gap['base_target_task']}.log"
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
    for _s, holo_name in ROT_VARIANTS:
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
    base = gap["base_target_task"]
    meta = partner_meta(gap)
    v1 = gap["seed_variant"] or "omnirt_v1"
    rec: dict[str, Any] = {"partner_case_id": gap["partner_case_id"], "object_key": gap["object_key"],
                           "base_target_task": base, "primary_variant": v1,
                           "seed_holosoma_root": C.rel(gap["seed_holosoma_root"])}
    if dry_run:
        rec["status"] = "dry_run"
        return rec

    seed(gap, meta, v1)
    rec["pass1_status"] = run_pipeline(gap, meta, v1, timeout_min)
    got1 = produced_rot(base, meta, v1)

    got2: set[str] = set()
    missing = {h for _s, h in ROT_VARIANTS} - got1
    if v1 == "omnirt_v1" and missing:
        seed(gap, meta, "omnirt_v2")
        rec["pass2_status"] = run_pipeline(gap, meta, "omnirt_v2", timeout_min)
        got2 = produced_rot(base, meta, "omnirt_v2")

    trimmed: dict[str, str] = {}
    for variant, got in ((v1, got1), ("omnirt_v2", got2)):
        if not got:
            continue
        for h in trim(base, meta, variant):
            if h not in trimmed or variant == v1:   # pass1 wins; rescue fills gaps
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

    gap = X.partner_gap_cases()
    if args.partners:
        keep = {p.strip() for p in args.partners.split(",") if p.strip()}
        gap = [g for g in gap if g["partner_case_id"] in keep]
    if not gap:
        raise SystemExit("no gap partner cases selected")
    missing_seed = [g["partner_case_id"] for g in gap if not g["seed_holosoma_root"]]
    if missing_seed:
        raise SystemExit(f"gap partners with no E199/E202 seed: {missing_seed}")

    if not args.dry_run:
        lock = C.RESULTS / ".locks/build_partner_aug.lock"
        C.SingleInstance(lock).__enter__()

    print(f"E215 partner phase: {len(gap)} gap partner cases")
    records: list[dict[str, Any]] = []
    for g in sorted(gap, key=lambda x: (x["object_key"], x["partner_case_id"])):
        rec = process(g, args.timeout_min, args.dry_run)
        records.append(rec)
        print(f"  [{rec.get('status','?'):10s}] {g['partner_case_id']:36s} "
              f"produced={rec.get('n_produced','-')} {rec.get('provenance','')}", flush=True)

    if not args.dry_run and records:
        X.PARTNER_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        fields: list[str] = []
        for r in records:
            r.setdefault("updated_at", C.now())
            for k in r:
                if k not in fields:
                    fields.append(k)
        C.write_tsv(X.PARTNER_MANIFEST, records, fields)
        print(f"\n-> {C.rel(X.PARTNER_MANIFEST)}")
    total = sum(r.get("n_produced", 0) for r in records)
    print(f"partner rot variants produced: {total} across {len(records)} partners")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
