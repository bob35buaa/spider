#!/usr/bin/env python3
"""E215 P2: drive pipeline.sh's augmentation branch (omnirt_v2, single pass).

For each case, run ``pipeline.sh --skip-contact --skip-spider`` with
``RETARGET_AUGMENTATION=1`` under the omnirt_v2 env, warm-started from the
``_original`` seeded by seed_warmstart.py.  Upstream generates all five native
configs, but E215 only consumes rot_0/rot_1; the trans_* it recomputes (warm
started, harmless) are ignored downstream.

Care taken (verbatim discipline from E208):
* **Never ``--force``.**  The seeded ``_original`` is loaded from disk as the IK
  warm start (robot_retarget.py:563-569); ``--force`` would recompute it and
  break the v2->v2 seeding guarantee.
* **Same-object cases serial.**  ``sync_generated_object_model`` (cp -a) and
  ``ensure_g1_object_xml`` (first-write) race on per-object files, so object
  groups run concurrently but each group is a queue.
* **Own sentinel.**  In aug mode pipeline.sh's done-marker is ``{task}_rot_1.npz``,
  which may legitimately never appear if rot_1 is infeasible -- so the step would
  be re-entered forever.  ``DP/sentinels/{base}.done`` records what actually landed.
* **``--skip-contact``.**  ``trim_start`` equals the seeded window, so the reused
  3 cm mask is valid frame-for-frame; regenerating it would create a second copy
  to drift from the one the overrides point at.

Usage:
    .venv/bin/python .../E215/run_upstream_retarget.py --dry-run
    ... --cases a,b --max-workers 5
    ... --objects box021,box023
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import e215_common as C  # noqa: E402

SingleInstance = C.SingleInstance

# 12 tab-separated columns; header starts with "# enabled".  source_scene_task
# must be non-empty even under --skip-spider (pipeline.sh reads with IFS=$'\t'
# and an empty field shifts target_task).
CASE_FIELDS = [
    "# enabled", "date", "seq", "person", "object_name", "object_model_rel",
    "source_scene_task", "target_task", "trim_start", "trim_frames", "data_id", "mask_slug",
]

SKIP_MARKER = re.compile(r"\[skip\] augmentation (?P<name>\S+) infeasible/failed: (?P<reason>.*)")
# The scipy shape crash (fixed in ../holosoma 9e544b1); if this ever fires again
# it means the driver picked up the stale holosoma copy.
ROTATION_BUG_MARKER = re.compile(
    r"Expected last dimension of .angles. to match number of sequence axes specified")


def case_file_path(meta: dict[str, str], base: str) -> Path:
    return C.DP_CASE_FILES / f"cases_e215_{base}.tsv"


def write_case_file(case: dict[str, str], meta: dict[str, str]) -> Path:
    path = case_file_path(meta, case["base_target_task"])
    path.parent.mkdir(parents=True, exist_ok=True)
    row = [
        "1", meta["date"], meta["seq"], meta["person"],
        meta["object_name"], meta["object_model_rel"],
        meta["source_scene_task"],
        case["base_target_task"],
        "auto", "auto", "0",
        case["base_target_task"],
    ]
    path.write_text("\t".join(CASE_FIELDS) + "\n" + "\t".join(row) + "\n", encoding="utf-8")
    return path


def pipeline_env() -> dict[str, str]:
    env = dict(os.environ)
    env.update({
        "REPO": str(C.REPO),
        "HOLOSOMA_DIR": str(C.HOLOSOMA_REPO),
        "CORE4D_REAL_ROOT": str(C.CORE4D_RAW_ROOT),
        "SMPLX_MODEL_DIR": str(C.SMPLX_MODEL_DIR),
        "RESULT_ROOT": C.rel(C.data_root()),
        "PYTHON_BIN": str(C.SPIDER_PYTHON_BIN),
        "RETARGET_PYTHON_BIN": str(C.RETARGET_PYTHON_BIN),
        "KEEP_GOING": "0",
        "RETARGET_AUGMENTATION": "1",
        "RETARGET_MAX_WORKERS": os.environ.get("E215_RETARGET_MAX_WORKERS", "6"),
    })
    env.update(C.PIPELINE_DATASET_ENV)
    env.update(C.OMNIRT_V2_ENV)   # 5 keys; REPLACE_WRIST_WITH_FINGERTIP left at pipeline default
    return env


def variant_npz(case: dict[str, str], meta: dict[str, str], holo_name: str) -> Path:
    return C.holosoma_dir(case["case_id"]) / "retargeted" / f"{meta['holosoma_task']}_{holo_name}.npz"


def scan_log(log_path: Path) -> tuple[dict[str, str], bool]:
    if not log_path.is_file():
        return {}, False
    hits: dict[str, str] = {}
    rotation_bug = False
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = SKIP_MARKER.search(line)
        if match:
            hits[match.group("name")] = match.group("reason").strip()[:300]
        elif ROTATION_BUG_MARKER.search(line):
            rotation_bug = True
    return hits, rotation_bug


def classify(case: dict[str, str], meta: dict[str, str],
             log_hits: dict[str, str], rotation_bug: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for short, holo in C.BUILD_VARIANTS:
        npz = variant_npz(case, meta, holo)
        present = npz.is_file() and npz.stat().st_size > 0
        reason_hit = log_hits.get(holo, "")
        if present:
            state = C.BUILT_STATE
        elif rotation_bug:
            # NOT an IK result: upstream crashed in augment_object_poses before
            # solving.  This should never happen with the fixed ../holosoma; if it
            # does, the driver used the stale copy -- stop and investigate.
            state = "upstream_rotation_bug"
        elif reason_hit:
            state = "rot_infeasible"
        else:
            state = "needs_triage"  # missing npz, no log line: crash/OOM/disk
        rows.append({
            "case_id": case["case_id"], "object_key": case["object_key"],
            "arm_group": case["arm_group"], "base_target_task": case["base_target_task"],
            "variant": short, "holosoma_variant": holo,
            "npz_present": int(present),
            "npz_sha256": C.sha256(npz) if present else "",
            "log_hit": reason_hit, "rotation_bug": int(rotation_bug),
            "state": state, "built": int(state == C.BUILT_STATE),
            "updated_at": C.now(),
        })
    return rows


def run_case(case: dict[str, str], *, dry_run: bool, force_rerun: bool,
             timeout_min: int) -> dict[str, Any]:
    meta = C.load_case_meta(case["base_target_task"])
    base = case["base_target_task"]
    sentinel = C.sentinel_path(case["case_id"])
    log_path = C.DP_LOGS / f"{base}.log"

    result: dict[str, Any] = {
        "case_id": case["case_id"], "object_key": case["object_key"], "base_target_task": base,
    }
    if sentinel.is_file() and not force_rerun:
        result["status"] = "skip_sentinel"
        result["wall_s"] = 0.0
        hits, rot_bug = scan_log(log_path)
        result["rows"] = classify(case, meta, hits, rot_bug)
        return result

    # warm start must be present
    warm = C.holosoma_dir(case["case_id"]) / "retargeted" / f"{meta['holosoma_task']}_original.npz"
    if not warm.is_file():
        result["status"] = "fail_no_warmstart"
        result["rows"] = []
        return result

    case_file = write_case_file(case, meta)
    cmd = [
        "bash", "workspace/core4d/data_preprocess/pipeline.sh",
        "--case-file", C.rel(case_file),
        "--skip-contact", "--skip-spider",
    ]
    result["cmd"] = " ".join(cmd)
    if dry_run:
        result["status"] = "dry_run"
        result["wall_s"] = 0.0
        result["rows"] = []
        return result

    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = pipeline_env()
    started = time.time()
    with log_path.open("w", encoding="utf-8") as stream:
        stream.write(f"# retarget_variant={C.RETARGET_VARIANT}\n")
        stream.write(f"# env={json.dumps({k: env[k] for k in sorted(C.OMNIRT_V2_ENV)})}\n")
        stream.write(f"# RESULT_ROOT={env['RESULT_ROOT']}\n# cmd={' '.join(cmd)}\n\n")
        stream.flush()
        try:
            proc = subprocess.run(cmd, cwd=C.REPO, env=env, stdout=stream,
                                  stderr=subprocess.STDOUT, timeout=timeout_min * 60, check=False)
            returncode, timed_out = proc.returncode, False
        except subprocess.TimeoutExpired:
            returncode, timed_out = -1, True
            stream.write(f"\n# TIMEOUT after {timeout_min} min\n")
    wall = time.time() - started

    log_hits, rotation_bug = scan_log(log_path)
    rows = classify(case, meta, log_hits, rotation_bug)
    for row in rows:
        row["wall_s"] = round(wall, 1)
        row["log_ref"] = C.rel(log_path)

    produced = {r["holosoma_variant"]: r["npz_sha256"] for r in rows if r["npz_present"]}
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text(json.dumps({
        "retarget_variant": C.RETARGET_VARIANT, "case_id": case["case_id"],
        "finished_at": C.now(), "returncode": returncode, "timed_out": timed_out,
        "wall_s": round(wall, 1),
        "artifacts": {holo: produced.get(holo) for _s, holo in C.BUILD_VARIANTS},
        "log": C.rel(log_path),
    }, indent=2) + "\n", encoding="utf-8")

    result.update({
        "status": "timeout" if timed_out else ("ok" if returncode == 0 else f"rc{returncode}"),
        "returncode": returncode, "wall_s": round(wall, 1), "rows": rows,
        "log": C.rel(log_path), "n_built": sum(r["built"] for r in rows),
    })
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="")
    ap.add_argument("--objects", default="")
    ap.add_argument("--max-workers", type=int, default=5,
                    help="concurrent OBJECT groups (cases inside a group stay serial)")
    ap.add_argument("--timeout-min", type=int, default=180)
    ap.add_argument("--force-rerun", action="store_true", help="ignore sentinels (never --force upstream)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.dry_run:
        _lock = SingleInstance(C.DP / "run_upstream_retarget.lock")
        _lock.__enter__()

    cases = C.load_e215_cases()
    if args.cases:
        keep = {c.strip() for c in args.cases.split(",") if c.strip()}
        unknown = keep - {c["case_id"] for c in cases}
        if unknown:
            raise SystemExit(f"unknown/non-buildable case_ids: {sorted(unknown)}")
        cases = [c for c in cases if c["case_id"] in keep]
    if args.objects:
        keep_obj = {o.strip() for o in args.objects.split(",") if o.strip()}
        cases = [c for c in cases if c["object_key"] in keep_obj]
    if not cases:
        raise SystemExit("no cases selected")

    groups: dict[str, list[dict[str, str]]] = {}
    for case in cases:
        groups.setdefault(case["object_key"], []).append(case)
    print(f"upstream rot retarget: {len(cases)} cases across {len(groups)} object groups "
          f"(<= {args.max_workers} groups in parallel, serial within a group)", flush=True)

    results: list[dict[str, Any]] = []

    def run_group(items: list[dict[str, str]]) -> list[dict[str, Any]]:
        out = []
        for case in items:
            res = run_case(case, dry_run=args.dry_run, force_rerun=args.force_rerun,
                           timeout_min=args.timeout_min)
            print(f"  [{res['status']:16s}] {res['case_id']:32s} "
                  f"{res.get('wall_s', 0):7.1f}s built={res.get('n_built', '-')}", flush=True)
            out.append(res)
        return out

    with futures.ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as pool:
        for group_results in pool.map(run_group, groups.values()):
            results.extend(group_results)

    rows = [r for res in results for r in res.get("rows", [])]
    if rows and not args.dry_run:
        prior = C.read_tsv(C.FEASIBILITY_TSV) if C.FEASIBILITY_TSV.is_file() else []
        fresh = {(r["case_id"], r["variant"]) for r in rows}
        merged = [r for r in prior if (r["case_id"], r["variant"]) not in fresh] + rows
        merged.sort(key=lambda r: (r["object_key"], r["case_id"], r["variant"]))
        C.write_tsv(C.FEASIBILITY_TSV, merged, C.FEASIBILITY_FIELDS)

    built = sum(1 for r in rows if r["built"])
    triage = sorted({r["case_id"] for r in rows if r["state"] in ("needs_triage", "upstream_rotation_bug")})
    print(f"\nrot variants built this pass = {built}/{len(cases) * len(C.BUILD_VARIANTS)}")
    if triage:
        print(f"  NEEDS ATTENTION (missing npz / rotation-bug): {triage}")
    if rows and not args.dry_run:
        print(f"  -> {C.rel(C.FEASIBILITY_TSV)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
