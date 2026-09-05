#!/usr/bin/env python3
"""E208 P2-P4: drive pipeline.sh's augmentation branch, twice (v1 pass, v2 rescue).

    --pass pass1          omnirt_v1 for the 21 v1-sourced cases, omnirt_v2 for the
                          one case whose source was already a v2 rescue
    --pass pass2-rescue   omnirt_v2, ONLY for the (case, variant) pairs that pass1
                          recorded as `v1_infeasible`

What this script is careful about
---------------------------------
* **Never passes ``--force``.**  P1 seeded ``_original`` byte-for-byte from E206
  and ``initialize_robot_pose`` (robot_retarget.py:563-569) *loads that file from
  disk* as the warm start for every trans variant.  ``--force`` would recompute
  it and reintroduce F15's IK non-determinism into the baseline.
* **Runs same-object cases serially.**  ``sync_generated_object_model`` (``cp -a``)
  and ``ensure_g1_object_xml`` (first-write) race on per-object model/XML files,
  so object groups run concurrently but each group is a queue.
* **Keeps its own sentinel.**  In augmentation mode pipeline.sh's done-marker is
  ``{task}_rot_1.npz`` (pipeline.sh:345-350), and under omnirt_v1 rot_1 is very
  likely infeasible -- so that marker may never appear and the step would be
  re-entered forever.  ``DP/sentinels/{base}.{pass}.done`` records what actually
  landed instead.
* **Splits v1 and v2 into separate RESULT_ROOTs.**  Upstream short-circuits on
  the output filename and both variants emit the same ``{task}_{trans_k}.npz``
  name; sharing a root would make the effective variant guessable only by mtime.
* **``--skip-contact``.**  Because ``trim_start`` is identical to E206's (P1/R4
  asserts it), E206's 3cm mask is valid frame-for-frame; regenerating it would
  only create a second copy to drift from the one the overrides point at.

Usage:
    .venv/bin/python .../E208/run_upstream_retarget.py --pass pass1 --dry-run
    ... --pass pass1 --cases <5 probes> --max-workers 5
    ... --pass pass2-rescue
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

import e208_common as C  # noqa: E402

PASSES = ("pass1", "pass2-rescue")


SingleInstance = C.SingleInstance  # moved to e208_common; both drivers need it


# 12 tab-separated columns, header line starts with "# enabled".
# `source_scene_task` must be non-empty even under --skip-spider: pipeline.sh
# reads the row with IFS=$'\t' and an empty field shifts `target_task`.
CASE_FIELDS = [
    "# enabled", "date", "seq", "person", "object_name", "object_model_rel",
    "source_scene_task", "target_task", "trim_start", "trim_frames", "data_id", "mask_slug",
]

# Upstream prints exactly this when a variant's IK is infeasible
# (parallel_robot_retarget.py:322).  It names the variant, so it is a far
# sharper signal than grepping for "cvxpy" somewhere in the log.
SKIP_MARKER = re.compile(r"\[skip\] augmentation (?P<name>\S+) infeasible/failed: (?P<reason>.*)")

# Upstream's rotation augmentation is broken against the installed scipy, and it
# fails BEFORE any IK is attempted, so it never reaches the try/except that emits
# SKIP_MARKER.  `holosoma .../src/utils.py:346` calls
#     R.from_euler("z", rotation_list)          # rotation_list.shape == (N,)
# but scipy 1.17.1 requires shape (N, 1) for a one-axis sequence, so it raises
#     ValueError: Expected last dimension of `angles` to match number of
#                 sequence axes specified, got <N>.
# and the whole file's processing aborts -- which is why rot_1 is never even
# attempted once rot_0 has been reached.
#
# This matters beyond E208: E199 concluded "rotation is systematically
# infeasible" from exactly this crash (97 rot_0 attempts, 0 rot_1 attempts,
# 0 rot npz produced, and only ONE genuine `[skip] ... infeasible` line in the
# whole experiment).  Rotation augmentation has never actually been evaluated
# here.  E208 is locked to translation-only so it does not act on this, but it
# must not repeat the misattribution: these rows get their own state, never
# `needs_triage` (which is reserved for a trans variant that needs a human) and
# never `*_infeasible` (which would claim an IK result we do not have).
ROTATION_BUG_MARKER = re.compile(
    r"Expected last dimension of .angles. to match number of sequence axes specified"
)


def case_file_path(meta: dict[str, str], case: dict[str, str]) -> Path:
    """One case file per case.

    E202 named these ``cases_e202_{object}_{seq}.tsv`` with no person component.
    E208 has six (object, seq) pairs that appear as both p1 and p2 --
    desk007_028, desk021_005/007/010, desk023_066 -- so that scheme would have
    silently processed one of each pair twice and the other never.
    """
    return C.DP_CASE_FILES / (
        f"cases_e208_{meta['object_name']}_{meta['seq']}_{meta['person']}.tsv"
    )


def write_case_file(case: dict[str, str], meta: dict[str, str]) -> Path:
    path = case_file_path(meta, case)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = [
        "1", meta["date"], meta["seq"], meta["person"],
        meta["object_name"],            # verbatim -- "Desk021"/"Desk023" are capitalised
        meta["object_model_rel"],
        meta["source_scene_task"],      # never empty (IFS shift hazard)
        case["base_target_task"],       # holosoma_{target_task} == the dir we seeded
        "auto", "auto",                 # re-infer the window from the seeded npz
        "0",
        case["base_target_task"],       # mask_slug (unused under --skip-contact)
    ]
    path.write_text(
        "\t".join(CASE_FIELDS) + "\n" + "\t".join(row) + "\n", encoding="utf-8"
    )
    return path


def pipeline_env(variant: str) -> dict[str, str]:
    env = dict(os.environ)
    env.update({
        "REPO": str(C.REPO),
        "HOLOSOMA_DIR": str(C.HOLOSOMA_REPO),
        "CORE4D_REAL_ROOT": str(C.env_path("CORE4D_RAW_ROOT")),
        "SMPLX_MODEL_DIR": str(C.env_path("SMPLX_MODEL_DIR")),
        "RESULT_ROOT": C.rel(C.data_root(variant)),
        "PYTHON_BIN": str(C.SPIDER_PYTHON_BIN),
        "RETARGET_PYTHON_BIN": str(C.retarget_python_bin()),
        "KEEP_GOING": "0",
        "RETARGET_AUGMENTATION": "1",
        "RETARGET_MAX_WORKERS": os.environ.get("E208_RETARGET_MAX_WORKERS", "6"),
    })
    env.update(C.PIPELINE_DATASET_ENV)
    env.update(C.OMNIRT_ENV_BY_VARIANT[variant])   # six keys, always explicit
    return env


def variant_npz(case: dict[str, str], meta: dict[str, str], variant: str, holo_name: str) -> Path:
    return (C.holosoma_dir(case["base_target_task"], variant) / "retargeted"
            / f"{meta['holosoma_task']}_{holo_name}.npz")


def scan_log(log_path: Path) -> tuple[dict[str, str], bool]:
    """(variant -> upstream's own infeasibility reason, rotation-bug seen?)."""
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


def classify(
    case: dict[str, str], meta: dict[str, str], variant: str, pass_id: str,
    log_hits: dict[str, str], rotation_bug: bool, is_source_v2: bool,
) -> list[dict[str, Any]]:
    """Two independent signals per variant; disagreement is NOT auto-escalated."""
    rows: list[dict[str, Any]] = []
    build_names = {h for _s, h in C.BUILD_VARIANTS}
    for short, holo in C.ALL_AUG_VARIANTS:
        npz = variant_npz(case, meta, variant, holo)
        present = npz.is_file() and npz.stat().st_size > 0
        reason_hit = log_hits.get(holo, "")

        if present:
            state = "source_v2_ok" if is_source_v2 else (
                "rescued_v2" if pass_id == "pass2-rescue" else "v1_ok")
            reason = "log_noise_but_produced" if reason_hit else ""
        elif rotation_bug and holo not in {h for _s, h in C.TRANS_VARIANTS}:
            # NOT an IK result: upstream crashed in augment_object_poses before
            # solving anything (see ROTATION_BUG_MARKER).  Reporting this as
            # "infeasible" is the mistake E199 made.
            state, reason = "upstream_rotation_bug", "scipy_from_euler_shape"
        elif reason_hit:
            if is_source_v2:
                state, reason = "source_v2_infeasible", "cvxpy_infeasible"
            elif pass_id == "pass2-rescue":
                state, reason = "rescue_failed", "cvxpy_infeasible"
            else:
                state, reason = "v1_infeasible", "cvxpy_infeasible"
        else:
            # Missing npz with NO infeasibility line: could equally be a crash,
            # an OOM or a failed write.  Escalating those to v2 as if they were
            # "mathematically infeasible" would corrupt the C2 yield and the
            # L-ladder verdict, and L3's whole job is telling bugs apart from
            # real infeasibility.  So: flag for a human, never auto-escalate.
            state, reason = "needs_triage", "missing_npz_no_log"

        rows.append({
            "case_id": case["case_id"], "object_key": case["object_key"],
            "base_target_task": case["base_target_task"],
            "variant": short, "holosoma_variant": holo,
            "pass": pass_id, "retarget_variant": variant,
            "npz_present": int(present),
            "npz_sha256": C.sha256(npz) if present else "",
            "log_hit": reason_hit,
            "rescue_reason": reason, "rescue_state": state,
            "built": int(short in dict(C.BUILD_VARIANTS) and state in C.BUILT_STATES),
            "updated_at": C.now(),
        })
    return rows


def run_case(
    case: dict[str, str], variant: str, pass_id: str, *,
    dry_run: bool, force_rerun: bool, timeout_min: int,
) -> dict[str, Any]:
    meta = C.load_case_meta(case["base_target_task"])
    base = case["base_target_task"]
    is_source_v2 = case["source_retarget_variant_id"] == "omnirt_v2"
    sentinel = C.sentinel_path(base, pass_id)
    log_path = C.DP / "logs" / pass_id / f"{base}.log"

    result: dict[str, Any] = {
        "case_id": case["case_id"], "object_key": case["object_key"],
        "base_target_task": base, "retarget_variant": variant, "pass": pass_id,
    }

    if sentinel.is_file() and not force_rerun:
        result["status"] = "skip_sentinel"
        result["wall_s"] = 0.0
        hits, rot_bug = scan_log(log_path)
        result["rows"] = classify(case, meta, variant, pass_id, hits, rot_bug, is_source_v2)
        return result

    case_file = write_case_file(case, meta)
    cmd = [
        "bash", "workspace/core4d/data_preprocess/pipeline.sh",
        "--case-file", C.rel(case_file),
        "--skip-contact",   # E206's 3cm mask is reused verbatim (P1/R4)
        "--skip-spider",    # SPIDER tasks are built by build_augmented_tasks.py
    ]
    result["cmd"] = " ".join(cmd)
    result["result_root"] = C.rel(C.data_root(variant))
    if dry_run:
        result["status"] = "dry_run"
        result["wall_s"] = 0.0
        result["rows"] = []
        return result

    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = pipeline_env(variant)
    started = time.time()
    with log_path.open("w", encoding="utf-8") as stream:
        stream.write(f"# pass={pass_id} retarget_variant={variant}\n")
        stream.write(f"# env={json.dumps({k: env[k] for k in sorted(C.OMNIRT_V1_ENV)})}\n")
        stream.write(f"# RESULT_ROOT={env['RESULT_ROOT']}\n# cmd={' '.join(cmd)}\n\n")
        stream.flush()
        try:
            proc = subprocess.run(cmd, cwd=C.REPO, env=env, stdout=stream,
                                  stderr=subprocess.STDOUT, timeout=timeout_min * 60, check=False)
            returncode = proc.returncode
            timed_out = False
        except subprocess.TimeoutExpired:
            returncode, timed_out = -1, True
            stream.write(f"\n# TIMEOUT after {timeout_min} min\n")
    wall = time.time() - started

    log_hits, rotation_bug = scan_log(log_path)
    rows = classify(case, meta, variant, pass_id, log_hits, rotation_bug, is_source_v2)
    for row in rows:
        row["wall_s"] = round(wall, 1)
        row["log_ref"] = C.rel(log_path)

    produced = {r["holosoma_variant"]: r["npz_sha256"] for r in rows if r["npz_present"]}
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text(json.dumps({
        "pass": pass_id, "retarget_variant": variant, "case_id": case["case_id"],
        "finished_at": C.now(), "returncode": returncode, "timed_out": timed_out,
        "wall_s": round(wall, 1),
        "artifacts": {holo: produced.get(holo) for _s, holo in C.ALL_AUG_VARIANTS},
        "log": C.rel(log_path),
    }, indent=2) + "\n", encoding="utf-8")

    result.update({
        "status": "timeout" if timed_out else ("ok" if returncode == 0 else f"rc{returncode}"),
        "returncode": returncode, "wall_s": round(wall, 1), "rows": rows,
        "log": C.rel(log_path),
        "n_built": sum(r["built"] for r in rows),
    })
    return result


def select_work(pass_id: str, cases: list[dict[str, str]]) -> list[tuple[dict[str, str], str]]:
    """(case, effective retarget variant) pairs this pass should run."""
    if pass_id == "pass1":
        return [(c, c["source_retarget_variant_id"]) for c in cases]

    if not C.FEASIBILITY_TSV.is_file():
        raise SystemExit(f"pass2-rescue needs {C.rel(C.FEASIBILITY_TSV)}; run --pass pass1 first")
    prior = C.read_tsv(C.FEASIBILITY_TSV)
    buildable = {s for s, _h in C.BUILD_VARIANTS}
    needs: set[str] = {
        row["case_id"] for row in prior
        if row["pass"] == "pass1" and row["variant"] in buildable
        and row["rescue_state"] == "v1_infeasible"
    }
    triage = sorted({
        row["case_id"] for row in prior
        if row["pass"] == "pass1" and row["variant"] in buildable
        and row["rescue_state"] in ("needs_triage", "upstream_rotation_bug")
    })
    if triage:
        print(f"NOTE: {len(triage)} case(s) have needs_triage variants and are NOT auto-rescued: "
              f"{triage}\n      inspect their logs before deciding (missing npz with no "
              f"infeasibility line == crash/OOM/disk, not a maths result)", flush=True)
    return [(c, "omnirt_v2") for c in cases if c["case_id"] in needs]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass", dest="pass_id", choices=PASSES, required=True)
    ap.add_argument("--cases", default="", help="comma-separated case_ids")
    ap.add_argument("--probes", action="store_true", help="shorthand for the 5 P2 probe cases")
    ap.add_argument("--objects", default="", help="comma-separated object_keys")
    ap.add_argument("--max-workers", type=int, default=5,
                    help="concurrent OBJECT groups (cases inside a group stay serial)")
    ap.add_argument("--timeout-min", type=int, default=180)
    ap.add_argument("--force-rerun", action="store_true", help="ignore sentinels (never --force upstream)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.dry_run:
        # one driver at a time, no matter which pass -- both passes call
        # pipeline.sh, and the per-object race does not care which pass you are in
        _lock = SingleInstance(C.DP / "run_upstream_retarget.lock")
        _lock.__enter__()

    cases = C.load_e208_cases()
    if args.probes:
        cases = [c for c in cases if c["case_id"] in C.PROBE_CASES]
    if args.cases:
        keep = {c.strip() for c in args.cases.split(",") if c.strip()}
        unknown = keep - {c["case_id"] for c in cases}
        if unknown:
            raise SystemExit(f"unknown case_ids: {sorted(unknown)}")
        cases = [c for c in cases if c["case_id"] in keep]
    if args.objects:
        keep_obj = {o.strip() for o in args.objects.split(",") if o.strip()}
        cases = [c for c in cases if c["object_key"] in keep_obj]
    if not cases:
        raise SystemExit("no cases selected")

    work = select_work(args.pass_id, cases)
    if not work:
        print(f"{args.pass_id}: nothing to do (no v1_infeasible variants to rescue)")
        return 0

    # group by object: concurrent across groups, serial within
    groups: dict[str, list[tuple[dict[str, str], str]]] = {}
    for case, variant in work:
        groups.setdefault(case["object_key"], []).append((case, variant))
    print(f"{args.pass_id}: {len(work)} case-runs across {len(groups)} object groups "
          f"(<= {args.max_workers} groups in parallel, serial within a group)", flush=True)

    results: list[dict[str, Any]] = []

    def run_group(items: list[tuple[dict[str, str], str]]) -> list[dict[str, Any]]:
        out = []
        for case, variant in items:
            res = run_case(case, variant, args.pass_id, dry_run=args.dry_run,
                           force_rerun=args.force_rerun, timeout_min=args.timeout_min)
            print(f"  [{res['status']:14s}] {res['case_id']:32s} {variant:10s} "
                  f"{res.get('wall_s', 0):7.1f}s built={res.get('n_built', '-')}",
                  flush=True)
            out.append(res)
        return out

    with futures.ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as pool:
        for group_results in pool.map(run_group, groups.values()):
            results.extend(group_results)

    rows = [r for res in results for r in res.get("rows", [])]
    if rows and not args.dry_run:
        prior = C.read_tsv(C.FEASIBILITY_TSV) if C.FEASIBILITY_TSV.is_file() else []
        fresh = {(r["case_id"], r["variant"], r["pass"]) for r in rows}
        merged = [r for r in prior if (r["case_id"], r["variant"], r["pass"]) not in fresh] + rows
        merged.sort(key=lambda r: (r["pass"], r["object_key"], r["case_id"], r["variant"]))
        C.write_tsv(C.FEASIBILITY_TSV, merged, C.FEASIBILITY_FIELDS)

    build_names = {s for s, _h in C.BUILD_VARIANTS}
    built = sum(1 for r in rows if r["variant"] in build_names and r["built"])
    triage = [r for r in rows if r["variant"] in build_names
              and r["rescue_state"] in ("needs_triage", "upstream_rotation_bug")]
    walls = sorted(res["wall_s"] for res in results if res.get("wall_s"))
    print(f"\n{args.pass_id}: {len(results)} case-runs, variants built this pass = {built}"
          f"/{len(work) * len(C.BUILD_VARIANTS)}")
    if walls:
        print(f"  wall per case: min={walls[0]:.1f}s median={walls[len(walls) // 2]:.1f}s max={walls[-1]:.1f}s")
    if triage:
        print(f"  needs_triage: {sorted({r['case_id'] for r in triage})}")
    if rows and not args.dry_run:
        print(f"  -> {C.rel(C.FEASIBILITY_TSV)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
