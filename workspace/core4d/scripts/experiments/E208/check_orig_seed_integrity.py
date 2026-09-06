#!/usr/bin/env python3
"""E208 P1 verify: prove the seeded `_original` artifacts are byte-identical to E206.

This is what closes C1.  It is the load-bearing check for the whole experiment:
if `_original` is not E206's exact bytes, then (a) the trim window shifts, which
silently invalidates reusing E206's 3cm contact mask, and (b) the aug variants
warm-start from a different pose, so every aug-vs-orig delta picks up F15's
IK non-determinism on top of the augmentation effect.

Hard gates (all 22 cases, every seeded variant root):
    R1  retargeted/{task}_original.npz  sha256 == E206
    R2  trimmed/{task}_original.npz     sha256 == E206
    R3  trim_window.json numeric fields == E206 (paths are expected to differ:
        pipeline.sh re-infers the file with E208-local paths)
    R4  the downstream inputs E208 will actually consume are unchanged --
        E206's 3cm contact mask, the base task's trajectory_kinematic.npz and
        the six task_info.json metadata fields load_case_meta() reads

Non-gate probes (opt-in, need the hsretargeting conda env; run them with P2 when
it is warm):
    R5  re-run the v1 retarget for the two F15-divergent cases plus two controls
        and report max|dqpos| -- confirmatory evidence that seeding avoided a
        real hazard, not a hypothetical one
    R6  run the same aug variant twice and report max|dqpos| -- feeds gate G4,
        which decides whether C4 is judged per-case or only distributionally

Usage:
    .venv/bin/python .../E208/check_orig_seed_integrity.py
    ... --probe-rerun           # adds R5/R6 (slow, needs conda)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import e208_common as C  # noqa: E402

SEED = C.load_e208_module("seed_from_e206")

TRIM_NUMERIC_FIELDS = (
    "untrimmed_frames", "trimmed_frames", "trim_start", "trim_end", "trim_frames",
)
META_FIELDS = ("date", "seq", "person", "object_name", "object_model_rel", "source_scene_task")


def check_case(case: dict[str, str], variant: str) -> dict[str, Any]:
    base = case["base_target_task"]
    meta = C.load_case_meta(base)
    task_name = meta["holosoma_task"]
    src_dir = SEED.e206_root(case["source_retarget_variant_id"]) / f"holosoma_{base}"
    dst_dir = C.holosoma_dir(base, variant)

    row: dict[str, Any] = {
        "case_id": case["case_id"], "object_key": case["object_key"],
        "base_target_task": base, "variant_root": variant,
        "f15_divergent": int(case["case_id"] in C.F15_DIVERGENT_CASES),
    }
    failures: list[str] = []

    # --- R1 / R2 -----------------------------------------------------------
    for code, sub in (("R1", "retargeted"), ("R2", "trimmed")):
        src = src_dir / sub / f"{task_name}_original.npz"
        dst = dst_dir / sub / f"{task_name}_original.npz"
        if not dst.is_file():
            failures.append(f"{code}: missing seeded {C.rel(dst)}")
            row[f"{code}_sha256"] = ""
            continue
        src_sha, dst_sha = C.sha256(src), C.sha256(dst)
        row[f"{code}_sha256"] = dst_sha
        row[f"{code}_match"] = int(src_sha == dst_sha)
        if src_sha != dst_sha:
            failures.append(f"{code}: {sub}/_original sha256 {dst_sha} != E206 {src_sha}")

    # --- R3 ----------------------------------------------------------------
    src_tw, dst_tw = src_dir / "trim_window.json", dst_dir / "trim_window.json"
    if not dst_tw.is_file():
        failures.append(f"R3: missing seeded {C.rel(dst_tw)}")
    else:
        want = json.loads(src_tw.read_text(encoding="utf-8"))
        got = json.loads(dst_tw.read_text(encoding="utf-8"))
        for field in TRIM_NUMERIC_FIELDS:
            if want.get(field) != got.get(field):
                failures.append(f"R3: trim_window.{field} {got.get(field)} != E206 {want.get(field)}")
        row["trim_start"] = got.get("trim_start")
        row["trim_frames"] = got.get("trim_frames")
        row["untrimmed_frames"] = got.get("untrimmed_frames")
        # the seeded npz must actually be consistent with the recorded window --
        # this is what makes E206's mask reusable frame-for-frame
        try:
            with np.load(dst_dir / "retargeted" / f"{task_name}_original.npz", allow_pickle=True) as data:
                n_untrimmed = int(np.asarray(data["qpos"]).shape[0])
            with np.load(dst_dir / "trimmed" / f"{task_name}_original.npz", allow_pickle=True) as data:
                n_trimmed = int(np.asarray(data["qpos"]).shape[0])
        except (OSError, KeyError, ValueError) as exc:
            failures.append(f"R3: cannot read seeded qpos ({exc})")
        else:
            row["npz_untrimmed_frames"] = n_untrimmed
            row["npz_trimmed_frames"] = n_trimmed
            if n_untrimmed != got.get("untrimmed_frames"):
                failures.append(f"R3: npz has {n_untrimmed} frames, window says {got.get('untrimmed_frames')}")
            if n_trimmed != got.get("trimmed_frames"):
                failures.append(f"R3: trimmed npz has {n_trimmed} frames, window says {got.get('trimmed_frames')}")
            if n_untrimmed - n_trimmed != got.get("trim_start"):
                failures.append(
                    f"R3: trim_start {got.get('trim_start')} != untrimmed-trimmed "
                    f"({n_untrimmed}-{n_trimmed}={n_untrimmed - n_trimmed}); "
                    "fixed_window_trim derives the aug window from this identity")

    # --- R4 ----------------------------------------------------------------
    # E206's 3cm mask carries THREE time axes and is self-describing:
    #   raw_*    (untrimmed_frames)              -- reference axis
    #   spider_* (trimmed_frames)                -- the axis SPIDER consumes
    #   eval_*   (trimmed * eval_fps / ref_fps)  -- the 50Hz eval axis
    # plus the `trim_start` it was generated at.  Reusing it unchanged for the
    # aug variants is valid iff that embedded trim_start equals the window the
    # aug variants are cut at -- which is exactly what seeding `_original`
    # guarantees.  Assert the identity rather than assuming it.
    mask = C.repo_path(case["orig_contact_mask"])
    traj = C.repo_path(case["orig_trajectory"])
    if not mask.is_file():
        failures.append(f"R4: missing E206 contact mask {mask}")
    else:
        row["contact_mask_sha256"] = C.sha256(mask)
        row["contact_mask"] = C.rel(mask)
        try:
            with np.load(mask, allow_pickle=True) as data:
                axes = {
                    "raw": int(np.asarray(data["raw_contact_mask_3cm"]).shape[0]),
                    "spider": int(np.asarray(data["spider_contact_mask_3cm"]).shape[0]),
                    "eval": int(np.asarray(data["eval_contact_mask_3cm"]).shape[0]),
                }
                mask_trim_start = int(data["trim_start"])
                ref_fps = float(data["ref_fps"])
                eval_fps = float(data["eval_fps"])
                threshold = float(data["threshold_m"])
        except (OSError, KeyError, ValueError) as exc:
            failures.append(f"R4: cannot read contact mask ({exc})")
        else:
            row.update({
                "mask_raw_frames": axes["raw"], "mask_spider_frames": axes["spider"],
                "mask_eval_frames": axes["eval"], "mask_trim_start": mask_trim_start,
                "mask_threshold_m": threshold,
            })
            if threshold != 0.03:
                failures.append(f"R4: mask threshold {threshold} != 0.03 (E206's PRIMARY_CONTACT_LABEL)")
            if mask_trim_start != row.get("trim_start"):
                failures.append(
                    f"R4: mask was generated at trim_start={mask_trim_start} but the seeded window "
                    f"says {row.get('trim_start')}; E206's mask is NOT reusable for this case")
            if axes["raw"] != row.get("untrimmed_frames"):
                failures.append(f"R4: mask raw axis {axes['raw']} != untrimmed {row.get('untrimmed_frames')}")
            if axes["spider"] != row.get("npz_trimmed_frames"):
                failures.append(
                    f"R4: mask spider axis {axes['spider']} != trimmed frames "
                    f"{row.get('npz_trimmed_frames')}")
            expected_eval = round(axes["spider"] * eval_fps / ref_fps)
            if abs(axes["eval"] - expected_eval) > 1:
                failures.append(
                    f"R4: mask eval axis {axes['eval']} != round({axes['spider']}*{eval_fps}/{ref_fps})"
                    f"={expected_eval}")

    if not traj.is_file():
        failures.append(f"R4: missing base trajectory {traj}")
    else:
        row["trajectory_sha256"] = C.sha256(traj)
        try:
            with np.load(traj, allow_pickle=True) as data:
                n_traj = int(np.asarray(data["qpos"]).shape[0])
        except (OSError, KeyError, ValueError) as exc:
            failures.append(f"R4: cannot read base trajectory ({exc})")
        else:
            row["trajectory_frames"] = n_traj
            # SPIDER's trajectory is cut at the same window; the aug variants are
            # cut at that window too, so all three axes have to agree here.
            if n_traj != row.get("npz_trimmed_frames"):
                failures.append(
                    f"R4: base trajectory has {n_traj} frames != trimmed "
                    f"{row.get('npz_trimmed_frames')}")

    info_path = C.TASK_ROOT / base / "task_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    for field in META_FIELDS:
        expected = meta[field]
        actual = info.get(field) if field != "source_scene_task" else Path(info["source_scene"]).parent.name
        if expected != actual:
            failures.append(f"R4: task_info.{field} {actual!r} != {expected!r}")
    row["object_name"] = meta["object_name"]
    row["source_scene_task"] = meta["source_scene_task"]

    row["failures"] = "; ".join(failures)
    row["status"] = "pass" if not failures else "fail"
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="")
    ap.add_argument("--probe-rerun", action="store_true",
                    help="also run the R5/R6 determinism probes (slow, needs the hsretargeting conda env)")
    ap.add_argument("--tsv-out", type=Path, default=C.SEED_INTEGRITY_TSV)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    cases = C.load_e208_cases()
    if args.cases:
        keep = {c.strip() for c in args.cases.split(",") if c.strip()}
        cases = [c for c in cases if c["case_id"] in keep]

    rows: list[dict[str, Any]] = []
    for case in cases:
        for variant in SEED.target_variants(case, C.RETARGET_VARIANTS):
            if not C.holosoma_dir(case["base_target_task"], variant).is_dir():
                continue
            rows.append(check_case(case, variant))

    for row in rows:
        mark = "PASS" if row["status"] == "pass" else "FAIL"
        flag = " [F15]" if row["f15_divergent"] else ""
        print(f"[{mark}] {row['case_id']:32s} {row['variant_root']:10s} "
              f"trim_start={row.get('trim_start', '?'):>4} "
              f"untrim/trim={row.get('npz_untrimmed_frames', '?'):>4}/{row.get('npz_trimmed_frames', '?'):<4} "
              f"mask raw/spider/eval="
              f"{row.get('mask_raw_frames', '?')}/{row.get('mask_spider_frames', '?')}/"
              f"{row.get('mask_eval_frames', '?')}{flag}")
        if row["failures"]:
            for failure in row["failures"].split("; "):
                print(f"        - {failure}")

    failures = [r for r in rows if r["status"] != "pass"]
    C.write_tsv(args.tsv_out, rows,
                [k for k in rows[0] if k != "failures"] + ["failures"] if rows else None)

    # Per-case (not per-variant-root) verdict is what C1 reports
    by_case = {}
    for row in rows:
        by_case.setdefault(row["case_id"], []).append(row["status"] == "pass")
    cases_pass = sum(1 for v in by_case.values() if all(v))

    payload: dict[str, Any] = {
        "experiment": C.EXP_ID, "generated_at": C.now(),
        "claim": "C1", "n_rows": len(rows), "n_row_failures": len(failures),
        "n_cases": len(by_case), "n_cases_pass": cases_pass,
        "verdict": "pass" if not failures and cases_pass == len(cases) else "fail",
        "f15_cases_in_registry": sorted(
            c for c in by_case if c in C.F15_DIVERGENT_CASES),
        "note": (
            "R1-R4 assert the seeded _original is E206's exact bytes, which is why "
            "F15 cannot reach E208's deltas. R5/R6 are confirmatory probes, run "
            "separately with --probe-rerun."
        ),
    }
    if args.probe_rerun:
        payload["probes"] = {
            "status": "not_implemented_yet",
            "reason": "R5/R6 need the hsretargeting conda env; they run in P2 alongside G1/G4",
        }
    out_json = args.json_out or args.tsv_out.with_suffix(".json")
    C.write_json(out_json, payload)

    print(f"\nC1: {cases_pass}/{len(cases)} cases pass "
          f"({len(rows) - len(failures)}/{len(rows)} case-variant rows) -> {C.rel(args.tsv_out)}")
    if payload["f15_cases_in_registry"]:
        print(f"F15-divergent cases present and seeded (not recomputed): "
              f"{payload['f15_cases_in_registry']}")
    return 1 if payload["verdict"] != "pass" else 0


if __name__ == "__main__":
    raise SystemExit(main())
