#!/usr/bin/env python3
"""E208 P2/P5: turn feasible aug retargets into SPIDER tasks with E206 PRG scenes.

Per (case, aug variant) -- all five: trans_0/1/2 and rot_0/1 -- in order:

1. **fixed-window trim** -- cut the aug retarget at the SAME ``trim_start`` as
   ``_original``.  Upstream's ``trim_no_contact.py`` only globs ``*_original.npz``
   (:362), so it never touches the aug variants; and because P1 seeded
   ``_original`` from E206 byte-for-byte, this window is E206's window, which is
   what makes E206's 3cm contact mask reusable frame-for-frame.
2. **SPIDER task dir** ``{base with variant relabelled}__aug_{variant}`` -- scene
   -> trajectory -> scene_act.  The scene is generated from the BASE task's
   ``scene.xml``, which E206 already rewrote to carry the hand-placed lowgeom box
   proxy (P4), so the aug task inherits the proxy for free.
3. **E206 arm scenes** via ``build_arm_scenes.build_one`` -- unchanged E206 code.
   Its ``task_dir()`` reads ``row["target_task"]`` first, so pointing that at the
   aug dir lands the whole rubberHull -> noPRG -> PRG chain there.  E208 only
   consumes the PRG scene, but the noPRG one is still built: the
   ``signature_without_pairs(noPRG) == signature_without_pairs(PRG)`` assertion
   is the only online evidence that the rubber-hull patch landed correctly in
   this new directory and that the 16N leg pairs are the *entire* difference.
4. **pose_diff** -- C3 evidence of how much augmentation SPIDER actually sees,
   and `effective_aug` (see below), which is not the same question.

Usage:
    .venv/bin/python .../E208/build_augmented_tasks.py --probes
    ... --cases a,b --overwrite-scenes --trim-only
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import e208_common as C  # noqa: E402

# same single-instance discipline as run_upstream_retarget: two builders on one
# case would race inside the shared task dir (scene -> trajectory -> scene_act ->
# arm scenes all write there). Bitten twice on 2026-09-05; use a lock, not care.
RUN = C.load_e208_module("run_upstream_retarget")

CREATE_SCENE = "workspace/core4d/data_preprocess/create_spider_scene_from_template.py"
CORE4D = "spider/process_datasets/core4d.py"

# object free-joint slice inside qpos (43 = 36 robot + 3 pos + 4 quat)
OBJ_POS = slice(36, 39)
OBJ_QUAT = slice(39, 43)


def _slice_time_axis(data: Any, trim_start: int) -> dict[str, Any]:
    """Slice every array whose axis 0 is the time axis; leave scalars alone."""
    n_frames = int(data["qpos"].shape[0])
    out: dict[str, Any] = {}
    for key in data.files:
        arr = data[key]
        if hasattr(arr, "shape") and arr.ndim >= 1 and arr.shape[0] == n_frames:
            out[key] = arr[trim_start:]
        else:
            out[key] = arr
    return out


def fixed_window_trim(
    base_task: str, meta: dict[str, str], retarget_variant: str,
    aug_variants: list[tuple[str, str]] | None = None, *, force: bool = False,
) -> tuple[int, list[str]]:
    """Trim feasible aug variants at ``_original``'s window. Missing npz = skip, not fatal."""
    C._drop_broken_torch()
    aug_variants = aug_variants or C.BUILD_VARIANTS
    root = C.holosoma_dir(base_task, retarget_variant)
    holo = meta["holosoma_task"]

    retargeted_orig = root / "retargeted" / f"{holo}_original.npz"
    trimmed_orig = root / "trimmed" / f"{holo}_original.npz"
    for path in (retargeted_orig, trimmed_orig):
        if not path.is_file():
            raise FileNotFoundError(path)
    with np.load(retargeted_orig, allow_pickle=True) as data:
        t_orig = int(data["qpos"].shape[0])
    with np.load(trimmed_orig, allow_pickle=True) as data:
        t_trim = int(data["qpos"].shape[0])
    trim_start = t_orig - t_trim
    if trim_start < 0:
        raise ValueError(f"negative trim_start for {base_task}: {t_orig} -> {t_trim}")

    (root / "trimmed").mkdir(parents=True, exist_ok=True)
    feasible: list[str] = []
    infeasible: list[str] = []
    for _short, holo_name in aug_variants:
        src = root / "retargeted" / f"{holo}_{holo_name}.npz"
        dst = root / "trimmed" / f"{holo}_{holo_name}.npz"
        if not src.is_file():
            infeasible.append(holo_name)
            continue
        if dst.is_file() and not force:
            feasible.append(holo_name)
            continue
        with np.load(src, allow_pickle=True) as data:
            saved = _slice_time_axis(data, trim_start)
        if int(saved["qpos"].shape[0]) != t_trim:
            raise ValueError(
                f"{base_task}/{holo_name}: trimmed to {saved['qpos'].shape[0]} frames, "
                f"expected {t_trim} -- the aug retarget has a different length than _original")
        np.savez(str(dst), **saved)
        feasible.append(holo_name)
    print(f"  [trim] {base_task} @{retarget_variant}: trim_start={trim_start} "
          f"({t_orig}->{t_trim}) feasible={len(feasible)}/{len(aug_variants)} "
          f"infeasible={infeasible}", flush=True)
    return trim_start, feasible


def build_variant_task(
    case: dict[str, str], meta: dict[str, str], variant: str, holo_name: str,
    retarget_variant: str, *, overwrite_scenes: bool,
) -> dict[str, Any]:
    base_task = case["base_target_task"]
    root = C.holosoma_dir(base_task, retarget_variant)
    trimmed_npz = root / "trimmed" / f"{meta['holosoma_task']}_{holo_name}.npz"
    if not trimmed_npz.is_file():
        raise FileNotFoundError(trimmed_npz)

    aug_task = C.aug_task_name(base_task, variant, retarget_variant)
    source_scene = C.TASK_ROOT / base_task / "scene.xml"
    if not source_scene.is_file():
        raise FileNotFoundError(f"base geometry template missing: {source_scene}")

    task_dir = C.TASK_ROOT / aug_task
    trajectory = task_dir / "0/trajectory_kinematic.npz"
    scene_act = task_dir / "scene_act.xml"
    scene_common = [
        str(C.SPIDER_PYTHON_BIN), CREATE_SCENE,
        "--source-scene", str(source_scene), "--task", aug_task,
        "--qpos", str(trimmed_npz), "--data-id", "0",
        "--date", meta["date"], "--seq", meta["seq"], "--person", meta["person"],
        "--object-name", meta["object_name"],
        "--object-model-rel", meta["object_model_rel"],
    ]
    if trajectory.is_file() and scene_act.is_file() and not overwrite_scenes:
        print(f"    [reuse] {aug_task}", flush=True)
    else:
        subprocess.run(scene_common, cwd=C.REPO, check=True)
        subprocess.run([
            str(C.SPIDER_PYTHON_BIN), CORE4D,
            "--source-npz", str(trimmed_npz), "--task", aug_task, "--data-id", "0",
            "--dataset-name", "core4d", "--robot-type", "unitree_g1",
            "--embodiment-type", "humanoid_object", "--no-show-viewer", "--no-save-video",
        ], cwd=C.REPO, check=True)
        subprocess.run(scene_common + ["--generate-scene-act"], cwd=C.REPO, check=True)

    # E206's chain, unchanged: target_task steers it at the aug dir
    arm = C.ARM.build_one(
        {"case_id": case["case_id"], "object_key": case["object_key"], "target_task": aug_task},
        dry_run=False,
    )
    if arm.get("status") != "built":
        raise RuntimeError(f"{aug_task}: arm scene build returned {arm.get('status')}")

    n_boxes = int(arm["object_geom_count"])
    prg_pairs = int(arm["prg_compiled"]["robot_object_pairs"])
    expected = C.expected_pair_counts(n_boxes)["prg"]
    if prg_pairs != expected:
        raise AssertionError(f"{aug_task}: {prg_pairs} PRG pairs != 18*{n_boxes}={expected}")

    prg_scene = task_dir / f"{C.SCENE_NAME}.xml"
    return {
        "aug_variant": variant, "holosoma_variant": holo_name,
        "effective_retarget_variant": retarget_variant,
        "target_task": aug_task,
        "target_scene": C.rel(task_dir / "scene.xml"),
        "trajectory": C.rel(trajectory), "trajectory_sha256": C.sha256(trajectory),
        "base_scene_act": C.rel(scene_act), "base_scene_sha256": C.sha256(scene_act),
        "scene_act": C.rel(prg_scene), "scene_name": C.SCENE_NAME,
        "effective_scene_sha256": C.sha256(prg_scene),
        "object_geom_count": n_boxes,
        "compiled_robot_object_pair_count": prg_pairs,
        "noprg_scene": C.rel(task_dir / f"{C.SCENE_NOPRG}.xml"),
        "trimmed_npz": C.rel(trimmed_npz),
    }


def _yaw_deg(quat_wxyz: np.ndarray) -> np.ndarray:
    C._drop_broken_torch()
    from scipy.spatial.transform import Rotation
    return np.degrees(Rotation.from_quat(quat_wxyz[:, [1, 2, 3, 0]]).as_euler("ZYX")[:, 0])


def pose_diff(
    base_task: str, meta: dict[str, str], holo_name: str, retarget_variant: str,
    approach_frames: int = 10,
) -> dict[str, Any]:
    """C3: is the 0.2 m approach offset actually there, and did trans-only stay yaw-free?"""
    C._drop_broken_torch()
    root = C.holosoma_dir(base_task, retarget_variant)
    holo = meta["holosoma_task"]
    with np.load(root / "trimmed" / f"{holo}_original.npz", allow_pickle=True) as data:
        q_orig = np.asarray(data["qpos"], dtype=np.float64)
    with np.load(root / "trimmed" / f"{holo}_{holo_name}.npz", allow_pickle=True) as data:
        q_aug = np.asarray(data["qpos"], dtype=np.float64)
    n = min(len(q_orig), len(q_aug))
    q_orig, q_aug = q_orig[:n], q_aug[:n]

    trans = np.linalg.norm(q_aug[:, OBJ_POS] - q_orig[:, OBJ_POS], axis=1)
    yaw = np.abs(((_yaw_deg(q_aug[:, OBJ_QUAT]) - _yaw_deg(q_orig[:, OBJ_QUAT])) + 180.0) % 360.0 - 180.0)
    k = min(approach_frames, n)
    peak = float(trans[:k].max())
    return {
        "n_frames": n,
        "approach_trans_offset_m_max": peak,
        "approach_trans_offset_m_mean": float(trans[:k].mean()),
        "approach_yaw_deg_max": float(yaw[:k].max()),
        "approach_yaw_deg_mean": float(yaw[:k].mean()),
        "endpoint_trans_offset_m": float(trans[-1]),
        "endpoint_yaw_deg": float(yaw[-1]),
        "endpoint_frac_of_approach_trans": float(trans[-1] / peak) if peak > 1e-9 else 0.0,
    }


def effective_variants(case: dict[str, str]) -> dict[str, str]:
    """Which retarget variant produced each trans variant, per the feasibility TSV."""
    if not C.FEASIBILITY_TSV.is_file():
        raise SystemExit(f"missing {C.rel(C.FEASIBILITY_TSV)}; run run_upstream_retarget.py first")
    chosen: dict[str, str] = {}
    for row in C.read_tsv(C.FEASIBILITY_TSV):
        if row["case_id"] != case["case_id"] or row["rescue_state"] not in C.BUILT_STATES:
            continue
        # pass1 wins; a rescue only fills a variant pass1 could not produce
        if row["variant"] not in chosen or row["pass"] == "pass1":
            chosen[row["variant"]] = row["retarget_variant"]
    return chosen


def process_case(case: dict[str, str], *, overwrite_scenes: bool, trim_only: bool) -> list[dict[str, Any]]:
    base_task = case["base_target_task"]
    meta = C.load_case_meta(base_task)
    chosen = effective_variants(case)
    print(f"\n=== {case['object_key']} {case['case_id']} "
          f"({len(chosen)}/{len(C.BUILD_VARIANTS)} feasible) ===", flush=True)
    if not chosen:
        return []

    # trim once per retarget-variant root that actually contributed
    for retarget_variant in sorted(set(chosen.values())):
        wanted = [(s, h) for s, h in C.BUILD_VARIANTS if chosen.get(s) == retarget_variant]
        fixed_window_trim(base_task, meta, retarget_variant, wanted)
    if trim_only:
        return []

    rows: list[dict[str, Any]] = []
    holo_by_short = dict(C.BUILD_VARIANTS)
    for short, _holo in C.BUILD_VARIANTS:
        retarget_variant = chosen.get(short)
        if retarget_variant is None:
            continue
        holo_name = holo_by_short[short]
        try:
            row = build_variant_task(case, meta, short, holo_name, retarget_variant,
                                     overwrite_scenes=overwrite_scenes)
            row.update(pose_diff(base_task, meta, holo_name, retarget_variant))
            # How much augmentation SPIDER actually sees, which is NOT the
            # nominal 0.2 m: the perturbation decays from object_moving_frame_idx
            # on translation_tau=50, and SPIDER only gets the contact-trimmed
            # window, so a late trim_start eats most of it. Measured on the P2
            # probes: chair005 (trim_start=113) arrives at 0.023 m -- a
            # near-duplicate of orig, which would pad the dataset without adding
            # diversity and flatter the C4 delta. Flag, do not silently ship.
            offset = float(row["approach_trans_offset_m_max"])
            row["effective_offset_frac"] = round(offset / C.NOMINAL_AUG_OFFSET_M, 4)
            row["effective_aug"] = int(offset >= C.EFFECTIVE_AUG_FLOOR_M)
            row["status"] = "built" if row["effective_aug"] else "built_degenerate_offset"
        except Exception as exc:  # noqa: BLE001 - one bad variant must not kill the batch
            row = {"aug_variant": short, "holosoma_variant": holo_name,
                   "effective_retarget_variant": retarget_variant,
                   "target_task": C.aug_task_name(base_task, short, retarget_variant),
                   "status": f"fail_{type(exc).__name__}", "error": str(exc)[:500]}
        row.update({
            "case_id": case["case_id"], "object_key": case["object_key"],
            "base_target_task": base_task,
            "source_retarget_variant_id": case["source_retarget_variant_id"],
            "aug_translation": C.aug_translation(short),
            "aug_rotation_rad": C.aug_rotation_rad(short),
            "orig_result_npz": case["orig_result_npz"],
            "contact_mask": case["orig_contact_mask"],
            "contact_mask_sha256": C.sha256(case["orig_contact_mask"]),
            "f15_divergent": int(case["case_id"] in C.F15_DIVERGENT_CASES),
            "updated_at": C.now(),
        })
        rows.append(row)
        mark = {"built": "OK ", "built_degenerate_offset": "DEG"}.get(row["status"], "ERR")
        print(f"  [{mark}] {short:7s} {row['target_task']}"
              + (f"  approach={row.get('approach_trans_offset_m_max', 0):.3f}m "
                 f"yaw={row.get('approach_yaw_deg_max', 0):.2f}deg "
                 f"pairs={row.get('compiled_robot_object_pair_count', '?')}"
                 if row["status"] == "built" else f"  {row.get('error', '')[:160]}"), flush=True)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="")
    ap.add_argument("--probes", action="store_true")
    ap.add_argument("--objects", default="")
    ap.add_argument("--overwrite-scenes", action="store_true")
    ap.add_argument("--trim-only", action="store_true", help="stop after the fixed-window trim")
    ap.add_argument("--artifacts", type=Path, default=C.ARTIFACTS_TSV)
    args = ap.parse_args()

    _lock = RUN.SingleInstance(C.DP / "build_augmented_tasks.lock")
    _lock.__enter__()

    cases = C.load_e208_cases()
    if args.probes:
        cases = [c for c in cases if c["case_id"] in C.PROBE_CASES]
    if args.cases:
        keep = {c.strip() for c in args.cases.split(",") if c.strip()}
        cases = [c for c in cases if c["case_id"] in keep]
    if args.objects:
        keep_obj = {o.strip() for o in args.objects.split(",") if o.strip()}
        cases = [c for c in cases if c["object_key"] in keep_obj]
    if not cases:
        raise SystemExit("no cases selected")

    rows: list[dict[str, Any]] = []
    for case in cases:
        rows.extend(process_case(case, overwrite_scenes=args.overwrite_scenes,
                                 trim_only=args.trim_only))

    if rows:
        prior = C.read_tsv(args.artifacts) if args.artifacts.is_file() else []
        fresh = {(r["case_id"], r["aug_variant"]) for r in rows}
        merged = [r for r in prior if (r["case_id"], r["aug_variant"]) not in fresh] + rows
        merged.sort(key=lambda r: (r["object_key"], r["case_id"], r["aug_variant"]))
        fields: list[str] = []
        for row in merged:
            for key in row:
                if key not in fields:
                    fields.append(key)
        C.write_tsv(args.artifacts, merged, fields)

    ok = [r for r in rows if r["status"].startswith("built")]
    effective = [r for r in ok if r.get("effective_aug")]
    degenerate = [r for r in ok if not r.get("effective_aug")]
    failed = [r for r in rows if not r["status"].startswith("built")]
    print(f"\nbuilt {len(ok)}/{len(rows)} aug tasks over {len(cases)} cases "
          f"({len(effective)} effective, {len(degenerate)} degenerate) -> {C.rel(args.artifacts)}")
    if ok:
        offsets = sorted(r["approach_trans_offset_m_max"] for r in ok)
        yaws = [r["approach_yaw_deg_max"] for r in ok]
        print(f"  approach offset: min={offsets[0]:.3f} median={offsets[len(offsets) // 2]:.3f} "
              f"max={offsets[-1]:.3f} m (nominal {C.NOMINAL_AUG_OFFSET_M}, "
              f"floor {C.EFFECTIVE_AUG_FLOOR_M})")
        print(f"  approach yaw:    max={max(yaws):.3f} deg "
              f"(~0 expected for trans_*; ~45 for rot_*)")
    for row in degenerate:
        print(f"  DEGENERATE {row['case_id']}/{row['aug_variant']}: "
              f"only {row['approach_trans_offset_m_max']:.4f} m reaches SPIDER "
              f"({row['effective_offset_frac']:.0%} of nominal)")
    for row in failed:
        print(f"  FAIL {row['case_id']}/{row['aug_variant']}: {row['status']} {row.get('error', '')[:200]}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
