#!/usr/bin/env python3
"""E199 data construction: build the 6 augmented SPIDER tasks per case.

Per case (one object), this:
  1. runs the legacy pipeline in augmentation mode (RETARGET_AUGMENTATION=1,
     --skip-spider): convert -> parallel_robot_retarget (original + 5 native
     object-interaction configs) -> trim `_original` -> 3cm contact mask;
  2. fixed-window trims the 5 augmented variants at the SAME trim_start as
     `_original` (augmentation anchors the manipulation to the original, so the
     contact-based trim window is identical by construction -> the original
     contact mask, which is a time axis, is reused for all 6 variants);
  3. for each of the 6 variants generates an independent SPIDER task
     `{base_task}__aug_{variant}` (standard scene from the base task geometry +
     augmented initial object pose, trajectory, and the E199 rubber_hull+PRG
     sidecar) -- never overwriting the historical base task;
  4. records C3 augmentation-correctness metrics (approach-segment offset vs
     original, endpoint anchoring) and an artifact manifest with SHAs.

Runs under the SPIDER venv (imports mujoco); the upstream retarget subprocess
uses the hsretargeting conda env via pipeline.sh.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e199_common as C  # noqa: E402

RESULT_ROOT_REL = "workspace/core4d/results/E199/data_preprocess"
PIPELINE = "workspace/core4d/data_preprocess/pipeline.sh"
CREATE_SCENE = "workspace/core4d/data_preprocess/create_spider_scene_from_template.py"
CORE4D = "spider/process_datasets/core4d.py"
CASE_FIELDS = [
    "# enabled", "date", "seq", "person", "object_name", "object_model_rel",
    "source_scene_task", "target_task", "trim_start", "trim_frames", "data_id", "mask_slug",
]


def case_root(base_task: str) -> Path:
    return C.REPO / RESULT_ROOT_REL / f"holosoma_{base_task}"


def mask_path(base_task: str) -> Path:
    return C.REPO / RESULT_ROOT_REL / "contact_masks" / base_task / "raw_contact_mask_3cm.npz"


def write_case_file(path: Path, meta: dict[str, str], base_task: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = ["1", meta["date"], meta["seq"], meta["person"], meta["object_name"],
           meta["object_model_rel"], meta["source_scene_task"], base_task,
           "auto", "auto", "0", base_task]
    with path.open("w", encoding="utf-8", newline="") as stream:
        stream.write("\t".join(CASE_FIELDS) + "\n")
        stream.write("\t".join(row) + "\n")


def run_upstream(base_task: str, meta: dict[str, str], *, force: bool, max_workers: int) -> None:
    """convert -> parallel retarget (6) -> trim _original -> contact mask (skip SPIDER)."""
    case_file = C.REPO / RESULT_ROOT_REL / f"cases_e199_{meta['object_name']}_{meta['seq']}.tsv"
    write_case_file(case_file, meta, base_task)
    import os
    env = os.environ.copy()
    env.update({
        "REPO": str(C.REPO),
        "HOLOSOMA_DIR": str(C.HOLOSOMA_REPO),
        "CORE4D_REAL_ROOT": str(C.CORE4D_RAW_ROOT),
        "SMPLX_MODEL_DIR": str(C.SMPLX_MODEL_DIR),
        "HOLOSOMA_DEPS_DIR": str(C.E173.HOLOSOMA_DEPS_DIR),
        "RESULT_ROOT": RESULT_ROOT_REL,
        "PYTHON_BIN": str(C.SPIDER_PYTHON_BIN),
        "RETARGET_PYTHON_BIN": str(C.RETARGET_PYTHON_BIN),
        "RETARGET_AUGMENTATION": "1",
        "RETARGET_MAX_WORKERS": str(max_workers),
        **C.OMNIRT_V2_ENV,  # E199 frozen retarget contract (Phase-4 relaxation)
    })
    cmd = ["bash", PIPELINE, "--case-file", C.rel(case_file), "--skip-spider"]
    if force:
        cmd.append("--force")
    print(f"[upstream] {base_task}: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=C.REPO, env=env, check=True)


def _slice_time_axis(data: "np.lib.npyio.NpzFile", trim_start: int) -> dict[str, Any]:
    T = int(data["qpos"].shape[0])
    out: dict[str, Any] = {}
    for key in data.files:
        arr = data[key]
        if hasattr(arr, "shape") and arr.ndim >= 1 and arr.shape[0] == T:
            out[key] = arr[trim_start:]
        else:
            out[key] = arr
    return out


def fixed_window_trim(base_task: str, meta: dict[str, str],
                      aug_variants: list[tuple[str, str]] = C.AUG_VARIANTS) -> tuple[int, list[str]]:
    """Trim the feasible aug variants at the SAME trim_start as `_original`.

    Returns (trim_start, feasible_holo_names). Aug variants whose retarget was
    infeasible (npz absent) are skipped, not fatal.
    """
    root = case_root(base_task)
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
    feasible: list[str] = ["original"]
    infeasible: list[str] = []
    for _, holo_name in aug_variants:
        src = root / "retargeted" / f"{holo}_{holo_name}.npz"
        if not src.is_file():
            infeasible.append(holo_name)
            continue
        with np.load(src, allow_pickle=True) as data:
            saved = _slice_time_axis(data, trim_start)
        np.savez(str(root / "trimmed" / f"{holo}_{holo_name}.npz"), **saved)
        feasible.append(holo_name)
    print(f"[trim] {base_task}: trim_start={trim_start} (orig {t_orig}->{t_trim}); "
          f"feasible aug={len(feasible) - 1}/{len(aug_variants)} infeasible={infeasible}", flush=True)
    return trim_start, feasible


def build_variant_task(base_task: str, meta: dict[str, str], e199_name: str, holo_name: str,
                       *, overwrite: bool, skip_existing: bool = False) -> dict[str, Any]:
    root = case_root(base_task)
    trimmed_npz = root / "trimmed" / f"{meta['holosoma_task']}_{holo_name}.npz"
    if not trimmed_npz.is_file():
        raise FileNotFoundError(trimmed_npz)
    aug_task = C.aug_task_name(base_task, e199_name)
    source_scene = C.TASK_ROOT / base_task / "scene.xml"
    if not source_scene.is_file():
        raise FileNotFoundError(f"base geometry template scene missing: {source_scene}")

    task_dir = C.TASK_ROOT / aug_task
    trajectory = task_dir / "0/trajectory_kinematic.npz"
    scene_act = task_dir / "scene_act.xml"
    scene_common = [
        str(C.SPIDER_PYTHON_BIN), CREATE_SCENE,
        "--source-scene", str(source_scene), "--task", aug_task,
        "--qpos", str(trimmed_npz), "--data-id", "0",
        "--date", meta["date"], "--seq", meta["seq"], "--person", meta["person"],
        "--object-name", meta["object_name"], "--object-model-rel", meta["object_model_rel"],
    ]
    if skip_existing and trajectory.is_file() and scene_act.is_file():
        # already built (idempotent resume) -- only re-derive the E199 PRG sidecar
        # (cheap, no CEM) below; skip the scene/trajectory regeneration subprocesses.
        print(f"    [reuse] {aug_task} (trajectory + scene_act present)", flush=True)
    else:
        # 1) standard scene.xml (base geometry + augmented initial object pose)
        subprocess.run(scene_common, cwd=C.REPO, check=True)
        # 2) SPIDER trajectory (scene_act euler detection below reads this)
        subprocess.run([
            str(C.SPIDER_PYTHON_BIN), CORE4D,
            "--source-npz", str(trimmed_npz), "--task", aug_task, "--data-id", "0",
            "--dataset-name", "core4d", "--robot-type", "unitree_g1",
            "--embodiment-type", "humanoid_object", "--no-show-viewer", "--no-save-video",
        ], cwd=C.REPO, check=True)
        # 3) scene_act (needs the trajectory to pick the euler convention)
        subprocess.run(scene_common + ["--generate-scene-act"], cwd=C.REPO, check=True)

    # 4) E199 rubber_hull + PRG sidecar (cheap; always (re)derived, idempotent)
    scene = C.build_prg_scene(aug_task, scene_act, trajectory, overwrite=overwrite)

    return {
        "aug_variant": e199_name, "holosoma_variant": holo_name, "target_task": aug_task,
        "target_scene": C.rel(task_dir / "scene.xml"),
        "trajectory": C.rel(trajectory), "trajectory_sha256": C.sha256(trajectory),
        "base_scene_act": scene["base_scene"], "base_scene_sha256": scene["base_scene_sha256"],
        "scene_act": scene["physical_scene"], "scene_name": C.SCENE_NAME,
        "effective_scene_sha256": scene["physical_scene_sha256"],
        "reference_first5_min_lowerbody_object_distance_m": scene["reference_first5_min_lowerbody_object_distance_m"],
    }


def _yaw_deg(quat_wxyz: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    xyzw = quat_wxyz[:, [1, 2, 3, 0]]
    return np.degrees(Rotation.from_quat(xyzw).as_euler("ZYX")[:, 0])


def pose_diff(base_task: str, meta: dict[str, str], e199_name: str, holo_name: str,
              approach_frames: int = 10) -> dict[str, Any]:
    """C3: approach-segment offset vs original + endpoint anchoring (from trimmed npz)."""
    root = case_root(base_task)
    holo = meta["holosoma_task"]
    with np.load(root / "trimmed" / f"{holo}_original.npz", allow_pickle=True) as data:
        qo = np.asarray(data["qpos"], dtype=np.float64)
    with np.load(root / "trimmed" / f"{holo}_{holo_name}.npz", allow_pickle=True) as data:
        qa = np.asarray(data["qpos"], dtype=np.float64)
    n = min(len(qo), len(qa))
    qo, qa = qo[:n], qa[:n]
    pos_o, pos_a = qo[:, 36:39], qa[:, 36:39]
    yaw_o, yaw_a = _yaw_deg(qo[:, 39:43]), _yaw_deg(qa[:, 39:43])
    trans = np.linalg.norm(pos_a - pos_o, axis=1)
    yaw = np.abs(((yaw_a - yaw_o) + 180.0) % 360.0 - 180.0)
    k = min(approach_frames, n)
    return {
        "n_frames": n,
        "approach_trans_offset_m_max": float(trans[:k].max()),
        "approach_trans_offset_m_mean": float(trans[:k].mean()),
        "approach_yaw_deg_max": float(yaw[:k].max()),
        "approach_yaw_deg_mean": float(yaw[:k].mean()),
        "endpoint_trans_offset_m": float(trans[-1]),
        "endpoint_yaw_deg": float(yaw[-1]),
        "endpoint_frac_of_approach_trans": float(trans[-1] / trans[:k].max()) if trans[:k].max() > 1e-9 else 0.0,
    }


def process_case(case: dict[str, str], *, force: bool, max_workers: int,
                 overwrite_scenes: bool, skip_upstream: bool,
                 variants: list[tuple[str, str]] = C.VARIANTS,
                 skip_existing: bool = False) -> list[dict[str, Any]]:
    base_task = case["base_target_task"]
    object_key = case["object_key"]
    meta = C.load_case_meta(base_task)
    aug_variants = [v for v in variants if v[0] != "orig"]
    print(f"\n=== E199 case {object_key}: {base_task} ===", flush=True)
    if not skip_upstream:
        run_upstream(base_task, meta, force=force, max_workers=max_workers)
    trim_start, feasible = fixed_window_trim(base_task, meta, aug_variants)
    fresh_mask = mask_path(base_task)
    if not fresh_mask.is_file():
        raise FileNotFoundError(f"contact mask not produced: {fresh_mask}")
    rows: list[dict[str, Any]] = []
    for e199_name, holo_name in variants:
        if holo_name not in feasible:
            print(f"  [skip] {e199_name} ({holo_name}) infeasible upstream -- no task built", flush=True)
            continue
        try:
            artifact = build_variant_task(base_task, meta, e199_name, holo_name,
                                          overwrite=overwrite_scenes, skip_existing=skip_existing)
            diff = pose_diff(base_task, meta, e199_name, holo_name)
        except Exception as exc:  # noqa: BLE001
            # e.g. build_prg_scene runtime_initial_overlap: an augmented pose whose
            # reference first frames already penetrate the object -- drop this
            # variant, keep the rest of the case.
            print(f"  [skip] {e199_name} ({holo_name}) scene build failed: "
                  f"{type(exc).__name__}: {exc}", flush=True)
            continue
        rows.append({
            "object_key": object_key, "base_target_task": base_task,
            "case_id": f"{object_key}_{meta['date']}_{meta['seq']}_{meta['person']}",
            "trim_start": trim_start,
            "aug_translation": C.aug_translation(e199_name),
            "aug_rotation_rad": C.aug_rotation_rad(e199_name),
            "contact_mask": C.rel(fresh_mask), "contact_mask_sha256": C.sha256(fresh_mask),
            **artifact, **{f"posediff_{k}": v for k, v in diff.items()},
            "updated_at": C.now(),
        })
        print(f"  [{e199_name}] {artifact['target_task']} approach_trans={diff['approach_trans_offset_m_max']:.3f}m "
              f"approach_yaw={diff['approach_yaw_deg_max']:.1f}deg endpoint_trans={diff['endpoint_trans_offset_m']:.3f}m",
              flush=True)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", default="pilot", choices=["pilot", "box_fullscale"],
                        help="pilot=8-object 6-variant; box_fullscale=all s6 box cases, translation-only")
    parser.add_argument("--cases", default="", help="comma list of object_keys or case_ids to restrict to")
    parser.add_argument("--force", action="store_true", help="force upstream re-run")
    parser.add_argument("--skip-upstream", action="store_true", help="reuse existing retarget/trim/mask")
    parser.add_argument("--overwrite-scenes", action="store_true")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--artifacts", default="",
                        help="override output artifacts TSV path (for parallel sharded builds)")
    args = parser.parse_args()

    fullscale = args.scope == "box_fullscale"
    variants = C.TRANS_VARIANTS if fullscale else C.VARIANTS
    registry = C.load_fullscale_cases() if fullscale else C.CASES
    out = C.repo_path(args.artifacts) if args.artifacts else (
        C.FULLSCALE_ARTIFACTS if fullscale else
        C.RESULTS / "data_preprocess/manifests/e199_aug_artifacts.tsv")

    wanted = {c.strip() for c in args.cases.split(",") if c.strip()}
    cases = [c for c in registry
             if not wanted or c["object_key"] in wanted or c.get("case_id") in wanted]
    if not cases:
        raise SystemExit(f"no cases match {wanted} (scope={args.scope})")
    print(f"[scope] {args.scope}: {len(cases)} case(s), variants={[v[0] for v in variants]}", flush=True)

    all_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for case in cases:
        try:
            all_rows.extend(process_case(
                case, force=args.force, max_workers=args.max_workers,
                overwrite_scenes=args.overwrite_scenes, skip_upstream=args.skip_upstream,
                variants=variants, skip_existing=fullscale))
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            failures.append({"object_key": case["object_key"],
                             "base_target_task": case["base_target_task"],
                             "error": f"{type(exc).__name__}: {exc}"})

    # merge with prior runs of the SAME scope: keep rows for cases not processed now
    processed = {c["base_target_task"] for c in cases}
    if C.repo_path(out).is_file():
        prior = [r for r in C.read_tsv(out) if r.get("base_target_task") not in processed]
        all_rows = prior + all_rows
    all_rows.sort(key=lambda r: (r.get("object_key", ""), r.get("case_id", ""), r.get("aug_variant", "")))
    C.write_tsv(out, all_rows)
    C.write_json(C.repo_path(out).with_suffix(".json"), all_rows)
    if failures:
        C.write_json(C.repo_path(out).with_name("e199_aug_failures.json"), failures)
    print(f"\n[done] {len(all_rows)} variant tasks across {len(cases) - len(failures)}/{len(cases)} cases; "
          f"failures={len(failures)} -> {C.rel(out)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
