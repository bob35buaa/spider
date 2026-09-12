#!/usr/bin/env python3
"""E215 P3: turn feasible rot retargets into SPIDER tasks with per-arm CEM scenes.

Per (case, rot variant) in order:

1. **fixed-window trim** -- cut the rot retarget at the SAME ``trim_start`` as the
   seeded ``_original`` (upstream's trim only globs ``*_original.npz``).  Because
   the window equals E199/E202's, their 3 cm mask is valid frame-for-frame.
2. **SPIDER task dir** ``dcv3_omnirt_v2_ref_fk_{case}__aug_rot{k}`` -- scene ->
   trajectory -> scene_act, generated from the BASE task's ``scene.xml`` geometry
   template.
3. **arm CEM scene** dispatched by object line:
     bucket -> e202_common.build_prg_scene (contact-aligned 5-seg proxy, 18-pair
               union) -> scene_act_E202_bucketAlignedTop_PRG
     box    -> e199_common.build_prg_scene (rubber_hull hand, 16 leg pairs) ->
               scene_act_E199_rubberHull (+ _PRG)
   then, for the two G1 arm groups, a single-variable object-gravcomp sidecar.
   The effective scene per group is picked from what the builder produced
   (rubberHull for box_noprg, PRG for box_prg, gravcomp for the G1 groups).
4. **pose_diff** -- C3 evidence: approach yaw (~45 deg expected; a late trim_start
   decays it) and the 0.2 m lateral offset rot_* also carries.

Usage:
    .venv/bin/python .../E215/build_augmented_tasks.py --cases a,b
    ... --objects box021 --overwrite-scenes --trim-only
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent / "E199", _HERE.parent / "E202"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import e215_common as C  # noqa: E402

RUN = C.load_e215_module("run_upstream_retarget")
GRAV = C.load_e215_module("build_gravcomp_sidecars")

CREATE_SCENE = "workspace/core4d/data_preprocess/create_spider_scene_from_template.py"
CORE4D = "spider/process_datasets/core4d.py"

OBJ_POS = slice(36, 39)
OBJ_QUAT = slice(39, 43)


def _slice_time_axis(data: Any, trim_start: int) -> dict[str, Any]:
    n_frames = int(data["qpos"].shape[0])
    out: dict[str, Any] = {}
    for key in data.files:
        arr = data[key]
        if hasattr(arr, "shape") and arr.ndim >= 1 and arr.shape[0] == n_frames:
            out[key] = arr[trim_start:]
        else:
            out[key] = arr
    return out


def fixed_window_trim(case: dict[str, str], meta: dict[str, str],
                      feasible_holo: list[str], *, force: bool = False) -> tuple[int, list[str]]:
    C._drop_broken_torch()
    root = C.holosoma_dir(case["case_id"])
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
        raise ValueError(f"negative trim_start for {case['case_id']}: {t_orig} -> {t_trim}")

    (root / "trimmed").mkdir(parents=True, exist_ok=True)
    done: list[str] = []
    for holo_name in feasible_holo:
        src = root / "retargeted" / f"{holo}_{holo_name}.npz"
        dst = root / "trimmed" / f"{holo}_{holo_name}.npz"
        if not src.is_file():
            continue
        if dst.is_file() and not force:
            done.append(holo_name)
            continue
        with np.load(src, allow_pickle=True) as data:
            saved = _slice_time_axis(data, trim_start)
        if int(saved["qpos"].shape[0]) != t_trim:
            raise ValueError(
                f"{case['case_id']}/{holo_name}: trimmed to {saved['qpos'].shape[0]} "
                f"frames, expected {t_trim}")
        np.savez(str(dst), **saved)
        done.append(holo_name)
    print(f"  [trim] {case['case_id']}: trim_start={trim_start} ({t_orig}->{t_trim}) "
          f"rot_trimmed={done}", flush=True)
    return trim_start, done


def _build_arm_scene(case: dict[str, str], meta: dict[str, str], task_dir: Path,
                     base_scene_act: Path, trajectory: Path, *, overwrite: bool) -> dict[str, Any]:
    """Dispatch the CEM scene build by arm group; return the effective-scene dict."""
    spec = C.group_spec(case["case_id"])
    obj = case["object_key"]
    case_id = case["case_id"]

    if spec["object_line"] == "bucket":
        import e202_common as E202
        info = E202.build_prg_scene(case_id, base_scene_act, trajectory,
                                    overwrite=overwrite, object_key=obj)
        prg_scene = C.repo_path(info["physical_scene"])
        object_geom_count = int(info["object_geom_count"])
        pair_count = int(info["compiled_robot_object_pair_count"])
        rubber_scene = C.repo_path(info["rubber_scene"])
    else:
        import e199_common as E199
        info = E199.build_prg_scene(case_id, base_scene_act, trajectory, overwrite=overwrite)
        prg_scene = C.repo_path(info["physical_scene"])
        object_geom_count = 1
        pair_count = int(info["compiled_pair_count"])
        rubber_scene = C.repo_path(info["rubber_scene"])

    out: dict[str, Any] = {
        "prg_scene": C.rel(prg_scene), "prg_scene_sha256": C.sha256(prg_scene),
        "rubber_scene": C.rel(rubber_scene), "rubber_scene_sha256": C.sha256(rubber_scene),
        "object_geom_count": object_geom_count,
        "compiled_robot_object_pair_count": pair_count,
    }

    group = case["arm_group"]
    if group == "box_noprg":
        effective = rubber_scene
        out["gravcomp_base_scene"] = ""
    elif spec.get("gravcomp"):
        gc_scene, action = GRAV.write_gravcomp_sidecar(prg_scene, spec["scene_name"],
                                                       overwrite=overwrite)
        effective = gc_scene
        out["gravcomp_base_scene"] = C.rel(prg_scene)
        out["gravcomp_action"] = action
    else:  # bucket_prg / box_prg
        effective = prg_scene
        out["gravcomp_base_scene"] = ""

    if effective.stem != spec["scene_name"]:
        raise AssertionError(
            f"{case_id}: effective scene {effective.name} != frozen {spec['scene_name']}")
    out["scene_act"] = C.rel(effective)
    out["scene_name"] = spec["scene_name"]
    out["effective_scene_sha256"] = C.sha256(effective)
    return out


def build_variant_task(case: dict[str, str], meta: dict[str, str], variant: str,
                       holo_name: str, *, overwrite_scenes: bool) -> dict[str, Any]:
    case_id = case["case_id"]
    base_task = case["base_target_task"]
    root = C.holosoma_dir(case_id)
    trimmed_npz = root / "trimmed" / f"{meta['holosoma_task']}_{holo_name}.npz"
    if not trimmed_npz.is_file():
        raise FileNotFoundError(trimmed_npz)

    aug_task = C.aug_task_name(case_id, variant)
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
        "--object-name", meta["object_name"], "--object-model-rel", meta["object_model_rel"],
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

    arm = _build_arm_scene(case, meta, task_dir, scene_act, trajectory, overwrite=overwrite_scenes)
    return {
        "aug_variant": variant, "holosoma_variant": holo_name,
        "effective_retarget_variant": C.RETARGET_VARIANT,
        "target_task": aug_task, "target_scene": C.rel(task_dir / "scene.xml"),
        "trajectory": C.rel(trajectory), "trajectory_sha256": C.sha256(trajectory),
        "base_scene_act": C.rel(scene_act), "base_scene_sha256": C.sha256(scene_act),
        "trimmed_npz": C.rel(trimmed_npz),
        **arm,
    }


def _yaw_deg(quat_wxyz: np.ndarray) -> np.ndarray:
    C._drop_broken_torch()
    from scipy.spatial.transform import Rotation
    return np.degrees(Rotation.from_quat(quat_wxyz[:, [1, 2, 3, 0]]).as_euler("ZYX")[:, 0])


def pose_diff(case: dict[str, str], meta: dict[str, str], holo_name: str,
              approach_frames: int = 10) -> dict[str, Any]:
    C._drop_broken_torch()
    root = C.holosoma_dir(case["case_id"])
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
    return {
        "n_frames": n,
        "approach_yaw_deg_max": float(yaw[:k].max()),
        "approach_yaw_deg_mean": float(yaw[:k].mean()),
        "approach_trans_offset_m_max": float(trans[:k].max()),
        "endpoint_yaw_deg": float(yaw[-1]),
        "endpoint_trans_offset_m": float(trans[-1]),
    }


def feasible_variants(case_id: str) -> list[str]:
    """rot variants the upstream scan marked rot_ok, from the feasibility TSV."""
    if not C.FEASIBILITY_TSV.is_file():
        raise SystemExit(f"missing {C.rel(C.FEASIBILITY_TSV)}; run run_upstream_retarget.py first")
    ok: list[str] = []
    for row in C.read_tsv(C.FEASIBILITY_TSV):
        if row["case_id"] == case_id and row["state"] == C.BUILT_STATE:
            ok.append(row["variant"])
    return ok


def process_case(case: dict[str, str], *, overwrite_scenes: bool, trim_only: bool) -> list[dict[str, Any]]:
    case_id = case["case_id"]
    meta = C.load_case_meta(case["base_target_task"])
    ok = feasible_variants(case_id)
    print(f"\n=== {case['object_key']} {case_id} ({len(ok)}/{len(C.BUILD_VARIANTS)} feasible rot) ===",
          flush=True)
    if not ok:
        return []

    holo_by_short = dict(C.BUILD_VARIANTS)
    fixed_window_trim(case, meta, [holo_by_short[s] for s in ok])
    if trim_only:
        return []

    rows: list[dict[str, Any]] = []
    for short in ok:
        holo_name = holo_by_short[short]
        try:
            row = build_variant_task(case, meta, short, holo_name, overwrite_scenes=overwrite_scenes)
            row.update(pose_diff(case, meta, holo_name))
            yaw = float(row["approach_yaw_deg_max"])
            row["degraded_yaw"] = int(yaw < C.EFFECTIVE_YAW_FLOOR_DEG)
            row["status"] = "built" if not row["degraded_yaw"] else "built_degenerate_yaw"
        except Exception as exc:  # noqa: BLE001 - one bad variant must not kill the batch
            row = {"aug_variant": short, "holosoma_variant": holo_name,
                   "effective_retarget_variant": C.RETARGET_VARIANT,
                   "target_task": C.aug_task_name(case_id, short),
                   "status": f"fail_{type(exc).__name__}", "error": str(exc)[:500]}
        row.update({
            "case_id": case_id, "object_key": case["object_key"],
            "arm_group": case["arm_group"], "base_variant": case["base_variant"],
            "base_target_task": case["base_target_task"],
            "aug_translation": C.aug_translation(short), "aug_rotation_rad": C.aug_rotation_rad(short),
            "updated_at": C.now(),
        })
        rows.append(row)
        mark = {"built": "OK ", "built_degenerate_yaw": "DEG"}.get(row["status"], "ERR")
        print(f"  [{mark}] {short:5s} {row['target_task']}"
              + (f"  yaw={row.get('approach_yaw_deg_max', 0):.1f}deg "
                 f"trans={row.get('approach_trans_offset_m_max', 0):.3f}m "
                 f"pairs={row.get('compiled_robot_object_pair_count', '?')} "
                 f"scene={row.get('scene_name', '?')}"
                 if row["status"].startswith("built") else f"  {row.get('error', '')[:180]}"),
              flush=True)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="")
    ap.add_argument("--objects", default="")
    ap.add_argument("--overwrite-scenes", action="store_true")
    ap.add_argument("--trim-only", action="store_true")
    ap.add_argument("--artifacts", type=Path, default=C.ARTIFACTS_TSV)
    args = ap.parse_args()

    _lock = RUN.SingleInstance(C.DP / "build_augmented_tasks.lock")
    _lock.__enter__()

    cases = C.load_e215_cases()
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
        rows.extend(process_case(case, overwrite_scenes=args.overwrite_scenes, trim_only=args.trim_only))

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
    deg = [r for r in ok if r["status"] == "built_degenerate_yaw"]
    failed = [r for r in rows if not r["status"].startswith("built")]
    print(f"\nbuilt {len(ok)}/{len(rows)} rot tasks over {len(cases)} cases "
          f"({len(deg)} degenerate-yaw) -> {C.rel(args.artifacts)}")
    if ok:
        yaws = sorted(r["approach_yaw_deg_max"] for r in ok)
        print(f"  approach yaw: min={yaws[0]:.1f} median={yaws[len(yaws) // 2]:.1f} "
              f"max={yaws[-1]:.1f} deg (nominal {C.NOMINAL_YAW_DEG}, floor {C.EFFECTIVE_YAW_FLOOR_DEG})")
    for row in deg:
        print(f"  DEGENERATE_YAW {row['case_id']}/{row['aug_variant']}: "
              f"only {row['approach_yaw_deg_max']:.1f} deg reaches SPIDER")
    for row in failed:
        print(f"  FAIL {row['case_id']}/{row['aug_variant']}: {row['status']} {row.get('error', '')[:200]}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
