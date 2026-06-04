"""Re-verify E143 hand-annotated case labels from OBJECT TRAJECTORY analysis.

For each of the 24 E143 cases we extract the rigid-body pose of the ``object``
body via MuJoCo forward kinematics (the most robust path, mirroring
``unified_replay_eval.py``) and compute quantitative metrics that confirm or
refute the human annotations (rotation ~180 deg, walk-up before lift, late lift,
object jitter, etc.).

Run (from repo root):
    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 .venv/bin/python \
        workspace/exp_diagnostic_v3/scripts/analyze_object_trajectory.py
"""

from __future__ import annotations

import csv
import glob
import json
import math
import os
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[3]
VARIANTS_TSV = REPO / "workspace/core4d/scripts/E143/variants.tsv"
OUT_DIR = REPO / "workspace/exp_diagnostic_v3/results/trajectory_analysis"
PER_CASE_DIR = OUT_DIR / "per_case"
CONTACT_DIR = REPO / "workspace/core4d/results/E143/contact_masks"
LEGOBJ_CSV_DIR = REPO / "workspace/core4d/results/E143/cem/full"
E109_METRICS = (
    REPO
    / "workspace/core4d/results/E109/expanded_24_work_cases"
    / "expanded_24_method_metrics.tsv"
)

# 0-indexed columns in variants.tsv (verified empirically).
COL_CASE_ID = 3
COL_E109_CASE_ID = 2
COL_OBJECT_KEY = 4
COL_DERIVED_TASK = 7
COL_PERSON_IDX = 8
COL_BASELINE_NPZ = 14
COL_OMNI_QPOS = 16
COL_OMNI_SCENE = 17
COL_SPIDER_SCENE = 18
COL_RUN_STATUS = 20
COL_REUSE_OUTDIR_NPZ = 24


def build_scene_index() -> dict[str, str]:
    """Index every available scene_act.xml under results by snapshot dir name.

    The variants.tsv spider_scene_xml points into example_datasets/ which is
    gitignored and often absent. The reproducible copies live in per-experiment
    scene_snapshot/{derived_task}/scene_act.xml. We index those by the snapshot
    directory basename (== derived_task) and pick the first valid one.
    """
    idx: dict[str, list[str]] = {}
    pattern = str(REPO / "workspace/core4d/results/**/scene_snapshot/*/scene_act.xml")
    for p in glob.glob(pattern, recursive=True):
        key = os.path.basename(os.path.dirname(p))
        idx.setdefault(key, []).append(p)
    return {k: sorted(v)[0] for k, v in idx.items()}


_SCENE_INDEX = build_scene_index()


def resolve_spider_scene(spider_scene_field: str, derived_task: str) -> Path | None:
    """Return a loadable nq=42 spider scene_act.xml for the case."""
    direct = resolve(spider_scene_field)
    if direct.exists():
        return direct
    hit = _SCENE_INDEX.get(derived_task)
    if hit:
        return Path(hit)
    return None

LIFT_THRESH_M = 0.05  # 5 cm lift threshold for lift onset / pre-lift phase

# Cases not in the 22-annotated set but present in the 24.
NOT_ANNOTATED = {
    "bucket004_20231002_022_p1",
    "bucket004_20231003_1_012_p1",
    "box026_039_p1",
}


def read_variants() -> list[list[str]]:
    rows = []
    with open(VARIANTS_TSV) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            rows.append(line.rstrip("\n").split("\t"))
    return rows


def resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else REPO / p


def load_qpos(npz_path: Path) -> np.ndarray | None:
    """Return (T, nq) for entity 0. Handles (T,2,nq) and (T,nq)."""
    if not npz_path.exists():
        return None
    d = np.load(npz_path, allow_pickle=True)
    if "qpos" not in d:
        return None
    arr = np.asarray(d["qpos"], dtype=np.float64)
    if arr.ndim == 3 and arr.shape[1] >= 2:
        return arr[:, 0, :]
    if arr.ndim == 2:
        return arr
    return None


def object_world_pose(scene_xml: Path, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Forward-kinematics the ``object`` body pose for each frame.

    Returns (pos (T,3), quat_wxyz (T,4)).
    """
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    if qpos.shape[1] != model.nq:
        raise ValueError(
            f"qpos nq={qpos.shape[1]} != scene nq={model.nq}: {scene_xml}"
        )
    data = mujoco.MjData(model)
    obj_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    if obj_bid < 0:
        raise ValueError(f"no 'object' body in {scene_xml}")
    pos = np.empty((qpos.shape[0], 3))
    quat = np.empty((qpos.shape[0], 4))
    for t in range(qpos.shape[0]):
        data.qpos[:] = qpos[t]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        pos[t] = data.xpos[obj_bid]
        quat[t] = data.xquat[obj_bid]
    # normalize quats defensively
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    return pos, quat


def quat_geodesic_deg(q0: np.ndarray, q1: np.ndarray) -> float:
    """Shortest-arc geodesic angle (deg) between two unit wxyz quaternions."""
    dot = abs(float(np.dot(q0, q1)))
    dot = min(1.0, max(-1.0, dot))
    return math.degrees(2.0 * math.acos(dot))


def rotation_metrics(quat: np.ndarray) -> dict:
    q0 = quat[0]
    rel = np.array([quat_geodesic_deg(q0, quat[t]) for t in range(len(quat))])
    incr = np.array(
        [quat_geodesic_deg(quat[t - 1], quat[t]) for t in range(1, len(quat))]
    )
    return {
        "object_rotation_total_deg": float(rel[-1]),
        "object_rotation_max_deg": float(rel.max()),
        "object_rotation_cumulative_deg": float(incr.sum()),
        "_rel_series": rel,
    }


def lift_metrics(pos: np.ndarray) -> dict:
    z = pos[:, 2]
    z0 = float(z[0])
    z_lift = float(z.max() - z0)
    z_range = float(z.max() - z.min())
    above = np.where(z - z0 > LIFT_THRESH_M)[0]
    if len(above) == 0:
        onset_frac = float("nan")
        onset_idx = None
    else:
        onset_idx = int(above[0])
        onset_frac = onset_idx / max(1, len(z) - 1)
    return {
        "object_z_lift_m": z_lift,
        "object_z_range_m": z_range,
        "object_z_init_m": z0,
        "lift_onset_frac": onset_frac,
        "_lift_onset_idx": onset_idx,
        "_z_series": z,
    }


def xy_metrics(pos: np.ndarray) -> dict:
    xy = pos[:, :2]
    steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    return {
        "object_xy_displacement_m": float(steps.sum()),
        "object_xy_net_m": float(np.linalg.norm(xy[-1] - xy[0])),
    }


def jitter_metric(pos: np.ndarray) -> dict:
    """Mean per-frame acceleration magnitude (2nd diff of position) +
    velocity sign-flip count, as proxies for mocap jitter."""
    acc = np.diff(pos, n=2, axis=0)
    acc_mag = np.linalg.norm(acc, axis=1)
    vel = np.diff(pos, axis=0)
    flips = 0
    for d in range(3):
        s = np.sign(vel[:, d])
        s = s[s != 0]
        flips += int(np.sum(np.diff(s) != 0))
    return {
        "object_jitter_accel_mean": float(acc_mag.mean()) if len(acc_mag) else float("nan"),
        "object_jitter_accel_max": float(acc_mag.max()) if len(acc_mag) else float("nan"),
        "object_jitter_vel_signflips": flips,
    }


def walk_up_metric(robot_xy: np.ndarray, lift_onset_idx: int | None) -> dict:
    """Robot-base xy path length during the pre-lift phase.

    If the object never lifts, the entire trajectory is "pre-lift".
    """
    if lift_onset_idx is None:
        end = len(robot_xy)
    else:
        end = max(1, lift_onset_idx)
    pre = robot_xy[:end]
    if len(pre) < 2:
        travel = 0.0
    else:
        travel = float(np.linalg.norm(np.diff(pre, axis=0), axis=1).sum())
    return {
        "prelift_robot_base_travel_m": travel,
        "prelift_n_frames": int(end),
    }


def gt_contact_metrics(case_id: str, person_idx: int) -> dict:
    """GT mocap contact fraction for the target person (either hand)."""
    path = CONTACT_DIR / case_id / "raw_contact_mask_3cm.npz"
    out = {
        "gt_contact_frac": float("nan"),
        "gt_contact_frac_5cm": float("nan"),
        "gt_contact_source": "missing_gt",
        "gt_min_dist_min_m": float("nan"),
    }
    if not path.exists():
        return out
    d = np.load(path, allow_pickle=True)
    persons = [str(p) for p in d["persons"]]
    # person_idx 0 -> person1, 1 -> person2
    pname = f"person{person_idx + 1}"
    if pname in persons:
        pi = persons.index(pname)
    else:
        pi = min(person_idx, len(persons) - 1)
    if "raw_min_dist_m" in d:
        md = np.asarray(d["raw_min_dist_m"], dtype=np.float64)  # (T,2,2)
        per = md[:, pi, :]  # (T, 2 hands)
        either = per.min(axis=1)  # min over the two hands
        out["gt_contact_frac"] = float(np.mean(either < 0.03))
        out["gt_contact_frac_5cm"] = float(np.mean(either < 0.05))
        out["gt_min_dist_min_m"] = float(np.nanmin(either))
        out["gt_contact_source"] = "raw_min_dist_m"
    else:
        mask = np.asarray(d["raw_contact_mask_3cm"], dtype=bool)  # (T,2,2)
        per = mask[:, pi, :]  # (T,2)
        either = per.any(axis=1)
        out["gt_contact_frac"] = float(np.mean(either))
        out["gt_contact_frac_5cm"] = float("nan")  # no dist signal available
        out["gt_contact_source"] = "raw_contact_mask_3cm"
    return out


def load_e109_metrics() -> dict:
    """case_id -> {method -> row dict}."""
    out: dict[str, dict[str, dict]] = {}
    if not E109_METRICS.exists():
        return out
    with open(E109_METRICS) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            out.setdefault(row["case_id"], {})[row["method"]] = row
    return out


def merge_e109(rec: dict, e109: dict, e109_case_id: str) -> None:
    """Attach OmniRetarget vs Spider CEM E109 object/contact metrics to rec."""
    m = e109.get(e109_case_id, {})
    omni = m.get("OmniRetarget", {})
    spd = m.get("Spider CEM", {})

    def g(d: dict, k: str) -> float:
        try:
            return float(d.get(k, ""))
        except (TypeError, ValueError):
            return float("nan")

    rec["e109_omni_object_z_range_m"] = g(omni, "object_z_range_m")
    rec["e109_spider_object_z_range_m"] = g(spd, "object_z_range_m")
    rec["e109_omni_object_xy_disp_m"] = g(omni, "object_xy_displacement_m")
    rec["e109_spider_object_xy_disp_m"] = g(spd, "object_xy_displacement_m")
    rec["e109_omni_hand_near_5cm_frac"] = g(omni, "hand_geom_near_5cm_frac")
    rec["e109_spider_hand_near_5cm_frac"] = g(spd, "hand_geom_near_5cm_frac")
    rec["e109_omni_object_floor_frac"] = g(omni, "object_floor_contact_frac")
    rec["e109_spider_object_floor_frac"] = g(spd, "object_floor_contact_frac")


def downsample(arr: np.ndarray, n: int = 50) -> list:
    if len(arr) <= n:
        return [float(x) for x in arr]
    idx = np.linspace(0, len(arr) - 1, n).round().astype(int)
    return [float(arr[i]) for i in idx]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PER_CASE_DIR.mkdir(parents=True, exist_ok=True)
    e109 = load_e109_metrics()

    tsv_cols = [
        "case_id",
        "object_key",
        "person_idx",
        "annotated",
        "status",
        "ref_source",
        "n_frames",
        "object_rotation_total_deg",
        "object_rotation_max_deg",
        "object_rotation_cumulative_deg",
        "object_z_lift_m",
        "object_z_range_m",
        "lift_onset_frac",
        "object_xy_displacement_m",
        "object_xy_net_m",
        "object_jitter_accel_mean",
        "object_jitter_accel_max",
        "object_jitter_vel_signflips",
        "prelift_robot_base_travel_m",
        "prelift_n_frames",
        "gt_contact_frac",
        "gt_contact_frac_5cm",
        "gt_contact_source",
        "gt_min_dist_min_m",
        # ref (OmniRetarget/ref_fk kinematic) object metrics
        "ref_rot_total_deg",
        "ref_rot_max_deg",
        "ref_z_lift_m",
        "ref_lift_onset_frac",
        # E109 comparison
        "e109_omni_object_z_range_m",
        "e109_spider_object_z_range_m",
        "e109_omni_object_xy_disp_m",
        "e109_spider_object_xy_disp_m",
        "e109_omni_hand_near_5cm_frac",
        "e109_spider_hand_near_5cm_frac",
        "e109_omni_object_floor_frac",
        "e109_spider_object_floor_frac",
    ]
    tsv_rows = []

    for row in read_variants():
        case_id = row[COL_CASE_ID]
        e109_case_id = row[COL_E109_CASE_ID]
        object_key = row[COL_OBJECT_KEY]
        person_idx = int(row[COL_PERSON_IDX])
        status = row[COL_RUN_STATUS]
        derived_task = row[COL_DERIVED_TASK]
        spider_scene = resolve_spider_scene(row[COL_SPIDER_SCENE], derived_task)
        baseline_npz = resolve(row[COL_BASELINE_NPZ])
        omni_qpos_path = resolve(row[COL_OMNI_QPOS])
        omni_scene = resolve(row[COL_OMNI_SCENE])

        if status == "already_done":
            sim_npz = resolve(row[COL_REUSE_OUTDIR_NPZ])
        else:
            sim_npz = (
                LEGOBJ_CSV_DIR
                / f"E143_{case_id}_raw_mask_ref_fk_outdir_full"
                / "trajectory_mjwp_act.npz"
            )

        rec: dict = {
            "case_id": case_id,
            "object_key": object_key,
            "person_idx": person_idx,
            "annotated": case_id not in NOT_ANNOTATED,
            "status": status,
        }

        sim_qpos = load_qpos(sim_npz)
        if sim_qpos is None:
            rec["status"] = "missing_sim"
            rec.update(gt_contact_metrics(case_id, person_idx))
            merge_e109(rec, e109, e109_case_id)
            print(f"[SKIP] {case_id}: missing sim qpos {sim_npz}")
            tsv_rows.append(rec)
            with open(PER_CASE_DIR / f"{case_id}.json", "w") as f:
                json.dump(rec, f, indent=2)
            continue

        if spider_scene is None:
            rec["status"] = "missing_scene"
            rec["ref_source"] = "none"
            # GT contact and E109 metrics need no scene
            rec.update(gt_contact_metrics(case_id, person_idx))
            merge_e109(rec, e109, e109_case_id)
            print(f"[SKIP] {case_id}: no loadable spider scene (derived={derived_task})")
            tsv_rows.append(rec)
            with open(PER_CASE_DIR / f"{case_id}.json", "w") as f:
                json.dump(rec, f, indent=2)
            continue

        try:
            pos, quat = object_world_pose(spider_scene, sim_qpos)
        except Exception as e:  # noqa: BLE001
            rec["status"] = f"sim_extract_error: {e}"
            print(f"[ERR] {case_id}: {e}")
            tsv_rows.append(rec)
            continue

        robot_xy = sim_qpos[:, :2]  # floating_base_joint xy (entity 0)

        rot = rotation_metrics(quat)
        lift = lift_metrics(pos)
        xy = xy_metrics(pos)
        jit = jitter_metric(pos)
        walk = walk_up_metric(robot_xy, lift["_lift_onset_idx"])
        gt = gt_contact_metrics(case_id, person_idx)

        rec.update(
            {
                "n_frames": int(len(pos)),
                "object_rotation_total_deg": rot["object_rotation_total_deg"],
                "object_rotation_max_deg": rot["object_rotation_max_deg"],
                "object_rotation_cumulative_deg": rot["object_rotation_cumulative_deg"],
                "object_z_lift_m": lift["object_z_lift_m"],
                "object_z_range_m": lift["object_z_range_m"],
                "object_z_init_m": lift["object_z_init_m"],
                "lift_onset_frac": lift["lift_onset_frac"],
                **xy,
                **jit,
                **walk,
                **gt,
            }
        )

        # --- reference (OmniRetarget / ref_fk kinematic) object trajectory ---
        ref_source = "none"
        ref_qpos = None
        ref_scene = None
        if omni_qpos_path.exists():
            ref_qpos = load_qpos(omni_qpos_path)
            ref_scene = omni_scene
            ref_source = "omni_qpos"
        if ref_qpos is None and baseline_npz.exists():
            ref_qpos = load_qpos(baseline_npz)
            ref_scene = spider_scene  # baseline is (T,2,42), same scene
            ref_source = "baseline_npz"
        rec["ref_source"] = ref_source
        if ref_qpos is not None and ref_scene is not None:
            try:
                rpos, rquat = object_world_pose(ref_scene, ref_qpos)
                rrot = rotation_metrics(rquat)
                rlift = lift_metrics(rpos)
                rec["ref_rot_total_deg"] = rrot["object_rotation_total_deg"]
                rec["ref_rot_max_deg"] = rrot["object_rotation_max_deg"]
                rec["ref_z_lift_m"] = rlift["object_z_lift_m"]
                rec["ref_lift_onset_frac"] = rlift["lift_onset_frac"]
                rec["_ref_z_series"] = downsample(rlift["_z_series"])
                rec["_ref_rot_series"] = downsample(rrot["_rel_series"])
            except Exception as e:  # noqa: BLE001
                rec["ref_source"] = f"{ref_source}_error: {e}"

        # --- E109 method-level comparison ---
        merge_e109(rec, e109, e109_case_id)

        # per-frame downsampled series for plotting
        rec["_z_series"] = downsample(lift["_z_series"])
        rec["_rot_series"] = downsample(rot["_rel_series"])
        rec["_robot_xy_series"] = {
            "x": downsample(robot_xy[:, 0]),
            "y": downsample(robot_xy[:, 1]),
        }
        rec["_obj_xy_series"] = {
            "x": downsample(pos[:, 0]),
            "y": downsample(pos[:, 1]),
        }

        with open(PER_CASE_DIR / f"{case_id}.json", "w") as f:
            json.dump(rec, f, indent=2)

        print(
            f"[OK] {case_id:26s} rot_tot={rec['object_rotation_total_deg']:6.1f} "
            f"rot_max={rec['object_rotation_max_deg']:6.1f} "
            f"lift={rec['object_z_lift_m']:.3f} "
            f"onset={rec['lift_onset_frac'] if not math.isnan(rec['lift_onset_frac']) else float('nan'):.2f} "
            f"gt_c={rec['gt_contact_frac']:.2f} "
            f"prelift_travel={rec['prelift_robot_base_travel_m']:.3f} "
            f"jit={rec['object_jitter_accel_mean']:.4f} ref={ref_source}"
        )
        tsv_rows.append(rec)

    # write TSV
    tsv_path = OUT_DIR / "object_trajectory_metrics.tsv"
    with open(tsv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=tsv_cols, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for r in tsv_rows:
            w.writerow(r)
    print(f"\nWrote {tsv_path}")
    print(f"Wrote {len(tsv_rows)} per-case JSON files to {PER_CASE_DIR}")


if __name__ == "__main__":
    main()
