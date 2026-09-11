#!/usr/bin/env python3
"""CORE4D SMPLX human GT reference, aligned to the rollout (scene) frame.

The holosoma-gauge metrics (foot sliding, contact precision) are defined against
a *human* demo in holosoma's original evaluator. Our pipeline is
    CORE4D SMPLX (raw)  ->  OmniRetarget  ->  SPIDER-CEM (rollout, scene frame).
This module rebuilds the SMPLX human toe / wrist keypoints and the object pose,
expressed in the SAME frame and scale as the rollout qpos, so those two metrics
compare the robot rollout against the true human GT.

Frames (verified empirically, box001_20231003_2_039_p1):
  * raw CORE4D is Y-up (toe y ~= 0 = floor; horizontal plane = x,z), human scale.
  * the rollout / kinematic-reference scene is Z-up, robot (G1) scale.
  * raw -> scene is a clean similarity (R, t, s): fitting the rigid OBJECT
    trajectory (translation + body-frame rigid points) gives ~0 mm residual, and
    the mapped SMPLX pelvis height matches the robot pelvis height.
  * the rollout has N (e.g. 100) frames vs raw's T_raw (e.g. 116); the rollout is
    a uniform resample of a trimmed raw window [a..b]. We recover (a, b) by a
    brute-force monotonic linear time map minimising the object-fit residual.

The rigid OBJECT is the only clean shared observable between raw and the rollout
(retargeting is non-linear on the body), so it anchors the transform; the mapped
pelvis height is an independent cross-check, not a fit target.

SMPLX native joint order (raw ``joints`` (T,127,3)): toe = 10/11, wrist = 20/21.
Wrists are taken from RAW (the true SMPLX wrist), never the converted
intermediate (whose wrist may be fingertip-replaced).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SMPLX_ROOT = Path(
    "/mnt/ali-sh-1/usr/xiayibo/xyb_data_tidal_alsh/other-datasets/mocap_data/"
    "CORE4D/CORE4D_Real/human_object_motions"
)

# SMPLX native joint indices (raw joints (T,127,3)).
TOE_JOINTS = (10, 11)      # L, R big-toe
WRIST_JOINTS = (20, 21)    # L, R wrist
PELVIS_JOINT = 0

NQ_ROBOT = 36              # rollout qpos: 36 robot dofs + 7 object freejoint
OBJ_POS_SLICE = slice(NQ_ROBOT, NQ_ROBOT + 3)
OBJ_QUAT_SLICE = slice(NQ_ROBOT + 3, NQ_ROBOT + 7)

_CASE_RE = re.compile(
    r"^(?P<obj>[a-z]+\d+)_(?P<date>\d{8})(?:_(?P<sub>\d+))?_(?P<seq>\d+)_p(?P<person>\d)$"
)


@dataclass
class SmplxReference:
    """SMPLX GT keypoints aligned 1:1 with the N rollout frames (scene frame)."""

    toe_scene: np.ndarray       # (N, 2, 3)
    wrist_scene: np.ndarray     # (N, 2, 3)
    obj_pos: np.ndarray         # (N, 3)  reference object position (scene)
    obj_quat: np.ndarray        # (N, 4)  reference object orientation (wxyz, scene)
    align_residual_m: float     # mean object-fit residual (gate)
    align_residual_max_m: float
    scale: float                # human -> scene similarity scale
    pelvis_check_m: float        # mean |mapped SMPLX pelvis - robot pelvis| (cross-check)
    time_map: tuple[int, int]    # (a, b) raw window resampled to N frames
    status: str                  # "ok" or "NO_GT_ALIGN:<reason>"


def parse_case(case_id: str) -> tuple[str, str, int]:
    """case_id -> (raw_date_dir, seq_dir, person_number[1-based])."""
    m = _CASE_RE.match(case_id)
    if not m:
        raise ValueError(f"cannot parse case_id: {case_id}")
    date = m.group("date")
    sub = m.group("sub")
    date_dir = f"{date}_{sub}" if sub else date
    return date_dir, m.group("seq"), int(m.group("person"))


def _load_raw(case_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load raw SMPLX joints + object pose. Returns (joints, obj_pos, obj_rot)."""
    date_dir, seq, person = parse_case(case_id)
    seq_dir = SMPLX_ROOT / date_dir / seq
    poses = seq_dir / f"person{person}_poses.npz"
    objp = seq_dir / "smooth_objposes.npy"
    if not poses.is_file() or not objp.is_file():
        raise FileNotFoundError(f"raw data missing for {case_id}: {seq_dir}")
    joints = np.load(poses, allow_pickle=True)["arr_0"].item()["joints"].astype(np.float64)
    op = np.load(objp).astype(np.float64)          # (T,4,4)
    return joints, op[:, :3, 3], op[:, :3, :3]


def _umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Similarity fit dst ~= s*R@src + t (Umeyama 1991, with scale)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    cov = (D.T @ S) / len(src)
    U, d, Vt = np.linalg.svd(cov)
    Dm = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        Dm[2, 2] = -1.0
    R = U @ Dm @ Vt
    var = (S ** 2).sum() / len(src)
    s = float(np.trace(np.diag(d) @ Dm) / var) if var > 1e-12 else 1.0
    t = mu_d - s * R @ mu_s
    return R, t, s


def _dtw_monotonic(scene_pts: np.ndarray, mapped_raw: np.ndarray) -> np.ndarray:
    """Assign each scene frame a raw index (monotonic non-decreasing) minimising the
    total scene<->mapped-raw distance. Handles non-uniform retarget time warps."""
    n, m = len(scene_pts), len(mapped_raw)
    dist = np.linalg.norm(scene_pts[:, None, :] - mapped_raw[None, :, :], axis=2)  # (n,m)
    cost = np.empty((n, m))
    cost[0] = dist[0]
    back = np.zeros((n, m), dtype=int)
    for i in range(1, n):
        prev = cost[i - 1]
        run_min = prev[0]
        run_arg = 0
        pm = np.empty(m)
        pa = np.empty(m, dtype=int)
        for j in range(m):
            if prev[j] < run_min:
                run_min = prev[j]
                run_arg = j
            pm[j] = run_min
            pa[j] = run_arg
        cost[i] = dist[i] + pm
        back[i] = pa
    j = int(np.argmin(cost[-1]))
    idx = np.empty(n, dtype=int)
    idx[-1] = j
    for i in range(n - 1, 0, -1):
        j = back[i, j]
        idx[i - 1] = j
    return idx


def _solve_transform(
    raw_obj_pos: np.ndarray, scene_obj_pos: np.ndarray, refine_iters: int = 6,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, float, float]:
    """Recover the raw->scene time map + similarity from the object TRANSLATION.

    Only the object translation trajectory is used: raw CORE4D and the MuJoCo scene
    define the object body frame differently (a constant mesh-canonical rotation
    offset), so the object *orientation* is NOT a consistent correspondence and
    would contaminate the fit. The translation path is a shared physical curve.

    Stage 1: brute-force a uniform (linear) monotonic time map raw[a..b]->N for the
    initialisation. Stage 2: ICP-style refinement -- alternate (fit similarity) and
    (monotonic-DTW re-assign frames) to absorb non-uniform retarget time warps.
    Returns (R, t, s, raw_index_map, residual_mean_m, residual_max_m).
    """
    n_roll = len(scene_obj_pos)
    t_raw = len(raw_obj_pos)
    best = None
    lo_max = max(1, t_raw // 3)
    hi_min = min(t_raw - 1, 2 * t_raw // 3)
    for a in range(0, lo_max + 1):
        for b in range(t_raw - 1, hi_min - 1, -1):
            if b - a < max(10, n_roll // 2):
                continue
            idx = np.linspace(a, b, n_roll).round().astype(int)
            R, t, s = _umeyama(raw_obj_pos[idx], scene_obj_pos)
            pred = s * (R @ raw_obj_pos[idx].T).T + t
            m = float(np.linalg.norm(pred - scene_obj_pos, axis=1).mean())
            if best is None or m < best[0]:
                best = (m, idx)
    idx = best[1]
    for _ in range(refine_iters):
        R, t, s = _umeyama(raw_obj_pos[idx], scene_obj_pos)
        mapped_all = s * (R @ raw_obj_pos.T).T + t          # all raw frames in scene
        new_idx = _dtw_monotonic(scene_obj_pos, mapped_all)
        if np.array_equal(new_idx, idx):
            break
        idx = new_idx
    R, t, s = _umeyama(raw_obj_pos[idx], scene_obj_pos)
    pred = s * (R @ raw_obj_pos[idx].T).T + t
    res = np.linalg.norm(pred - scene_obj_pos, axis=1)
    return R, t, s, idx, float(res.mean()), float(res.max())


_CACHE_FIELDS = ("toe_scene", "wrist_scene", "obj_pos", "obj_quat")


def build_smplx_reference(
    case_id: str, kin_ref_qpos: np.ndarray, residual_gate_m: float = 0.03,
    cache_dir: Path | str | None = None,
) -> SmplxReference:
    """Build the scene-frame SMPLX GT reference for one case.

    Args:
        case_id: e.g. ``box001_20231003_2_039_p1``.
        kin_ref_qpos: (N, nq) rollout kinematic reference (>= NQ_ROBOT+7 columns).
        residual_gate_m: object-fit residual above which the case is NO_GT_ALIGN.
        cache_dir: if set, ``<case_id>.npz`` there is loaded when present and
            written after a build (the reference is shared across a case's series).
    """
    cache_path = Path(cache_dir) / f"{case_id}.npz" if cache_dir is not None else None
    if cache_path is not None and cache_path.is_file():
        d = np.load(cache_path, allow_pickle=False)
        return SmplxReference(
            toe_scene=d["toe_scene"], wrist_scene=d["wrist_scene"],
            obj_pos=d["obj_pos"], obj_quat=d["obj_quat"],
            align_residual_m=float(d["align_residual_m"]),
            align_residual_max_m=float(d["align_residual_max_m"]),
            scale=float(d["scale"]), pelvis_check_m=float(d["pelvis_check_m"]),
            time_map=(int(d["time_map"][0]), int(d["time_map"][1])),
            status=str(d["status"]),
        )

    kin = np.asarray(kin_ref_qpos, dtype=np.float64)
    if kin.ndim == 3:
        kin = kin[:, 0, :]
    n_roll = len(kin)
    scene_obj_pos = kin[:, OBJ_POS_SLICE]

    joints, raw_obj_pos, _raw_obj_rot = _load_raw(case_id)

    R, t, s, idx, res_mean, res_max = _solve_transform(raw_obj_pos, scene_obj_pos)

    def to_scene(pts: np.ndarray) -> np.ndarray:  # (..,3) raw -> scene
        return (s * (R @ pts.reshape(-1, 3).T).T + t).reshape(pts.shape)

    toe_scene = to_scene(joints[idx][:, TOE_JOINTS, :])       # (N,2,3)
    wrist_scene = to_scene(joints[idx][:, WRIST_JOINTS, :])   # (N,2,3)
    pelvis_scene = to_scene(joints[idx][:, PELVIS_JOINT, :])  # (N,3)
    pelvis_check = float(np.linalg.norm(pelvis_scene - kin[:, 0:3], axis=1).mean())

    # reference object pose in the scene frame == the rollout object pose (the fit
    # maps raw object onto it to ~0 mm); use it directly for the reference boxes.
    obj_pos = scene_obj_pos.copy()
    obj_quat = kin[:, OBJ_QUAT_SLICE].copy()

    status = "ok" if res_mean <= residual_gate_m else f"NO_GT_ALIGN:residual={res_mean*1000:.1f}mm"
    ref = SmplxReference(
        toe_scene=toe_scene, wrist_scene=wrist_scene,
        obj_pos=obj_pos, obj_quat=obj_quat,
        align_residual_m=res_mean, align_residual_max_m=res_max,
        scale=s, pelvis_check_m=pelvis_check, time_map=(int(idx[0]), int(idx[-1])),
        status=status,
    )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache_path, toe_scene=toe_scene, wrist_scene=wrist_scene,
            obj_pos=obj_pos, obj_quat=obj_quat,
            align_residual_m=res_mean, align_residual_max_m=res_max,
            scale=s, pelvis_check_m=pelvis_check,
            time_map=np.array(ref.time_map), status=status,
        )
    return ref
