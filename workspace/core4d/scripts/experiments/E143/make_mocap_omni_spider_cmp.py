#!/usr/bin/env python3
"""Build E143 three-panel videos: raw SMPL-X/object mesh, E143 ref, E143 sim."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import trimesh


REPO = Path(__file__).resolve().parents[4]
VARIANTS_TSV = REPO / "workspace/core4d/scripts/E143/variants.tsv"
OUT_ROOT = REPO / "workspace/core4d/results/E143/mocap_omni_spider_cmp"
TASK_ROOT = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
RAW_ROOT = Path("/mnt/a0ccc676-9496-49f8-a861-f8a1797dec52/mocap_data/CORE4D/CORE4D_Real")
RAW_MOTION_ROOT = RAW_ROOT / "human_object_motions"
OBJECT_ROOT = RAW_ROOT / "object_models"
SMPLX_MODEL_NPZ = Path("/mnt/a0ccc676-9496-49f8-a861-f8a1797dec52/mocap_data/smplx/smplx/SMPLX_NEUTRAL.npz")

FIELDS = [
    "ordinal",
    "variant",
    "e109_case_id",
    "case_id",
    "object_key",
    "target_variant_id",
    "source_task",
    "derived_task",
    "person_idx",
    "split",
    "ablation",
    "mask_path",
    "mask_kind",
    "source_mask_path",
    "baseline_npz_path",
    "baseline_run_id",
    "omni_qpos_path",
    "omni_scene_xml",
    "spider_scene_xml",
    "override",
    "run_status",
    "reuse_source_exp",
    "reuse_npz_path",
    "reuse_video_path",
    "reuse_outdir_npz_path",
]

LEGACY = {
    "box023_person1": ("20231008", "045", "person1", "Box023"),
    "box023_person2": ("20231008", "045", "person2", "Box023"),
    "box021_person1": ("20231018", "030", "person1", "Box021"),
    "box025_person1": ("20231011", "048", "person1", "Box025"),
    "box025_person2": ("20231011", "048", "person2", "Box025"),
}


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def repo_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO / p


def read_variants() -> list[dict[str, str]]:
    with VARIANTS_TSV.open("r", encoding="utf-8", newline="") as f:
        lines = (line for line in f if line.strip() and not line.startswith("#"))
        return list(csv.DictReader(lines, fieldnames=FIELDS, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def put_label(img: np.ndarray, text: str, org: tuple[int, int], scale: float = 0.58, color: tuple[int, int, int] = (255, 255, 255)) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def source_info(row: dict[str, str]) -> tuple[str, str, str, str]:
    if row["source_task"] in LEGACY:
        return LEGACY[row["source_task"]]
    info_path = TASK_ROOT / row["source_task"] / "task_info.json"
    if info_path.is_file():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        if info.get("date") and info.get("seq"):
            return (
                str(info["date"]),
                str(info["seq"]),
                str(info.get("person", f"person{int(row['person_idx']) + 1}")),
                str(info.get("object_name", row["object_key"].replace("box", "Box").replace("bucket", "Bucket"))),
            )
    parts = row["source_task"].split("_")
    date_idx = next((i for i, part in enumerate(parts) if len(part) == 8 and part.isdigit()), None)
    if date_idx is None:
        raise ValueError(f"cannot parse raw date/seq from {row['source_task']}")
    date = parts[date_idx]
    next_part = parts[date_idx + 1]
    if len(next_part) == 1 and next_part.isdigit():
        date = f"{date}_{next_part}"
        seq = parts[date_idx + 2]
    else:
        seq = next_part
    person = f"person{int(row['person_idx']) + 1}"
    object_name = row["object_key"].replace("box", "Box").replace("bucket", "Bucket")
    return date, seq, person, object_name


def yup_to_zup_points(points: np.ndarray) -> np.ndarray:
    out = np.empty_like(points)
    out[..., 0] = points[..., 0]
    out[..., 1] = -points[..., 2]
    out[..., 2] = points[..., 1]
    return out


def load_smplx_faces() -> np.ndarray:
    data = np.load(SMPLX_MODEL_NPZ, allow_pickle=True)
    faces = np.asarray(data["f"], dtype=np.int32)
    if faces.shape != (20908, 3):
        raise ValueError(f"unexpected SMPL-X faces shape={faces.shape}")
    return faces


def object_mesh_path(object_name: str) -> Path:
    obj = object_name.lower()
    family = "".join(ch for ch in obj if not ch.isdigit())
    path = OBJECT_ROOT / family / f"{obj}_m.obj"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_object_mesh(object_name: str) -> tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.load(object_mesh_path(object_name), force="mesh", process=False)
    return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int32)


def load_raw(row: dict[str, str]) -> dict[str, Any]:
    date, seq, target_person, object_name = source_info(row)
    root = RAW_MOTION_ROOT / date / seq
    people = {}
    frame_count = None
    for person in ("person1", "person2"):
        path = root / f"{person}_poses.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        raw = np.load(path, allow_pickle=True)["arr_0"].item()
        verts = yup_to_zup_points(np.asarray(raw["vertices"], dtype=np.float64))
        people[person] = verts
        frame_count = verts.shape[0] if frame_count is None else min(frame_count, verts.shape[0])
    obj_pose_path = root / "smooth_objposes.npy"
    if not obj_pose_path.is_file():
        raise FileNotFoundError(obj_pose_path)
    obj_T_yup = np.asarray(np.load(obj_pose_path), dtype=np.float64)
    frame_count = min(int(frame_count or obj_T_yup.shape[0]), obj_T_yup.shape[0])
    obj_vertices, obj_faces = load_object_mesh(object_name)
    return {
        "date": date,
        "seq": seq,
        "target_person": target_person,
        "object_name": object_name,
        "raw_root": root,
        "people": {person: verts[:frame_count] for person, verts in people.items()},
        "obj_T_yup": obj_T_yup[:frame_count],
        "obj_vertices_yup": obj_vertices,
        "obj_faces": obj_faces,
        "frame_count": frame_count,
    }


def task_info_paths(row: dict[str, str]) -> list[Path]:
    paths = []
    for key in ("source_task", "derived_task"):
        task = row.get(key, "")
        if task:
            paths.append(TASK_ROOT / task / "task_info.json")
    return paths


def source_qpos_candidates(source_qpos: str, task_info_path: Path) -> list[Path]:
    p = Path(source_qpos)
    roots = [
        Path("/home/ubuntu/Workspace"),
        REPO,
        task_info_path.parent,
        TASK_ROOT,
    ]
    candidates = [p] if p.is_absolute() else [root / p for root in roots]
    fixed = []
    for candidate in candidates:
        fixed.append(candidate)
        if "data_constructon" in str(candidate):
            fixed.append(Path(str(candidate).replace("data_constructon", "data_construction")))
    unique = []
    seen = set()
    for candidate in fixed:
        normalized = candidate.resolve(strict=False)
        if normalized not in seen:
            seen.add(normalized)
            unique.append(normalized)
    return unique


def trim_candidates_from_task_info(row: dict[str, str]) -> list[Path]:
    candidates = []
    for info_path in task_info_paths(row):
        if not info_path.is_file():
            continue
        info = json.loads(info_path.read_text(encoding="utf-8"))
        source_qpos = info.get("source_qpos")
        if not source_qpos:
            continue
        for qpos_path in source_qpos_candidates(str(source_qpos), info_path):
            candidates.extend(
                [
                    qpos_path.parent / "trim_window.json",
                    qpos_path.parent.parent / "trim_window.json",
                ]
            )
    return candidates


def fallback_trim_candidates(row: dict[str, str]) -> list[Path]:
    source = row["source_task"]
    case_id = row["case_id"]
    candidates = []
    if source.startswith("e091_"):
        candidates.append(
            Path("/home/ubuntu/Workspace/holosoma/workspace/v3/data_construction_v2/results/stage2b_medium/results")
            / f"holosoma_{source}"
            / "trim_window.json"
        )
    if source.startswith("d003_box021_"):
        candidates.append(
            Path("/home/ubuntu/Workspace/holosoma/workspace/v3/data_construction/results/d003_omniretarget_spider_production/results")
            / f"holosoma_{source}"
            / "trim_window.json"
        )
    if source.startswith("bucket004_"):
        candidates.append(
            REPO
            / "workspace/core4d/results/E108/s3_retarget/omnirt_v1/ref_fk/results/omnirt_v1_ref_fk"
            / f"holosoma_dcv3_omnirt_v1_ref_fk_{source}"
            / "trim_window.json"
        )
        candidates.append(
            REPO
            / "workspace/core4d/results/E108/archive_legacy/stage2b_bucket004_person1_execute/results/omnirt_v1_ref_fk"
            / f"holosoma_dcv3_omnirt_v1_ref_fk_{source}"
            / "trim_window.json"
        )
    if source == "box023_person2" or case_id == "box023_person2":
        candidates.extend(
            [
                REPO / "workspace/core4d/results/E077/holosoma_box023_person2/trim_window.json",
                REPO / "workspace/core4d/results/E079/holosoma_box023_person2/trim_window.json",
            ]
        )
    return candidates


def resolve_trim_window(row: dict[str, str], raw: dict[str, Any]) -> dict[str, Any]:
    candidates = fallback_trim_candidates(row) + trim_candidates_from_task_info(row)
    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve(strict=False)
        if candidate in seen:
            continue
        seen.add(candidate)
        if not candidate.is_file():
            continue
        trim = json.loads(candidate.read_text(encoding="utf-8"))
        start = int(trim["trim_start"])
        end = int(trim["trim_end"])
        trim_frames = int(trim.get("trim_frames", trim.get("trimmed_frames", end - start)))
        if start < 0 or end <= start:
            raise ValueError(f"invalid trim window {candidate}: {start}:{end}")
        clipped_start = min(start, raw["frame_count"] - 1)
        clipped_end = min(end, raw["frame_count"])
        if clipped_end <= clipped_start:
            raise ValueError(f"trim window outside raw frame range {candidate}: {start}:{end}, raw={raw['frame_count']}")
        return {
            "trim_start": clipped_start,
            "trim_end": clipped_end,
            "trim_frames": trim_frames,
            "trim_window_path": candidate,
            "trim_was_clipped": clipped_start != start or clipped_end != end,
        }
    raise FileNotFoundError(f"trim_window.json not found for {row['case_id']} ({row['source_task']})")


def spider_video_path(row: dict[str, str]) -> Path:
    if row["run_status"] == "already_done":
        return repo_path(row["reuse_video_path"])
    return REPO / "workspace/core4d/results/E143/cem/full" / f"{row['variant']}_full.mp4"


def world_object_vertices_zup(raw: dict[str, Any], idx: int) -> np.ndarray:
    T = raw["obj_T_yup"][idx]
    verts = raw["obj_vertices_yup"]
    world_yup = (T[:3, :3] @ verts.T).T + T[:3, 3]
    return yup_to_zup_points(world_yup)


def projection_basis(yaw_deg: float, pitch_deg: float = 15.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yaw = np.deg2rad(float(yaw_deg))
    pitch = np.deg2rad(float(pitch_deg))
    forward = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)], dtype=np.float64)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)
    return right, up, forward


def build_projection(raw: dict[str, Any], width: int, height: int, trim_start: int, trim_end: int, yaw_deg: float) -> dict[str, Any]:
    samples = []
    for person in ("person1", "person2"):
        verts = raw["people"][person]
        idxs = np.linspace(trim_start, trim_end - 1, min(12, trim_end - trim_start), dtype=int)
        samples.append(verts[idxs].reshape(-1, 3))
    obj_samples = []
    idxs = np.linspace(trim_start, trim_end - 1, min(12, trim_end - trim_start), dtype=int)
    for idx in idxs:
        obj_samples.append(world_object_vertices_zup(raw, int(idx)))
    samples.append(np.concatenate(obj_samples, axis=0))
    all_pts = np.concatenate(samples, axis=0)
    center = np.nanmean(all_pts, axis=0)
    right, up, forward = projection_basis(yaw_deg)
    xy = np.stack([(all_pts - center) @ right, (all_pts - center) @ up], axis=-1)
    extent = np.nanmax(np.abs(xy), axis=0)
    scale = 0.86 * min(width / max(1e-6, 2 * extent[0]), height / max(1e-6, 2 * extent[1]))
    return {"center": center, "right": right, "up": up, "forward": forward, "scale": scale, "width": width, "height": height, "yaw_deg": float(yaw_deg)}


def bbox2d(points: np.ndarray, proj: dict[str, Any]) -> np.ndarray:
    pix, _ = project(points, proj)
    return np.array([pix[:, 0].min(), pix[:, 1].min(), pix[:, 0].max(), pix[:, 1].max()], dtype=np.float64)


def bbox_area(box: np.ndarray) -> float:
    return float(max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]))


def bbox_overlap(a: np.ndarray, b: np.ndarray) -> float:
    x0 = max(float(a[0]), float(b[0]))
    y0 = max(float(a[1]), float(b[1]))
    x1 = min(float(a[2]), float(b[2]))
    y1 = min(float(a[3]), float(b[3]))
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def choose_mocap_yaw(raw: dict[str, Any], width: int, height: int, trim_start: int, trim_end: int) -> tuple[float, float]:
    candidates = [-135.0, -90.0, -45.0, 0.0, 45.0, 90.0, 135.0, 180.0]
    sample_count = min(5, trim_end - trim_start)
    frame_idxs = np.linspace(trim_start, trim_end - 1, sample_count, dtype=int)
    best = None
    for yaw_deg in candidates:
        proj = build_projection(raw, width, height, trim_start, trim_end, yaw_deg)
        penalties = []
        for idx in frame_idxs:
            p1 = bbox2d(raw["people"]["person1"][int(idx)], proj)
            p2 = bbox2d(raw["people"]["person2"][int(idx)], proj)
            obj = bbox2d(world_object_vertices_zup(raw, int(idx)), proj)
            p1_area = max(bbox_area(p1), 1.0)
            p2_area = max(bbox_area(p2), 1.0)
            obj_area = max(bbox_area(obj), 1.0)
            person_overlap = bbox_overlap(p1, p2) / max(1.0, min(p1_area, p2_area))
            object_overlap = (
                bbox_overlap(obj, p1) / max(1.0, min(obj_area, p1_area))
                + bbox_overlap(obj, p2) / max(1.0, min(obj_area, p2_area))
            )
            obj_wh = np.array([max(1.0, obj[2] - obj[0]), max(1.0, obj[3] - obj[1])])
            obj_aspect_penalty = max(obj_wh) / max(1.0, min(obj_wh)) * 0.02
            penalties.append(2.0 * person_overlap + object_overlap + obj_aspect_penalty)
        score = float(np.mean(penalties))
        if best is None or score < best[1]:
            best = (yaw_deg, score)
    if best is None:
        return -45.0, float("nan")
    return best


def project(points: np.ndarray, proj: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    rel = points - proj["center"]
    x = rel @ proj["right"]
    y = rel @ proj["up"]
    z = rel @ proj["forward"]
    pix = np.stack(
        [
            proj["width"] * 0.5 + x * proj["scale"],
            proj["height"] * 0.56 - y * proj["scale"],
        ],
        axis=-1,
    )
    return pix.astype(np.float32), z.astype(np.float32)


def face_lighting(vertices: np.ndarray, faces: np.ndarray, base: tuple[int, int, int]) -> np.ndarray:
    tri = vertices[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(norms, 1e-8)
    light = np.array([0.25, -0.45, 0.86], dtype=np.float64)
    light /= np.linalg.norm(light)
    shade = np.clip(0.48 + 0.52 * (normals @ light), 0.30, 1.0)
    return (np.asarray(base, dtype=np.float64)[None, :] * shade[:, None]).clip(0, 255).astype(np.uint8)


def draw_mesh(
    img: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    proj: dict[str, Any],
    base_color: tuple[int, int, int],
    stride: int = 2,
) -> None:
    faces_use = faces[::stride] if stride > 1 else faces
    pts, depth = project(vertices, proj)
    tri_pts = pts[faces_use]
    tri_depth = depth[faces_use].mean(axis=1)
    colors = face_lighting(vertices, faces_use, base_color)
    order = np.argsort(tri_depth)
    h, w = img.shape[:2]
    for face_idx in order:
        poly = np.rint(tri_pts[face_idx]).astype(np.int32)
        if poly[:, 0].max() < 0 or poly[:, 0].min() >= w or poly[:, 1].max() < 0 or poly[:, 1].min() >= h:
            continue
        cv2.fillConvexPoly(img, poly, tuple(int(x) for x in colors[face_idx]), lineType=cv2.LINE_AA)


def draw_meshes_global_depth(
    img: np.ndarray,
    meshes: list[tuple[np.ndarray, np.ndarray, tuple[int, int, int], int]],
    proj: dict[str, Any],
) -> None:
    triangles = []
    h, w = img.shape[:2]
    for vertices, faces, base_color, stride in meshes:
        faces_use = faces[::stride] if stride > 1 else faces
        pts, depth = project(vertices, proj)
        tri_pts = pts[faces_use]
        tri_depth = depth[faces_use].mean(axis=1)
        colors = face_lighting(vertices, faces_use, base_color)
        for i in range(faces_use.shape[0]):
            poly = np.rint(tri_pts[i]).astype(np.int32)
            if poly[:, 0].max() < 0 or poly[:, 0].min() >= w or poly[:, 1].max() < 0 or poly[:, 1].min() >= h:
                continue
            triangles.append((float(tri_depth[i]), poly, tuple(int(x) for x in colors[i])))
    for _, poly, color in sorted(triangles, key=lambda item: item[0]):
        cv2.fillConvexPoly(img, poly, color, lineType=cv2.LINE_AA)


def raw_mesh_frame(raw: dict[str, Any], frame_idx: int, proj: dict[str, Any], smplx_faces: np.ndarray, size: tuple[int, int], case_id: str) -> np.ndarray:
    width, height = size
    img = np.full((height, width, 3), 245, dtype=np.uint8)
    target = raw["target_person"]
    person_colors = {
        "person1": (60, 170, 75) if target == "person1" else (155, 155, 155),
        "person2": (60, 170, 75) if target == "person2" else (195, 115, 45),
    }
    obj_vertices = world_object_vertices_zup(raw, frame_idx)
    draw_meshes_global_depth(
        img,
        [
            (raw["people"]["person1"][frame_idx], smplx_faces, person_colors["person1"], 1),
            (raw["people"]["person2"][frame_idx], smplx_faces, person_colors["person2"], 1),
            (obj_vertices, raw["obj_faces"], (50, 75, 220), 1),
        ],
        proj,
    )

    for person in ("person1", "person2"):
        pelvis, _ = project(raw["people"][person][frame_idx][0:1], proj)
        pos = tuple(np.rint(pelvis[0]).astype(int))
        label = f"{person}{' RETARGET' if person == target else ''}"
        put_label(img, label, (pos[0] + 6, pos[1] - 8), 0.48, person_colors[person])

    put_label(img, "MOCAP RAW (SMPL-X + object mesh)", (14, 28), 0.62)
    put_label(
        img,
        f"{case_id} | target={target} | {raw['date']}/{raw['seq']} | trim={raw['trim_start']}:{raw['trim_end']} | yaw={raw['mocap_yaw_deg']:.0f}",
        (14, height - 16),
        0.46,
    )
    return img


def split_ref_sim(frame: np.ndarray, size: tuple[int, int], case_id: str, target_person: str) -> tuple[np.ndarray, np.ndarray]:
    h, w = frame.shape[:2]
    mid = w // 2
    ref = cv2.resize(frame[:, :mid], size, interpolation=cv2.INTER_AREA)
    sim = cv2.resize(frame[:, mid:], size, interpolation=cv2.INTER_AREA)
    put_label(ref, "OmniRetarget/ref (from E143 left half)", (14, 28), 0.55)
    put_label(ref, f"{case_id} | retarget={target_person}", (14, size[1] - 16), 0.46)
    put_label(sim, "SPIDER raw_mask_ref_fk sim (from E143 right half)", (14, 28), 0.55)
    put_label(sim, f"{case_id} | retarget={target_person}", (14, size[1] - 16), 0.46)
    return ref, sim


def make_video(
    row: dict[str, str],
    out_dir: Path,
    smplx_faces: np.ndarray,
    panel_size: tuple[int, int],
    fps: int,
    max_frames: int | None,
    yaw_deg: float | None,
) -> dict[str, Any]:
    case_id = row["case_id"]
    raw = load_raw(row)
    trim = resolve_trim_window(row, raw)
    raw.update(trim)
    if yaw_deg is None:
        chosen_yaw, yaw_score = choose_mocap_yaw(raw, panel_size[0], panel_size[1], raw["trim_start"], raw["trim_end"])
    else:
        chosen_yaw, yaw_score = float(yaw_deg), float("nan")
    raw["mocap_yaw_deg"] = float(chosen_yaw)
    raw["mocap_yaw_score"] = float(yaw_score)
    src_video = spider_video_path(row)
    if not src_video.is_file():
        raise FileNotFoundError(src_video)
    cap = cv2.VideoCapture(str(src_video))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {src_video}")
    src_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    n = src_frames
    if max_frames:
        n = min(n, max_frames)
    if n <= 0:
        raise RuntimeError(f"no frames for {case_id}")
    raw_idxs = np.linspace(raw["trim_start"], raw["trim_end"] - 1, n, dtype=int)
    src_idxs = np.linspace(0, src_frames - 1, n, dtype=int)
    proj = build_projection(raw, panel_size[0], panel_size[1], raw["trim_start"], raw["trim_end"], raw["mocap_yaw_deg"])
    out_path = out_dir / f"{case_id}_mocap_omni_spider_cmp.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (panel_size[0] * 3, panel_size[1]))
    for i in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(src_idxs[i]))
        ok, src = cap.read()
        if not ok:
            break
        left = raw_mesh_frame(raw, int(raw_idxs[i]), proj, smplx_faces, panel_size, case_id)
        mid, right = split_ref_sim(src, panel_size, case_id, raw["target_person"])
        writer.write(np.concatenate([left, mid, right], axis=1))
    cap.release()
    writer.release()
    return {
        "case_id": case_id,
        "source_task": row["source_task"],
        "target_person": raw["target_person"],
        "raw_date": raw["date"],
        "raw_seq": raw["seq"],
        "frames": n,
        "status": "ok",
        "comparison_video": rel(out_path),
        "e143_ref_sim_video": rel(src_video),
        "raw_root": str(raw["raw_root"]),
        "object_name": raw["object_name"],
        "trim_window_path": rel(raw["trim_window_path"]),
        "trim_start": raw["trim_start"],
        "trim_end": raw["trim_end"],
        "trim_frames": raw["trim_frames"],
        "trim_was_clipped": raw["trim_was_clipped"],
        "mocap_yaw_deg": raw["mocap_yaw_deg"],
        "mocap_yaw_score": raw["mocap_yaw_score"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT_ROOT)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--only-case", default="")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--yaw-deg", type=float, default=None, help="Override automatic mocap yaw selection.")
    args = parser.parse_args()

    rows = read_variants()
    if args.only_case:
        rows = [row for row in rows if row["case_id"] == args.only_case]
    if args.limit:
        rows = rows[: args.limit]
    out_dir = repo_path(args.out_dir)
    smplx_faces = load_smplx_faces()
    manifest_rows = []
    for row in rows:
        try:
            result = make_video(row, out_dir, smplx_faces, (args.width, args.height), args.fps, args.max_frames, args.yaw_deg)
        except Exception as exc:
            result = {
                "case_id": row["case_id"],
                "source_task": row["source_task"],
                "target_person": f"person{int(row['person_idx']) + 1}",
                "raw_date": "",
                "raw_seq": "",
                "frames": "",
                "status": "error",
                "comparison_video": "",
                "e143_ref_sim_video": rel(spider_video_path(row)),
                "raw_root": "",
                "object_name": "",
                "trim_window_path": "",
                "trim_start": "",
                "trim_end": "",
                "trim_frames": "",
                "trim_was_clipped": "",
                "mocap_yaw_deg": "",
                "mocap_yaw_score": "",
                "error": f"{type(exc).__name__}: {exc}",
            }
        manifest_rows.append(result)
        print(f"{result['case_id']}: {result['status']} {result.get('comparison_video', '')}")

    fields = [
        "case_id",
        "source_task",
        "target_person",
        "raw_date",
        "raw_seq",
        "frames",
        "status",
        "comparison_video",
        "e143_ref_sim_video",
        "raw_root",
        "object_name",
        "trim_window_path",
        "trim_start",
        "trim_end",
        "trim_frames",
        "trim_was_clipped",
        "mocap_yaw_deg",
        "mocap_yaw_score",
        "error",
    ]
    write_tsv(out_dir / "mocap_omni_spider_cmp_manifest.tsv", manifest_rows, fields)
    summary = {
        "rows": len(manifest_rows),
        "ok": sum(1 for row in manifest_rows if row["status"] == "ok"),
        "error": sum(1 for row in manifest_rows if row["status"] != "ok"),
        "out_dir": rel(out_dir),
        "layout": "left raw SMPL-X/object mesh; middle E143 ref left half; right E143 sim right half",
    }
    (out_dir / "mocap_omni_spider_cmp_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
