#!/usr/bin/env python3
"""Build deterministic five-event visual evidence for E170 paired videos.

Each E170 paired video gets frames for:

1. the frame immediately before first physical hand/object contact;
2. deepest lower-body/object SDF;
3. deepest hand/object SDF;
4. maximum object translational speed; and
5. the final frame.

The rollout is evaluated at the same flattened sim-substep resolution used by
the E168/E170 renderer.  Frame extraction is delegated to the installed
``video-frames`` skill so the evidence path is explicit and reproducible.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.core.core_metrics import (  # noqa: E402
    HAND_GEOMS,
    LOWERBODY_GEOMS,
    geom_object_sdf,
    mj_id,
    object_collision_geoms,
)


REPO = Path(__file__).resolve().parents[5]
DEFAULT_MANIFEST = REPO / "workspace/core4d/results/E170/s6_downstream/manifests/analysis_manifest.tsv"
DEFAULT_OUTPUT = REPO / "workspace/core4d/results/E170/s6_downstream/evidence/visual_qc"
DEFAULT_FRAME_SCRIPT = Path.home() / ".codex/skills/video-frames/scripts/frame.sh"
PAIR_ROOT = REPO / "workspace/core4d/results/E170/s6_downstream/render/full/paired"
EVENT_ORDER = (
    "pre_contact",
    "max_lower_body_penetration",
    "max_hand_penetration",
    "max_object_speed",
    "final",
)


def repo_path(raw: str | Path) -> Path:
    path = Path(raw)
    if path.exists():
        return path.resolve()
    text = str(raw)
    for marker in ("example_datasets/", "workspace/", "logs/"):
        if marker in text:
            return REPO / (marker + text.split(marker, 1)[1])
    return path if path.is_absolute() else REPO / path


def rel(path: Path | str) -> str:
    value = Path(path)
    resolved = value.resolve()
    # ``workspace/core4d/results`` is intentionally symlinked to persistent
    # storage on this workstation.  Preserve the logical repo-owned path in
    # manifests instead of leaking the machine-specific symlink target.
    logical_roots = (
        "workspace/core4d/results",
        "workspace",
        "example_datasets",
        "logs",
        "",
    )
    for logical in logical_roots:
        root = (REPO / logical).resolve()
        try:
            suffix = resolved.relative_to(root)
        except ValueError:
            continue
        return str(Path(logical) / suffix) if logical else str(suffix)
    return str(value)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def rollout(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    with np.load(path, allow_pickle=True) as data:
        raw_qpos = np.asarray(data["qpos"], dtype=np.float64)
        raw_time = np.asarray(data["time"], dtype=np.float64) if "time" in data.files else None
    if raw_qpos.ndim == 3:
        substeps = int(raw_qpos.shape[1])
        qpos = raw_qpos.reshape(-1, raw_qpos.shape[-1])
    elif raw_qpos.ndim == 2:
        substeps = 1
        qpos = raw_qpos
    else:
        raise ValueError(f"unsupported qpos shape {raw_qpos.shape}: {path}")
    if raw_time is None:
        time = np.arange(len(qpos), dtype=np.float64) / 60.0
    else:
        time = raw_time.reshape(-1)
        if len(time) != len(qpos):
            raise ValueError(f"time/qpos length mismatch: {path}: {len(time)} != {len(qpos)}")
    return qpos, time, substeps


def video_info(path: Path) -> dict[str, Any]:
    proc = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=nb_frames,avg_frame_rate,width,height,duration",
            "-of", "json", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(proc.stdout)["streams"][0]
    frames = int(stream.get("nb_frames") or 0)
    if frames <= 0:
        raise ValueError(f"ffprobe did not report a positive frame count: {path}")
    return {
        "frames": frames,
        "fps": stream.get("avg_frame_rate", ""),
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "duration": float(stream.get("duration") or math.nan),
    }


def first_reference_contact_frame(row: dict[str, str], substeps: int, frame_count: int) -> int | None:
    mask_path = repo_path(row["contact_mask"])
    with np.load(mask_path, allow_pickle=True) as data:
        if "spider_contact_mask_3cm" not in data.files:
            return None
        mask = np.asarray(data["spider_contact_mask_3cm"], dtype=bool)
    person = 0 if str(row.get("source_person", "")).lower() in {"person1", "p1", "0"} else 1
    if mask.ndim != 3 or person >= mask.shape[1]:
        raise ValueError(f"invalid contact mask shape/person: {mask_path}: {mask.shape}, person={person}")
    indices = np.flatnonzero(np.any(mask[:, person, :], axis=1))
    if not len(indices):
        return None
    return min(frame_count - 1, int(indices[0]) * substeps)


def event_diagnostics(row: dict[str, str]) -> tuple[list[dict[str, Any]], int]:
    qpos, times, substeps = rollout(repo_path(row["outdir_npz"]))
    model = mujoco.MjModel.from_xml_path(str(repo_path(row["scene_act"])))
    if qpos.shape[1] != model.nq:
        raise ValueError(f"{row['case_id']}: rollout nq={qpos.shape[1]} != scene nq={model.nq}")
    data = mujoco.MjData(model)
    object_gids = object_collision_geoms(model)
    object_set = set(object_gids)
    object_body = mj_id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    hand_gids = [gid for name in HAND_GEOMS if (gid := mj_id(model, mujoco.mjtObj.mjOBJ_GEOM, name)) >= 0]
    lower_gids = [gid for name in LOWERBODY_GEOMS if (gid := mj_id(model, mujoco.mjtObj.mjOBJ_GEOM, name)) >= 0]
    if object_body < 0 or not hand_gids or not lower_gids:
        raise ValueError(f"{row['case_id']}: missing object/hand/lower-body ids")

    hand_sdf: list[float] = []
    lower_sdf: list[float] = []
    object_xyz: list[np.ndarray] = []
    physical_contact: list[bool] = []
    for frame in qpos:
        data.qpos[:] = frame
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        hand_sdf.append(min(geom_object_sdf(model, data, gid, object_gids) for gid in hand_gids))
        lower_sdf.append(min(geom_object_sdf(model, data, gid, object_gids) for gid in lower_gids))
        object_xyz.append(data.xpos[object_body].copy())
        contact = False
        for ci in range(data.ncon):
            pair = {int(data.contact[ci].geom1), int(data.contact[ci].geom2)}
            if object_set & pair and any(gid in hand_gids for gid in pair - object_set):
                contact = True
                break
        physical_contact.append(contact)

    first_physical = np.flatnonzero(np.asarray(physical_contact, dtype=bool))
    if len(first_physical):
        contact_frame = int(first_physical[0])
        pre_basis = "frame_before_first_physical_hand_object_contact"
    else:
        fallback = first_reference_contact_frame(row, substeps, len(qpos))
        contact_frame = 0 if fallback is None else fallback
        pre_basis = "frame_before_first_reference_3cm_contact" if fallback is not None else "no_contact_detected_fallback_first_frame"
    pre_frame = max(0, contact_frame - 1)

    lower_frame = int(np.nanargmin(np.asarray(lower_sdf, dtype=np.float64)))
    hand_frame = int(np.nanargmin(np.asarray(hand_sdf, dtype=np.float64)))
    xyz = np.asarray(object_xyz, dtype=np.float64)
    dt = np.diff(times)
    displacement = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    valid = np.isfinite(dt) & (dt > 0)
    speed = np.full_like(displacement, np.nan, dtype=np.float64)
    speed[valid] = displacement[valid] / dt[valid]
    speed_frame = int(np.nanargmax(speed)) + 1 if np.any(np.isfinite(speed)) else 0

    events = [
        {
            "event": "pre_contact", "frame_index": pre_frame,
            "metric_name": "", "metric_value": "", "metric_unit": "",
            "selection_basis": pre_basis,
        },
        {
            "event": "max_lower_body_penetration", "frame_index": lower_frame,
            "metric_name": "lower_body_object_min_sdf", "metric_value": lower_sdf[lower_frame], "metric_unit": "m",
            "selection_basis": "minimum_mesh_aware_lower_body_object_sdf_over_flattened_rollout",
        },
        {
            "event": "max_hand_penetration", "frame_index": hand_frame,
            "metric_name": "hand_object_min_sdf", "metric_value": hand_sdf[hand_frame], "metric_unit": "m",
            "selection_basis": "minimum_mesh_aware_hand_object_sdf_over_flattened_rollout",
        },
        {
            "event": "max_object_speed", "frame_index": speed_frame,
            "metric_name": "object_speed", "metric_value": speed[speed_frame - 1] if speed_frame > 0 else 0.0, "metric_unit": "m/s",
            "selection_basis": "maximum_world_object_translation_speed_using_npz_time",
        },
        {
            "event": "final", "frame_index": len(qpos) - 1,
            "metric_name": "", "metric_value": "", "metric_unit": "",
            "selection_basis": "last_flattened_rollout_frame",
        },
    ]
    for event in events:
        event["simulation_time_s"] = float(times[event["frame_index"]])
    return events, len(qpos)


def format_label(row: dict[str, Any]) -> str:
    label = f"{row['event']}  frame={row['frame_index']}  sim_t={float(row['simulation_time_s']):.3f}s"
    if row["metric_name"]:
        label += f"  {row['metric_name']}={float(row['metric_value']):.5f}{row['metric_unit']}"
    return label


def build_sheet(case_id: str, rows: list[dict[str, Any]], sheet_path: Path) -> None:
    by_event = {row["event"]: row for row in rows}
    ordered = [by_event[event] for event in EVENT_ORDER]
    tile_width, tile_height, label_height = 960, 270, 42
    canvas = Image.new("RGB", (tile_width * 2, (tile_height + label_height) * 3), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=18)
    for index, row in enumerate(ordered):
        image = Image.open(repo_path(row["image_path"])).convert("RGB")
        image.thumbnail((tile_width, tile_height), Image.Resampling.LANCZOS)
        x = (index % 2) * tile_width
        y = (index // 2) * (tile_height + label_height)
        cell = Image.new("RGB", (tile_width, tile_height), "black")
        cell.paste(image, ((tile_width - image.width) // 2, (tile_height - image.height) // 2))
        canvas.paste(cell, (x, y))
        draw.text((x + 8, y + tile_height + 8), format_label(row), fill="black", font=font)
    draw.text((tile_width + 8, 2 * (tile_height + label_height) + tile_height + 8), case_id, fill="black", font=font)
    sheet_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = sheet_path.with_suffix(".tmp.jpg")
    canvas.save(tmp, format="JPEG", quality=92)
    tmp.replace(sheet_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frame-script", type=Path, default=DEFAULT_FRAME_SCRIPT)
    parser.add_argument("--cases", nargs="*", default=[])
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()

    manifest = repo_path(args.manifest)
    output_root = repo_path(args.output_root)
    frame_script = args.frame_script.expanduser().resolve()
    if not frame_script.is_file():
        raise SystemExit(f"missing video-frames entrypoint: {frame_script}")
    rows = read_tsv(manifest)
    selectors = set(args.cases)
    selected = [row for row in rows if not selectors or row["case_id"] in selectors]
    if selectors - {row["case_id"] for row in selected}:
        raise SystemExit(f"unknown cases: {sorted(selectors - {row['case_id'] for row in selected})}")
    if args.require_all and len(selected) != 28:
        raise SystemExit(f"strict keyframe evidence requires 28 cases, got {len(selected)}")

    evidence_rows: list[dict[str, Any]] = []
    for index, row in enumerate(selected, 1):
        case_id = row["case_id"]
        paired = PAIR_ROOT / f"{case_id}_E168_vs_E170_PRG.mp4"
        required = [repo_path(row[key]) for key in ("outdir_npz", "scene_act", "contact_mask")]
        missing = [str(path) for path in [paired, *required] if not path.is_file()]
        if missing:
            raise SystemExit(f"{case_id}: missing keyframe inputs: {missing}")
        info = video_info(paired)
        events, qpos_frames = event_diagnostics(row)
        if info["frames"] != qpos_frames:
            raise SystemExit(f"{case_id}: paired video/qpos frame mismatch: {info['frames']} != {qpos_frames}")
        sheet_path = output_root / f"{case_id}_keyframes.jpg"
        case_rows: list[dict[str, Any]] = []
        for event in events:
            frame_index = int(event["frame_index"])
            image_path = output_root / "frames" / case_id / f"{event['event']}_f{frame_index:04d}.png"
            subprocess.run(
                ["bash", str(frame_script), str(paired), "--index", str(frame_index), "--out", str(image_path)],
                check=True,
                stdout=subprocess.DEVNULL,
            )
            item = {
                "case_id": case_id,
                "variant": row["variant"],
                **event,
                "paired_video": rel(paired),
                "image_path": rel(image_path),
                "sheet_path": rel(sheet_path),
                "video_frame_count": info["frames"],
                "qpos_flattened_frame_count": qpos_frames,
                "video_fps": info["fps"],
                "mapping_status": "exact_flattened_qpos_to_video_frame",
            }
            case_rows.append(item)
            evidence_rows.append(item)
        build_sheet(case_id, case_rows, sheet_path)
        print(f"[{index}/{len(selected)}] {case_id}: 5 frames -> {rel(sheet_path)}", flush=True)

    manifest_path = output_root / "keyframe_manifest.tsv"
    write_tsv(manifest_path, evidence_rows)
    summary = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_manifest": rel(manifest),
        "selected_cases": len(selected),
        "evidence_rows": len(evidence_rows),
        "events_per_case": list(EVENT_ORDER),
        "frame_extractor": str(frame_script),
        "status": "pass" if len(evidence_rows) == 5 * len(selected) else "fail",
    }
    (output_root / "keyframe_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.require_all and (len(evidence_rows) != 140 or summary["status"] != "pass"):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
