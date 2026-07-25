#!/usr/bin/env python3
"""Render MMHOI 30skip RGB frames at a selected rate without interpolation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

LOGGER = logging.getLogger("render_sparse_rgb_videos")
REQUIRED_CASE_COLUMNS = ("case_id", "scenario_code", "selection_reason")
ALLOWED_SCENARIOS = {"C_2", "C_8"}
SOURCE_CAPTURE_FPS = 30
SOURCE_ANNOTATION_STRIDE = 30
DEFAULT_PLAYBACK_FPS = 1
DEFAULT_CRF = 20


@dataclass(frozen=True)
class CaseSpec:
    """One selected scenario capture."""

    case_id: str
    scenario_code: str
    selection_reason: str


@dataclass(frozen=True)
class RgbFrame:
    """One source RGB image with its original source-frame id."""

    frame_id: int
    frame_folder: str
    image_path: Path


def _validate_case_id(case_id: str) -> None:
    parts = case_id.split("/")
    if (
        len(parts) != 2
        or any(not part or part in {".", ".."} for part in parts)
        or any("\\" in part for part in parts)
    ):
        raise ValueError(
            f"case_id must be sequence/scenario with no traversal: {case_id!r}"
        )


def load_case_specs(path: Path) -> tuple[CaseSpec, ...]:
    """Load and validate a selected-case TSV."""
    if not path.is_file():
        raise FileNotFoundError(f"case manifest not found: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"case manifest has no header: {path}")
        missing = set(REQUIRED_CASE_COLUMNS) - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"case manifest missing columns {sorted(missing)}: {path}"
            )
        rows = tuple(reader)
    if not rows:
        raise ValueError(f"case manifest is empty: {path}")

    specs: list[CaseSpec] = []
    seen: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        case_id = row["case_id"].strip()
        scenario_code = row["scenario_code"].strip()
        selection_reason = row["selection_reason"].strip()
        _validate_case_id(case_id)
        if scenario_code not in ALLOWED_SCENARIOS:
            raise ValueError(
                f"row {row_number}: unsupported scenario {scenario_code!r}"
            )
        if not selection_reason:
            raise ValueError(f"row {row_number}: empty selection_reason")
        if case_id in seen:
            raise ValueError(f"row {row_number}: duplicate case_id {case_id!r}")
        seen.add(case_id)
        specs.append(
            CaseSpec(
                case_id=case_id,
                scenario_code=scenario_code,
                selection_reason=selection_reason,
            )
        )
    return tuple(specs)


def safe_case_slug(case_id: str) -> str:
    """Return a stable filesystem-safe case slug."""
    _validate_case_id(case_id)
    safe_parts = []
    for part in case_id.split("/"):
        cleaned = re.sub(r"[^A-Za-z0-9]+", "_", part).strip("_")
        safe_parts.append(cleaned)
    return "__".join(safe_parts)


def discover_rgb_frames(
    case_dir: Path,
    camera_id: int,
) -> tuple[RgbFrame, ...]:
    """Discover all numeric frame folders and require one RGB per folder."""
    if not case_dir.is_dir():
        raise FileNotFoundError(f"case directory not found: {case_dir}")
    numeric_dirs = sorted(
        (
            child
            for child in case_dir.iterdir()
            if child.is_dir() and child.name.isdigit()
        ),
        key=lambda path: int(path.name),
    )
    frames: list[RgbFrame] = []
    missing: list[Path] = []
    for frame_dir in numeric_dirs:
        image_path = frame_dir / f"{camera_id}_{frame_dir.name}.jpg"
        if not image_path.is_file():
            missing.append(image_path)
            continue
        frames.append(
            RgbFrame(
                frame_id=int(frame_dir.name),
                frame_folder=frame_dir.name,
                image_path=image_path.resolve(),
            )
        )
    if not frames:
        raise ValueError(f"no camera-{camera_id} RGB frames in {case_dir}")
    if missing:
        raise ValueError(
            f"camera-{camera_id} RGB missing for {len(missing)} frame folders; "
            f"first missing path: {missing[0]}"
        )
    return tuple(frames)


def frame_gap_distribution(frame_ids: Sequence[int]) -> dict[int, int]:
    """Count adjacent source-frame-id gaps."""
    gaps = Counter(
        current - previous
        for previous, current in zip(
            frame_ids,
            frame_ids[1:],
            strict=False,
        )
    )
    return dict(sorted(gaps.items()))


def _require_binary(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise FileNotFoundError(f"required binary not found on PATH: {name}")
    return path


def _link_ordered_frames(
    frames: Sequence[RgbFrame],
    staging_dir: Path,
) -> None:
    for index, frame in enumerate(frames):
        link_path = staging_dir / f"{index:06d}.jpg"
        link_path.symlink_to(frame.image_path)


def _run_ffmpeg(
    frames: Sequence[RgbFrame],
    output_path: Path,
    playback_fps: int,
    crf: int,
    overwrite: bool,
) -> list[str]:
    ffmpeg = _require_binary("ffmpeg")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"output exists; use --overwrite to replace: {output_path}"
        )
    with tempfile.TemporaryDirectory(
        prefix=".rgb_frames_",
        dir=output_path.parent,
    ) as temp_dir:
        staging_dir = Path(temp_dir)
        _link_ordered_frames(frames, staging_dir)
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y" if overwrite else "-n",
            "-framerate",
            str(playback_fps),
            "-start_number",
            "0",
            "-i",
            str(staging_dir / "%06d.jpg"),
            "-frames:v",
            str(len(frames)),
            "-vsync",
            "0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        subprocess.run(command, check=True)
    return command


def _probe_video(path: Path) -> dict[str, object]:
    ffprobe = _require_binary("ffprobe")
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        (
            "stream=codec_name,width,height,avg_frame_rate,r_frame_rate,"
            "nb_frames,duration:format=duration"
        ),
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    streams = payload.get("streams", [])
    if len(streams) != 1:
        raise ValueError(f"expected one video stream in {path}, got {len(streams)}")
    return {
        "stream": streams[0],
        "format": payload.get("format", {}),
    }


def _validate_probe(
    probe: Mapping[str, object],
    expected_frames: int,
    playback_fps: int,
    video_path: Path,
) -> None:
    stream = probe["stream"]
    if not isinstance(stream, Mapping):
        raise TypeError(f"invalid ffprobe stream payload for {video_path}")
    if stream.get("avg_frame_rate") != f"{playback_fps}/1":
        raise ValueError(
            f"unexpected avg_frame_rate for {video_path}: "
            f"{stream.get('avg_frame_rate')}"
        )
    if int(str(stream.get("nb_frames", -1))) != expected_frames:
        raise ValueError(
            f"unexpected frame count for {video_path}: "
            f"{stream.get('nb_frames')} != {expected_frames}"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_frame_manifest(
    path: Path,
    case_spec: CaseSpec,
    camera_id: int,
    frames: Sequence[RgbFrame],
) -> None:
    fieldnames = (
        "case_id",
        "scenario_code",
        "camera_id",
        "frame_index",
        "source_frame_id",
        "source_frame_folder",
        "source_rgb_path",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for index, frame in enumerate(frames):
            writer.writerow(
                {
                    "case_id": case_spec.case_id,
                    "scenario_code": case_spec.scenario_code,
                    "camera_id": camera_id,
                    "frame_index": index,
                    "source_frame_id": frame.frame_id,
                    "source_frame_folder": frame.frame_folder,
                    "source_rgb_path": str(frame.image_path),
                }
            )


def _render_case(
    spec: CaseSpec,
    data_root: Path,
    output_dir: Path,
    camera_id: int,
    playback_fps: int,
    crf: int,
    overwrite: bool,
) -> dict[str, object]:
    case_dir = data_root / "sequences" / spec.case_id
    frames = discover_rgb_frames(case_dir, camera_id)
    slug = safe_case_slug(spec.case_id)
    video_path = (
        output_dir
        / "videos"
        / f"{slug}_cam{camera_id}_{playback_fps}fps.mp4"
    )
    frame_manifest = output_dir / "frames" / f"{slug}_cam{camera_id}.tsv"
    frame_manifest.parent.mkdir(parents=True, exist_ok=True)
    command = _run_ffmpeg(
        frames=frames,
        output_path=video_path,
        playback_fps=playback_fps,
        crf=crf,
        overwrite=overwrite,
    )
    probe = _probe_video(video_path)
    _validate_probe(probe, len(frames), playback_fps, video_path)
    _write_frame_manifest(frame_manifest, spec, camera_id, frames)
    stream = probe["stream"]
    file_size = video_path.stat().st_size
    return {
        "case_id": spec.case_id,
        "scenario_code": spec.scenario_code,
        "selection_reason": spec.selection_reason,
        "camera_id": camera_id,
        "source_capture_fps": SOURCE_CAPTURE_FPS,
        "source_annotation_stride": SOURCE_ANNOTATION_STRIDE,
        "playback_fps": playback_fps,
        "interpolation": "none",
        "input_frame_count": len(frames),
        "first_source_frame_id": frames[0].frame_id,
        "last_source_frame_id": frames[-1].frame_id,
        "source_gap_distribution": json.dumps(
            frame_gap_distribution([frame.frame_id for frame in frames]),
            sort_keys=True,
        ),
        "width": stream["width"],
        "height": stream["height"],
        "codec": stream["codec_name"],
        "avg_frame_rate": stream["avg_frame_rate"],
        "duration_seconds": probe["format"]["duration"],
        "output_size_bytes": file_size,
        "output_sha256": _sha256(video_path),
        "video_path": str(video_path.resolve()),
        "frame_manifest": str(frame_manifest.resolve()),
        "ffmpeg_command": command,
    }


def _write_render_manifest(
    path: Path,
    records: Sequence[Mapping[str, object]],
) -> None:
    fieldnames = tuple(
        key for key in records[0] if key != "ffmpeg_command"
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for record in records:
            writer.writerow(
                {key: record[key] for key in fieldnames}
            )


def _write_summary(
    path: Path,
    records: Sequence[Mapping[str, object]],
) -> None:
    lines = [
        "# MMHOI sparse RGB visual QC",
        "",
        (
            "所有视频只使用公开 release 中已有的 30skip camera-0 RGB，"
            f"按 {records[0]['playback_fps']} fps 播放；没有插值、补帧、"
            "光流或运动平滑。"
        ),
        "",
        "| Scenario | Case | Frames | Duration | Video |",
        "|---|---|---:|---:|---|",
    ]
    for record in records:
        video_path = Path(str(record["video_path"]))
        relative_video = video_path.relative_to(path.parent.resolve())
        lines.append(
            f"| `{record['scenario_code']}` | `{record['case_id']}` | "
            f"{record['input_frame_count']} | "
            f"{float(str(record['duration_seconds'])):.1f}s | "
            f"[MP4]({relative_video.as_posix()}) |"
        )
    lines.extend(
        [
            "",
            "时间解释：源 capture clock 为 30 fps，release stride 为 30，"
            "因此每个输入 RGB 对应约 1 秒源时间。",
            (
                ""
                if int(records[0]["playback_fps"]) == 1
                else (
                    f"当前 {records[0]['playback_fps']} fps 版本只是把稀疏帧"
                    f"加速 {records[0]['playback_fps']} 倍播放，不代表恢复了"
                    "帧间轨迹。"
                )
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def render_selected_cases(
    data_root: Path,
    case_manifest: Path,
    output_dir: Path,
    camera_id: int,
    playback_fps: int,
    crf: int,
    overwrite: bool,
) -> tuple[dict[str, object], ...]:
    """Render all selected cases and write persistent evidence."""
    if camera_id < 0:
        raise ValueError("camera_id must be non-negative")
    if playback_fps <= 0:
        raise ValueError("playback_fps must be positive")
    if crf < 0 or crf > 51:
        raise ValueError("crf must be in [0, 51]")
    if not (data_root / "sequences").is_dir():
        raise FileNotFoundError(f"MMHOI sequences directory missing: {data_root}")
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = load_case_specs(case_manifest)
    records = tuple(
        _render_case(
            spec=spec,
            data_root=data_root,
            output_dir=output_dir,
            camera_id=camera_id,
            playback_fps=playback_fps,
            crf=crf,
            overwrite=overwrite,
        )
        for spec in specs
    )
    _write_render_manifest(output_dir / "render_manifest.tsv", records)
    _write_summary(output_dir / "README.md", records)
    run_manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "data_root": str(data_root.resolve()),
        "case_manifest": str(case_manifest.resolve()),
        "output_dir": str(output_dir.resolve()),
        "camera_id": camera_id,
        "source_capture_fps": SOURCE_CAPTURE_FPS,
        "source_annotation_stride": SOURCE_ANNOTATION_STRIDE,
        "playback_fps": playback_fps,
        "interpolation": "none",
        "optical_flow": False,
        "frame_blending": False,
        "case_count": len(records),
        "records": records,
    }
    with (output_dir / "run_manifest.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(run_manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return records


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--case-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument(
        "--playback-fps",
        type=int,
        default=DEFAULT_PLAYBACK_FPS,
    )
    parser.add_argument("--crf", type=int, default=DEFAULT_CRF)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Render selected videos and return a process exit code."""
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(message)s",
    )
    try:
        records = render_selected_cases(
            data_root=args.data_root,
            case_manifest=args.case_manifest,
            output_dir=args.output_dir,
            camera_id=args.camera_id,
            playback_fps=args.playback_fps,
            crf=args.crf,
            overwrite=args.overwrite,
        )
    except (
        FileExistsError,
        FileNotFoundError,
        OSError,
        subprocess.CalledProcessError,
        TypeError,
        ValueError,
    ):
        LOGGER.exception("sparse RGB render failed")
        return 1
    LOGGER.info("rendered %d sparse RGB videos", len(records))
    return 0


if __name__ == "__main__":
    sys.exit(main())
