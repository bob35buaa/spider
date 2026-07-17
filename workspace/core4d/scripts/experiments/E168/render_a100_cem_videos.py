#!/usr/bin/env python3
"""Render compute-only E168 A100 CEM results as local MuJoCo replay videos."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")


REPO = Path(__file__).resolve().parents[5]
DEFAULT_MANIFEST = (
    REPO
    / "workspace/core4d/results/E168/s6_downstream/cem/manifests/cem_production_manifest.tsv"
)


def repo_path(raw: str | Path) -> Path:
    """Rebase repo-owned paths saved on a remote worker to this checkout."""
    path = Path(raw)
    if path.exists():
        return path.resolve()
    text = str(raw)
    for marker in ("example_datasets/", "workspace/", "logs/"):
        if marker in text:
            return REPO / (marker + text.split(marker, 1)[1])
    return path if path.is_absolute() else REPO / path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def load_render_config(row: dict[str, str], config_yaml: Path) -> Any:
    from spider.config import Config, filter_config_fields, load_config_yaml

    cfg_dict = load_config_yaml(str(config_yaml))
    config = Config(**filter_config_fields(cfg_dict))
    config.device = "cpu"
    config.model_path = str(repo_path(row["scene_act"]))
    config.data_path = str(repo_path(row["trajectory"]))
    config.save_video = True
    config.video_camera = "auto"
    return config


def converted_reference_qpos(config: Any) -> Any:
    import mujoco
    import numpy as np
    from scipy.spatial.transform import Rotation as R

    from spider.io import load_data

    qpos_ref, _qvel_ref, _ctrl_ref, _contact, _contact_pos_ref = load_data(
        config, config.data_path
    )
    qpos = qpos_ref.detach().cpu().numpy()
    if qpos.shape[1] == config.nq:
        return qpos
    if qpos.shape[1] <= config.nq or config.nq < 6:
        raise ValueError(
            f"cannot convert reference qpos shape {qpos.shape} to nq={config.nq}"
        )

    model = mujoco.MjModel.from_xml_path(config.model_path)
    nq_model = int(config.nq)
    nq_robot = nq_model - 6
    obj_pos_world = qpos[:, nq_robot : nq_robot + 3]
    obj_quat_wxyz = qpos[:, nq_robot + 3 : nq_robot + 7]

    obj_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    if obj_body_id < 0:
        raise ValueError(f"scene has no body named object: {config.model_path}")
    body_pos = model.body_pos[obj_body_id]
    body_quat_wxyz = model.body_quat[obj_body_id]
    body_rot = R.from_quat(
        [
            body_quat_wxyz[1],
            body_quat_wxyz[2],
            body_quat_wxyz[3],
            body_quat_wxyz[0],
        ]
    )
    obj_slide_pos = body_rot.inv().apply(obj_pos_world - body_pos[np.newaxis, :])

    meta_path = Path(config.model_path).with_name("scene_act_meta.json")
    euler_convention = str(getattr(config, "euler_convention", "XYZ") or "XYZ")
    if meta_path.is_file():
        import json

        euler_convention = json.loads(meta_path.read_text(encoding="utf-8")).get(
            "euler_convention", euler_convention
        )
    obj_quat_xyzw = np.column_stack(
        [
            obj_quat_wxyz[:, 1],
            obj_quat_wxyz[:, 2],
            obj_quat_wxyz[:, 3],
            obj_quat_wxyz[:, 0],
        ]
    )
    obj_euler = (body_rot.inv() * R.from_quat(obj_quat_xyzw)).as_euler(
        euler_convention
    )

    converted = np.zeros((qpos.shape[0], nq_model), dtype=np.float64)
    converted[:, :nq_robot] = qpos[:, :nq_robot]
    converted[:, nq_robot : nq_robot + 3] = obj_slide_pos
    converted[:, nq_robot + 3 : nq_robot + 6] = obj_euler
    return converted


def rollout_qpos(npz_path: Path) -> Any:
    import numpy as np

    with np.load(npz_path, allow_pickle=True) as data:
        qpos = np.asarray(data["qpos"], dtype=np.float64)
    if qpos.ndim == 3:
        # Saved shape is (control ticks, sim steps per tick, nq).
        return qpos.reshape(-1, qpos.shape[-1])
    if qpos.ndim == 2:
        return qpos
    raise ValueError(f"unsupported qpos shape {qpos.shape} in {npz_path}")


def render_row(
    row: dict[str, str],
    *,
    out_path: Path,
    max_frames: int,
) -> tuple[int, int]:
    import imageio.v2 as imageio
    import mujoco

    from spider.viewers import render_image, setup_renderer

    rollout_path = repo_path(row["outdir_npz"])
    config_path = repo_path(row["config_act"])
    config = load_render_config(row, config_path)
    sim_qpos = rollout_qpos(rollout_path)
    ref_qpos = converted_reference_qpos(config)
    model = mujoco.MjModel.from_xml_path(config.model_path)

    if sim_qpos.shape[1] != model.nq:
        raise ValueError(
            f"{row['variant']}: rollout nq={sim_qpos.shape[1]} != model nq={model.nq}"
        )
    if ref_qpos.shape[1] != model.nq:
        raise ValueError(
            f"{row['variant']}: reference nq={ref_qpos.shape[1]} != model nq={model.nq}"
        )

    stride = max(1, int(round(float(config.render_dt) / float(config.sim_dt))))
    frame_ids = list(range(0, min(len(sim_qpos), len(ref_qpos)), stride))
    if max_frames > 0:
        frame_ids = frame_ids[:max_frames]
    if not frame_ids:
        raise ValueError(f"{row['variant']}: no replay frames")

    data = mujoco.MjData(model)
    data_ref = mujoco.MjData(model)
    renderer = setup_renderer(config, model)
    if renderer is None:
        raise RuntimeError("MuJoCo renderer was not created")

    fps = max(1, int(round(1.0 / float(config.render_dt))))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{out_path.stem}.", suffix=".tmp.mp4", dir=out_path.parent
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    writer = imageio.get_writer(tmp_path, fps=fps, codec="libx264", quality=8)
    try:
        for frame_id in frame_ids:
            data.qpos[:] = sim_qpos[frame_id]
            data.qvel[:] = 0.0
            data_ref.qpos[:] = ref_qpos[frame_id]
            data_ref.qvel[:] = 0.0
            writer.append_data(render_image(config, renderer, model, data, data_ref))
        writer.close()
        renderer.close()
        tmp_path.replace(out_path)
    except Exception:
        writer.close()
        renderer.close()
        tmp_path.unlink(missing_ok=True)
        raise
    return len(frame_ids), fps


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render E168 compute-only CEM results from the production manifest"
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--pool",
        choices=("a100", "a6000", "all"),
        default="a100",
        help="Manifest worker pool to render (default: a100)",
    )
    parser.add_argument(
        "--cases",
        nargs="*",
        default=[],
        help="Optional exact case_id or variant selectors",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override manifest video destinations, useful for smoke tests",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="Fail when any selected row is not yet ready to render",
    )
    args = parser.parse_args()

    manifest = repo_path(args.manifest)
    rows = read_rows(manifest)
    selectors = set(args.cases)
    selected = [
        row
        for row in rows
        if (args.pool == "all" or row.get("preferred_pool") == args.pool)
        and (
            not selectors
            or row.get("case_id") in selectors
            or row.get("variant") in selectors
        )
    ]
    if args.limit > 0:
        selected = selected[: args.limit]
    if not selected:
        raise SystemExit("no manifest rows matched the requested pool/cases")

    counts: Counter[str] = Counter()
    failures: list[str] = []
    for row in selected:
        rollout_path = repo_path(row["outdir_npz"])
        config_path = repo_path(row["config_act"])
        scene_path = repo_path(row["scene_act"])
        trajectory_path = repo_path(row["trajectory"])
        missing = [
            name
            for name, path in (
                ("outdir_npz", rollout_path),
                ("config_act", config_path),
                ("scene_act", scene_path),
                ("trajectory", trajectory_path),
            )
            if not path.is_file()
        ]
        if missing:
            counts["not_ready"] += 1
            print(f"[not-ready] {row['variant']}: missing={','.join(missing)}")
            continue

        if args.output_dir is None:
            out_path = repo_path(row["video"])
        else:
            out_path = repo_path(args.output_dir) / Path(row["video"]).name
        if out_path.is_file() and not args.overwrite:
            counts["existing"] += 1
            print(f"[existing] {row['variant']} -> {out_path}")
            continue
        if args.dry_run:
            counts["ready"] += 1
            print(f"[ready] {row['variant']} -> {out_path}")
            continue

        print(f"[render] {row['variant']} -> {out_path}", flush=True)
        try:
            frame_count, fps = render_row(
                row, out_path=out_path, max_frames=args.max_frames
            )
        except Exception as exc:
            counts["failed"] += 1
            failures.append(f"{row['variant']}: {exc}")
            print(f"[failed] {failures[-1]}")
            continue
        counts["rendered"] += 1
        print(f"[ok] {row['variant']}: frames={frame_count} fps={fps}")

    print(
        "summary "
        f"selected={len(selected)} rendered={counts['rendered']} "
        f"ready={counts['ready']} existing={counts['existing']} "
        f"not_ready={counts['not_ready']} failed={counts['failed']}"
    )
    if failures:
        return 1
    if args.require_all and counts["not_ready"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
