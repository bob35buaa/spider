#!/usr/bin/env python3
"""Re-render E094 CEM rollouts with the full-body auto MuJoCo camera."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from spider.config import Config, filter_config_fields, load_config_yaml
from spider.io import load_data
from spider.viewers import render_image, setup_renderer


REPO = Path(__file__).resolve().parents[4]
DEFAULT_VARIANTS = REPO / "workspace/core4d/scripts/E094/variants.tsv"
DEFAULT_RESULTS = REPO / "workspace/core4d/results/E094/cem/full"

FIELDS = [
    "route",
    "case_id",
    "variant",
    "source_task",
    "derived_task",
    "person_idx",
    "split",
    "object",
    "role",
    "note",
    "target_case_id",
    "target_npz",
]


def repo_path(raw: str | Path) -> Path:
    path = Path(raw)
    if path.is_absolute() and path.exists():
        return path
    if not path.is_absolute() and path.exists():
        return path
    text = str(raw)
    for marker in ("example_datasets/", "workspace/"):
        if marker in text:
            candidate = REPO / (marker + text.split(marker, 1)[1])
            if candidate.exists():
                return candidate
    return path if path.is_absolute() else REPO / path


def read_variants(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(
            (line for line in f if line.strip() and not line.startswith("#")),
            delimiter="\t",
            fieldnames=FIELDS,
        )
        for row in reader:
            if row["route"] == "cem_hbproj":
                rows[row["variant"]] = row
    return rows


def load_config(config_yaml: Path) -> Config:
    cfg_dict = load_config_yaml(str(config_yaml))
    config = Config(**filter_config_fields(cfg_dict))
    config.device = "cpu"
    config.model_path = str(repo_path(config.model_path))
    config.data_path = str(repo_path(config.data_path))
    config.save_video = True
    config.video_camera = "auto"
    return config


def converted_reference_qpos(config: Config) -> np.ndarray:
    qpos_ref, _qvel_ref, _ctrl_ref, _contact, _contact_pos = load_data(
        config, config.data_path
    )
    if (config.object_pd_override or config.contact_guidance) and qpos_ref.shape[1] > config.nq:
        nq_model = int(config.nq)
        nq_robot = nq_model - 6
        obj_pos_world = qpos_ref[:, nq_robot : nq_robot + 3].detach().cpu().numpy()
        obj_quat_wxyz = qpos_ref[:, nq_robot + 3 : nq_robot + 7].detach().cpu().numpy()

        model = mujoco.MjModel.from_xml_path(config.model_path)
        obj_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
        body_pos = model.body_pos[obj_body_id]
        body_quat_wxyz = model.body_quat[obj_body_id]
        body_quat_xyzw = [
            body_quat_wxyz[1],
            body_quat_wxyz[2],
            body_quat_wxyz[3],
            body_quat_wxyz[0],
        ]
        body_rot = R.from_quat(body_quat_xyzw)
        obj_slide_pos = body_rot.inv().apply(obj_pos_world - body_pos[np.newaxis, :])

        meta_path = Path(config.model_path).with_name("scene_act_meta.json")
        euler_conv = str(getattr(config, "euler_convention", "XYZ") or "XYZ")
        if meta_path.is_file():
            import json

            euler_conv = json.loads(meta_path.read_text(encoding="utf-8")).get(
                "euler_convention", euler_conv
            )

        obj_quat_xyzw = np.column_stack(
            [
                obj_quat_wxyz[:, 1],
                obj_quat_wxyz[:, 2],
                obj_quat_wxyz[:, 3],
                obj_quat_wxyz[:, 0],
            ]
        )
        world_rot = R.from_quat(obj_quat_xyzw)
        obj_euler = (body_rot.inv() * world_rot).as_euler(euler_conv)

        qpos_new = np.zeros((qpos_ref.shape[0], nq_model), dtype=np.float32)
        qpos_new[:, :nq_robot] = qpos_ref[:, :nq_robot].detach().cpu().numpy()
        qpos_new[:, nq_robot : nq_robot + 3] = obj_slide_pos.astype(np.float32)
        qpos_new[:, nq_robot + 3 : nq_robot + 6] = obj_euler.astype(np.float32)
        return qpos_new
    return qpos_ref.detach().cpu().numpy()


def rollout_qpos(npz_path: Path) -> np.ndarray:
    data = np.load(npz_path, allow_pickle=True)
    qpos = data["qpos"]
    if qpos.ndim == 3:
        return qpos.reshape(-1, qpos.shape[-1])
    if qpos.ndim == 2:
        return qpos
    raise ValueError(f"Unsupported qpos shape {qpos.shape} in {npz_path}")


def render_variant(
    variant: str,
    results_dir: Path,
    stage: str,
    max_frames: int,
    overwrite: bool,
) -> Path | None:
    out_dir = results_dir / f"{variant}_outdir_{stage}"
    npz_path = out_dir / "trajectory_mjwp_act.npz"
    config_yaml = out_dir / "config_act.yaml"
    if not npz_path.is_file() or not config_yaml.is_file():
        print(f"[SKIP] {variant}: missing {npz_path.name} or config_act.yaml")
        return None

    out_video = results_dir / f"{variant}_{stage}_autocam.mp4"
    if out_video.exists() and not overwrite:
        print(f"[SKIP] {variant}: {out_video} exists")
        return out_video

    config = load_config(config_yaml)
    sim_q = rollout_qpos(npz_path)
    ref_q = converted_reference_qpos(config)
    model = mujoco.MjModel.from_xml_path(config.model_path)
    if sim_q.shape[1] != model.nq:
        raise ValueError(f"{variant}: rollout nq {sim_q.shape[1]} != model.nq {model.nq}")
    if ref_q.shape[1] != model.nq:
        raise ValueError(f"{variant}: ref nq {ref_q.shape[1]} != model.nq {model.nq}")

    stride = max(1, int(round(float(config.render_dt) / float(config.sim_dt))))
    frame_ids = list(range(0, min(len(sim_q), len(ref_q)), stride))
    if max_frames > 0:
        frame_ids = frame_ids[:max_frames]

    data = mujoco.MjData(model)
    data_ref = mujoco.MjData(model)
    renderer = setup_renderer(config, model)
    out_video.parent.mkdir(parents=True, exist_ok=True)
    key_dir = results_dir / "keyframes_autocam" / variant
    key_dir.mkdir(parents=True, exist_ok=True)
    key_pick = set(np.linspace(0, max(len(frame_ids) - 1, 0), num=min(10, len(frame_ids)), dtype=int).tolist())

    writer = imageio.get_writer(out_video, fps=max(1, int(round(1.0 / float(config.render_dt)))), codec="libx264", quality=7)
    try:
        for out_i, q_i in enumerate(frame_ids):
            data.qpos[:] = sim_q[q_i]
            data_ref.qpos[:] = ref_q[min(q_i, len(ref_q) - 1)]
            image = render_image(config, renderer, model, data, data_ref)
            writer.append_data(image)
            if out_i in key_pick:
                imageio.imwrite(key_dir / f"f{q_i:04d}.jpg", image)
    finally:
        writer.close()
        renderer.close()
    print(f"[OK] {variant}: {out_video}")
    return out_video


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="full")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--variants-file", type=Path, default=DEFAULT_VARIANTS)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("variants", nargs="*")
    args = parser.parse_args()

    variants = read_variants(repo_path(args.variants_file))
    selected = args.variants or list(variants)
    results_dir = repo_path(args.results)
    for variant in selected:
        if variant not in variants:
            print(f"[SKIP] unknown variant {variant}")
            continue
        render_variant(variant, results_dir, args.stage, args.max_frames, args.overwrite)


if __name__ == "__main__":
    main()
