#!/usr/bin/env python3
"""E151 preflight checks for mesh SDF reward and external targets."""

from __future__ import annotations

import csv
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mujoco
import numpy as np
import torch

from spider.simulators.mjwp import _geom_box_sdf_min

REPO = Path(__file__).resolve().parents[4]
RESULT_ROOT = REPO / "workspace/core4d/results/E151/route_b_hand_surface_contact"
VARIANTS_TSV = REPO / "workspace/core4d/scripts/E151/variants.tsv"
E147_EVAL = REPO / "workspace/core4d/scripts/eval/eval_E147_rubber_hand_collision.py"
OUT_DIR = RESULT_ROOT / "preflight"


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(REPO.resolve()))
    except Exception:
        return str(path)


def repo_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO / p


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def load_e147_eval():
    spec = importlib.util.spec_from_file_location("eval_E147_for_E151_check", E147_EVAL)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {E147_EVAL}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_cases() -> list[tuple[str, Path, Path]]:
    if not VARIANTS_TSV.is_file():
        raise SystemExit(f"missing variants; run build_route_b_manifest.py first: {VARIANTS_TSV}")
    rows = [row for row in read_tsv(VARIANTS_TSV) if row["method"] == "baseline"]
    return [
        (row["short_case_id"], repo_path(row["rubber_scene_act"]), repo_path(row["outdir_npz"]))
        for row in rows
    ]


def load_scene_act_qpos(path: Path) -> np.ndarray:
    qpos = np.load(path, allow_pickle=True)["qpos"]
    if qpos.ndim == 3:
        return qpos[:, 0, :]
    return qpos


def tensors_from_data(data: mujoco.MjData) -> tuple[torch.Tensor, torch.Tensor]:
    geom_xpos = torch.tensor(data.geom_xpos[None, :, :], dtype=torch.float64)
    geom_xmat = torch.tensor(data.geom_xmat.reshape(-1, 3, 3)[None, :, :, :], dtype=torch.float64)
    return geom_xpos, geom_xmat


def legacy_primitive_sdf(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_ids: list[int],
    object_gid: int,
) -> float:
    obj_pos = data.geom_xpos[object_gid]
    obj_mat = data.geom_xmat[object_gid].reshape(3, 3)
    half = model.geom_size[object_gid, :3]
    vals: list[float] = []
    for gid in geom_ids:
        center = data.geom_xpos[gid]
        mat = data.geom_xmat[gid].reshape(3, 3)
        axis = mat[:, 2]
        radius = float(model.geom_size[gid, 0])
        half_len = (
            float(model.geom_size[gid, 1])
            if int(model.geom_type[gid]) == int(mujoco.mjtGeom.mjGEOM_CAPSULE)
            else 0.0
        )
        for s in (-half_len, 0.0, half_len):
            point = center + axis * s
            local = obj_mat.T @ (point - obj_pos)
            q = np.abs(local) - half
            outside = np.linalg.norm(np.maximum(q, 0.0))
            inside = min(float(q.max()), 0.0)
            vals.append(outside + inside - radius)
    return float(min(vals))


def mesh_sdf_rows(eval_mod: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    max_abs = 0.0
    for short, scene, traj in check_cases():
        model = mujoco.MjModel.from_xml_path(str(scene))
        data = mujoco.MjData(model)
        qpos = load_scene_act_qpos(traj)
        frames = sorted({0, int(qpos.shape[0] // 2), int(qpos.shape[0] - 1)})
        object_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_collision")
        hand_gids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "lh"),
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "rh"),
        ]
        config = SimpleNamespace(
            device="cpu",
            hand_approach_obj_half_extents=[float(x) for x in model.geom_size[object_gid, :3]],
        )
        env = SimpleNamespace(model_cpu=model)
        for frame in frames:
            data.qpos[:] = qpos[frame]
            mujoco.mj_forward(model, data)
            geom_xpos, geom_xmat = tensors_from_data(data)
            runtime = float(
                _geom_box_sdf_min(
                    config,
                    env,
                    hand_gids,
                    object_gid,
                    geom_xpos=geom_xpos,
                    geom_xmat=geom_xmat,
                )[0].item()
            )
            reference = min(eval_mod.geom_object_sdf(model, data, gid, [object_gid]) for gid in hand_gids)
            abs_diff = abs(runtime - reference)
            max_abs = max(max_abs, abs_diff)
            rows.append(
                {
                    "case": short,
                    "frame": frame,
                    "runtime_mesh_sdf_m": f"{runtime:.10f}",
                    "eval_E147_mesh_sdf_m": f"{reference:.10f}",
                    "abs_diff_m": f"{abs_diff:.10f}",
                    "pass_1mm": str(abs_diff <= 0.001),
                    "scene": rel(scene),
                }
            )
    if max_abs > 0.001:
        raise SystemExit(f"mesh SDF check failed: max_abs_diff={max_abs:.6f}m")
    return rows


def primitive_regression_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    short, scene, traj = check_cases()[0]
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    qpos = load_scene_act_qpos(traj)
    object_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_collision")
    primitive_gids = []
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        gtype = int(model.geom_type[gid])
        if name in {"floor", "object_collision", "object_visual", "lh", "rh"}:
            continue
        if gtype in {
            int(mujoco.mjtGeom.mjGEOM_SPHERE),
            int(mujoco.mjtGeom.mjGEOM_CAPSULE),
            int(mujoco.mjtGeom.mjGEOM_BOX),
        }:
            primitive_gids.append(gid)
        if len(primitive_gids) >= 8:
            break
    if not primitive_gids:
        raise SystemExit("primitive regression found no primitive geoms")
    config = SimpleNamespace(
        device="cpu",
        hand_approach_obj_half_extents=[float(x) for x in model.geom_size[object_gid, :3]],
    )
    env = SimpleNamespace(model_cpu=model)
    for frame in (0, int(qpos.shape[0] // 2)):
        data.qpos[:] = qpos[frame]
        mujoco.mj_forward(model, data)
        geom_xpos, geom_xmat = tensors_from_data(data)
        runtime = float(
            _geom_box_sdf_min(
                config,
                env,
                primitive_gids,
                object_gid,
                geom_xpos=geom_xpos,
                geom_xmat=geom_xmat,
            )[0].item()
        )
        legacy = legacy_primitive_sdf(model, data, primitive_gids, object_gid)
        abs_diff = abs(runtime - legacy)
        rows.append(
            {
                "case": short,
                "frame": frame,
                "geom_count": len(primitive_gids),
                "runtime_primitive_sdf_m": f"{runtime:.10f}",
                "legacy_primitive_sdf_m": f"{legacy:.10f}",
                "abs_diff_m": f"{abs_diff:.12f}",
                "pass_1e-9": str(abs_diff <= 1e-9),
            }
        )
        if abs_diff > 1e-9:
            raise SystemExit(f"primitive regression failed: abs_diff={abs_diff:.12f}")
    return rows


def target_rows() -> list[dict[str, Any]]:
    if not VARIANTS_TSV.is_file():
        raise SystemExit(f"missing variants; run build_route_b_manifest.py first: {VARIANTS_TSV}")
    rows = []
    for row in read_tsv(VARIANTS_TSV):
        if row["method"] not in {"b2_sup", "b2_tip"}:
            continue
        target = repo_path(row["target_npz"])
        data = np.load(target, allow_pickle=True)
        arr = data["spider_contact_target_object_local"]
        qpos_len = int(np.load(repo_path(row["trajectory"]), allow_pickle=True)["qpos"].shape[0])
        ok = bool(arr.ndim == 3 and arr.shape == (qpos_len, 2, 3) and np.isfinite(arr).all())
        rows.append(
            {
                "variant": row["variant"],
                "case": row["short_case_id"],
                "method": row["method"],
                "target_npz": row["target_npz"],
                "shape": "x".join(map(str, arr.shape)),
                "qpos_len": qpos_len,
                "finite": str(bool(np.isfinite(arr).all())),
                "pass": str(ok),
                "box021_clean_max_diff_m": row["target_clean_max_diff_m"],
                "box021_selected": row["target_clean_selected"],
            }
        )
        if not ok:
            raise SystemExit(f"target validation failed for {row['variant']}: {target}")
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    eval_mod = load_e147_eval()
    mesh_rows = mesh_sdf_rows(eval_mod)
    primitive_rows = primitive_regression_rows()
    tgt_rows = target_rows()
    write_tsv(
        OUT_DIR / "mesh_sdf_parity.tsv",
        mesh_rows,
        ["case", "frame", "runtime_mesh_sdf_m", "eval_E147_mesh_sdf_m", "abs_diff_m", "pass_1mm", "scene"],
    )
    write_tsv(
        OUT_DIR / "primitive_sdf_regression.tsv",
        primitive_rows,
        ["case", "frame", "geom_count", "runtime_primitive_sdf_m", "legacy_primitive_sdf_m", "abs_diff_m", "pass_1e-9"],
    )
    write_tsv(
        OUT_DIR / "target_validation.tsv",
        tgt_rows,
        ["variant", "case", "method", "target_npz", "shape", "qpos_len", "finite", "pass", "box021_clean_max_diff_m", "box021_selected"],
    )
    summary = {
        "mesh_rows": len(mesh_rows),
        "primitive_rows": len(primitive_rows),
        "target_rows": len(tgt_rows),
        "mesh_max_abs_diff_m": max(float(r["abs_diff_m"]) for r in mesh_rows),
        "primitive_max_abs_diff_m": max(float(r["abs_diff_m"]) for r in primitive_rows),
        "status": "pass",
    }
    (OUT_DIR / "preflight_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
