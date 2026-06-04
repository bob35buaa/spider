#!/usr/bin/env python3
"""Build E122 snap-warmstart carry-prior diagnostic manifest."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from spider.preprocess.hand_snap_ik import find_object_mesh, snap_hands_to_object


REPO = Path(__file__).resolve().parents[4]
TASK_ROOT = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
OVERRIDE_ROOT = REPO / "examples/config/override"
SCRIPTS_ROOT = REPO / "workspace/core4d/scripts/E122"
RESULTS_ROOT = REPO / "workspace/core4d/results/E122"
WARMSTART_ROOT = RESULTS_ROOT / "warmstarts"
PREFLIGHT_ROOT = RESULTS_ROOT / "preflight"
E121_VARIANTS = REPO / "workspace/core4d/scripts/E121/variants.tsv"
VARIANTS_TSV = SCRIPTS_ROOT / "variants.tsv"
PREFLIGHT_TSV = PREFLIGHT_ROOT / "phaseA_preflight.tsv"
SUMMARY_JSON = PREFLIGHT_ROOT / "phaseA_manifest_summary.json"
SUMMARY_MD = PREFLIGHT_ROOT / "phaseA_manifest_summary.md"

FIELDS = [
    "ordinal",
    "variant",
    "source_task",
    "derived_task",
    "person_idx",
    "split",
    "case_id",
    "e109_case_id",
    "object_key",
    "ablation",
    "base_e121_variant",
    "base_e121_override",
    "warmstart_path",
    "warmstart_mask_start",
    "warmstart_mask_end",
    "mask_path",
    "source_override",
    "baseline_npz_path",
    "e113_hold_npz_path",
    "e113_hold_video_path",
    "phase_scope",
    "diagnostic_queue",
    "holdout_reason",
]


@dataclass(frozen=True)
class VariantSpec:
    ablation: str
    base_ablation: str
    terminal_enabled: bool | None = None
    terminal_mode: str | None = None


RUNNABLE_VARIANTS = [
    VariantSpec(
        "snap_support",
        base_ablation="terminal_soft_surface",
        terminal_enabled=False,
    ),
    VariantSpec(
        "snap_terminal_soft",
        base_ablation="terminal_soft_surface",
        terminal_enabled=True,
        terminal_mode="soft",
    ),
    VariantSpec(
        "snap_hard_surface",
        base_ablation="terminal_hard_surface",
        terminal_enabled=True,
        terminal_mode="hard_soft",
    ),
]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def repo_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO / p


def read_commented_tsv(path: Path) -> list[dict[str, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    header: list[str] | None = None
    rows: list[dict[str, str]] = []
    for line in lines:
        if not line.strip():
            continue
        if line.startswith("#"):
            maybe = line.lstrip("# ").split("\t")
            if maybe and maybe[0] == "ordinal":
                header = maybe
            continue
        if header is None:
            raise RuntimeError(f"{path} missing commented header")
        vals = line.split("\t")
        rows.append(dict(zip(header, vals)))
    return rows


def resize_bool_mask(mask: np.ndarray, target_len: int) -> np.ndarray:
    idx = np.linspace(0, mask.shape[0] - 1, target_len).round().astype(int)
    return mask[idx].astype(bool)


def generate_warmstart(row: dict[str, str]) -> dict[str, Any]:
    task = row["derived_task"]
    traj_path = TASK_ROOT / task / "0/trajectory_kinematic.npz"
    scene_path = TASK_ROOT / task / "scene.xml"
    qpos_ref = np.load(traj_path, allow_pickle=True)["qpos"].astype(np.float64)

    mask_data = np.load(repo_path(row["mask_path"]), allow_pickle=True)
    mask_key = (
        "eval_contact_mask_3cm"
        if "eval_contact_mask_3cm" in mask_data.files
        else "spider_contact_mask_3cm"
    )
    mask = np.asarray(mask_data[mask_key])
    person_idx = int(row["person_idx"])
    person_mask = mask[:, person_idx, :].any(axis=1)
    qpos_mask = resize_bool_mask(person_mask, qpos_ref.shape[0])
    active = np.where(qpos_mask)[0]
    if active.size == 0:
        raise RuntimeError(f"{row['case_id']} has no active resized contact frames")

    active_start = int(active[0])
    active_end = int(active[-1])
    blend = 8
    mask_start = max(0, active_start - blend)
    mask_end = active_end

    mesh_path = find_object_mesh(str(scene_path))
    qpos_snap, diag = snap_hands_to_object(
        str(scene_path),
        qpos_ref,
        (active_start, active_end),
        "both",
        str(mesh_path),
        surface_offset=0.01,
        approach_blend_frames=blend,
        max_ik_iter=30,
    )
    if not np.isfinite(qpos_snap).all():
        raise RuntimeError(f"{row['case_id']} warmstart qpos_snap contains non-finite values")
    snap_mask = np.zeros(qpos_ref.shape[0], dtype=bool)
    snap_mask[mask_start : mask_end + 1] = True

    out = WARMSTART_ROOT / f"{row['case_id']}_snap_carry_warmstart_qpos.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        qpos_ref=qpos_ref,
        qpos_snap=qpos_snap,
        snap_mask=snap_mask,
        intent_window=np.array([active_start, active_end], dtype=np.int32),
        snap_mask_window=np.array([mask_start, mask_end], dtype=np.int32),
    )

    diag_path = WARMSTART_ROOT / f"{row['case_id']}_snap_carry_diag.tsv"
    with diag_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame",
                "hand",
                "palm_to_surface_init",
                "palm_to_surface_final",
                "ik_iterations",
                "ik_residual",
                "joint_in_limits",
                "target_xyz",
            ],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for item in diag:
            writer.writerow(
                {
                    "frame": item.frame,
                    "hand": item.hand,
                    "palm_to_surface_init": f"{item.palm_to_surface_init:.8f}",
                    "palm_to_surface_final": f"{item.palm_to_surface_final:.8f}",
                    "ik_iterations": item.ik_iterations,
                    "ik_residual": f"{item.ik_residual:.8f}",
                    "joint_in_limits": int(item.joint_in_limits),
                    "target_xyz": ",".join(f"{v:.8f}" for v in item.target_xyz),
                }
            )

    final = np.array([item.palm_to_surface_final for item in diag], dtype=np.float64)
    init = np.array([item.palm_to_surface_init for item in diag], dtype=np.float64)
    return {
        "warmstart_path": rel(out),
        "warmstart_diag_path": rel(diag_path),
        "warmstart_mask_start": mask_start,
        "warmstart_mask_end": mask_end,
        "active_start": active_start,
        "active_end": active_end,
        "snap_mask_frac": float(snap_mask.mean()),
        "diag_rows": int(len(diag)),
        "init_surface_dist_mean_m": float(init.mean()) if init.size else float("nan"),
        "final_surface_dist_mean_m": float(final.mean()) if final.size else float("nan"),
        "final_surface_dist_max_m": float(final.max()) if final.size else float("nan"),
    }


def build_rows() -> list[dict[str, Any]]:
    e121_rows = read_commented_tsv(E121_VARIANTS)
    by_case_ablation = {
        (row["case_id"], row["ablation"]): row
        for row in e121_rows
    }
    warmstarts: dict[str, dict[str, Any]] = {}
    for row in e121_rows:
        if row["case_id"] not in warmstarts:
            warmstarts[row["case_id"]] = generate_warmstart(row)

    rows: list[dict[str, Any]] = []
    ordinal = 1
    case_order = ["box021_029_p2", "box021_035_p2", "box021_035_p1", "box004_083_p2"]
    for case_id in case_order:
        for spec in RUNNABLE_VARIANTS:
            base = by_case_ablation[(case_id, spec.base_ablation)]
            ws = warmstarts[case_id]
            rows.append(
                {
                    "ordinal": ordinal,
                    "variant": f"E122_{case_id}_{spec.ablation}",
                    "source_task": base["source_task"],
                    "derived_task": base["derived_task"],
                    "person_idx": base["person_idx"],
                    "split": base["split"],
                    "case_id": case_id,
                    "e109_case_id": base["e109_case_id"],
                    "object_key": base["object_key"],
                    "ablation": spec.ablation,
                    "base_e121_variant": base["variant"],
                    "base_e121_override": f"core4d_{base['variant']}",
                    "warmstart_path": ws["warmstart_path"],
                    "warmstart_mask_start": ws["warmstart_mask_start"],
                    "warmstart_mask_end": ws["warmstart_mask_end"],
                    "mask_path": base["mask_path"],
                    "source_override": base["source_override"],
                    "baseline_npz_path": base["baseline_npz_path"],
                    "e113_hold_npz_path": base["e113_hold_npz_path"],
                    "e113_hold_video_path": base["e113_hold_video_path"],
                    "phase_scope": base["phase_scope"],
                    "diagnostic_queue": base["diagnostic_queue"],
                    "holdout_reason": base.get("holdout_reason", ""),
                    "_spec": spec,
                    "_warmstart": ws,
                }
            )
            ordinal += 1
    return rows


def override_text(row: dict[str, Any]) -> str:
    spec: VariantSpec = row["_spec"]
    terminal_lines = ""
    if spec.terminal_enabled is not None:
        terminal_lines += f"terminal_carry_gate_enabled: {'true' if spec.terminal_enabled else 'false'}\n"
    if spec.terminal_mode is not None:
        terminal_lines += f"terminal_carry_gate_mode: {spec.terminal_mode}\n"
    return f"""# @package _global_
# Auto-generated by workspace/core4d/scripts/E122/build_snap_warmstart_manifest.py.
# E122 snap warmstart carry prior; case={row['case_id']} ablation={row['ablation']}.
defaults:
  - {row['base_e121_override']}
  - _self_

task: {row['derived_task']}

warmstart_qpos_path: {row['warmstart_path']}
warmstart_update_ctrl_from_qpos: true
warmup_steps: 1
{terminal_lines}video_camera: auto
"""


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str], comment: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write(f"# {comment}\n")
        f.write("# " + "\t".join(fields) + "\n")
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_overrides(rows: list[dict[str, Any]]) -> list[str]:
    OVERRIDE_ROOT.mkdir(parents=True, exist_ok=True)
    paths = []
    for row in rows:
        path = OVERRIDE_ROOT / f"core4d_{row['variant']}.yaml"
        path.write_text(override_text(row), encoding="utf-8")
        paths.append(rel(path))
    return paths


def write_preflight(rows: list[dict[str, Any]], overrides: list[str]) -> dict[str, Any]:
    preflight_rows: list[dict[str, Any]] = []
    for row in rows:
        ws_path = repo_path(row["warmstart_path"])
        ws = np.load(ws_path, allow_pickle=True)
        scene_act = TASK_ROOT / row["derived_task"] / "scene_act.xml"
        mujoco.MjModel.from_xml_path(str(scene_act))
        qpos_snap = np.asarray(ws["qpos_snap"])
        snap_mask = np.asarray(ws["snap_mask"]).astype(bool)
        ok = (
            qpos_snap.ndim == 2
            and qpos_snap.shape[1] == 43
            and snap_mask.ndim == 1
            and snap_mask.shape[0] == qpos_snap.shape[0]
            and bool(snap_mask.any())
            and np.isfinite(qpos_snap).all()
            and repo_path(row["mask_path"]).is_file()
            and repo_path(row["baseline_npz_path"]).is_file()
            and repo_path(row["e113_hold_npz_path"]).is_file()
        )
        preflight_rows.append(
            {
                "variant": row["variant"],
                "case_id": row["case_id"],
                "ablation": row["ablation"],
                "split": row["split"],
                "warmstart_exists": ws_path.is_file(),
                "warmstart_qpos_shape": "x".join(map(str, qpos_snap.shape)),
                "snap_mask_frac": f"{float(snap_mask.mean()):.6f}",
                "warmstart_mask_start": row["warmstart_mask_start"],
                "warmstart_mask_end": row["warmstart_mask_end"],
                "scene_act_loads": True,
                "mask_exists": repo_path(row["mask_path"]).is_file(),
                "baseline_exists": repo_path(row["baseline_npz_path"]).is_file(),
                "e113_hold_exists": repo_path(row["e113_hold_npz_path"]).is_file(),
                "preflight_ok": ok,
            }
        )
    fields = list(preflight_rows[0].keys())
    write_tsv(PREFLIGHT_TSV, preflight_rows, fields, "E122 snap warmstart preflight")
    all_ok = all(bool(row["preflight_ok"]) for row in preflight_rows)
    warmstart_cases = {
        row["case_id"]: row["_warmstart"]
        for row in rows
    }
    summary = {
        "experiment": "E122",
        "variants": len(rows),
        "cases": len(warmstart_cases),
        "overrides": overrides,
        "preflight_tsv": rel(PREFLIGHT_TSV),
        "all_preflight_ok": all_ok,
        "warmstarts": warmstart_cases,
    }
    PREFLIGHT_ROOT.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# E122 Phase A Manifest Summary",
        "",
        f"- variants: `{len(rows)}`",
        f"- cases: `{len(warmstart_cases)}`",
        f"- all_preflight_ok: `{all_ok}`",
        f"- variants file: `{rel(VARIANTS_TSV)}`",
        f"- preflight: `{rel(PREFLIGHT_TSV)}`",
        "",
        "| case | mask window | snap frac | final surface mean | final surface max |",
        "|---|---:|---:|---:|---:|",
    ]
    for case_id, ws in warmstart_cases.items():
        lines.append(
            f"| `{case_id}` | {ws['warmstart_mask_start']}-{ws['warmstart_mask_end']} | "
            f"{ws['snap_mask_frac'] * 100:.1f}% | {ws['final_surface_dist_mean_m']:.4f}m | "
            f"{ws['final_surface_dist_max_m']:.4f}m |"
        )
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    rows = build_rows()
    write_tsv(VARIANTS_TSV, rows, FIELDS, "E122 snap warmstart variants")
    overrides = write_overrides(rows)
    summary = write_preflight(rows, overrides)
    print(
        json.dumps(
            {
                "variants": summary["variants"],
                "cases": summary["cases"],
                "all_preflight_ok": summary["all_preflight_ok"],
                "variants_tsv": rel(VARIANTS_TSV),
                "summary_json": rel(SUMMARY_JSON),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
