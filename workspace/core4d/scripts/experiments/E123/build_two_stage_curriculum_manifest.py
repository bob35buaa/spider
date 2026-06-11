#!/usr/bin/env python3
"""Build E123 two-stage carry curriculum manifest and warmstart bridge."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R


REPO = Path(__file__).resolve().parents[4]
TASK_ROOT = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
OVERRIDE_ROOT = REPO / "examples/config/override"
SCRIPTS_ROOT = REPO / "workspace/core4d/scripts/E123"
RESULTS_ROOT = REPO / "workspace/core4d/results/E123"
WARMSTART_ROOT = RESULTS_ROOT / "warmstarts"
PREFLIGHT_ROOT = RESULTS_ROOT / "preflight"

E119_VARIANTS = REPO / "workspace/core4d/scripts/E119/variants.tsv"
E120_VARIANTS = REPO / "workspace/core4d/scripts/E120/variants.tsv"
E121_VARIANTS = REPO / "workspace/core4d/scripts/E121/variants.tsv"

STAGE1_TSV = SCRIPTS_ROOT / "stage1.tsv"
VARIANTS_TSV = SCRIPTS_ROOT / "variants.tsv"
PREFLIGHT_TSV = PREFLIGHT_ROOT / "phaseA_preflight.tsv"
SUMMARY_JSON = PREFLIGHT_ROOT / "phaseA_manifest_summary.json"
SUMMARY_MD = PREFLIGHT_ROOT / "phaseA_manifest_summary.md"

STAGE1_FIELDS = [
    "ordinal",
    "stage1_variant",
    "source_task",
    "derived_task",
    "person_idx",
    "split",
    "case_id",
    "e109_case_id",
    "object_key",
    "seed_ablation",
    "seed_source_variant",
    "seed_source_override",
    "mask_path",
    "source_override",
    "baseline_npz_path",
    "e113_hold_npz_path",
    "e113_hold_video_path",
    "phase_scope",
    "diagnostic_queue",
    "holdout_reason",
    "expected_warmstart_path",
]

STAGE2_FIELDS = [
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
    "stage1_variant",
    "stage1_override",
    "stage1_seed_ablation",
    "base_stage2_variant",
    "base_stage2_override",
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

CASE_ORDER = ["box021_029_p2", "box021_035_p2", "box021_035_p1", "box004_083_p2"]


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
        rows.append(dict(zip(header, line.split("\t"))))
    return rows


def resize_bool_mask(mask: np.ndarray, target_len: int) -> np.ndarray:
    if mask.shape[0] == target_len:
        return mask.astype(bool)
    idx = np.linspace(0, mask.shape[0] - 1, target_len).round().astype(np.int64)
    return mask[idx].astype(bool)


def resize_rows(arr: np.ndarray, target_len: int) -> np.ndarray:
    if arr.shape[0] == target_len:
        return arr
    idx = np.linspace(0, arr.shape[0] - 1, target_len).round().astype(np.int64)
    return arr[idx]


def scene_act_to_freejoint(scene_act_xml: Path, source_qpos: np.ndarray, scene_qpos: np.ndarray) -> np.ndarray:
    model = mujoco.MjModel.from_xml_path(str(scene_act_xml))
    object_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    if object_body < 0:
        raise RuntimeError(f"{scene_act_xml} missing object body")
    meta_path = scene_act_xml.parent / "scene_act_meta.json"
    euler_conv = "XYZ"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        euler_conv = meta.get("euler_convention", euler_conv)

    nq_robot = scene_qpos.shape[1] - 6
    if source_qpos.shape[1] != nq_robot + 7:
        raise RuntimeError(
            f"Unexpected qpos dims for {scene_act_xml}: source={source_qpos.shape}, scene={scene_qpos.shape}"
        )

    body_pos = model.body_pos[object_body].copy()
    body_quat_wxyz = model.body_quat[object_body].copy()
    body_rot = R.from_quat(
        [body_quat_wxyz[1], body_quat_wxyz[2], body_quat_wxyz[3], body_quat_wxyz[0]]
    )

    out = source_qpos.copy()
    out[:, :nq_robot] = scene_qpos[:, :nq_robot]
    slide = scene_qpos[:, nq_robot : nq_robot + 3]
    euler = scene_qpos[:, nq_robot + 3 : nq_robot + 6]
    world_pos = body_pos[None, :] + body_rot.apply(slide)
    world_rot = body_rot * R.from_euler(euler_conv, euler)
    quat_xyzw = world_rot.as_quat()
    out[:, nq_robot : nq_robot + 3] = world_pos
    out[:, nq_robot + 3 : nq_robot + 7] = np.column_stack(
        [quat_xyzw[:, 3], quat_xyzw[:, 0], quat_xyzw[:, 1], quat_xyzw[:, 2]]
    )
    return out


def case_mask(row: dict[str, str], target_len: int) -> tuple[np.ndarray, int, int]:
    data = np.load(repo_path(row["mask_path"]), allow_pickle=True)
    key = "eval_contact_mask_3cm" if "eval_contact_mask_3cm" in data.files else "spider_contact_mask_3cm"
    raw = np.asarray(data[key])
    person_idx = int(row["person_idx"])
    person_mask = raw[:, person_idx, :].any(axis=1)
    mask = resize_bool_mask(person_mask, target_len)
    active = np.where(mask)[0]
    if active.size == 0:
        raise RuntimeError(f"{row['case_id']} has no active contact frames")
    start = max(0, int(active[0]) - 8)
    end = int(active[-1])
    snap_mask = np.zeros(target_len, dtype=bool)
    snap_mask[start : end + 1] = True
    return snap_mask, start, end


def select_stage1_rows() -> list[dict[str, Any]]:
    e119 = read_commented_tsv(E119_VARIANTS)
    e121 = read_commented_tsv(E121_VARIANTS)
    e119_by_case = {(row["case_id"], row["ablation"]): row for row in e119}
    e121_by_case = {(row["case_id"], row["ablation"]): row for row in e121}
    rows: list[dict[str, Any]] = []
    for ordinal, case_id in enumerate(CASE_ORDER, start=1):
        if (case_id, "corridor_ref_pose_bodyguard") in e119_by_case:
            base = e119_by_case[(case_id, "corridor_ref_pose_bodyguard")]
            seed_ablation = "pose_bodyguard_seed"
        else:
            base = e121_by_case[(case_id, "terminal_soft_surface")]
            seed_ablation = "guard_terminal_soft_seed"
        stage1_variant = f"E123_{case_id}_{seed_ablation}"
        rows.append(
            {
                "ordinal": ordinal,
                "stage1_variant": stage1_variant,
                "source_task": base["source_task"],
                "derived_task": base["derived_task"],
                "person_idx": base["person_idx"],
                "split": base["split"],
                "case_id": case_id,
                "e109_case_id": base["e109_case_id"],
                "object_key": base["object_key"],
                "seed_ablation": seed_ablation,
                "seed_source_variant": base["variant"],
                "seed_source_override": f"core4d_{base['variant']}",
                "mask_path": base["mask_path"],
                "source_override": base["source_override"],
                "baseline_npz_path": base["baseline_npz_path"],
                "e113_hold_npz_path": base["e113_hold_npz_path"],
                "e113_hold_video_path": base["e113_hold_video_path"],
                "phase_scope": base["phase_scope"],
                "diagnostic_queue": base["diagnostic_queue"],
                "holdout_reason": base.get("holdout_reason", ""),
                "expected_warmstart_path": rel(
                    WARMSTART_ROOT / f"{case_id}_{seed_ablation}_warmstart_qpos.npz"
                ),
            }
        )
    return rows


def build_stage2_rows(stage1_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    e120 = read_commented_tsv(E120_VARIANTS)
    e121 = read_commented_tsv(E121_VARIANTS)
    by_case_ablation = {
        (row["case_id"], row["ablation"]): row
        for row in [*e120, *e121]
    }
    stage1_by_case = {row["case_id"]: row for row in stage1_rows}
    specs = [
        ("stage2_support_surface", "support_surface_direct"),
        ("stage2_terminal_soft", "terminal_soft_surface"),
    ]
    rows: list[dict[str, Any]] = []
    ordinal = 1
    for case_id in CASE_ORDER:
        stage1 = stage1_by_case[case_id]
        for ablation, base_ablation in specs:
            base = by_case_ablation[(case_id, base_ablation)]
            rows.append(
                {
                    "ordinal": ordinal,
                    "variant": f"E123_{case_id}_{ablation}",
                    "source_task": base["source_task"],
                    "derived_task": base["derived_task"],
                    "person_idx": base["person_idx"],
                    "split": base["split"],
                    "case_id": case_id,
                    "e109_case_id": base["e109_case_id"],
                    "object_key": base["object_key"],
                    "ablation": ablation,
                    "stage1_variant": stage1["stage1_variant"],
                    "stage1_override": f"core4d_{stage1['stage1_variant']}",
                    "stage1_seed_ablation": stage1["seed_ablation"],
                    "base_stage2_variant": base["variant"],
                    "base_stage2_override": f"core4d_{base['variant']}",
                    "warmstart_path": stage1["expected_warmstart_path"],
                    "warmstart_mask_start": "",
                    "warmstart_mask_end": "",
                    "mask_path": base["mask_path"],
                    "source_override": base["source_override"],
                    "baseline_npz_path": base["baseline_npz_path"],
                    "e113_hold_npz_path": base["e113_hold_npz_path"],
                    "e113_hold_video_path": base["e113_hold_video_path"],
                    "phase_scope": base["phase_scope"],
                    "diagnostic_queue": base["diagnostic_queue"],
                    "holdout_reason": base.get("holdout_reason", ""),
                }
            )
            ordinal += 1
    return rows


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str], comment: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write(f"# {comment}\n")
        f.write("# " + "\t".join(fields) + "\n")
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def stage1_override_text(row: dict[str, Any]) -> str:
    return f"""# @package _global_
# Auto-generated by workspace/core4d/scripts/E123/build_two_stage_curriculum_manifest.py.
# E123 stage1 seed; case={row['case_id']} seed={row['seed_ablation']}.
defaults:
  - {row['seed_source_override']}
  - _self_

task: {row['derived_task']}
video_camera: auto
"""


def stage2_override_text(row: dict[str, Any]) -> str:
    return f"""# @package _global_
# Auto-generated by workspace/core4d/scripts/E123/build_two_stage_curriculum_manifest.py.
# E123 stage2; case={row['case_id']} ablation={row['ablation']}.
defaults:
  - {row['base_stage2_override']}
  - _self_

task: {row['derived_task']}

warmstart_qpos_path: {row['warmstart_path']}
warmstart_update_ctrl_from_qpos: true
warmup_steps: 1
video_camera: auto
"""


def write_overrides(stage1_rows: list[dict[str, Any]], stage2_rows: list[dict[str, Any]]) -> list[str]:
    OVERRIDE_ROOT.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for row in stage1_rows:
        path = OVERRIDE_ROOT / f"core4d_{row['stage1_variant']}.yaml"
        path.write_text(stage1_override_text(row), encoding="utf-8")
        paths.append(rel(path))
    for row in stage2_rows:
        path = OVERRIDE_ROOT / f"core4d_{row['variant']}.yaml"
        path.write_text(stage2_override_text(row), encoding="utf-8")
        paths.append(rel(path))
    return paths


def write_preflight(stage1_rows: list[dict[str, Any]], stage2_rows: list[dict[str, Any]], overrides: list[str]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for row in stage1_rows:
        scene = TASK_ROOT / row["derived_task"] / "scene_act.xml"
        source = TASK_ROOT / row["derived_task"] / "0/trajectory_kinematic.npz"
        ok = (
            scene.is_file()
            and source.is_file()
            and repo_path(row["mask_path"]).is_file()
            and repo_path(row["baseline_npz_path"]).is_file()
            and repo_path(row["e113_hold_npz_path"]).is_file()
        )
        if scene.is_file():
            mujoco.MjModel.from_xml_path(str(scene))
        rows.append(
            {
                "row_type": "stage1",
                "variant": row["stage1_variant"],
                "case_id": row["case_id"],
                "ablation": row["seed_ablation"],
                "split": row["split"],
                "scene_act_loads": scene.is_file(),
                "source_qpos_exists": source.is_file(),
                "mask_exists": repo_path(row["mask_path"]).is_file(),
                "baseline_exists": repo_path(row["baseline_npz_path"]).is_file(),
                "e113_hold_exists": repo_path(row["e113_hold_npz_path"]).is_file(),
                "warmstart_expected": row["expected_warmstart_path"],
                "preflight_ok": ok,
            }
        )
    for row in stage2_rows:
        scene = TASK_ROOT / row["derived_task"] / "scene_act.xml"
        rows.append(
            {
                "row_type": "stage2",
                "variant": row["variant"],
                "case_id": row["case_id"],
                "ablation": row["ablation"],
                "split": row["split"],
                "scene_act_loads": scene.is_file(),
                "source_qpos_exists": (TASK_ROOT / row["derived_task"] / "0/trajectory_kinematic.npz").is_file(),
                "mask_exists": repo_path(row["mask_path"]).is_file(),
                "baseline_exists": repo_path(row["baseline_npz_path"]).is_file(),
                "e113_hold_exists": repo_path(row["e113_hold_npz_path"]).is_file(),
                "warmstart_expected": row["warmstart_path"],
                "preflight_ok": (
                    scene.is_file()
                    and repo_path(row["mask_path"]).is_file()
                    and repo_path(row["baseline_npz_path"]).is_file()
                    and repo_path(row["e113_hold_npz_path"]).is_file()
                ),
            }
        )
    write_tsv(PREFLIGHT_TSV, rows, list(rows[0].keys()), "E123 two-stage preflight")
    all_ok = all(bool(row["preflight_ok"]) for row in rows)
    summary = {
        "experiment": "E123",
        "stage1_rows": len(stage1_rows),
        "stage2_rows": len(stage2_rows),
        "overrides": overrides,
        "preflight_tsv": rel(PREFLIGHT_TSV),
        "all_preflight_ok": all_ok,
        "note": "Stage2 warmstart files are expected after Stage1 conversion, not at initial manifest build.",
    }
    PREFLIGHT_ROOT.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# E123 Phase A Manifest Summary",
        "",
        f"- stage1 rows: `{len(stage1_rows)}`",
        f"- stage2 rows: `{len(stage2_rows)}`",
        f"- all_preflight_ok: `{all_ok}`",
        f"- stage1 file: `{rel(STAGE1_TSV)}`",
        f"- variants file: `{rel(VARIANTS_TSV)}`",
        f"- preflight: `{rel(PREFLIGHT_TSV)}`",
        "",
        "| case | split | stage1 seed | stage2 variants |",
        "|---|---|---|---|",
    ]
    by_case: dict[str, list[str]] = {}
    for row in stage2_rows:
        by_case.setdefault(row["case_id"], []).append(row["ablation"])
    for row in stage1_rows:
        lines.append(
            f"| `{row['case_id']}` | `{row['split']}` | `{row['seed_ablation']}` | "
            f"{', '.join(f'`{v}`' for v in by_case[row['case_id']])} |"
        )
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def build_manifest() -> dict[str, Any]:
    stage1_rows = select_stage1_rows()
    stage2_rows = build_stage2_rows(stage1_rows)
    write_tsv(STAGE1_TSV, stage1_rows, STAGE1_FIELDS, "E123 stage1 seeds")
    write_tsv(VARIANTS_TSV, stage2_rows, STAGE2_FIELDS, "E123 stage2 variants")
    overrides = write_overrides(stage1_rows, stage2_rows)
    return write_preflight(stage1_rows, stage2_rows, overrides)


def convert_stage1_outputs(stage: str, split: str | None = None) -> list[dict[str, Any]]:
    if not STAGE1_TSV.is_file():
        raise SystemExit(f"Missing {STAGE1_TSV}; build manifest first")
    rows = read_commented_tsv(STAGE1_TSV)
    converted: list[dict[str, Any]] = []
    for row in rows:
        if split and row["split"] != split:
            continue
        stage1_npz = (
            RESULTS_ROOT
            / "cem"
            / stage
            / f"{row['stage1_variant']}_outdir_{stage}"
            / "trajectory_mjwp_act.npz"
        )
        if not stage1_npz.is_file():
            continue
        source_path = TASK_ROOT / row["derived_task"] / "0/trajectory_kinematic.npz"
        scene_act = TASK_ROOT / row["derived_task"] / "scene_act.xml"
        source_qpos = np.load(source_path, allow_pickle=True)["qpos"].astype(np.float64)
        data = np.load(stage1_npz, allow_pickle=True)
        qpos = np.asarray(data["qpos"], dtype=np.float64)
        if qpos.ndim == 3:
            scene_qpos = qpos[:, -1, :]
        elif qpos.ndim == 2:
            scene_qpos = qpos
        else:
            raise RuntimeError(f"{stage1_npz} qpos has unsupported shape {qpos.shape}")
        scene_qpos = resize_rows(scene_qpos, source_qpos.shape[0])
        qpos_snap = scene_act_to_freejoint(scene_act, source_qpos, scene_qpos)
        snap_mask, mask_start, mask_end = case_mask(row, source_qpos.shape[0])
        out = repo_path(row["expected_warmstart_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out,
            qpos_ref=source_qpos,
            qpos_snap=qpos_snap,
            snap_mask=snap_mask,
            stage1_npz=rel(stage1_npz),
            source_qpos_path=rel(source_path),
            scene_act_xml=rel(scene_act),
            snap_mask_window=np.array([mask_start, mask_end], dtype=np.int32),
            source_shape=np.array(source_qpos.shape, dtype=np.int32),
            stage1_scene_shape=np.array(scene_qpos.shape, dtype=np.int32),
        )
        converted.append(
            {
                "case_id": row["case_id"],
                "stage1_variant": row["stage1_variant"],
                "warmstart_path": rel(out),
                "mask_start": mask_start,
                "mask_end": mask_end,
                "snap_mask_frac": float(snap_mask.mean()),
                "source_shape": list(source_qpos.shape),
                "stage1_scene_shape": list(scene_qpos.shape),
            }
        )
    report = {
        "stage": stage,
        "split": split or "all",
        "converted": converted,
        "converted_count": len(converted),
    }
    out_json = PREFLIGHT_ROOT / f"stage1_conversion_{stage}_{split or 'all'}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return converted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--convert-stage1", action="store_true")
    parser.add_argument("--stage", default="smoke", choices=["smoke", "full"])
    parser.add_argument("--split", default="")
    args = parser.parse_args()
    if args.convert_stage1:
        converted = convert_stage1_outputs(args.stage, args.split or None)
        print(json.dumps({"converted": len(converted), "stage": args.stage, "split": args.split or "all"}, indent=2))
        return
    summary = build_manifest()
    print(
        json.dumps(
            {
                "stage1_rows": summary["stage1_rows"],
                "stage2_rows": summary["stage2_rows"],
                "all_preflight_ok": summary["all_preflight_ok"],
                "stage1_tsv": rel(STAGE1_TSV),
                "variants_tsv": rel(VARIANTS_TSV),
                "summary_json": rel(SUMMARY_JSON),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
