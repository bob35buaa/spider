#!/usr/bin/env python3
"""Build E112 contact-aware CEM Phase A variants, overrides, and preflight."""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


REPO = Path(__file__).resolve().parents[4]
TASK_ROOT = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
OVERRIDE_ROOT = REPO / "examples/config/override"
SCRIPTS_ROOT = REPO / "workspace/core4d/scripts/E112"
RESULTS_ROOT = REPO / "workspace/core4d/results/E112"
CONTACT_MASK_ROOT = RESULTS_ROOT / "contact_masks"
PREFLIGHT_ROOT = RESULTS_ROOT / "preflight"
VARIANTS_TSV = SCRIPTS_ROOT / "variants.tsv"
PREFLIGHT_TSV = PREFLIGHT_ROOT / "phaseA_preflight.tsv"
SUMMARY_JSON = PREFLIGHT_ROOT / "phaseA_manifest_summary.json"
SUMMARY_MD = PREFLIGHT_ROOT / "phaseA_manifest_summary.md"


VARIANT_FIELDS = [
    "ordinal",
    "variant",
    "source_task",
    "derived_task",
    "person_idx",
    "split",
    "case_id",
    "object_key",
    "ablation",
    "mask_path",
    "source_override",
]


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    object_key: str
    source_task: str
    derived_task: str
    person_idx: int
    source_override: str
    mask_path: str
    mask_kind: str


CASES = [
    CaseSpec(
        case_id="box004_082_p1",
        object_key="box004",
        source_task="e091_box004_20231003_2_082_p1",
        derived_task="e091_box004_20231003_2_082_p1_e096b_mask_cem",
        person_idx=0,
        source_override="core4d_E096bP2_box004_082_p1_mask_cem",
        mask_path="workspace/core4d/results/E096b/contact_masks/e091_box004_20231003_2_082_p1/raw_contact_mask_3cm.npz",
        mask_kind="mjwp_compatible",
    ),
    CaseSpec(
        case_id="box021_035_p1",
        object_key="box021",
        source_task="d003_box021_20231011_035_p1",
        derived_task="d003_box021_20231011_035_p1_e107_clean",
        person_idx=0,
        source_override="core4d_E107C02_box021_20231011_035_p1_ref_fk_clean",
        mask_path="workspace/core4d/results/E079/contact_masks/box021_person1/raw_contact_mask_3cm.npz",
        mask_kind="mjwp_compatible",
    ),
    CaseSpec(
        case_id="box026_135_p1",
        object_key="box026",
        source_task="e091_box026_20231020_135_p1",
        derived_task="e091_box026_20231020_135_p1_e106_clean",
        person_idx=0,
        source_override="core4d_E106B22_box026_20231020_135_p1_ref_fk_clean",
        mask_path="workspace/core4d/results/E104/d002_medium_box_multithreshold/per_sequence/20231020_135_box026/raw_contact_proxy.npz",
        mask_kind="e104_raw_proxy",
    ),
]


ABLATIONS = [
    ("baseline_ref_fk", "local-gpu0"),
    ("raw_mask_ref_fk", "remote-gpu0"),
    ("hold_band", "remote-gpu1"),
]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def repo_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO / p


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str], comment: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write(f"# {comment}\n")
        f.write("# " + "\t".join(fields) + "\n")
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def resize_nearest(mask: np.ndarray, target_len: int) -> np.ndarray:
    if mask.shape[0] == target_len:
        return mask.astype(bool)
    idx = np.round(np.linspace(0, mask.shape[0] - 1, target_len)).astype(np.int64)
    return mask[idx].astype(bool)


def task_qpos_len(case: CaseSpec) -> int:
    traj = TASK_ROOT / case.derived_task / "0/trajectory_kinematic.npz"
    if not traj.is_file():
        raise FileNotFoundError(traj)
    return int(np.load(traj, allow_pickle=True)["qpos"].shape[0])


def make_e104_mask(case: CaseSpec) -> Path:
    source = repo_path(case.mask_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    data = np.load(source, allow_pickle=True)
    thresholds = np.asarray(data["thresholds_m"], dtype=float)
    idx = int(np.argmin(np.abs(thresholds - 0.03)))
    if abs(float(thresholds[idx]) - 0.03) > 1e-6:
        raise ValueError(f"{source} has no 0.03m threshold: {thresholds}")
    raw_mask = np.asarray(data["masks"][..., idx], dtype=bool)
    qpos_len = task_qpos_len(case)
    spider_mask = resize_nearest(raw_mask, qpos_len)
    eval_mask = spider_mask.copy()
    out_dir = CONTACT_MASK_ROOT / case.case_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "raw_contact_mask_3cm.npz"
    np.savez(
        out,
        raw_contact_mask_3cm=raw_mask,
        spider_contact_mask_3cm=spider_mask,
        eval_contact_mask_3cm=eval_mask,
        persons=data["persons"],
        hands=data["hands"],
        threshold_m=np.array(0.03, dtype=np.float64),
        source_raw_contact_proxy=np.array(rel(source)),
        conversion_note=np.array("E112: nearest-neighbor resize E104 raw 3cm mask to task qpos length for MJWP auto axis."),
    )
    return out


def ensure_mask(case: CaseSpec) -> Path:
    if case.mask_kind == "e104_raw_proxy":
        return make_e104_mask(case)
    path = repo_path(case.mask_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def validate_mask(path: Path, person_idx: int) -> dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    keys = set(data.files)
    axis = "eval" if "eval_contact_mask_3cm" in keys else "spider"
    key = f"{axis}_contact_mask_3cm"
    if key not in keys:
        raise KeyError(f"{path} missing eval_contact_mask_3cm or spider_contact_mask_3cm")
    mask = np.asarray(data[key])
    if mask.ndim != 3 or mask.shape[2] != 2:
        raise ValueError(f"{path}:{key} expected shape (T, person, 2), got {mask.shape}")
    if person_idx >= mask.shape[1]:
        raise ValueError(f"{path}:{key} person_idx={person_idx} out of shape {mask.shape}")
    per_hand = mask[:, person_idx, :].astype(bool)
    return {
        "mask_key": key,
        "mask_shape": "x".join(str(x) for x in mask.shape),
        "mask_left_active_frac": float(per_hand[:, 0].mean()),
        "mask_right_active_frac": float(per_hand[:, 1].mean()),
        "mask_either_active_frac": float(per_hand.any(axis=1).mean()),
    }


def validate_task(case: CaseSpec) -> dict[str, Any]:
    task_dir = TASK_ROOT / case.derived_task
    scene = task_dir / "scene.xml"
    scene_act = task_dir / "scene_act.xml"
    traj = task_dir / "0/trajectory_kinematic.npz"
    for path in [scene, scene_act, traj]:
        if not path.is_file():
            raise FileNotFoundError(path)
    model_scene = mujoco.MjModel.from_xml_path(str(scene))
    model_act = mujoco.MjModel.from_xml_path(str(scene_act))
    qpos = np.load(traj, allow_pickle=True)["qpos"]
    ok = (
        model_scene.nq == 43
        and model_scene.nv == 41
        and model_scene.nu == 29
        and model_act.nq == 42
        and model_act.nv == 41
        and model_act.nu == 35
        and qpos.ndim == 2
        and qpos.shape[1] == 43
    )
    return {
        "task_dir": rel(task_dir),
        "scene_xml": rel(scene),
        "scene_act_xml": rel(scene_act),
        "trajectory_npz": rel(traj),
        "scene_nq": int(model_scene.nq),
        "scene_nv": int(model_scene.nv),
        "scene_nu": int(model_scene.nu),
        "scene_act_nq": int(model_act.nq),
        "scene_act_nv": int(model_act.nv),
        "scene_act_nu": int(model_act.nu),
        "qpos_shape": "x".join(str(x) for x in qpos.shape),
        "task_validation_ok": str(ok),
    }


def variant_name(case: CaseSpec, ablation: str) -> str:
    return f"E112_{case.case_id}_{ablation}"


def write_override(case: CaseSpec, ablation: str, mask: Path) -> Path:
    variant = variant_name(case, ablation)
    mask_enabled = ablation in {"raw_mask_ref_fk", "hold_band"}
    hold_enabled = ablation == "hold_band"
    path = OVERRIDE_ROOT / f"core4d_{variant}.yaml"
    mask_source = "core4d_3cm" if mask_enabled else ""
    mask_path = rel(mask) if mask_enabled else ""
    contact_gain = "5.0" if mask_enabled else "0.0"
    hold_scale = "1.0" if hold_enabled else "0.0"
    hold_sigma = "0.05"
    hold_start = "0.0"
    hold_end = "999.0" if hold_enabled else "0.0"
    content = f"""# @package _global_
# Auto-generated by workspace/core4d/scripts/E112/build_contact_aware_cem_manifest.py.
# E112 Phase A contact-aware CEM ablation; case={case.case_id} ablation={ablation}.
defaults:
  - core4d_E089A_box021_person1_upperobj
  - _self_

task: {case.derived_task}

contact_hdmi_target_source: ref_fk
contact_hdmi_target_path: ""
contact_hdmi_target_uses_eef_offset: true
contact_hdmi_gain: {contact_gain}
contact_hdmi_mask_source: "{mask_source}"
contact_hdmi_mask_path: "{mask_path}"
contact_hdmi_mask_person_idx: {case.person_idx}
contact_hdmi_mask_time_axis: auto

hold_contact_rew_scale: {hold_scale}
hold_contact_sigma: {hold_sigma}
hold_contact_start_eval_time: {hold_start}
hold_contact_end_eval_time: {hold_end}
hold_contact_require_ref_contact: true

video_camera: auto
"""
    path.write_text(content, encoding="utf-8")
    return path


def build() -> tuple[list[dict[str, str]], list[dict[str, Any]], list[Path]]:
    rows: list[dict[str, str]] = []
    preflight: list[dict[str, Any]] = []
    overrides: list[Path] = []
    ordinal = 1
    for ablation, split in ABLATIONS:
        for case in CASES:
            mask = ensure_mask(case)
            validation = validate_task(case)
            mask_validation = validate_mask(mask, case.person_idx)
            override = write_override(case, ablation, mask)
            variant = variant_name(case, ablation)
            row = {
                "ordinal": str(ordinal),
                "variant": variant,
                "source_task": case.source_task,
                "derived_task": case.derived_task,
                "person_idx": str(case.person_idx),
                "split": split,
                "case_id": case.case_id,
                "object_key": case.object_key,
                "ablation": ablation,
                "mask_path": rel(mask),
                "source_override": case.source_override,
            }
            rows.append(row)
            preflight.append(
                {
                    **row,
                    **validation,
                    **mask_validation,
                    "override": rel(override),
                    "source_mask_path": case.mask_path,
                    "mask_kind": case.mask_kind,
                    "preflight_ok": "True",
                }
            )
            overrides.append(override)
            ordinal += 1
    return rows, preflight, overrides


def main() -> None:
    SCRIPTS_ROOT.mkdir(parents=True, exist_ok=True)
    PREFLIGHT_ROOT.mkdir(parents=True, exist_ok=True)
    rows, preflight, overrides = build()
    write_tsv(VARIANTS_TSV, rows, VARIANT_FIELDS, "E112 Phase A contact-aware CEM variants")
    fields = sorted({key for row in preflight for key in row})
    write_tsv(PREFLIGHT_TSV, preflight, fields, "E112 Phase A preflight")
    summary = {
        "rows": len(rows),
        "cases": [case.case_id for case in CASES],
        "ablations": [name for name, _ in ABLATIONS],
        "variants_tsv": rel(VARIANTS_TSV),
        "preflight_tsv": rel(PREFLIGHT_TSV),
        "overrides": [rel(path) for path in overrides],
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# E112 Phase A Manifest Summary",
        "",
        f"- variants: `{len(rows)}`",
        f"- variants tsv: `{rel(VARIANTS_TSV)}`",
        f"- preflight: `{rel(PREFLIGHT_TSV)}`",
        "",
        "| variant | split | task | mask active either | preflight |",
        "|---|---|---|---:|---|",
    ]
    for row in preflight:
        lines.append(
            f"| `{row['variant']}` | `{row['split']}` | `{row['derived_task']}` | "
            f"{float(row['mask_either_active_frac']) * 100:.1f}% | `{row['preflight_ok']}` |"
        )
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {rel(VARIANTS_TSV)} rows={len(rows)}")
    print(f"wrote {rel(PREFLIGHT_TSV)}")
    print(f"wrote {rel(SUMMARY_MD)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"E112 manifest build failed: {exc}", file=sys.stderr)
        raise
