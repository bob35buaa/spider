#!/usr/bin/env python3
"""Build E143 raw-mask ref-FK 24-case manifest, masks, and overrides."""

from __future__ import annotations

import csv
import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


REPO = Path(__file__).resolve().parents[4]
TASK_ROOT = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
OVERRIDE_ROOT = REPO / "examples/config/override"
SCRIPTS_ROOT = REPO / "workspace/core4d/scripts/E143"
RESULTS_ROOT = REPO / "workspace/core4d/results/E143"
CONTACT_MASK_ROOT = RESULTS_ROOT / "contact_masks"
PREFLIGHT_ROOT = RESULTS_ROOT / "preflight"
CEM_FULL_ROOT = RESULTS_ROOT / "cem/full"
E109_CASES = REPO / "workspace/core4d/results/E109/expanded_24_work_cases/expanded_24_case_comparison.tsv"
E109_METHODS = REPO / "workspace/core4d/results/E109/expanded_24_work_cases/expanded_24_method_metrics.tsv"
E110_CONTACT = REPO / "workspace/core4d/results/E110/contact_metric_audit/contact_band_metrics.tsv"
E112_SUMMARY = REPO / "workspace/core4d/results/E112/cem/full/full_eval_summary.csv"
E104_PROXY_ROOT = REPO / "workspace/core4d/results/E104/d002_medium_box_multithreshold/per_sequence"
VARIANTS_TSV = SCRIPTS_ROOT / "variants.tsv"
PREFLIGHT_TSV = PREFLIGHT_ROOT / "raw_mask_ref_fk_24case_preflight.tsv"
SUMMARY_JSON = PREFLIGHT_ROOT / "raw_mask_ref_fk_24case_manifest_summary.json"
SUMMARY_MD = PREFLIGHT_ROOT / "raw_mask_ref_fk_24case_manifest_summary.md"


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


@dataclass(frozen=True)
class MethodRows:
    omni: dict[str, str]
    spider: dict[str, str]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def repo_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO / p


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str], comment: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write(f"# {comment}\n")
        f.write("# " + "\t".join(fields) + "\n")
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def method_rows() -> dict[str, MethodRows]:
    grouped: dict[str, dict[str, dict[str, str]]] = {}
    for row in read_tsv(E109_METHODS):
        grouped.setdefault(row["case_id"], {})[row["method"]] = row
    out: dict[str, MethodRows] = {}
    for case_id, rows in grouped.items():
        if "OmniRetarget" in rows and "Spider CEM" in rows:
            out[case_id] = MethodRows(omni=rows["OmniRetarget"], spider=rows["Spider CEM"])
    return out


def base_short_case_id(e109_case_id: str, object_key: str) -> str:
    if e109_case_id.startswith("box023_person"):
        return e109_case_id
    m = re.match(r"^[a-z]\d+_(box\d+)_(\d{8}(?:_\d+)?)_(\d+)_p([12])$", e109_case_id)
    if m:
        return f"{m.group(1)}_{m.group(3)}_p{m.group(4)}"
    m = re.match(r"^(bucket\d+)_(\d{8}(?:_\d+)?)_(\d+)_p([12])$", e109_case_id)
    if m:
        return f"{m.group(1)}_{m.group(2)}_{m.group(3)}_p{m.group(4)}"
    return re.sub(r"[^A-Za-z0-9_]+", "_", e109_case_id)


def disambiguated_short_case_id(e109_case_id: str, object_key: str, short_counts: dict[str, int]) -> str:
    base = base_short_case_id(e109_case_id, object_key)
    if short_counts.get(base, 0) <= 1:
        return base
    m = re.match(r"^[a-z]\d+_(box\d+)_(\d{8}(?:_\d+)?)_(\d+)_p([12])$", e109_case_id)
    if m:
        return f"{m.group(1)}_{m.group(2)}_{m.group(3)}_p{m.group(4)}"
    return e109_case_id


def parse_person_idx(case_id: str) -> int:
    if case_id.endswith("_p2") or case_id.endswith("person2"):
        return 1
    return 0


def derived_task_from_scene(scene_xml: str, e109_case_id: str) -> str:
    scene = repo_path(scene_xml)
    if str(scene).startswith(str(TASK_ROOT)):
        return scene.parent.name
    if e109_case_id == "box023_person2":
        return "box023_person2_legobj_e026_e081"
    raise ValueError(f"Cannot infer TASK_ROOT derived_task for {e109_case_id}: {scene_xml}")


def e104_proxy_for_case(e109_case_id: str, object_key: str) -> Path:
    parts = e109_case_id.split("_")
    if object_key.startswith("box") and len(parts) >= 5:
        # e091_box026_20231020_135_p1 -> 20231020_135_box026
        date = parts[2]
        seq = parts[3]
        if parts[0].startswith("e") and parts[1] == object_key:
            return E104_PROXY_ROOT / f"{date}_{seq}_{object_key}/raw_contact_proxy.npz"
    raise FileNotFoundError(f"No E104 proxy rule for {e109_case_id}")


def explicit_mask_candidates(e109_case_id: str, short_id: str, object_key: str) -> list[Path]:
    candidates: list[Path] = []
    if e109_case_id == "box023_person2":
        candidates += [
            REPO / "workspace/core4d/results/E081/contact_masks/box023_person2/raw_contact_mask_3cm.npz",
            REPO / "workspace/core4d/results/E079/contact_masks/box023_person2/raw_contact_mask_3cm.npz",
            REPO / "workspace/core4d/results/E077/contact_masks/box023/raw_contact_mask_3cm.npz",
        ]
    if e109_case_id.startswith("bucket004_"):
        task = f"dcv3_omnirt_v1_ref_fk_{e109_case_id}"
        candidates.append(
            REPO
            / "workspace/core4d/results/E108/s3_retarget/omnirt_v1/ref_fk/results/omnirt_v1_ref_fk/contact_masks"
            / task
            / "raw_contact_mask_3cm.npz"
        )
    if e109_case_id.startswith("d003_box021_20231011_035_p1"):
        candidates.append(REPO / "workspace/core4d/results/E079/contact_masks/box021_person1/raw_contact_mask_3cm.npz")
    if e109_case_id.startswith("d003_box021_20231011_035_p2"):
        candidates.append(REPO / "workspace/core4d/results/E082/contact_masks/d003_box021_20231011_035_p2/raw_contact_mask_3cm.npz")
    if e109_case_id.startswith("d003_box021_20231018_029_p2"):
        candidates.append(REPO / "workspace/core4d/results/E082/contact_masks/d003_box021_20231018_029_p2/raw_contact_mask_3cm.npz")
    if e109_case_id.startswith("e091_box004_20231003_2_082_p1"):
        candidates.append(REPO / "workspace/core4d/results/E096b/contact_masks/e091_box004_20231003_2_082_p1/raw_contact_mask_3cm.npz")
    if e109_case_id.startswith("e091_box004_20231003_2_083_p1"):
        candidates.append(REPO / "workspace/core4d/results/E096b/contact_masks/e091_box004_20231003_2_083_p1/raw_contact_mask_3cm.npz")
    if e109_case_id.startswith("e091_box004_20231003_2_083_p2"):
        candidates.append(REPO / "workspace/core4d/results/E113/contact_masks/box004_083_p2/raw_contact_mask_3cm.npz")
    if e109_case_id == "e091_box026_20231020_135_p1":
        candidates.append(REPO / "workspace/core4d/results/E112/contact_masks/box026_135_p1/raw_contact_mask_3cm.npz")
    return candidates


def task_qpos_len(derived_task: str) -> int:
    traj = TASK_ROOT / derived_task / "0/trajectory_kinematic.npz"
    if not traj.is_file():
        raise FileNotFoundError(traj)
    return int(np.load(traj, allow_pickle=True)["qpos"].shape[0])


def resize_nearest(mask: np.ndarray, target_len: int) -> np.ndarray:
    if mask.shape[0] == target_len:
        return mask.astype(bool)
    idx = np.round(np.linspace(0, mask.shape[0] - 1, target_len)).astype(np.int64)
    return mask[idx].astype(bool)


def make_e104_mask(e109_case_id: str, short_id: str, object_key: str, derived_task: str) -> tuple[Path, str]:
    source = e104_proxy_for_case(e109_case_id, object_key)
    if not source.is_file():
        raise FileNotFoundError(source)
    data = np.load(source, allow_pickle=True)
    thresholds = np.asarray(data["thresholds_m"], dtype=float)
    idx = int(np.argmin(np.abs(thresholds - 0.03)))
    if abs(float(thresholds[idx]) - 0.03) > 1e-6:
        raise ValueError(f"{source} has no 0.03m threshold")
    raw_mask = np.asarray(data["masks"][..., idx], dtype=bool)
    spider_mask = resize_nearest(raw_mask, task_qpos_len(derived_task))
    out = CONTACT_MASK_ROOT / short_id / "raw_contact_mask_3cm.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        raw_contact_mask_3cm=raw_mask,
        spider_contact_mask_3cm=spider_mask,
        eval_contact_mask_3cm=spider_mask.copy(),
        persons=data["persons"],
        hands=data["hands"],
        threshold_m=np.array(0.03, dtype=np.float64),
        source_raw_contact_proxy=np.array(rel(source)),
        conversion_note=np.array("E143: nearest-neighbor resize E104 raw 3cm mask to task qpos length."),
    )
    return out, rel(source)


def resolve_mask(e109_case_id: str, short_id: str, object_key: str, derived_task: str) -> tuple[Path, str, str]:
    for candidate in explicit_mask_candidates(e109_case_id, short_id, object_key):
        if candidate.is_file():
            out = CONTACT_MASK_ROOT / short_id / "raw_contact_mask_3cm.npz"
            out.parent.mkdir(parents=True, exist_ok=True)
            if candidate.resolve() != out.resolve():
                shutil.copy2(candidate, out)
            return out, "mjwp_compatible_copied", rel(candidate)
    if object_key in {"box004", "box026"}:
        mask, source = make_e104_mask(e109_case_id, short_id, object_key, derived_task)
        return mask, "e104_raw_proxy_converted", source
    raise FileNotFoundError(f"No contact mask found for {e109_case_id}")


def validate_task(derived_task: str) -> dict[str, Any]:
    task_dir = TASK_ROOT / derived_task
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


def validate_mask(path: Path, person_idx: int) -> dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    key = "eval_contact_mask_3cm" if "eval_contact_mask_3cm" in data.files else "spider_contact_mask_3cm"
    if key not in data.files:
        raise KeyError(f"{path} missing eval_contact_mask_3cm or spider_contact_mask_3cm")
    mask = np.asarray(data[key])
    if mask.ndim != 3 or mask.shape[2] != 2:
        raise ValueError(f"{path}:{key} expected (T, person, 2), got {mask.shape}")
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


def completed_e112_raw() -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in read_csv(E112_SUMMARY):
        if row.get("ablation") != "raw_mask_ref_fk":
            continue
        e109_case = row.get("source_task") or row.get("derived_task") or row.get("case_id")
        required = [row.get("root_npz_path", ""), row.get("video_path", ""), row.get("npz_path", "")]
        if all(repo_path(path).is_file() for path in required if path):
            out[e109_case] = row
    return out


def split_for_index(index: int, run_status: str) -> str:
    if run_status == "already_done":
        return "already_done"
    splits = ["local-gpu0", "remote-gpu0", "remote-gpu1"]
    return splits[index % len(splits)]


def write_override(variant: str, derived_task: str, mask: Path, person_idx: int) -> Path:
    path = OVERRIDE_ROOT / f"core4d_{variant}.yaml"
    content = f"""# @package _global_
# Auto-generated by workspace/core4d/scripts/E143/build_raw_mask_ref_fk_24case_manifest.py.
# E143 24-case raw_mask_ref_fk sweep; variant={variant}.
defaults:
  - core4d_E089A_box021_person1_upperobj
  - _self_

task: {derived_task}

contact_hdmi_target_source: ref_fk
contact_hdmi_target_path: ""
contact_hdmi_target_uses_eef_offset: true
contact_hdmi_gain: 5.0
contact_hdmi_mask_source: "core4d_3cm"
contact_hdmi_mask_path: "{rel(mask)}"
contact_hdmi_mask_person_idx: {person_idx}
contact_hdmi_mask_time_axis: auto

hold_contact_rew_scale: 0.0
hold_contact_sigma: 0.05
hold_contact_start_eval_time: 0.0
hold_contact_end_eval_time: 0.0
hold_contact_require_ref_contact: true

video_camera: auto
"""
    path.write_text(content, encoding="utf-8")
    return path


def build() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cases = read_tsv(E109_CASES)
    short_counts: dict[str, int] = {}
    for case in cases:
        base = base_short_case_id(case["case_id"], case["object_key"])
        short_counts[base] = short_counts.get(base, 0) + 1
    methods = method_rows()
    e112_done = completed_e112_raw()
    rows: list[dict[str, Any]] = []
    preflight: list[dict[str, Any]] = []
    to_run_idx = 0
    for ordinal, case in enumerate(cases, start=1):
        e109_case_id = case["case_id"]
        pair = methods.get(e109_case_id)
        if pair is None:
            raise KeyError(f"E109 method metrics missing Omni/Spider pair for {e109_case_id}")
        short_id = disambiguated_short_case_id(e109_case_id, case["object_key"], short_counts)
        variant = f"E143_{short_id}_raw_mask_ref_fk"
        person_idx = parse_person_idx(e109_case_id)
        derived_task = derived_task_from_scene(pair.spider["scene_xml"], e109_case_id)
        mask, mask_kind, source_mask = resolve_mask(e109_case_id, short_id, case["object_key"], derived_task)
        task_validation = validate_task(derived_task)
        mask_validation = validate_mask(mask, person_idx)
        reuse = e112_done.get(e109_case_id)
        run_status = "already_done" if reuse else "to_run"
        split = split_for_index(to_run_idx, run_status)
        if run_status == "to_run":
            to_run_idx += 1
        override = write_override(variant, derived_task, mask, person_idx)
        row = {
            "ordinal": str(ordinal),
            "variant": variant,
            "e109_case_id": e109_case_id,
            "case_id": short_id,
            "object_key": case["object_key"],
            "target_variant_id": case["target_variant_id"],
            "source_task": e109_case_id,
            "derived_task": derived_task,
            "person_idx": str(person_idx),
            "split": split,
            "ablation": "raw_mask_ref_fk",
            "mask_path": rel(mask),
            "mask_kind": mask_kind,
            "source_mask_path": source_mask,
            "baseline_npz_path": pair.spider["qpos_path"],
            "baseline_run_id": pair.spider["run_id"],
            "omni_qpos_path": pair.omni["qpos_path"],
            "omni_scene_xml": pair.omni["scene_xml"],
            "spider_scene_xml": pair.spider["scene_xml"],
            "override": rel(override),
            "run_status": run_status,
            "reuse_source_exp": "E112" if reuse else "",
            "reuse_npz_path": reuse.get("root_npz_path", "") if reuse else "",
            "reuse_video_path": reuse.get("video_path", "") if reuse else "",
            "reuse_outdir_npz_path": reuse.get("npz_path", "") if reuse else "",
        }
        rows.append(row)
        preflight.append(
            {
                **row,
                **task_validation,
                **mask_validation,
                "baseline_exists": str(repo_path(row["baseline_npz_path"]).is_file()),
                "omni_qpos_exists": str(repo_path(row["omni_qpos_path"]).is_file()),
                "override_exists": str(repo_path(row["override"]).is_file()),
                "preflight_ok": "True",
            }
        )
    return rows, preflight


def main() -> None:
    SCRIPTS_ROOT.mkdir(parents=True, exist_ok=True)
    PREFLIGHT_ROOT.mkdir(parents=True, exist_ok=True)
    for path in OVERRIDE_ROOT.glob("core4d_E143_*_raw_mask_ref_fk.yaml"):
        path.unlink()
    rows, preflight = build()
    if len(rows) != 24:
        raise RuntimeError(f"E143 expected 24 rows, got {len(rows)}")
    write_tsv(VARIANTS_TSV, rows, FIELDS, "E143 raw_mask_ref_fk 24-case variants")
    preflight_fields = sorted({key for row in preflight for key in row})
    write_tsv(PREFLIGHT_TSV, preflight, preflight_fields, "E143 raw_mask_ref_fk 24-case preflight")
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["run_status"]] = counts.get(row["run_status"], 0) + 1
    split_counts: dict[str, int] = {}
    for row in rows:
        split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1
    summary = {
        "rows": len(rows),
        "run_status_counts": counts,
        "split_counts": split_counts,
        "variants_tsv": rel(VARIANTS_TSV),
        "preflight_tsv": rel(PREFLIGHT_TSV),
        "e109_case_comparison": rel(E109_CASES),
        "e109_method_metrics": rel(E109_METHODS),
        "e110_contact_metrics": rel(E110_CONTACT),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# E143 Raw-Mask Ref-FK 24-Case Manifest Summary",
        "",
        f"- variants: `{len(rows)}`",
        f"- run status counts: `{counts}`",
        f"- split counts: `{split_counts}`",
        f"- variants tsv: `{rel(VARIANTS_TSV)}`",
        f"- preflight: `{rel(PREFLIGHT_TSV)}`",
        "",
        "| case | split | status | task | mask active either | mask kind |",
        "|---|---|---|---|---:|---|",
    ]
    for row in preflight:
        lines.append(
            f"| `{row['case_id']}` | `{row['split']}` | `{row['run_status']}` | "
            f"`{row['derived_task']}` | {float(row['mask_either_active_frac']) * 100:.1f}% | `{row['mask_kind']}` |"
        )
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {rel(VARIANTS_TSV)} rows={len(rows)}")
    print(f"wrote {rel(PREFLIGHT_TSV)}")
    print(f"wrote {rel(SUMMARY_MD)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"E143 manifest build failed: {exc}", file=sys.stderr)
        raise
