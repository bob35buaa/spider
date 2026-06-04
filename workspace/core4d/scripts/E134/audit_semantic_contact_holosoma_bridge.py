#!/usr/bin/env python3
"""Audit whether semantic contact masks can bridge into Holosoma exports.

This audit is deliberately read-only. It does not resize masks, does not write
Holosoma motion exports, and does not treat E131 geometry proxies as semantic
contact labels.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[4]
OUT_DIR = REPO / "workspace/core4d/results/E134/semantic_contact_holosoma_bridge_audit"
MANIFEST = OUT_DIR / "e134_semantic_contact_holosoma_bridge_manifest.tsv"
SUMMARY_JSON = OUT_DIR / "e134_semantic_contact_holosoma_bridge_summary.json"
SUMMARY_MD = OUT_DIR / "e134_semantic_contact_holosoma_bridge_summary.md"
REGISTRY_SOURCES = (
    REPO / "workspace/core4d/data_construction_v3/existing_cases.tsv",
    REPO / "workspace/core4d/results/E111/contact_chain_smoke/E111_contact_chain_smoke/registries/case_state_registry.tsv",
)


FIELDS = [
    "case_id",
    "role",
    "person",
    "person_idx",
    "semantic_mask_npz",
    "semantic_mask_provenance",
    "semantic_mask_exists",
    "mask_keys_available",
    "preferred_mask_key",
    "preferred_mask_shape",
    "preferred_mask_frames",
    "preferred_mask_person_axis",
    "preferred_left_active_frac",
    "preferred_right_active_frac",
    "preferred_either_active_frac",
    "raw_mask_frames",
    "spider_mask_frames",
    "eval_mask_frames",
    "has_eval_ref_idx",
    "has_eval_raw_idx",
    "trim_start",
    "target_npz",
    "target_exists",
    "target_frames",
    "target_active_frac",
    "v3_registry_case_id",
    "v3_registry_source",
    "v3_raw_contact_3cm_status",
    "v3_raw_contact_5cm_status",
    "v3_contact_mask_npz",
    "v3_raw_contact_artifact_npz",
    "holosoma_export_npz",
    "holosoma_export_kind",
    "holosoma_export_exists",
    "holosoma_frames",
    "holosoma_frame_key",
    "has_existing_object_contact",
    "existing_object_contact_source",
    "existing_object_contact_total",
    "existing_object_contact_shape",
    "timeline_exact_match",
    "timeline_matching_mask_keys",
    "semantic_bridge_candidate",
    "e126_e131_semantic_ready",
    "structural_proxy_ready",
    "semantic_holosoma_mask_ready",
    "rl_ready",
    "training_launched",
    "cem_launched",
    "remote_jobs_launched",
    "failure_mode",
    "notes",
]


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    role: str
    person: str
    person_idx: int
    semantic_mask_npz: str
    semantic_mask_provenance: str
    target_npz: str
    v3_registry_case_id: str
    exports: tuple[tuple[str, str], ...]


CASES = [
    CaseSpec(
        case_id="box021_035_p1",
        role="fragment_release_guard",
        person="person1",
        person_idx=0,
        semantic_mask_npz="workspace/core4d/results/E079/contact_masks/box021_person1/raw_contact_mask_3cm.npz",
        semantic_mask_provenance="legacy_box021_person1_mask_used_by_E112",
        target_npz="",
        v3_registry_case_id="box021_20231011_035_p1",
        exports=(
            (
                "E126_fragment_export",
                "workspace/core4d/results/E126/holosoma_fragment_adapter_preflight/exports/E126_box021_035_p1_with_partner_box021_035_p2_mj_w_obj_w_partner.npz",
            ),
            (
                "E131_geometry_proxy_export",
                "workspace/core4d/results/E131/holosoma_object_contact_proxy/exports/E126_box021_035_p1_with_partner_box021_035_p2_object_contact_proxy5cm.npz",
            ),
            (
                "E107_full_cem_omnirt_trimmed",
                "workspace/core4d/results/E107/s6_downstream/rl_export/full_cem_omnirt/results/omnirt_v1/holosoma_e107_full_cem_omnirt_box021_20231011_035_p1/trimmed/20231011-035-person1-Box021_with_obj_original.npz",
            ),
        ),
    ),
    CaseSpec(
        case_id="box021_035_p2",
        role="fragment_lowerbody_risk",
        person="person2",
        person_idx=1,
        semantic_mask_npz="workspace/core4d/results/E082/contact_masks/d003_box021_20231011_035_p2/raw_contact_mask_3cm.npz",
        semantic_mask_provenance="case_specific_E082_raw_contact_mask",
        target_npz="workspace/core4d/results/E100/fingertip_targets/d003_box021_20231011_035_p2/spider_contact_target_object_local.npz",
        v3_registry_case_id="box021_20231011_035_p2",
        exports=(
            (
                "E126_fragment_export",
                "workspace/core4d/results/E126/holosoma_fragment_adapter_preflight/exports/E126_box021_035_p2_with_partner_box021_035_p1_mj_w_obj_w_partner.npz",
            ),
            (
                "E131_geometry_proxy_export",
                "workspace/core4d/results/E131/holosoma_object_contact_proxy/exports/E126_box021_035_p2_with_partner_box021_035_p1_object_contact_proxy5cm.npz",
            ),
            (
                "E107_full_cem_omnirt_trimmed",
                "workspace/core4d/results/E107/s6_downstream/rl_export/full_cem_omnirt/results/omnirt_v1/holosoma_e107_full_cem_omnirt_box021_20231011_035_p2/trimmed/20231011-035-person2-Box021_with_obj_original.npz",
            ),
        ),
    ),
    CaseSpec(
        case_id="box021_029_p2",
        role="main_lowerbody_aware_contact",
        person="person2",
        person_idx=1,
        semantic_mask_npz="workspace/core4d/results/E084/contact_masks/d003_box021_20231018_029_p2/raw_contact_mask_3cm.npz",
        semantic_mask_provenance="case_specific_E084_raw_contact_mask",
        target_npz="workspace/core4d/results/E100/fingertip_targets/d003_box021_20231018_029_p2/spider_contact_target_object_local.npz",
        v3_registry_case_id="box021_20231018_029_p2",
        exports=(
            (
                "E107_full_cem_omnirt_trimmed",
                "workspace/core4d/results/E107/s6_downstream/rl_export/full_cem_omnirt/results/omnirt_v1/holosoma_e107_full_cem_omnirt_box021_20231018_029_p2/trimmed/20231018-029-person2-Box021_with_obj_original.npz",
            ),
        ),
    ),
]


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


def repo_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else REPO / path


def scalar_text(value: Any) -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        item = arr.item()
        if isinstance(item, bytes):
            return item.decode("utf-8", errors="replace")
        return str(item)
    return ",".join(str(x.decode("utf-8", errors="replace") if isinstance(x, bytes) else x) for x in arr.reshape(-1).tolist())


def fmt_frac(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def load_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader((line for line in f if not line.startswith("#")), delimiter="\t"))


def registry_summary(case_id: str) -> dict[str, str]:
    out = {
        "v3_registry_case_id": case_id,
        "v3_registry_source": "",
        "v3_raw_contact_3cm_status": "",
        "v3_raw_contact_5cm_status": "",
        "v3_contact_mask_npz": "",
        "v3_raw_contact_artifact_npz": "",
    }
    for path in REGISTRY_SOURCES:
        for row in read_tsv(path):
            if row.get("case_id") != case_id:
                continue
            out.update(
                {
                    "v3_registry_source": rel(path),
                    "v3_raw_contact_3cm_status": row.get("raw_contact_3cm_status", ""),
                    "v3_raw_contact_5cm_status": row.get("raw_contact_5cm_status", ""),
                    "v3_contact_mask_npz": row.get("contact_mask_npz", ""),
                    "v3_raw_contact_artifact_npz": row.get("raw_contact_artifact_npz", ""),
                }
            )
            return out
    return out


def mask_summary(path: Path, person_idx: int) -> dict[str, Any]:
    out: dict[str, Any] = {
        "semantic_mask_exists": str(path.is_file()).lower(),
        "mask_keys_available": "",
        "preferred_mask_key": "",
        "preferred_mask_shape": "",
        "preferred_mask_frames": "",
        "preferred_mask_person_axis": "",
        "preferred_left_active_frac": "",
        "preferred_right_active_frac": "",
        "preferred_either_active_frac": "",
        "raw_mask_frames": "",
        "spider_mask_frames": "",
        "eval_mask_frames": "",
        "has_eval_ref_idx": "false",
        "has_eval_raw_idx": "false",
        "trim_start": "",
        "mask_error": "",
    }
    if not path.is_file():
        out["mask_error"] = "semantic_mask_missing"
        return out
    data = load_npz(path)
    keys = sorted(data)
    out["mask_keys_available"] = ",".join(keys)
    out["has_eval_ref_idx"] = str("eval_ref_idx" in data).lower()
    out["has_eval_raw_idx"] = str("eval_raw_idx" in data).lower()
    if "trim_start" in data:
        out["trim_start"] = scalar_text(data["trim_start"])
    for key in ("raw_contact_mask_3cm", "spider_contact_mask_3cm", "eval_contact_mask_3cm"):
        if key in data:
            out[key.split("_", 1)[0] + "_mask_frames"] = int(np.asarray(data[key]).shape[0])
    preferred = ""
    for key in ("eval_contact_mask_3cm", "spider_contact_mask_3cm", "raw_contact_mask_3cm"):
        if key in data:
            preferred = key
            break
    if not preferred:
        out["mask_error"] = "semantic_mask_key_missing"
        return out
    mask = np.asarray(data[preferred]).astype(bool)
    out["preferred_mask_key"] = preferred
    out["preferred_mask_shape"] = "x".join(str(v) for v in mask.shape)
    out["preferred_mask_frames"] = int(mask.shape[0])
    if mask.ndim == 3 and mask.shape[2] >= 2:
        out["preferred_mask_person_axis"] = int(mask.shape[1])
        if person_idx >= mask.shape[1]:
            out["mask_error"] = "person_idx_out_of_bounds"
            return out
        hand_mask = mask[:, person_idx, :2]
    elif mask.ndim == 2 and mask.shape[1] >= 2:
        out["preferred_mask_person_axis"] = "none"
        hand_mask = mask[:, :2]
    else:
        out["mask_error"] = f"semantic_mask_shape_unsupported:{list(mask.shape)}"
        return out
    left = hand_mask[:, 0]
    right = hand_mask[:, 1]
    out["preferred_left_active_frac"] = fmt_frac(float(np.mean(left)))
    out["preferred_right_active_frac"] = fmt_frac(float(np.mean(right)))
    out["preferred_either_active_frac"] = fmt_frac(float(np.mean(left | right)))
    return out


def target_summary(path: Path) -> dict[str, Any]:
    out = {"target_exists": str(path.is_file()).lower(), "target_frames": "", "target_active_frac": ""}
    if not path.is_file():
        return out
    data = load_npz(path)
    target_key = "spider_contact_target_object_local" if "spider_contact_target_object_local" in data else ""
    if target_key:
        out["target_frames"] = int(np.asarray(data[target_key]).shape[0])
    if "active" in data:
        active = np.asarray(data["active"]).astype(bool)
        out["target_active_frac"] = fmt_frac(float(np.mean(active)))
    return out


def export_summary(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {
        "holosoma_export_exists": str(path.is_file()).lower(),
        "holosoma_frames": "",
        "holosoma_frame_key": "",
        "has_existing_object_contact": "false",
        "existing_object_contact_source": "",
        "existing_object_contact_total": "",
        "existing_object_contact_shape": "",
        "export_error": "",
    }
    if not path.is_file():
        out["export_error"] = "holosoma_export_missing"
        return out
    data = load_npz(path)
    for key in ("joint_pos", "qpos", "body_pos_w", "object_pos_w", "object_pos"):
        if key in data:
            arr = np.asarray(data[key])
            if arr.ndim >= 1:
                out["holosoma_frames"] = int(arr.shape[0])
                out["holosoma_frame_key"] = key
                break
    if not out["holosoma_frames"]:
        out["export_error"] = "frame_key_missing"
    if "object_contact" in data:
        contact = np.asarray(data["object_contact"]).astype(bool)
        out["has_existing_object_contact"] = "true"
        out["existing_object_contact_total"] = int(contact.sum())
        out["existing_object_contact_shape"] = "x".join(str(v) for v in contact.shape)
    if "object_contact_source" in data:
        out["existing_object_contact_source"] = scalar_text(data["object_contact_source"])
    return out


def matching_mask_keys(mask: dict[str, Any], export_frames: str) -> list[str]:
    if not export_frames:
        return []
    matches: list[str] = []
    for key, field in (
        ("raw_contact_mask_3cm", "raw_mask_frames"),
        ("spider_contact_mask_3cm", "spider_mask_frames"),
        ("eval_contact_mask_3cm", "eval_mask_frames"),
    ):
        if str(mask.get(field, "")) == str(export_frames):
            matches.append(key)
    return matches


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in CASES:
        mask_path = repo_path(spec.semantic_mask_npz)
        target_path = repo_path(spec.target_npz) if spec.target_npz else Path("")
        mask = mask_summary(mask_path, spec.person_idx)
        target = target_summary(target_path) if spec.target_npz else {"target_exists": "false", "target_frames": "", "target_active_frac": ""}
        registry = registry_summary(spec.v3_registry_case_id)
        for export_kind, export_text in spec.exports:
            export_path = repo_path(export_text)
            export = export_summary(export_path)
            failure_mode = ""
            notes = "read-only audit; no mask export generated"
            mask_frames = str(mask.get("preferred_mask_frames", ""))
            export_frames = str(export.get("holosoma_frames", ""))
            matches = matching_mask_keys(mask, export_frames)
            timeline_exact = bool(matches)
            semantic_bridge = (
                mask.get("semantic_mask_exists") == "true"
                and not mask.get("mask_error")
                and registry.get("v3_raw_contact_3cm_status") in {"pass", "review"}
                and bool(registry.get("v3_raw_contact_artifact_npz"))
                and export.get("holosoma_export_exists") == "true"
                and not export.get("export_error")
                and timeline_exact
            )
            if mask.get("mask_error"):
                failure_mode = str(mask["mask_error"])
            elif export.get("export_error"):
                failure_mode = str(export["export_error"])
            elif registry.get("v3_raw_contact_3cm_status") not in {"pass", "review"} or not registry.get("v3_raw_contact_artifact_npz"):
                failure_mode = "v3_semantic_contact_not_ready"
                notes = "legacy mask may exist, but v3 S1 contact chain is not ready for this case"
            elif not timeline_exact:
                failure_mode = "timeline_bridge_blocked"
                notes = "semantic mask exists but time axis does not exactly match this Holosoma export"
            e126_e131 = export_kind in {"E126_fragment_export", "E131_geometry_proxy_export"}
            e126_e131_ready = semantic_bridge and e126_e131
            row: dict[str, Any] = {
                "case_id": spec.case_id,
                "role": spec.role,
                "person": spec.person,
                "person_idx": spec.person_idx,
                "semantic_mask_npz": spec.semantic_mask_npz,
                "semantic_mask_provenance": spec.semantic_mask_provenance,
                "target_npz": spec.target_npz,
                "holosoma_export_npz": export_text,
                "holosoma_export_kind": export_kind,
                "timeline_exact_match": str(timeline_exact).lower(),
                "timeline_matching_mask_keys": ",".join(matches),
                "semantic_bridge_candidate": str(semantic_bridge).lower(),
                "e126_e131_semantic_ready": str(e126_e131_ready).lower(),
                "structural_proxy_ready": str(export.get("has_existing_object_contact") == "true").lower(),
                "semantic_holosoma_mask_ready": str(semantic_bridge).lower(),
                "rl_ready": "false",
                "training_launched": "false",
                "cem_launched": "false",
                "remote_jobs_launched": "false",
                "failure_mode": failure_mode,
                "notes": notes,
            }
            row.update(mask)
            row.update(target)
            row.update(registry)
            row.update(export)
            row.pop("mask_error", None)
            row.pop("export_error", None)
            rows.append(row)
    return rows


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def write_summary(rows: list[dict[str, Any]]) -> None:
    semantic_candidates = [row for row in rows if row["semantic_bridge_candidate"] == "true"]
    e126_e131_ready = [row for row in rows if row["e126_e131_semantic_ready"] == "true"]
    blocked = [row for row in rows if row["failure_mode"]]
    summary = {
        "experiment": "E134",
        "status": "pass",
        "rows": len(rows),
        "semantic_mask_available_rows": sum(1 for row in rows if row["semantic_mask_exists"] == "true"),
        "v3_semantic_contact_ready_rows": sum(
            1
            for row in rows
            if row["v3_raw_contact_3cm_status"] in {"pass", "review"} and bool(row["v3_raw_contact_artifact_npz"])
        ),
        "timeline_exact_match_rows": sum(1 for row in rows if row["timeline_exact_match"] == "true"),
        "semantic_bridge_candidate_rows": len(semantic_candidates),
        "e126_e131_semantic_ready_rows": len(e126_e131_ready),
        "semantic_holosoma_mask_ready_rows": len(semantic_candidates),
        "structural_proxy_ready_rows": sum(1 for row in rows if row["structural_proxy_ready"] == "true"),
        "blocked_rows": len(blocked),
        "rl_ready_rows": 0,
        "training_launched": False,
        "cem_launched": False,
        "remote_jobs_launched": False,
        "notes": [
            "E134 is read-only and generates no Holosoma semantic object_contact export.",
            "Exact time-axis match is required; resize/pad/nearest mapping remains disallowed.",
            "E131 geometry proxy rows remain structural only, not semantic contact evidence.",
        ],
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# E134 Semantic Contact to Holosoma Bridge Audit Summary",
        "",
        f"- rows: `{summary['rows']}`",
        f"- semantic mask available rows: `{summary['semantic_mask_available_rows']}`",
        f"- v3 semantic contact ready rows: `{summary['v3_semantic_contact_ready_rows']}`",
        f"- timeline exact match rows: `{summary['timeline_exact_match_rows']}`",
        f"- semantic bridge candidate rows: `{summary['semantic_bridge_candidate_rows']}`",
        f"- E126/E131 semantic-ready rows: `{summary['e126_e131_semantic_ready_rows']}`",
        f"- structural proxy ready rows: `{summary['structural_proxy_ready_rows']}`",
        "- RL-ready rows: `0`",
        "- training launched: `false`",
        "- CEM launched: `false`",
        "- remote jobs launched: `false`",
        f"- status: `{summary['status']}`",
        "",
        "| case | export kind | v3 3cm | preferred frames | export frames | matching mask keys | semantic candidate | E126/E131 ready | existing object_contact | failure |",
        "|---|---|---|---:|---:|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['case_id']}` | `{row['holosoma_export_kind']}` | "
            f"`{row['v3_raw_contact_3cm_status']}` | "
            f"{row['preferred_mask_frames']} | {row['holosoma_frames']} | "
            f"`{row['timeline_matching_mask_keys']}` | `{row['semantic_bridge_candidate']}` | "
            f"`{row['e126_e131_semantic_ready']}` | `{row['has_existing_object_contact']}` | "
            f"{row['failure_mode']} |"
        )
    lines.extend(
        [
            "",
            "Interpretation: E134 only audits bridge compatibility. A semantic Holosoma `object_contact` export still requires an exact source-to-export time-axis mapping for the target motion; E134 does not launch CEM/PPO/RL.",
            "",
        ]
    )
    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    write_tsv(MANIFEST, rows)
    write_summary(rows)
    print(f"wrote {SUMMARY_MD} rows={len(rows)}")


if __name__ == "__main__":
    main()
