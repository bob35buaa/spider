#!/usr/bin/env python3
"""Audit E135 raw-contact masks against Holosoma export time axes.

E136 is deliberately audit-only. It identifies safe bridge candidates and
blockers, but it does not write modified Holosoma motion NPZ files.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[4]
OUT_DIR = REPO / "workspace/core4d/results/E136/e135_semantic_contact_holosoma_bridge"
MANIFEST = OUT_DIR / "e136_e135_semantic_contact_bridge_manifest.tsv"
SUMMARY_JSON = OUT_DIR / "e136_e135_semantic_contact_bridge_summary.json"
SUMMARY_MD = OUT_DIR / "e136_e135_semantic_contact_bridge_summary.md"
REGISTRY = REPO / "workspace/core4d/results/E135/box021_v3_s1_raw_contact_remine/registries_combined_5cm_then_3cm/case_state_registry.tsv"
THRESHOLDS = ("3cm", "5cm")

FIELDS = [
    "case_id",
    "raw_case_id",
    "sequence_slug",
    "person",
    "person_idx",
    "threshold_label",
    "raw_contact_npz",
    "raw_contact_exists",
    "raw_contact_key",
    "raw_contact_shape",
    "raw_frames",
    "raw_to_trimmed_frame_index_status",
    "registry_raw_contact_3cm_status",
    "registry_raw_contact_5cm_status",
    "export_kind",
    "export_npz",
    "export_exists",
    "export_frame_key",
    "export_frames",
    "has_existing_object_contact",
    "existing_object_contact_source",
    "existing_object_contact_shape",
    "mapping_type",
    "mapping_status",
    "mapping_proof",
    "trim_window_json",
    "trim_start",
    "trim_end",
    "mapped_frames",
    "object_contact_shape_if_written",
    "mapped_left_active_frac",
    "mapped_right_active_frac",
    "mapped_both_active_frac",
    "semantic_bridge_candidate",
    "e107_semantic_bridge_candidate",
    "e126_e131_semantic_ready",
    "semantic_object_contact_export_written",
    "rl_ready",
    "training_launched",
    "cem_launched",
    "remote_jobs_launched",
    "failure_mode",
    "notes",
]


@dataclass(frozen=True)
class ExportSpec:
    kind: str
    path: str
    mapping_type: str
    trim_window: str = ""


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    raw_case_id: str
    sequence_slug: str
    person: str
    person_idx: int
    raw_npz: str
    exports: tuple[ExportSpec, ...]


E107_ROOT = "workspace/core4d/results/E107/s6_downstream/rl_export/full_cem_omnirt/results/omnirt_v1"
E126_ROOT = "workspace/core4d/results/E126/holosoma_fragment_adapter_preflight/exports"
E131_ROOT = "workspace/core4d/results/E131/holosoma_object_contact_proxy/exports"
E135_RAW_ROOT = "workspace/core4d/results/E135/box021_v3_s1_raw_contact_remine/s1_raw_contact/raw_contact/per_sequence"


CASES = (
    CaseSpec(
        case_id="box021_035_p1",
        raw_case_id="box021_20231011_035_p1",
        sequence_slug="20231011_035_box021",
        person="person1",
        person_idx=0,
        raw_npz=f"{E135_RAW_ROOT}/20231011_035_box021/raw_contact_proxy.npz",
        exports=(
            ExportSpec(
                "E107_retargeted_untrimmed",
                f"{E107_ROOT}/holosoma_e107_full_cem_omnirt_box021_20231011_035_p1/retargeted/20231011-035-person1-Box021_with_obj_original.npz",
                "direct_raw_axis",
            ),
            ExportSpec(
                "E107_trimmed",
                f"{E107_ROOT}/holosoma_e107_full_cem_omnirt_box021_20231011_035_p1/trimmed/20231011-035-person1-Box021_with_obj_original.npz",
                "trim_window_slice",
                f"{E107_ROOT}/holosoma_e107_full_cem_omnirt_box021_20231011_035_p1/trim_window.json",
            ),
            ExportSpec(
                "E126_fragment_export",
                f"{E126_ROOT}/E126_box021_035_p1_with_partner_box021_035_p2_mj_w_obj_w_partner.npz",
                "blocked_missing_raw_window",
            ),
            ExportSpec(
                "E131_geometry_proxy_export",
                f"{E131_ROOT}/E126_box021_035_p1_with_partner_box021_035_p2_object_contact_proxy5cm.npz",
                "blocked_missing_raw_window",
            ),
        ),
    ),
    CaseSpec(
        case_id="box021_035_p2",
        raw_case_id="box021_20231011_035_p2",
        sequence_slug="20231011_035_box021",
        person="person2",
        person_idx=1,
        raw_npz=f"{E135_RAW_ROOT}/20231011_035_box021/raw_contact_proxy.npz",
        exports=(
            ExportSpec(
                "E107_retargeted_untrimmed",
                f"{E107_ROOT}/holosoma_e107_full_cem_omnirt_box021_20231011_035_p2/retargeted/20231011-035-person2-Box021_with_obj_original.npz",
                "direct_raw_axis",
            ),
            ExportSpec(
                "E107_trimmed",
                f"{E107_ROOT}/holosoma_e107_full_cem_omnirt_box021_20231011_035_p2/trimmed/20231011-035-person2-Box021_with_obj_original.npz",
                "trim_window_slice",
                f"{E107_ROOT}/holosoma_e107_full_cem_omnirt_box021_20231011_035_p2/trim_window.json",
            ),
            ExportSpec(
                "E126_fragment_export",
                f"{E126_ROOT}/E126_box021_035_p2_with_partner_box021_035_p1_mj_w_obj_w_partner.npz",
                "blocked_missing_raw_window",
            ),
            ExportSpec(
                "E131_geometry_proxy_export",
                f"{E131_ROOT}/E126_box021_035_p2_with_partner_box021_035_p1_object_contact_proxy5cm.npz",
                "blocked_missing_raw_window",
            ),
        ),
    ),
    CaseSpec(
        case_id="box021_029_p1",
        raw_case_id="box021_20231018_029_p1",
        sequence_slug="20231018_029_box021",
        person="person1",
        person_idx=0,
        raw_npz=f"{E135_RAW_ROOT}/20231018_029_box021/raw_contact_proxy.npz",
        exports=(
            ExportSpec(
                "E107_retargeted_untrimmed",
                f"{E107_ROOT}/holosoma_e107_full_cem_omnirt_box021_20231018_029_p1/retargeted/20231018-029-person1-Box021_with_obj_original.npz",
                "direct_raw_axis",
            ),
            ExportSpec(
                "E107_trimmed",
                f"{E107_ROOT}/holosoma_e107_full_cem_omnirt_box021_20231018_029_p1/trimmed/20231018-029-person1-Box021_with_obj_original.npz",
                "trim_window_slice",
                f"{E107_ROOT}/holosoma_e107_full_cem_omnirt_box021_20231018_029_p1/trim_window.json",
            ),
        ),
    ),
    CaseSpec(
        case_id="box021_029_p2",
        raw_case_id="box021_20231018_029_p2",
        sequence_slug="20231018_029_box021",
        person="person2",
        person_idx=1,
        raw_npz=f"{E135_RAW_ROOT}/20231018_029_box021/raw_contact_proxy.npz",
        exports=(
            ExportSpec(
                "E107_retargeted_untrimmed",
                f"{E107_ROOT}/holosoma_e107_full_cem_omnirt_box021_20231018_029_p2/retargeted/20231018-029-person2-Box021_with_obj_original.npz",
                "direct_raw_axis",
            ),
            ExportSpec(
                "E107_trimmed",
                f"{E107_ROOT}/holosoma_e107_full_cem_omnirt_box021_20231018_029_p2/trimmed/20231018-029-person2-Box021_with_obj_original.npz",
                "trim_window_slice",
                f"{E107_ROOT}/holosoma_e107_full_cem_omnirt_box021_20231018_029_p2/trim_window.json",
            ),
        ),
    ),
)


def repo_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else REPO / path


def rel(path: str | Path) -> str:
    path_obj = Path(path)
    try:
        return str(path_obj.resolve().relative_to(REPO))
    except ValueError:
        return str(path_obj)


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def scalar_text(value: Any) -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        item = arr.item()
        if isinstance(item, bytes):
            return item.decode("utf-8", errors="replace")
        return str(item)
    items = arr.reshape(-1).tolist()
    return ",".join(str(x.decode("utf-8", errors="replace") if isinstance(x, bytes) else x) for x in items)


def registry_by_case() -> dict[str, dict[str, str]]:
    return {row.get("case_id", ""): row for row in read_tsv(REGISTRY)}


def frame_summary(path: Path) -> dict[str, Any]:
    out = {
        "export_exists": str(path.is_file()).lower(),
        "export_frame_key": "",
        "export_frames": "",
        "has_existing_object_contact": "false",
        "existing_object_contact_source": "",
        "existing_object_contact_shape": "",
    }
    if not path.is_file():
        return out
    with np.load(path, allow_pickle=True) as data:
        for key in ("joint_pos", "qpos", "body_pos_w", "object_pos_w", "object_pos"):
            if key in data:
                arr = np.asarray(data[key])
                if arr.ndim >= 1:
                    out["export_frame_key"] = key
                    out["export_frames"] = int(arr.shape[0])
                    break
        if "object_contact" in data:
            contact = np.asarray(data["object_contact"]).astype(bool)
            out["has_existing_object_contact"] = "true"
            out["existing_object_contact_shape"] = "x".join(map(str, contact.shape))
        if "object_contact_source" in data:
            out["existing_object_contact_source"] = scalar_text(data["object_contact_source"])
    return out


def load_trim_window(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def fraction(mask: np.ndarray) -> str:
    if mask.size == 0:
        return ""
    return f"{float(mask.mean()):.6f}"


def mapping_result(export: ExportSpec, export_frames: int | str, raw_frames: int, trim: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "mapping_status": "blocked",
        "mapping_proof": "",
        "trim_start": "",
        "trim_end": "",
        "mapped_slice": None,
        "failure_mode": "",
        "notes": "",
    }
    try:
        export_frame_count = int(export_frames)
    except (TypeError, ValueError):
        out["failure_mode"] = "export_frame_count_missing"
        out["notes"] = "cannot map without export frame count"
        return out

    if export.mapping_type == "direct_raw_axis":
        if export_frame_count == raw_frames:
            out.update(
                {
                    "mapping_status": "pass",
                    "mapping_proof": f"export_frames={export_frame_count} equals E135 raw_frames={raw_frames}",
                    "trim_start": 0,
                    "trim_end": raw_frames,
                    "mapped_slice": slice(0, raw_frames),
                }
            )
        else:
            out["failure_mode"] = "direct_raw_frame_mismatch"
            out["notes"] = "direct raw-axis export frame count does not match E135 raw frames"
        return out

    if export.mapping_type == "trim_window_slice":
        required = ("untrimmed_frames", "trimmed_frames", "trim_start", "trim_end", "trim_frames")
        if not all(key in trim for key in required):
            out["failure_mode"] = "trim_window_missing_fields"
            out["notes"] = "trim-window slice requires explicit untrimmed/trimmed frame metadata"
            return out
        trim_start = int(trim["trim_start"])
        trim_end = int(trim["trim_end"])
        trim_frames = int(trim["trim_frames"])
        conditions = {
            "untrimmed_frames_match_raw": int(trim["untrimmed_frames"]) == raw_frames,
            "trimmed_frames_match_export": int(trim["trimmed_frames"]) == export_frame_count,
            "trim_span_match_export": trim_end - trim_start == export_frame_count,
            "trim_frames_match_export": trim_frames == export_frame_count,
            "trim_bounds_valid": 0 <= trim_start < trim_end <= raw_frames,
        }
        if all(conditions.values()):
            out.update(
                {
                    "mapping_status": "pass",
                    "mapping_proof": ";".join(f"{key}={value}" for key, value in conditions.items()),
                    "trim_start": trim_start,
                    "trim_end": trim_end,
                    "mapped_slice": slice(trim_start, trim_end),
                }
            )
        else:
            out["failure_mode"] = "trim_window_incompatible"
            out["mapping_proof"] = ";".join(f"{key}={value}" for key, value in conditions.items())
            out["notes"] = "trim window does not prove a safe raw-to-export slice"
        return out

    out["failure_mode"] = "fragment_raw_window_missing"
    out["notes"] = "E126/E131 fragment export has no explicit raw-window mapping; shape-only bridge is disallowed"
    return out


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    registry = registry_by_case()
    for spec in CASES:
        raw_path = repo_path(spec.raw_npz)
        reg = registry.get(spec.raw_case_id, {})
        raw_exists = raw_path.is_file()
        raw_data: dict[str, Any] = {}
        if raw_exists:
            with np.load(raw_path, allow_pickle=True) as data:
                raw_data = {key: data[key] for key in data.files}
        raw_to_trimmed_status = "missing"
        if "raw_to_trimmed_frame_index" in raw_data:
            mapping = np.asarray(raw_data["raw_to_trimmed_frame_index"])
            raw_to_trimmed_status = "all_missing" if mapping.size and np.all(mapping < 0) else "present"

        for label in THRESHOLDS:
            raw_key = f"raw_contact_mask_{label}"
            raw_mask = np.asarray(raw_data[raw_key]).astype(bool) if raw_key in raw_data else np.zeros((0, 2, 2), dtype=bool)
            raw_frames = int(raw_mask.shape[0]) if raw_mask.ndim >= 1 else 0
            person_mask = raw_mask[:, spec.person_idx, :] if raw_mask.ndim == 3 and raw_mask.shape[1] > spec.person_idx else np.zeros((0, 2), dtype=bool)
            for export in spec.exports:
                export_path = repo_path(export.path)
                export_summary = frame_summary(export_path)
                trim_path = repo_path(export.trim_window) if export.trim_window else Path("")
                trim = load_trim_window(trim_path) if export.trim_window else {}
                mapping = mapping_result(export, export_summary["export_frames"], raw_frames, trim)
                mapped_slice = mapping.pop("mapped_slice")
                mapped = person_mask[mapped_slice] if mapped_slice is not None else np.zeros((0, 2), dtype=bool)
                mapped_frames = int(mapped.shape[0])
                bridge = (
                    raw_exists
                    and reg.get(f"raw_contact_{label}_status") == "pass"
                    and export_summary["export_exists"] == "true"
                    and mapping["mapping_status"] == "pass"
                    and export.kind.startswith("E107_")
                )
                e126_e131 = export.kind in {"E126_fragment_export", "E131_geometry_proxy_export"}
                failure_mode = mapping.get("failure_mode", "")
                notes = mapping.get("notes", "")
                if bridge:
                    failure_mode = ""
                    notes = "safe mapping candidate; audit-only, no semantic object_contact export written"
                elif not raw_exists:
                    failure_mode = "raw_contact_npz_missing"
                elif reg.get(f"raw_contact_{label}_status") != "pass":
                    failure_mode = "e135_raw_contact_not_pass"
                elif export_summary["export_exists"] != "true":
                    failure_mode = "export_missing"

                row: dict[str, Any] = {
                    "case_id": spec.case_id,
                    "raw_case_id": spec.raw_case_id,
                    "sequence_slug": spec.sequence_slug,
                    "person": spec.person,
                    "person_idx": spec.person_idx,
                    "threshold_label": label,
                    "raw_contact_npz": rel(raw_path),
                    "raw_contact_exists": str(raw_exists).lower(),
                    "raw_contact_key": raw_key,
                    "raw_contact_shape": "x".join(map(str, raw_mask.shape)) if raw_mask.size else "",
                    "raw_frames": raw_frames,
                    "raw_to_trimmed_frame_index_status": raw_to_trimmed_status,
                    "registry_raw_contact_3cm_status": reg.get("raw_contact_3cm_status", ""),
                    "registry_raw_contact_5cm_status": reg.get("raw_contact_5cm_status", ""),
                    "export_kind": export.kind,
                    "export_npz": export.path,
                    "mapping_type": export.mapping_type,
                    "mapping_status": mapping["mapping_status"],
                    "mapping_proof": mapping.get("mapping_proof", ""),
                    "trim_window_json": export.trim_window,
                    "trim_start": mapping.get("trim_start", ""),
                    "trim_end": mapping.get("trim_end", ""),
                    "mapped_frames": mapped_frames if mapped_frames else "",
                    "object_contact_shape_if_written": f"{mapped_frames}x2" if mapped_frames else "",
                    "mapped_left_active_frac": fraction(mapped[:, 0]) if mapped_frames else "",
                    "mapped_right_active_frac": fraction(mapped[:, 1]) if mapped_frames else "",
                    "mapped_both_active_frac": fraction(np.logical_and(mapped[:, 0], mapped[:, 1])) if mapped_frames else "",
                    "semantic_bridge_candidate": str(bridge).lower(),
                    "e107_semantic_bridge_candidate": str(bridge and export.kind.startswith("E107_")).lower(),
                    "e126_e131_semantic_ready": str(bridge and e126_e131).lower(),
                    "semantic_object_contact_export_written": "false",
                    "rl_ready": "false",
                    "training_launched": "false",
                    "cem_launched": "false",
                    "remote_jobs_launched": "false",
                    "failure_mode": failure_mode,
                    "notes": notes,
                }
                row.update(export_summary)
                rows.append(row)
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "status": "pass",
        "audit_rows": len(rows),
        "raw_contact_available_rows": sum(1 for row in rows if row["raw_contact_exists"] == "true"),
        "direct_raw_axis_candidate_rows": sum(
            1 for row in rows if row["mapping_type"] == "direct_raw_axis" and row["semantic_bridge_candidate"] == "true"
        ),
        "trim_window_slice_candidate_rows": sum(
            1 for row in rows if row["mapping_type"] == "trim_window_slice" and row["semantic_bridge_candidate"] == "true"
        ),
        "semantic_bridge_candidate_rows": sum(1 for row in rows if row["semantic_bridge_candidate"] == "true"),
        "e107_semantic_bridge_candidate_rows": sum(1 for row in rows if row["e107_semantic_bridge_candidate"] == "true"),
        "e126_e131_semantic_ready_rows": sum(1 for row in rows if row["e126_e131_semantic_ready"] == "true"),
        "fragment_mapping_blocked_rows": sum(1 for row in rows if row["failure_mode"] == "fragment_raw_window_missing"),
        "structural_proxy_rows": sum(1 for row in rows if row["has_existing_object_contact"] == "true"),
        "semantic_object_contact_exports_written": 0,
        "training_launched": False,
        "cem_launched": False,
        "remote_jobs_launched": False,
        "notes": (
            "E136 proves E107 raw/trim-window bridge candidates only. "
            "E126/E131 fragment exports remain blocked because no raw-window mapping exists."
        ),
    }


def markdown_summary(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# E136 E135 semantic contact to Holosoma bridge audit",
        "",
        f"- audit rows: `{summary['audit_rows']}`",
        f"- semantic bridge candidate rows: `{summary['semantic_bridge_candidate_rows']}`",
        f"- direct raw-axis candidates: `{summary['direct_raw_axis_candidate_rows']}`",
        f"- trim-window slice candidates: `{summary['trim_window_slice_candidate_rows']}`",
        f"- E126/E131 semantic-ready rows: `{summary['e126_e131_semantic_ready_rows']}`",
        f"- semantic object_contact exports written: `{summary['semantic_object_contact_exports_written']}`",
        f"- training/CEM/remote launched: `{str(summary['training_launched']).lower()}`/"
        f"`{str(summary['cem_launched']).lower()}`/`{str(summary['remote_jobs_launched']).lower()}`",
        "",
        "## Rows",
        "",
        "| case | threshold | export | raw/export/mapped frames | mapping | candidate | failure |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['case_id']}` | `{row['threshold_label']}` | `{row['export_kind']}` | "
            f"{row['raw_frames']}/{row['export_frames']}/{row['mapped_frames']} | "
            f"`{row['mapping_type']}:{row['mapping_status']}` | "
            f"`{row['semantic_bridge_candidate']}` | `{row['failure_mode']}` |"
        )
    lines.extend(
        [
            "",
            "Interpretation: E107 retargeted-untrimmed rows have direct raw-axis candidates, and E107 trimmed rows have explicit trim-window slice candidates. E126/E131 fragment rows remain blocked because the adapter policy is fragment-only and records no raw-window mapping; E131's existing `object_contact` is still structural proxy evidence, not semantic raw contact.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    rows = build_rows()
    summary = summarize(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_tsv(MANIFEST, rows)
    write_json(SUMMARY_JSON, summary)
    SUMMARY_MD.write_text(markdown_summary(summary, rows), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
