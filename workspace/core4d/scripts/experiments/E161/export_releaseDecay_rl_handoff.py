#!/usr/bin/env python3
"""Export E161 releaseDecay clean8 rows as S6 RL handoff inputs."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[5]
RESULT_ROOT = REPO / "workspace/core4d/results/E161/releaseDecay_rl_export"
E161_VARIANTS = REPO / "workspace/core4d/scripts/experiments/E161/variants.tsv"
E161_METRICS = REPO / "workspace/core4d/results/E161/surface_release_ablation/eval/full/e161_method_metrics.tsv"
E161_METRICS_REF = "workspace/core4d/results/E161/surface_release_ablation/eval/full/e161_method_metrics.tsv"
S6_ROOT = REPO / "workspace/core4d/scripts/data_construction_v3/stages/s6_downstream"
RAW_ROOT = Path("/mnt/a0ccc676-9496-49f8-a861-f8a1797dec52/mocap_data/CORE4D/CORE4D_Real")
SMPLX_DIR = Path("/mnt/a0ccc676-9496-49f8-a861-f8a1797dec52/mocap_data/smplx")
HOLOSOMA_REPO = Path("/home/ubuntu/Workspace/holosoma")

TARGET_SHORT_CASES = [
    "box021_035_p1",
    "box021_035_p2",
    "box021_029_p2",
    "box004_083_p1",
    "box004_083_p2",
    "box023_person2",
    "box004_082_p1",
    "box026_139_p1",
]

OBJECT_NAME = {
    "box021": "Box021",
    "box023": "Box023",
    "box004": "box004",
    "box026": "Box026",
}

HANDOFF_FIELDS = [
    "case_id",
    "short_case_id",
    "object_key",
    "object_name",
    "date",
    "seq",
    "person",
    "person_idx",
    "retarget_variant_id",
    "target_variant_id",
    "hand_collision_variant_id",
    "handoff_decision",
    "candidate_decision",
    "target_gate_status",
    "visual_qc_status",
    "target_scene",
    "trajectory",
    "scene_act",
    "contact_mask",
    "stage2b_target_task",
    "stage2b_result_root",
    "stage2b_manifest_ref",
    "raw_contact_threshold_label",
    "source_exp",
    "source_variant",
    "source_metrics_ref",
    "notes",
]

EVIDENCE_FIELDS = [
    "case_id",
    "object_key",
    "object_name",
    "date",
    "seq",
    "person",
    "person_idx",
    "retarget_variant_id",
    "target_variant_id",
    "hand_collision_variant_id",
    "cem_status",
    "rl_status",
    "downstream_failure_mode",
    "downstream_notes",
    "cem_run_id",
    "cem_result_npz",
    "cem_video",
    "cem_metrics_ref",
]

SOURCE_FIELDS = [
    "short_case_id",
    "case_id",
    "variant",
    "object_key",
    "object_name",
    "date",
    "seq",
    "person",
    "person_idx",
    "derived_task",
    "target_scene",
    "trajectory",
    "scene_act",
    "contact_mask",
    "cem_result_npz",
    "cem_outdir_npz",
    "cem_video",
    "success_tracked",
    "fall_flag",
    "track_pelvis_z_err_terminal_m",
    "inmask_contact_3mm",
    "phys_penetration_3mm",
    "release_false_3mm",
    "leg_penetration_frac",
    "obj_err_mean_m",
]


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def rel(path: str | Path) -> str:
    if not path:
        return ""
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
            writer.writerow({field: "" if row.get(field) is None else row.get(field, "") for field in fields})


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_identity(row: dict[str, str]) -> tuple[str, str, str, str]:
    case_id = row["case_id"]
    object_key = row["object_key"]
    if row["short_case_id"] == "box023_person2":
        task_info = repo_path(row["target_scene"]).parent / "task_info.json"
        info = json.loads(task_info.read_text(encoding="utf-8"))
        source_qpos = Path(info["source_qpos"]).name
        match = re.match(r"(?P<date>\d{8})-(?P<seq>\d{3})-(?P<person>person[12])-(?P<object>[^_]+)_with_obj", source_qpos)
        if not match:
            raise ValueError(f"cannot parse box023 source identity from {source_qpos}")
        return match.group("date"), match.group("seq"), match.group("person"), match.group("object")

    pattern = rf".*_{re.escape(object_key)}_(?P<date>\d{{8}}(?:_\d+)?)_(?P<seq>\d{{3}})_p(?P<person_idx>[12])$"
    match = re.match(pattern, case_id)
    if not match:
        raise ValueError(f"cannot parse identity from case_id={case_id}")
    person = f"person{match.group('person_idx')}"
    object_name = OBJECT_NAME.get(object_key, object_key)
    return match.group("date"), match.group("seq"), person, object_name


def load_sources() -> list[dict[str, str]]:
    variants = {
        row["short_case_id"]: row
        for row in read_tsv(E161_VARIANTS)
        if row.get("short_case_id") in TARGET_SHORT_CASES and row.get("method_group") == "releaseDecay"
    }
    missing = [case for case in TARGET_SHORT_CASES if case not in variants]
    if missing:
        raise SystemExit(f"missing E161 releaseDecay variants: {missing}")

    metrics = {
        row["short_case_id"]: row
        for row in read_tsv(E161_METRICS)
        if row.get("short_case_id") in TARGET_SHORT_CASES and row.get("method_group") == "releaseDecay"
    }
    missing_metrics = [case for case in TARGET_SHORT_CASES if case not in metrics]
    if missing_metrics:
        raise SystemExit(f"missing E161 releaseDecay metrics: {missing_metrics}")

    rows: list[dict[str, str]] = []
    for short_case in TARGET_SHORT_CASES:
        variant = variants[short_case]
        metric = metrics[short_case]
        date, seq, person, object_name = parse_identity(variant)
        row = {
            "short_case_id": short_case,
            "case_id": variant["case_id"],
            "variant": variant["variant"],
            "object_key": variant["object_key"],
            "object_name": object_name,
            "date": date,
            "seq": seq,
            "person": person,
            "person_idx": variant["person_idx"],
            "derived_task": variant["derived_task"],
            "target_scene": variant["target_scene"],
            "trajectory": variant["trajectory"],
            "scene_act": variant["rubber_scene_act"],
            "contact_mask": variant["mask_path"],
            "cem_result_npz": variant["result_npz"],
            "cem_outdir_npz": variant["outdir_npz"],
            "cem_video": variant["video"],
            "success_tracked": metric.get("success_tracked", ""),
            "fall_flag": metric.get("fall_flag", ""),
            "track_pelvis_z_err_terminal_m": metric.get("track_pelvis_z_err_terminal_m", ""),
            "inmask_contact_3mm": metric.get("hand_object_physics_contact_3mm_in_mask_frac", ""),
            "phys_penetration_3mm": metric.get("hand_object_physics_penetration_3mm_frame_frac", ""),
            "release_false_3mm": metric.get("hand_object_release_false_contact_3mm_frac", ""),
            "leg_penetration_frac": metric.get("leg_penetration_frac", ""),
            "obj_err_mean_m": metric.get("obj_err_mean_m", ""),
        }
        rows.append(row)
    return rows


def notes(row: dict[str, str]) -> str:
    return (
        "E161 releaseDecay clean8; "
        f"tracked={row['success_tracked']}; fall={row['fall_flag']}; "
        f"pelvis_z_terminal={row['track_pelvis_z_err_terminal_m']}; "
        f"inmaskC3={row['inmask_contact_3mm']}; "
        f"physPen3={row['phys_penetration_3mm']}; "
        f"releaseF3={row['release_false_3mm']}; "
        f"legPen={row['leg_penetration_frac']}; "
        f"objErr={row['obj_err_mean_m']}"
    )


def required_paths(row: dict[str, str]) -> list[str]:
    return ["target_scene", "trajectory", "scene_act", "contact_mask", "cem_result_npz", "cem_outdir_npz", "cem_video"]


def validate_source_rows(rows: list[dict[str, str]]) -> None:
    failures: list[str] = []
    for row in rows:
        for field in required_paths(row):
            path = repo_path(row[field])
            if not path.exists() or path.stat().st_size <= 0:
                failures.append(f"{row['short_case_id']}:{field}:{row[field]}")
        if row["success_tracked"] != "true":
            failures.append(f"{row['short_case_id']}:success_tracked={row['success_tracked']}")
        if row["fall_flag"] != "false":
            failures.append(f"{row['short_case_id']}:fall_flag={row['fall_flag']}")
    if failures:
        raise SystemExit("source validation failed:\n" + "\n".join(failures))


def build_handoff_rows(source_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for source in source_rows:
        rows.append(
            {
                "case_id": source["case_id"],
                "short_case_id": source["short_case_id"],
                "object_key": source["object_key"],
                "object_name": source["object_name"],
                "date": source["date"],
                "seq": source["seq"],
                "person": source["person"],
                "person_idx": source["person_idx"],
                "retarget_variant_id": "omnirt_v1",
                "target_variant_id": "ref_fk",
                "hand_collision_variant_id": "rubber_hull",
                "handoff_decision": "HANDOFF_READY",
                "candidate_decision": "E161_RELEASE_DECAY_READY",
                "target_gate_status": "pass",
                "visual_qc_status": "pass",
                "target_scene": source["target_scene"],
                "trajectory": source["trajectory"],
                "scene_act": source["scene_act"],
                "contact_mask": source["contact_mask"],
                "stage2b_target_task": source["derived_task"],
                "stage2b_result_root": "workspace/core4d/results/E161/surface_release_ablation",
                "stage2b_manifest_ref": "workspace/core4d/scripts/experiments/E161/variants.tsv",
                "raw_contact_threshold_label": "3cm",
                "source_exp": "E161",
                "source_variant": source["variant"],
                "source_metrics_ref": E161_METRICS_REF,
                "notes": notes(source),
            }
        )
    return rows


def build_evidence_rows(source_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for source in source_rows:
        rows.append(
            {
                "case_id": source["case_id"],
                "object_key": source["object_key"],
                "object_name": source["object_name"],
                "date": source["date"],
                "seq": source["seq"],
                "person": source["person"],
                "person_idx": source["person_idx"],
                "retarget_variant_id": "omnirt_v1",
                "target_variant_id": "ref_fk",
                "hand_collision_variant_id": "rubber_hull",
                "cem_status": "pass",
                "rl_status": "not_run",
                "downstream_failure_mode": "",
                "downstream_notes": notes(source),
                "cem_run_id": source["variant"],
                "cem_result_npz": source["cem_result_npz"],
                "cem_video": source["cem_video"],
                "cem_metrics_ref": E161_METRICS_REF,
            }
        )
    return rows


def run_cmd(cmd: list[str]) -> None:
    print("+ " + " ".join(str(x) for x in cmd), flush=True)
    subprocess.run([str(x) for x in cmd], cwd=REPO, check=True)


def partner_base_cmd(args: argparse.Namespace, rl_export_dir: Path, partner_dir: Path) -> list[Any]:
    return [
        sys.executable,
        S6_ROOT / "export_rl_partner_omnirt.py",
        "--rl-export-input-tsv",
        rl_export_dir / "rl_export_input.tsv",
        "--out-dir",
        partner_dir,
        "--spider-repo",
        REPO,
        "--holosoma-repo",
        args.holosoma_repo,
        "--core4d-raw-root",
        args.core4d_raw_root,
        "--smplx-model-dir",
        args.smplx_model_dir,
        "--python-bin",
        args.python_bin,
    ]


def path_exists(path_text: str) -> bool:
    if not path_text:
        return False
    path = repo_path(path_text)
    return path.exists() and path.stat().st_size > 0


def finalize_partner_manifest(partner_manifest: Path, partner_dir: Path) -> None:
    rows = read_tsv(partner_manifest)
    if not rows:
        return
    fields = list(rows[0].keys())
    for row in rows:
        if row.get("pipeline_enabled") != "1":
            continue
        required = ["converted_npz", "omniretarget_output_npz", "trimmed_npz", "trim_window_json"]
        missing = [field for field in required if not path_exists(row.get(field, ""))]
        if missing:
            row["partner_status"] = "missing_outputs"
            row["failure_mode"] = "partner_omnirt_outputs_missing"
            row["decision_notes"] = "missing outputs: " + ",".join(missing)
        else:
            row["partner_status"] = "pass"
            row["failure_mode"] = ""
            row["decision_notes"] = "temporary partner OmniRetarget outputs exist"
    write_tsv(partner_manifest, rows, fields)
    write_json(partner_manifest.with_suffix(".json"), rows)
    summary = {
        "stage": "S6_rl_partner_omnirt_finalized",
        "created_at": now(),
        "rows": len(rows),
        "partner_status_counts": dict(Counter(row.get("partner_status", "") for row in rows)),
        "manifest_tsv": str(partner_manifest),
        "out_dir": str(partner_dir),
    }
    write_json(partner_dir / "rl_partner_omnirt_summary.json", summary)
    lines = [
        "# S6 RL partner OmniRetarget summary",
        "",
        f"- rows: `{summary['rows']}`",
        f"- out_dir: `{summary['out_dir']}`",
        "",
        "## statuses",
        "",
        "| status | count |",
        "|---|---:|",
    ]
    for status, count in summary["partner_status_counts"].items():
        lines.append(f"| `{status}` | {count} |")
    lines.extend(["", "## partner rows", "", "| source | partner | status | failure | trimmed_npz |", "|---|---|---|---|---|"])
    for row in rows:
        lines.append(
            f"| `{row.get('source_case_id', '')}` | `{row.get('partner_case_id', '')}` | "
            f"`{row.get('partner_status', '')}` | `{row.get('failure_mode', '')}` | `{row.get('trimmed_npz', '')}` |"
        )
    (partner_dir / "rl_partner_omnirt_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_partner_export(args: argparse.Namespace, source_rows: list[dict[str, str]], rl_export_dir: Path, partner_dir: Path) -> None:
    base_cmd = partner_base_cmd(args, rl_export_dir, partner_dir)
    if not args.execute_partner:
        run_cmd(base_cmd)
        return

    if not args.allow_partner_failure:
        cmd = [*base_cmd, "--execute"]
        if args.force_partner:
            cmd.append("--force")
        run_cmd(cmd)
        return

    failures: list[str] = []
    for source in source_rows:
        cmd = [*base_cmd, "--case-id", source["case_id"], "--execute"]
        if args.force_partner:
            cmd.append("--force")
        try:
            run_cmd(cmd)
        except subprocess.CalledProcessError as exc:
            failures.append(f"{source['short_case_id']}:{source['case_id']}:returncode={exc.returncode}")
            print(f"partner OmniRetarget failed for {source['short_case_id']}; continuing", file=sys.stderr)

    run_cmd(base_cmd)
    partner_manifest = partner_dir / "rl_partner_omnirt_manifest.tsv"
    finalize_partner_manifest(partner_manifest, partner_dir)
    if failures:
        print("partner OmniRetarget failures recorded:\n" + "\n".join(failures), file=sys.stderr)


def summarize(
    *,
    source_rows: list[dict[str, str]],
    rl_export_input: Path,
    partner_manifest: Path,
    out_path: Path,
) -> dict[str, Any]:
    rl_rows = read_tsv(rl_export_input)
    partner_rows = read_tsv(partner_manifest) if partner_manifest.is_file() else []
    summary = {
        "experiment": "E161",
        "created_at": now(),
        "source": "E161 surfaceBandReleaseDecay clean8",
        "result_root": rel(RESULT_ROOT),
        "source_cases": [row["short_case_id"] for row in source_rows],
        "rl_export_rows": len(rl_rows),
        "rl_export_decision_counts": dict(Counter(row.get("rl_export_decision", "") for row in rl_rows)),
        "partner_rows": len(partner_rows),
        "partner_status_counts": dict(Counter(row.get("partner_status", "") for row in partner_rows)),
        "partner_cases": [
            {
                "source_case_id": row.get("source_case_id", ""),
                "partner_case_id": row.get("partner_case_id", ""),
                "partner_status": row.get("partner_status", ""),
                "failure_mode": row.get("failure_mode", ""),
                "trimmed_npz": row.get("trimmed_npz", ""),
            }
            for row in partner_rows
        ],
    }
    write_json(out_path.with_suffix(".json"), summary)

    lines = [
        "# E161 releaseDecay RL export summary",
        "",
        f"- created_at: `{summary['created_at']}`",
        f"- result_root: `{summary['result_root']}`",
        f"- RL export rows: `{summary['rl_export_rows']}`",
        f"- RL export decisions: `{summary['rl_export_decision_counts']}`",
        f"- partner statuses: `{summary['partner_status_counts']}`",
        "",
        "## Source cases",
        "",
        "| short case | case_id | person | object | scene_act | CEM result |",
        "|---|---|---|---|---|---|",
    ]
    for row in source_rows:
        lines.append(
            f"| `{row['short_case_id']}` | `{row['case_id']}` | `{row['person']}` | `{row['object_name']}` | "
            f"`{row['scene_act']}` | `{row['cem_result_npz']}` |"
        )
    lines.extend(["", "## Partner OmniRetarget", "", "| source | partner | status | failure | trimmed_npz |", "|---|---|---|---|---|"])
    for row in partner_rows:
        lines.append(
            f"| `{row.get('source_case_id', '')}` | `{row.get('partner_case_id', '')}` | "
            f"`{row.get('partner_status', '')}` | `{row.get('failure_mode', '')}` | `{row.get('trimmed_npz', '')}` |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def validate_exports(rl_export_input: Path, partner_manifest: Path, *, require_partner_pass: bool) -> None:
    rl_rows = read_tsv(rl_export_input)
    if len(rl_rows) != len(TARGET_SHORT_CASES):
        raise SystemExit(f"expected {len(TARGET_SHORT_CASES)} RL rows, got {len(rl_rows)}")
    not_ready = [row for row in rl_rows if row.get("rl_export_decision") != "RL_EXPORT_READY"]
    if not_ready:
        raise SystemExit(f"RL export has non-ready rows: {not_ready}")
    missing_required = []
    for row in rl_rows:
        for field in ["scene_act", "trajectory", "contact_mask", "cem_result_npz"]:
            path = repo_path(row[field])
            if not path.exists() or path.stat().st_size <= 0:
                missing_required.append(f"{row['case_id']}:{field}:{row[field]}")
    if missing_required:
        raise SystemExit("missing RL required files:\n" + "\n".join(missing_required))

    partner_rows = read_tsv(partner_manifest)
    if len(partner_rows) != len(TARGET_SHORT_CASES):
        raise SystemExit(f"expected {len(TARGET_SHORT_CASES)} partner rows, got {len(partner_rows)}")
    if require_partner_pass:
        failed = [row for row in partner_rows if row.get("partner_status") != "pass"]
        if failed:
            raise SystemExit(f"partner OmniRetarget failed rows: {failed}")
        missing_partner = []
        for row in partner_rows:
            for field in ["converted_npz", "omniretarget_output_npz", "trimmed_npz", "trim_window_json"]:
                path = repo_path(row[field])
                if not path.exists() or path.stat().st_size <= 0:
                    missing_partner.append(f"{row['partner_case_id']}:{field}:{row[field]}")
        if missing_partner:
            raise SystemExit("missing partner files:\n" + "\n".join(missing_partner))


def main() -> None:
    global RESULT_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=RESULT_ROOT)
    parser.add_argument("--core4d-raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--smplx-model-dir", type=Path, default=SMPLX_DIR)
    parser.add_argument("--holosoma-repo", type=Path, default=HOLOSOMA_REPO)
    parser.add_argument("--python-bin", default=".venv/bin/python")
    parser.add_argument("--execute-partner", action="store_true")
    parser.add_argument("--force-partner", action="store_true")
    parser.add_argument("--allow-partner-failure", action="store_true")
    args = parser.parse_args()

    RESULT_ROOT = args.out_root.expanduser().resolve()
    source_rows = load_sources()
    validate_source_rows(source_rows)

    manifest_dir = RESULT_ROOT / "manifest"
    handoff_dir = RESULT_ROOT / "s5_handoff"
    evidence_dir = RESULT_ROOT / "s6_downstream/evidence"
    rl_export_dir = RESULT_ROOT / "s6_downstream/rl_export"
    partner_dir = rl_export_dir / "partner_omnirt"
    for path in [manifest_dir, handoff_dir, evidence_dir, rl_export_dir, partner_dir]:
        path.mkdir(parents=True, exist_ok=True)

    source_path = manifest_dir / "releaseDecay_source_rows.tsv"
    handoff_path = handoff_dir / "handoff_manifest.tsv"
    evidence_input = evidence_dir / "downstream_evidence_input.tsv"
    write_tsv(source_path, source_rows, SOURCE_FIELDS)
    write_tsv(handoff_path, build_handoff_rows(source_rows), HANDOFF_FIELDS)
    write_tsv(evidence_input, build_evidence_rows(source_rows), EVIDENCE_FIELDS)

    run_cmd(
        [
            sys.executable,
            S6_ROOT / "record_downstream_evidence.py",
            "--handoff-manifest-tsv",
            handoff_path,
            "--evidence-tsv",
            evidence_input,
            "--out-dir",
            evidence_dir,
            "--evidence-root",
            RESULT_ROOT,
            "--source-ref",
            "E161_releaseDecay_clean8",
        ]
    )
    evidence_manifest = evidence_dir / "downstream_evidence_manifest.tsv"
    run_cmd(
        [
            sys.executable,
            S6_ROOT / "export_rl_inputs.py",
            "--handoff-manifest-tsv",
            handoff_path,
            "--cem-evidence-tsv",
            evidence_manifest,
            "--out-dir",
            rl_export_dir,
            "--spider-repo",
            REPO,
        ]
    )

    run_partner_export(args, source_rows, rl_export_dir, partner_dir)

    partner_manifest = partner_dir / "rl_partner_omnirt_manifest.tsv"
    validate_exports(
        rl_export_dir / "rl_export_input.tsv",
        partner_manifest,
        require_partner_pass=args.execute_partner and not args.allow_partner_failure,
    )
    summary = summarize(
        source_rows=source_rows,
        rl_export_input=rl_export_dir / "rl_export_input.tsv",
        partner_manifest=partner_manifest,
        out_path=RESULT_ROOT / "summary.md",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
