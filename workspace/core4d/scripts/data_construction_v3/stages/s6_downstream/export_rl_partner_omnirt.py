#!/usr/bin/env python3
"""Generate temporary partner OmniRetarget outputs for S6 RL export rows.

This script is intentionally downstream-only: it does not update the v3
registry, S5 handoff, CEM evidence, or RL export readiness decisions.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import sys
from collections import Counter
from pathlib import Path

SCRIPT_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "data_construction_v3")
for _path in (SCRIPT_ROOT / "lib", SCRIPT_ROOT / "state", SCRIPT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
from typing import Any

from common import (
    SCHEMA_VERSION,
    find_spider_repo,
    json_dumps,
    read_tsv,
    resolve_holosoma_repo,
    timestamp,
    write_json,
    write_tsv,
)


PERSON_PARTNER = {"person1": "person2", "person2": "person1"}
PERSON_INDEX = {"person1": "0", "person2": "1"}
PERSON_SHORT = {"person1": "p1", "person2": "p2"}
RETARGET_ENV_BY_VARIANT = {
    "omnirt_v1": {
        "RETARGET_ENABLE_CONSTRAINT_RELAXATION": "0",
        "RETARGET_ENABLE_FOOT_Z_CONSTRAINT": "0",
        "RETARGET_FOOT_SLIDE_PENALTY_WEIGHT": "0.0",
        "RETARGET_ENABLE_CONTACT_PRESERVATION": "0",
        "RETARGET_OBJECT_PENETRATION_TOLERANCE_SCALE": "1.0",
    },
    "omnirt_v2": {
        "RETARGET_ENABLE_CONSTRAINT_RELAXATION": "1",
        "RETARGET_ENABLE_FOOT_Z_CONSTRAINT": "1",
        "RETARGET_FOOT_SLIDE_PENALTY_WEIGHT": "1.0",
        "RETARGET_ENABLE_CONTACT_PRESERVATION": "1",
        "RETARGET_OBJECT_PENETRATION_TOLERANCE_SCALE": "0.8",
    },
}

FIELDS = [
    "source_case_id",
    "source_rl_export_decision",
    "source_handoff_decision",
    "source_cem_status",
    "source_person",
    "source_person_idx",
    "partner_case_id",
    "partner_person",
    "partner_person_idx",
    "object_key",
    "object_name",
    "object_model_rel",
    "date",
    "seq",
    "retarget_variant_id",
    "target_variant_id",
    "generation_mode",
    "pipeline_enabled",
    "partner_status",
    "failure_mode",
    "decision_notes",
    "partner_target_task",
    "result_root",
    "holosoma_case_root",
    "converted_npz",
    "omniretarget_output_npz",
    "retargeted_npz",
    "trimmed_npz",
    "trim_window_json",
    "trim_start",
    "trim_end",
    "trim_frames",
    "untrimmed_frames",
    "trimmed_frames",
    "raw_window_start_frame",
    "raw_window_end_frame",
    "raw_window_num_frames",
    "case_file",
    "run_script",
    "command_line",
    "replace_wrist_with_fingertip",
    "retarget_params_json",
    "source_rl_export_input",
    "schema_version",
    "updated_at",
]


def safe_id(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_")


def rel_to_repo(path: Path, repo: Path) -> str:
    return os.path.relpath(path.resolve(), repo.resolve())


def resolve_existing(path_text: str, repo: Path) -> bool:
    if not path_text:
        return False
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = repo / path
    return path.exists() and path.stat().st_size > 0


def object_model_rel(core4d_raw_root: Path, object_name: str, object_key: str) -> str:
    object_root = core4d_raw_root / "object_models"
    candidates: list[Path] = []
    for name in [object_name, object_key]:
        if not name:
            continue
        candidates.extend(sorted(object_root.glob(f"*/{name}_m.obj")))
        candidates.extend(sorted(object_root.glob(f"*/{name}.obj")))
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    if not unique:
        raise FileNotFoundError(f"object mesh not found under {object_root}: object_name={object_name} object_key={object_key}")
    if len(unique) > 1:
        rels = ", ".join(str(path.relative_to(object_root)) for path in unique)
        raise RuntimeError(f"ambiguous object mesh for {object_name or object_key}: {rels}")
    return str(unique[0].relative_to(object_root))


def case_id(object_key: str, date: str, seq: str, person: str) -> str:
    pshort = PERSON_SHORT.get(person, person)
    return safe_id(f"{object_key}_{date}_{seq}_{pshort}")


def task_name(date: str, seq: str, person: str, object_name: str) -> str:
    return f"{date}-{seq}-{person}-{object_name}_with_obj"


def partner_rows(
    rl_rows: list[dict[str, str]],
    *,
    core4d_raw_root: Path,
    source_rl_export_input: Path,
    out_dir: Path,
    result_root: Path,
    spider_repo: Path,
    retarget_variant_id: str,
    target_variant_id: str,
    replace_wrist: bool,
    retarget_env: dict[str, str],
    case_ids: set[str],
    include_non_ready: bool,
) -> tuple[list[dict[str, Any]], list[list[str]]]:
    rows: list[dict[str, Any]] = []
    case_file_rows_by_partner: dict[str, list[str]] = {}
    result_root_rel = rel_to_repo(result_root, spider_repo)
    case_file = out_dir / "cases_rl_partner_omnirt.tsv"
    run_script = out_dir / "run_rl_partner_omnirt.sh"
    case_file_rel = rel_to_repo(case_file, spider_repo)

    for source in rl_rows:
        source_case = source.get("case_id", "")
        if case_ids and source_case not in case_ids:
            continue
        if not include_non_ready and source.get("rl_export_decision") != "RL_EXPORT_READY":
            continue
        source_person = source.get("person", "")
        partner_person = PERSON_PARTNER.get(source_person, "")
        object_key = source.get("object_key", "")
        object_name = source.get("object_name", "")
        date = source.get("date", "")
        seq = source.get("seq", "")
        partner_case = case_id(object_key, date, seq, partner_person) if partner_person else ""
        target_task = safe_id(f"rl_partner_omnirt_{retarget_variant_id}_{partner_case}") if partner_case else ""
        case_root = result_root / f"holosoma_{target_task}" if target_task else Path("")
        htask = task_name(date, seq, partner_person, object_name) if partner_person else ""
        converted_npz = case_root / "converted" / f"{htask}.npz" if htask else Path("")
        retargeted_npz = case_root / "retargeted" / f"{htask}_original.npz" if htask else Path("")
        trimmed_npz = case_root / "trimmed" / f"{htask}_original.npz" if htask else Path("")

        enabled = 1
        status = "ready_to_execute"
        failure = ""
        notes = "temporary partner OmniRetarget output for RL; not v3 handoff evidence"
        model_rel = ""
        if not partner_person:
            enabled = 0
            status = "blocked"
            failure = "unsupported_source_person"
            notes = f"cannot infer partner for source_person={source_person}"
        elif not all([object_key, object_name, date, seq]):
            enabled = 0
            status = "blocked"
            failure = "missing_source_identity"
            notes = "source row missing one of object_key/object_name/date/seq"
        else:
            try:
                model_rel = object_model_rel(core4d_raw_root, object_name, object_key)
            except Exception as exc:  # noqa: BLE001 - retain per-row blocking reason in manifest.
                enabled = 0
                status = "blocked"
                failure = "object_model_missing_or_ambiguous"
                notes = str(exc)

        command = [
            "env",
            f"REPO={spider_repo}",
            f"RESULT_ROOT={result_root_rel}",
            f"REPLACE_WRIST_WITH_FINGERTIP={'1' if replace_wrist else '0'}",
            *(f"{key}={value}" for key, value in retarget_env.items()),
            "bash",
            "workspace/core4d/data_preprocess/pipeline.sh",
            "--case-file",
            case_file_rel,
            "--skip-contact",
            "--skip-spider",
        ]
        row = {
            "source_case_id": source_case,
            "source_rl_export_decision": source.get("rl_export_decision", ""),
            "source_handoff_decision": source.get("handoff_decision", ""),
            "source_cem_status": source.get("cem_status", ""),
            "source_person": source_person,
            "source_person_idx": source.get("person_idx", ""),
            "partner_case_id": partner_case,
            "partner_person": partner_person,
            "partner_person_idx": PERSON_INDEX.get(partner_person, ""),
            "object_key": object_key,
            "object_name": object_name,
            "object_model_rel": model_rel,
            "date": date,
            "seq": seq,
            "retarget_variant_id": retarget_variant_id,
            "target_variant_id": target_variant_id,
            "generation_mode": "direct_omnirt_partner_temp",
            "pipeline_enabled": enabled,
            "partner_status": status,
            "failure_mode": failure,
            "decision_notes": notes,
            "partner_target_task": target_task,
            "result_root": str(result_root),
            "holosoma_case_root": str(case_root) if target_task else "",
            "converted_npz": str(converted_npz) if htask else "",
            "omniretarget_output_npz": str(retargeted_npz) if htask else "",
            "retargeted_npz": str(retargeted_npz) if htask else "",
            "trimmed_npz": str(trimmed_npz) if htask else "",
            "trim_window_json": str(case_root / "trim_window.json") if target_task else "",
            "trim_start": "",
            "trim_end": "",
            "trim_frames": "",
            "untrimmed_frames": "",
            "trimmed_frames": "",
            "raw_window_start_frame": "",
            "raw_window_end_frame": "",
            "raw_window_num_frames": "",
            "case_file": str(case_file),
            "run_script": str(run_script),
            "command_line": " ".join(shlex.quote(x) for x in command),
            "replace_wrist_with_fingertip": "1" if replace_wrist else "0",
            "retarget_params_json": json.dumps(
                {
                    **retarget_env,
                    "REPLACE_WRIST_WITH_FINGERTIP": "1" if replace_wrist else "0",
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            "source_rl_export_input": str(source_rl_export_input),
            "schema_version": SCHEMA_VERSION,
            "updated_at": timestamp(),
        }
        rows.append(row)
        if enabled:
            case_file_rows_by_partner.setdefault(
                partner_case,
                [
                    "1",
                    date,
                    seq,
                    partner_person,
                    object_name,
                    model_rel,
                    "unused_source_scene",
                    target_task,
                    "auto",
                    "auto",
                    "0",
                    target_task,
                ],
            )
    return rows, list(case_file_rows_by_partner.values())


def write_case_file(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "# enabled",
        "date",
        "seq",
        "person",
        "object_name",
        "object_model_rel",
        "source_scene_task",
        "target_task",
        "trim_start",
        "trim_frames",
        "data_id",
        "mask_slug",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def write_run_script(path: Path, command: list[str], *, dry_run: bool, force: bool, holosoma_repo: Path, core4d_raw_root: Path, smplx_model_dir: Path, python_bin: str) -> None:
    cmd = [
        "env",
        f"HOLOSOMA_DIR={holosoma_repo}",
        f"CORE4D_REAL_ROOT={core4d_raw_root}",
        f"SMPLX_MODEL_DIR={smplx_model_dir}",
        f"PYTHON_BIN={python_bin}",
        *command,
    ]
    if force:
        cmd.append("--force")
    if dry_run:
        cmd.append("--dry-run")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\ncd \"$(git rev-parse --show-toplevel)\"\n"
        + " ".join(shlex.quote(x) for x in cmd)
        + "\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def load_trim_window(path_text: str, repo: Path) -> dict[str, Any]:
    if not path_text:
        return {}
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = repo / path
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def refresh_outputs(rows: list[dict[str, Any]], repo: Path, *, mark_missing: bool) -> None:
    for row in rows:
        if str(row.get("pipeline_enabled", "")) != "1":
            continue
        missing = [
            field
            for field in ["converted_npz", "omniretarget_output_npz", "trimmed_npz"]
            if not resolve_existing(str(row.get(field, "")), repo)
        ]
        trim_window = load_trim_window(str(row.get("trim_window_json", "")), repo)
        if trim_window:
            row["trim_start"] = trim_window.get("trim_start", "")
            row["trim_end"] = trim_window.get("trim_end", "")
            row["trim_frames"] = trim_window.get("trim_frames", "")
            row["untrimmed_frames"] = trim_window.get("untrimmed_frames", "")
            row["trimmed_frames"] = trim_window.get("trimmed_frames", "")
            row["raw_window_start_frame"] = trim_window.get("trim_start", "")
            row["raw_window_end_frame"] = trim_window.get("trim_end", "")
            row["raw_window_num_frames"] = trim_window.get("trim_frames", "")
        row["updated_at"] = timestamp()
        if not trim_window and mark_missing:
            missing.append("trim_window_json")
        if missing and mark_missing:
            row["partner_status"] = "missing_outputs"
            row["failure_mode"] = "partner_omnirt_outputs_missing"
            row["decision_notes"] = "missing outputs: " + ",".join(missing)
        elif not missing and trim_window:
            row["partner_status"] = "pass"
            row["failure_mode"] = ""
            row["decision_notes"] = "temporary partner OmniRetarget outputs exist"


def markdown_summary(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# S6 RL partner OmniRetarget summary",
        "",
        f"- rows: `{summary['rows']}`",
        f"- runnable partner cases: `{summary['runnable_partner_cases']}`",
        f"- executed: `{summary['executed']}`",
        f"- out_dir: `{summary['out_dir']}`",
        "",
        "## statuses",
        "",
        "| status | count |",
        "|---|---:|",
    ]
    for status, count in summary["partner_status_counts"].items():
        lines.append(f"| `{status}` | {count} |")
    lines.extend(["", "## partner rows", "", "| source | partner | status | retargeted |", "|---|---|---|---|"])
    for row in rows:
        lines.append(
            f"| `{row['source_case_id']}` | `{row['partner_case_id']}` | `{row['partner_status']}` | "
            f"`{row['omniretarget_output_npz']}` |"
        )
    lines.extend(
        [
            "",
            "说明：本表只为 RL 临时 partner motion 输入服务，不回写 v3 registry、S5 handoff 或 S6 CEM evidence。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rl-export-input-tsv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--spider-repo", type=Path, default=None)
    parser.add_argument("--holosoma-repo", type=Path, default=None)
    parser.add_argument("--core4d-raw-root", type=Path, default=None)
    parser.add_argument("--smplx-model-dir", type=Path, default=None)
    parser.add_argument("--python-bin", default=".venv/bin/python")
    parser.add_argument("--retarget-variant-id", default="omnirt_v1")
    parser.add_argument("--target-variant-id", default="partner_temp_omnirt_only")
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--include-non-ready", action="store_true")
    parser.add_argument("--replace-wrist-with-fingertip", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    repo = (args.spider_repo or find_spider_repo()).expanduser().resolve()
    holosoma_repo = (args.holosoma_repo or resolve_holosoma_repo()).expanduser().resolve()
    raw_root = args.core4d_raw_root or (Path(os.environ["CORE4D_RAW_ROOT"]) if os.environ.get("CORE4D_RAW_ROOT") else None)
    smplx_model_dir = args.smplx_model_dir or (Path(os.environ["SMPLX_MODEL_DIR"]) if os.environ.get("SMPLX_MODEL_DIR") else None)
    if raw_root is None:
        raise SystemExit("missing --core4d-raw-root or CORE4D_RAW_ROOT")
    if smplx_model_dir is None:
        raise SystemExit("missing --smplx-model-dir or SMPLX_MODEL_DIR")
    raw_root = raw_root.expanduser().resolve()
    smplx_model_dir = smplx_model_dir.expanduser().resolve()
    if not (raw_root / "object_models").is_dir() or not (raw_root / "human_object_motions").is_dir():
        raise SystemExit(f"invalid CORE4D raw root: {raw_root}")
    if not smplx_model_dir.is_dir():
        raise SystemExit(f"invalid SMPL-X model dir: {smplx_model_dir}")
    if args.retarget_variant_id not in RETARGET_ENV_BY_VARIANT:
        raise SystemExit(
            "partner exporter supports explicit solver parameters only for "
            f"{sorted(RETARGET_ENV_BY_VARIANT)}, got {args.retarget_variant_id}"
        )
    if args.retarget_variant_id == "omnirt_v2" and args.replace_wrist_with_fingertip:
        raise SystemExit("omnirt_v2 Phase4 rescue requires wrist targets (replacement=0)")
    retarget_env = RETARGET_ENV_BY_VARIANT[args.retarget_variant_id]

    rl_export_input = args.rl_export_input_tsv.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    result_root = out_dir / "results" / safe_id(args.retarget_variant_id)
    case_file = out_dir / "cases_rl_partner_omnirt.tsv"
    run_script = out_dir / "run_rl_partner_omnirt.sh"
    manifest_tsv = out_dir / "rl_partner_omnirt_manifest.tsv"
    out_dir.mkdir(parents=True, exist_ok=True)

    command = [
        "env",
        f"REPO={repo}",
        f"RESULT_ROOT={rel_to_repo(result_root, repo)}",
        f"REPLACE_WRIST_WITH_FINGERTIP={'1' if args.replace_wrist_with_fingertip else '0'}",
        *(f"{key}={value}" for key, value in retarget_env.items()),
        "bash",
        "workspace/core4d/data_preprocess/pipeline.sh",
        "--case-file",
        rel_to_repo(case_file, repo),
        "--skip-contact",
        "--skip-spider",
    ]
    rows, pipeline_cases = partner_rows(
        read_tsv(rl_export_input),
        core4d_raw_root=raw_root,
        source_rl_export_input=rl_export_input,
        out_dir=out_dir,
        result_root=result_root,
        spider_repo=repo,
        retarget_variant_id=args.retarget_variant_id,
        target_variant_id=args.target_variant_id,
        replace_wrist=args.replace_wrist_with_fingertip,
        retarget_env=retarget_env,
        case_ids=set(args.case_id),
        include_non_ready=args.include_non_ready,
    )
    write_case_file(case_file, pipeline_cases)
    write_run_script(
        run_script,
        command,
        dry_run=not args.execute,
        force=args.force,
        holosoma_repo=holosoma_repo,
        core4d_raw_root=raw_root,
        smplx_model_dir=smplx_model_dir,
        python_bin=args.python_bin,
    )

    returncode = 0
    if args.execute and pipeline_cases:
        completed = subprocess.run([str(run_script)], cwd=repo, check=False)
        returncode = completed.returncode
    refresh_outputs(rows, repo, mark_missing=args.execute)
    write_tsv(manifest_tsv, rows, FIELDS)
    write_json(out_dir / "rl_partner_omnirt_manifest.json", rows)
    summary = {
        "stage": "S6_rl_partner_omnirt",
        "created_at": timestamp(),
        "schema_version": SCHEMA_VERSION,
        "rows": len(rows),
        "runnable_partner_cases": len(pipeline_cases),
        "executed": str(args.execute).lower(),
        "returncode": returncode,
        "partner_status_counts": dict(Counter(row["partner_status"] for row in rows)),
        "source_rl_export_input": str(rl_export_input),
        "case_file": str(case_file),
        "run_script": str(run_script),
        "manifest_tsv": str(manifest_tsv),
        "out_dir": str(out_dir),
        "result_root": str(result_root),
    }
    write_json(out_dir / "rl_partner_omnirt_summary.json", summary)
    (out_dir / "rl_partner_omnirt_summary.md").write_text(markdown_summary(summary, rows), encoding="utf-8")
    print(json_dumps(summary))
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
