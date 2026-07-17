#!/usr/bin/env python3
"""Install E168 rubber-hand sidecars and finalize the S5 handoff."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[5]
S5_DIR = REPO / "workspace/core4d/scripts/data_construction_v3/stages/s5_handoff"
sys.path.insert(0, str(S5_DIR))

from patch_hand_collision import patch_scene  # noqa: E402


SCENE_NAME = "scene_act_E168_rubber_hull"


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_tsv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader), list(reader.fieldnames or [])


def repo_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO / path


def rel(path: Path) -> str:
    try:
        return str(path.absolute().relative_to(REPO.absolute()))
    except ValueError:
        pass
    for symlink_root in (REPO / "workspace/core4d/results",):
        if not symlink_root.exists():
            continue
        try:
            suffix = path.resolve().relative_to(symlink_root.resolve())
            return str(symlink_root.relative_to(REPO) / suffix)
        except ValueError:
            pass
    try:
        return str(path.resolve().relative_to(REPO.resolve()))
    except ValueError:
        return str(path.resolve())


def portable_path(value: str) -> str:
    if not value:
        return ""
    path = Path(value).expanduser()
    if not path.is_absolute():
        return value
    return rel(path)


PATH_FIELDS = {
    "source_scene",
    "target_scene",
    "trajectory",
    "scene_act",
    "contact_mask",
    "contact_mask_npz",
    "contact_mask_3cm_npz",
    "contact_mask_5cm_npz",
    "raw_contact_artifact_npz",
    "raw_contact_3cm_artifact_npz",
    "raw_contact_5cm_artifact_npz",
    "contact_target_npz",
    "target_npz",
    "evidence_root",
    "stage2b_result_root",
    "route_diagnostic_ref",
    "contact_route_diagnostic_ref",
    "downstream_evidence_root",
    "cem_result_npz",
    "cem_video",
    "cem_metrics_ref",
    "rl_checkpoint",
    "rl_metrics_ref",
    "rl_video",
    "base_scene_act",
    "hand_collision_adapter_ref",
}


def write_tsv(
    path: Path, rows: list[dict[str, Any]], fields: list[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--handoff-manifest-tsv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cem-override-manifest-tsv", type=Path, default=None)
    args = parser.parse_args()

    source_rows, fields = read_tsv(args.handoff_manifest_tsv.expanduser().resolve())
    ready_rows = [row for row in source_rows if row.get("handoff_decision") == "HANDOFF_READY"]
    if len(source_rows) != 40 or len(ready_rows) != 40:
        raise SystemExit(
            f"expected 40/40 S5 rows ready, got {len(source_rows)}/{len(ready_rows)}"
        )
    out_dir = args.out_dir.expanduser().resolve()
    adapter_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    for row in ready_rows:
        base_scene = repo_path(row["scene_act"])
        task_dir = base_scene.parent
        adapter = patch_scene(
            base_scene_act=base_scene,
            out_dir=out_dir / "hand_collision_scenes" / row["case_id"],
            case_id=row["case_id"],
            hand_collision_variant_id="rubber_hull",
            scene_name=SCENE_NAME,
            install_dir=task_dir,
            repo=REPO,
        )
        adapter_rows.append(adapter)
        installed = task_dir / f"{SCENE_NAME}.xml"
        final = {
            key: portable_path(value) if key in PATH_FIELDS else value
            for key, value in row.items()
        }
        final.update(
            {
                "hand_collision_variant_id": "rubber_hull",
                "base_scene_act": row["scene_act"],
                "scene_act": rel(installed),
                "scene_name": SCENE_NAME,
                "hand_collision_adapter_status": adapter["status"],
                "hand_collision_adapter_ref": adapter["patched_scene_act"],
                "base_scene_act_sha256": adapter["base_scene_sha256"],
                "scene_act_sha256": adapter["patched_scene_sha256"],
                "updated_at": now(),
            }
        )
        final_rows.append(final)

    extra_fields = [
        "base_scene_act",
        "scene_name",
        "hand_collision_adapter_status",
        "hand_collision_adapter_ref",
        "base_scene_act_sha256",
        "scene_act_sha256",
    ]
    final_fields = fields + [field for field in extra_fields if field not in fields]
    write_tsv(out_dir / "handoff_manifest.tsv", final_rows, final_fields)
    (out_dir / "handoff_manifest.json").write_text(
        json.dumps(final_rows, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    adapter_fields = sorted({key for row in adapter_rows for key in row})
    write_tsv(out_dir / "hand_collision_adapter_manifest.tsv", adapter_rows, adapter_fields)
    (out_dir / "hand_collision_adapter_manifest.json").write_text(
        json.dumps(adapter_rows, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "created_at": now(),
        "status": "pass",
        "rows": len(final_rows),
        "handoff_decision_counts": dict(
            Counter(row["handoff_decision"] for row in final_rows)
        ),
        "retarget_variant_counts": dict(
            Counter(row["retarget_variant_id"] for row in final_rows)
        ),
        "hand_collision_variant_counts": dict(
            Counter(row["hand_collision_variant_id"] for row in final_rows)
        ),
        "adapter_status_counts": dict(Counter(row["status"] for row in adapter_rows)),
        "scene_name": SCENE_NAME,
        "source_handoff_manifest": str(args.handoff_manifest_tsv.expanduser().resolve()),
        "handoff_manifest": str(out_dir / "handoff_manifest.tsv"),
    }
    (out_dir / "handoff_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.cem_override_manifest_tsv:
        override_rows, _ = read_tsv(
            args.cem_override_manifest_tsv.expanduser().resolve()
        )
        if len(override_rows) != 40 or any(
            row.get("override_status") != "pass" for row in override_rows
        ):
            raise SystemExit("expected 40 pass CEM override rows before installation")
        installed_overrides = []
        install_dir = REPO / "examples/config/override"
        for row in override_rows:
            source = repo_path(row["override_config"])
            destination = install_dir / source.name
            destination.write_bytes(source.read_bytes())
            installed_overrides.append(
                {
                    "case_id": row["case_id"],
                    "retarget_variant_id": row["retarget_variant_id"],
                    "target_variant_id": row["target_variant_id"],
                    "source_override": rel(source),
                    "installed_override": rel(destination),
                    "override_id": destination.stem,
                    "status": "pass",
                    "updated_at": now(),
                }
            )
        install_fields = list(installed_overrides[0].keys())
        write_tsv(
            out_dir / "cem_override_install_manifest.tsv",
            installed_overrides,
            install_fields,
        )
        (out_dir / "cem_override_install_manifest.json").write_text(
            json.dumps(installed_overrides, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        summary["installed_cem_overrides"] = len(installed_overrides)
        summary["cem_override_install_manifest"] = str(
            out_dir / "cem_override_install_manifest.tsv"
        )
        (out_dir / "handoff_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
