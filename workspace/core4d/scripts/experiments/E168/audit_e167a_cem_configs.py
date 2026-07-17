#!/usr/bin/env python3
"""Compose and audit every E168 E167A CEM override."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


REPO = Path(__file__).resolve().parents[5]


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO / path


def normalize(value: Any) -> Any:
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def equal(actual: Any, expected: Any) -> bool:
    actual = normalize(actual)
    if isinstance(expected, float):
        try:
            return abs(float(actual) - expected) <= 1e-9
        except (TypeError, ValueError):
            return False
    return actual == expected


def runtime_mask_time_axis(axis: str) -> str:
    axis = (axis or "auto").strip()
    if axis in {"auto", "raw", "spider", "eval", "method"}:
        return axis
    return "auto"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--handoff-manifest-tsv", type=Path, required=True)
    parser.add_argument("--override-install-manifest-tsv", type=Path, required=True)
    parser.add_argument("--profile-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    handoff_rows = read_tsv(args.handoff_manifest_tsv.expanduser().resolve())
    handoff = {
        (row["case_id"], row["retarget_variant_id"], row["target_variant_id"]): row
        for row in handoff_rows
    }
    install_rows = read_tsv(args.override_install_manifest_tsv.expanduser().resolve())
    profile_doc = json.loads(args.profile_json.expanduser().resolve().read_text())
    expected = profile_doc["profile"]
    expected["cem_smooth_enabled"] = False

    audit_rows = []
    config_dir = REPO / "examples/config"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir.resolve())):
        for install in install_rows:
            key = (
                install["case_id"],
                install["retarget_variant_id"],
                install["target_variant_id"],
            )
            source = handoff[key]
            cfg = compose(
                config_name="default",
                overrides=[f"+override={install['override_id']}"],
            )
            failures = []
            for field, expected_value in expected.items():
                actual = OmegaConf.select(cfg, field, default=None)
                if field == "cem_smooth_enabled" and actual is None:
                    actual = False
                if not equal(actual, expected_value):
                    failures.append(
                        f"{field}:actual={normalize(actual)!r}:expected={expected_value!r}"
                    )
            route_expected = {
                "task": source["stage2b_target_task"],
                "scene_name": "scene_act_E168_rubber_hull",
                "contact_hdmi_target_source": "ref_fk",
                "contact_hdmi_mask_source": "core4d_3cm",
                "contact_hdmi_mask_path": source["contact_mask_npz"],
                "contact_hdmi_mask_person_idx": int(source["contact_mask_person_idx"]),
                "contact_hdmi_mask_time_axis": runtime_mask_time_axis(source["contact_mask_time_axis"]),
            }
            for field, expected_value in route_expected.items():
                actual = OmegaConf.select(cfg, field, default=None)
                if not equal(actual, expected_value):
                    failures.append(
                        f"{field}:actual={normalize(actual)!r}:expected={expected_value!r}"
                    )
            for field in ("target_scene", "trajectory", "scene_act", "contact_mask_npz"):
                if not resolve_path(source[field]).is_file():
                    failures.append(f"missing_path:{field}={source[field]}")
            audit_rows.append(
                {
                    "case_id": source["case_id"],
                    "retarget_variant_id": source["retarget_variant_id"],
                    "target_variant_id": source["target_variant_id"],
                    "hand_collision_variant_id": source["hand_collision_variant_id"],
                    "override_id": install["override_id"],
                    "audit_status": "pass" if not failures else "fail",
                    "failure_count": len(failures),
                    "failures_json": json.dumps(failures, sort_keys=True),
                    "task": str(cfg.task),
                    "scene_name": str(cfg.scene_name),
                    "contact_mask_path": str(cfg.contact_hdmi_mask_path),
                    "contact_mask_person_idx": str(cfg.contact_hdmi_mask_person_idx),
                    "updated_at": now(),
                }
            )

    if len(audit_rows) != 40:
        raise SystemExit(f"expected 40 config audit rows, got {len(audit_rows)}")
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = list(audit_rows[0].keys())
    with (out_dir / "e167a_config_audit.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(audit_rows)
    summary = {
        "created_at": now(),
        "rows": len(audit_rows),
        "audit_status_counts": dict(Counter(row["audit_status"] for row in audit_rows)),
        "profile_json": str(args.profile_json.expanduser().resolve()),
        "profile_sha256": profile_doc["profile_sha256"],
        "status": "pass" if all(row["audit_status"] == "pass" for row in audit_rows) else "fail",
        "out_dir": str(out_dir),
    }
    (out_dir / "e167a_config_audit.json").write_text(
        json.dumps(audit_rows, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "e167a_config_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if summary["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
