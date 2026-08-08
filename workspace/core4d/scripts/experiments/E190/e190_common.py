#!/usr/bin/env python3
"""Shared IO helpers and per-object source recipes for the E190 38-case noPRG RL export.

E190 does not run new physics simulation. It packages already-completed noPRG CEM
evidence from four source experiments (E189, E179, E168, E167) into the noPRG
counterpart of the frozen PRG-side 38-case RL export (E170/E172/E173).
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[5]
RUN_ROOT = REPO / "workspace/core4d/results/E190"
RL_DIR = RUN_ROOT / "s6_downstream/rl_export"
S6_SCRIPTS = REPO / "workspace/core4d/scripts/data_construction_v3/stages/s6_downstream"

SPIDER_METHOD_ID = "E167A_zOnlyBody_noPRG"
HAND_COLLISION_VARIANT_ID = "rubber_hull"

# PRG-side authority: the already-approved 38-case export this counterpart mirrors.
PRG_RL_EXPORT = {
    "box004": REPO / "workspace/core4d/results/E172/s6_downstream/rl_export/box004_user_approved/rl_export_input.tsv",
    "box001": REPO / "workspace/core4d/results/E173/s6_downstream/rl_export/box001_user_approved/rl_export_input.tsv",
    "box023": REPO / "workspace/core4d/results/E173/s6_downstream/rl_export/box023_user_approved/rl_export_input.tsv",
    "box024": REPO / "workspace/core4d/results/E173/s6_downstream/rl_export/box024_user_approved/rl_export_input.tsv",
    "box021": REPO / "workspace/core4d/results/E170/s6_downstream/rl_export/rl_export_input.tsv",
}
PRG_PARTNER_MANIFEST = {
    "box004": REPO / "workspace/core4d/results/E172/s6_downstream/rl_export/box004_user_approved/partner_omnirt/rl_partner_omnirt_manifest.tsv",
    "box001": REPO / "workspace/core4d/results/E173/s6_downstream/rl_export/box001_user_approved/partner_omnirt/rl_partner_omnirt_manifest.tsv",
    "box023": REPO / "workspace/core4d/results/E173/s6_downstream/rl_export/box023_user_approved/partner_omnirt/rl_partner_omnirt_manifest.tsv",
    "box024": REPO / "workspace/core4d/results/E173/s6_downstream/rl_export/box024_user_approved/partner_omnirt/rl_partner_omnirt_manifest.tsv",
    "box021": REPO / "workspace/core4d/results/E170/s6_downstream/rl_export/partner_omnirt/rl_partner_omnirt_manifest.tsv",
}
# box021's two "bridge" cases were never run under PRG at all (approved via the
# pre-PRG E167/E167A method as a documented USER_APPROVED_BRIDGE_OVERRIDE, see
# workspace/core4d/exp_analysis_0726.md); they are excluded from PRG_RL_EXPORT["box021"]
# (E170's 18-row export) and sourced from E167 directly instead.
BOX021_BRIDGE_CASE_IDS = {"box021_20231018_029_p2", "box021_20231011_035_p1"}

E167_RL_EXPORT = REPO / "workspace/core4d/results/E167/holosoma_zonly/rl_export/s6_downstream/rl_export/rl_export_input.tsv"
E167_PARTNER_MANIFEST = REPO / "workspace/core4d/results/E167/holosoma_zonly/rl_export/s6_downstream/rl_export/partner_omnirt/rl_partner_omnirt_manifest.tsv"

# noPRG source recipes.
E172_HANDOFF = REPO / "workspace/core4d/results/E172/s5_handoff/handoff_manifest.tsv"
E173_HANDOFF = REPO / "workspace/core4d/results/E173/s5_handoff/handoff_manifest.tsv"
E168_HANDOFF = REPO / "workspace/core4d/results/E168/s5_handoff/rubber_hull/handoff_manifest.tsv"

E189_CEM_MANIFEST = REPO / "workspace/core4d/results/E189/s6_downstream/manifests/cem_full_manifest.tsv"
E179_CEM_MANIFEST = REPO / "workspace/core4d/results/E179/s6_downstream/manifests/cem_full_manifest.tsv"
E168_CEM_MANIFEST = REPO / "workspace/core4d/results/E168/s6_downstream/cem/manifests/cem_production_manifest.tsv"

# object_key -> (source_exp_id, handoff manifest, cem manifest, cem manifest "flavor")
# flavor "full_manifest": E189/E179 s6_downstream/manifests/cem_full_manifest.tsv schema
# flavor "production_manifest": E168 s6_downstream/cem/manifests/cem_production_manifest.tsv schema
OBJECT_RECIPES = {
    "box004": ("E189", E172_HANDOFF, E189_CEM_MANIFEST, "full_manifest"),
    "box001": ("E189", E173_HANDOFF, E189_CEM_MANIFEST, "full_manifest"),
    "box024": ("E189", E173_HANDOFF, E189_CEM_MANIFEST, "full_manifest"),
    "box023": ("E179", E173_HANDOFF, E179_CEM_MANIFEST, "full_manifest"),
    "box021": ("E168", E168_HANDOFF, E168_CEM_MANIFEST, "production_manifest"),
}


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def repo_path(raw: str | Path) -> Path:
    path = Path(str(raw))
    if not path.is_absolute():
        return (REPO / path).resolve()
    if path.exists():
        return path
    # Some older (pre-E170) manifests record absolute paths from the machine that
    # originally ran them (e.g. /mnt/<uuid>/spider_workdirs/core4d/results/...,
    # missing the "workspace/" prefix this repo actually uses). Remap by the
    # stable "core4d/..." or "example_datasets/..." suffix.
    text = path.as_posix()
    for marker, prefix in (("core4d/", "workspace/"), ("example_datasets/", "")):
        idx = text.rfind(f"/{marker}")
        if idx != -1:
            return (REPO / f"{prefix}{text[idx + 1:]}").resolve()
        if text.startswith(marker):
            return (REPO / f"{prefix}{text}").resolve()
    return path


def rel(raw: str | Path) -> str:
    path = repo_path(raw)
    return path.relative_to(REPO.resolve()).as_posix()


def require_file(raw: str | Path, label: str) -> Path:
    path = repo_path(raw)
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"missing {label}: {raw}")
    return path


def sha256(raw: str | Path) -> str:
    path = require_file(raw, "sha256 input")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unique_by_case(rows: list[dict[str, str]], label: str) -> dict[str, dict[str, str]]:
    output: dict[str, dict[str, str]] = {}
    for row in rows:
        case_id = row.get("case_id")
        if not case_id:
            continue
        if case_id in output:
            raise SystemExit(f"duplicate case_id in {label}: {case_id}")
        output[case_id] = row
    return output


FROZEN_RL_LABELS = REPO / "workspace/core4d/results/E180/rl_metric_separability/frozen_rl_labels.tsv"


def load_authority() -> dict[str, list[str]]:
    """Read the true 38-case authority from E180's frozen_rl_labels.tsv.

    This is NOT the same as "all rows in each PRG rl_export_input.tsv" — e.g. E170's
    box021 export has 18 RL_EXPORT_READY rows, but only 9 of those (plus 2 bridge
    cases sourced from E167, outside E170's export entirely) are part of the 38-case
    set. frozen_rl_labels.tsv is the actual RL-release authority.
    """
    rows = read_tsv(require_file(FROZEN_RL_LABELS, "E180 frozen_rl_labels.tsv"))
    authority: dict[str, list[str]] = {}
    for row in rows:
        authority.setdefault(row["object_key"], []).append(row["case_id"])
    for object_key in authority:
        authority[object_key] = sorted(authority[object_key])

    expected_counts = {"box001": 13, "box004": 4, "box021": 11, "box023": 7, "box024": 3}
    for object_key, expected in expected_counts.items():
        if len(authority.get(object_key, [])) != expected:
            raise SystemExit(
                f"authority drift for {object_key}: expected {expected}, got {len(authority.get(object_key, []))}"
            )
    total = sum(len(v) for v in authority.values())
    if total != 38:
        raise SystemExit(f"authority drift: expected 38 total cases, got {total}")

    # Cross-check: every non-bridge case must also appear in the PRG-side export
    # (bridge cases are the sole documented exception - see BOX021_BRIDGE_CASE_IDS).
    for object_key, case_ids in authority.items():
        prg_rows = read_tsv(require_file(PRG_RL_EXPORT[object_key], f"{object_key} PRG rl_export_input.tsv"))
        prg_case_ids = {row["case_id"] for row in prg_rows if row.get("object_key") == object_key}
        missing = sorted(set(case_ids) - prg_case_ids - BOX021_BRIDGE_CASE_IDS)
        if missing:
            raise SystemExit(f"{object_key}: authority cases missing from PRG-side export: {missing}")
    return authority
