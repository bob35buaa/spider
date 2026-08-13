#!/usr/bin/env python3
"""Frozen contract + authority loader for E198 (G1+A2) and the E192 A2 expansion.

This module builds the joint 103-run priority queue that completes the 2x2 factorial
(none / G1 / A2 / G1+A2) for box004/box021/box023/box024:

  Tier P0  box024 G1+A2 (9)        -> experiment E198
  Tier P1  box021+box023 A2 (44)   -> experiment E192-ext
  Tier P2  box021+box023 G1+A2 (44)-> experiment E198
  Tier P3  box004 G1+A2 (6)        -> experiment E198

Authority is reused, never re-derived:
  * box021/box023 -> E194 g1_expansion_source_authority.tsv (base + G1 sidecar + override + SHAs)
  * box004/box024 -> E172/E173 cem_full_manifest.tsv (base) + E194 original G1 sidecar
The G1 sidecars already exist on disk from E194 and are reused, not rebuilt.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[5]
SCRIPT_DIR = REPO / "workspace/core4d/scripts/experiments/E198"
RESULTS_E198 = REPO / "workspace/core4d/results/E198"
RESULTS_E192 = REPO / "workspace/core4d/results/E192"
TASK_ROOT = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"

# --- authority sources (read-only) ------------------------------------------
G1_EXPANSION_AUTHORITY = (
    REPO / "workspace/core4d/results/E194/s6_downstream/manifests/g1_expansion_source_authority.tsv"
)
SOURCE_MANIFEST = {  # box004/box024 base PRG authority (A0 side, already landed)
    "box004": ("E172", REPO / "workspace/core4d/results/E172/s6_downstream/manifests/cem_full_manifest.tsv"),
    "box024": ("E173", REPO / "workspace/core4d/results/E173/s6_downstream/manifests/cem_full_manifest.tsv"),
}
BOX2404_G1_SIDECAR = "scene_act_E194_rubberHull_PRG_gravcomp.xml"  # E194 original (box024/box004)
EXPANSION_G1_SIDECAR = "scene_act_E194_G1_expansion_rubberHull_PRG_gravcomp.xml"  # E194 expansion

BOX2404_CASES = {
    "box024": [
        "box024_20231011_026_p1", "box024_20231011_026_p2",
        "box024_20231011_027_p1", "box024_20231011_027_p2",
        "box024_20231011_028_p1", "box024_20231011_028_p2",
        "box024_20231011_030_p1",
        "box024_20231011_031_p1", "box024_20231011_031_p2",
    ],
    "box004": [
        "box004_20231003_2_082_p1", "box004_20231003_2_082_p2",
        "box004_20231003_2_083_p1", "box004_20231003_2_083_p2",
        "box004_20231003_2_086_p1", "box004_20231003_2_086_p2",
    ],
}

# --- frozen intervention parameters ------------------------------------------
# A2 hand-gate pack, identical to E192 A2_GATE (audit verifies equality).
A2_GATE = {
    "cem_hand_gate_min_sdf_m": -0.010,
    "cem_hand_gate_max_violation_pct": 0.05,
    "cem_hand_gate_hard_floor_m": -0.015,
}
KP_POS, KP_ROT = 500.0, 50.0
FULL_SAMPLES, FULL_OPT_STEPS = 1024, 32
CANARY_SAMPLES, CANARY_OPT_STEPS = 64, 4
CEM_SEED = 0

OBJECT_COUNTS = {"box024": 9, "box004": 6, "box021": 28, "box023": 16}

# --- priority tiers ----------------------------------------------------------
# (tier, experiment, arm, objects) -- strict P0->P3 dispatch order.
TIERS = [
    ("P0", "E198",     "G1A2", ["box024"]),
    ("P1", "E192-ext", "A2",   ["box021", "box023"]),
    ("P2", "E198",     "G1A2", ["box021", "box023"]),
    ("P3", "E198",     "G1A2", ["box004"]),
]
TIER_RANK = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}

MANIFEST_DIR = RESULTS_E198 / "s6_downstream/manifests"
FULL_MANIFEST = MANIFEST_DIR / "e198_priority_full_manifest.tsv"
CANARY_MANIFEST = MANIFEST_DIR / "e198_priority_canary_manifest.tsv"
SENTINEL_MANIFEST = MANIFEST_DIR / "e198_priority_sentinel_manifest.tsv"
AUTHORITY_TSV = MANIFEST_DIR / "e198_factorial_authority.tsv"

FIELDS = [
    "ordinal", "tier", "experiment", "arm", "object_key", "case_id",
    "retarget_variant_id", "target_task", "target_scene",
    "trajectory", "trajectory_sha256", "contact_mask", "contact_mask_sha256",
    "override_id", "override_path", "override_sha256",
    "base_scene_act", "base_scene_sha256", "scene_act", "scene_name", "effective_scene_sha256",
    "extra_overrides", "kp_pos", "kp_rot", "gravcomp",
    "cem_samples", "cem_opt_steps", "cem_seed",
    "sentinel", "canary_representative",
    "variant", "result_npz", "outdir_npz", "config_act", "video", "log",
    "gpu_id", "status", "failure_mode", "execution_mode", "updated_at",
]


# --- small IO helpers (self-contained, no cross-experiment imports) ----------
def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def repo_path(value: str | Path) -> Path:
    path = Path(str(value))
    if path.is_file() or path.is_dir():
        return path
    text = str(value)
    for marker in ("example_datasets/", "workspace/", "logs/", "examples/"):
        if marker in text:
            return REPO / (marker + text.split(marker, 1)[1])
    return path if path.is_absolute() else REPO / path


def rel(value: str | Path) -> str:
    path = repo_path(value)
    try:
        return path.relative_to(REPO).as_posix()
    except ValueError:
        return Path(str(value)).as_posix()


def sha256(value: str | Path) -> str:
    path = repo_path(value)
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"missing/empty: {value}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_tsv(value: str | Path) -> list[dict[str, str]]:
    with repo_path(value).open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def read_with_fields(value: str | Path) -> tuple[list[dict[str, str]], list[str]]:
    with repo_path(value).open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        rows = list(reader)
        return rows, list(reader.fieldnames or [])


def serial(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def write_tsv(value: str | Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path = repo_path(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: serial(row.get(key, "")) for key in fields})


def write_json(value: str | Path, payload: Any) -> None:
    path = repo_path(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def truth(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def a2_overrides() -> str:
    return " ".join(f"{key}={value:.6f}" for key, value in A2_GATE.items())


def extra_overrides_for(arm: str, g1_scene_name: str) -> str:
    if arm == "A2":
        return a2_overrides()
    if arm == "G1A2":
        return f"scene_name={g1_scene_name} {a2_overrides()}"
    raise ValueError(f"unknown arm {arm}")


# --- authority loading -------------------------------------------------------
def _expansion_rows() -> dict[str, dict[str, str]]:
    """box021/box023 authority keyed by case_id (from E194 g1 expansion)."""
    out: dict[str, dict[str, str]] = {}
    for raw in read_tsv(G1_EXPANSION_AUTHORITY):
        if raw.get("object_key") not in {"box021", "box023"}:
            continue
        out[raw["case_id"]] = raw
    return out


def _box2404_rows() -> dict[str, dict[str, str]]:
    """box004/box024 base PRG authority keyed by case_id (from E172/E173)."""
    out: dict[str, dict[str, str]] = {}
    for object_key, (exp_id, manifest) in SOURCE_MANIFEST.items():
        by_case = {row["case_id"]: row for row in read_tsv(manifest)}
        for case_id in BOX2404_CASES[object_key]:
            if case_id not in by_case:
                raise SystemExit(f"{case_id} missing from {manifest}")
            out[case_id] = {**by_case[case_id], "_source_exp": exp_id}
    return out


def _cell_from_expansion(raw: dict[str, str], arm: str, object_key: str) -> dict[str, Any]:
    base_scene = repo_path(raw["base_scene_act"])
    g1_scene = repo_path(raw["g1_scene_act"])
    scene = g1_scene if arm == "G1A2" else base_scene
    g1_stem = Path(raw["g1_scene_act"]).stem
    return {
        "arm": arm, "object_key": object_key, "case_id": raw["case_id"],
        "retarget_variant_id": raw["retarget_variant_id"], "target_task": raw["target_task"],
        "target_scene": rel(TASK_ROOT / raw["target_task"] / "scene.xml"),
        "trajectory": rel(raw["trajectory"]), "trajectory_sha256": raw["trajectory_sha256"],
        "contact_mask": rel(raw["contact_mask"]), "contact_mask_sha256": raw["contact_mask_sha256"],
        "override_id": Path(raw["override_path"]).stem, "override_path": rel(raw["override_path"]),
        "override_sha256": raw["override_sha256"],
        "base_scene_act": rel(base_scene), "base_scene_sha256": raw["source_effective_scene_sha256"],
        "scene_act": rel(scene), "scene_name": Path(scene).stem,
        "extra_overrides": extra_overrides_for(arm, g1_stem),
        "gravcomp": 1.0 if arm == "G1A2" else 0.0,
    }


def _cell_from_box2404(raw: dict[str, str], arm: str, object_key: str) -> dict[str, Any]:
    base_scene = repo_path(raw["scene_act"])
    g1_scene = base_scene.with_name(BOX2404_G1_SIDECAR)
    scene = g1_scene if arm == "G1A2" else base_scene
    g1_stem = Path(BOX2404_G1_SIDECAR).stem
    return {
        "arm": arm, "object_key": object_key, "case_id": raw["case_id"],
        "retarget_variant_id": raw.get("retarget_variant_id", ""), "target_task": raw["target_task"],
        "target_scene": rel(raw.get("target_scene", TASK_ROOT / raw["target_task"] / "scene.xml")),
        "trajectory": rel(raw["trajectory"]), "trajectory_sha256": raw["trajectory_sha256"],
        "contact_mask": rel(raw["contact_mask"]), "contact_mask_sha256": raw["contact_mask_sha256"],
        "override_id": raw["override_id"], "override_path": rel(raw["override_path"]),
        "override_sha256": raw["override_sha256"],
        "base_scene_act": rel(base_scene), "base_scene_sha256": raw.get("scene_sha256", ""),
        "scene_act": rel(scene), "scene_name": Path(scene).stem,
        "extra_overrides": extra_overrides_for(arm, g1_stem),
        "gravcomp": 1.0 if arm == "G1A2" else 0.0,
    }


def build_cells() -> list[dict[str, Any]]:
    """Return the 103 run cells in strict tier order (P0->P3, then object,case_id)."""
    expansion = _expansion_rows()
    box2404 = _box2404_rows()
    cells: list[dict[str, Any]] = []
    for tier, experiment, arm, objects in TIERS:
        for object_key in objects:
            if object_key in ("box021", "box023"):
                case_ids = sorted(cid for cid, r in expansion.items() if r["object_key"] == object_key)
                for case_id in case_ids:
                    cell = _cell_from_expansion(expansion[case_id], arm, object_key)
                    cell.update({"tier": tier, "experiment": experiment})
                    cells.append(cell)
            else:  # box024 / box004
                for case_id in sorted(BOX2404_CASES[object_key]):
                    cell = _cell_from_box2404(box2404[case_id], arm, object_key)
                    cell.update({"tier": tier, "experiment": experiment})
                    cells.append(cell)
    return cells
