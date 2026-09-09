#!/usr/bin/env python3
"""Paper results: SPIDER-CEM vs OmniRetarget metric aggregation over the paper case set.

For every case in ``tmp/paper_case_id.txt`` this resolves the user-fixed SPIDER-CEM
"selected rollout" per object group, plus the matched OmniRetarget kinematic trajectory,
and reports a single consistent metric table (per-case + per-object + overall) in
Markdown / xlsx / LaTeX.

Policy (user choice): reuse precomputed ``*_case_metrics.tsv`` values where the column
exists; recompute the gaps from the case's selected ``cem_result_npz`` via the shared
``eval.core.core_metrics.evaluate_sequence`` + ``motion_health.run_health`` (identical
definitions).  OmniRetarget physics metrics are obtained by MuJoCo position-servo replay
of its kinematic trajectory (same protocol as E197/E109).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "workspace/core4d/scripts"))

from eval.core.core_metrics import (  # noqa: E402
    EVAL_METRIC_STANDARD_ID,
    EvalConfig,
    evaluate_sequence,
    person_idx_from_case,
)
from eval.core.motion_health import run_health  # noqa: E402

# Reuse the proven OmniRetarget world-pose → scene_act converter/replayer verbatim.
sys.path.insert(0, str(REPO / "workspace/core4d/scripts/eval/reports"))
from gen_E197_full_cem_omnirt_vs_prg_metrics import (  # noqa: E402
    angular_error_deg,  # noqa: F401  (imported for parity / potential reuse)
    compiled_euler_convention,  # noqa: F401
    convert_omni as _e197_convert_omni,
)
from eval.runners.eval_E194_G1_expansion import fixed_reference_z_metrics  # noqa: E402

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

CASE_FILE = REPO / "tmp/paper_case_id.txt"
OUT = REPO / "workspace/core4d/report/0908/paper_results"
OMNI_QPOS_DIR = OUT / "omni_scene_act_qpos"
CACHE = OUT / "_cache_method_metrics.jsonl"
MISSING_MOUNT = "/mnt/a0ccc676-9496-49f8-a861-f8a1797dec52/spider_workdirs"

HAND_COLLISION_VARIANT = "rubber_hull"

# rl-export tsv column names shared across every source experiment.
RL_TRAJECTORY = "trajectory"        # kinematic reference npz
RL_SCENE_ACT = "scene_act"          # physics scene xml
RL_CONTACT_MASK = "contact_mask"    # raw 3cm mask npz
RL_CEM_NPZ = "cem_result_npz"       # selected SPIDER-CEM rollout npz
RL_ARM = "arm"
RL_PARTNER_OMNI = "partner_omniretarget_output_npz"

# Per object-group SPIDER-CEM source (version selection fixed by the user).
#   rl:     selected-rollout authority tsv (row keyed on case_id, optionally + arm)
#   table:  precomputed *_case_metrics.tsv to reuse
#   suffix: factorial cell suffix appended to metric column names (E198 only)
#   arm_key: rl column that selects the arm inside `table` (E212/E207)
GROUPS: dict[str, dict[str, Any]] = {
    "box021": {
        "exp": "E170_PRG",
        "rl": "workspace/core4d/results/E170/s6_downstream/rl_export/paired_rl_export_input.tsv",
        "table": "workspace/core4d/results/E170/s6_downstream/eval/full/e170_case_metrics.tsv",
        "suffix": "",
        "arm_key": None,
    },
    "box023": {
        "exp": "E190_noPRG(E179)",
        "rl": "workspace/core4d/results/E190/s6_downstream/rl_export/box023_noPRG_user_approved/paired_rl_export_input.tsv",
        "table": "workspace/core4d/results/E179/s6_downstream/eval/full/e179_case_metrics.tsv",
        "suffix": "",
        "arm_key": None,
    },
    "box001": {
        "exp": "E198_G1A2",
        "rl": "workspace/core4d/results/E198/s6_downstream/rl_export/box001_user_approved/rl_export_input.tsv",
        "table": "workspace/core4d/results/E198/s6_downstream/eval/full_factorial/e198_factorial_by_case.tsv",
        "suffix": "__G1A2",
        "arm_key": None,
    },
    "box004": {
        "exp": "E198_G1A2",
        "rl": "workspace/core4d/results/E198/s6_downstream/rl_export/box004_user_approved/rl_export_input.tsv",
        "table": "workspace/core4d/results/E198/s6_downstream/eval/full_factorial/e198_factorial_by_case.tsv",
        "suffix": "__G1A2",
        "arm_key": None,
    },
    "box024": {
        "exp": "E198_G1A2",
        "rl": "workspace/core4d/results/E198/s6_downstream/rl_export/box024_user_approved/rl_export_input.tsv",
        "table": "workspace/core4d/results/E198/s6_downstream/eval/full_factorial/e198_factorial_by_case.tsv",
        "suffix": "__G1A2",
        "arm_key": None,
    },
    "bucket003": {
        "exp": "E178",
        "rl": "workspace/core4d/results/E178/s6_downstream/rl_export/paired_rl_export_input.tsv",
        "table": "workspace/core4d/results/E178/s6_downstream/eval/full/e178_case_metrics.tsv",
        "suffix": "",
        "arm_key": None,
    },
    "bucket007": {
        "exp": "E207",
        "rl": "workspace/core4d/results/E207/s6_downstream/rl_export/paired_rl_export_input.tsv",
        "table": "workspace/core4d/results/E207/s6_downstream/eval/four_arm/e207_arm_case_metrics.tsv",
        "suffix": "",
        "arm_key": "arm",
    },
}
# desk/chair share E212.
for _obj in ("chair006", "desk007", "desk021", "desk023"):
    GROUPS[_obj] = {
        "exp": "E212",
        "rl": "workspace/core4d/results/E212/s6_downstream/rl_export/paired_rl_export_input.tsv",
        "table": "workspace/core4d/results/E212/s6_downstream/eval/g_sweep/e212_g_sweep_rollout.tsv",
        "suffix": "",
        "arm_key": "arm",
    }

# Per-case overrides: cases whose selected rollout lives outside the object-group source.
# These two box021 cases are not in E170; the user selected the E167A (zOnlyBody) variant
# from the E167 holosoma_zonly export.  table=None forces recompute from cem_result_npz.
_E167_RL = "workspace/core4d/results/E167/holosoma_zonly/rl_export/s6_downstream/rl_export/rl_export_input.tsv"
_E167_TABLE = "workspace/core4d/results/E167/holosoma_zonly/eval/cem_metrics/full/e167_arm_metrics.tsv"
CASE_OVERRIDES: dict[str, dict[str, Any]] = {
    "box021_20231018_029_p2": {"exp": "E167A_zOnlyBody", "rl": _E167_RL,
                               "rl_case_id": "d003_box021_20231018_029_p2__E167_E167A",
                               "table": None, "suffix": "", "arm_key": None,
                               "e167_table": _E167_TABLE, "e167_short": "box021_029_p2", "e167_arm": "E167A"},
    "box021_20231011_035_p1": {"exp": "E167A_zOnlyBody", "rl": _E167_RL,
                               "rl_case_id": "d003_box021_20231011_035_p1__E167_E167A",
                               "table": None, "suffix": "", "arm_key": None,
                               "e167_table": _E167_TABLE, "e167_short": "box021_035_p1", "e167_arm": "E167A"},
}


def case_cfg(case_id: str) -> dict[str, Any]:
    """Resolve the source config for a case (per-case override wins over object group)."""
    if case_id in CASE_OVERRIDES:
        return CASE_OVERRIDES[case_id]
    cfg = dict(GROUPS[object_key_of(case_id)])
    cfg["rl_case_id"] = case_id
    return cfg


# Object display order for grouping.
OBJECT_ORDER = ("box001", "box004", "box021", "box023", "box024", "bucket003", "bucket007",
                "chair006", "desk007", "desk021", "desk023")

# Metric spec.  src: where a *recompute* value comes from (evaluate_sequence result /
# run_health / None=table-only).  kind: pct (frame fraction 0..1) / cm / deg / m / mps /
# jerk / flag.  omni=True -> also reported for OmniRetarget.
Metric = dict[str, Any]
METRICS: list[Metric] = [
    {"key": "raw_contact",   "field": "hand_object_physics_contact_in_mask_frac",        "src": "result", "kind": "pct", "dir": "higher", "omni": True,  "label": "raw contact"},
    {"key": "contact_3mm",   "field": "hand_object_physics_contact_3mm_in_mask_frac",     "src": "result", "kind": "pct", "dir": "higher", "omni": True,  "label": "physical contact@3mm"},
    {"key": "pen_3mm",       "field": "hand_object_physics_penetration_3mm_frame_frac",   "src": "result", "kind": "pct", "dir": "lower",  "omni": True,  "label": "physical penetration@3mm"},
    {"key": "geom_2mm",      "field": "hand_geom_penetration_2mm_frac",                   "src": "result", "kind": "pct", "dir": "lower",  "omni": True,  "label": "geometric penetration@2mm"},
    {"key": "track_root_pos", "field": "track_root_pos_err_cm_mean",                      "src": "result", "kind": "cm",  "dir": "lower",  "omni": False, "label": "root pos err"},
    {"key": "track_root_ori", "field": "track_root_ori_err_deg_mean",                     "src": "result", "kind": "deg", "dir": "lower",  "omni": False, "label": "root ori err"},
    {"key": "track_eef_pos",  "field": "track_eef_pos_err_cm_mean",                       "src": "result", "kind": "cm",  "dir": "lower",  "omni": False, "label": "eef pos err"},
    {"key": "track_eef_ori",  "field": "track_eef_ori_err_deg_mean",                      "src": "result", "kind": "deg", "dir": "lower",  "omni": False, "label": "eef ori err"},
    {"key": "track_obj_pos",  "field": "track_obj_pos_err_cm_mean",                       "src": "result", "kind": "cm",  "dir": "lower",  "omni": False, "label": "obj pos err"},
    {"key": "track_obj_ori",  "field": "track_obj_ori_err_deg_mean",                      "src": "result", "kind": "deg", "dir": "lower",  "omni": False, "label": "obj ori err"},
    {"key": "fall_flag",      "field": "fall_flag",                                       "src": "result", "kind": "flag", "dir": "lower", "omni": False, "label": "fall"},
    {"key": "body_z",         "field": "body_z_err_p95_m",                                "src": "zmetric", "kind": "m",  "dir": "lower",  "omni": False, "label": "body-z err p95"},
    {"key": "ankle_jerk",     "field": "ankle_jerk_p95",                                  "src": "health", "kind": "jerk", "dir": "lower", "omni": False, "label": "ankle jerk p95"},
    {"key": "obj_speed",      "field": "obj_speed_max",                                   "src": "health", "kind": "mps", "dir": "lower",  "omni": False, "label": "obj speed max"},
    {"key": "foot_slip",      "field": "foot_slip_max_m",                                 "src": "health", "kind": "m",   "dir": "lower",  "omni": False, "label": "foot slip max"},
]
METRIC_BY_KEY = {m["key"]: m for m in METRICS}
OMNI_KEYS = [m["key"] for m in METRICS if m["omni"]]
SPIDER_KEYS = [m["key"] for m in METRICS]
# Metrics that must be present in a table for a case to reuse without any recompute.
REUSE_REQUIRED = [m["key"] for m in METRICS if m["src"] in ("result", "health", "zmetric")]


# --------------------------------------------------------------------------------------
# small utilities
# --------------------------------------------------------------------------------------

def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def numeric(value: Any) -> float:
    if isinstance(value, str):
        low = value.strip().lower()
        if low == "true":
            return 1.0
        if low == "false":
            return 0.0
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def remap(value: str) -> str:
    """Remap the missing /mnt/a0ccc676 mount to the on-disk workspace path.

    ``.../spider_workdirs/core4d/...`` lives on disk at ``<repo>/workspace/core4d/...``.
    """
    if value.startswith(MISSING_MOUNT):
        tail = value[len(MISSING_MOUNT):].lstrip("/")
        cand = REPO / "workspace" / tail
        if cand.exists() or tail.startswith("core4d/"):
            return str(cand)
        return str(REPO / tail)
    return value


def resolve(value: str | Path) -> Path | None:
    """Resolve a possibly-relative / mount-prefixed path to an existing file."""
    if value in (None, ""):
        return None
    text = remap(str(value))
    p = Path(text)
    if p.is_file():
        return p.resolve()
    if not p.is_absolute() and (REPO / p).is_file():
        return (REPO / p).resolve()
    for marker in ("workspace/core4d/", "example_datasets/", "logs/"):
        if marker in text:
            cand = REPO / (marker + text.split(marker, 1)[1])
            if cand.is_file():
                return cand.resolve()
    return None


def rel(path: Path | str | None) -> str:
    if path is None:
        return ""
    p = Path(path)
    try:
        return str(p.resolve().relative_to(REPO))
    except ValueError:
        return str(p)


def object_key_of(case_id: str) -> str:
    return case_id.split("_", 1)[0]


# --------------------------------------------------------------------------------------
# source resolution
# --------------------------------------------------------------------------------------

class SourceIndex:
    """Lazily-loaded rl-export rows and reuse tables, indexed by case_id."""

    def __init__(self) -> None:
        self._rl: dict[str, list[dict[str, str]]] = {}
        self._table: dict[str, list[dict[str, str]]] = {}
        self._table_cols: dict[str, set[str]] = {}

    def rl_rows(self, path: str) -> list[dict[str, str]]:
        if path not in self._rl:
            self._rl[path] = read_tsv(REPO / path)
        return self._rl[path]

    def table(self, path: str) -> tuple[list[dict[str, str]], set[str]]:
        if path not in self._table:
            rows = read_tsv(REPO / path)
            self._table[path] = rows
            self._table_cols[path] = set(rows[0].keys()) if rows else set()
        return self._table[path], self._table_cols[path]


def find_rl_row(idx: SourceIndex, case_id: str, cfg: dict[str, Any]) -> dict[str, str] | None:
    key = cfg.get("rl_case_id", case_id)
    rows = [r for r in idx.rl_rows(cfg["rl"]) if r.get("case_id") == key]
    if not rows:
        return None
    return rows[0]


def find_table_row(
    idx: SourceIndex, case_id: str, cfg: dict[str, Any], rl_row: dict[str, str],
    cem_npz: Path | None,
) -> dict[str, str] | None:
    if not cfg.get("table"):
        return None
    rows, _cols = idx.table(cfg["table"])
    cands = [r for r in rows if r.get("case_id") == case_id]
    if not cands:
        return None
    # Disambiguate multi-row tables.
    arm_key = cfg.get("arm_key")
    if arm_key and rl_row.get(arm_key):
        arm = rl_row[arm_key]
        armed = [r for r in cands if r.get("arm") == arm]
        if armed:
            cands = armed
    if len(cands) > 1 and cem_npz is not None:
        base = cem_npz.name
        matched = [
            r for r in cands
            if base in " ".join(str(r.get(c, "")) for c in ("result_npz", "outdir_npz", "qpos_path", "cem_result_npz"))
        ]
        if matched:
            cands = matched
    return cands[0]


def table_field(cols: set[str], base_field: str, suffix: str) -> str | None:
    if suffix and (base_field + suffix) in cols:
        return base_field + suffix
    if base_field in cols:
        return base_field
    return None


# --------------------------------------------------------------------------------------
# metric computation
# --------------------------------------------------------------------------------------

def eval_row_for(case_id: str) -> dict[str, str]:
    obj = object_key_of(case_id)
    cat = "box" if obj.startswith("box") else ("bucket" if obj.startswith("bucket") else "furniture")
    return {
        "case_id": case_id,
        "variant": f"paper_{case_id}",
        "object_key": obj,
        "object_category": cat,
        "expected_quality": "paper_selected",
    }


def recompute_metrics(
    case_id: str, method: str, qpos: Path, scene: Path, kin_ref: Path | None, mask: Path | None,
    keys: list[str],
) -> dict[str, float]:
    """Run evaluate_sequence (+ run_health if any health key requested) once."""
    srcs = {METRIC_BY_KEY[k]["src"] for k in keys}
    out: dict[str, float] = {}
    if "result" in srcs:
        result = evaluate_sequence(
            row=eval_row_for(case_id),
            method=method,
            hand_collision_variant_id=HAND_COLLISION_VARIANT,
            qpos_path=qpos,
            scene_xml=scene,
            kin_ref_path=kin_ref,
            contact_mask_path=mask,
            person_idx=person_idx_from_case(case_id),
        )
        for k in keys:
            if METRIC_BY_KEY[k]["src"] == "result":
                out[k] = numeric(result.get(METRIC_BY_KEY[k]["field"]))
    if "health" in srcs:
        health = run_health(qpos, scene, EvalConfig())
        for k in keys:
            if METRIC_BY_KEY[k]["src"] == "health":
                out[k] = numeric(health.get(METRIC_BY_KEY[k]["field"]))
    if "zmetric" in srcs:
        if kin_ref is None:
            for k in keys:
                if METRIC_BY_KEY[k]["src"] == "zmetric":
                    out[k] = math.nan
        else:
            z = fixed_reference_z_metrics(qpos, scene, kin_ref)
            for k in keys:
                if METRIC_BY_KEY[k]["src"] == "zmetric":
                    out[k] = numeric(z.get(METRIC_BY_KEY[k]["field"]))
    return out


def compute_spider(idx: SourceIndex, case_id: str) -> dict[str, Any]:
    obj = object_key_of(case_id)
    cfg = case_cfg(case_id)
    rl_row = find_rl_row(idx, case_id, cfg) or {}
    rec: dict[str, Any] = {"case_id": case_id, "object_key": obj, "method": "SPIDER-CEM",
                           "source_exp": cfg["exp"], "status": "ok", "notes": []}
    cem_npz = resolve(rl_row.get(RL_CEM_NPZ, ""))
    scene = resolve(rl_row.get(RL_SCENE_ACT, ""))
    kin_ref = resolve(rl_row.get(RL_TRAJECTORY, ""))
    mask = resolve(rl_row.get(RL_CONTACT_MASK, ""))
    rec["cem_npz"] = rel(cem_npz)
    rec["scene_act"] = rel(scene)

    # E167 override: reuse the E167 arm-metrics table (its mask alignment) + recompute body_z.
    if cfg.get("e167_table"):
        return _spider_from_e167(idx, case_id, cfg, cem_npz, scene, kin_ref, rec)

    cols: set[str] = idx.table(cfg["table"])[1] if cfg.get("table") else set()
    trow = find_table_row(idx, case_id, cfg, rl_row, cem_npz)
    suffix = cfg["suffix"]

    # reuse values available in the table
    reused: dict[str, float] = {}
    reuse_src: dict[str, str] = {}
    if trow is not None:
        for k in SPIDER_KEYS:
            fld = table_field(cols, METRIC_BY_KEY[k]["field"], suffix)
            if fld is not None and str(trow.get(fld, "")).strip() != "":
                reused[k] = numeric(trow[fld])
                reuse_src[k] = fld

    # decide recompute
    missing = [k for k in REUSE_REQUIRED if k not in reused]
    provenance: dict[str, str] = {}
    values: dict[str, float] = {}
    xcheck: dict[str, float] = {}

    if missing and cem_npz is not None and scene is not None:
        recomputed = recompute_metrics(case_id, "SPIDER-CEM", cem_npz, scene, kin_ref, mask,
                                       keys=[k for k in SPIDER_KEYS if METRIC_BY_KEY[k]["src"]])
        for k in SPIDER_KEYS:
            m = METRIC_BY_KEY[k]
            if m["src"] is None:
                if k in reused:
                    values[k] = reused[k]; provenance[k] = f"table:{reuse_src[k]}"
                else:
                    values[k] = math.nan; provenance[k] = "MISSING"
            else:
                values[k] = recomputed.get(k, math.nan)
                provenance[k] = "recompute"
                if k in reused and math.isfinite(reused[k]) and math.isfinite(values[k]):
                    xcheck[k] = abs(reused[k] - values[k])
    else:
        # pure reuse
        for k in SPIDER_KEYS:
            if k in reused:
                values[k] = reused[k]; provenance[k] = f"table:{reuse_src[k]}"
            else:
                values[k] = math.nan; provenance[k] = "MISSING"
        if trow is None:
            rec["status"] = "UNRESOLVED"
            rec["notes"].append("no rl-row and no case_metrics table row")
        elif missing and (cem_npz is None or scene is None):
            rec["status"] = "REUSE_INCOMPLETE_NO_NPZ"
            rec["notes"].append(f"missing {missing}; npz/scene unresolved")

    rec["metrics"] = values
    rec["provenance"] = provenance
    rec["xcheck_abs"] = xcheck
    return rec


_OMNI_INDEX: dict[str, list[Path]] | None = None
OMNI_SEARCH_ROOTS = (
    "workspace/core4d/results/E206/s3_retarget",
    "workspace/core4d/results/E173/s3_retarget",
    "workspace/core4d/results/E178/s6_downstream",
    "workspace/core4d/results/E207/s6_downstream",
    "workspace/core4d/results/E198/s6_downstream",
    "workspace/core4d/results/E170/s6_downstream",
)
# holosoma roots (sibling repo) — omniretarget production trees.
OMNI_SEARCH_ROOTS_ABS = (
    "holosoma/workspace/v3/data_construction",
    "holosoma/workspace/v3/data_construction_v2",
    "holosoma/workspace/v2/results",
    "holosoma/workspace/pipeline/results",
)


def _omni_index() -> dict[str, list[Path]]:
    """One-time index: filename -> list of on-disk *_with_obj_original.npz paths."""
    global _OMNI_INDEX
    if _OMNI_INDEX is not None:
        return _OMNI_INDEX
    index: dict[str, list[Path]] = defaultdict(list)
    roots = [REPO / r for r in OMNI_SEARCH_ROOTS] + [REPO.parent / r for r in OMNI_SEARCH_ROOTS_ABS]
    for root in roots:
        if not root.exists():
            continue
        for cand in root.rglob("*_with_obj_original.npz"):
            index[cand.name.lower()].append(cand)
    _OMNI_INDEX = index
    return index


E167_FIELD_MAP = {  # metric key -> column in e167_arm_metrics.tsv (body_z absent -> recompute)
    "raw_contact": "hand_object_physics_contact_in_mask_frac",
    "contact_3mm": "hand_object_physics_contact_3mm_in_mask_frac",
    "pen_3mm": "hand_object_physics_penetration_3mm_frame_frac",
    "geom_2mm": "hand_geom_penetration_2mm_frac",
    "track_root_pos": "track_root_pos_err_cm_mean", "track_root_ori": "track_root_ori_err_deg_mean",
    "track_eef_pos": "track_eef_pos_err_cm_mean", "track_eef_ori": "track_eef_ori_err_deg_mean",
    "track_obj_pos": "track_obj_pos_err_cm_mean", "track_obj_ori": "track_obj_ori_err_deg_mean",
    "fall_flag": "fall_flag", "ankle_jerk": "ankle_jerk_p95", "obj_speed": "obj_speed_max",
    "foot_slip": "foot_slip_max_m",
}


def _spider_from_e167(idx: SourceIndex, case_id: str, cfg: dict[str, Any],
                      cem_npz: Path | None, scene: Path | None, kin_ref: Path | None,
                      rec: dict[str, Any]) -> dict[str, Any]:
    rows = [r for r in idx.rl_rows(cfg["e167_table"])
            if r.get("short_case_id") == cfg["e167_short"] and r.get("arm") == cfg["e167_arm"]]
    values: dict[str, float] = {}
    provenance: dict[str, str] = {}
    if rows:
        trow = rows[0]
        for k, col in E167_FIELD_MAP.items():
            values[k] = numeric(trow.get(col))
            provenance[k] = f"table:e167:{col}"
    else:
        rec["status"] = "UNRESOLVED"; rec["notes"].append("no E167 arm-metrics row")
    # body_z recomputed (no contact mask needed)
    if cem_npz is not None and scene is not None and kin_ref is not None:
        z = recompute_metrics(case_id, "SPIDER-CEM", cem_npz, scene, kin_ref, None, keys=["body_z"])
        values["body_z"] = z.get("body_z", math.nan)
        provenance["body_z"] = "recompute"
    else:
        values["body_z"] = math.nan; provenance["body_z"] = "MISSING"
    rec["metrics"] = values
    rec["provenance"] = provenance
    rec["xcheck_abs"] = {}
    return rec


def _aligned_mask(mask_path: Path, n_frames: int) -> Path:
    """Return a mask npz whose spider_contact_mask_3cm has n_frames rows.

    E167-era masks were built for a different trim window; nearest-neighbour resample the
    boolean contact mask onto the rollout timeline so in-mask fractions are defined.
    """
    data = dict(np.load(mask_path, allow_pickle=True))
    sm = np.asarray(data["spider_contact_mask_3cm"])
    if sm.shape[0] == n_frames:
        return mask_path
    idxs = np.rint(np.linspace(0, sm.shape[0] - 1, n_frames)).astype(int)
    data["spider_contact_mask_3cm"] = sm[idxs]
    OMNI_QPOS_DIR.mkdir(parents=True, exist_ok=True)
    out = OMNI_QPOS_DIR / f"_resampled_mask_{mask_path.parent.name}_{n_frames}.npz"
    np.savez_compressed(out, **data)
    return out


def resolve_omnirt_npz(case_id: str, rl_row: dict[str, str] | None) -> Path | None:
    """Locate the raw OmniRetarget kinematic npz for this case's own person.

    Naming convention: case_id ``<obj>_<date...>_<seq>_p<idx>`` maps to
    ``<date...>-<seq>-person<idx>-<Obj>_with_obj_original.npz``.
    """
    obj = object_key_of(case_id)
    parts = case_id.split("_")
    pidx = parts[-1][-1]
    seq = parts[-2]
    date_str = "_".join(parts[1:-2])
    date_dash = date_str.replace("_", "_")  # keep sub-index token as-is
    obj_variants = {obj, obj.capitalize(), obj[:-3].capitalize() + obj[-3:], obj.upper()}
    infix = f"-{seq}-person{pidx}-"
    index = _omni_index()
    matches: list[Path] = []
    for ov in obj_variants:
        key = f"{date_str}-{seq}-person{pidx}-{ov}_with_obj_original.npz".lower()
        matches.extend(index.get(key, []))
    if not matches:
        # looser: any indexed file whose name matches infix + object variant + date prefix
        want_dates = {date_str, date_str.split("_")[0]}
        for name_lower, paths in index.items():
            if infix.lower() not in name_lower:
                continue
            if not any(name_lower.endswith(f"-{ov.lower()}_with_obj_original.npz") for ov in obj_variants):
                continue
            if not any(name_lower.startswith(d.lower() + "-") for d in want_dates):
                continue
            matches.extend(paths)
    # prefer trimmed > retargeted; prefer v3 production trees; shortest path
    def rank(p: Path) -> tuple[int, int, int]:
        s = str(p)
        return (0 if "trimmed" in s else 1,
                0 if ("d003" in s or "stage2b" in s or "E206" in s or "E173" in s) else 1,
                len(s))
    matches = sorted(set(matches), key=rank)
    return matches[0] if matches else None


def compute_omnirt(idx: SourceIndex, case_id: str, spider_rec: dict[str, Any]) -> dict[str, Any]:
    obj = object_key_of(case_id)
    cfg = case_cfg(case_id)
    rl_row = find_rl_row(idx, case_id, cfg)
    rec: dict[str, Any] = {"case_id": case_id, "object_key": obj, "method": "OmniRetarget",
                           "status": "ok", "notes": []}
    scene = resolve(rl_row.get(RL_SCENE_ACT, "")) if rl_row else None
    kin_ref = resolve(rl_row.get(RL_TRAJECTORY, "")) if rl_row else None
    mask = resolve(rl_row.get(RL_CONTACT_MASK, "")) if rl_row else None
    if scene is None:
        rec["status"] = "UNRESOLVED"; rec["notes"].append("no scene_act"); return rec
    # The SPIDER kinematic reference IS the OmniRetarget output in raw world-freejoint
    # form (nq_robot + 7), the exact convert_omni input, in the same scene/window/mask as
    # SPIDER-CEM.  Prefer it; fall back to a filesystem-resolved raw npz only if absent.
    raw = kin_ref if kin_ref is not None else resolve_omnirt_npz(case_id, rl_row)
    if raw is None:
        rec["status"] = "UNRESOLVED"; rec["notes"].append("omnirt trajectory not found"); return rec
    rec["omnirt_raw_npz"] = rel(raw)
    rec["scene_act"] = rel(scene)
    try:
        OMNI_QPOS_DIR.mkdir(parents=True, exist_ok=True)
        # reuse E197 converter by pointing its OMNI_DIR at ours
        import gen_E197_full_cem_omnirt_vs_prg_metrics as e197
        e197.OMNI_DIR = OMNI_QPOS_DIR
        omni_qpos, conv = _e197_convert_omni({"case_id": case_id}, raw, scene)
        rec["euler_convention"] = conv["euler_convention"]
        rec["roundtrip_pos_cm_max"] = conv["roundtrip_position_error_cm_max"]
        rec["roundtrip_ori_deg_max"] = conv["roundtrip_orientation_error_deg_max"]
    except Exception as exc:  # noqa: BLE001 - report, do not crash the whole run
        rec["status"] = "REPLAY_FAILED"; rec["notes"].append(str(exc)); return rec
    # Align a mismatched (E167-era) mask to the rollout timeline so in-mask metrics exist.
    if mask is not None:
        try:
            aligned = _aligned_mask(mask, int(conv["frames"]))
            if aligned != mask:
                rec["notes"].append("mask_resampled_to_rollout")
            mask = aligned
        except Exception as exc:  # noqa: BLE001
            rec["notes"].append(f"mask_align_failed:{exc}")
    recomputed = recompute_metrics(case_id, "OmniRetarget", omni_qpos, scene, kin_ref, mask,
                                   keys=OMNI_KEYS)
    rec["metrics"] = {k: recomputed.get(k, math.nan) for k in OMNI_KEYS}
    rec["provenance"] = {k: "recompute" for k in OMNI_KEYS}
    return rec


# --------------------------------------------------------------------------------------
# cache
# --------------------------------------------------------------------------------------

def load_cache() -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    if CACHE.is_file():
        for line in CACHE.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            out[(rec["case_id"], rec["method"])] = rec
    return out


def append_cache(rec: dict[str, Any]) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with CACHE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False, default=_json_default) + "\n")


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# --------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------

def load_cases(limit: int | None, only_object: str | None) -> list[str]:
    cases = [ln.strip() for ln in CASE_FILE.read_text().splitlines() if ln.strip()]
    if only_object:
        cases = [c for c in cases if object_key_of(c) == only_object]
    if limit:
        cases = cases[:limit]
    return cases


def compute_all(cases: list[str], do_spider: bool, do_omni: bool, use_cache: bool) -> list[dict[str, Any]]:
    idx = SourceIndex()
    cache = load_cache() if use_cache else {}
    records: list[dict[str, Any]] = []
    for i, case_id in enumerate(cases, 1):
        if do_spider:
            key = (case_id, "SPIDER-CEM")
            if key in cache:
                rec = cache[key]
            else:
                rec = compute_spider(idx, case_id)
                append_cache(rec)
            records.append(rec)
            print(f"[{i:02d}/{len(cases)}] SPIDER  {case_id} {rec['status']}", flush=True)
        if do_omni:
            key = (case_id, "OmniRetarget")
            if key in cache:
                rec = cache[key]
            else:
                rec = compute_omnirt(idx, case_id, {})
                append_cache(rec)
            records.append(rec)
            print(f"[{i:02d}/{len(cases)}] OMNI    {case_id} {rec['status']}", flush=True)
    return records


def audit_all(cases: list[str]) -> list[dict[str, Any]]:
    """Fast path-resolution audit (no MuJoCo replay)."""
    idx = SourceIndex()
    rows: list[dict[str, Any]] = []
    for case_id in cases:
        obj = object_key_of(case_id)
        cfg = case_cfg(case_id)
        rl_row = find_rl_row(idx, case_id, cfg) or {}
        cem_npz = resolve(rl_row.get(RL_CEM_NPZ, ""))
        scene = resolve(rl_row.get(RL_SCENE_ACT, ""))
        kin_ref = resolve(rl_row.get(RL_TRAJECTORY, ""))
        mask = resolve(rl_row.get(RL_CONTACT_MASK, ""))
        trow = find_table_row(idx, case_id, cfg, rl_row, cem_npz)
        cols = idx.table(cfg["table"])[1] if cfg.get("table") else set()
        reuse_ok = trow is not None and all(
            table_field(cols, METRIC_BY_KEY[k]["field"], cfg["suffix"]) is not None for k in REUSE_REQUIRED
        )
        omni = resolve_omnirt_npz(case_id, rl_row)
        rows.append({
            "case_id": case_id, "object_key": obj, "exp": cfg["exp"],
            "rl_row": bool(rl_row), "table_row": trow is not None, "reuse_all": reuse_ok,
            "cem_npz": bool(cem_npz), "scene": bool(scene), "kin_ref": bool(kin_ref),
            "mask": bool(mask), "omni_npz": bool(omni),
            "omni_path": rel(omni), "cem_path": rel(cem_npz),
        })
    return rows


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", action="store_true", help="resolve paths only, no replay")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only-object", default=None)
    ap.add_argument("--no-spider", action="store_true")
    ap.add_argument("--no-omni", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="ignore cache")
    ap.add_argument("--compute-only", action="store_true", help="skip report writing")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.fresh and CACHE.is_file():
        CACHE.unlink()
    cases = load_cases(args.limit, args.only_object)
    if args.audit:
        rows = audit_all(cases)
        bad = [r for r in rows if not (r["reuse_all"] or (r["cem_npz"] and r["scene"]))]
        no_omni = [r for r in rows if not r["omni_npz"]]
        for r in rows:
            print(f"{r['case_id']:32s} exp={r['exp']:16s} rl={int(r['rl_row'])} tbl={int(r['table_row'])} "
                  f"reuse_all={int(r['reuse_all'])} cem={int(r['cem_npz'])} scn={int(r['scene'])} "
                  f"kin={int(r['kin_ref'])} msk={int(r['mask'])} omni={int(r['omni_npz'])}")
        print(f"\nTOTAL {len(rows)} | SPIDER unresolved {len(bad)}: {[r['case_id'] for r in bad]}")
        print(f"OMNI unresolved {len(no_omni)}: {[r['case_id'] for r in no_omni]}")
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "audit.json").write_text(json.dumps(rows, indent=2, default=_json_default), encoding="utf-8")
        return 0
    records = compute_all(cases, not args.no_spider, not args.no_omni, use_cache=not args.fresh)
    if args.compute_only:
        print(json.dumps({"records": len(records), "cases": len(cases)}, indent=2))
        return 0
    # report writers are added in the next stage
    from paper_report import write_all_reports  # noqa: PLC0415
    write_all_reports(records, cases)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
