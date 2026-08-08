#!/usr/bin/env python3
"""Frozen contract for E194 object gravity-compensation 2x2 (mechanism (b)).

E194 asks whether the servo sag on CORE4D's position-actuated object
(sag = m*g/kp) is fixable, and -- crucially -- whether hand->object penetration
comes from the object being in the wrong PLACE or from the servo actively
PUSHING it.  The 2x2 separates the two:

    arm  kp     gravcomp   steady displacement   steady servo force
    A0   500    0          -9.81 cm              49.05 N     (baseline)
    G1   500    1           0.00 cm               0    N     (gravity comp)
    G2   2500   0          -1.96 cm              49.05 N     (stiffer servo)
    G3   2500   1           0.00 cm               0    N     (both)

All three arms only touch the *translational* gain; `init_rot_actuator_gain`
stays 50 for every arm (see C3 -- we must not stiffen rotation, or the
"centre-of-mass force cannot fix distal tilt" test is confounded).

Every arm reuses the A0 baseline's PRG override (`+override=core4d_E173/E172_*_PRG`)
and differs ONLY by extra CLI tokens appended to run_mjwp:

    A0  reuse E172/E173 landed results (NOT re-run here)
    G1  scene_name=scene_act_E194_rubberHull_PRG_gravcomp
    G2  init_pos_actuator_gain=2500
    G3  scene_name=scene_act_E194_rubberHull_PRG_gravcomp init_pos_actuator_gain=2500

The gravcomp sidecar is the base E173/E172 PRG sidecar with `gravcomp="1"`
injected on the object body -- built per-case into the same case directory by
build_gravcomp_manifest.py.  Verified against source at plan time:
  - run_mjwp.py:1102-1108   scene_name -> model_path selects the sidecar
  - run_mjwp.py:1217-1227   init_pos_actuator_gain drives non-`_rot_` kp
  - Hydra compose of the base PRG override yields kp=500 / rot=50 / PRG on
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
SCRIPT_DIR = REPO / "workspace/core4d/scripts/experiments/E194"
RESULTS = REPO / "workspace/core4d/results/E194"

E194_METHOD_ID = "E167A_zOnlyBody_PRG_gravcomp2x2"
HAND_COLLISION_VARIANT_ID = "rubber_hull"

# --- the gravity-compensation sidecar --------------------------------------
# Built per-case by build_gravcomp_manifest.py: base PRG sidecar + object body
# gravcomp="1".  The user approved this file name (does NOT overwrite A0's XML).
GRAVCOMP_SCENE_NAME = "scene_act_E194_rubberHull_PRG_gravcomp"
GRAVCOMP_SIDECAR_FILE = f"{GRAVCOMP_SCENE_NAME}.xml"

# --- gains ------------------------------------------------------------------
KP_BASELINE = 500.0   # A0/G1 translational gain (resolved from the PRG override)
KP_HARD = 2500.0      # G2/G3 translational gain (user-approved 5x)
ROT_GAIN = 50.0       # all four arms -- MUST NOT change (see C3)

# --- A0 baseline source manifests (PRG side, already landed, read-only) -----
# object_key -> (source_exp_id, cem_full_manifest.tsv)
SOURCE_MANIFEST = {
    "box004": ("E172", REPO / "workspace/core4d/results/E172/s6_downstream/manifests/cem_full_manifest.tsv"),
    "box024": ("E173", REPO / "workspace/core4d/results/E173/s6_downstream/manifests/cem_full_manifest.tsv"),
}

# --- the 15-case set (identical to E191/E192/E193; v1/v2 retarget mixed) -----
CASES = {
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
N_CASES = sum(len(v) for v in CASES.values())  # 15


def all_case_ids() -> list[str]:
    return [cid for key in ("box024", "box004") for cid in CASES[key]]


def object_key_of(case_id: str) -> str:
    for key, ids in CASES.items():
        if case_id in ids:
            return key
    raise KeyError(f"unknown case_id: {case_id}")


# --- arms -------------------------------------------------------------------
# needs_gravcomp_sidecar -> uses scene_name=GRAVCOMP_SCENE_NAME (else base PRG)
# kp                      -> translational gain the config MUST resolve to
# A0 is documented for parity but is never re-run by E194 (reuse=True).
ARMS = {
    "A0": {"gravcomp": False, "kp": KP_BASELINE, "reuse": True},
    "G1": {"gravcomp": True,  "kp": KP_BASELINE, "reuse": False},
    "G2": {"gravcomp": False, "kp": KP_HARD,     "reuse": False},
    "G3": {"gravcomp": True,  "kp": KP_HARD,     "reuse": False},
}
RUN_ARMS = ["G1", "G2", "G3"]  # 3 arms x 15 cases = 45 Full CEM rows


def arm_extra_overrides(arm: str) -> str:
    """Space-separated run_mjwp CLI tokens that turn the base PRG override into `arm`."""
    spec = ARMS[arm]
    tokens: list[str] = []
    if spec["gravcomp"]:
        tokens.append(f"scene_name={GRAVCOMP_SCENE_NAME}")
    if spec["kp"] != KP_BASELINE:
        tokens.append(f"init_pos_actuator_gain={int(spec['kp'])}")
    return " ".join(tokens)


# --- CEM budget -------------------------------------------------------------
FULL_SAMPLES, FULL_OPT_STEPS = 1024, 32
CANARY_SAMPLES, CANARY_OPT_STEPS = 64, 4
CEM_SEED = 0

# Full CEM runs only on local GPUs 4-7 (user directive, 2026-08-08). Rows are
# round-robin assigned over this pool by ordinal for balance (45 rows -> ~11-12
# per GPU). Canary keeps its source assigned_gpu (it runs on 0/4/6).
FULL_GPU_POOL = ["4", "5", "6", "7"]

# canary: 3 arms x 2 cases = 6 rows. box024 026_p1 (worst) + 027_p2 (best) +
# box004 082_p1, each under all three arms (the plan's canary set).
CANARY_CASES = ["box024_20231011_026_p1", "box024_20231011_027_p2", "box004_20231003_2_082_p1"]


# --- generic IO helpers (self-contained; mirrors e189/e190_common) ----------
def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def repo_path(value: str | Path) -> Path:
    path = Path(str(value))
    # Do NOT .resolve(): workspace/core4d/results is a symlink to another mount,
    # and resolving it escapes REPO (breaking rel()). Lexical REPO/<rel> traverses
    # the symlink fine for all filesystem ops (open/stat/sha256/mkdir).
    return path if path.is_absolute() else (REPO / path)


def rel(value: str | Path) -> str:
    path = repo_path(value)
    try:
        return path.relative_to(REPO).as_posix()
    except ValueError:
        return Path(str(value)).as_posix()


def require_file(value: str | Path, label: str) -> Path:
    path = repo_path(value)
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"missing {label}: {value}")
    return path


def sha256(value: str | Path) -> str:
    path = require_file(value, "sha256 input")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_tsv(value: str | Path) -> list[dict[str, str]]:
    with repo_path(value).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_with_fields(value: str | Path) -> tuple[list[dict[str, str]], list[str]]:
    with repo_path(value).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
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


def write_tsv(value: str | Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path = repo_path(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, delimiter="\t", lineterminator="\n", extrasaction="ignore"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: serial(row.get(field, "")) for field in fields})


def write_json(value: str | Path, payload: Any) -> None:
    path = repo_path(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def boolish(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def load_source_rows() -> dict[str, dict[str, str]]:
    """case_id -> A0 baseline manifest row, from E172/E173 cem_full_manifest.tsv.

    This is the single source of truth for each case's PRG override, base scene
    sidecar, trajectory, contact_mask, task and assigned_gpu -- never
    reconstructed from a path template (retarget variant is v1/v2 mixed).
    """
    out: dict[str, dict[str, str]] = {}
    for object_key, (exp_id, manifest) in SOURCE_MANIFEST.items():
        rows = {r["case_id"]: r for r in read_tsv(require_file(manifest, f"{exp_id} full manifest"))}
        for case_id in CASES[object_key]:
            if case_id not in rows:
                raise SystemExit(f"{case_id} absent from {exp_id} full manifest: {manifest}")
            row = dict(rows[case_id])
            row["_source_exp"] = exp_id
            out[case_id] = row
    if len(out) != N_CASES:
        raise SystemExit(f"expected {N_CASES} source rows, got {len(out)}")
    return out
