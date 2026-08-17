#!/usr/bin/env python3
"""Shared contract + helpers for E200 (E199 translation augmentation -> two more arms).

E200 reuses the *already-built* E199 full-scale translation augmentation
(249 feasible trans0/1/2 tasks across the 87 box s6-full-CEM cases) and re-runs
formal full CEM under two additional downstream arms, WITHOUT re-running any
upstream retarget or rebuilding any trajectory:

  * arm "prg_g1a2" : PRG scene + object gravcomp=1 (G1) + hand-gate (A2).
                     scene = scene_act_E199_rubberHull_PRG_gravcomp.xml
                     (single-variable gravcomp diff of the E199 PRG sidecar);
                     override = the E199 PRG override + CLI scene_name + A2 gate.
                     orig baseline (eval) = E198 G1A2 arm (all 87 box cases).
  * arm "noprg"    : E167A base arm (rubber_hull hand, NO 16 lower-body/object
                     pairs, NO leg gate). scene = scene_act_E199_rubberHull.xml
                     (the rubber_hull intermediate E199 already persisted);
                     override = the E199 augmented base task yaml (chains E167A)
                     + CLI scene_name + explicit leg-penalty/leg-gate = off.
                     orig baseline (eval) = E190 38-case noPRG (only-read).

The augmentation lives entirely in the reference trajectory (arm-independent),
so E200 is a pure downstream re-run: swap scene_act/override, re-run CEM.

Single source of truth:
  * E199 aug artifacts / IO helpers / rubber-hull scene names <- e199_common.
  * A2 hand-gate pack (A2_GATE, a2_overrides)                 <- e198_common.
  * The gravcomp single-variable assertion is ported verbatim from
    E198 build_g1a2_manifest.assert_gravcomp_diff (kept local so this module
    has no import-time dependency on that script's CLI module).
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

# --- reuse E199 + E198 contracts as single sources of truth ------------------
_E199_DIR = Path(__file__).resolve().parent.parent / "E199"
_E198_DIR = Path(__file__).resolve().parent.parent / "E198"
for _p in (str(_E199_DIR), str(_E198_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import e199_common as E199  # noqa: E402
import e198_common as E198  # noqa: E402

# re-export the E199 IO helpers verbatim (JuiceFS-EIO-safe write_tsv included)
REPO = E199.REPO
repo_path = E199.repo_path
rel = E199.rel
sha256 = E199.sha256
read_tsv = E199.read_tsv
read_with_fields = E199.read_with_fields
write_tsv = E199.write_tsv
write_json = E199.write_json
now = E199.now
serial = E199.serial
safe_id = E199.safe_id
truth = E199.truth
TASK_ROOT = E199.TASK_ROOT

# --- E200 roots --------------------------------------------------------------
SCRIPT_DIR = REPO / "workspace/core4d/scripts/experiments/E200"
RESULTS = REPO / "workspace/core4d/results/E200"
MANIFEST_DIR = RESULTS / "s6_downstream/manifests"
SNAPSHOT_DIR = RESULTS / "scene_snapshot"

# --- source: the E199 full-scale feasible aug set (249 rows, 87 box cases) ---
E199_FULLSCALE_MANIFEST = E199.FULLSCALE_MANIFEST

# --- arm contract ------------------------------------------------------------
ARMS = ("noprg", "prg_g1a2")
ARM_TAG = {"noprg": "noPRG", "prg_g1a2": "PRG_G1A2"}
ARM_LABEL = {"noprg": "noPRG (E167A)", "prg_g1a2": "PRG+G1+A2"}

# scene basenames that E199 already persisted in every aug task dir
NOPRG_SCENE = E199.RUBBER_INTERMEDIATE          # scene_act_E199_rubberHull
PRG_SCENE = E199.SCENE_NAME                      # scene_act_E199_rubberHull_PRG
GRAVCOMP_SCENE = f"{PRG_SCENE}_gravcomp"         # scene_act_E199_rubberHull_PRG_gravcomp

# A2 hand-gate pack (single source = E198/E192)
A2_GATE = E198.A2_GATE
a2_overrides = E198.a2_overrides

# frozen CEM budget (identical to E199/E198 full)
CEM_SEED = E199.CEM_SEED
CEM_FULL_SAMPLES = E199.CEM_FULL_SAMPLES
CEM_FULL_OPT_STEPS = E199.CEM_FULL_OPT_STEPS

# priority tiers: all rows are translation variants -> single tier P1.
TIER_RANK = {"P1": 1}

FIELDS = [
    "ordinal", "tier", "experiment", "arm", "object_key", "case_id",
    "aug_variant", "aug_translation", "aug_rotation_rad",
    "base_target_task", "target_task", "target_scene",
    "trajectory", "trajectory_sha256", "contact_mask", "contact_mask_sha256",
    "override_id", "override_path", "override_sha256",
    "base_scene_act", "base_scene_sha256", "scene_act", "scene_name", "effective_scene_sha256",
    "extra_overrides", "cem_samples", "cem_opt_steps", "cem_seed",
    "variant", "result_npz", "outdir_npz", "config_act", "video", "log",
    "gpu_id", "status", "failure_mode", "execution_mode", "updated_at",
]


def manifest_path(arm: str) -> Path:
    return MANIFEST_DIR / f"e200_{arm}_priority_manifest.tsv"


def scene_registry_path(arm: str) -> Path:
    return MANIFEST_DIR / f"e200_{arm}_scene_registry.tsv"


# --- source rows: read the E199 full-scale feasible aug set ------------------
def load_aug_rows() -> list[dict[str, str]]:
    """Every feasible E199 full-scale trans variant (arm-independent inputs).

    Returns the E199 rows verbatim; E200 only reuses trajectory / contact_mask /
    scenes / override_id from them and re-targets the downstream arm.
    """
    rows = read_tsv(E199_FULLSCALE_MANIFEST)
    rows.sort(key=lambda r: (r["object_key"], r["case_id"], r["aug_variant"]))
    return rows


# --- gravcomp single-variable sidecar (G1) -----------------------------------
def _signature(element: ET.Element) -> Any:
    return (element.tag, tuple(sorted(element.attrib.items())), (element.text or "").strip(),
            tuple(_signature(child) for child in element))


def assert_gravcomp_diff(base: Path, sidecar: Path) -> None:
    """Fail unless sidecar == base with only object body gravcomp 0/absent -> 1.

    Ported verbatim from E198 build_g1a2_manifest.assert_gravcomp_diff.
    """
    base = repo_path(base)
    sidecar = repo_path(sidecar)
    base_root = ET.parse(base).getroot()
    objs = [b for b in base_root.iter("body") if b.get("name") == "object"]
    if len(objs) != 1 or objs[0].get("gravcomp") not in (None, "0", "0.0"):
        raise ValueError(f"unexpected base object gravcomp in {base}")
    expected = ET.parse(base).getroot()
    next(b for b in expected.iter("body") if b.get("name") == "object").set("gravcomp", "1")
    side_root = ET.parse(sidecar).getroot()
    if _signature(side_root) != _signature(expected):
        raise AssertionError(f"G1 sidecar is not a single-variable gravcomp diff: {sidecar}")


def build_gravcomp_sidecar(prg_scene: str | Path, *, overwrite: bool = False) -> Path:
    """Write scene_act_E199_rubberHull_PRG_gravcomp.xml next to the PRG scene.

    Single-variable diff: the `object` body gravcomp 0/absent -> 1, nothing else.
    Idempotent: an existing sidecar is re-verified (not rewritten) unless overwrite.
    """
    prg = repo_path(prg_scene)
    if not prg.is_file():
        raise FileNotFoundError(prg)
    out = prg.with_name(f"{GRAVCOMP_SCENE}.xml")
    if out.is_file() and not overwrite:
        assert_gravcomp_diff(prg, out)
        return out
    tree = ET.parse(prg)
    root = tree.getroot()
    objs = [b for b in root.iter("body") if b.get("name") == "object"]
    if len(objs) != 1:
        raise ValueError(f"expected exactly one object body in {prg}, found {len(objs)}")
    if objs[0].get("gravcomp") not in (None, "0", "0.0"):
        raise ValueError(f"object already has gravcomp in {prg}: {objs[0].get('gravcomp')}")
    objs[0].set("gravcomp", "1")
    ET.indent(tree, space="  ")
    tree.write(out, encoding="utf-8", xml_declaration=True)
    assert_gravcomp_diff(prg, out)
    return out


# --- per-arm override wiring --------------------------------------------------
def base_override_id(target_task: str) -> str:
    """The E199 augmented base task yaml id (chains the E167A base reward)."""
    return f"core4d_{target_task}"


def arm_scene(arm: str, aug_task_dir: Path) -> Path:
    """The scene_act xml this arm loads (already on disk from E199)."""
    if arm == "noprg":
        return aug_task_dir / f"{NOPRG_SCENE}.xml"
    if arm == "prg_g1a2":
        return aug_task_dir / f"{GRAVCOMP_SCENE}.xml"
    raise ValueError(f"unknown arm {arm}")


def arm_scene_name(arm: str) -> str:
    return {"noprg": NOPRG_SCENE, "prg_g1a2": GRAVCOMP_SCENE}[arm]


def arm_override_id(arm: str, aug_row: dict[str, str]) -> str:
    """noprg uses the E167A augmented base yaml; prg_g1a2 reuses the E199 PRG override."""
    if arm == "noprg":
        return base_override_id(aug_row["target_task"])
    if arm == "prg_g1a2":
        return aug_row["override_id"]
    raise ValueError(f"unknown arm {arm}")


def arm_extra_overrides(arm: str) -> str:
    """CLI overrides that turn the shared override_id into this arm.

    noprg    : force the rubber_hull (no-PRG) scene. leg_object_penalty_scale /
               cem_leg_gate_enabled stay at their SPIDER defaults (0.0 / False) --
               the E167A base yaml never declares them, so a bare CLI override is
               rejected by Hydra ("not in struct"); the noPRG-negative state is
               instead VERIFIED from config_act.yaml post-run (same as E190).
    prg_g1a2 : force the gravcomp (G1) scene + the A2 hand-gate pack.
    """
    if arm == "noprg":
        return f"scene_name={NOPRG_SCENE}"
    if arm == "prg_g1a2":
        return f"scene_name={GRAVCOMP_SCENE} {a2_overrides()}"
    raise ValueError(f"unknown arm {arm}")
