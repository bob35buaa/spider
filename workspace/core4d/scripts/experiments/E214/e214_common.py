#!/usr/bin/env python3
"""E214 shared contract: four leave-one-out ablations of the core method.

E214 verifies the two algorithmic contributions of the final line (Multi-Scale
Contact Reward + Part-wise Penetration Constraints) by removing one sub-component
at a time and re-running CEM on the paper case set (50 cases = paper 52 minus the
two E167A box021 overrides; box023 uses the full-stack E173 baseline instead of
the noPRG E179/E190 version, so all 50 baselines are homogeneous full-stack).

Mechanism (no new Hydra override files): ``examples/run_mjwp.py`` supports
``load_config_path=<config_act.yaml>`` -- it loads a case's fully-resolved
baseline config and applies CLI overrides on top.  So each ablation =
"load baseline config_act.yaml + flip the ablation keys + redirect output_dir to
the E214 tree".  Nothing under any baseline experiment is modified; only the
ablation's own config_act.yaml + npz are written, into results/E214/.

Baseline "full" column numbers are NOT recomputed here (see the report script):
the 43 non-box023 cases reuse the paper cache; box023 uses its E173 rollouts.

The per-case baseline config_act.yaml is resolved exactly like
report/0908/code/gen_paper_results.py (GROUPS + rl-export tsv -> cem_result_npz),
except box023 points at E173's user-approved export.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[5]
SCRIPT_DIR = Path(__file__).resolve().parent

# Light IO helpers only (e199_common imports fast; e208_common hangs on import).
sys.path.insert(0, str(REPO / "workspace/core4d/scripts/experiments/E199"))
import e199_common as E199  # noqa: E402

now = E199.now
rel = E199.rel
repo_path = E199.repo_path
sha256 = E199.sha256
read_tsv = E199.read_tsv
read_with_fields = E199.read_with_fields
write_tsv = E199.write_tsv
safe_id = E199.safe_id
SPIDER_PYTHON_BIN = E199.SPIDER_PYTHON_BIN

# --------------------------------------------------------------------------
# Identity
# --------------------------------------------------------------------------
EXP_ID = "E214"
RUN_ID = "R300"
PLAN_SLOT = 245
LOG_SLOT = 303

# GPU pool defaults (local 8-GPU run; not sharded).
GPU_DEFAULT = "0,1,2,3,4,5,6,7"
PER_GPU_MEM_MIB = 5000
PER_TASK_TIMEOUT_MIN = 180  # same bound as E208/E213

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
CASE_FILE = REPO / "tmp/E214_case_id.txt"
PAPER_CASE_FILE = REPO / "tmp/paper_case_id.txt"
# The two E167A box021 cases dropped from the paper set (no full-stack baseline).
DROP_CASES = {"box021_20231018_029_p2", "box021_20231011_035_p1"}

RESULTS = REPO / "workspace/core4d/results" / EXP_ID
CEM_DIR = RESULTS / "cem"
MANIFEST_DIR = RESULTS / "manifests"
EVAL_DIR = RESULTS / "eval"
REPORT_DIR = RESULTS / "reports"
SNAPSHOT_DIR = RESULTS / "scene_snapshot"
PREFLIGHT_DIR = RESULTS / "preflight"
LOCKS_DIR = RESULTS / ".locks"
CEM_LOG_DIR = REPO / "logs/E214/cem"

MANIFEST = MANIFEST_DIR / "e214_ablation_cem.tsv"

MISSING_MOUNT = "/mnt/a0ccc676-9496-49f8-a861-f8a1797dec52/spider_workdirs"

# Baseline config_act.yaml stores absolute paths from the machine it ran on
# (e.g. /home/dataset-assist-0/...) that do NOT exist here.  These input path
# fields must be re-resolved to local files and overridden on the CLI so
# load_config_path reproduces the exact baseline scene/data/mask locally.
INPUT_PATH_FIELDS = ("model_path", "data_path", "contact_hdmi_mask_path")

# Hydra strict-override bookkeeping: keys present in default.yaml take `key=`;
# keys absent (but still valid Config fields) need `+key=`.
DEFAULT_YAML = REPO / "examples/config/default.yaml"


def _default_yaml_keys() -> set[str]:
    try:
        data = yaml.safe_load(DEFAULT_YAML.read_text(encoding="utf-8")) or {}
    except OSError:
        return set()
    return {k for k in data if k != "defaults"}


DEFAULT_KEYS = _default_yaml_keys()


def hydra_token(key: str, val: Any) -> str:
    """Format one run_mjwp.py CLI override, adding '+' for non-default keys."""
    if isinstance(val, bool):
        sval = str(val).lower()
    else:
        sval = str(val)
    prefix = "" if key in DEFAULT_KEYS else "+"
    return f"{prefix}{key}={sval}"

# rl-export tsv column names (shared across every source experiment).
RL_TRAJECTORY = "trajectory"
RL_SCENE_ACT = "scene_act"
RL_CONTACT_MASK = "contact_mask"
RL_CEM_NPZ = "cem_result_npz"

# --------------------------------------------------------------------------
# Ablation definitions: name -> {label, desc, toggles}
# toggles are CLI overrides applied on top of the loaded baseline config.
# Booleans are lowercased for Hydra dotlist parsing.
# --------------------------------------------------------------------------
ABLATIONS: dict[str, dict[str, Any]] = {
    "A1_contactHDMI_only": {
        "label": "仅 contact_hdmi (去 surface_band 细项)",
        "toggles": {
            "surface_band_rew_scale": 0.0,
            "surface_band_penalty_scale": 0.0,
        },
    },
    "A2_surfaceBand_only": {
        "label": "仅 surface_band (去 contact_hdmi 粗项 + palm-normal)",
        "toggles": {
            "contact_hdmi_gain": 0.0,
            "contact_hdmi_ori_weight": 0.0,
        },
    },
    "A3_softPenalty_only": {
        "label": "仅软惩罚 (去全部硬候选剔除门)",
        "toggles": {
            "cem_safety_gate_enabled": False,
            "cem_hand_gate_enabled": False,
            "cem_leg_gate_enabled": False,
            "cem_posture_gate_enabled": False,
            "cem_peak_margin_enabled": False,
        },
    },
    "A4_hardGate_only": {
        "label": "仅硬候选门 (去全部软惩罚 + E167A z)",
        "toggles": {
            "robot_object_penalty_scale": 0.0,
            "leg_object_penalty_scale": 0.0,
            "hand_floor_penalty_scale": 0.0,
            "e167_body_z_enabled": False,
            "e167_ground_z_enabled": False,
        },
    },
}
ABLATION_ORDER = list(ABLATIONS.keys())

# Keys that legitimately differ between baseline and ablation config_act.yaml
# (run-plumbing, not the ablation itself).  The single-variable audit asserts the
# diff is a subset of {these} + the ablation's own toggled keys + derived keys
# (e.g. *_geom_ids that stay resolved even when a gate flag flips off).
PLUMBING_KEYS = {
    "load_config_path", "output_dir", "video_output_path", "save_video",
    "save_config", "seed", "num_samples", "max_num_iterations",
    "use_torch_compile", "video_camera", "viewer",
    # local-resolved baseline input paths (config stores another machine's paths):
    "model_path", "data_path", "contact_hdmi_mask_path",
}
DERIVED_KEY_SUFFIXES = ("_geom_ids", "_body_ids", "_ids")

# --------------------------------------------------------------------------
# Source registry (baseline config_act.yaml resolution).
# Mirrors gen_paper_results.GROUPS, EXCEPT box023 -> E173 full-stack export.
# suffix / arm_key are only used to disambiguate multi-row tables downstream;
# for E214 we only need the rl-export row's cem_result_npz.
# --------------------------------------------------------------------------
GROUPS: dict[str, dict[str, Any]] = {
    "box021": {"exp": "E170_PRG",
               "rl": "workspace/core4d/results/E170/s6_downstream/rl_export/paired_rl_export_input.tsv"},
    "box023": {"exp": "E173_PRG_fullstack",
               "rl": "workspace/core4d/results/E173/s6_downstream/rl_export/box023_user_approved/rl_export_input.tsv"},
    "box001": {"exp": "E198_G1A2",
               "rl": "workspace/core4d/results/E198/s6_downstream/rl_export/box001_user_approved/rl_export_input.tsv"},
    "box004": {"exp": "E198_G1A2",
               "rl": "workspace/core4d/results/E198/s6_downstream/rl_export/box004_user_approved/rl_export_input.tsv"},
    "box024": {"exp": "E198_G1A2",
               "rl": "workspace/core4d/results/E198/s6_downstream/rl_export/box024_user_approved/rl_export_input.tsv"},
    "bucket003": {"exp": "E178",
                  "rl": "workspace/core4d/results/E178/s6_downstream/rl_export/paired_rl_export_input.tsv"},
    "bucket007": {"exp": "E207",
                  "rl": "workspace/core4d/results/E207/s6_downstream/rl_export/paired_rl_export_input.tsv"},
}
for _obj in ("chair006", "desk007", "desk021", "desk023"):
    GROUPS[_obj] = {"exp": "E212",
                    "rl": "workspace/core4d/results/E212/s6_downstream/rl_export/paired_rl_export_input.tsv"}


def object_key_of(case_id: str) -> str:
    return case_id.split("_", 1)[0]


# --------------------------------------------------------------------------
# Path remap / resolve (verbatim policy from gen_paper_results.py)
# --------------------------------------------------------------------------
def remap(value: str) -> str:
    if value.startswith(MISSING_MOUNT):
        tail = value[len(MISSING_MOUNT):].lstrip("/")
        cand = REPO / "workspace" / tail
        if cand.exists() or tail.startswith("core4d/"):
            return str(cand)
        return str(REPO / tail)
    return value


def resolve(value: str | Path | None) -> Path | None:
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


# --------------------------------------------------------------------------
# Case model
# --------------------------------------------------------------------------
def load_cases() -> list[str]:
    cases = [ln.strip() for ln in CASE_FILE.read_text().splitlines() if ln.strip()]
    return cases


def _rl_row(rl_tsv: str, case_id: str) -> dict[str, str] | None:
    for row in read_tsv(REPO / rl_tsv):
        if row.get("case_id") == case_id:
            return row
    return None


def baseline_paths(case_id: str) -> dict[str, Any]:
    """Resolve a case's baseline (config_act.yaml, cem_npz, scene/traj/mask).

    config_act.yaml is the sibling of the selected cem_result_npz.  Returns a
    dict with resolved absolute paths (or None) plus the source exp id and any
    resolution note.  Never raises: the contract test aggregates failures.
    """
    obj = object_key_of(case_id)
    grp = GROUPS.get(obj)
    out: dict[str, Any] = {"case_id": case_id, "object_key": obj,
                           "exp": grp["exp"] if grp else "UNKNOWN", "note": ""}
    if grp is None:
        out["note"] = f"no GROUPS entry for object {obj!r}"
        return out
    row = _rl_row(grp["rl"], case_id)
    if row is None:
        out["note"] = f"case not in rl tsv {grp['rl']}"
        return out
    cem_npz = resolve(row.get(RL_CEM_NPZ, ""))
    out["cem_npz"] = cem_npz
    out["scene_act"] = resolve(row.get(RL_SCENE_ACT, ""))
    out["trajectory"] = resolve(row.get(RL_TRAJECTORY, ""))
    out["contact_mask"] = resolve(row.get(RL_CONTACT_MASK, ""))
    if cem_npz is None:
        out["note"] = "cem_result_npz unresolved"
        return out
    cfg = resolve_config_act(cem_npz)
    out["config_act"] = cfg
    if cfg is None:
        out["note"] = f"config_act.yaml not found for {rel(cem_npz)}"
        return out
    # Re-resolve the baseline config's own input paths to local files.
    run_inputs, missing = resolve_config_inputs(cfg)
    out["run_inputs"] = run_inputs
    if missing:
        out["note"] = "config input(s) unresolved: " + ",".join(missing)
    return out


def resolve_config_inputs(cfg_path: Path) -> tuple[dict[str, Path], list[str]]:
    """Resolve the baseline config's own model_path/data_path/contact_hdmi_mask_path
    to local files (they are stored as another machine's absolute paths).

    Returns (resolved{field->Path}, missing[field]).  A field that is empty in
    the config is skipped (not missing).  contact_hdmi_mask_path may be empty
    when the mask source is not core4d_3cm.
    """
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    resolved: dict[str, Path] = {}
    missing: list[str] = []
    for field in INPUT_PATH_FIELDS:
        raw = data.get(field) or ""
        if not raw:
            continue
        p = resolve(raw)
        if p is None:
            missing.append(field)
        else:
            resolved[field] = p
    return resolved, missing


def resolve_config_act(cem_npz: Path) -> Path | None:
    """Locate the config_act.yaml that produced this selected rollout.

    Two on-disk layouts across the source experiments:
      1. cem_npz IS an outdir's ``trajectory_mjwp_act.npz`` (E207/E212->E209):
         config_act.yaml sits in the same directory.
      2. cem_npz is a flattened ``<TAG>.npz`` copy in the stage dir (E170/E173/
         E178/E198): the run output lives in a sibling ``<TAG>_outdir*/`` dir
         (suffix varies: _outdir / _outdir_full / ...).  The flattened npz is
         byte-identical to that dir's trajectory_mjwp_act.npz.
    """
    if cem_npz.name == "trajectory_mjwp_act.npz":
        cfg = cem_npz.parent / "config_act.yaml"
        return cfg.resolve() if cfg.is_file() else None
    stem = cem_npz.stem
    cands = sorted(cem_npz.parent.glob(f"{stem}_outdir*/config_act.yaml"))
    if cands:
        return cands[0].resolve()
    # last resort: same-dir config_act.yaml
    cfg = cem_npz.parent / "config_act.yaml"
    return cfg.resolve() if cfg.is_file() else None


# --------------------------------------------------------------------------
# CEM output paths (E214 tree)
# --------------------------------------------------------------------------
def run_tag(case_id: str, ablation: str) -> str:
    return f"{EXP_ID}_{case_id}__{ablation}"


def cem_out_dir(case_id: str, ablation: str) -> Path:
    return CEM_DIR / run_tag(case_id, ablation)


def result_npz(case_id: str, ablation: str) -> Path:
    return cem_out_dir(case_id, ablation) / "trajectory_mjwp_act.npz"


def config_act_out(case_id: str, ablation: str) -> Path:
    return cem_out_dir(case_id, ablation) / "config_act.yaml"


def cem_log_path(case_id: str, ablation: str) -> Path:
    return CEM_LOG_DIR / f"{run_tag(case_id, ablation)}.log"


def video_out(case_id: str, ablation: str) -> Path:
    return cem_out_dir(case_id, ablation) / "visualization_mjwp_act.mp4"


# --------------------------------------------------------------------------
# CLI toggle formatting for run_mjwp.py
# --------------------------------------------------------------------------
def toggle_tokens(ablation: str) -> list[str]:
    """Ablation config overrides as run_mjwp.py CLI dotlist tokens."""
    toks: list[str] = []
    for key, val in ABLATIONS[ablation]["toggles"].items():
        if isinstance(val, bool):
            toks.append(f"{key}={str(val).lower()}")
        else:
            toks.append(f"{key}={val}")
    return toks


# --------------------------------------------------------------------------
# Manifest schema
# --------------------------------------------------------------------------
MANIFEST_FIELDS = [
    "ordinal", "object_key", "case_id", "ablation", "ablation_label",
    "source_exp", "baseline_config_act", "baseline_config_act_sha256",
    "baseline_cem_npz",
    # local-resolved baseline inputs (overridden on the CLI, verbatim reuse):
    "run_model_path", "run_data_path", "run_contact_mask_path",
    "toggles",
    "outdir_npz", "config_act", "log",
    "gpu_id", "host", "status", "failure_mode", "wall_min", "updated_at",
]


class SingleInstance:
    """flock-based single-instance guard (copied from e208_common)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle = None

    def __enter__(self) -> "SingleInstance":
        import fcntl
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a+")
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._handle.seek(0)
            holder = self._handle.read().strip() or "<unknown>"
            self._handle.close()
            raise SystemExit(
                f"another {Path(sys.argv[0]).name} is already running ({holder}).\n"
                f"lock: {self.path}\nWait for it or stop it first."
            ) from None
        self._handle.seek(0)
        self._handle.truncate()
        self._handle.write(f"pid={os.getpid()} started={now()} argv={' '.join(sys.argv[1:])}\n")
        self._handle.flush()
        return self

    def __exit__(self, *exc: object) -> None:
        if self._handle is not None:
            self._handle.close()
