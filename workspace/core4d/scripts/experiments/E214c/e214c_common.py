#!/usr/bin/env python3
"""E214c shared contract: two part-wise-threshold ablations of Contribution 2.

E214 (R300) leave-one-out ablated the penetration constraints *wholesale*
(A3 = remove ALL hard gates, A4 = remove ALL soft penalties).  E214c instead
ablates the **part-wise threshold** itself -- the actual "part-wise" essence of
Contribution 2 (technical_summary §2.2: "阈值本身编码部位差异：leg 严禁碰、hand 允许贴").

Baseline survey (50 cases, all 11 object groups) established:
  * SOFT penalties are already part-uniform (robot/leg/hand-floor all
    scale=2.0, margin=2cm; hand_object_deep OFF) -> nothing part-wise to unify.
    Per user decision the soft layer is left UNCHANGED.
  * HARD gates carry the part-wise thresholds (asymmetric):
        body  (cem_safety_gate): min_sdf -5mm, hard_floor NaN,  max_vio 0%
        hand  (cem_hand_gate)  : min_sdf -10mm, hard_floor -15mm(E198)/-20mm(rest), max_vio 5%/10%
        leg   (cem_leg_gate)   : min_sdf +5mm, hard_floor -5mm, max_vio 2%
    hand floor/vio vary by source experiment -> toggles are computed PER CASE.

Two ablations, unifying all three threshold params (min_sdf + hard_floor + max_vio):
  * A5_uniform_hand : body & leg gates <- that case's OWN hand gate values (most permissive).
  * A6_uniform_leg  : body & hand gates <- leg gate values (most strict; leg uniform across cases).
The source gate (hand for A5, leg for A6) is left untouched; all gates stay enabled;
soft penalties / posture gate / e167 z / peak_margin stay at baseline.

Mechanism is identical to E214 (run_mjwp.py ``load_config_path`` + CLI overrides on
top; ``_parse_override_tokens`` strips the leading '+', validates against Config, and
parses the float).  ZERO core code change -- only existing config keys are flipped.
Baseline resolution is imported verbatim from e214_common (same baselines as E214).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml

# --------------------------------------------------------------------------
# Reuse E214's baseline resolution + IO helpers verbatim (same baselines).
# --------------------------------------------------------------------------
_E214_DIR = Path(__file__).resolve().parents[1] / "E214"
sys.path.insert(0, str(_E214_DIR))
import e214_common as E214  # noqa: E402

# Light IO / path helpers (re-exported so the runner/audit/eval can use C.*).
now = E214.now
rel = E214.rel
repo_path = E214.repo_path
sha256 = E214.sha256
read_tsv = E214.read_tsv
read_with_fields = E214.read_with_fields
write_tsv = E214.write_tsv
safe_id = E214.safe_id
SPIDER_PYTHON_BIN = E214.SPIDER_PYTHON_BIN
hydra_token = E214.hydra_token
remap = E214.remap
resolve = E214.resolve
object_key_of = E214.object_key_of
baseline_paths = E214.baseline_paths
resolve_config_act = E214.resolve_config_act
resolve_config_inputs = E214.resolve_config_inputs
SingleInstance = E214.SingleInstance
GROUPS = E214.GROUPS
INPUT_PATH_FIELDS = E214.INPUT_PATH_FIELDS
PLUMBING_KEYS = E214.PLUMBING_KEYS
DERIVED_KEY_SUFFIXES = E214.DERIVED_KEY_SUFFIXES
PAPER_CASE_FILE = E214.PAPER_CASE_FILE
DROP_CASES = E214.DROP_CASES
GPU_DEFAULT = E214.GPU_DEFAULT
PER_GPU_MEM_MIB = E214.PER_GPU_MEM_MIB
PER_TASK_TIMEOUT_MIN = E214.PER_TASK_TIMEOUT_MIN

# --------------------------------------------------------------------------
# Identity
# --------------------------------------------------------------------------
EXP_ID = "E214c"
RUN_ID = "R303"
PLAN_SLOT = 248
LOG_SLOT = 307

# --------------------------------------------------------------------------
# Paths (E214c output tree; baselines still resolved from source experiments)
# --------------------------------------------------------------------------
REPO = E214.REPO
CASE_FILE = REPO / "tmp/E214c_case_id.txt"
# E214's case file -- the contract test asserts E214c's set is identical.
E214_CASE_FILE = E214.CASE_FILE

RESULTS = REPO / "workspace/core4d/results" / EXP_ID
CEM_DIR = RESULTS / "cem"
MANIFEST_DIR = RESULTS / "manifests"
EVAL_DIR = RESULTS / "eval"
REPORT_DIR = RESULTS / "reports"
SNAPSHOT_DIR = RESULTS / "scene_snapshot"
PREFLIGHT_DIR = RESULTS / "preflight"
LOCKS_DIR = RESULTS / ".locks"
CEM_LOG_DIR = REPO / "logs/E214c/cem"

MANIFEST = MANIFEST_DIR / "e214c_partwise_cem.tsv"


def load_cases() -> list[str]:
    return [ln.strip() for ln in CASE_FILE.read_text().splitlines() if ln.strip()]


# --------------------------------------------------------------------------
# Ablation definitions: part-wise threshold unification.
#   source_gate  : the gate whose thresholds are copied (left untouched itself)
#   target_gates : the two gates overwritten with the source's thresholds
# Toggle VALUES are computed per case (per_case_toggles) because the hand gate's
# hard_floor / max_vio differ between source experiments.
# --------------------------------------------------------------------------
GATE_PREFIX = {
    "safety": "cem_safety_gate",
    "hand": "cem_hand_gate",
    "leg": "cem_leg_gate",
}
# The three threshold params unified per gate (user decision: all three).
THRESHOLD_KEYS = ("min_sdf_m", "hard_floor_m", "max_violation_pct")

ABLATIONS: dict[str, dict[str, Any]] = {
    "A5_uniform_hand": {
        "label": "统一手阈值 (body/leg 门 ← hand, 最宽松)",
        "source_gate": "hand",
        "target_gates": ("safety", "leg"),
    },
    "A6_uniform_leg": {
        "label": "统一腿阈值 (body/hand 门 ← leg, 最严格)",
        "source_gate": "leg",
        "target_gates": ("safety", "hand"),
    },
}
ABLATION_ORDER = list(ABLATIONS.keys())

# The 9 distinct config keys any E214c ablation may flip (union over both arms).
ALL_TOGGLE_KEYS = frozenset(
    f"{GATE_PREFIX[g]}_{tk}"
    for spec in ABLATIONS.values()
    for g in spec["target_gates"]
    for tk in THRESHOLD_KEYS
)


def _fmt_num(v: Any) -> str:
    """Format a baseline float threshold for a Hydra dotlist token (round-trips)."""
    f = float(v)
    if f != f:  # NaN -- never expected for a source gate threshold
        raise ValueError("source gate threshold is NaN; cannot unify")
    return repr(f) if abs(f) >= 1e-4 else f"{f:.6f}".rstrip("0").rstrip(".")


def per_case_toggles(case_id: str, ablation: str, baseline_cfg: Path) -> dict[str, float]:
    """Read the case's baseline source-gate thresholds; map onto the target gates.

    Returns {config_key: float_value} for the 6 keys (2 target gates x 3 params).
    The source gate is NOT in the result (left at baseline).
    """
    data = yaml.safe_load(Path(baseline_cfg).read_text(encoding="utf-8")) or {}
    spec = ABLATIONS[ablation]
    src_prefix = GATE_PREFIX[spec["source_gate"]]
    src_vals: dict[str, float] = {}
    for tk in THRESHOLD_KEYS:
        key = f"{src_prefix}_{tk}"
        if key not in data or data[key] is None:
            raise ValueError(f"{case_id}: baseline missing source-gate key {key}")
        src_vals[tk] = float(data[key])
    toggles: dict[str, float] = {}
    for tgt in spec["target_gates"]:
        tp = GATE_PREFIX[tgt]
        for tk in THRESHOLD_KEYS:
            toggles[f"{tp}_{tk}"] = src_vals[tk]
    return toggles


def toggle_tokens(case_id: str, ablation: str, baseline_cfg: Path) -> list[str]:
    """Per-case CLI override tokens (key=value) for one ablation."""
    return [f"{k}={_fmt_num(v)}" for k, v in per_case_toggles(case_id, ablation, baseline_cfg).items()]


# --------------------------------------------------------------------------
# CEM output paths (E214c tree)
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
# Manifest schema (E214's + source_gate for audit transparency)
# --------------------------------------------------------------------------
MANIFEST_FIELDS = [
    "ordinal", "object_key", "case_id", "ablation", "ablation_label", "source_gate",
    "source_exp", "baseline_config_act", "baseline_config_act_sha256",
    "baseline_cem_npz",
    "run_model_path", "run_data_path", "run_contact_mask_path",
    "toggles",
    "outdir_npz", "config_act", "log",
    "gpu_id", "host", "status", "failure_mode", "wall_min", "updated_at",
]
