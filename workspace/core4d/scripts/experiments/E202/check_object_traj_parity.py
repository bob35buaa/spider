#!/usr/bin/env python3
"""E202-export C2: verify the perturbed object trajectory is identical for the
source person and the partner person under the same trans_k.

Object augmentation must perturb the SHARED object identically for both CORE4D
persons, else the two robots disagree about where the object is during approach.
The trimmed retarget qpos carries the object freejoint at [36:43] (robot base7 +
29 joints = 36, then object pos3+quat4 = 7). We crop both persons' trimmed object
channels to their common raw window and compare frame-for-frame.

Also reports the approach-segment offset + endpoint anchoring per side (reuses the
E202 pose_diff), confirming each retarget actually applied the ~0.2 m perturbation.

Pass criteria (per case x trans):
  * common_raw_frames >= 2,
  * object position max-abs diff over the common window <= POS_TOL_M,
  * object quaternion geodesic max diff <= QUAT_TOL_DEG,
  * source & partner approach offset within [APPROACH_MIN_M, APPROACH_MAX_M],
  * endpoint residual fraction <= ENDPOINT_FRAC_MAX.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e202_common as C  # noqa: E402
import e202_export_common as X  # noqa: E402


def _load_e202_module(name: str):
    """Load E202's build_augmented_tasks by path (e202_common shadows E199's)."""
    import importlib.util
    path = Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"e202_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BAT = _load_e202_module("build_augmented_tasks")  # E202's, not E199's

POS_TOL_M = 0.01          # 1 cm; deterministic perturbation -> expect ~0
QUAT_TOL_DEG = 2.0
APPROACH_MIN_M = 0.15
APPROACH_MAX_M = 0.25
ENDPOINT_FRAC_MAX = 0.25

PARITY_FIELDS = [
    "case_id", "partner_case_id", "aug_variant", "holo_name",
    "common_raw_start", "common_raw_end", "common_raw_frames",
    "obj_pos_maxdiff_m", "obj_quat_maxdiff_deg",
    "src_approach_offset_m", "partner_approach_offset_m",
    "src_endpoint_frac", "partner_endpoint_frac",
    "status", "failure_mode",
]


def _load_trim(base_task: str) -> tuple[int, int]:
    p = C.REPO / C.DATA_PREPROCESS_REL / f"holosoma_{base_task}" / "trim_window.json"
    j = json.loads(p.read_text())
    return int(j["trim_start"]), int(j["trim_end"])


def _object_channel(base_task: str, meta: dict[str, str], holo_name: str) -> np.ndarray:
    root = BAT.case_root(base_task)
    holo = meta["holosoma_task"]
    with np.load(root / "trimmed" / f"{holo}_{holo_name}.npz", allow_pickle=True) as d:
        q = np.asarray(d["qpos"], dtype=np.float64)
    return q[:, 36:43]  # object freejoint: pos3 + quat(wxyz)4


def _quat_maxdiff_deg(a: np.ndarray, b: np.ndarray) -> float:
    # a,b: (T,4) wxyz. geodesic angle = 2*acos(|<a,b>|)
    dot = np.abs(np.sum(a * b, axis=1)) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
    dot = np.clip(dot, -1.0, 1.0)
    return float(np.degrees(2.0 * np.arccos(dot)).max())


def check_case(source: dict[str, Any]) -> list[dict[str, Any]]:
    s_base = source["base_target_task"]
    s_meta = {"holosoma_task": source["holosoma_task"]}
    p_base = X.partner_base_task(s_base)
    p_case = X.partner_case_id(source["case_id"])
    p_meta = {"holosoma_task": C.load_case_meta(p_base)["holosoma_task"]}

    ss, se = _load_trim(s_base)
    ps, pe = _load_trim(p_base)
    cs, ce = max(ss, ps), min(se, pe)
    common = ce - cs

    rows: list[dict[str, Any]] = []
    for e202_name in sorted(source["source_variants"]):           # trans0/1/2
        holo_name = f"trans_{e202_name[-1]}"
        failures: list[str] = []
        obj_pos_diff = obj_quat_diff = float("nan")
        s_app = p_app = s_end = p_end = float("nan")
        try:
            s_obj = _object_channel(s_base, s_meta, holo_name)
            p_obj = _object_channel(p_base, p_meta, holo_name)
            if common < 2:
                failures.append("no_usable_common_raw_window")
            else:
                s_slice = s_obj[cs - ss: ce - ss]
                p_slice = p_obj[cs - ps: ce - ps]
                n = min(len(s_slice), len(p_slice))
                s_slice, p_slice = s_slice[:n], p_slice[:n]
                obj_pos_diff = float(np.linalg.norm(s_slice[:, :3] - p_slice[:, :3], axis=1).max())
                obj_quat_diff = _quat_maxdiff_deg(s_slice[:, 3:7], p_slice[:, 3:7])
                if obj_pos_diff > POS_TOL_M:
                    failures.append(f"object_pos_parity>{POS_TOL_M}m")
                if obj_quat_diff > QUAT_TOL_DEG:
                    failures.append(f"object_ori_parity>{QUAT_TOL_DEG}deg")
            sd = BAT.pose_diff(s_base, s_meta, holo_name)
            pd = BAT.pose_diff(p_base, p_meta, holo_name)
            s_app, s_end = sd["approach_trans_offset_m_max"], sd["endpoint_frac_of_approach_trans"]
            p_app, p_end = pd["approach_trans_offset_m_max"], pd["endpoint_frac_of_approach_trans"]
            for lbl, app in (("source", s_app), ("partner", p_app)):
                if not (APPROACH_MIN_M <= app <= APPROACH_MAX_M):
                    failures.append(f"{lbl}_approach_offset_out_of_range")
            for lbl, end in (("source", s_end), ("partner", p_end)):
                if end > ENDPOINT_FRAC_MAX:
                    failures.append(f"{lbl}_endpoint_not_anchored")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"error:{type(exc).__name__}:{exc}")
        rows.append({
            "case_id": source["case_id"], "partner_case_id": p_case,
            "aug_variant": e202_name, "holo_name": holo_name,
            "common_raw_start": cs, "common_raw_end": ce, "common_raw_frames": common,
            "obj_pos_maxdiff_m": round(obj_pos_diff, 6),
            "obj_quat_maxdiff_deg": round(obj_quat_diff, 4),
            "src_approach_offset_m": round(s_app, 4), "partner_approach_offset_m": round(p_app, 4),
            "src_endpoint_frac": round(s_end, 4), "partner_endpoint_frac": round(p_end, 4),
            "status": "PASS" if not failures else "FAIL",
            "failure_mode": ",".join(failures),
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    all_rows: list[dict[str, Any]] = []
    for source in X.source_use_cases():
        if only and source["case_id"] not in only:
            continue
        all_rows.extend(check_case(source))
    C.write_tsv(X.PARITY_REPORT, all_rows, PARITY_FIELDS)
    n_pass = sum(r["status"] == "PASS" for r in all_rows)
    n_fail = len(all_rows) - n_pass
    print(f"[C2 parity] {n_pass}/{len(all_rows)} PASS, {n_fail} FAIL -> {C.rel(X.PARITY_REPORT)}")
    for r in all_rows:
        if r["status"] != "PASS":
            print(f"  FAIL {r['case_id']} {r['aug_variant']}: {r['failure_mode']} "
                  f"(objΔ={r['obj_pos_maxdiff_m']}m)")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
