#!/usr/bin/env python3
"""E208 P9: build the viser review feed -- orig + 5 aug variants as a 6-arm sweep.

Registered as exp ``E208AUG`` in ``review_index.py``; launched with
    bash workspace/core4d/scripts/eval/wrappers/review_player.sh E208AUG

Shape follows E199's precedent rather than plan238's suggestion.  plan238 wanted
``case_id = {orig}__aug_{variant}``, which would give the player 105 unrelated
cases; making ``case_id`` the ORIG case and ``arm`` the variant instead gives 21
cases x 6 arms, so the reviewer sees orig and all five augmentations of the same
demonstration side by side.  That is the comparison the human is actually being
asked to make -- "is this aug as usable as its own orig" -- and it is how the
reviewer already reads E199/E200.

**No re-scoring.**  Metrics come from ``e208_aug_rollout.tsv``; only the
12-gate view is recomputed, and its thresholds are imported BY VALUE from E206's
own review builder so both review sessions are read on one scale.  Note those 12
gates are E206's release caliber, deliberately NOT the 6-gate augmentation
caliber C4 uses -- the player shows the reviewer what E206 showed them, while C4
compares orig and aug on the augmentation line's own standard.  Both numbers
appear in the feed (``numeric_release_pass`` = 12-gate, ``c4_all_gates_pass`` =
6-gate) so the difference is visible rather than surprising.

Every aug row is prefilled with its orig's E206 human verdict, so the reviewer
can see what the same demonstration was judged before augmenting.

Usage:
    .venv/bin/python .../E208/build_aug_review_tsv.py
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import e208_common as C  # noqa: E402


def _load_e206_review_builder():
    """E206's GATE_MAP/METRIC_COLUMNS/gate_pass, by value -- one shared scale."""
    path = C.EXPERIMENTS / "E206/build_arm_review_tsv.py"
    spec = importlib.util.spec_from_file_location("e208_e206_review_builder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ARM_LABEL = {
    "orig": "ORIG", "trans0": "TRANS0", "trans1": "TRANS1",
    "trans2": "TRANS2", "rot0": "ROT0", "rot1": "ROT1",
}

# Blank columns the reviewer fills, same names as E206's C6 sheet so the two
# filled TSVs can be concatenated without renaming anything.
REVIEW_BLANKS = (
    "user_manual_review_status", "manual_use_decision", "manual_quality_label",
    "manual_failure_taxonomy", "manual_review_note", "manual_reviewer",
    "manual_reviewed_at",
)

DELTA_ANCHORS = (
    "track_obj_pos_err_cm_mean",
    "hand_object_physics_contact_in_mask_frac",
    "leg_penetration_frac",
)


def finite(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def e206_manual() -> dict[str, dict[str, str]]:
    """E206's human verdicts for the PRG arm, keyed by plain case_id.

    E206's filled sheet has no ``arm`` column -- the arm is folded into the key
    as ``{case_id}#PRG`` / ``#noPRG``.  Joining on the raw key silently yields an
    all-blank prefill (no error, just an emptier review sheet), so the suffix is
    split off explicitly and non-PRG rows dropped.
    """
    path = C.MANUAL_REVIEW_TSV
    if not path.is_file():
        return {}
    got = C.sha256(path)
    if got != C.EXPECTED_MANUAL_REVIEW_SHA256:
        print(f"[warn] E206 manual review sha {got[:12]} != pinned "
              f"{C.EXPECTED_MANUAL_REVIEW_SHA256[:12]}; prefill may be stale",
              file=sys.stderr)
    out: dict[str, dict[str, str]] = {}
    for row in C.read_tsv(path):
        key = row.get("case_id", "")
        case_id, _, arm = key.partition("#")
        if arm and arm.upper() != "PRG":
            continue
        out[case_id] = row
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=C.EVAL_DIR / "e208_aug_case_metrics.tsv")
    ap.add_argument("--rollout-tsv", type=Path, default=None,
                    help="the evaluator's scored rows (e208_aug_rollout.tsv)")
    args = ap.parse_args()

    E206R = _load_e206_review_builder()
    eval_tsv = args.rollout_tsv or (C.EVAL_DIR / "e208_aug_rollout.tsv")
    if not C.repo_path(eval_tsv).is_file():
        raise SystemExit(f"missing {C.rel(eval_tsv)} -- run eval_E208_aug.py first")
    # Naming follows E206: the evaluator owns `*_rollout.tsv`, this builder owns
    # `*_case_metrics.tsv` (the review-player feed). Sharing one filename would
    # make whichever script ran last silently break the other one's consumer.
    scored = C.read_tsv(C.repo_path(eval_tsv))
    by_key = {(r["case_id"], r["aug_variant"]): r for r in scored}

    manifest = {(r["case_id"], r["aug_variant"]): r
                for r in C.read_tsv(C.PRIORITY_MANIFEST)}
    authority = {r["case_id"]: r for r in C.read_tsv(C.AUTHORITY_TSV)
                 if r["aug_variant"] == "orig"}
    manual = e206_manual()

    cols = (["exp_id", "arm", "case_id", "variant", "object_key", "retarget_variant_id",
             "numeric_release_pass", "numeric_failure_modes", "status",
             "outdir_npz", "scene_xml", "config_act", "trajectory", "video"]
            + list(E206R.GATE_MAP) + list(E206R.METRIC_COLUMNS)
            + ["aug_variant", "offset_band", "approach_trans_offset_m_max",
               "effective_retarget_variant", "rescue_state",
               "c4_all_gates_pass", "c4_failure_modes",
               "f15_divergent", "contact_mask_blind_frac"]
            + [f"delta_{m}" for m in DELTA_ANCHORS]
            + ["orig_manual_use_decision", "orig_manual_quality_label",
               "orig_manual_failure_taxonomy", "orig_numeric_release_pass"]
            + list(REVIEW_BLANKS) + ["paired_video"])

    deltas = {}
    delta_tsv = C.EVAL_DIR / "e208_aug_deltas.tsv"
    if delta_tsv.is_file():
        deltas = {(r["case_id"], r["aug_variant"]): r for r in C.read_tsv(delta_tsv)}

    rows: list[dict[str, Any]] = []
    unplayable: list[str] = []
    for (case_id, variant), item in sorted(by_key.items()):
        src = manifest.get((case_id, variant)) or authority.get(case_id, {})
        is_orig = variant == "orig"
        auth = authority.get(case_id, {})
        if is_orig:
            npz = auth.get("result_npz", "")
            scene = auth.get("scene_act", "")
            traj = auth.get("trajectory", "")
            video = auth.get("video", "")
            config_act = ""
        else:
            npz = src.get("outdir_npz", "")
            scene = src.get("scene_act", "")
            traj = src.get("trajectory", "")
            video = src.get("video", "")
            config_act = src.get("config_act", "")

        m = manual.get(case_id, {})
        d = deltas.get((case_id, variant), {})
        rec: dict[str, Any] = {
            "exp_id": C.EXP_ID,
            "arm": ARM_LABEL.get(variant, variant.upper()),
            "case_id": case_id,
            "variant": variant,
            "object_key": item.get("object_key", case_id.split("_")[0]),
            "retarget_variant_id": item.get("effective_retarget_variant")
            or src.get("effective_retarget_variant", ""),
            "status": "REUSED_E206" if is_orig else src.get("status", ""),
            "outdir_npz": npz if npz and C.repo_path(npz).is_file() else "",
            "scene_xml": scene,
            "config_act": config_act if config_act and C.repo_path(config_act).is_file() else "",
            "trajectory": traj,
            "video": video if video and C.repo_path(video).is_file() else "",
            "aug_variant": variant,
            "offset_band": item.get("offset_band", ""),
            "approach_trans_offset_m_max": item.get("approach_trans_offset_m_max", ""),
            "effective_retarget_variant": item.get("effective_retarget_variant", ""),
            "rescue_state": item.get("rescue_state", ""),
            "c4_all_gates_pass": item.get("all_gates_pass", ""),
            "c4_failure_modes": item.get("numeric_failure_modes", ""),
            "f15_divergent": item.get("f15_divergent", "0"),
            "contact_mask_blind_frac": item.get("contact_mask_blind_frac", ""),
            "orig_manual_use_decision": m.get("manual_use_decision", ""),
            "orig_manual_quality_label": m.get("manual_quality_label", ""),
            "orig_manual_failure_taxonomy": m.get("manual_failure_taxonomy", ""),
            "orig_numeric_release_pass": m.get("numeric_release_pass", ""),
            "paired_video": auth.get("video", ""),
        }
        for gf, (field, op, thr) in E206R.GATE_MAP.items():
            rec[gf] = E206R.gate_pass(field, op, thr, item)
        for metric in E206R.METRIC_COLUMNS:
            rec[metric] = item.get(metric, "")
        # the player's headline flag stays E206's 12-gate caliber (see docstring)
        failed = [g.replace("_gate_pass", "") for g in E206R.GATE_MAP
                  if rec[g] not in ("True", True)]
        rec["numeric_release_pass"] = str(not failed)
        rec["numeric_failure_modes"] = ",".join(failed)
        for metric in DELTA_ANCHORS:
            rec[f"delta_{metric}"] = d.get(f"delta_{metric}", "")
        for blank in REVIEW_BLANKS:
            rec[blank] = ""
        if not rec["outdir_npz"]:
            unplayable.append(f"{case_id}/{variant}")
        rows.append(rec)

    rows.sort(key=lambda r: (r["object_key"], r["case_id"],
                             list(ARM_LABEL).index(r["variant"])
                             if r["variant"] in ARM_LABEL else 99))
    out = C.repo_path(args.out)
    C.write_tsv(out, rows, cols)

    by_arm = Counter(r["arm"] for r in rows)
    with_video = sum(1 for r in rows if r["video"])
    print(f"[done] {C.rel(out)}  rows={len(rows)}  arms={dict(by_arm)}")
    print(f"  playable={len(rows) - len(unplayable)}  with_video={with_video}")
    # write_tsv serialises booleans lowercase while gate_pass() emits str(bool):
    # comparing against the literal "True" silently counted zero of both.
    print(f"  12-gate pass={sum(1 for r in rows if C.truth(r['numeric_release_pass']))}"
          f"  6-gate(C4) pass={sum(1 for r in rows if C.truth(r['c4_all_gates_pass']))}"
          f"  (of {len(rows)})")
    if unplayable:
        print(f"  [warn] {len(unplayable)} rows have no rollout npz: {unplayable[:5]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
