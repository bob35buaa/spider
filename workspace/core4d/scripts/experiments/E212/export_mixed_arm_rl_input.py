#!/usr/bin/env python3
"""E212: combined desk/chair RL export with a DIFFERENT arm chosen per case.

Scope: 21 cases (E206's 22 minus ``chair005_20231030_043_p1``, excluded by the
user).  One row per case.  The arm for each case is picked from four different
experiments -- see ``ARM_BY_CASE`` -- so this is the first export in the family
whose rollouts are not all from one run.

Why per-case arms
-----------------
E209 put ``gravcomp=1`` on all 22 desk/chair cases and failed its main gate, but
the damage was not uniform: ``chair006`` improved on every metric while
``desk007`` degraded on every one.  E211 (desk007) and E212 (desk023) then swept
partial compensation.  The per-case winner therefore differs by object and, on
desk007, by case.  This export takes the best available arm for each.

Design: seed-and-override (the E207 pattern)
--------------------------------------------
Each row starts as a copy of E206's already-validated ``rl_export_input.tsv``
row for that case, then only the arm-dependent fields are overwritten.  Two
reasons this beats rebuilding from Stage2b:

  * The 74-column schema and all static provenance (object_name/date/seq/person/
    trajectory/contact_mask/target_scene/stage2b_*) are inherited byte-for-byte,
    so the output stays drop-in compatible with the Holosoma consumer.
  * E206's ``paired_rl_export_input.tsv`` is the SHA-pinned authority that
    ``e209_common`` (and through it E211/E212) validates against.  Seeding from
    it means any drift in the shared provenance shows up as an assertion here
    rather than as a silent mismatch downstream.

The partner side is copied wholesale and is **arm-independent**: partner
identity is a ``_p1``/``_p2`` string flip, and the partner artifacts come from
``results/E206/s3_retarget/...``, which every one of these experiments reuses.
The same 21 partner rows would be produced for any arm choice.  The alignment
audit is still re-run per case, because it compares the *source rollout's* frame
count against the trajectory, and a divergent arm could in principle differ.

Honesty about review status
---------------------------
Only ``desk007_20231030_034_p1`` keeps ``manual_use_decision=USE``: it is the one
case whose exported rollout is the E206 PRG rollout a human actually reviewed.
Every other row is a G-arm that was never manually reviewed, so it is marked
``NOT_REVIEWED`` and E206's verdict is demoted to the ``prior_arm_manual_*``
columns (the E207 precedent).  ``arm`` / ``arm_experiment`` / ``arm_gravcomp`` /
``arm_selection_reason`` are appended so the mixture is auditable from the TSV
alone.

Usage:
    .venv/bin/python workspace/core4d/scripts/experiments/E212/export_mixed_arm_rl_input.py --dry-run
    .venv/bin/python workspace/core4d/scripts/experiments/E212/export_mixed_arm_rl_input.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[5]
_EXP = REPO / "workspace/core4d/scripts/experiments"
for _d in ("E212", "E211", "E209", "E206", "E200", "E199"):
    _p = str(_EXP / _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load_by_path(alias: str, path: Path):
    """Import by explicit path -- several E2xx dirs ship same-named modules and
    this file puts six of them on sys.path (E211 F5)."""
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


import e212_common as C12  # noqa: E402
import e211_common as C11  # noqa: E402
import e209_common as C09  # noqa: E402
import e206_common as C06  # noqa: E402

#: E206's exporter is reused as a LIBRARY for the partner + alignment block.
#: It has no CLI and its main() would re-validate E206's review authority, so we
#: import the module and call only the pieces we need (the E202 precedent).
E206X = _load_by_path(
    "e212_e206_exporter", _EXP / "E206/export_manual_use_partner_rl.py"
)
PARTNER = E206X.PARTNER

EXP = "E212"
OUT_DEFAULT = C12.S6_DIR / "rl_export"
E206_RL = REPO / "workspace/core4d/results/E206/s6_downstream/rl_export"
SEED_TSV = E206_RL / "rl_export_input.tsv"
SEED_PAIRED = E206_RL / "paired_rl_export_input.tsv"

#: E206's paired export is the pinned authority e209_common validates against;
#: if it moved, the static provenance we inherit is no longer what E209/E211/E212
#: ran against and this export would silently mix two generations of metadata.
EXPECTED_SEED_PAIRED_SHA256 = C09.EXPECTED_SOURCE_SHA256

#: Excluded by explicit user decision (it is E206/E209's 22nd case).
EXCLUDED_CASES = ("chair005_20231030_043_p1",)

#: ---------------------------------------------------------------------------
#: THE authority for "which rollout backs this case".  (experiment, arm).
#: Arm spelling is the METRICS/manifest spelling, not the CEM directory tag.
#: ---------------------------------------------------------------------------
_G1 = ("E209", "G1")
ARM_BY_CASE: dict[str, tuple[str, str]] = {
    # chair006 -- the only object that improved on EVERY metric under gravcomp=1
    "chair006_20231003_1_003_p1": _G1,
    "chair006_20231003_1_005_p1": _G1,
    "chair006_20231003_2_011_p2": _G1,
    "chair006_20231003_2_015_p1": _G1,
    "chair006_20231011_076_p2": _G1,
    # desk021 -- neutral under gravcomp=1 (narrow 4/4 -> 4/4), object side better
    "desk021_20231008_005_p1": _G1,
    "desk021_20231008_005_p2": _G1,
    "desk021_20231008_007_p1": _G1,
    "desk021_20231008_007_p2": _G1,
    "desk021_20231011_010_p1": _G1,
    "desk021_20231011_010_p2": _G1,
    "desk021_20231011_014_p2": _G1,
    # desk023 -- E212 Stage A; g=0.6 is the C1-closest arm (5/6 clauses)
    "desk023_20231008_066_p1": ("E212", "G06"),
    "desk023_20231008_066_p2": ("E212", "G06"),
    "desk023_20231011_005_p1": ("E212", "G06"),
    "desk023_20231030_019_p1": ("E212", "G06"),
    # desk007 -- E211 Stage A, per-case winner; 034_p1 keeps the PRG baseline
    "desk007_20231030_028_p1": ("E211", "G08"),
    "desk007_20231030_028_p2": ("E211", "G08"),
    "desk007_20231030_030_p2": ("E211", "G04"),
    "desk007_20231030_032_p2": ("E211", "G06"),
    "desk007_20231030_034_p1": ("E206", "PRG"),
}
EXPECTED_ROWS = 21

ARM_GRAVCOMP = {("E206", "PRG"): "0.0", ("E209", "G1"): "1.0",
                ("E211", "G04"): "0.4", ("E211", "G06"): "0.6", ("E211", "G08"): "0.8",
                ("E212", "G04"): "0.4", ("E212", "G06"): "0.6", ("E212", "G08"): "0.8"}

ARM_REASON = {
    ("E209", "G1"): "E209 g=1: object side improves, object-level no regression",
    ("E212", "G06"): "E212 Stage A: C1-closest arm (5/6 clauses, misses z by 0.026cm)",
    ("E211", "G08"): "E211 Stage A: per-case selection",
    ("E211", "G06"): "E211 Stage A: per-case selection",
    ("E211", "G04"): "E211 Stage A: per-case selection",
    ("E206", "PRG"): "E206 PRG baseline kept: every g>0 arm degraded this case",
}

#: Per-experiment evidence tables.
METRICS_TSV = {
    "E206": REPO / "workspace/core4d/results/E206/s6_downstream/eval/two_arm/e206_arm_case_metrics.tsv",
    "E209": REPO / "workspace/core4d/results/E209/s6_downstream/eval/two_arm/e209_arm_case_metrics.tsv",
}
SWEEP_TSV = {
    "E211": REPO / "workspace/core4d/results/E211/s6_downstream/eval/g_sweep/e211_g_sweep_rollout.tsv",
    "E212": REPO / "workspace/core4d/results/E212/s6_downstream/eval/g_sweep/e212_g_sweep_rollout.tsv",
}
MANIFEST_TSV = {
    "E211": REPO / "workspace/core4d/results/E211/s6_downstream/manifests/e211_stageA_full_manifest.tsv",
    "E212": REPO / "workspace/core4d/results/E212/s6_downstream/manifests/e212_stageA_full_manifest.tsv",
}
RUN_ID = {"E206": "R292", "E209": "R295", "E211": "R297", "E212": "R298"}

#: The 12 funnel gates, in the order E206's SOURCE_FIELDS lists them, paired with
#: the g_sweep column that carries the same verdict for E211/E212.
G12_TO_GATE = {
    "g12_fall": "fall_gate_pass",
    "g12_body_z": "body_z_gate_pass",
    "g12_contact": "contact_gate_pass",
    "g12_release": "release_gate_pass",
    "g12_hand_penetration": "hand_penetration_gate_pass",
    "g12_lower_body": "lower_body_gate_pass",
    "g12_root_pos": "root_pos_gate_pass",
    "g12_root_ori": "root_ori_gate_pass",
    "g12_hand_pos": "hand_pos_gate_pass",
    "g12_hand_ori": "hand_ori_gate_pass",
    "g12_object_pos": "object_pos_gate_pass",
    "g12_object_ori": "object_ori_gate_pass",
}

EXTRA_FIELDS = [
    "arm", "arm_experiment", "arm_gravcomp", "arm_scene_name", "arm_selection_reason",
    "selection_authority",
    "prior_arm_manual_use_decision", "prior_arm_manual_quality_label",
    "prior_arm_manual_reviewer", "prior_arm_cem_run_id",
]


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    p = Path(path)
    try:
        return p.relative_to(REPO).as_posix()
    except ValueError:
        return p.as_posix()


def expected_scene_name(exp: str, arm: str) -> str:
    if exp == "E206":
        return C06.SCENE_BY_ARM["prg"]
    if exp == "E209":
        return C09.SCENE
    if exp == "E211":
        return C11.SCENE_BY_ARM[arm]
    if exp == "E212":
        return C12.SCENE_BY_ARM[arm]
    raise ValueError(exp)


# ---------------------------------------------------------------------------
# Per-experiment adapters: (case_id, arm) -> the arm-dependent fields
# ---------------------------------------------------------------------------
def _from_metrics(exp: str, case_id: str, arm: str, cache: dict) -> dict[str, Any]:
    """E206 / E209: one 37-column metrics TSV carrying paths AND gate verdicts."""
    key = (exp, arm)
    if key not in cache:
        rows = read_tsv(METRICS_TSV[exp])
        cache[key] = {r["case_id"]: r for r in rows if r["arm"] == arm}
    m = cache[key].get(case_id)
    if m is None:
        raise SystemExit(f"{case_id}: no {exp} metrics row for arm={arm}")
    out = {
        "scene_act": m["scene_xml"],
        "cem_result_npz": m["outdir_npz"],
        "cem_video": m["video"],
        "cem_status": m["status"],
        "numeric_release_pass": m["numeric_release_pass"],
        "numeric_failure_modes": m["numeric_failure_modes"],
        "cem_metrics_ref": rel(METRICS_TSV[exp]),
        "metrics_sha256": sha256(METRICS_TSV[exp]),
        "_leg_pen": m["leg_penetration_frac"],
    }
    for gate in G12_TO_GATE.values():
        out[gate] = m[gate]
    return out


def _from_sweep(exp: str, case_id: str, arm: str, cache: dict) -> dict[str, Any]:
    """E211 / E212: paths come from the CEM manifest, gates from the g_sweep TSV.

    The two are separate because the sweep evaluator scores five arms (including
    the two frozen endpoints) and therefore carries no asset paths of its own.
    """
    mkey = ("man", exp)
    if mkey not in cache:
        cache[mkey] = {(r["case_id"], r["arm"]): r for r in read_tsv(MANIFEST_TSV[exp])}
    skey = ("sweep", exp)
    if skey not in cache:
        cache[skey] = {(r["case_id"], r["arm"]): r for r in read_tsv(SWEEP_TSV[exp])}
    man = cache[mkey].get((case_id, arm))
    swp = cache[skey].get((case_id, arm))
    if man is None:
        raise SystemExit(f"{case_id}: no {exp} manifest row for arm={arm}")
    if swp is None:
        raise SystemExit(f"{case_id}: no {exp} g_sweep row for arm={arm}")
    if man["status"] != "run_complete_pending_eval":
        raise SystemExit(f"{case_id}/{arm}: manifest status={man['status']!r}, not complete")
    out = {
        "scene_act": man["scene_act"],
        "cem_result_npz": man["outdir_npz"],
        "cem_video": man["video"],
        "cem_status": "pass",
        # narrow_pass is the sweep's equivalent of the 12-gate "all clear" verdict
        "numeric_release_pass": swp["narrow_pass"],
        "numeric_failure_modes": swp.get("narrow_failed", ""),
        "cem_metrics_ref": rel(SWEEP_TSV[exp]),
        "metrics_sha256": sha256(SWEEP_TSV[exp]),
        "_scene_sha256": man["effective_scene_sha256"],
        "_leg_pen": swp["leg_penetration_frac"],
    }
    for g12, gate in G12_TO_GATE.items():
        out[gate] = swp[g12]
    return out


def arm_fields(exp: str, case_id: str, arm: str, cache: dict) -> dict[str, Any]:
    if exp in METRICS_TSV:
        return _from_metrics(exp, case_id, arm, cache)
    return _from_sweep(exp, case_id, arm, cache)


# ---------------------------------------------------------------------------
def build_rows(dry_run: bool) -> tuple[list[dict[str, Any]], list[str]]:
    if not SEED_TSV.is_file():
        raise SystemExit(f"missing seed export: {SEED_TSV}")
    got = sha256(SEED_PAIRED)
    if got != EXPECTED_SEED_PAIRED_SHA256:
        raise SystemExit(
            f"seed authority drifted: {SEED_PAIRED}\n  sha256 {got}\n  expected "
            f"{EXPECTED_SEED_PAIRED_SHA256} (e209_common.EXPECTED_SOURCE_SHA256)"
        )
    seed = {r["case_id"]: r for r in read_tsv(SEED_TSV)}
    base_fields = list(next(iter(seed.values())))

    missing = sorted(set(ARM_BY_CASE) - set(seed))
    if missing:
        raise SystemExit(f"cases not in E206's export: {missing}")
    overlap = sorted(set(ARM_BY_CASE) & set(EXCLUDED_CASES))
    if overlap:
        raise SystemExit(f"excluded case is also mapped: {overlap}")
    unmapped = sorted(set(seed) - set(ARM_BY_CASE) - set(EXCLUDED_CASES))
    if unmapped:
        raise SystemExit(f"seed case neither mapped nor explicitly excluded: {unmapped}")
    if len(ARM_BY_CASE) != EXPECTED_ROWS:
        raise SystemExit(f"{len(ARM_BY_CASE)} mapped cases, expected {EXPECTED_ROWS}")

    cache: dict = {}
    rows: list[dict[str, Any]] = []
    for case_id in sorted(ARM_BY_CASE):
        exp, arm = ARM_BY_CASE[case_id]
        row = dict(seed[case_id])
        af = arm_fields(exp, case_id, arm, cache)

        scene_act = REPO / af["scene_act"]
        cem_npz = REPO / af["cem_result_npz"]
        for label, p in (("scene_act", scene_act), ("cem_result_npz", cem_npz)):
            if not p.is_file():
                raise SystemExit(f"{case_id}/{exp}/{arm}: missing {label}: {p}")

        # Per-case arm guard (E206 line 310 hardcodes ONE scene name; a mixed-arm
        # export must assert the per-case expectation instead of dropping the check).
        want_scene = expected_scene_name(exp, arm)
        if want_scene not in scene_act.name:
            raise SystemExit(
                f"{case_id}: scene_act {scene_act.name!r} does not match arm "
                f"{exp}/{arm} (expected {want_scene!r})"
            )

        # Static provenance is inherited; the trajectory/contact_mask must not move.
        traj = REPO / row["trajectory"]
        if not traj.is_file():
            raise SystemExit(f"{case_id}: inherited trajectory missing: {traj}")

        leg_pen = af.get("_leg_pen", "")
        try:
            leg_ok = str(float(leg_pen) <= 0.20)
        except (TypeError, ValueError):
            leg_ok = ""

        row.update({
            "source_exp_id": exp,
            "scene_act": af["scene_act"],
            "cem_result_npz": af["cem_result_npz"],
            "cem_video": af["cem_video"],
            "cem_status": af["cem_status"],
            "cem_run_id": RUN_ID[exp],
            "cem_metrics_ref": af["cem_metrics_ref"],
            "metrics_sha256": af["metrics_sha256"],
            "numeric_release_pass": af["numeric_release_pass"],
            "numeric_failure_modes": af["numeric_failure_modes"],
            "leg_gate_health_pass": leg_ok,
            "result_sha256": sha256(cem_npz),
            "scene_sha256": af.get("_scene_sha256") or sha256(scene_act),
            "trajectory_sha256": sha256(traj),
            "rl_export_decision": "RL_EXPORT_READY",
            "skip_reason": "",
            "scene_act_exists": "True",
            "trajectory_exists": "True",
            "cem_result_exists": "True",
            "execution_kind": f"{exp}_{arm}_FULL_COMPLETE",
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "evaluation_manifest_sha256": af["metrics_sha256"],
        })
        for gate in G12_TO_GATE.values():
            row[gate] = af[gate]

        # Review honesty: only the E206 PRG row carries a real human verdict.
        reviewed = (exp, arm) == ("E206", "PRG")
        row.update({
            "arm": arm,
            "arm_experiment": exp,
            "arm_gravcomp": ARM_GRAVCOMP[(exp, arm)],
            "arm_scene_name": want_scene,
            "arm_selection_reason": ARM_REASON[(exp, arm)],
            "selection_authority": (
                "E206 manual review (USE)" if reviewed
                else "E212 mixed-arm map (no manual review of this arm)"
            ),
            "prior_arm_manual_use_decision": seed[case_id].get("manual_use_decision", ""),
            "prior_arm_manual_quality_label": seed[case_id].get("manual_quality_label", ""),
            "prior_arm_manual_reviewer": seed[case_id].get("manual_reviewer", ""),
            "prior_arm_cem_run_id": seed[case_id].get("cem_run_id", ""),
        })
        if not reviewed:
            row.update({
                "manual_use_decision": "NOT_REVIEWED",
                "manual_quality_label": "",
                "manual_failure_taxonomy": "",
                "manual_review_note": "",
                "manual_reviewer": "",
                "manual_reviewed_at": "",
                "manual_review_ref": "",
                "manual_review_sha256": "",
                "visual_qc_status": "not_reviewed",
            })
        rows.append(row)

    fields = base_fields + [f for f in EXTRA_FIELDS if f not in base_fields]
    return rows, fields


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows, fields = build_rows(args.dry_run)
    by_obj: dict[str, int] = {}
    by_arm: dict[str, int] = {}
    for r in rows:
        by_obj[r["object_key"]] = by_obj.get(r["object_key"], 0) + 1
        tag = f"{r['arm_experiment']}/{r['arm']}"
        by_arm[tag] = by_arm.get(tag, 0) + 1
    for r in rows:
        print(f"  {r['case_id']:32s} {r['arm_experiment']}/{r['arm']:4s} "
              f"g={r['arm_gravcomp']}  narrow={r['numeric_release_pass']:5s}  "
              f"{Path(r['cem_result_npz']).parent.name}")
    print(f"\n  objects={by_obj}\n  arms={by_arm}")

    if args.dry_run:
        print(f"\n[dry-run] {len(rows)} rows, {len(fields)} columns; nothing written")
        return 0

    # Partner resolution -- arm-independent, wired EXACTLY as E206's main() does
    # (mirrors export_manual_use_partner_rl.py:643-742). Partner identity is a
    # _p1/_p2 flip and the artifacts come from E206's s3_retarget tree, so the
    # 21 partner rows are identical regardless of which source arm we picked.
    stage2b_index = PARTNER.load_stage2b_index(E206X.STAGE2B_MANIFESTS)
    direct_partners = E206X.load_direct_partners()
    BLOCKED = E206X.BLOCKED_DECISION

    # Source-side stage2b provenance/evidence, needed by alignment_audit. Every
    # source case here is one E206 exported, so it must have exactly one passing
    # ref_fk row (source_stage2b raises otherwise).
    source_stage: dict[str, tuple[Path, dict[str, str]]] = {}
    for src in rows:
        source_stage[src["case_id"]] = E206X.source_stage2b(src["case_id"], stage2b_index)

    staging = args.out_dir.parent / f".{args.out_dir.name}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "partner_omnirt").mkdir(parents=True)

    source_path = staging / "rl_export_input.tsv"
    write_tsv(source_path, rows, fields)
    (staging / "rl_export_input.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    partner_rows, audits, blocked = [], [], []
    for src in rows:
        partner_case, partner_person, partner_idx = PARTNER.infer_partner(src)
        candidates = stage2b_index.get(partner_case, [])
        direct = direct_partners.get(partner_case)
        source_provenance, source_evidence = source_stage[src["case_id"]]
        if candidates:
            partner_provenance, partner_evidence = PARTNER.choose_partner(
                candidates, PARTNER.DEFAULT_VARIANT_PREFERENCE, partner_case)
            prow = PARTNER.build_partner_row(
                src, partner_case=partner_case, partner_person=partner_person,
                partner_person_idx=partner_idx, evidence=partner_evidence,
                provenance=partner_provenance, source_rl=source_path, repo=REPO,
                generation_mode="reuse_e206_stage2b_partner")
        elif direct:
            partner_evidence = direct
            partner_provenance = Path(direct["_manifest"])
            prow = E206X.direct_partner_row(
                src, direct, partner_case=partner_case, partner_person=partner_person,
                partner_person_idx=partner_idx, source_rl=source_path)
        else:
            partner_provenance, partner_evidence = None, None
            prow = E206X.blocked_partner_row(
                src, partner_case, partner_person, partner_idx, source_path,
                "partner never entered E206: no Stage2b row and no direct fallback")
            blocked.append(src["case_id"])
        partner_rows.append(prow)
        audits.append(E206X.alignment_audit(
            src, source_evidence, source_provenance,
            prow, partner_evidence, partner_provenance))

    write_tsv(staging / "partner_resolution_audit.tsv", audits, E206X.ALIGNMENT_FIELDS)
    # A resolved-but-misaligned partner is a real defect; a structurally blocked
    # one is a reported scope gap, not a failure (E206's distinction).
    misaligned = [a for a in audits if a["alignment_status"] not in ("RL_EXPORT_READY", BLOCKED)]
    if misaligned:
        for a in misaligned:
            print(f"  MISALIGNED {a['source_case_id']}: {a.get('alignment_failure_mode')}")
        raise SystemExit(f"partner alignment failed for {len(misaligned)} case(s)")

    pdir = staging / "partner_omnirt"
    partner_manifest = pdir / "rl_partner_omnirt_manifest.tsv"
    write_tsv(partner_manifest, partner_rows, PARTNER.PARTNER_FIELDS)
    (pdir / "rl_partner_omnirt_manifest.json").write_text(
        json.dumps(partner_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    paired_fields = fields + [f for f in PARTNER.PAIRED_EXTRA_FIELDS if f not in fields]
    manifest_hash = sha256(partner_manifest)
    paired = [PARTNER.paired_row(s, p, manifest_ref=rel(partner_manifest),
                                 manifest_hash=manifest_hash, repo=REPO)
              for s, p in zip(rows, partner_rows)]
    write_tsv(staging / "paired_rl_export_input.tsv", paired, paired_fields)
    (staging / "paired_rl_export_input.json").write_text(
        json.dumps(paired, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    ready = sum(1 for p in paired if p.get("paired_rl_export_decision") == "RL_EXPORT_READY")
    summary = {
        "experiment": EXP,
        "stage": "s6_downstream/rl_export",
        "kind": "mixed_arm_per_case",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "RL_EXPORT_READY" if not blocked else "PARTIAL",
        "source_rows": len(rows),
        "expected_rows": EXPECTED_ROWS,
        "excluded_cases": list(EXCLUDED_CASES),
        "excluded_reason": "user decision: chair005 out of scope for this delivery",
        "arm_by_case": {k: list(v) for k, v in sorted(ARM_BY_CASE.items())},
        "arm_counts": by_arm,
        "object_counts": by_obj,
        "seed_export": rel(SEED_TSV),
        "seed_paired_sha256": EXPECTED_SEED_PAIRED_SHA256,
        "partner_rows": len(partner_rows),
        "paired_ready_rows": ready,
        "paired_blocked_cases": blocked,
        "numeric_release_counts": {
            k: sum(1 for r in rows if str(r["numeric_release_pass"]) == k)
            for k in ("True", "False")
        },
        "manual_review_note": (
            "Only desk007_20231030_034_p1 (E206 PRG) carries a real human verdict; "
            "every other row is a G-arm marked NOT_REVIEWED with the E206 verdict "
            "demoted to prior_arm_manual_*."
        ),
        "claim_boundary": (
            "Arm selection is per-case and NOT the output of a single pre-registered "
            "gate: E209 G1 for chair006/desk021, E212 g=0.6 for desk023 (C1-closest, "
            "C1 itself FAILED), E211 per-case for desk007. See log298/log300/log301."
        ),
    }
    (staging / "rl_export_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    staging.rename(args.out_dir)

    # Re-pin the partner manifest's back-reference + hash to the PUBLISHED source
    # tsv, then re-emit the manifest and paired files (E206:813-823). source_rl
    # was written into partner rows as the staging path; publish moved it.
    published_source = args.out_dir / "rl_export_input.tsv"
    published_sha = sha256(published_source)
    for prow in partner_rows:
        prow["source_rl_export_input"] = rel(published_source)
        prow["source_rl_export_input_sha256"] = published_sha
    pdir = args.out_dir / "partner_omnirt"
    partner_manifest = pdir / "rl_partner_omnirt_manifest.tsv"
    write_tsv(partner_manifest, partner_rows, PARTNER.PARTNER_FIELDS)
    (pdir / "rl_partner_omnirt_manifest.json").write_text(
        json.dumps(partner_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_hash = sha256(partner_manifest)
    paired = [PARTNER.paired_row(s, p, manifest_ref=rel(partner_manifest),
                                 manifest_hash=manifest_hash, repo=REPO)
              for s, p in zip(rows, partner_rows)]
    write_tsv(args.out_dir / "paired_rl_export_input.tsv", paired, paired_fields)
    (args.out_dir / "paired_rl_export_input.json").write_text(
        json.dumps(paired, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    validation = {
        **summary,
        "checks": {
            "seed_authority_sha256_pinned": True,
            "all_cases_mapped_or_explicitly_excluded": True,
            "per_case_arm_scene_name_matches": True,
            "source_required_files_loadable": True,
            "partner_resolution_complete": len(partner_rows) == len(rows),
            "partner_alignment_no_misalignment": not misaligned,
            "paired_rows_equal_source_rows": len(paired) == len(rows),
        },
        "artifact_sha256": {
            p.name: sha256(p) for p in sorted(args.out_dir.rglob("*.tsv"))
        },
    }
    (args.out_dir / "validation_report.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\nwrote {len(rows)} source + {len(partner_rows)} partner rows -> "
          f"{rel(args.out_dir)}")
    print(f"  paired ready {ready}/{len(paired)}  blocked={blocked}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
