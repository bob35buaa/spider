#!/usr/bin/env python3
"""E216 keep-best → downstream RL source export (81-col rl_export_input).

Phase 1 of the E216 paired RL export. Joins the user-selected keep-best
augmented sequences (``E216_aug_keepbest1.xlsx`` / sheet ``Selected_best``)
with the frozen priority manifest (paths + sha) and the 14-gate funnel detail
(NARROW gate pass/fail), emitting the same 81-column source schema the E213
export used. Downstream-only: does not touch the v3 registry or CEM evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

import openpyxl

SCRIPT_ROOT = next(p for p in Path(__file__).resolve().parents if p.name == "data_construction_v3") \
    if any(p.name == "data_construction_v3" for p in Path(__file__).resolve().parents) else None
# common.py lives under the data_construction_v3 tree; add it explicitly.
_COMMON = Path("workspace/core4d/scripts/data_construction_v3/lib")
if _COMMON.is_dir() and str(_COMMON) not in sys.path:
    sys.path.insert(0, str(_COMMON))
from common import read_tsv, timestamp, write_json, write_tsv  # noqa: E402

SCHEMA_VERSION = "core4d_data_construction_v3.0"
SOURCE_EXP_ID = "E216-export"
CEM_RUN_ID = "R304"

# 81-column source schema (exact order from E213 rl_export_input.tsv).
SOURCE_FIELDS = [
    "case_id", "object_key", "object_name", "date", "seq", "person", "person_idx",
    "retarget_variant_id", "target_variant_id", "hand_collision_variant_id",
    "source_exp_id", "spider_method_id", "handoff_decision", "candidate_decision",
    "target_gate_status", "visual_qc_status", "target_scene", "trajectory",
    "scene_act", "contact_mask", "stage2b_target_task", "stage2b_result_root",
    "stage2b_manifest_ref", "raw_contact_threshold_label", "cem_status",
    "cem_run_id", "cem_result_npz", "cem_video", "cem_metrics_ref",
    "downstream_decision", "downstream_failure_mode", "downstream_notes",
    "rl_export_decision", "skip_reason", "scene_act_exists", "trajectory_exists",
    "contact_mask_exists", "cem_result_exists", "source_handoff_manifest",
    "source_cem_evidence", "schema_version", "updated_at", "manual_use_decision",
    "manual_quality_label", "manual_failure_taxonomy", "manual_review_note",
    "manual_reviewer", "manual_reviewed_at", "manual_review_ref",
    "manual_review_sha256", "numeric_release_pass", "numeric_failure_modes",
    "leg_gate_health_pass", "fall_gate_pass", "body_z_gate_pass",
    "contact_gate_pass", "release_gate_pass", "hand_penetration_gate_pass",
    "lower_body_gate_pass", "root_pos_gate_pass", "root_ori_gate_pass",
    "hand_pos_gate_pass", "hand_ori_gate_pass", "object_pos_gate_pass",
    "object_ori_gate_pass", "c9_technical_status", "c9_progression_authority",
    "execution_kind", "result_sha256", "scene_sha256", "trajectory_sha256",
    "contact_mask_sha256", "metrics_sha256", "evaluation_manifest_sha256",
    "aug_variant", "arm", "arm_experiment", "arm_gravcomp", "arm_scene_name",
    "arm_selection_reason", "selection_authority",
]

# funnel NARROW gate column -> schema gate column.
GATE_MAP = {
    "fall": "fall_gate_pass",
    "body_z": "body_z_gate_pass",
    "contact": "contact_gate_pass",
    "release": "release_gate_pass",
    "hand_pen": "hand_penetration_gate_pass",
    "leg_pen": "lower_body_gate_pass",
    "root_pos": "root_pos_gate_pass",
    "root_ori": "root_ori_gate_pass",
    "eef_pos": "hand_pos_gate_pass",
    "eef_ori": "hand_ori_gate_pass",
    "obj_pos": "object_pos_gate_pass",
    "obj_ori": "object_ori_gate_pass",
}
# failure-mode naming for the 4 physics dims.
FAILMODE = {"hand_pen": "hand_penetration", "release": "release",
            "leg_pen": "lower_body_penetration", "contact": "contact"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_case(case_id: str, object_key: str) -> tuple[str, str, str, str]:
    """Return (date, seq, person, person_idx) from a per-person case_id."""
    rest = case_id[len(object_key) + 1:]
    m = re.match(r"(.+)_p(\d+)$", rest)
    if not m:
        raise SystemExit(f"cannot parse person from case_id: {case_id}")
    date_seq, pn = m.group(1), m.group(2)
    date, seq = date_seq.rsplit("_", 1)
    return date, seq, f"person{pn}", str(int(pn) - 1)


def read_selected(xlsx: Path) -> list[dict]:
    wb = openpyxl.load_workbook(xlsx, data_only=True)
    ws = wb["Selected_best"]
    hdr = [c.value for c in ws[1]]
    idx = {h: i for i, h in enumerate(hdr)}
    rows = []
    for r in ws.iter_rows(min_row=2, values_only=True):
        rows.append({
            "track": r[idx["track"]],
            "object_key": r[idx["object_key"]],
            "case_id": r[idx["case_id"]],
            "variant": r[idx["best_variant"]],
            "layer": r[idx["layer"]],
            "npass": r[idx["NARROW_pass(/14)"]],
            "trk": r[idx["track_err(↓)"]],
        })
    return rows


def read_funnel_gates(xlsx: Path) -> dict[tuple[str, str], dict[str, str]]:
    """Map (case_id, aug_variant) -> {narrow gate name -> PASS/FAIL}."""
    wb = openpyxl.load_workbook(xlsx, data_only=True)
    ws = wb["Detail"]
    hdr = [c.value for c in ws[1]]
    idx = {h: i for i, h in enumerate(hdr)}
    # hard gate pass cols are "<name>✓"; banded NARROW cols are "<name> N(..)".
    narrow_col = {}
    for name in GATE_MAP:
        if name in ("fall", "body_z"):
            narrow_col[name] = idx[f"{name}✓"]
        else:
            col = next(i for i, h in enumerate(hdr)
                       if isinstance(h, str) and h.startswith(f"{name} N("))
            narrow_col[name] = col
    out = {}
    ci, vi = idx["case_id"], idx["aug_variant"]
    for r in ws.iter_rows(min_row=2, values_only=True):
        key = (r[ci], r[vi])
        out[key] = {name: r[col] for name, col in narrow_col.items()}
    return out


def tf(pass_fail) -> str:
    return "true" if pass_fail == "PASS" else "false"


def build(args) -> int:
    repo = Path.cwd()
    selected = read_selected(args.keepbest_xlsx)

    # Merge both tracks' frozen priority manifests, tagging each row with its
    # track provenance (manifest path, rollout evidence, run id, exp id).
    man_idx: dict[tuple[str, str], tuple[dict, dict]] = {}
    man_sha_cache: dict[str, str] = {}
    for spec in args.manifest_specs:
        mpath = Path(spec["manifest"])
        for row in read_tsv(mpath):
            man_idx[(row["case_id"], row["aug_variant"])] = (row, spec)
        man_sha_cache[spec["manifest"]] = sha256(mpath)
        rpath = Path(spec["rollout"])
        man_sha_cache[spec["rollout"]] = sha256(rpath) if rpath.is_file() else ""

    gates = {}
    for fx in args.funnel_xlsx:
        gates.update(read_funnel_gates(Path(fx)))

    selection_authority = str(args.selection_snapshot)

    out_rows: list[dict] = []
    missing = []
    for sel in selected:
        key = (sel["case_id"], sel["variant"])
        hit = man_idx.get(key)
        if hit is None:
            missing.append(key)
            continue
        man, spec = hit
        manifest_path = spec["manifest"]
        rollout_tsv = spec["rollout"]
        manifest_sha = man_sha_cache[manifest_path]
        metrics_sha = man_sha_cache[rollout_tsv]
        object_key = sel["object_key"]
        date, seq, person, person_idx = parse_case(sel["case_id"], object_key)
        g = gates.get(key, {})

        # per-row artifact paths from the frozen manifest
        trajectory = man["trajectory"]
        scene_act = man["scene_act"]
        contact_mask = man["contact_mask"]
        cem_result = man["outdir_npz"]  # trajectory_mjwp_act.npz (matches E213)

        def exists(rel: str) -> str:
            p = Path(rel)
            p = p if p.is_absolute() else repo / p
            return str(p.is_file() and p.stat().st_size > 0)

        cem_path = Path(cem_result if Path(cem_result).is_absolute() else repo / cem_result)
        result_sha = sha256(cem_path) if cem_path.is_file() else ""

        # gate failure bookkeeping on the 4 physics dims
        failed = [FAILMODE[d] for d in ("hand_pen", "release", "leg_pen", "contact")
                  if g.get(d) not in ("PASS", None)]
        fail_modes = ",".join(failed)

        case_case = sel["case_id"]
        row = {
            "case_id": case_case,
            "object_key": object_key,
            "object_name": object_key,
            "date": date,
            "seq": seq,
            "person": person,
            "person_idx": person_idx,
            "retarget_variant_id": man.get("base_variant", "omnirt_v1"),
            "target_variant_id": "ref_fk",
            "hand_collision_variant_id": "rubber_hull",
            "source_exp_id": spec["exp_id"],
            "spider_method_id": f"E216_selected_keepbest_aug_{man.get('arm_group', '')}",
            "handoff_decision": "HANDOFF_READY",
            "candidate_decision": "USER_SELECTED_E216_KEEPBEST_XLSX",
            "target_gate_status": "pass",
            "visual_qc_status": "xlsx_selected",
            "target_scene": man["target_scene"],
            "trajectory": trajectory,
            "scene_act": scene_act,
            "contact_mask": contact_mask,
            "stage2b_target_task": man["target_task"],
            "stage2b_result_root": "",
            "stage2b_manifest_ref": manifest_path,
            "raw_contact_threshold_label": "3cm",
            "cem_status": "pass",
            "cem_run_id": spec["run_id"],
            "cem_result_npz": cem_result,
            "cem_video": man.get("video", ""),
            "cem_metrics_ref": rollout_tsv,
            "downstream_decision": "DOWNSTREAM_USER_APPROVED_FOR_RL_VALIDATION",
            "downstream_failure_mode": failed[0] if failed else "",
            "downstream_notes": (
                f"E216-export keep-best aug unit ({sel['layer']}, {sel['variant']}); "
                f"xlsx-selected; arm={man.get('arm_group', '')} "
                f"effective_retarget={man.get('effective_retarget_variant', '')}; "
                f"NARROW_pass={sel['npass']}/14 track_err={sel['trk']}; "
                "two-person object consistency via downstream partner re-anchor"
            ),
            "rl_export_decision": "RL_EXPORT_READY",
            "skip_reason": "",
            "scene_act_exists": exists(scene_act),
            "trajectory_exists": exists(trajectory),
            "contact_mask_exists": exists(contact_mask),
            "cem_result_exists": exists(cem_result),
            "source_handoff_manifest": manifest_path,
            "source_cem_evidence": rollout_tsv,
            "schema_version": SCHEMA_VERSION,
            "updated_at": timestamp(),
            "manual_use_decision": "USER_SELECTED_E216_KEEPBEST_XLSX",
            "manual_quality_label": "",
            "manual_failure_taxonomy": "",
            "manual_review_note": "",
            "manual_reviewer": "",
            "manual_reviewed_at": "",
            "manual_review_ref": str(args.selection_snapshot),
            "manual_review_sha256": args.selection_snapshot_sha,
            "numeric_release_pass": tf(g.get("release")),
            "numeric_failure_modes": fail_modes,
            "leg_gate_health_pass": tf(g.get("leg_pen")),
            "c9_technical_status": "",
            "c9_progression_authority": "USER_SELECTED_E216_KEEPBEST_XLSX",
            "execution_kind": "E216_keepbest_aug_FULL_COMPLETE",
            "result_sha256": result_sha,
            "scene_sha256": man.get("effective_scene_sha256", ""),
            "trajectory_sha256": man.get("trajectory_sha256", ""),
            "contact_mask_sha256": man.get("contact_mask_sha256", ""),
            "metrics_sha256": metrics_sha,
            "evaluation_manifest_sha256": manifest_sha,
            "aug_variant": sel["variant"],
            "arm": man.get("arm_group", ""),
            "arm_experiment": "E216",
            "arm_gravcomp": man.get("gravcomp", ""),
            "arm_scene_name": man.get("scene_name", ""),
            "arm_selection_reason": (
                "E216 keep-best: layer>NARROW-pass>track-err per two-person case"
            ),
            "selection_authority": selection_authority,
        }
        # fill all gate columns
        for name, col in GATE_MAP.items():
            row[col] = tf(g.get(name))
        out_rows.append(row)

    if missing:
        raise SystemExit(f"{len(missing)} selected rows not found in priority manifest: {missing[:5]}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_tsv = args.out_dir / "rl_export_input.tsv"
    write_tsv(out_tsv, out_rows, SOURCE_FIELDS)
    write_json(args.out_dir / "rl_export_input.json", out_rows)

    # report
    missing_art = [r["case_id"] for r in out_rows
                   if "False" in (r["trajectory_exists"], r["scene_act_exists"],
                                  r["contact_mask_exists"], r["cem_result_exists"])]
    print(f"wrote {len(out_rows)} source rows -> {out_tsv}")
    print(f"artifacts missing on disk: {len(missing_art)} {missing_art[:10]}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    base = Path("workspace/core4d/results/E216")
    ap.add_argument("--keepbest-xlsx", type=Path,
                    default=base / "s6_downstream/eval/aug/E216_aug_keepbest1.xlsx")
    ap.add_argument("--funnel-xlsx", action="append", default=[
        str(base / "s6_downstream/eval/aug/E216_trans_vs_orig_funnel.xlsx"),
        str(base / "chair_desk/s6_downstream/eval/aug/E216cd_trans_0p1m_vs_0p2m.xlsx"),
    ])
    ap.add_argument("--selection-snapshot", type=Path,
                    default=base / "s6_downstream/export/selection_snapshot.xlsx")
    ap.add_argument("--selection-snapshot-sha", default="")
    ap.add_argument("--out-dir", type=Path, default=base / "s6_downstream/export")
    args = ap.parse_args()
    # Two tracks, each with its own frozen priority manifest + rollout evidence.
    args.manifest_specs = [
        {
            "manifest": str(base / "s6_downstream/manifests/e216_priority_manifest.tsv"),
            "rollout": str(base / "s6_downstream/eval/aug/e216_trans_rollout.tsv"),
            "run_id": "R304",
            "exp_id": "E216-export",
        },
        {
            "manifest": str(base / "chair_desk/s6_downstream/manifests/e216cd_priority_manifest.tsv"),
            "rollout": str(base / "chair_desk/s6_downstream/eval/aug/e216cd_trans_rollout.tsv"),
            "run_id": "E216CD",
            "exp_id": "E216CD-export",
        },
    ]
    return build(args)


if __name__ == "__main__":
    raise SystemExit(main())
