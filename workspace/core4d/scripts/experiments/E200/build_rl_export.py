#!/usr/bin/env python3
"""Assemble the three E200 paired RL-export versions (63-col historical schema).

For every selected case:
  * orig  -> reuse the fully-built paired row from the source experiment's
             paired_rl_export_input.tsv (E198 G1A2 / E190 noPRG / E170 PRG),
             which already carries a validated partner.
  * aug   -> synthesize the base row from the arm's priority manifest (scene_act,
             trajectory, contact_mask, cem_result_npz) and attach the augmented
             partner from partner_aug_motion_index.tsv.

A master table is built over the 5object_all selection (the superset), then each
version is emitted by filtering to its own selection.tsv. Field order matches the
E198/E190 63-column paired schema exactly. Nothing is faked: rows whose files or
partner motion are missing are marked non-ready and reported.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

DCV3 = Path(__file__).resolve().parents[1].parent / "data_construction_v3"
sys.path.insert(0, str(DCV3 / "stages/s6_downstream"))
import finalize_reused_partner_rl as F  # noqa: E402  (stdlib-only; schema + helpers)

# base 42-col rl_export schema (copied from export_rl_inputs.FIELDS to avoid mujoco import)
BASE_FIELDS = [
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
    "source_cem_evidence", "schema_version", "updated_at",
]
FULL_FIELDS = BASE_FIELDS + F.PAIRED_EXTRA_FIELDS  # 42 + 21 = 63

RES = Path("workspace/core4d/results")
PAIRED_BY_OBJECT = {
    "box001": RES / "E198/s6_downstream/rl_export/box001_user_approved/partner_omnirt/paired_rl_export_input.tsv",
    "box024": RES / "E198/s6_downstream/rl_export/box024_user_approved/partner_omnirt/paired_rl_export_input.tsv",
    "box004": RES / "E198/s6_downstream/rl_export/box004_user_approved/partner_omnirt/paired_rl_export_input.tsv",
    "box023": RES / "E190/s6_downstream/rl_export/box023_noPRG_user_approved/paired_rl_export_input.tsv",
    "box021": RES / "E170/s6_downstream/rl_export/paired_rl_export_input.tsv",
}
ARM_MANIFEST = {
    "box001": RES / "E200/s6_downstream/manifests/e200_prg_g1a2_priority_manifest.tsv",
    "box024": RES / "E200/s6_downstream/manifests/e200_prg_g1a2_priority_manifest.tsv",
    "box004": RES / "E200/s6_downstream/manifests/e200_prg_g1a2_priority_manifest.tsv",
    "box023": RES / "E200/s6_downstream/manifests/e200_noprg_priority_manifest.tsv",
    "box021": RES / "E199/s6_downstream/manifests/e199_fullscale_priority_manifest.tsv",
}
ARM_METHOD = {
    "PRG+G1+A2": "E200_PRG_G1A2_omnirt_v2_aug",
    "noPRG": "E200_noPRG_omnirt_v2_aug",
    "PRG": "E199_PRG_omnirt_v2_aug",
}
TASK_ROOT = Path("example_datasets/processed/core4d/unitree_g1/humanoid_object")
PERSON_IDX = {"person1": "0", "person2": "1"}
SHORT_PERSON = {"p1": "person1", "p2": "person2"}


def norm_pn(case_id: str) -> str:
    """Normalize a case id to the ``_p1``/``_p2`` short-person form."""
    return case_id.replace("_person1", "_p1").replace("_person2", "_p2")


# path columns that may carry a stale absolute path from a reused historical
# manifest (E170/E168 stored old-mount absolutes; E198/E190 stored repo-relative).
PATH_FIELDS = [
    "target_scene", "trajectory", "scene_act", "contact_mask", "cem_result_npz",
    "cem_video", "cem_metrics_ref", "stage2b_result_root", "stage2b_manifest_ref",
    "source_handoff_manifest", "source_cem_evidence", "partner_trimmed_npz",
    "partner_omniretarget_output_npz", "partner_trim_window_json", "partner_manifest_ref",
]


def remap_path(value: str) -> str:
    """Re-root a stale absolute artifact path onto the current repo (idempotent).

    Reused E170/E168 paired rows stored absolute paths under an old mount
    (``/.../spider_workdirs/core4d/results/...``). Map any absolute path back to
    a repo-relative one via the stable ``core4d/results/`` and ``example_datasets/``
    markers; leave already-relative paths untouched.
    """
    out = value
    if value.startswith("/"):
        for marker, prefix in (
            ("/example_datasets/", "example_datasets/"),
            ("/core4d/results/", "workspace/core4d/results/"),
        ):
            idx = value.find(marker)
            if idx >= 0:
                out = prefix + value[idx + len(marker):]
                break
    # collapse a historical typo ``results/results/`` (present in the E198 source
    # manifest); legitimate paths never contain consecutive ``results/`` segments.
    while "results/results/" in out:
        out = out.replace("results/results/", "results/")
    return out


def remap_row_paths(row: dict[str, str]) -> dict[str, str]:
    """Return a copy with PATH_FIELDS re-rooted onto the current repo."""
    return {k: (remap_path(v) if k in PATH_FIELDS else v) for k, v in row.items()}


def read_tsv_rows(path: Path) -> list[dict[str, str]]:
    """Read a TSV into dict rows."""
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def exists_and_sha(value: str, repo: Path) -> tuple[bool, str]:
    """Return (exists_nonempty, sha256-or-empty) for a repo-relative artifact."""
    if not value:
        return False, ""
    path = F.local_path(value, repo)
    if path.is_file() and path.stat().st_size > 0:
        return True, F.sha256(path)
    return False, ""


def load_orig_index() -> dict[tuple[str, str], dict[str, str]]:
    """Index every source paired manifest by (object, case_id)."""
    idx: dict[tuple[str, str], dict[str, str]] = {}
    for obj, path in PAIRED_BY_OBJECT.items():
        for row in read_tsv_rows(path):
            idx[(obj, row["case_id"])] = row
    return idx


def load_aug_index() -> dict[tuple[str, str, str], dict[str, str]]:
    """Index aug rows by (object, case_id_pn, variant), each object from ITS arm's manifest.

    An object appears in more than one arm's manifest (every arm ran every object),
    so we take each object only from the manifest assigned to it in ARM_MANIFEST to
    avoid cross-arm key collisions (e.g. box001 must come from prg_g1a2, not noprg).
    """
    idx: dict[tuple[str, str, str], dict[str, str]] = {}
    for obj, path in ARM_MANIFEST.items():
        for row in read_tsv_rows(path):
            if row["object_key"] != obj:
                continue
            idx[(obj, norm_pn(row["case_id"]), row["aug_variant"])] = row
    return idx


def task_info(target_task: str) -> dict[str, str]:
    """Read object_name/date/seq/person from an aug task_info.json."""
    return json.loads((TASK_ROOT / target_task / "task_info.json").read_text(encoding="utf-8"))


def build_aug_row(
    sel: dict[str, str],
    man: dict[str, str],
    partner: dict[str, str] | None,
    repo: Path,
    partner_index_ref: str,
    partner_index_sha: str,
) -> dict[str, str]:
    """Synthesize a 63-col paired row for one augmented case."""
    obj, case_pn, variant = sel["object"], norm_pn(sel["case_id"]), sel["variant"]
    info = task_info(man["target_task"])
    person = info["person"]
    scene_ok, scene_sha = exists_and_sha(man["scene_act"], repo)
    traj_ok, traj_sha = exists_and_sha(man["trajectory"], repo)
    mask_ok, mask_sha = exists_and_sha(man["contact_mask"], repo)
    cem_ok, cem_sha = exists_and_sha(man["result_npz"], repo)

    source_ready = scene_ok and traj_ok and cem_ok
    row: dict[str, str] = {name: "" for name in FULL_FIELDS}
    row.update({
        "case_id": case_pn,
        "object_key": obj,
        "object_name": info["object_name"],
        "date": info["date"],
        "seq": info["seq"],
        "person": person,
        "person_idx": PERSON_IDX.get(person, ""),
        "retarget_variant_id": "omnirt_v2",
        "target_variant_id": "ref_fk",
        "hand_collision_variant_id": "rubber_hull",
        "source_exp_id": man.get("experiment", ""),
        "spider_method_id": ARM_METHOD[sel["arm"]],
        "handoff_decision": "HANDOFF_READY",
        "candidate_decision": "PASS",
        "target_gate_status": "pass",
        "visual_qc_status": "pass",
        "target_scene": man.get("target_scene", ""),
        "trajectory": man["trajectory"],
        "scene_act": man["scene_act"],
        "contact_mask": man["contact_mask"],
        "stage2b_target_task": man["target_task"],
        "stage2b_result_root": F.path_ref(ARM_MANIFEST[obj], repo),
        "stage2b_manifest_ref": F.path_ref(ARM_MANIFEST[obj], repo),
        "raw_contact_threshold_label": "3cm",
        "cem_status": "pass",
        "cem_run_id": f"{man.get('experiment', '')}_{man.get('arm', '')}_{variant}",
        "cem_result_npz": man["result_npz"],
        "cem_video": man.get("video", ""),
        "cem_metrics_ref": "",
        "downstream_decision": "",
        "downstream_failure_mode": "",
        "downstream_notes": f"aug={variant}; manual={sel['manual']}; manual_label={sel['manual_label']}",
        "scene_act_exists": str(scene_ok),
        "trajectory_exists": str(traj_ok),
        "contact_mask_exists": str(mask_ok),
        "cem_result_exists": str(cem_ok),
        "source_handoff_manifest": F.path_ref(ARM_MANIFEST[obj], repo),
        "source_cem_evidence": man.get("result_npz", ""),
        "schema_version": F.SCHEMA_VERSION,
        "updated_at": F.timestamp(),
        "trajectory_sha256": traj_sha,
        "scene_act_sha256": scene_sha,
        "contact_mask_sha256": mask_sha,
        "cem_result_sha256": cem_sha,
    })
    row["rl_export_decision"] = "RL_EXPORT_READY" if source_ready else "BLOCKED_MISSING_REQUIRED_FILE"
    row["skip_reason"] = "" if source_ready else ",".join(
        f for f, ok in (("scene_act", scene_ok), ("trajectory", traj_ok), ("cem_result_npz", cem_ok)) if not ok
    )

    # partner
    resolved = partner is not None and partner["partner_status"] in ("present", "generated")
    ptrim_ok = pomni_ok = ptw_ok = False
    ptrim_sha = pomni_sha = ptw_sha = ""
    if resolved:
        ptrim_ok, ptrim_sha = exists_and_sha(partner["partner_trimmed_npz"], repo)
        pomni_ok, pomni_sha = exists_and_sha(partner["partner_omniretarget_output_npz"], repo)
        ptw_ok, ptw_sha = exists_and_sha(partner["partner_trim_window_json"], repo)
    partner_ready = resolved and ptrim_ok
    if partner is not None:
        row.update({
            "partner_case_id": partner["partner_case_id"],
            "partner_person": partner["partner_person"],
            "partner_person_idx": partner["partner_person_idx"],
            "partner_retarget_variant_id": "omnirt_v2",
            "partner_target_variant_id": "ref_fk",
            "partner_generation_mode": partner["partner_generation_mode"],
            "partner_trimmed_npz": partner["partner_trimmed_npz"],
            "partner_trimmed_npz_sha256": ptrim_sha,
            "partner_omniretarget_output_npz": partner["partner_omniretarget_output_npz"],
            "partner_omniretarget_output_npz_sha256": pomni_sha,
            "partner_trim_window_json": partner["partner_trim_window_json"],
            "partner_trim_window_json_sha256": ptw_sha,
            "partner_manifest_ref": partner_index_ref,
            "partner_manifest_sha256": partner_index_sha,
        })
    if partner_ready:
        row["partner_status"] = "pass"
        row["pair_status"] = "PAIR_COMPLETE"
    elif resolved:
        row["partner_status"] = "missing_file"
        row["pair_status"] = "PARTNER_FILE_MISSING"
    else:
        row["partner_status"] = partner["partner_status"] if partner else "missing"
        row["pair_status"] = "PARTNER_PENDING"
    row["paired_rl_export_decision"] = (
        "RL_EXPORT_READY" if (source_ready and partner_ready) else "BLOCKED_PARTNER_OR_SOURCE"
    )
    return row


def normalize_orig_row(row: dict[str, str]) -> dict[str, str]:
    """Coerce a reused orig paired row to exactly the 63-col schema."""
    return {name: row.get(name, "") for name in FULL_FIELDS}


def build_master(
    superset_sel: list[dict[str, str]],
    orig_idx: dict[tuple[str, str], dict[str, str]],
    aug_idx: dict[tuple[str, str, str], dict[str, str]],
    partner_idx: dict[tuple[str, str, str], dict[str, str]],
    repo: Path,
    partner_index_path: Path,
) -> dict[tuple[str, str, str], dict[str, str]]:
    """Build every paired row for the superset selection, keyed by (obj, case_pn, variant)."""
    partner_ref = F.path_ref(partner_index_path, repo)
    partner_sha = F.sha256(partner_index_path)
    master: dict[tuple[str, str, str], dict[str, str]] = {}
    for sel in superset_sel:
        obj, case_pn, variant = sel["object"], norm_pn(sel["case_id"]), sel["variant"]
        if sel["orig_or_aug"] == "orig":
            src = orig_idx.get((obj, case_pn))
            if src is None:
                raise SystemExit(f"orig case not in source paired manifest: {obj} {case_pn}")
            row = normalize_orig_row(src)
        else:
            man = aug_idx.get((obj, case_pn, variant))
            if man is None:
                raise SystemExit(f"aug case not in priority manifest: {obj} {case_pn} {variant}")
            partner = partner_idx.get((obj, case_pn, variant))
            row = build_aug_row(sel, man, partner, repo, partner_ref, partner_sha)
        master[(obj, case_pn, variant)] = remap_row_paths(row)
    return master


def summarize(rows: list[dict[str, str]]) -> dict[str, object]:
    """Compact counts for the version summary."""
    return {
        "rows": len(rows),
        "orig": sum(1 for r in rows if not r["spider_method_id"].endswith("_aug")),
        "aug": sum(1 for r in rows if r["spider_method_id"].endswith("_aug")),
        "paired_rl_export_decision": dict(Counter(r["paired_rl_export_decision"] for r in rows)),
        "pair_status": dict(Counter(r["pair_status"] for r in rows)),
        "by_object": dict(Counter(r["object_key"] for r in rows)),
    }


def write_version(out_dir: Path, rows: list[dict[str, str]], repo: Path) -> None:
    """Write the paired TSV/JSON + summary + audit for one version."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda r: (r["object_key"], r["case_id"], r["retarget_variant_id"]))
    F.write_tsv(out_dir / "paired_rl_export_input.tsv", ordered, FULL_FIELDS)
    F.write_json(out_dir / "paired_rl_export_input.json", ordered)
    summary = summarize(ordered)
    summary["out_dir"] = F.path_ref(out_dir, repo)
    summary["created_at"] = F.timestamp()
    F.write_json(out_dir / "rl_export_summary.json", summary)
    lines = [
        f"# E200 RL export — {out_dir.name}",
        "",
        f"- rows: `{summary['rows']}` (orig `{summary['orig']}` / aug `{summary['aug']}`)",
        f"- paired decisions: `{summary['paired_rl_export_decision']}`",
        f"- pair status: `{summary['pair_status']}`",
        f"- by object: `{summary['by_object']}`",
    ]
    (out_dir / "rl_export_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    """Build and write the three RL-export versions."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--rl-export-root", type=Path, required=True)
    parser.add_argument("--partner-index", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo.expanduser().resolve()
    root = args.rl_export_root.expanduser()
    partner_index_path = args.partner_index.expanduser()

    orig_idx = load_orig_index()
    aug_idx = load_aug_index()
    partner_idx = {
        (r["object_key"], r["source_case_id"], r["source_variant"]): r
        for r in read_tsv_rows(partner_index_path)
    }
    superset_sel = read_tsv_rows(root / "5object_all" / "selection.tsv")
    master = build_master(superset_sel, orig_idx, aug_idx, partner_idx, repo, partner_index_path)

    for version in ("box001_only", "5object_all", "5object_box001clean"):
        sel = read_tsv_rows(root / version / "selection.tsv")
        keys = [(r["object"], norm_pn(r["case_id"]), r["variant"]) for r in sel]
        rows = [master[k] for k in keys]
        if len(rows) != len(sel):
            raise SystemExit(f"[{version}] row/selection count mismatch")
        write_version(root / version, rows, repo)
        s = summarize(rows)
        print(f"[{version}] {s['rows']} rows  paired={s['paired_rl_export_decision']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
