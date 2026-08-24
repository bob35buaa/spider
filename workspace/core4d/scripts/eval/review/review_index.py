#!/usr/bin/env python3
"""Unified case index + annotation store for the PRG review player (E170-E194).

Pure-python (no viser / no spider imports) so it is importable headless for the
`--check` self-test. Reads each experiment's ``*_case_metrics.tsv`` by column
name (E170 is a 281-col superset of the shared 218; E172's file is mislabeled
``e171_``), normalizes artifact paths back to the repo root, and reads/writes a
non-destructive ``user_manual_review_filled.tsv`` per experiment (never touches
the ``..._template.tsv``).
"""

from __future__ import annotations

import csv
import datetime as _dt
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

# repo root: .../spider/workspace/core4d/scripts/eval/review/review_index.py
REPO = Path(__file__).resolve().parents[5]
DEFAULT_EXPS = (
    "E170", "E171", "E172", "E173", "E174", "E178", "E187", "E188", "E189", "E192", "E194",
)
# E198 is the G1xA2 2x2 factorial (4 arms x 59 cases). Opt-in via --exps E198.
SOURCE_OVERRIDES = {
    # Alias for the older E194 four-arm sweep under eval/full. Keep E194 itself
    # bound to the canonical 72-case G1 expansion review set.
    "E194_FULL": {
        "result_exp": "E194",
        "eval_subdir": "full",
        "case_metrics": "e194_arm_case_metrics_G1.tsv",
        "arm_sweep": True,
    },
    "E192": {
        "eval_subdir": "full",
        "case_metrics": "e192_case_metrics.tsv",
        "summary": "e192_eval_summary.json",
        "arm": "A2",
    },
    "E194": {
        "eval_subdir": "full_g1_expansion",
        "case_metrics": "e194_g1_expansion_case_metrics.tsv",
        "arm": "G1",
        "threshold_exp": "E173",
    },
    # E198 G1xA2 factorial: keep all four arms (A0/G1/A2/G1+A2) as an arm sweep,
    # so the player shows every 2x2 cell per case. Live qpos playback (no MP4).
    "E198": {
        "eval_subdir": "full_factorial",
        "case_metrics": "e198_arm_cache.tsv",
        "arm_sweep": True,
        "threshold_exp": "E194",
    },
    # E199 full-scale box translation augmentation (plan229/log287): orig +
    # trans0/1/2 shown as a 4-arm sweep per case. Live qpos playback (CEM ran
    # save_video=false). Aug metrics/paths come from the fullscale eval
    # case_metrics joined with the fullscale priority manifest by
    # (case_id, aug_variant); orig replay paths are rebuilt from the A0 row.
    # Opt-in via --exps E199.
    "E199": {
        "eval_subdir": "fullscale_augmentation",
        "case_metrics": "e199_fullscale_case_metrics.tsv",
        "arm_sweep": True,
        "threshold_exp": "E194",
    },
    # E199P = the E199 pilot (plan228/log286): 8 objects (5 box + bucket003/004/007),
    # 31 rows = orig + trans0/1/2 per case, shown as an arm sweep. This is the ONLY
    # E199 source that carries the bucket cases (the fullscale放量 is box-only).
    # Same replay/join mechanics as E199, but points at the pilot eval + manifest and
    # keeps orig as a 4th arm. Opt-in via --exps E199P.
    "E199P": {
        "result_exp": "E199",
        "eval_subdir": "full_augmentation",
        "case_metrics": "e199_aug_case_metrics.tsv",
        "arm_sweep": True,
        "threshold_exp": "E194",
    },
    # E200 = E199 translation aug replayed under two more downstream arms (plan230/
    # log289). Same replay/join mechanics as E199 fullscale (orig + trans0/1/2 arm
    # sweep, live qpos playback). Two arms, opt-in separately:
    #   E200N = noPRG (E167A rubber-hull, no PRG pairs). aug(249) + orig(37, E190).
    #   E200G = PRG+G1+A2 (gravcomp + hand-gate). aug(249); orig via --exps E198.
    "E200N": {
        "result_exp": "E200",
        "eval_subdir": "noprg",
        "case_metrics": "e200_noprg_case_metrics.tsv",
        "arm_sweep": True,
        "threshold_exp": "E194",
    },
    "E200G": {
        "result_exp": "E200",
        "eval_subdir": "prg_g1a2",
        "case_metrics": "e200_prg_g1a2_case_metrics.tsv",
        "arm_sweep": True,
        "threshold_exp": "E194",
    },
    # E203 P1 = CORE4D v2 human-motion orig retarget + CEM (box001/box024),
    # evaluated by run_E203_p1_eval.py. Single-arm live-qpos playback (the CEM
    # outdir trajectory_mjwp_act.npz holds the executed qpos + reference). The
    # eval output lives at s6_downstream/eval_P1 (sibling of eval/), so the
    # subdir hops up one level from the eval/ base. Opt-in via --exps E203.
    "E203": {
        "eval_subdir": "../eval_P1",
        "case_metrics": "e203_case_metrics.tsv",
        "threshold_exp": "E173",
    },
}

E194_CORRECTED_EVAL = (
    REPO / "workspace/core4d/results/E196/s6_downstream/eval/full_reference_fix"
)
E194_CORRECTED_RENDER = (
    REPO / "workspace/core4d/results/E196/s6_downstream/render/full_reference_fix"
)

GATE_FIELDS = (
    "fall_gate_pass",
    "body_z_gate_pass",
    "contact_gate_pass",
    "release_gate_pass",
    "hand_penetration_gate_pass",
    "lower_body_gate_pass",
    "root_pos_gate_pass",
    "root_ori_gate_pass",
    "hand_pos_gate_pass",
    "hand_ori_gate_pass",
    "object_pos_gate_pass",
    "object_ori_gate_pass",
)
# continuous metric columns shown in the top metrics bar (raw TSV column names)
METRIC_COLUMNS = (
    "body_z_err_p95_m",
    "leg_penetration_frac",
    "hand_object_physics_penetration_3mm_frame_frac",
    "hand_object_release_false_contact_3mm_frac",
    "hand_object_physics_contact_3mm_in_mask_frac",
    "fall_flag",
    # tracking errors (numeric gates for E178; display-only for legacy rows)
    "track_root_pos_err_cm_mean",
    "track_root_ori_err_deg_mean",
    "track_eef_pos_err_cm_mean",
    "track_eef_ori_err_deg_mean",
    "track_obj_pos_err_cm_mean",
    "track_obj_ori_err_deg_mean",
    "foot_slip_max_m",
)
# the 9-column schema shared by every user_manual_review_template.tsv
REVIEW_FIELDS = (
    "case_id",
    "user_manual_review_status",
    "manual_use_decision",
    "manual_quality_label",
    "manual_failure_taxonomy",
    "manual_review_note",
    "manual_reviewer",
    "manual_reviewed_at",
    "paired_video",
)


def _as_bool(raw: str) -> bool | None:
    text = str(raw).strip().lower()
    if text in ("true", "1", "yes"):
        return True
    if text in ("false", "0", "no"):
        return False
    return None


def _as_float(raw: str) -> float | None:
    text = str(raw).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def normalize_path(raw: str) -> str:
    """Re-root any foreign/absolute artifact path onto the current repo.

    Same rule as E168 ``repo_path``: split on the first known marker segment.
    """
    text = str(raw).strip()
    if not text:
        return ""
    # tidal relocation: results moved under .../spider_workdirs/core4d/results/results/
    # (workspace/core4d/results is a symlink to that dir), so collapse the double
    # results/ before the single-results legacy rule below.
    if "spider_workdirs/core4d/results/results/" in text:
        return str(
            REPO
            / "workspace/core4d/results"
            / text.split("spider_workdirs/core4d/results/results/", 1)[1]
        )
    # legacy foreign mount: /mnt/<uuid>/spider_workdirs/core4d/results/... -> workspace/core4d/...
    if "spider_workdirs/" in text:
        return str(REPO / "workspace" / text.split("spider_workdirs/", 1)[1])
    for marker in ("example_datasets/", "workspace/", "logs/"):
        if marker in text:
            return str(REPO / (marker + text.split(marker, 1)[1]))
    p = Path(text)
    return text if p.is_absolute() else str(REPO / text)


def resolve_scene(exp_id: str, case_id: str, scene_xml: str) -> str:
    """Live scene XML if present, else fall back to the experiment's scene_snapshot.

    E170's live scene dir was overwritten by later experiments; its exact scene
    XML survives only under ``results/{exp}/scene_snapshot/{case_id}/``.
    """
    p = Path(scene_xml)
    if p.is_file():
        return str(p)
    snap = REPO / "workspace/core4d/results" / exp_id / "scene_snapshot"
    if not snap.is_dir() or not p.name:
        return scene_xml
    cands = list(snap.rglob(p.name))
    for c in cands:
        if case_id in str(c):
            return str(c)
    return str(cands[0]) if cands else scene_xml


@dataclass
class CaseRecord:
    exp_id: str
    arm: str
    case_id: str
    variant: str
    object_key: str
    retarget_variant_id: str
    numeric_release_pass: bool | None
    numeric_failure_modes: list[str]
    gates: dict[str, bool | None]
    status: str
    outdir_npz: str
    scene_xml: str
    config_act: str
    trajectory: str
    video: str
    # annotation (merged from filled TSV; blank until reviewed)
    annotation: dict[str, str] = field(default_factory=dict)
    # continuous numeric metrics for the top bar
    metrics: dict[str, float | None] = field(default_factory=dict)
    # kinematic playback: replay outdir_npz's full qpos directly against scene_xml
    # (no CEM config / physics rollout). Used by E197 RL-export motions.
    kinematic: bool = False

    @property
    def key(self) -> str:
        # Arm-inclusive so arm-sweep records (multiple arms share a case_id) get
        # distinct cache / lookup keys — otherwise one arm's geometry would be
        # served for another arm of the same case.
        return f"{self.exp_id}/{self.case_id}/{self.arm}"

    @property
    def ann_id(self) -> str:
        # Annotation identity within a per-exp filled TSV: arm-namespaced so each
        # arm of an arm-sweep case is annotated independently.
        return _ann_id(self.case_id, self.arm)

    @property
    def reviewed(self) -> bool:
        return (self.annotation.get("user_manual_review_status") or "") == "reviewed"

    @property
    def playable(self) -> bool:
        return Path(self.outdir_npz).is_file() and Path(self.scene_xml).is_file()


def eval_dir(exp: str) -> Path:
    subdir = SOURCE_OVERRIDES.get(exp, {}).get("eval_subdir", "full")
    result_exp = SOURCE_OVERRIDES.get(exp, {}).get("result_exp", exp)
    base = REPO / "workspace/core4d/results" / result_exp / "s6_downstream/eval"
    # normpath collapses a "../eval_P1"-style subdir lexically (the eval/ base may
    # not exist on disk for experiments that write a sibling eval dir); a plain
    # subdir is unchanged.
    return Path(os.path.normpath(base / subdir))


def _case_metrics_path(exp: str) -> Path | None:
    filename = SOURCE_OVERRIDES.get(exp, {}).get("case_metrics")
    if filename:
        path = eval_dir(exp) / filename
        return path if path.is_file() else None
    hits = sorted(eval_dir(exp).glob("*_case_metrics.tsv"))
    return hits[0] if hits else None


def filled_path(exp: str) -> Path:
    return eval_dir(exp) / "user_manual_review_filled.tsv"


def load_thresholds(exp: str) -> dict[str, float]:
    """Numeric thresholds from the experiment's summary.json (metric standard)."""
    import json

    summ = eval_dir(exp) / SOURCE_OVERRIDES.get(exp, {}).get("summary", "summary.json")
    thresholds = {}
    if summ.is_file():
        thresholds = json.loads(summ.read_text(encoding="utf-8")).get("thresholds", {})
    fallback_exp = SOURCE_OVERRIDES.get(exp, {}).get("threshold_exp")
    if not thresholds and fallback_exp:
        fallback = eval_dir(fallback_exp) / "summary.json"
        if fallback.is_file():
            thresholds = json.loads(fallback.read_text(encoding="utf-8")).get("thresholds", {})
    if exp == "E192" and not thresholds:
        # E192's evaluator stores the frozen gate contract in the source code
        # rather than repeating a summary threshold object.
        thresholds = {
            "body_z_err_p95_m_max": 0.20,
            "leg_penetration_max": 0.10,
            "hand_penetration_3mm_max": 0.30,
            "release_false_3mm_max": 0.30,
            "raw_contact_min": 0.50,
        }
    return thresholds


def _ann_id(case_id: str, arm: str) -> str:
    """Arm-namespaced annotation id. Plain case_id when there is no arm (keeps
    single-arm experiments' filled TSVs backward compatible); ``case_id#arm`` for
    arm-sweep records so each arm is annotated independently."""
    arm = (arm or "").strip()
    return f"{case_id}#{arm}" if arm else case_id


def load_annotations(exp: str) -> dict[str, dict[str, str]]:
    """ann_id -> annotation row (from the non-destructive filled TSV).

    The ``case_id`` column stores the arm-namespaced ``ann_id`` (see `_ann_id`)."""
    path = filled_path(exp)
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        return {r["case_id"]: dict(r) for r in reader if r.get("case_id")}


def save_annotation(exp: str, ann_id: str, values: dict[str, str]) -> Path:
    """Upsert one arm's annotation into the filled TSV (keyed by ann_id). Atomic."""
    rows = load_annotations(exp)
    row = {k: "" for k in REVIEW_FIELDS}
    row.update(rows.get(ann_id, {}))
    row.update(values)
    row["case_id"] = ann_id
    row["user_manual_review_status"] = "reviewed"
    row["manual_reviewed_at"] = (
        _dt.datetime.now().astimezone().isoformat(timespec="seconds")
    )
    rows[ann_id] = row

    path = filled_path(exp)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tsv.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=list(REVIEW_FIELDS), delimiter="\t", extrasaction="ignore"
        )
        writer.writeheader()
        for cid in sorted(rows):
            out = {k: rows[cid].get(k, "") for k in REVIEW_FIELDS}
            out["case_id"] = cid
            writer.writerow(out)
    os.replace(tmp, path)
    return path


def _read_exp(exp: str) -> list[CaseRecord]:
    path = _case_metrics_path(exp)
    if path is None:
        return []
    anns = load_annotations(exp)
    required_arm = SOURCE_OVERRIDES.get(exp, {}).get("arm")
    records: list[CaseRecord] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            arm = (row.get("arm") or "").strip()
            if required_arm and arm != required_arm:
                continue
            case_id = (row.get("case_id") or "").strip()
            if not case_id:
                continue
            modes = (row.get("numeric_failure_modes") or "").replace(";", ",")
            outdir_npz = normalize_path(row.get("outdir_npz", ""))
            config_act = normalize_path(row.get("config_act", ""))
            if not config_act:
                # E192 evaluator metrics intentionally omit config/video columns;
                # the canonical CEM outdir always carries config_act.yaml.
                candidate = Path(outdir_npz).parent / "config_act.yaml"
                config_act = str(candidate) if candidate.is_file() else ""
            video = normalize_path(row.get("video", ""))
            if not video and exp == "E192":
                candidate = REPO / "workspace/core4d/results/E192/s6_downstream/render/full" / f"E192_{case_id}_A2_full.mp4"
                video = str(candidate) if candidate.is_file() else ""
            scene_xml = resolve_scene(
                exp, case_id, normalize_path(row.get("scene_xml", ""))
            )
            records.append(
                CaseRecord(
                    exp_id=exp,
                    arm=arm,
                    case_id=case_id,
                    variant=(row.get("variant") or "").strip(),
                    object_key=(row.get("object_key") or "").strip(),
                    retarget_variant_id=(row.get("retarget_variant_id") or "").strip(),
                    numeric_release_pass=_as_bool(
                        row.get(
                            "numeric_release_pass_12gate"
                            if exp == "E192"
                            else "numeric_release_pass",
                            row.get("numeric_release_pass", ""),
                        )
                    ),
                    numeric_failure_modes=[
                        m.strip() for m in modes.split(",") if m.strip()
                    ],
                    gates={g: _as_bool(row.get(g, "")) for g in GATE_FIELDS},
                    status=(row.get("status") or (f"{arm}_FULL_COMPLETE" if arm else "")).strip(),
                    outdir_npz=outdir_npz,
                    scene_xml=scene_xml,
                    config_act=config_act,
                    trajectory=normalize_path(row.get("trajectory", "")),
                    video=video,
                    annotation=anns.get(_ann_id(case_id, arm), {}),
                    metrics={c: _as_float(row.get(c, "")) for c in METRIC_COLUMNS},
                )
            )
    return records


def _read_e194_corrected_overlay() -> list[CaseRecord]:
    """Read the E194 72-case G1 review set with the E196 fix overlaid.

    Only the 29 Euler-mismatch cases are taken from E196 corrected output;
    the remaining 43 G1 rows remain the original E194 expansion results.
    """
    import csv

    base_path = REPO / "workspace/core4d/results/E194/s6_downstream/eval/full_g1_expansion/e194_g1_expansion_case_metrics.tsv"
    corrected_path = E194_CORRECTED_EVAL / "e196_reference_fix_case_metrics.tsv"
    if not base_path.is_file() or not corrected_path.is_file():
        return _read_exp("E194")
    annotations = load_annotations("E194")
    # E196 is a new corrected rollout. Do not carry over human decisions made
    # on the contaminated E194 videos for exactly these 29 case IDs. The
    # source TSV is cleaned once below; this filter is also fail-safe for an
    # older checkout or a stale annotation file.
    with base_path.open("r", encoding="utf-8", newline="") as fh:
        base_rows = [r for r in csv.DictReader(fh, delimiter="\t") if r.get("arm") == "G1"]
    with corrected_path.open("r", encoding="utf-8", newline="") as fh:
        fixed_rows = [r for r in csv.DictReader(fh, delimiter="\t") if r.get("arm") == "G1_corrected"]
    fixed = {r["case_id"]: r for r in fixed_rows}
    if len(base_rows) != 72 or len(fixed) != 29:
        raise ValueError(f"E194 review overlay cardinality: base={len(base_rows)} corrected={len(fixed)}")
    rows = [fixed.get(r["case_id"], r) for r in base_rows]
    records: list[CaseRecord] = []
    for row in rows:
        case_id = row["case_id"]
        corrected = case_id in fixed
        video = (
            E194_CORRECTED_RENDER / f"E196_{case_id}_G1_reference_fix.mp4"
            if corrected else
            REPO / "workspace/core4d/results/E194/s6_downstream/render/full_g1_expansion" / f"E194_{case_id}_G1_expansion.mp4"
        )
        modes = (row.get("numeric_failure_modes") or "").replace(";", ",")
        records.append(CaseRecord(
            exp_id="E194", arm="G1", case_id=case_id,
            variant=(row.get("variant") or "").strip(),
            object_key=(row.get("object_key") or "").strip(),
            retarget_variant_id=(row.get("retarget_variant_id") or "").strip(),
            # E194's public strict review gate is the frozen 12-gate field;
            # `numeric_release_pass` may additionally include legacy health.
            numeric_release_pass=_as_bool(row.get("numeric_release_pass_12gate", row.get("numeric_release_pass", ""))),
            numeric_failure_modes=[m.strip() for m in modes.split(",") if m.strip()],
            gates={g: _as_bool(row.get(g, "")) for g in GATE_FIELDS},
            status=(row.get("status") or "G1_FULL_COMPLETE").strip(),
            outdir_npz=normalize_path(row.get("outdir_npz", "")),
            scene_xml=resolve_scene("E196" if corrected else "E194", case_id, normalize_path(row.get("scene_xml", ""))),
            config_act=normalize_path(row.get("config_act", "")),
            trajectory=normalize_path(row.get("trajectory", "")),
            video=str(video),
            annotation={} if corrected else annotations.get(_ann_id(case_id, "G1"), {}),
            metrics={c: _as_float(row.get(c, "")) for c in METRIC_COLUMNS},
        ))
    return records


E199_FS_MANIFEST = (
    REPO / "workspace/core4d/results/E199/s6_downstream/manifests/e199_fullscale_priority_manifest.tsv"
)


def _e199_person_cid(case_id: str) -> str:
    """Normalize an orig `_p1/_p2` case id to the aug `_person1/_person2` form so
    orig groups with its trans0/1/2 siblings under one case in the arm sweep."""
    if case_id.endswith("_p1"):
        return case_id[:-3] + "_person1"
    if case_id.endswith("_p2"):
        return case_id[:-3] + "_person2"
    return case_id


def _read_e199_fullscale() -> list[CaseRecord]:
    """E199 full-scale augmentation review set: orig + trans0/1/2 per case.

    Aug rows join the fullscale eval case_metrics (metrics + 6 physics gates) with
    the fullscale priority manifest (replay paths) by (case_id, aug_variant); each
    translation variant is one arm. The reused A0/PRG `orig` rows are NOT in that
    manifest, so their replay paths (outdir/config/scene/ref) are reconstructed
    from the case row itself and orig is exposed as a 4th arm (case id normalized
    to the `_person` form so it groups with its trans siblings). Live qpos playback.
    """
    metrics_path = _case_metrics_path("E199")
    if metrics_path is None or not E199_FS_MANIFEST.is_file():
        return _read_exp("E199") if metrics_path else []
    manifest: dict[tuple[str, str], dict[str, str]] = {}
    with E199_FS_MANIFEST.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            manifest[(row.get("case_id", ""), row.get("aug_variant", ""))] = row
    anns = load_annotations("E199")
    records: list[CaseRecord] = []
    with metrics_path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            group = (row.get("group") or "").strip()
            case_id = (row.get("case_id") or "").strip()
            variant = (row.get("aug_variant") or "").strip()
            if not case_id:
                continue
            if group == "aug":
                mrow = manifest.get((case_id, variant))
                if mrow is None:
                    continue
                outdir_npz = normalize_path(mrow.get("outdir_npz", ""))
                config_act = normalize_path(mrow.get("config_act", ""))
                scene_src = mrow.get("scene_act", "") or row.get("scene_xml", "")
                trajectory = normalize_path(mrow.get("trajectory", ""))
                video = normalize_path(mrow.get("video", ""))
                disp_cid = case_id
                rvar, arm = "omnirt_v2", variant
            elif group == "orig":
                # orig has no manifest row: rebuild replay paths from the A0 row.
                outdir_npz = normalize_path(row.get("qpos_path", ""))
                config_act = ""
                scene_src = row.get("scene_xml", "")
                trajectory = ""
                video = ""
                disp_cid = _e199_person_cid(case_id)
                rvar, arm = "omnirt_v1", "orig"
            else:
                continue
            if not config_act and outdir_npz:
                candidate = Path(outdir_npz).parent / "config_act.yaml"
                config_act = str(candidate) if candidate.is_file() else ""
            scene_xml = resolve_scene("E199", disp_cid, normalize_path(scene_src))
            if group == "orig" and scene_xml:
                cand_traj = Path(scene_xml).parent / "0" / "trajectory_kinematic.npz"
                trajectory = str(cand_traj) if cand_traj.is_file() else ""
            if video and not Path(video).is_file():
                video = ""  # CEM ran save_video=false -> live qpos playback
            modes = (row.get("numeric_failure_modes") or "").replace(";", ",")
            records.append(
                CaseRecord(
                    exp_id="E199",
                    arm=arm,
                    case_id=disp_cid,
                    variant=variant or arm,
                    object_key=(row.get("object_key") or "").strip(),
                    retarget_variant_id=rvar,
                    numeric_release_pass=_as_bool(row.get("all_gates_pass", "")),
                    numeric_failure_modes=[m.strip() for m in modes.split(",") if m.strip()],
                    gates={g: _as_bool(row.get(g, "")) for g in GATE_FIELDS},
                    status=(row.get("status") or ("ORIG_A0_REUSED" if group == "orig" else "AUG_FULL_COMPLETE")).strip(),
                    outdir_npz=outdir_npz,
                    scene_xml=scene_xml,
                    config_act=config_act,
                    trajectory=trajectory,
                    video=video,
                    annotation=anns.get(_ann_id(disp_cid, arm), {}),
                    metrics={c: _as_float(row.get(c, "")) for c in METRIC_COLUMNS},
                )
            )
    return records


E199_PILOT_MANIFEST = (
    REPO / "workspace/core4d/results/E199/s6_downstream/manifests/e199_priority_full_manifest.tsv"
)


def _read_e199_pilot() -> list[CaseRecord]:
    """E199 pilot review set (plan228/log286): 8 objects, orig + trans0/1/2 per case.

    Same join mechanics as `_read_e199_fullscale` — the pilot eval case_metrics
    (metrics + 6 physics gates) joined with the pilot priority manifest by
    (case_id, aug_variant) — but this file has no `group` column, so ALL variants
    are kept (orig exposed as a 4th arm) and this is the only E199 source carrying
    the bucket003/004/007 cases. Live qpos playback (CEM ran save_video=false).
    """
    metrics_path = _case_metrics_path("E199P")
    if metrics_path is None or not E199_PILOT_MANIFEST.is_file():
        return _read_exp("E199P") if metrics_path else []
    manifest: dict[tuple[str, str], dict[str, str]] = {}
    with E199_PILOT_MANIFEST.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            manifest[(row.get("case_id", ""), row.get("aug_variant", ""))] = row
    anns = load_annotations("E199P")
    records: list[CaseRecord] = []
    with metrics_path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            case_id = (row.get("case_id") or "").strip()
            variant = (row.get("aug_variant") or "").strip()
            mrow = manifest.get((case_id, variant))
            if not case_id or mrow is None:
                continue
            outdir_npz = normalize_path(mrow.get("outdir_npz", ""))
            config_act = normalize_path(mrow.get("config_act", ""))
            if not config_act and outdir_npz:
                candidate = Path(outdir_npz).parent / "config_act.yaml"
                config_act = str(candidate) if candidate.is_file() else ""
            scene_xml = resolve_scene(
                "E199P", case_id,
                normalize_path(mrow.get("scene_act", "") or row.get("scene_xml", "")),
            )
            video = normalize_path(mrow.get("video", ""))
            if video and not Path(video).is_file():
                video = ""  # CEM ran save_video=false -> live qpos playback
            modes = (row.get("numeric_failure_modes") or "").replace(";", ",")
            records.append(
                CaseRecord(
                    exp_id="E199P",
                    arm=variant,
                    case_id=case_id,
                    variant=variant,
                    object_key=(row.get("object_key") or "").strip(),
                    retarget_variant_id="omnirt_v1" if variant == "orig" else "omnirt_v2",
                    numeric_release_pass=_as_bool(row.get("all_gates_pass", "")),
                    numeric_failure_modes=[m.strip() for m in modes.split(",") if m.strip()],
                    gates={g: _as_bool(row.get(g, "")) for g in GATE_FIELDS},
                    status=(row.get("status") or "AUG_PILOT_COMPLETE").strip(),
                    outdir_npz=outdir_npz,
                    scene_xml=scene_xml,
                    config_act=config_act,
                    trajectory=normalize_path(mrow.get("trajectory", "")),
                    video=video,
                    annotation=anns.get(_ann_id(case_id, variant), {}),
                    metrics={c: _as_float(row.get(c, "")) for c in METRIC_COLUMNS},
                )
            )
    return records


E200_MANIFEST = {
    "E200N": REPO / "workspace/core4d/results/E200/s6_downstream/manifests/e200_noprg_priority_manifest.tsv",
    "E200G": REPO / "workspace/core4d/results/E200/s6_downstream/manifests/e200_prg_g1a2_priority_manifest.tsv",
}
# G1A2 arm has no same-arm orig CEM run in E200 (the prg_g1a2 CEM re-ran aug only);
# its orig baseline is the E198 factorial G1A2 arm. Expose it inside E200G so the
# review player shows orig alongside its trans0/1/2 siblings (symmetric to E200N,
# whose orig comes from E190). Read-only: replay paths + metrics already live in
# the E198 arm_cache, so this needs no eval re-run.
E198_ARM_CACHE = REPO / "workspace/core4d/results/E198/s6_downstream/eval/full_factorial/e198_arm_cache.tsv"


def _read_e198_g1a2_orig(exp: str, anns: dict[str, dict[str, str]]) -> list[CaseRecord]:
    """G1A2 orig rollouts (E198 factorial arm=G1A2) as orig arm records for E200G."""
    if not E198_ARM_CACHE.is_file():
        return []
    records: list[CaseRecord] = []
    with E198_ARM_CACHE.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if (row.get("arm") or "").strip() != "G1A2":
                continue
            case_id = (row.get("case_id") or "").strip()
            if not case_id:
                continue
            disp_cid = _e199_person_cid(case_id)
            outdir_npz = normalize_path(row.get("qpos_path", ""))
            scene_xml = resolve_scene(exp, disp_cid, normalize_path(row.get("scene_xml", "")))
            trajectory = normalize_path(row.get("trajectory", ""))
            if (not trajectory) and scene_xml:
                cand = Path(scene_xml).parent / "0" / "trajectory_kinematic.npz"
                trajectory = str(cand) if cand.is_file() else ""
            config_act = normalize_path(row.get("config_act", ""))
            if not config_act and outdir_npz:
                cand = Path(outdir_npz).parent / "config_act.yaml"
                config_act = str(cand) if cand.is_file() else ""
            modes = (row.get("numeric_failure_modes") or "").replace(";", ",")
            records.append(
                CaseRecord(
                    exp_id=exp,
                    arm="orig",
                    case_id=disp_cid,
                    variant="orig",
                    object_key=(row.get("object_key") or "").strip(),
                    retarget_variant_id="omnirt_v1",
                    numeric_release_pass=_as_bool(row.get("numeric_release_pass", "")),
                    numeric_failure_modes=[m.strip() for m in modes.split(",") if m.strip()],
                    gates={g: _as_bool(row.get(g, "")) for g in GATE_FIELDS},
                    status="ORIG_E198_G1A2",
                    outdir_npz=outdir_npz,
                    scene_xml=scene_xml,
                    config_act=config_act,
                    trajectory=trajectory,
                    video="",
                    annotation=anns.get(_ann_id(disp_cid, "orig"), {}),
                    metrics={c: _as_float(row.get(c, "")) for c in METRIC_COLUMNS},
                )
            )
    return records


# E197 partner re-export v2 (plan?/log277): the RL-export set of 52 target cases,
# sourced mostly from E170/E172/E173 CEM + 4 E197_rerun. Each case ships a
# 43-dim g1_expansion qpos (root7 + G1 29dof + object7) — a full MuJoCo qpos we
# replay kinematically against the source case's nq=43 scene.xml. There is NO
# per-case numeric metrics file, so gates/metrics are blank (visual review only).
E197_REEXPORT = REPO / "workspace/core4d/results/E197/s6_downstream/rl_export/partner_reexport_v2"
_HUMANOID_OBJECT = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"


def _read_e197_reexport() -> list[CaseRecord]:
    """E197 partner_reexport_v2 target motions as kinematic-playback records."""
    pairing = E197_REEXPORT / "manifests/source_pairing_manifest.tsv"
    prov_path = E197_REEXPORT / "manifests/provenance.tsv"
    if not pairing.is_file():
        return []
    prov: dict[str, dict[str, str]] = {}
    if prov_path.is_file():
        with prov_path.open("r", encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                prov[r.get("case_id", "")] = r
    anns = load_annotations("E197")
    records: list[CaseRecord] = []
    with pairing.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            case_id = (row.get("case_id") or "").strip()
            if not case_id:
                continue
            pr = prov.get(case_id, {})
            variant = (pr.get("source_variant") or row.get("upstream_version") or "").strip() or "omnirt_v1"
            outdir_npz = normalize_path(row.get("target_g1_expansion_npz", ""))
            # scene: the source case's nq=43 scene.xml (object pose is driven by qpos,
            # so any G1+object scene for this case replays correctly).
            slug = (pr.get("source_slug") or "").strip()
            scene_xml = str(_HUMANOID_OBJECT / slug / "scene.xml") if slug else ""
            records.append(
                CaseRecord(
                    exp_id="E197",
                    arm=variant,
                    case_id=case_id,
                    variant=variant,
                    object_key=(row.get("object_name") or "").strip().lower(),
                    retarget_variant_id=variant,
                    numeric_release_pass=None,
                    numeric_failure_modes=[],
                    gates={g: None for g in GATE_FIELDS},
                    status=f"E197_REEXPORT_V2:{pr.get('source_exp', '')}",
                    outdir_npz=outdir_npz,
                    scene_xml=scene_xml,
                    config_act="",
                    trajectory="",
                    video="",
                    annotation=anns.get(_ann_id(case_id, variant), {}),
                    metrics={c: None for c in METRIC_COLUMNS},
                    kinematic=True,
                )
            )
    return records


def _read_e200_arm(exp: str) -> list[CaseRecord]:
    """E200 arm review set (plan230/log289): orig + trans0/1/2 per case, one arm each.

    Same join mechanics as `_read_e199_fullscale`: aug rows join the arm's priority
    manifest (replay paths) by (case_id, aug_variant); orig rows (present only for
    the noPRG arm, from the reused E190 baseline) rebuild their replay paths from
    the case row. Live qpos playback (CEM ran save_video=false)."""
    metrics_path = _case_metrics_path(exp)
    man_path = E200_MANIFEST.get(exp)
    if metrics_path is None or man_path is None:
        return []
    manifest: dict[tuple[str, str], dict[str, str]] = {}
    if man_path.is_file():
        with man_path.open("r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                manifest[(row.get("case_id", ""), row.get("aug_variant", ""))] = row
    anns = load_annotations(exp)
    records: list[CaseRecord] = []
    with metrics_path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            group = (row.get("group") or "").strip()
            case_id = (row.get("case_id") or "").strip()
            variant = (row.get("aug_variant") or "").strip()
            if not case_id:
                continue
            if group == "aug":
                mrow = manifest.get((case_id, variant))
                if mrow is None:
                    continue
                outdir_npz = normalize_path(mrow.get("outdir_npz", ""))
                config_act = normalize_path(mrow.get("config_act", ""))
                scene_src = mrow.get("scene_act", "") or row.get("scene_xml", "")
                trajectory = normalize_path(mrow.get("trajectory", ""))
                video = normalize_path(mrow.get("video", ""))
                disp_cid = case_id
                rvar, arm = "omnirt_v2", variant
            elif group == "orig":
                outdir_npz = normalize_path(row.get("qpos_path", ""))
                config_act = ""
                scene_src = row.get("scene_xml", "")
                trajectory = ""
                video = ""
                disp_cid = _e199_person_cid(case_id)
                rvar, arm = "omnirt_v1", "orig"
            else:
                continue
            if not config_act and outdir_npz:
                # E190/E168 orig qpos npz lives flat in cem/full/<stem>.npz, and its
                # config sits in the sibling <stem>_outdir_full/config_act.yaml — not
                # next to the npz. Try that first, then the adjacent-file layout.
                p = Path(outdir_npz)
                for candidate in (p.with_name(p.stem + "_outdir_full") / "config_act.yaml",
                                  p.parent / "config_act.yaml"):
                    if candidate.is_file():
                        config_act = str(candidate)
                        break
            scene_xml = resolve_scene(exp, disp_cid, normalize_path(scene_src))
            if group == "orig" and scene_xml:
                cand_traj = Path(scene_xml).parent / "0" / "trajectory_kinematic.npz"
                trajectory = str(cand_traj) if cand_traj.is_file() else ""
            if video and not Path(video).is_file():
                video = ""
            modes = (row.get("numeric_failure_modes") or "").replace(";", ",")
            records.append(
                CaseRecord(
                    exp_id=exp,
                    arm=arm,
                    case_id=disp_cid,
                    variant=variant or arm,
                    object_key=(row.get("object_key") or "").strip(),
                    retarget_variant_id=rvar,
                    numeric_release_pass=_as_bool(row.get("all_gates_pass", "")),
                    numeric_failure_modes=[m.strip() for m in modes.split(",") if m.strip()],
                    gates={g: _as_bool(row.get(g, "")) for g in GATE_FIELDS},
                    status=(row.get("status") or ("ORIG_REUSED" if group == "orig" else "AUG_FULL_COMPLETE")).strip(),
                    outdir_npz=outdir_npz,
                    scene_xml=scene_xml,
                    config_act=config_act,
                    trajectory=trajectory,
                    video=video,
                    annotation=anns.get(_ann_id(disp_cid, arm), {}),
                    metrics={c: _as_float(row.get(c, "")) for c in METRIC_COLUMNS},
                )
            )
    # G1A2 arm's orig baseline lives in E198, not in this arm's metrics -> inject it.
    if exp == "E200G":
        records.extend(_read_e198_g1a2_orig(exp, anns))
    return records


def build_index(exps: tuple[str, ...] = DEFAULT_EXPS) -> list[CaseRecord]:
    out: list[CaseRecord] = []
    for exp in exps:
        if exp == "E194":
            out.extend(_read_e194_corrected_overlay())
        elif exp == "E199":
            out.extend(_read_e199_fullscale())
        elif exp == "E199P":
            out.extend(_read_e199_pilot())
        elif exp == "E197":
            out.extend(_read_e197_reexport())
        elif exp in ("E200N", "E200G"):
            out.extend(_read_e200_arm(exp))
        else:
            out.extend(_read_exp(exp))
    return out


# --- filter helpers used by the viser app -----------------------------------
def objects_for(records: list[CaseRecord]) -> list[str]:
    return sorted({r.object_key for r in records if r.object_key})


def failure_modes_for(records: list[CaseRecord]) -> list[str]:
    modes: set[str] = set()
    for r in records:
        modes.update(r.numeric_failure_modes)
    return sorted(modes)


def variants_for(records: list[CaseRecord]) -> list[str]:
    return sorted({r.retarget_variant_id for r in records if r.retarget_variant_id})


def _check(exps: tuple[str, ...] = DEFAULT_EXPS) -> int:
    """Print per-exp counts and cross-check against summary.json; exit non-zero on mismatch."""
    import json

    records = build_index(exps)
    ok = True
    print(
        f"{'exp':6} {'indexed':>7} {'evaluated':>9} {'npass':>5} {'reviewed':>8} {'playable':>8}"
    )
    for exp in exps:
        recs = [r for r in records if r.exp_id == exp]
        summ = eval_dir(exp) / SOURCE_OVERRIDES.get(exp, {}).get("summary", "summary.json")
        evaluated = npass = -1
        if summ.is_file():
            summary = json.loads(summ.read_text(encoding="utf-8"))
            counts = summary.get("counts", {})
            evaluated = int(counts.get("evaluated", summary.get("evaluated", -1)))
            npass = int(counts.get("numeric_pass", -1))
            if exp == "E192":
                evaluated = int(summary.get("evaluated", -1))
                npass = sum(1 for r in recs if r.numeric_release_pass)
            if exp in ("E199", "E199P", "E200N", "E200G"):
                # E199/E200 arm-sweep summaries report aug_scored/scored (not
                # `evaluated`); they index orig + trans as arms, so cross-check the
                # review set against the index itself rather than the summary count.
                evaluated = len(recs)
                npass = sum(1 for r in recs if r.numeric_release_pass)
        elif exp == "E197":
            # E197 is kinematic-playback only: no summary.json / no numeric gates.
            evaluated = len(recs)
            npass = sum(1 for r in recs if r.numeric_release_pass)
        elif exp == "E194_FULL":
            evaluated = len(recs)
            npass = sum(1 for r in recs if r.numeric_release_pass)
            if exp == "E194_FULL":
                evaluated = len(recs)
                npass = sum(1 for r in recs if r.numeric_release_pass)
            if exp == "E194":
                evaluated = int(summary.get("g1_scored", -1))
                # The review player uses the E196-corrected 29-row overlay,
                # so its expected strict count must be derived from the
                # overlaid records rather than the historical E194 summary.
                npass = sum(1 for r in recs if r.numeric_release_pass)
        idx_pass = sum(1 for r in recs if r.numeric_release_pass)
        reviewed = sum(1 for r in recs if r.reviewed)
        playable = sum(1 for r in recs if r.playable)
        flag = ""
        if len(recs) != evaluated or idx_pass != npass:
            ok = False
            flag = "  <-- MISMATCH"
        print(
            f"{exp:6} {len(recs):7d} {evaluated:9d} {idx_pass:5d} {reviewed:8d} "
            f"{playable:8d}{flag}"
        )
    print(
        f"total indexed: {len(records)}  playable: {sum(1 for r in records if r.playable)}"
    )
    print(f"objects: {objects_for(records)}")
    print(f"failure modes: {failure_modes_for(records)}")
    print(f"variants: {variants_for(records)}")
    print(f"REPO={REPO}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_check())
