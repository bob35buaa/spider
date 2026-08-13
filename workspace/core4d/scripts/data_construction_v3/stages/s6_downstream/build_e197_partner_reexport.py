#!/usr/bin/env python3
"""Assemble the E197 OmniRetarget partner re-export for SUGAR R018-23.

Downstream (R018-23) needs, per gate-pass case, the full OmniRetarget
``s3_retarget`` middle-products that ``core4d_partner.bundle`` consumes:
per person ``retargeted``/``trimmed`` ``*_with_obj_original.npz`` (qpos(T,43)
f64 + human_joints(T,22,3) f32 + fps + cost), ``trim_window.json`` and a
scene-level ``raw_contact_mask_3cm.npz``.  These were never in E197's slim
``s6/rl_export`` product, but the native, self-consistent artifacts still
exist in the source retarget trees (E168/E170/E172/E173) plus the E197
``partner_omnirt_rerun`` tree for four partners that the main run skipped.

This script is a pure assembler: it locates the source artifacts for all 70
unique person-cases (52 gate-pass targets + their partners), validates every
one against the bundle's gates (schema / trim proof / contact axes / frame
counts), and only then mirrors them into a clean E197 delivery tree with
rewritten stable ``trim_window`` paths, SHA256 sidecars and a 52-row
target<->partner pairing manifest.  It is fail-closed: any case that fails
validation aborts the whole build before a single file is copied.

Design note: the delivered qpos is the source retarget's NATIVE G1 43-dim
free-joint qpos: root position + root quaternion (wxyz), 29 G1 joints, then
world-object pose in [:,36:43] (quaternion wxyz).  ``human_joints`` is an
additional FK product used by the partner anchor path; it does not make qpos
a human-body coordinate vector.  The same trimmed qpos is also exported as an
explicit ``target_g1_expansion_npz`` so SUGAR can run its standard Isaac
converter on the re-export clock without ambiguity.
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path("/home/ubuntu/Workspace/spider")
RESULTS = REPO / "workspace/core4d/results"  # symlink -> /mnt/.../core4d/results
ANALYSIS = REPO / "workspace/core4d/analysis/E197_full_cem_omnirt_vs_prg_metrics"
GATE_TSV = ANALYSIS / "e197_omni_absolute_wide_gate_filter.tsv"
METHOD_TSV = ANALYSIS / "e197_method_metrics.tsv"
RERUN = (
    RESULTS
    / "E197/s6_downstream/rl_export/partner_omnirt_rerun/omnirt_v1/results/omnirt_v1"
)
DELIVERY = REPO / "workspace/core4d/results/E197/s6_downstream/rl_export/partner_reexport_v2"
RELEASE = DELIVERY / "release/omnirt_v1_ref_fk"
MANIFESTS = DELIVERY / "manifests"

RAW_FPS = 30
CANONICAL_FPS = 50
THRESHOLD_M = 0.03
CONTACT_KEY = "raw_contact_mask_3cm"
GATE_PASS = "RL_CANDIDATE_OMNI_WIDE_GATE_PASS"

# Partners the main E170/E173 runs never retargeted; only present in the E197
# partner_omnirt_rerun tree (produced with --skip-contact).  Their raw contact
# is sourced from the gate-pass sibling (scene-level, frame-aligned).
RERUN_PARTNERS = {
    "box021_20231018_029_p2",
    "box021_20231018_034_p1",
    "box021_20231018_035_p1",
    "box023_20231020_039_p1",
}

G1_EXPANSION = DELIVERY / "target_g1_expansion"
SPIDER_PROCESSOR = REPO / "spider/process_datasets/core4d.py"
SCENE_ROOT = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
G1_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def partner_cid(cid: str) -> str:
    return cid[:-2] + ("p1" if cid.endswith("p2") else "p2")


def parse_case(cid: str) -> dict[str, str]:
    object_key = cid.split("_", 1)[0]
    person = cid[-2:]  # p1 / p2
    remainder = cid[len(object_key) + 1 : -3]  # strip 'boxNNN_' and '_pN'
    date, seq = remainder.rsplit("_", 1)
    actor = "person1" if person == "p1" else "person2"
    return {
        "object_key": object_key,
        "object_name": "Box" + object_key[3:],
        "scene_id": f"{object_key}_{remainder}",
        "date": date,
        "sequence": seq,
        "person": person,
        "actor": actor,
        "actor_index": "0" if person == "p1" else "1",
    }


def load_targets() -> list[str]:
    rows = read_tsv(GATE_TSV)
    return [r["case_id"] for r in rows if r.get("rl_filter_decision") == GATE_PASS]


def load_method_metrics() -> dict[str, dict[str, str]]:
    return {r["case_id"]: r for r in read_tsv(METHOD_TSV) if r.get("method") == "OmniRetarget"}


class SourceResolveError(RuntimeError):
    pass


def resolve_source(cid: str, mm: dict[str, dict[str, str]]) -> dict[str, Any]:
    """Locate source artifacts for one person-case.

    Returns dict with absolute Paths: untrimmed, trimmed, trim_window, contact
    (contact may be None for rerun partners -> filled from sibling later), plus
    provenance (source_exp, variant, slug, tree).
    """
    if cid in RERUN_PARTNERS:
        hits = glob.glob(str(RERUN / f"holosoma_rl_partner_omnirt_omnirt_v1_{cid}"))
        if not hits:
            raise SourceResolveError(f"{cid}: rerun dir missing")
        hd = Path(hits[0])
        return {
            "untrimmed": _one(hd / "retargeted", "*_with_obj_original.npz", cid),
            "trimmed": _one(hd / "trimmed", "*_with_obj_original.npz", cid),
            "trim_window": hd / "trim_window.json",
            "contact": None,
            "source_exp": "E197_rerun",
            "variant": "omnirt_v1",
            "slug": hd.name.replace("holosoma_", ""),
            "tree": str(hd.parent),
        }
    row = mm.get(cid)
    if not row or not row.get("contact_mask"):
        # Not in method_metrics: native retarget exists in some s3_retarget tree
        # (retargeted this scene but not gate-evaluated).  Glob for a complete
        # dcv3 dir (retargeted+trimmed+trim_window+contact), prefer omnirt_v1.
        return _resolve_by_glob(cid)
    cmp = Path(row["contact_mask"])
    # .../<tree>/contact_masks/<slug>/raw_contact_mask_3cm.npz
    slug = cmp.parent.name
    tree = cmp.parent.parent.parent  # strip contact_masks/<slug>/file
    hd = tree / f"holosoma_{slug}"
    variant = "omnirt_v2" if "omnirt_v2" in slug else "omnirt_v1"
    return {
        "untrimmed": _one(hd / "retargeted", "*_with_obj_original.npz", cid),
        "trimmed": _one(hd / "trimmed", "*_with_obj_original.npz", cid),
        "trim_window": hd / "trim_window.json",
        "contact": cmp,
        "source_exp": row.get("source_exp", ""),
        "variant": variant,
        "slug": slug,
        "tree": str(tree),
    }


def _one(directory: Path, pattern: str, cid: str) -> Path:
    hits = sorted(directory.glob(pattern))
    if len(hits) != 1:
        raise SourceResolveError(f"{cid}: expected exactly one {pattern} in {directory}, got {len(hits)}")
    return hits[0]


def _resolve_by_glob(cid: str) -> dict[str, Any]:
    """Fallback for native retargets absent from method_metrics: find a complete
    dcv3 s3_retarget dir on disk (retargeted+trimmed+trim_window+contact)."""
    roots = sorted(
        glob.glob(str(RESULTS / "E1*/s3_retarget/omnirt_v*/ref_fk*/results/omnirt_v*_ref_fk"))
    )
    # prefer omnirt_v1 trees.
    roots.sort(key=lambda r: ("omnirt_v2" in r, r))
    for tree in roots:
        if "archive_legacy" in tree:
            continue
        hits = glob.glob(f"{tree}/holosoma_dcv3_*_ref_fk_{cid}")
        if not hits:
            continue
        hd = Path(hits[0])
        slug = hd.name.replace("holosoma_", "")
        contact = Path(tree) / "contact_masks" / slug / "raw_contact_mask_3cm.npz"
        rtg = sorted((hd / "retargeted").glob("*_with_obj_original.npz"))
        trm = sorted((hd / "trimmed").glob("*_with_obj_original.npz"))
        tw = hd / "trim_window.json"
        if not (rtg and trm and tw.exists() and contact.exists()):
            continue
        variant = "omnirt_v2" if "omnirt_v2" in slug else "omnirt_v1"
        parts = Path(tree).parts
        source_exp = parts[parts.index("results") + 1] if "results" in parts else ""
        return {
            "untrimmed": rtg[0],
            "trimmed": trm[0],
            "trim_window": tw,
            "contact": contact,
            "source_exp": source_exp,
            "variant": variant,
            "slug": slug,
            "tree": tree,
        }
    raise SourceResolveError(f"{cid}: no method_metrics row and no complete native s3_retarget dir")


def validate_case(cid: str, src: dict[str, Any], contact_path: Path) -> dict[str, Any]:
    """Run the same gates bundle.py enforces.  Raise on any failure."""
    un = np.load(src["untrimmed"], allow_pickle=True)
    tr = np.load(src["trimmed"], allow_pickle=True)
    tw = json.loads(Path(src["trim_window"]).read_text())
    contact = np.load(contact_path, allow_pickle=True)

    for key in ("qpos", "human_joints", "fps", "cost"):
        if key not in un.files or key not in tr.files:
            raise SourceResolveError(f"{cid}: missing '{key}' in npz")

    start, end = int(tw["trim_start"]), int(tw["trim_end"])
    uf, tf = int(tw["untrimmed_frames"]), int(tw["trimmed_frames"])
    raw_frames = int(contact[CONTACT_KEY].shape[0])

    # frame-count consistency
    if un["qpos"].shape[0] != uf or un["human_joints"].shape[0] != uf:
        raise SourceResolveError(f"{cid}: untrimmed frames {un['qpos'].shape[0]} != untrimmed_frames {uf}")
    if end - start != tf:
        raise SourceResolveError(f"{cid}: trim_end-trim_start {end - start} != trimmed_frames {tf}")
    if raw_frames != uf:
        raise SourceResolveError(f"{cid}: raw_contact frames {raw_frames} != untrimmed_frames {uf}")

    # per-frame trim proof (exact)
    for key in ("qpos", "human_joints"):
        if not np.array_equal(tr[key], un[key][start:end]):
            raise SourceResolveError(f"{cid}: {key} trim proof failed")

    # schema gate
    if tr["qpos"].shape[1] < 43 or tuple(tr["human_joints"].shape[1:]) != (22, 3):
        raise SourceResolveError(
            f"{cid}: schema incompatible qpos{tr['qpos'].shape} human_joints{tr['human_joints'].shape}"
        )
    # unit quaternion (object orientation)
    qnorm = np.linalg.norm(tr["qpos"][:, 39:43], axis=1)
    if not np.allclose(qnorm, 1.0, atol=1e-4):
        raise SourceResolveError(f"{cid}: object quat not unit (max dev {np.abs(qnorm - 1).max():.2e})")

    # contact axes
    persons = [str(x) for x in contact["persons"]]
    hands = [str(x) for x in contact["hands"]]
    if persons != ["person1", "person2"]:
        raise SourceResolveError(f"{cid}: persons axis {persons}")
    if hands != ["left", "right"]:
        raise SourceResolveError(f"{cid}: hands axis {hands}")
    if abs(float(contact["threshold_m"]) - THRESHOLD_M) > 1e-9:
        raise SourceResolveError(f"{cid}: threshold_m {float(contact['threshold_m'])}")
    if tuple(contact[CONTACT_KEY].shape[1:]) != (2, 2):
        raise SourceResolveError(f"{cid}: contact shape {contact[CONTACT_KEY].shape}")
    if contact[CONTACT_KEY].dtype != np.bool_:
        raise SourceResolveError(f"{cid}: contact dtype {contact[CONTACT_KEY].dtype}")

    return {
        "raw_frame_count": raw_frames,
        "trim_start": start,
        "trim_end": end,
        "trimmed_frames": tf,
        "fps": int(un["fps"]),
    }


def spider_rel(path: Path) -> str:
    """SPIDER_ROOT::<repo-relative> path.

    Must NOT resolve() the path: the ``workspace/core4d/results`` segment is a
    symlink to /mnt, and resolving it yields an unstable ``../../../../mnt/...``
    string.  Pure-string relpath keeps the stable, downstream-resolvable form
    ``workspace/core4d/results/E197/...``.
    """
    return "SPIDER_ROOT::" + os.path.relpath(os.path.abspath(str(path)), str(REPO))


def copy_case(cid: str, src: dict[str, Any], contact_path: Path) -> dict[str, Any]:
    """Copy artifacts into the delivery tree under a uniform slug; rewrite trim_window paths."""
    slug = f"dcv3_omnirt_v1_ref_fk_{cid}"
    hd = RELEASE / f"holosoma_{slug}"
    (hd / "retargeted").mkdir(parents=True, exist_ok=True)
    (hd / "trimmed").mkdir(parents=True, exist_ok=True)
    (RELEASE / "contact_masks" / slug).mkdir(parents=True, exist_ok=True)

    un_name = Path(src["untrimmed"]).name
    tr_name = Path(src["trimmed"]).name
    dst_un = hd / "retargeted" / un_name
    dst_tr = hd / "trimmed" / tr_name
    dst_contact = RELEASE / "contact_masks" / slug / "raw_contact_mask_3cm.npz"
    shutil.copy2(src["untrimmed"], dst_un)
    shutil.copy2(src["trimmed"], dst_tr)
    shutil.copy2(contact_path, dst_contact)

    tw = json.loads(Path(src["trim_window"]).read_text())
    tw_out = {
        "untrimmed": spider_rel(dst_un),
        "trimmed": spider_rel(dst_tr),
        "untrimmed_frames": int(tw["untrimmed_frames"]),
        "trimmed_frames": int(tw["trimmed_frames"]),
        "trim_start": int(tw["trim_start"]),
        "trim_end": int(tw["trim_end"]),
        "trim_frames": int(tw.get("trim_frames", tw["trimmed_frames"])),
    }
    dst_tw = hd / "trim_window.json"
    dst_tw.write_text(json.dumps(tw_out, indent=1) + "\n")
    return {
        "slug": slug,
        "untrimmed_npz": dst_un,
        "trimmed_npz": dst_tr,
        "trim_window_json": dst_tw,
        "raw_contact_path": dst_contact,
    }


def _write_g1_expansion(cid: str, trimmed_npz: Path) -> dict[str, Path | int]:
    """Export the target's native G1 free-joint qpos on the re-export 30Hz clock."""
    with np.load(trimmed_npz, allow_pickle=True) as z:
        qpos = np.asarray(z["qpos"], dtype=np.float64)
        fps = int(z["fps"])
    if qpos.ndim != 2 or qpos.shape[1] != 43 or fps != RAW_FPS:
        raise SourceResolveError(
            f"{cid}: G1 expansion requires qpos(T,43) at 30Hz, "
            f"got {qpos.shape}, fps={fps}"
        )
    for sl, label in ((slice(3, 7), "floating-base"), (slice(39, 43), "object")):
        dev = np.max(np.abs(np.linalg.norm(qpos[:, sl], axis=1) - 1.0))
        if dev > 1e-4:
            raise SourceResolveError(
                f"{cid}: {label} quaternion not unit (max dev {dev:.2e})"
            )
    out = G1_EXPANSION / f"{cid}_g1_expansion_30hz.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        qpos=qpos,
        fps=np.array(RAW_FPS, dtype=np.int32),
        joint_names=np.asarray(G1_JOINT_NAMES),
        qpos_layout=np.array(
            "root_pos_xyz,root_quat_wxyz,g1_29dof,"
            "object_pos_xyz,object_quat_wxyz"
        ),
        source_case_id=np.array(cid),
    )
    return {"path": out, "frames": len(qpos)}


def _resolve_scene(cid: str) -> Path:
    """Resolve the matching SPIDER scene, preferring the v1 scene."""
    for variant in ("omnirt_v1", "omnirt_v2"):
        path = SCENE_ROOT / f"dcv3_{variant}_ref_fk_{cid}" / "scene.xml"
        if path.is_file():
            return path
    raise SourceResolveError(f"{cid}: no dcv3 scene.xml found for core4d processor")


def _run_spider_processor(cid: str, trimmed_npz: Path, out_root: Path) -> dict[str, Path | int]:
    """Run SPIDER's MuJoCo converter to produce qvel-bearing trajectory data."""
    scene = _resolve_scene(cid)
    task_root = scene.parent
    task = task_root.name
    source_trajectory = task_root / "0/trajectory_kinematic.npz"
    if source_trajectory.is_file():
        with np.load(source_trajectory, allow_pickle=False) as existing, np.load(trimmed_npz, allow_pickle=True) as src:
            if existing["qpos"].shape == src["qpos"].shape and np.array_equal(existing["qpos"], src["qpos"]):
                trajectory = out_root / "processed/trajectory_kinematic" / f"{cid}.npz"
                trajectory.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_trajectory, trajectory)
                with np.load(trajectory, allow_pickle=False) as data:
                    if data["qvel"].shape[1] != 41:
                        raise SourceResolveError(f"{cid}: existing trajectory qvel shape {data['qvel'].shape}")
                    return {"path": trajectory, "frames": int(data["qvel"].shape[0]), "scene": scene}
    cmd = [
        "uv", "run", "python", str(SPIDER_PROCESSOR),
        "--source-npz", str(trimmed_npz),
        "--dataset-dir", str(REPO / "example_datasets"),
        "--task", task,
        "--data-id", "0",
        "--no-save-video",
    ]
    proc = subprocess.run(cmd, cwd=REPO, text=True, capture_output=True)
    if proc.returncode:
        raise SourceResolveError(
            f"{cid}: core4d.py failed ({proc.returncode}): {proc.stderr[-1200:]}"
        )
    if not source_trajectory.is_file():
        raise SourceResolveError(f"{cid}: core4d.py did not write {source_trajectory}")
    trajectory = out_root / "processed/trajectory_kinematic" / f"{cid}.npz"
    trajectory.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_trajectory, trajectory)
    with np.load(trajectory, allow_pickle=False) as data:
        if data["qpos"].shape[1] != 43 or data["qvel"].shape[1] != 41:
            raise SourceResolveError(
                f"{cid}: unexpected trajectory schema qpos={data['qpos'].shape} qvel={data['qvel'].shape}"
            )
        if data["qpos"].shape[0] != data["qvel"].shape[0]:
            raise SourceResolveError(f"{cid}: trajectory qpos/qvel frame mismatch")
    with np.load(trajectory, allow_pickle=False) as data:
        frames = int(data["qvel"].shape[0])
    return {"path": trajectory, "frames": frames, "scene": scene}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="overwrite existing delivery tree")
    args = ap.parse_args()

    git_commit = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()

    targets = load_targets()
    mm = load_method_metrics()
    assert len(targets) == 52, f"expected 52 gate-pass targets, got {len(targets)}"

    # Build the 70-unique person-case set (targets + partners).
    needed: list[str] = []
    for c in targets:
        for x in (c, partner_cid(c)):
            if x not in needed:
                needed.append(x)

    # 1) Resolve + validate ALL cases (fail-closed) before any copy.
    print(f"[1/3] resolving + validating {len(needed)} person-cases ...")
    resolved: dict[str, dict[str, Any]] = {}
    contact_for: dict[str, Path] = {}
    errors: list[str] = []
    for cid in needed:
        try:
            resolved[cid] = resolve_source(cid, mm)
        except SourceResolveError as exc:
            errors.append(str(exc))
    if errors:
        for e in errors:
            print("  RESOLVE FAIL:", e)
        raise SystemExit("aborting: unresolved sources")

    # rerun partners take contact from their gate-pass sibling (scene-level).
    for cid in needed:
        src = resolved[cid]
        if src["contact"] is not None:
            contact_for[cid] = Path(src["contact"])
        else:
            sib = partner_cid(cid)
            sib_contact = resolved.get(sib, {}).get("contact")
            if not sib_contact:
                errors.append(f"{cid}: rerun partner has no sibling contact ({sib})")
                continue
            contact_for[cid] = Path(sib_contact)

    validated: dict[str, dict[str, Any]] = {}
    for cid in needed:
        if cid not in contact_for:
            continue
        try:
            validated[cid] = validate_case(cid, resolved[cid], contact_for[cid])
        except SourceResolveError as exc:
            errors.append(str(exc))
    if errors:
        for e in errors:
            print("  VALIDATE FAIL:", e)
        raise SystemExit("aborting: validation failures")

    # cross-check: target and partner share raw_frame_count per scene.
    for c in targets:
        p = partner_cid(c)
        if validated[c]["raw_frame_count"] != validated[p]["raw_frame_count"]:
            errors.append(
                f"{c}: target raw_frame_count {validated[c]['raw_frame_count']} != "
                f"partner {p} {validated[p]['raw_frame_count']}"
            )
    if errors:
        for e in errors:
            print("  SCENE FAIL:", e)
        raise SystemExit("aborting: target/partner frame mismatch")
    print(f"      all {len(needed)} cases pass bundle gates")

    # 2) Copy into delivery tree.
    if DELIVERY.exists():
        if not args.force:
            raise SystemExit(f"delivery exists: {DELIVERY} (use --force)")
        shutil.rmtree(DELIVERY)
    RELEASE.mkdir(parents=True, exist_ok=True)
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    print(f"[2/3] copying {len(needed)} person-cases into {DELIVERY} ...")
    placed: dict[str, dict[str, Any]] = {}
    for cid in needed:
        placed[cid] = copy_case(cid, resolved[cid], contact_for[cid])

    print("      packaging 52 target G1 expansion inputs ...")
    g1_expansion: dict[str, dict[str, Path | int]] = {}
    for cid in sorted(targets):
        g1_expansion[cid] = _write_g1_expansion(cid, placed[cid]["trimmed_npz"])

    print("      running SPIDER core4d processor for qvel-bearing trajectories ...")
    spider_trajectories: dict[str, dict[str, Path | int]] = {}
    for cid in sorted(targets):
        spider_trajectories[cid] = _run_spider_processor(
            cid, placed[cid]["trimmed_npz"], DELIVERY
        )

    # 3) SHA256 sidecar (every delivered file), pairing manifest, provenance, summary.
    print("[3/3] hashing + writing manifests ...")
    sha_rows: list[dict[str, str]] = []
    sha_cache: dict[Path, str] = {}

    def h(path: Path) -> str:
        if path not in sha_cache:
            sha_cache[path] = sha256_file(path)
            sha_rows.append({"path": spider_rel(path), "sha256": sha_cache[path], "bytes": str(path.stat().st_size)})
        return sha_cache[path]

    for cid in needed:
        pl = placed[cid]
        for k in ("untrimmed_npz", "trimmed_npz", "trim_window_json", "raw_contact_path"):
            h(pl[k])

    # pairing manifest: 52 target rows, target<->partner file mapping.
    pair_cols = [
        "case_id", "scene_id", "date", "sequence", "object_name",
        "target_actor", "target_actor_index", "partner_actor", "partner_actor_index",
        "raw_frame_count", "raw_fps", "canonical_fps",
        "raw_contact_key", "partner_contact_person_index", "contact_left_index",
        "contact_right_index", "contact_threshold_m",
        "source_pairing_case_id", "spider_git_commit",
        "upstream_version", "upstream_method_id", "partner_status",
        "target_untrimmed_npz", "target_trimmed_npz", "target_trim_window_json",
        "partner_untrimmed_npz", "partner_trimmed_npz", "partner_trim_window_json",
        "target_raw_contact_path", "raw_contact_path",
        "target_untrimmed_npz_sha256", "target_trimmed_npz_sha256", "target_trim_window_json_sha256",
        "partner_untrimmed_npz_sha256", "partner_trimmed_npz_sha256", "partner_trim_window_json_sha256",
        "target_raw_contact_path_sha256", "raw_contact_path_sha256",
        "target_g1_expansion_npz", "target_g1_expansion_npz_sha256",
        "target_g1_expansion_frames", "target_g1_expansion_fps",
        "target_g1_qpos_layout", "target_g1_joint_order",
        "target_spider_trajectory_kinematic_npz",
        "target_spider_trajectory_kinematic_npz_sha256",
        "target_spider_trajectory_frames", "target_spider_qvel_shape",
    ]
    pair_rows: list[dict[str, str]] = []
    for cid in sorted(targets):
        p = partner_cid(cid)
        ci = parse_case(cid)
        tp, pp = placed[cid], placed[p]
        row = {
            "case_id": cid,
            "scene_id": ci["scene_id"],
            "date": ci["date"],
            "sequence": ci["sequence"],
            "object_name": ci["object_name"],
            "target_actor": ci["actor"],
            "target_actor_index": ci["actor_index"],
            "partner_actor": parse_case(p)["actor"],
            "partner_actor_index": parse_case(p)["actor_index"],
            "raw_frame_count": str(validated[cid]["raw_frame_count"]),
            "raw_fps": str(RAW_FPS),
            "canonical_fps": str(CANONICAL_FPS),
            "raw_contact_key": CONTACT_KEY,
            "partner_contact_person_index": parse_case(p)["actor_index"],
            "contact_left_index": "0",
            "contact_right_index": "1",
            "contact_threshold_m": str(THRESHOLD_M),
            "source_pairing_case_id": cid,
            "spider_git_commit": git_commit,
            "upstream_version": resolved[cid]["source_exp"],
            "upstream_method_id": f"{resolved[cid]['variant']}_TSV_TRAJECTORY",
            "partner_status": "pass",
            "target_untrimmed_npz": spider_rel(tp["untrimmed_npz"]),
            "target_trimmed_npz": spider_rel(tp["trimmed_npz"]),
            "target_trim_window_json": spider_rel(tp["trim_window_json"]),
            "partner_untrimmed_npz": spider_rel(pp["untrimmed_npz"]),
            "partner_trimmed_npz": spider_rel(pp["trimmed_npz"]),
            "partner_trim_window_json": spider_rel(pp["trim_window_json"]),
            "target_raw_contact_path": spider_rel(tp["raw_contact_path"]),
            "raw_contact_path": spider_rel(pp["raw_contact_path"]),
            "target_untrimmed_npz_sha256": h(tp["untrimmed_npz"]),
            "target_trimmed_npz_sha256": h(tp["trimmed_npz"]),
            "target_trim_window_json_sha256": h(tp["trim_window_json"]),
            "partner_untrimmed_npz_sha256": h(pp["untrimmed_npz"]),
            "partner_trimmed_npz_sha256": h(pp["trimmed_npz"]),
            "partner_trim_window_json_sha256": h(pp["trim_window_json"]),
            "target_raw_contact_path_sha256": h(tp["raw_contact_path"]),
            "raw_contact_path_sha256": h(pp["raw_contact_path"]),
        }
        g1 = g1_expansion[cid]
        row["target_g1_expansion_npz"] = spider_rel(g1["path"])
        row["target_g1_expansion_npz_sha256"] = h(g1["path"])
        row["target_g1_expansion_frames"] = str(g1["frames"])
        row["target_g1_expansion_fps"] = str(RAW_FPS)
        row["target_g1_qpos_layout"] = (
            "root_pos_xyz,root_quat_wxyz,g1_29dof,"
            "object_pos_xyz,object_quat_wxyz"
        )
        row["target_g1_joint_order"] = ",".join(G1_JOINT_NAMES)
        traj = spider_trajectories[cid]
        row["target_spider_trajectory_kinematic_npz"] = spider_rel(traj["path"])
        row["target_spider_trajectory_kinematic_npz_sha256"] = h(traj["path"])
        row["target_spider_trajectory_frames"] = str(traj["frames"])
        row["target_spider_qvel_shape"] = f"({traj['frames']},41)"
        pair_rows.append(row)

    _write_tsv(MANIFESTS / "source_pairing_manifest.tsv", pair_rows, pair_cols)
    _write_tsv(MANIFESTS / "file_sha256.tsv", sha_rows, ["path", "sha256", "bytes"])

    prov_rows = [
        {
            "case_id": cid,
            "is_target": str(cid in targets).lower(),
            "source_exp": resolved[cid]["source_exp"],
            "source_variant": resolved[cid]["variant"],
            "source_slug": resolved[cid]["slug"],
            "source_tree": resolved[cid]["tree"],
            "contact_source_case": cid if resolved[cid]["contact"] else partner_cid(cid),
            "raw_frame_count": str(validated[cid]["raw_frame_count"]),
            "trimmed_frames": str(validated[cid]["trimmed_frames"]),
            "fps": str(validated[cid]["fps"]),
        }
        for cid in sorted(needed)
    ]
    _write_tsv(
        MANIFESTS / "provenance.tsv",
        prov_rows,
        ["case_id", "is_target", "source_exp", "source_variant", "source_slug", "source_tree",
         "contact_source_case", "raw_frame_count", "trimmed_frames", "fps"],
    )

    summary = {
        "deliverable": "E197 OmniRetarget partner re-export for R018-23",
        "spider_git_commit": git_commit,
        "gate_pass_targets": len(targets),
        "unique_person_cases": len(needed),
        "rerun_partners_contact_from_sibling": sorted(RERUN_PARTNERS),
        "variant_tally_targets": _tally(resolved[c]["variant"] for c in targets),
        "source_exp_tally_targets": _tally(resolved[c]["source_exp"] for c in targets),
        "raw_fps": RAW_FPS,
        "canonical_fps": CANONICAL_FPS,
        "contact_threshold_m": THRESHOLD_M,
        "delivery_root": spider_rel(DELIVERY),
        "release_root": spider_rel(RELEASE),
        "pairing_manifest": spider_rel(MANIFESTS / "source_pairing_manifest.tsv"),
        "files_hashed": len(sha_rows),
        "target_g1_expansion_cases": len(g1_expansion),
        "target_g1_contract": (
            "qpos(T,43): root pos+quat(wxyz), G1 29DoF, "
            "object pos+quat(wxyz), fps=30"
        ),
        "target_spider_processor": "spider/process_datasets/core4d.py",
        "target_spider_trajectory_cases": len(spider_trajectories),
    }
    (MANIFESTS / "reexport_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    validation = {cid: validated[cid] for cid in sorted(needed)}
    (MANIFESTS / "validation_report.json").write_text(json.dumps(validation, indent=1) + "\n")

    print(json.dumps(summary, indent=2))
    print("OK")
    return 0


def _tally(it: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for x in it:
        out[x] = out.get(x, 0) + 1
    return out


def _write_tsv(path: Path, rows: list[dict[str, str]], cols: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", extrasaction="raise")
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    sys.exit(main())
