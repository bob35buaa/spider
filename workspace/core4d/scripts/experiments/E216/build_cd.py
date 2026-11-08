#!/usr/bin/env python3
"""E216 chair/desk driver: seed -> retarget@0.1m -> SPIDER task + E206 lowgeom PRG
scene + per-case fractional gravcomp sidecar -> frozen CEM manifest.

One cohesive driver for the 20-case Phase 2 (desk/chair) track.  Mirrors the
box/bucket e216 flow but on the E208/E213 scene family (omnirt_v1, E206 lowgeom
PRG via build_arm_scenes.build_one, fractional object gravcomp G1/G0.8/G0.6/G0.4).

Stages (each idempotent):
  seed       copy _original (+converted+trim_window) from E208's v{bv} tree.
             trans_* NOT seeded -> upstream recomputes at 0.1m.
  retarget   pipeline.sh aug branch, OMNIRT_V1_ENV + OMNIRT_TRANS_OFFSET_M=0.1 +
             OMNIRT_TRANS_ONLY=1.  Object groups parallel, cases serial.
  build      fixed-window trim + create_spider_scene + core4d.py + ARM.build_one
             (scene_act_E206_lowgeom_PRG) + fractional gravcomp sidecar.
  manifest   clone E208's 0.2m aug base yaml (task swapped to _0p1m) + extra
             scene_name=<gravcomp>; freeze; emit priority manifest for run_E216_cem.

Usage:
    .venv/bin/python .../E216/build_cd.py --stage all [--objects chair006] [--cases ..]
    ... --stage retarget --max-workers 5
    ... --stage manifest --freeze
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import os
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "E212"))

import e216cd_common as C  # noqa: E402
import e212_common as E212C  # noqa: E402  (assert_gravcomp_diff_value)

CREATE_SCENE = "workspace/core4d/data_preprocess/create_spider_scene_from_template.py"
CORE4D = "spider/process_datasets/core4d.py"
OBJ_POS = slice(36, 39)
SKIP_MARKER = re.compile(r"\[skip\] augmentation (?P<name>\S+) infeasible/failed: (?P<reason>.*)")
CASE_FIELDS = ["# enabled", "date", "seq", "person", "object_name", "object_model_rel",
               "source_scene_task", "target_task", "trim_start", "trim_frames", "data_id", "mask_slug"]


# ==========================================================================
# seed
# ==========================================================================
def _link_or_copy(src: Path, dst: Path) -> None:
    import shutil
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    tmp = dst.with_name(f".{dst.name}.cdtmp")
    tmp.unlink(missing_ok=True)
    try:
        os.link(src, tmp)
    except OSError:
        shutil.copy2(src, tmp)
    tmp.replace(dst)


def seed_case(case: dict[str, str]) -> str:
    import shutil
    meta = C.load_case_meta(case["base_target_task"])
    holo = meta["holosoma_task"]
    src = C.seed_source_dir(case["case_id"])
    dst = C.holosoma_dir(case["case_id"])
    if not src.is_dir():
        return f"fail_no_source:{C.rel(src)}"
    conv = src / "converted"
    if conv.is_dir():
        for f in sorted(conv.glob("*.npz")):
            _link_or_copy(f, dst / "converted" / f.name)
    for sub in ("retargeted", "trimmed"):
        s = src / sub / f"{holo}_original.npz"
        if not s.is_file():
            return f"fail_missing_{sub}_original"
        _link_or_copy(s, dst / sub / s.name)
    tw = src / "trim_window.json"
    if not tw.is_file():
        return "fail_missing_trim_window"
    dst.mkdir(parents=True, exist_ok=True)
    if not (dst / "trim_window.json").exists():
        shutil.copy2(tw, dst / "trim_window.json")
    return "seeded"


# ==========================================================================
# retarget @0.1m
# ==========================================================================
def pipeline_env(case_id: str) -> dict[str, str]:
    env = dict(os.environ)
    env.update({
        "REPO": str(C.REPO), "HOLOSOMA_DIR": str(C.HOLOSOMA_REPO),
        "CORE4D_REAL_ROOT": str(C.CORE4D_RAW_ROOT), "SMPLX_MODEL_DIR": str(C.SMPLX_MODEL_DIR),
        "RESULT_ROOT": C.rel(C.data_root()),
        "PYTHON_BIN": str(C.SPIDER_PYTHON_BIN), "RETARGET_PYTHON_BIN": str(C.RETARGET_PYTHON_BIN),
        "KEEP_GOING": "0", "RETARGET_AUGMENTATION": "1",
        "RETARGET_MAX_WORKERS": os.environ.get("E216CD_RETARGET_MAX_WORKERS", "6"),
        "OMNIRT_TRANS_OFFSET_M": C.OMNIRT_TRANS_OFFSET_M, "OMNIRT_TRANS_ONLY": "1",
    })
    env.update(C.PIPELINE_DATASET_ENV)
    env.update(C.omnirt_env(case_id))   # per-case: v2-base case runs v2, rest v1
    return env


def write_case_file(case: dict[str, str], meta: dict[str, str]) -> Path:
    path = C.DP_CASE_FILES / f"cases_e216cd_{case['base_target_task']}.tsv"
    path.parent.mkdir(parents=True, exist_ok=True)
    row = ["1", meta["date"], meta["seq"], meta["person"], meta["object_name"],
           meta["object_model_rel"], meta["source_scene_task"], case["base_target_task"],
           "auto", "auto", "0", case["base_target_task"]]
    path.write_text("\t".join(CASE_FIELDS) + "\n" + "\t".join(row) + "\n", encoding="utf-8")
    return path


def retarget_case(case: dict[str, str], *, timeout_min: int = 180) -> dict[str, Any]:
    meta = C.load_case_meta(case["base_target_task"])
    base = case["base_target_task"]
    sentinel = C.sentinel_path(case["case_id"])
    log_path = C.DP_LOGS / f"{base}.log"
    if sentinel.is_file():
        return {"case_id": case["case_id"], "status": "skip_sentinel"}
    warm = C.holosoma_dir(case["case_id"]) / "retargeted" / f"{meta['holosoma_task']}_original.npz"
    if not warm.is_file():
        return {"case_id": case["case_id"], "status": "fail_no_warmstart"}
    cf = write_case_file(case, meta)
    cmd = ["bash", "workspace/core4d/data_preprocess/pipeline.sh", "--case-file", C.rel(cf),
           "--skip-contact", "--skip-spider"]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log_path.open("w", encoding="utf-8") as stream:
        stream.write(f"# E216CD retarget_variant={C.RETARGET_VARIANT} offset={C.OMNIRT_TRANS_OFFSET_M}\n# cmd={' '.join(cmd)}\n\n")
        stream.flush()
        try:
            proc = subprocess.run(cmd, cwd=C.REPO, env=pipeline_env(case["case_id"]), stdout=stream,
                                  stderr=subprocess.STDOUT, timeout=timeout_min * 60, check=False)
            rc, timed_out = proc.returncode, False
        except subprocess.TimeoutExpired:
            rc, timed_out = -1, True
    wall = time.time() - started
    built = []
    for short, holo in C.BUILD_VARIANTS:
        npz = C.holosoma_dir(case["case_id"]) / "retargeted" / f"{meta['holosoma_task']}_{holo}.npz"
        if npz.is_file() and npz.stat().st_size > 0:
            built.append(short)
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text(json.dumps({"case_id": case["case_id"], "rc": rc, "timed_out": timed_out,
                                    "wall_s": round(wall, 1), "built": built}, indent=2), encoding="utf-8")
    return {"case_id": case["case_id"], "status": "timeout" if timed_out else ("ok" if rc == 0 else f"rc{rc}"),
            "built": built, "wall_s": round(wall, 1)}


# ==========================================================================
# build: trim + scene + gravcomp sidecar
# ==========================================================================
def _slice_time(data: Any, trim_start: int) -> dict[str, Any]:
    n = int(data["qpos"].shape[0])
    return {k: (data[k][trim_start:] if hasattr(data[k], "shape") and data[k].ndim >= 1
                and data[k].shape[0] == n else data[k]) for k in data.files}


def fixed_window_trim(case: dict[str, str], meta: dict[str, str], feasible_holo: list[str]) -> int:
    C._drop_broken_torch()
    root = C.holosoma_dir(case["case_id"])
    holo = meta["holosoma_task"]
    with np.load(root / "retargeted" / f"{holo}_original.npz", allow_pickle=True) as d:
        t_orig = int(d["qpos"].shape[0])
    with np.load(root / "trimmed" / f"{holo}_original.npz", allow_pickle=True) as d:
        t_trim = int(d["qpos"].shape[0])
    trim_start = t_orig - t_trim
    (root / "trimmed").mkdir(parents=True, exist_ok=True)
    for holo_name in feasible_holo:
        src = root / "retargeted" / f"{holo}_{holo_name}.npz"
        dst = root / "trimmed" / f"{holo}_{holo_name}.npz"
        if not src.is_file() or dst.is_file():
            continue
        with np.load(src, allow_pickle=True) as d:
            np.savez(str(dst), **_slice_time(d, trim_start))
    return trim_start


def write_gravcomp_sidecar(prg_scene: Path, value: str, *, overwrite: bool = False) -> Path:
    out = prg_scene.with_name(f"{C.SCENE_GRAVCOMP}.xml")
    if out.is_file() and not overwrite:
        E212C.assert_gravcomp_diff_value(prg_scene, out, value)
        return out
    tree = ET.parse(prg_scene)
    objs = [b for b in tree.getroot().iter("body") if b.get("name") == "object"]
    if len(objs) != 1:
        raise ValueError(f"expected 1 object body in {C.rel(prg_scene)}, found {len(objs)}")
    if objs[0].get("gravcomp") not in (None, "0", "0.0"):
        raise ValueError(f"object already has gravcomp: {objs[0].get('gravcomp')!r}")
    objs[0].set("gravcomp", value)
    ET.indent(tree, space="  ")
    tree.write(out, encoding="utf-8", xml_declaration=True)
    E212C.assert_gravcomp_diff_value(prg_scene, out, value)
    return out


def pose_diff(case: dict[str, str], meta: dict[str, str], holo_name: str, k: int = 10) -> dict[str, float]:
    C._drop_broken_torch()
    root = C.holosoma_dir(case["case_id"])
    holo = meta["holosoma_task"]
    with np.load(root / "trimmed" / f"{holo}_original.npz", allow_pickle=True) as d:
        qo = np.asarray(d["qpos"], np.float64)
    with np.load(root / "trimmed" / f"{holo}_{holo_name}.npz", allow_pickle=True) as d:
        qa = np.asarray(d["qpos"], np.float64)
    n = min(len(qo), len(qa))
    tr = np.linalg.norm(qa[:n, OBJ_POS] - qo[:n, OBJ_POS], axis=1)
    kk = min(k, n)
    return {"approach_trans_offset_m_max": float(tr[:kk].max()), "endpoint_trans_offset_m": float(tr[-1])}


def build_variant(case: dict[str, str], meta: dict[str, str], variant: str, holo_name: str,
                  *, overwrite: bool) -> dict[str, Any]:
    root = C.holosoma_dir(case["case_id"])
    trimmed = root / "trimmed" / f"{meta['holosoma_task']}_{holo_name}.npz"
    if not trimmed.is_file():
        raise FileNotFoundError(trimmed)
    aug_task = C.aug_task_name(case["case_id"], variant)
    source_scene = C.TASK_ROOT / case["base_target_task"] / "scene.xml"
    if not source_scene.is_file():
        raise FileNotFoundError(f"base template missing: {source_scene}")
    task_dir = C.TASK_ROOT / aug_task
    trajectory = task_dir / "0/trajectory_kinematic.npz"
    scene_act = task_dir / "scene_act.xml"
    scene_common = [str(C.SPIDER_PYTHON_BIN), CREATE_SCENE, "--source-scene", str(source_scene),
                    "--task", aug_task, "--qpos", str(trimmed), "--data-id", "0",
                    "--date", meta["date"], "--seq", meta["seq"], "--person", meta["person"],
                    "--object-name", meta["object_name"], "--object-model-rel", meta["object_model_rel"]]
    if not (trajectory.is_file() and scene_act.is_file() and not overwrite):
        subprocess.run(scene_common, cwd=C.REPO, check=True)
        subprocess.run([str(C.SPIDER_PYTHON_BIN), CORE4D, "--source-npz", str(trimmed), "--task", aug_task,
                        "--data-id", "0", "--dataset-name", "core4d", "--robot-type", "unitree_g1",
                        "--embodiment-type", "humanoid_object", "--no-show-viewer", "--no-save-video"],
                       cwd=C.REPO, check=True)
        subprocess.run(scene_common + ["--generate-scene-act"], cwd=C.REPO, check=True)
    arm = C.ARM.build_one({"case_id": case["case_id"], "object_key": case["object_key"],
                           "target_task": aug_task}, dry_run=False)
    if arm.get("status") != "built":
        raise RuntimeError(f"{aug_task}: arm build returned {arm.get('status')}")
    prg_scene = task_dir / f"{C.SCENE_NAME}.xml"
    gc_scene = write_gravcomp_sidecar(prg_scene, case["gravcomp"], overwrite=overwrite)
    row = {"aug_variant": variant, "holosoma_variant": holo_name,
           "effective_retarget_variant": C.RETARGET_VARIANT, "target_task": aug_task,
           "target_scene": C.rel(task_dir / "scene.xml"), "trajectory": C.rel(trajectory),
           "trajectory_sha256": C.sha256(trajectory), "base_scene_act": C.rel(scene_act),
           "base_scene_sha256": C.sha256(scene_act), "scene_act": C.rel(gc_scene),
           "scene_name": C.SCENE_GRAVCOMP, "effective_scene_sha256": C.sha256(gc_scene),
           "prg_scene": C.rel(prg_scene), "gravcomp": case["gravcomp"],
           "object_geom_count": int(arm["object_geom_count"]),
           "compiled_robot_object_pair_count": int(arm["prg_compiled"]["robot_object_pairs"]),
           "trimmed_npz": C.rel(trimmed)}
    row.update(pose_diff(case, meta, holo_name))
    row["weak_trans"] = int(float(row["approach_trans_offset_m_max"]) < C.WEAK_TRANS_FLOOR_M)
    row["status"] = "built" if not row["weak_trans"] else "built_weak_trans"
    return row


def feasible_variants(case_id: str) -> list[str]:
    """trans variants present on disk (retarget produced them)."""
    meta = C.load_case_meta(C.base_target_task(case_id))
    root = C.holosoma_dir(case_id)
    ok = []
    for short, holo in C.BUILD_VARIANTS:
        if (root / "retargeted" / f"{meta['holosoma_task']}_{holo}.npz").is_file():
            ok.append(short)
    return ok


def build_case(case: dict[str, str], *, overwrite: bool) -> list[dict[str, Any]]:
    meta = C.load_case_meta(case["base_target_task"])
    ok = feasible_variants(case["case_id"])
    if not ok:
        return []
    holo_by = dict(C.BUILD_VARIANTS)
    fixed_window_trim(case, meta, [holo_by[s] for s in ok])
    rows = []
    for short in ok:
        try:
            r = build_variant(case, meta, short, holo_by[short], overwrite=overwrite)
        except Exception as exc:  # noqa: BLE001
            r = {"aug_variant": short, "holosoma_variant": holo_by[short], "target_task": C.aug_task_name(case["case_id"], short),
                 "status": f"fail_{type(exc).__name__}", "error": str(exc)[:400]}
        r.update({"case_id": case["case_id"], "object_key": case["object_key"], "arm_group": case["arm_group"],
                  "base_variant": case["base_variant"], "base_target_task": case["base_target_task"],
                  "aug_translation": C.aug_translation(short), "aug_rotation_rad": C.aug_rotation_rad(short),
                  "updated_at": C.now()})
        rows.append(r)
        mark = "OK " if r["status"].startswith("built") else "ERR"
        print(f"  [{mark}] {short} {r['target_task']} "
              + (f"trans={r.get('approach_trans_offset_m_max', 0):.3f}m g={case['gravcomp']} pairs={r.get('compiled_robot_object_pair_count','?')}"
                 if r["status"].startswith("built") else r.get("error", "")[:160]), flush=True)
    return rows


# ==========================================================================
# manifest + overrides
# ==========================================================================
CEM_ROOT = None  # set in main via C


def clone_override(case: dict[str, str], aug_task: str, variant: str) -> tuple[Path, str]:
    """Clone E208's 0.2m aug base yaml (task swapped to the 0p1m dir)."""
    sib = C.OVERRIDE_DIR / f"core4d_{aug_task.replace('_0p1m', '')}.yaml"  # E208 0.2m sibling
    src = sib if sib.is_file() else (C.OVERRIDE_DIR / f"core4d_{case['base_target_task']}.yaml")
    kind = "e208_0p2m_sibling" if sib.is_file() else "dcv3_base"
    if not src.is_file():
        raise FileNotFoundError(f"{case['case_id']}: no clone source ({sib.name})")
    lines, swapped = [], 0
    for ln in src.read_text(encoding="utf-8").splitlines(keepends=True):
        if ln.startswith("task:"):
            lines.append(f"task: {aug_task}\n"); swapped += 1
        else:
            lines.append(ln)
    if swapped != 1:
        raise AssertionError(f"expected 1 task: line in {src.name}, found {swapped}")
    lines.insert(1, f"# E216CD: cloned from {src.name} ({kind}); task retargeted at 0.1m dir.\n")
    out = C.OVERRIDE_DIR / f"core4d_{aug_task}.yaml"
    out.write_text("".join(lines), encoding="utf-8")
    mask = ""
    for ln in out.read_text(encoding="utf-8").splitlines():
        if ln.strip().startswith("contact_hdmi_mask_path:"):
            mask = ln.split(":", 1)[1].strip().strip('"')
    return out, mask


def artifact_paths(vid: str) -> dict[str, str]:
    root = C.rel(C.S6_DIR / "cem/full")
    return {"variant": vid, "result_npz": f"{root}/{vid}.npz",
            "outdir_npz": f"{root}/{vid}/trajectory_mjwp_act.npz",
            "config_act": f"{root}/{vid}/config_act.yaml",
            "video": f"{C.rel(C.S6_DIR / 'render/full')}/{vid}_full.mp4",
            "log": f"logs/E216CD/cem/full/{vid}.log"}


def build_manifest(*, freeze: bool) -> int:
    arts = [r for r in C.read_tsv(C.ARTIFACTS_TSV) if r.get("status", "").startswith("built")]
    if not arts:
        raise SystemExit(f"no built rows in {C.rel(C.ARTIFACTS_TSV)}")
    budget = C.frozen_budget()
    rows, blockers = [], []
    seen_obj: set[str] = set()
    for art in sorted(arts, key=lambda r: (r["object_key"], r["case_id"], r["aug_variant"])):
        case = {"case_id": art["case_id"], "object_key": art["object_key"], "base_variant": art["base_variant"],
                "base_target_task": art["base_target_task"], "gravcomp": art.get("gravcomp", "")}
        variant, aug_task = art["aug_variant"], art["target_task"]
        base_yaml, mask = clone_override(case, aug_task, variant)
        vid = C.aug_variant_id(art["case_id"], variant)
        is_first = variant == "trans0" and art["object_key"] not in seen_obj
        if is_first:
            seen_obj.add(art["object_key"])
        tier = "P0" if is_first else ("P1" if variant == "trans0" else "P2")
        extra = f"scene_name={C.SCENE_GRAVCOMP}"
        row = {"ordinal": 0, "tier": tier, "experiment": C.EXP_ID, "arm": art["arm_group"],
               "arm_group": art["arm_group"], "object_key": art["object_key"], "case_id": art["case_id"],
               "aug_variant": variant, "aug_translation": art.get("aug_translation", ""),
               "aug_rotation_rad": art.get("aug_rotation_rad", ""), "base_variant": art["base_variant"],
               "base_target_task": art["base_target_task"], "target_task": aug_task,
               "target_scene": art["target_scene"], "trajectory": art["trajectory"],
               "trajectory_sha256": art["trajectory_sha256"], "contact_mask": mask,
               "contact_mask_sha256": C.sha256(mask) if mask and C.repo_path(mask).is_file() else "",
               "override_id": base_yaml.stem, "override_path": C.rel(base_yaml),
               "override_sha256": C.sha256(base_yaml), "base_scene_act": art["base_scene_act"],
               "base_scene_sha256": art["base_scene_sha256"], "scene_act": art["scene_act"],
               "scene_name": art["scene_name"], "effective_scene_sha256": art["effective_scene_sha256"],
               "extra_overrides": extra, "cem_samples": budget["num_samples"],
               "cem_opt_steps": budget["max_num_iterations"], "cem_seed": budget["seed"],
               "execution_mode": "production", "status": "", "failure_mode": "", "gpu_id": "",
               "wall_min": "", "updated_at": C.now(), "effective_retarget_variant": C.RETARGET_VARIANT,
               "gravcomp": art.get("gravcomp", ""), "object_geom_count": art.get("object_geom_count", ""),
               "compiled_robot_object_pair_count": art.get("compiled_robot_object_pair_count", ""),
               "approach_trans_offset_m_max": art.get("approach_trans_offset_m_max", ""),
               "endpoint_trans_offset_m": art.get("endpoint_trans_offset_m", ""),
               "weak_trans": art.get("weak_trans", "0"), "orig_result_npz": "", **artifact_paths(vid)}
        for label, key in (("scene", "scene_act"), ("trajectory", "trajectory"), ("target_scene", "target_scene")):
            if not C.repo_path(row[key]).is_file():
                blockers.append(f"{vid}:missing:{label}")
        if not mask or not C.repo_path(mask).is_file():
            blockers.append(f"{vid}:missing:contact_mask")
        rows.append(row)
    objects = sorted({r["object_key"] for r in rows})
    oi = {o: i for i, o in enumerate(objects)}
    vi = {"trans0": 0, "trans1": 1, "trans2": 2}
    rows.sort(key=lambda r: (C.TIER_RANK[r["tier"]], (oi[r["object_key"]] + vi[r["aug_variant"]]) % max(len(objects), 1),
                             oi[r["object_key"]], r["case_id"], r["aug_variant"]))
    for i, r in enumerate(rows, 1):
        r["ordinal"] = i
    for m in blockers:
        print(f"[BLOCK] {m}")
    C.write_tsv(C.PRIORITY_MANIFEST, rows, C.FIELDS)
    C.write_tsv(C.AUTHORITY_TSV, rows, C.FIELDS)
    by_obj: dict[str, int] = {}
    for r in rows:
        by_obj[r["object_key"]] = by_obj.get(r["object_key"], 0) + 1
    print(f"\n{len(rows)} queue rows -> {C.rel(C.PRIORITY_MANIFEST)}  by_object {by_obj}  blockers {len(blockers)}")
    if freeze and not blockers:
        C.write_tsv(C.FROZEN_MANIFEST, rows, C.FIELDS)
        pairs = sorted(f"{r['case_id']}|{r['aug_variant']}" for r in rows)
        C.write_json(C.FREEZE_JSON, {"experiment": C.EXP_ID, "run_id": C.RUN_ID, "frozen_at": C.now(),
                                     "n_rows": len(rows), "frozen_manifest": C.rel(C.FROZEN_MANIFEST),
                                     "frozen_manifest_sha256": C.sha256(C.FROZEN_MANIFEST),
                                     "priority_manifest": C.rel(C.PRIORITY_MANIFEST),
                                     "case_variant_set_sha256": C.sha256_text("\n".join(pairs)),
                                     "budget": budget})
        print(f"frozen: {len(rows)} rows, set sha {C.sha256_text(chr(10).join(pairs))[:12]}")
    elif freeze:
        print("NOT frozen: blockers present")
    return 1 if blockers else 0


# ==========================================================================
# main
# ==========================================================================
def _merge_artifacts(new_rows: list[dict[str, Any]]) -> None:
    prior = C.read_tsv(C.ARTIFACTS_TSV) if C.ARTIFACTS_TSV.is_file() else []
    fresh = {(r["case_id"], r["aug_variant"]) for r in new_rows}
    merged = [r for r in prior if (r["case_id"], r["aug_variant"]) not in fresh] + new_rows
    merged.sort(key=lambda r: (r["object_key"], r["case_id"], r["aug_variant"]))
    fields: list[str] = []
    for r in merged:
        for k in r:
            if k not in fields:
                fields.append(k)
    C.write_tsv(C.ARTIFACTS_TSV, merged, fields)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["seed", "retarget", "build", "manifest", "all"], default="all")
    ap.add_argument("--objects", default="")
    ap.add_argument("--cases", default="")
    ap.add_argument("--max-workers", type=int, default=5)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--freeze", action="store_true")
    args = ap.parse_args()

    cases = C.load_cases(objects=args.objects)
    if args.cases:
        keep = {c.strip() for c in args.cases.split(",") if c.strip()}
        cases = [c for c in cases if c["case_id"] in keep]
    if not cases and args.stage != "manifest":
        raise SystemExit("no cases selected")

    if args.stage in ("seed", "all"):
        print(f"=== seed {len(cases)} cases ===")
        for c in cases:
            print(f"  [{seed_case(c):22s}] {c['case_id']}")

    if args.stage in ("retarget", "all"):
        print(f"=== retarget @0.1m (v1), {len(cases)} cases, {args.max_workers} groups parallel ===")
        groups: dict[str, list] = {}
        for c in cases:
            groups.setdefault(c["object_key"], []).append(c)

        def run_group(items):
            out = []
            for c in items:
                r = retarget_case(c)
                print(f"  [{r['status']:14s}] {r['case_id']} built={r.get('built', '-')}", flush=True)
                out.append(r)
            return out
        with futures.ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as pool:
            list(pool.map(run_group, groups.values()))

    if args.stage in ("build", "all"):
        print(f"=== build {len(cases)} cases ===")
        all_rows = []
        for c in cases:
            all_rows.extend(build_case(c, overwrite=args.overwrite))
        if all_rows:
            _merge_artifacts(all_rows)
        ok = [r for r in all_rows if r["status"].startswith("built")]
        print(f"built {len(ok)}/{len(all_rows)} -> {C.rel(C.ARTIFACTS_TSV)}")

    if args.stage in ("manifest", "all"):
        print("=== manifest ===")
        return build_manifest(freeze=args.freeze)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
