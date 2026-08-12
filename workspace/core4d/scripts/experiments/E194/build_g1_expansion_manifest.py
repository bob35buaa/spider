#!/usr/bin/env python3
"""Restore exact inputs, freeze authority, and build E194 G1 expansion manifests."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e194_g1_expansion_common as C  # noqa: E402

CREATE_SCENE = C.REPO / "workspace/core4d/data_preprocess/create_spider_scene_from_template.py"
PROCESS_CORE4D = C.REPO / "spider/process_datasets/core4d.py"
RESTORE_ROOT = C.RESULTS / "s6_downstream/evidence/g1_expansion/stage2b_restore"
PRIMARY_ARTIFACTS = ("scene.xml", "scene_act.xml", "scene_act_meta.json", "task_info.json", "0/trajectory_kinematic.npz")
FIELDS = [
    "ordinal", "case_id", "object_key", "arm", "source_exp", "execution_source", "reused_full",
    "retarget_variant_id", "target_task", "target_scene", "trajectory", "contact_mask",
    "override_id", "override_path", "override_sha256", "base_scene_act", "scene_act", "scene_name",
    "source_effective_scene_sha256", "effective_scene_sha256", "trajectory_sha256", "contact_mask_sha256",
    "extra_overrides", "kp_pos", "kp_rot", "gravcomp", "worker", "execution_profile", "assigned_gpu", "gpu_id",
    "canary_representative", "sentinel", "a0_track_obj_z_abs_err_cm_mean", "cem_samples", "cem_opt_steps", "cem_seed",
    "variant", "result_npz", "outdir_npz", "config_act", "video", "log",
    "status", "failure_mode", "execution_mode", "updated_at",
]


def signature(element: ET.Element) -> Any:
    return (element.tag, tuple(sorted(element.attrib.items())), (element.text or "").strip(), tuple(signature(child) for child in element))


def semantic_xml(path: Path) -> bytes:
    return ET.tostring(ET.parse(path).getroot())


def stage2b_index() -> dict[tuple[str, str], dict[str, str]]:
    out: dict[tuple[str, str], dict[str, str]] = {}
    for variant in ("omnirt_v1", "omnirt_v2"):
        manifest = C.REPO / f"workspace/core4d/results/E173/s3_retarget/{variant}/ref_fk/stage2b_manifest_{variant}_ref_fk.tsv"
        for row in C.read_tsv(manifest):
            if row.get("stage2b_status") == "pass":
                out[(row["case_id"], variant)] = row
    return out


def primary_state(task: Path) -> dict[str, bool]:
    return {name: (task / name).is_file() for name in PRIMARY_ARTIFACTS}


def run_logged(command: list[str], log: Path) -> None:
    done = subprocess.run(command, cwd=C.REPO, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as stream:
        stream.write("$ " + " ".join(command) + "\n" + done.stdout)
        if done.stdout and not done.stdout.endswith("\n"):
            stream.write("\n")
    if done.returncode:
        raise RuntimeError(f"Stage2b adapter failed ({done.returncode}); see {log}")


def replay_stage2b(row: dict[str, str], stage: dict[str, str], temporary_task: str, log: Path) -> Path:
    source = C.RESULTS.parent / f"E173/scene_snapshot/source_templates/{row['object_key']}_{stage['person']}/scene.xml"
    trimmed = C.repo_path(stage["trimmed_npz"])
    common = [
        sys.executable, str(CREATE_SCENE), "--source-scene", str(source), "--task", temporary_task,
        "--qpos", str(trimmed), "--data-id", stage.get("data_id", "0") or "0", "--date", stage["date"],
        "--seq", stage["seq"], "--person", stage["person"], "--object-name", stage["object_name"],
        "--object-model-rel", stage["object_model_rel"],
    ]
    run_logged(common, log)
    run_logged([sys.executable, str(PROCESS_CORE4D), "--source-npz", str(trimmed), "--task", temporary_task,
                "--data-id", stage.get("data_id", "0") or "0", "--no-show-viewer", "--no-save-video"], log)
    run_logged(common + ["--generate-scene-act"], log)
    return C.TASK_ROOT / temporary_task


def audit_stage2b(row: dict[str, str], stage: dict[str, str], task: Path) -> dict[str, Any]:
    trajectory = task / "0/trajectory_kinematic.npz"
    pristine = C.RESULTS.parent / f"E173/scene_snapshot/cem_sidecars/{row['case_id']}/scene_act.xml"
    with np.load(trajectory, allow_pickle=True) as generated, np.load(C.repo_path(stage["trimmed_npz"]), allow_pickle=True) as trimmed:
        keys_ok = all(key in generated.files for key in ("qpos", "qvel", "ctrl", "contact", "contact_pos"))
        qpos_exact = keys_ok and np.array_equal(generated["qpos"], trimmed["qpos"])
    scene = mujoco.MjModel.from_xml_path(str(task / "scene.xml"))
    scene_act = mujoco.MjModel.from_xml_path(str(task / "scene_act.xml"))
    payload = {
        "trajectory_sha256": C.sha256(trajectory),
        "trajectory_authority_sha_exact": C.sha256(trajectory) == row["trajectory_sha256"],
        "trimmed_qpos_exact": bool(qpos_exact),
        "scene_act_semantic_exact": semantic_xml(task / "scene_act.xml") == semantic_xml(pristine),
        "compile_pass": (scene.nq, scene.nv, scene.nu) == (43, 41, 29) and (scene_act.nq, scene_act.nv, scene_act.nu) == (42, 41, 35),
    }
    payload["pass"] = all(payload[key] for key in ("trajectory_authority_sha_exact", "trimmed_qpos_exact", "scene_act_semantic_exact", "compile_pass"))
    return payload


def restore_e173_missing(rows: list[dict[str, str]], *, apply: bool) -> list[dict[str, Any]]:
    index = stage2b_index()
    reports: list[dict[str, Any]] = []
    for row in rows:
        if row["source_exp"] != "E173":
            continue
        task = C.TASK_ROOT / row["target_task"]
        state = primary_state(task)
        stage = index[(row["case_id"], row["retarget_variant_id"])]
        runtime_complete = all(state.values())
        if runtime_complete:
            trajectory_exact = C.sha256(task / "0/trajectory_kinematic.npz") == row["trajectory_sha256"]
            scene = mujoco.MjModel.from_xml_path(str(task / "scene.xml"))
            scene_compile = (scene.nq, scene.nv, scene.nu) == (43, 41, 29)
            if not trajectory_exact or not scene_compile:
                raise ValueError(f"existing runtime input differs from E173 authority: {row['case_id']}")
            reports.append({"case_id": row["case_id"], "action": "preserved_runtime_authority_match",
                            "trajectory_authority_sha_exact": trajectory_exact, "scene_compile_pass": scene_compile, "pass": True})
            continue
        if not apply:
            reports.append({"case_id": row["case_id"], "action": "runtime_restore_required", "state": state, "pass": False})
            continue
        log = RESTORE_ROOT / "logs" / f"{row['case_id']}.log"
        with tempfile.TemporaryDirectory(prefix=f"__e194_g1_restore_{row['case_id']}_", dir=C.TASK_ROOT) as temporary:
            staged = replay_stage2b(row, stage, Path(temporary).name, log)
            audit = audit_stage2b(row, stage, staged)
            if not audit["pass"]:
                raise RuntimeError(f"exact Stage2b recovery failed: {row['case_id']} {audit}")
            task.mkdir(parents=True, exist_ok=True)
            (task / "0").mkdir(parents=True, exist_ok=True)
            for name in PRIMARY_ARTIFACTS:
                target = task / name
                if target.exists():
                    continue
                if name == "task_info.json":
                    meta = json.loads((staged / name).read_text(encoding="utf-8")); meta["task"] = row["target_task"]
                    target.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
                else:
                    shutil.copy2(staged / name, target)
        final = audit_stage2b(row, stage, task)
        reports.append({"case_id": row["case_id"], "action": "restored_missing_exact_stage2b", **final})
    C.write_tsv(RESTORE_ROOT / "restore_manifest.tsv", reports)
    C.write_json(RESTORE_ROOT / "restore_summary.json", {"created_at": C.now(), "rows": len(reports),
        "actions": dict(Counter(row["action"] for row in reports)), "pass": sum(bool(row.get("pass")) for row in reports)})
    return reports


def build_sidecar(base: Path, *, apply: bool) -> tuple[Path, str, str]:
    base_sha = C.sha256(base)
    tree = ET.parse(base); root = tree.getroot()
    objects = [body for body in root.iter("body") if body.get("name") == "object"]
    if len(objects) != 1 or objects[0].get("gravcomp") not in (None, "0", "0.0"):
        raise ValueError(f"unexpected object gravcomp in {base}")
    expected = ET.parse(base).getroot()
    next(body for body in expected.iter("body") if body.get("name") == "object").set("gravcomp", "1")
    objects[0].set("gravcomp", "1")
    if signature(root) != signature(expected):
        raise AssertionError(f"non-gravcomp XML drift: {base}")
    out = base.with_name(C.SIDECAR_FILE)
    if apply:
        if out.exists() and signature(ET.parse(out).getroot()) != signature(expected):
            raise FileExistsError(f"different sidecar exists: {out}")
        if not out.exists():
            ET.indent(tree, space="  "); tree.write(out, encoding="utf-8", xml_declaration=True)
        if signature(ET.parse(out).getroot()) != signature(expected):
            raise AssertionError(f"physical sidecar drift: {out}")
    return out, base_sha, C.sha256(out) if apply else ""


def a0_z_metrics(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    cached = {row["case_id"]: row for row in C.read_tsv(C.E173_Z_METRICS) if row.get("object_key") in {"box001", "box023"}}
    sys.path.insert(0, str(C.REPO / "workspace/core4d/scripts"))
    from eval.core.core_metrics import _table4_tracking_metrics, npz_qpos  # noqa: PLC0415
    output: list[dict[str, Any]] = []
    for row in rows:
        if row["case_id"] in cached:
            source = cached[row["case_id"]]
            value = float(source["track_obj_z_abs_err_cm_mean"]); frames = int(source["frames"])
        else:
            run_qpos, _ = npz_qpos(C.repo_path(row["outdir_npz"]))
            with np.load(C.repo_path(row["trajectory"]), allow_pickle=True) as archive:
                kin_qpos = np.asarray(archive["qpos"], dtype=np.float64)
            if kin_qpos.ndim == 3:
                kin_qpos = kin_qpos[:, 0, :]
            model = mujoco.MjModel.from_xml_path(str(C.repo_path(row["scene_act"])))
            metrics = _table4_tracking_metrics(run_qpos, kin_qpos, model)
            value = float(metrics["track_obj_z_abs_err_cm_mean"])
            frames = min(len(run_qpos), len(kin_qpos))
        output.append({"case_id": row["case_id"], "object_key": row["object_key"], "source_exp": row["source_exp"],
                       "frames": frames, "track_obj_z_abs_err_cm_mean": value})
    C.write_tsv(C.A0_METRICS, output)
    return output


def selected_cases(metrics: list[dict[str, Any]]) -> tuple[dict[str, str], set[str]]:
    canary: dict[str, str] = {}; sentinels: set[str] = set()
    for object_key in C.OBJECT_ORDER:
        group = sorted((row for row in metrics if row["object_key"] == object_key), key=lambda row: row["case_id"])
        worst = min(group, key=lambda row: (-float(row["track_obj_z_abs_err_cm_mean"]), row["case_id"]))
        values = sorted(float(row["track_obj_z_abs_err_cm_mean"]) for row in group)
        n = len(values); median = values[n // 2] if n % 2 else (values[n // 2 - 1] + values[n // 2]) / 2
        middle = min(group, key=lambda row: (abs(float(row["track_obj_z_abs_err_cm_mean"]) - median), row["case_id"]))
        canary[object_key] = worst["case_id"]; sentinels.update((worst["case_id"], middle["case_id"]))
    if len(sentinels) != 6:
        raise ValueError(f"expected 6 unique sentinels, got {sentinels}")
    return canary, sentinels


def artifact_paths(case_id: str, stage: str, worker: str) -> dict[str, str]:
    canary = stage == "canary"; suffix = f"_{worker.replace('-', '_')}" if canary else ""
    variant = f"E194_{case_id}_G1_expansion" + (f"_canary{suffix}" if canary else "")
    root = f"workspace/core4d/results/E194/s6_downstream/cem/{'canary_g1_expansion' if canary else 'full_g1_expansion'}"
    return {"variant": variant, "result_npz": f"{root}/{variant}.npz", "outdir_npz": f"{root}/{variant}_outdir/trajectory_mjwp_act.npz",
            "config_act": f"{root}/{variant}_outdir/config_act.yaml", "video": f"workspace/core4d/results/E194/s6_downstream/render/{'canary_g1_expansion' if canary else 'full_g1_expansion'}/{variant}.mp4",
            "log": f"logs/E194/cem/{'canary_g1_expansion' if canary else 'full_g1_expansion'}/{variant}.log"}


def make_row(src: dict[str, str], sidecar: Path, source_sha: str, effective_sha: str, a0_z: float, *, stage: str, worker: str, sentinel: bool, canary_rep: bool) -> dict[str, Any]:
    row: dict[str, Any] = {
        "ordinal": 0, "case_id": src["case_id"], "object_key": src["object_key"], "arm": C.ARM,
        "source_exp": src["source_exp"], "execution_source": src.get("execution_source", src["source_exp"]), "reused_full": src.get("reused_full", "false"),
        "retarget_variant_id": src["retarget_variant_id"], "target_task": src["target_task"], "target_scene": C.rel(C.TASK_ROOT / src["target_task"] / "scene.xml"),
        "trajectory": C.rel(C.TASK_ROOT / src["target_task"] / "0/trajectory_kinematic.npz"), "contact_mask": C.rel(src["contact_mask"]),
        "override_id": src["override_id"], "override_path": C.rel(src["override_path"]), "override_sha256": C.sha256(src["override_path"]),
        "base_scene_act": C.rel(src["scene_act"]), "scene_act": C.rel(sidecar), "scene_name": C.SCENE_NAME,
        "source_effective_scene_sha256": source_sha, "effective_scene_sha256": effective_sha,
        "trajectory_sha256": C.sha256(C.TASK_ROOT / src["target_task"] / "0/trajectory_kinematic.npz"), "contact_mask_sha256": C.sha256(src["contact_mask"]),
        "extra_overrides": f"scene_name={C.SCENE_NAME}", "kp_pos": C.KP_POS, "kp_rot": C.KP_ROT, "gravcomp": C.GRAVCOMP,
        "worker": worker, "execution_profile": worker, "assigned_gpu": C.WORKER_GPU[worker], "gpu_id": "",
        "canary_representative": canary_rep, "sentinel": sentinel, "a0_track_obj_z_abs_err_cm_mean": a0_z,
        "cem_samples": C.CANARY_SAMPLES if stage == "canary" else C.FULL_SAMPLES,
        "cem_opt_steps": C.CANARY_OPT_STEPS if stage == "canary" else C.FULL_OPT_STEPS, "cem_seed": C.CEM_SEED,
        "status": "READY_FOR_CANARY" if stage == "canary" else "READY_FOR_FULL", "failure_mode": "",
        "execution_mode": stage, "updated_at": C.now(),
    }
    row.update(artifact_paths(src["case_id"], stage, worker)); return row


def build(*, apply: bool, snapshot: bool, restore_missing: bool) -> int:
    sources = C.source_rows()
    restore = restore_e173_missing(sources, apply=restore_missing)
    if any(not row.get("pass") for row in restore):
        raise RuntimeError("E173 runtime inputs are not fully restored; run with --restore-missing")
    metrics = a0_z_metrics(sources); z_by_case = {row["case_id"]: float(row["track_obj_z_abs_err_cm_mean"]) for row in metrics}
    canary_case, sentinel_cases = selected_cases(metrics)
    authority: list[dict[str, Any]] = []; full: list[dict[str, Any]] = []; canary: list[dict[str, Any]] = []
    snapshots = C.RESULTS / "scene_snapshot/g1_expansion"
    object_index = Counter()
    for src in sources:
        task = C.TASK_ROOT / src["target_task"]
        for field, path, expected in (("trajectory", task / "0/trajectory_kinematic.npz", src["trajectory_sha256"]),
                                      ("scene", C.repo_path(src["scene_act"]), src["effective_scene_sha256"]),
                                      ("override", C.repo_path(src["override_path"]), src["override_sha256"]),
                                      ("contact", C.repo_path(src["contact_mask"]), src["contact_mask_sha256"])):
            actual = C.sha256(path)
            if actual != expected:
                raise ValueError(f"{src['case_id']} {field} SHA drift: {actual} != {expected}")
        sidecar, source_sha, effective_sha = build_sidecar(C.repo_path(src["scene_act"]), apply=apply)
        if snapshot and apply:
            snap = snapshots / src["case_id"]; snap.mkdir(parents=True, exist_ok=True)
            shutil.copy2(C.repo_path(src["scene_act"]), snap / Path(src["scene_act"]).name); shutil.copy2(sidecar, snap / sidecar.name)
        worker = C.worker_for_object_index(object_index[src["object_key"]]); object_index[src["object_key"]] += 1
        full.append(make_row(src, sidecar, source_sha, effective_sha, z_by_case[src["case_id"]], stage="full", worker=worker,
                             sentinel=src["case_id"] in sentinel_cases, canary_rep=src["case_id"] == canary_case[src["object_key"]]))
        if src["case_id"] == canary_case[src["object_key"]]:
            for canary_worker in C.WORKERS:
                canary.append(make_row(src, sidecar, source_sha, effective_sha, z_by_case[src["case_id"]], stage="canary", worker=canary_worker,
                                        sentinel=False, canary_rep=True))
        authority.append({"case_id": src["case_id"], "object_key": src["object_key"], "source_exp": src["source_exp"],
            "execution_source": src.get("execution_source", src["source_exp"]), "reused_full": src.get("reused_full", "false"),
            "retarget_variant_id": src["retarget_variant_id"], "target_task": src["target_task"], "trajectory": C.rel(task / "0/trajectory_kinematic.npz"),
            "trajectory_sha256": C.sha256(task / "0/trajectory_kinematic.npz"), "base_scene_act": C.rel(src["scene_act"]),
            "source_effective_scene_sha256": source_sha, "g1_scene_act": C.rel(sidecar), "g1_scene_sha256": effective_sha,
            "override_path": C.rel(src["override_path"]), "override_sha256": C.sha256(src["override_path"]),
            "contact_mask": C.rel(src["contact_mask"]), "contact_mask_sha256": C.sha256(src["contact_mask"]),
            "a0_track_obj_z_abs_err_cm_mean": z_by_case[src["case_id"]], "worker": worker,
            "canary_representative": src["case_id"] == canary_case[src["object_key"]], "sentinel": src["case_id"] in sentinel_cases})
    for ordinal, row in enumerate(full, 1): row["ordinal"] = ordinal
    for ordinal, row in enumerate(canary, 1): row["ordinal"] = ordinal
    C.validate_worker_balance(full)
    if len(full) != 72 or len(canary) != 9 or sum(C.truth(row["sentinel"]) for row in full) != 6:
        raise ValueError("manifest cardinality contract failed")
    sentinels = [dict(row) for row in full if C.truth(row["sentinel"])]
    C.write_tsv(C.SOURCE_AUTHORITY, authority); C.write_json(C.SOURCE_AUTHORITY.with_suffix(".json"), authority)
    C.write_tsv(C.FULL_MANIFEST, full, FIELDS); C.write_tsv(C.CANARY_MANIFEST, canary, FIELDS); C.write_tsv(C.SENTINEL_MANIFEST, sentinels, FIELDS)
    summary = {"created_at": C.now(), "apply": apply, "snapshot": snapshot, "restored": dict(Counter(row["action"] for row in restore)),
        "objects": dict(Counter(row["object_key"] for row in full)), "workers": dict(Counter(row["worker"] for row in full)),
        "full_rows": len(full), "canary_rows": len(canary), "sentinel_rows": len(sentinels), "canary_cases": canary_case,
        "sidecar": C.SIDECAR_FILE, "budget_full": [C.FULL_SAMPLES, C.FULL_OPT_STEPS], "budget_canary": [C.CANARY_SAMPLES, C.CANARY_OPT_STEPS]}
    C.write_json(C.MANIFEST_DIR / "g1_expansion_build_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True)); return 0


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--apply", action="store_true"); parser.add_argument("--snapshot", action="store_true")
    parser.add_argument("--restore-missing", action="store_true", help="exactly replay missing E173 Stage2b primary inputs")
    args = parser.parse_args(); return build(apply=args.apply, snapshot=args.snapshot, restore_missing=args.restore_missing)


if __name__ == "__main__":
    raise SystemExit(main())
