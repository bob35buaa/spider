#!/usr/bin/env python3
"""Build E194 gravity-compensation manifests + per-case gravcomp scene sidecars.

For each of the 15 cases this:
  1. reads the A0 baseline row from E172/E173 cem_full_manifest.tsv (the single
     source of truth for override / base sidecar / trajectory / gpu);
  2. builds the G1/G3 gravcomp sidecar = base PRG sidecar with `gravcomp="1"`
     injected on the object body, saved next to the base as
     scene_act_E194_rubberHull_PRG_gravcomp.xml.  A tree_signature check asserts
     the ONLY difference from the base is that single attribute;
  3. emits three arm rows (G1/G2/G3) whose sole difference from A0 is the extra
     run_mjwp CLI tokens in `extra_overrides` (see e194_common.arm_extra_overrides).

Outputs (45 Full rows, 9 canary rows):
  workspace/core4d/results/E194/s6_downstream/manifests/cem_full_manifest.tsv
  workspace/core4d/results/E194/s6_downstream/manifests/cem_canary_manifest.tsv
  workspace/core4d/results/E194/s6_downstream/manifests/a0_baseline_manifest.tsv
  workspace/core4d/results/E194/preflight/scene_audit.tsv

Usage:
  .venv/bin/python .../build_gravcomp_manifest.py --apply --snapshot
  .venv/bin/python .../build_gravcomp_manifest.py            # dry-run (no writes to XML)
"""

from __future__ import annotations

import argparse
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e194_common as C  # noqa: E402


def tree_signature(element: ET.Element) -> Any:
    return (
        element.tag,
        tuple(sorted(element.attrib.items())),
        (element.text or "").strip(),
        tuple(tree_signature(child) for child in element),
    )


def build_gravcomp_sidecar(base_scene: Path, *, apply: bool) -> tuple[Path, str, str]:
    """Return (out_path, base_sha, effective_sha); inject object-body gravcomp=1."""
    if not base_scene.is_file():
        raise FileNotFoundError(base_scene)
    tree = ET.parse(base_scene)
    root = tree.getroot()
    objects = [b for b in root.iter("body") if b.get("name") == "object"]
    if len(objects) != 1:
        raise ValueError(f"expected exactly one object body, found {len(objects)} in {base_scene}")
    obj = objects[0]
    if obj.get("gravcomp") not in (None, "0"):
        raise ValueError(f"object body already has gravcomp={obj.get('gravcomp')!r} in {base_scene}")

    out_path = base_scene.with_name(C.GRAVCOMP_SIDECAR_FILE)

    # Build the expected signature: base tree with ONLY the object body gaining
    # gravcomp="1".  We diff against this to guarantee no other field moved.
    expected = ET.parse(base_scene).getroot()
    for body in expected.iter("body"):
        if body.get("name") == "object":
            body.set("gravcomp", "1")

    obj.set("gravcomp", "1")
    if tree_signature(root) != tree_signature(expected):
        raise AssertionError(f"gravcomp injection changed more than object gravcomp: {base_scene}")

    if apply:
        if out_path.exists():
            existing = tree_signature(ET.parse(out_path).getroot())
            if existing != tree_signature(root):
                raise FileExistsError(f"different gravcomp sidecar already exists: {out_path}")
        else:
            ET.indent(tree, space="  ")
            tree.write(out_path, encoding="utf-8", xml_declaration=True)
        # Re-verify the physical file matches base + only-gravcomp.
        physical = ET.parse(out_path).getroot()
        if tree_signature(physical) != tree_signature(expected):
            raise AssertionError(f"physical gravcomp sidecar diverges from base+gravcomp: {out_path}")
        base_sha, eff_sha = C.sha256(base_scene), C.sha256(out_path)
    else:
        base_sha, eff_sha = C.sha256(base_scene), ""
    return out_path, base_sha, eff_sha


def artifact_paths(case_id: str, arm: str, stage: str) -> dict[str, str]:
    suffix = "canary" if stage == "canary" else "full"
    variant = f"E194_{case_id}_{arm}" + ("_canary" if stage == "canary" else "")
    root = f"workspace/core4d/results/E194/s6_downstream/cem/{stage}"
    return {
        "variant": variant,
        "result_npz": f"{root}/{variant}.npz",
        "outdir_npz": f"{root}/{variant}_outdir_{suffix}/trajectory_mjwp_act.npz",
        "config_act": f"{root}/{variant}_outdir_{suffix}/config_act.yaml",
        "video": f"workspace/core4d/results/E194/s6_downstream/render/{stage}/{variant}_{suffix}.mp4",
        "log": f"logs/E194/cem/{stage}/{variant}.log",
    }


FIELDS = [
    "ordinal", "case_id", "object_key", "arm", "source_exp", "retarget_variant_id",
    "target_task", "target_scene", "trajectory", "contact_mask",
    "override_id", "override_path", "override_sha256",
    "scene_act", "scene_name", "base_scene_sha256", "effective_scene_sha256",
    "trajectory_sha256", "contact_mask_sha256",
    "extra_overrides", "kp_pos", "gravcomp",
    "assigned_gpu", "gpu_id",
    "cem_samples", "cem_opt_steps", "cem_seed",
    "variant", "result_npz", "outdir_npz", "config_act", "video", "log",
    "status", "failure_mode", "execution_mode", "updated_at",
]


def make_row(src: dict[str, str], arm: str, stage: str,
             base_scene: str, gravcomp_scene: str,
             base_sha: str, eff_base_sha: str, eff_grav_sha: str) -> dict[str, Any]:
    spec = C.ARMS[arm]
    case_id = src["case_id"]
    if spec["gravcomp"]:
        scene_act, scene_name = gravcomp_scene, C.GRAVCOMP_SCENE_NAME
        base_scene_sha, eff_sha = base_sha, eff_grav_sha
    else:
        scene_act, scene_name = base_scene, src["scene_name"]
        base_scene_sha, eff_sha = base_sha, eff_base_sha
    samples = C.CANARY_SAMPLES if stage == "canary" else C.FULL_SAMPLES
    steps = C.CANARY_OPT_STEPS if stage == "canary" else C.FULL_OPT_STEPS
    row: dict[str, Any] = {
        "ordinal": 0,
        "case_id": case_id,
        "object_key": C.object_key_of(case_id),
        "arm": arm,
        "source_exp": src["_source_exp"],
        "retarget_variant_id": src.get("retarget_variant_id", ""),
        "target_task": src["target_task"],
        "target_scene": src["target_scene"],
        "trajectory": src["trajectory"],
        "contact_mask": src["contact_mask"],
        "override_id": src["override_id"],
        "override_path": src["override_path"],
        "override_sha256": C.sha256(src["override_path"]) if C.repo_path(src["override_path"]).is_file() else "",
        "scene_act": C.rel(scene_act),
        "scene_name": scene_name,
        "base_scene_sha256": base_scene_sha,
        "effective_scene_sha256": eff_sha,
        "trajectory_sha256": C.sha256(src["trajectory"]) if C.repo_path(src["trajectory"]).is_file() else "",
        "contact_mask_sha256": C.sha256(src["contact_mask"]) if C.repo_path(src["contact_mask"]).is_file() else "",
        "extra_overrides": C.arm_extra_overrides(arm),
        "kp_pos": spec["kp"],
        "gravcomp": spec["gravcomp"],
        "assigned_gpu": src["assigned_gpu"],
        "gpu_id": "",
        "cem_samples": samples,
        "cem_opt_steps": steps,
        "cem_seed": C.CEM_SEED,
        "status": "READY_FOR_CANARY" if stage == "canary" else "READY_FOR_FULL",
        "failure_mode": "",
        "execution_mode": "canary" if stage == "canary" else "production",
        "updated_at": C.now(),
    }
    row.update(artifact_paths(case_id, arm, stage))
    return row


def build(*, apply: bool, snapshot: bool) -> int:
    sources = C.load_source_rows()
    full_rows: list[dict[str, Any]] = []
    canary_rows: list[dict[str, Any]] = []
    a0_rows: list[dict[str, Any]] = []
    scene_audit: list[dict[str, Any]] = []

    for case_id in C.all_case_ids():
        src = sources[case_id]
        base_scene = C.repo_path(src["scene_act"])
        out_path, base_sha, eff_grav_sha = build_gravcomp_sidecar(base_scene, apply=apply)
        eff_base_sha = base_sha  # G2 uses the base sidecar verbatim
        scene_audit.append({
            "case_id": case_id,
            "base_scene": C.rel(base_scene),
            "base_scene_sha256": base_sha,
            "gravcomp_scene": C.rel(out_path),
            "gravcomp_scene_sha256": eff_grav_sha,
            "applied": "true" if apply else "false",
        })
        if snapshot and apply:
            snap = C.RESULTS / "scene_snapshot" / case_id
            snap.mkdir(parents=True, exist_ok=True)
            shutil.copy2(base_scene, snap / base_scene.name)
            shutil.copy2(out_path, snap / out_path.name)

        # A0 baseline pointer row (for eval): reuse E172/E173 landed results.
        a0_rows.append({
            "case_id": case_id, "object_key": C.object_key_of(case_id), "arm": "A0",
            "source_exp": src["_source_exp"],
            "result_npz": src.get("result_npz", ""), "outdir_npz": src.get("outdir_npz", ""),
            "config_act": src.get("config_act", ""), "video": src.get("video", ""),
            "scene_act": C.rel(base_scene), "scene_name": src["scene_name"],
            "override_id": src["override_id"],
        })

        for arm in C.RUN_ARMS:
            full_rows.append(make_row(src, arm, "full", C.rel(base_scene), C.rel(out_path),
                                      base_sha, eff_base_sha, eff_grav_sha))
            if case_id in C.CANARY_CASES:
                canary_rows.append(make_row(src, arm, "canary", C.rel(base_scene), C.rel(out_path),
                                            base_sha, eff_base_sha, eff_grav_sha))

    for ordinal, row in enumerate(full_rows, 1):
        row["ordinal"] = ordinal
        # Full CEM is restricted to GPUs 4-7 (user directive); round-robin for balance.
        row["assigned_gpu"] = C.FULL_GPU_POOL[(ordinal - 1) % len(C.FULL_GPU_POOL)]
    for ordinal, row in enumerate(canary_rows, 1):
        row["ordinal"] = ordinal

    if len(full_rows) != 45:
        raise SystemExit(f"expected 45 Full rows, got {len(full_rows)}")
    if len(canary_rows) != 9:
        raise SystemExit(f"expected 9 canary rows (3 cases x 3 arms), got {len(canary_rows)}")

    man = C.RESULTS / "s6_downstream/manifests"
    C.write_tsv(man / "cem_full_manifest.tsv", full_rows, FIELDS)
    C.write_tsv(man / "cem_canary_manifest.tsv", canary_rows, FIELDS)
    C.write_tsv(man / "a0_baseline_manifest.tsv", a0_rows, list(a0_rows[0]))
    C.write_tsv(C.RESULTS / "preflight/scene_audit.tsv", scene_audit, list(scene_audit[0]))

    summary = {
        "created_at": C.now(), "applied": apply, "snapshot": snapshot,
        "method_id": C.E194_METHOD_ID, "n_cases": C.N_CASES,
        "full_rows": len(full_rows), "canary_rows": len(canary_rows),
        "arms": C.RUN_ARMS, "kp_baseline": C.KP_BASELINE, "kp_hard": C.KP_HARD, "rot_gain": C.ROT_GAIN,
        "gravcomp_sidecar": C.GRAVCOMP_SIDECAR_FILE,
    }
    C.write_json(C.RESULTS / "s6_downstream/manifests/build_summary.json", summary)
    print(f"[build] apply={apply} snapshot={snapshot} full={len(full_rows)} canary={len(canary_rows)} a0={len(a0_rows)}")
    for row in scene_audit:
        print(f"  {row['case_id']}: base={row['base_scene_sha256'][:12]} grav={row['gravcomp_scene_sha256'][:12] or '(dry)'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="write gravcomp sidecars to disk")
    parser.add_argument("--snapshot", action="store_true", help="copy base+gravcomp sidecars into results/E194/scene_snapshot")
    args = parser.parse_args()
    return build(apply=args.apply, snapshot=args.snapshot)


if __name__ == "__main__":
    raise SystemExit(main())
