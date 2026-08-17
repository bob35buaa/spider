#!/usr/bin/env python3
"""Build/verify the per-arm scene_act sidecars for E200 (reuses E199 aug tasks).

For every feasible E199 full-scale trans variant:
  * prg_g1a2 -> build scene_act_E199_rubberHull_PRG_gravcomp.xml (object gravcomp
               0->1 single-variable diff of the E199 PRG sidecar) + snapshot it.
  * noprg    -> verify scene_act_E199_rubberHull.xml (rubber-hull, no-PRG) exists
               (already built + snapshotted by E199; nothing new to write).

Writes a per-arm scene registry TSV (path + sha256 of each arm scene). Idempotent
and resume-safe: existing gravcomp sidecars are re-verified, not rewritten.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e200_common as C  # noqa: E402


def aug_task_dir(row: dict[str, str]) -> Path:
    # every scene lives in the aug task dir; derive it from the PRG scene path.
    return C.repo_path(row["scene_act"]).parent


def build_for_arm(arm: str, rows: list[dict[str, str]], *, overwrite: bool) -> list[dict[str, str]]:
    registry: list[dict[str, str]] = []
    snap_root = C.SNAPSHOT_DIR / ("g1_sidecars" if arm == "prg_g1a2" else "noprg_scenes")
    built = verified = 0
    for row in rows:
        task_dir = aug_task_dir(row)
        prg_scene = task_dir / f"{C.PRG_SCENE}.xml"
        if arm == "prg_g1a2":
            existed = (task_dir / f"{C.GRAVCOMP_SCENE}.xml").is_file()
            scene = C.build_gravcomp_sidecar(prg_scene, overwrite=overwrite)
            built += int(not existed)
            verified += int(existed and not overwrite)
        else:  # noprg
            scene = task_dir / f"{C.NOPRG_SCENE}.xml"
            if not scene.is_file():
                raise FileNotFoundError(f"missing rubber-hull (no-PRG) scene: {scene}")
            verified += 1
        # snapshot the exact scene used (git HEAD + sha256 recorded in manifest.txt below)
        snap_dir = snap_root / task_dir.name
        snap_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(C.repo_path(scene), snap_dir / Path(scene).name)
        registry.append({
            "arm": arm, "object_key": row["object_key"], "case_id": row["case_id"],
            "aug_variant": row["aug_variant"], "target_task": row["target_task"],
            "scene_act": C.rel(scene), "scene_name": C.arm_scene_name(arm),
            "scene_sha256": C.sha256(scene),
            "base_prg_scene": C.rel(prg_scene) if arm == "prg_g1a2" else "",
        })
    _write_snapshot_manifest(snap_root, registry)
    print(f"[{arm}] scenes: {built} built, {verified} verified, {len(registry)} total")
    return registry


def _write_snapshot_manifest(snap_root: Path, registry: list[dict[str, str]]) -> None:
    import subprocess
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=C.REPO, text=True,
                          capture_output=True, check=False).stdout.strip()
    lines = [f"git_head\t{head}"]
    for r in registry:
        lines.append(f"{r['scene_sha256']}\t{r['scene_act']}")
    snap_root.mkdir(parents=True, exist_ok=True)
    (snap_root / "manifest.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=",".join(C.ARMS))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    rows = C.load_aug_rows()
    print(f"loaded {len(rows)} feasible E199 aug rows "
          f"({len({r['case_id'] for r in rows})} cases)")
    for arm in arms:
        if arm not in C.ARMS:
            raise SystemExit(f"unknown arm {arm}; expected {C.ARMS}")
        registry = build_for_arm(arm, rows, overwrite=args.overwrite)
        C.write_tsv(C.scene_registry_path(arm), registry)
        print(f"[{arm}] registry -> {C.rel(C.scene_registry_path(arm))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
