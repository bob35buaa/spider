#!/usr/bin/env python3
"""Build the 22-row E209 G1 CEM manifest for the local 8-GPU scheduler.

Column contract is E199's ``run_local_priority_queue.py`` (tier/arm/aug_variant
are scheduler sort keys; E209 has a single tier and arm so they are constants).

Every input path is carried with its sha256 so the scheduler's ``input_failures``
preflight refuses to launch against a drifted scene / trajectory / mask /
override.  Reference inputs are additionally asserted byte-identical to what
E206's PRG arm consumed -- that is what makes "E209 vs E206 PRG is
single-variable" true at the *data* level, not only at the config level (C0).

The mask path is taken from E206's own resolved ``config_act.yaml`` rather than
from the delivery TSV: the TSV records an absolute /mnt path while the run
consumed the workspace-relative symlink.  They hash the same, but the config is
what the process actually opened, so it is the honest reference.

Usage:
    .venv/bin/python workspace/core4d/scripts/experiments/E209/build_manifest.py
    ... --stage smoke --limit 1 --samples 64 --opt-steps 4 --cases <id>
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[5]
for _d in ("E209", "E199"):
    sys.path.insert(0, str(REPO / "workspace/core4d/scripts/experiments" / _d))

import e199_common as E199  # noqa: E402
import e209_common as C  # noqa: E402

FIELDS = [
    # `variant` and `preferred_pool` are consumed by E168/render_a100_cem_videos.py;
    # the scheduler ignores them.
    "case_id", "object_key", "person", "experiment", "arm", "tier", "aug_variant",
    "variant", "preferred_pool",
    "target_task", "target_scene", "scene_act", "scene_name", "effective_scene_sha256",
    "trajectory", "trajectory_sha256", "contact_mask", "contact_mask_sha256",
    "override_id", "override_path", "override_sha256",
    "baseline_outdir_npz", "baseline_scene_sha256",
    "cem_samples", "cem_opt_steps", "cem_seed", "execution_mode",
    "result_npz", "outdir_npz", "config_act", "video", "log",
    "status", "failure_mode", "gpu_id", "updated_at",
]


def rel(path: Path) -> str:
    return path.relative_to(REPO).as_posix()


def baseline_config(case_id: str) -> dict:
    path = C.baseline_out_dir(case_id) / "config_act.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"{case_id}: missing E206 PRG config_act.yaml: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def build(stage: str, cases: list[str], samples: int, opt_steps: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for source in C.sources():
        case_id = source["case_id"]
        if case_id not in cases:
            continue
        base_cfg = baseline_config(case_id)

        scene = C.scene_path(source)
        base_scene = C.base_scene_path(source)
        trajectory = C.kinematic_npz(source)
        contact = REPO / base_cfg["contact_hdmi_mask_path"]
        override = C.override_path(case_id)
        target_scene = C.task_dir(source) / "scene.xml"
        for label, path in (
            ("scene_act", scene), ("base_scene", base_scene), ("trajectory", trajectory),
            ("contact_mask", contact), ("override", override), ("target_scene", target_scene),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"{case_id}: missing {label}: {path}")

        # --- C0: reference inputs identical to what E206 PRG consumed --------
        traj_sha, mask_sha = E199.sha256(trajectory), E199.sha256(contact)
        if traj_sha != source["trajectory_sha256"]:
            raise ValueError(f"{case_id}: trajectory drifted from E206 delivery")
        if mask_sha != source["contact_mask_sha256"]:
            raise ValueError(f"{case_id}: contact mask drifted from E206 delivery")
        base_scene_sha = E199.sha256(base_scene)
        if base_scene_sha != source["scene_act_sha256"]:
            raise ValueError(f"{case_id}: E206 PRG scene drifted from E206 delivery")
        # The baseline arm must be the E206 PRG scene, not something re-pointed.
        if base_cfg.get("scene_name") != C.BASE_SCENE:
            raise ValueError(
                f"{case_id}: E206 baseline scene_name={base_cfg.get('scene_name')!r}"
                f" != {C.BASE_SCENE!r}"
            )

        out = C.out_dir(case_id, stage)
        rows.append({
            "case_id": case_id,
            "object_key": source["object_key"],
            "person": source.get("person", ""),
            "experiment": C.EXP,
            "arm": C.ARM_TAG,
            "tier": "P0",
            "aug_variant": "orig",
            "variant": f"{C.EXP}_{case_id}_{C.ARM_TAG}",
            "preferred_pool": "local",
            "target_task": C.target_task(source),
            "target_scene": rel(target_scene),
            "scene_act": rel(scene),
            "scene_name": C.SCENE,
            "effective_scene_sha256": E199.sha256(scene),
            "trajectory": rel(trajectory),
            "trajectory_sha256": traj_sha,
            "contact_mask": rel(contact),
            "contact_mask_sha256": mask_sha,
            "override_id": C.override_id(case_id),
            "override_path": rel(override),
            "override_sha256": E199.sha256(override),
            "baseline_outdir_npz": rel(C.baseline_npz(case_id)),
            "baseline_scene_sha256": base_scene_sha,
            "cem_samples": str(samples),
            "cem_opt_steps": str(opt_steps),
            "cem_seed": str(C.CEM_SEED),
            "execution_mode": "production" if stage == "full" else stage,
            "result_npz": rel(out / f"{C.EXP}_{case_id}_{C.ARM_TAG}.npz"),
            "outdir_npz": rel(C.result_npz(case_id, stage)),
            "config_act": rel(C.config_act(case_id, stage)),
            "video": rel(C.S6_DIR / f"render/{stage}/{C.EXP}_{case_id}_{C.ARM_TAG}.mp4"),
            "log": f"logs/{C.EXP}/cem/{stage}/{C.EXP}_{case_id}_{C.ARM_TAG}.log",
            "status": "READY_FOR_FULL",
            "failure_mode": "",
            "gpu_id": "",
            "updated_at": E199.now(),
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="full", choices=("full", "smoke"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cases", default="", help="comma-separated case_id subset")
    ap.add_argument("--samples", type=int, default=C.CEM_NUM_SAMPLES)
    ap.add_argument("--opt-steps", type=int, default=C.CEM_MAX_ITERATIONS)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    C.audit(verbose=False)
    cases = [c.strip() for c in args.cases.split(",") if c.strip()] or list(C.CASES)
    unknown = [c for c in cases if c not in C.CASES]
    if unknown:
        raise SystemExit(f"unknown case_id: {unknown}")
    if args.limit:
        cases = cases[: args.limit]

    rows = build(args.stage, cases, args.samples, args.opt_steps)
    out = args.out or (
        C.MANIFEST if args.stage == "full"
        else C.MANIFEST.with_name("e209_g1_smoke_manifest.tsv")
    )
    E199.write_tsv(out, rows, FIELDS)

    scene_shas = {r["effective_scene_sha256"] for r in rows}
    base_shas = {r["baseline_scene_sha256"] for r in rows}
    if len(scene_shas) != len(rows):
        raise SystemExit("duplicate effective_scene_sha256 across rows")
    if scene_shas & base_shas:
        raise SystemExit("an E209 scene equals an E206 baseline scene byte-for-byte")

    print(f"wrote {len(rows)} rows -> {rel(out)}")
    print(f"  objects={dict(Counter(r['object_key'] for r in rows))}")
    print(f"  scene={C.SCENE}  cem={args.samples}x{args.opt_steps} seed={C.CEM_SEED} stage={args.stage}")
    print("  C0 PASS: trajectory + contact_mask + baseline scene sha256 == E206 delivery, all rows")
    print(f"  {len(scene_shas)} distinct E209 scene hashes, none equal to a baseline hash")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
