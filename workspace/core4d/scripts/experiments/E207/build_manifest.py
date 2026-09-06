#!/usr/bin/env python3
"""Build the 9-row E207 G1only full-CEM manifest for the local 8-GPU scheduler.

Column contract is E199's ``run_local_priority_queue.py`` (tier/arm/aug_variant
are scheduler sort keys; E207 has a single tier and arm so they are constants).

Every input path is carried with its sha256 so the scheduler's ``input_failures``
preflight refuses to launch against a drifted scene / trajectory / mask / override.
Reference inputs are additionally asserted to be byte-identical to what E178 ran,
which is what makes "E207 vs E178 is a single-variable comparison" true at the
data level and not just at the config level.

Usage:
    .venv/bin/python workspace/core4d/scripts/experiments/E207/build_manifest.py
    ... --stage smoke --limit 1 --samples 64 --opt-steps 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import e207_common as C

sys.path.insert(0, str(C.REPO / "workspace/core4d/scripts/experiments/E199"))
import e199_common as E199  # noqa: E402

E178_MANIFEST = (
    C.REPO / "workspace/core4d/results/E178/s6_downstream/eval/full/evaluated_manifest_snapshot.tsv"
)

FIELDS = [
    # `variant` and `preferred_pool` are consumed by E168/render_a100_cem_videos.py;
    # the scheduler ignores them.
    "case_id", "object_key", "person", "experiment", "arm", "tier", "aug_variant",
    "variant", "preferred_pool",
    "target_task", "target_scene", "scene_act", "scene_name", "effective_scene_sha256",
    "trajectory", "trajectory_sha256", "contact_mask", "contact_mask_sha256",
    "override_id", "override_path", "override_sha256",
    "cem_samples", "cem_opt_steps", "cem_seed", "execution_mode",
    "result_npz", "outdir_npz", "config_act", "video", "log",
    "status", "failure_mode", "gpu_id", "updated_at",
]


def rel(path: Path) -> str:
    return path.relative_to(C.REPO).as_posix()


def build(stage: str, limit: int, samples: int, opt_steps: int) -> list[dict[str, str]]:
    e178 = {r["case_id"]: r for r in E199.read_with_fields(E178_MANIFEST)[0]}
    rows: list[dict[str, str]] = []
    for source in C.sources()[: limit or None]:
        case_id = source["case_id"]
        ref = e178[case_id]

        scene = C.scene_path(case_id)
        trajectory = C.REPO / ref["trajectory"]
        contact = C.REPO / ref["contact_mask"]
        override = C.override_path(case_id)
        target_scene = C.task_dir(case_id) / "scene.xml"
        for label, path in (("scene_act", scene), ("trajectory", trajectory),
                            ("contact_mask", contact), ("override", override),
                            ("target_scene", target_scene)):
            if not path.is_file():
                raise FileNotFoundError(f"{case_id}: missing {label}: {path}")

        # C0: the reference inputs must be exactly what E178 consumed.
        if E199.sha256(trajectory) != ref["trajectory_sha256"]:
            raise ValueError(f"{case_id}: trajectory drifted from E178")
        if E199.sha256(contact) != ref["contact_mask_sha256"]:
            raise ValueError(f"{case_id}: contact mask drifted from E178")

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
            "target_task": C.task_of(case_id),
            "target_scene": rel(target_scene),
            "scene_act": rel(scene),
            "scene_name": C.SCENE,
            "effective_scene_sha256": E199.sha256(scene),
            "trajectory": rel(trajectory),
            "trajectory_sha256": ref["trajectory_sha256"],
            "contact_mask": rel(contact),
            "contact_mask_sha256": ref["contact_mask_sha256"],
            "override_id": C.override_id(case_id),
            "override_path": rel(override),
            "override_sha256": E199.sha256(override),
            "cem_samples": str(samples),
            "cem_opt_steps": str(opt_steps),
            "cem_seed": str(C.CEM_SEED),
            "execution_mode": "production" if stage == "full" else stage,
            "result_npz": rel(out / f"{C.EXP}_{case_id}_{C.ARM_TAG}.npz"),
            "outdir_npz": rel(C.result_npz(case_id, stage)),
            "config_act": rel(C.config_act(case_id, stage)),
            "video": rel(C.RESULTS / f"s6_downstream/render/{stage}/{C.EXP}_{case_id}_{C.ARM_TAG}.mp4"),
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
    ap.add_argument("--samples", type=int, default=C.FULL_SAMPLES)
    ap.add_argument("--opt-steps", type=int, default=C.FULL_OPT_STEPS)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    C.audit(verbose=False)
    rows = build(args.stage, args.limit, args.samples, args.opt_steps)
    out = args.out or (C.MANIFEST if args.stage == "full" else
                       C.MANIFEST.with_name("g1only_smoke_manifest.tsv"))
    out.parent.mkdir(parents=True, exist_ok=True)
    E199.write_tsv(out, rows, FIELDS)

    from collections import Counter

    dist = dict(Counter(r["object_key"] for r in rows))
    print(f"wrote {len(rows)} rows -> {rel(out)}")
    print(f"  objects={dist}  scene={C.SCENE}  cem={args.samples}x{args.opt_steps} seed={C.CEM_SEED}")
    print("  C0 PASS: trajectory + contact_mask sha256 identical to E178 for every row")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
