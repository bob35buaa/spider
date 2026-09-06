#!/usr/bin/env python3
"""Build the 12-row E212 Stage A manifest, plus the two execution shards.

Column contract is E199's ``run_local_priority_queue.py`` (tier/arm/aug_variant
are scheduler sort keys).  Every input path carries its sha256 so the
scheduler's ``input_failures`` preflight refuses to launch against a drifted
scene / trajectory / mask / override.  Reference inputs are additionally
asserted byte-identical to what E206's PRG arm consumed -- that is what makes
"the g-curve is single-variable" true at the *data* level, not only at the
config level.

Why two shard files
-------------------
Stage A runs on two 8-GPU machines that share this /mnt.  The queue rewrites the
WHOLE manifest TSV on every status change (``run_local_priority_queue.py``
write_tsv calls), so two instances pointed at one file would each overwrite the
other's status with their own stale in-memory snapshot.  Rows would be claimed
twice or dropped silently.  The fix is disjoint files: each machine reads and
writes only its own shard, and ``merge_shards.py`` reconciles them afterwards.

The split is a Latin-square diagonal over the canonical (case_id, arm) order
(``e212_common.shard_of``), not two contiguous halves, so each shard spans all 4
cases and all 3 g values.  If one machine dies, the surviving shard is still an
interpretable cross-section of the sweep instead of "only the 066 rows".

The predecessor used index parity, which for 12 rows yields 6/6; this run is
8 local + 4 remote.  The naive ``index % 3 == 2`` would give 8/4 but puts the
ENTIRE G08 arm on the remote box -- and G08 is the predicted-best arm, so losing
that machine would lose a whole g value.  Hence the diagonal.

Usage:
    .venv/bin/python workspace/core4d/scripts/experiments/E212/build_manifest.py
    ... --stage smoke --samples 64 --opt-steps 4 --cases <id> --arms G06
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[5]
for _d in ("E212", "E199"):
    sys.path.insert(0, str(REPO / "workspace/core4d/scripts/experiments" / _d))

import e199_common as E199  # noqa: E402
import e212_common as C  # noqa: E402

FIELDS = [
    # `variant` and `preferred_pool` are consumed by the render tooling;
    # the scheduler ignores them.
    "case_id", "object_key", "person", "experiment", "arm", "gravcomp", "tier", "aug_variant",
    "variant", "preferred_pool", "shard",
    "target_task", "target_scene", "scene_act", "scene_name", "effective_scene_sha256",
    "trajectory", "trajectory_sha256", "contact_mask", "contact_mask_sha256",
    "override_id", "override_path", "override_sha256",
    "baseline_outdir_npz", "baseline_scene_sha256",
    "g1_outdir_npz",
    "cem_samples", "cem_opt_steps", "cem_seed", "execution_mode",
    "result_npz", "outdir_npz", "config_act", "video", "log",
    # `host` is written by the queue wrapper so a two-machine run can always
    # answer "where did this row actually run".
    "status", "failure_mode", "gpu_id", "host", "updated_at",
]

#: Shard labels and sizes live in e212_common so the launcher, the manifest
#: builder and merge_shards.py can never disagree about the split.
#: A = local (8 rows), B = remote (4 rows).
SHARDS = C.SHARDS
EXPECTED_SHARD_ROWS = C.EXPECTED_SHARD_ROWS


def rel(path: Path) -> str:
    return path.relative_to(REPO).as_posix()


def baseline_config(case_id: str) -> dict:
    path = C.prg_out_dir(case_id) / "config_act.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"{case_id}: missing E206 PRG config_act.yaml: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def build(stage: str, cases: list[str], arms: list[str], samples: int, opt_steps: int):
    rows: list[dict[str, str]] = []
    # Canonical order: (case_id, arm). The shard assignment is a pure function of
    # (case_id, arm) itself -- not of the row index -- so a subset run assigns the
    # same shard a full run would, and the split is reproducible from the manifest.
    pairs = [(c, a) for c in cases for a in arms]
    pairs.sort()
    for case_id, arm in pairs:
        source = C.source_row(case_id)
        base_cfg = baseline_config(case_id)

        scene = C.scene_path(source, arm)
        base_scene = C.base_scene_path(source)
        trajectory = C.kinematic_npz(source)
        contact = REPO / base_cfg["contact_hdmi_mask_path"]
        override = C.override_path(case_id, arm)
        target_scene = C.task_dir(source) / "scene.xml"
        for label, path in (
            ("scene_act", scene), ("base_scene", base_scene), ("trajectory", trajectory),
            ("contact_mask", contact), ("override", override), ("target_scene", target_scene),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"{case_id}/{arm}: missing {label}: {path}")

        # --- reference inputs identical to what E206 PRG consumed ------------
        traj_sha, mask_sha = E199.sha256(trajectory), E199.sha256(contact)
        if traj_sha != source["trajectory_sha256"]:
            raise ValueError(f"{case_id}: trajectory drifted from E206 delivery")
        if mask_sha != source["contact_mask_sha256"]:
            raise ValueError(f"{case_id}: contact mask drifted from E206 delivery")
        base_scene_sha = E199.sha256(base_scene)
        if base_scene_sha != source["scene_act_sha256"]:
            raise ValueError(f"{case_id}: E206 PRG scene drifted from E206 delivery")
        if base_cfg.get("scene_name") != C.BASE_SCENE:
            raise ValueError(
                f"{case_id}: E206 baseline scene_name={base_cfg.get('scene_name')!r}"
                f" != {C.BASE_SCENE!r}"
            )

        out = C.out_dir(case_id, arm, stage)
        rows.append({
            "case_id": case_id,
            "object_key": source["object_key"],
            "person": source.get("person", ""),
            "experiment": C.EXP,
            "arm": arm,
            "gravcomp": C.gravcomp_str(arm),
            "tier": "P0",
            "aug_variant": "orig",
            "variant": f"{C.EXP}_{case_id}_{arm}",
            "preferred_pool": "local",
            "shard": C.shard_of(case_id, arm),
            "target_task": C.target_task(source),
            "target_scene": rel(target_scene),
            "scene_act": rel(scene),
            "scene_name": C.SCENE_BY_ARM[arm],
            "effective_scene_sha256": E199.sha256(scene),
            "trajectory": rel(trajectory),
            "trajectory_sha256": traj_sha,
            "contact_mask": rel(contact),
            "contact_mask_sha256": mask_sha,
            "override_id": C.override_id(case_id, arm),
            "override_path": rel(override),
            "override_sha256": E199.sha256(override),
            "baseline_outdir_npz": rel(C.prg_out_dir(case_id) / "trajectory_mjwp_act.npz"),
            "baseline_scene_sha256": base_scene_sha,
            "g1_outdir_npz": rel(C.g1_out_dir(case_id) / "trajectory_mjwp_act.npz"),
            "cem_samples": str(samples),
            "cem_opt_steps": str(opt_steps),
            "cem_seed": str(C.CEM_SEED),
            "execution_mode": "production" if stage == "full" else stage,
            "result_npz": rel(out / f"{C.EXP}_{case_id}_{arm}.npz"),
            "outdir_npz": rel(C.result_npz(case_id, arm, stage)),
            "config_act": rel(C.config_act(case_id, arm, stage)),
            "video": rel(C.S6_DIR / f"render/{stage}/{C.EXP}_{case_id}_{arm}.mp4"),
            "log": f"logs/{C.EXP}/cem/{stage}/{C.EXP}_{case_id}_{arm}.log",
            "status": "READY_FOR_FULL",
            "failure_mode": "",
            "gpu_id": "",
            "host": "",
            "updated_at": E199.now(),
        })
    return rows


def check_shards(rows: list[dict[str, str]], stage: str) -> None:
    """Disjointness and coverage -- the invariants the two-machine run rests on."""
    by_shard: dict[str, list[dict[str, str]]] = {s: [] for s in SHARDS}
    for row in rows:
        by_shard[row["shard"]].append(row)

    keys = {s: {(r["case_id"], r["arm"]) for r in rs} for s, rs in by_shard.items()}
    if keys["A"] & keys["B"]:
        raise SystemExit(f"shards overlap on {sorted(keys['A'] & keys['B'])}")
    if keys["A"] | keys["B"] != {(r["case_id"], r["arm"]) for r in rows}:
        raise SystemExit("shard union != full manifest")
    if len(rows) != C.EXPECTED_ROWS:
        # Subset run (smoke / --cases): the counts and coverage below only apply
        # to a full build.  Say so out loud -- a guard that silently does not
        # fire reads exactly like a guard that passed.
        print(
            f"  shard split NOT fully checked ({stage}): {len(rows)} rows != "
            f"{C.EXPECTED_ROWS}; run a full build to exercise count+coverage asserts"
        )
        return
    for shard, want in EXPECTED_SHARD_ROWS.items():
        if len(by_shard[shard]) != want:
            raise SystemExit(f"shard {shard}: {len(by_shard[shard])} rows, want {want}")
    # The diagonal should give each shard every case and every arm; if a future
    # edit breaks that, a dead machine would leave an uninterpretable remainder.
    for shard, rs in by_shard.items():
        if {r["case_id"] for r in rs} != set(C.CASES):
            raise SystemExit(f"shard {shard} does not span all {C.EXPECTED_CASES} cases")
        if {r["arm"] for r in rs} != set(C.ARM_ORDER):
            raise SystemExit(f"shard {shard} does not span all {len(C.ARMS)} arms")
    print(f"  shard split OK ({stage}): " + ", ".join(
        f"{s}={len(by_shard[s])}" for s in SHARDS
    ))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="full", choices=("full", "smoke"))
    ap.add_argument("--cases", default="", help="comma-separated case_id subset")
    ap.add_argument("--arms", default="", help="comma-separated arm subset")
    ap.add_argument("--samples", type=int, default=C.CEM_NUM_SAMPLES)
    ap.add_argument("--opt-steps", type=int, default=C.CEM_MAX_ITERATIONS)
    args = ap.parse_args()

    C.audit(verbose=False)
    cases = [c.strip() for c in args.cases.split(",") if c.strip()] or list(C.CASES)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()] or list(C.ARM_ORDER)
    unknown = [c for c in cases if c not in C.CASES] + [a for a in arms if a not in C.ARMS]
    if unknown:
        raise SystemExit(f"unknown case_id/arm: {unknown}")

    rows = build(args.stage, cases, arms, args.samples, args.opt_steps)
    check_shards(rows, args.stage)

    main_path = C.manifest_path(args.stage)
    main_path.parent.mkdir(parents=True, exist_ok=True)
    E199.write_tsv(main_path, rows, FIELDS)
    for shard in SHARDS:
        subset = [r for r in rows if r["shard"] == shard]
        if subset:
            E199.write_tsv(C.manifest_path(args.stage, shard), subset, FIELDS)

    scene_shas = {r["effective_scene_sha256"] for r in rows}
    base_shas = {r["baseline_scene_sha256"] for r in rows}
    if len(scene_shas) != len(rows):
        raise SystemExit("duplicate effective_scene_sha256 across rows")
    if scene_shas & base_shas:
        raise SystemExit("an E212 scene equals the E206 baseline scene byte-for-byte")

    print(f"wrote {len(rows)} rows -> {rel(main_path)}")
    for shard in SHARDS:
        p = C.manifest_path(args.stage, shard)
        if p.is_file():
            print(f"  shard {shard}: {sum(r['shard'] == shard for r in rows):2d} rows -> {rel(p)}")
    print(f"  arms={dict(Counter(r['arm'] for r in rows))}  gravcomp={sorted({r['gravcomp'] for r in rows})}")
    print(f"  cem={args.samples}x{args.opt_steps} seed={C.CEM_SEED} stage={args.stage}")
    print("  C0 PASS: trajectory + contact_mask + baseline scene sha256 == E206 delivery, all rows")
    print(f"  {len(scene_shas)} distinct E212 scene hashes, none equal to a baseline hash")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
