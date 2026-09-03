#!/usr/bin/env python3
"""E206 P8: 65 cases x 2 arms = 130 CEM runs over an 8-GPU slot pool.

Both arms share a task dir, so each job writes to a DISTINCT per-arm output_dir
and skip-already-done keys on that per-arm rollout npz -- the run is resumable
and safe to interrupt (plan236 A2).

Two deliberate deviations from run_e204e205_cem.py, both evidence-backed:
  * `use_torch_compile` comes from admission_decision.json instead of being
    hardcoded false. P3 measured it: compile ON was 2.0-2.7 min SLOWER on both
    probes, so the frozen value happens to be false -- but it is now measured,
    not assumed.
  * `object_collision_sdf_batch_groups: true` on BOTH arms (E176 verified it is
    numerically equivalent), so it is not an arm variable.

Queue order is the A2b priority: all desk007 first (the C5a lifeline vs E174),
then per-object round-robin, then the remainder -- so a halted queue still has
every object represented and C5a still has data.

Usage:
    .venv/bin/python .../run_e206_cem.py --gpus 0,1,2,3,4,5,6,7
    ... --stage smoke --limit 2          # 2 cases x 2 arms, then diff config_act
    ... --arms prg --dry-run
Env: E206_FORCE=1 ignores skip-already-done.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import e206_common as C  # noqa: E402

PY = str(C.REPO / ".venv/bin/python")
MANIFESTS = (
    C.S3_DIR / "omnirt_v1/ref_fk/stage2b_manifest_omnirt_v1_ref_fk.tsv",
    C.S3_DIR / "omnirt_v2/ref_fk/stage2b_manifest_omnirt_v2_ref_fk.tsv",
)


def passing_cases() -> list[tuple[str, str]]:
    """(case_id, object_key) for every Stage2b pass; v2 rescue wins over v1 fail."""
    by_case: dict[str, str] = {}
    for manifest in MANIFESTS:
        if not manifest.is_file():
            continue
        for row in C.read_tsv(manifest):
            if row.get("stage2b_status") == "pass":
                by_case[row["case_id"]] = row["object_key"]
    return [(k, by_case[k]) for k in sorted(by_case)]


def priority_order(cases: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """A2b: desk007 first (C5a lifeline), then per-object round-robin."""
    lifeline = [c for c in cases if c[1] == "desk007"]
    rest: dict[str, list[tuple[str, str]]] = collections.defaultdict(list)
    for case in cases:
        if case[1] != "desk007":
            rest[case[1]].append(case)
    ordered = list(lifeline)
    while any(rest.values()):
        for key in sorted(rest):
            if rest[key]:
                ordered.append(rest[key].pop(0))
    return ordered


def admission() -> dict[str, Any]:
    path = C.S6_DIR / "cem/throughput/admission_decision.json"
    if not path.is_file():
        raise SystemExit(
            f"missing {path} -- plan236 P3 requires the admission decision to be "
            "frozen and logged BEFORE the CEM queue is dispatched"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("A1_single_task", {}).get("verdict") != "pass":
        raise SystemExit("admission A1 did not pass; apply the A3 ladder first")
    if payload.get("A2_queue", {}).get("verdict") != "pass":
        raise SystemExit("admission A2 did not pass; apply the A3 ladder first")
    return payload


def cem_env(gpu: str | None, *, compile_on: bool) -> dict:
    env = dict(os.environ)
    env.setdefault("MUJOCO_GL", os.environ.get("E206_MUJOCO_GL", "disable"))
    if not compile_on:
        env["TORCHDYNAMO_DISABLE"] = "1"
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONUNBUFFERED"] = "1"
    return env


def prepare(arm: str, case_id: str, *, samples: int, iters: int, seed: int,
            stage: str, compile_on: bool) -> dict[str, Any]:
    r: dict[str, Any] = {"arm": arm, "case_id": case_id, "status": "", "note": ""}
    override = C.OVERRIDE_DIR / f"{C.override_name(case_id, arm)}.yaml"
    if not override.is_file():
        r.update({"status": "skip_missing_override", "note": "run build_overrides.py"})
        return r
    out_dir = C.arm_out_dir(arm, case_id, stage)
    rollout = out_dir / "trajectory_mjwp_act.npz"
    if rollout.is_file() and os.environ.get("E206_FORCE", "0") != "1":
        r["status"] = "skip_already_done"
        return r
    r["cmd"] = [
        PY, "-u", "examples/run_mjwp.py",
        f"+override={C.override_name(case_id, arm)}",
        "save_video=false", "video_camera=auto",
        f"+use_torch_compile={'true' if compile_on else 'false'}",
        f"seed={seed}", f"num_samples={samples}", f"max_num_iterations={iters}",
        f"output_dir={out_dir}",
    ]
    r["output_dir"] = str(out_dir.relative_to(C.REPO))
    r["status"] = "ready"
    return r


def run_pool(jobs: list[dict[str, Any]], gpus: list[str], log_dir: Path,
             *, dry_run: bool, compile_on: bool) -> list[dict[str, Any]]:
    log_dir.mkdir(parents=True, exist_ok=True)
    prepared = [j for j in jobs if j["status"] == "ready"]
    results = [j for j in jobs if j["status"] != "ready"]
    for j in results:
        print(f"   prep {j['arm']}/{j['case_id']} -> {j['status']} {j.get('note','')}",
              flush=True)
    if dry_run:
        for j in prepared:
            j["command"] = " ".join(j["cmd"])
            j["status"] = "dry_run"
            j.pop("cmd", None)
            print(f"   DRY {j['arm']}/{j['case_id']}", flush=True)
        return results + prepared

    running: dict[str, tuple] = {}
    pending = list(prepared)
    free = list(gpus) or [None]
    total = len(pending)
    started = 0
    while pending or running:
        while pending and free:
            j = pending.pop(0)
            gpu = free.pop(0)
            tag = f"{j['arm']}_{j['case_id']}"
            fh = (log_dir / f"{tag}.log").open("w")
            fh.write(f"# gpu={gpu} command={' '.join(j['cmd'])}\n\n")
            fh.flush()
            j["gpu"] = gpu
            j["t0"] = time.monotonic()
            proc = subprocess.Popen(j["cmd"], cwd=C.REPO, stdout=fh,
                                    stderr=subprocess.STDOUT,
                                    env=cem_env(gpu, compile_on=compile_on))
            running[tag] = (proc, j, fh, gpu)
            started += 1
            print(f"   launch [{started}/{total}] {tag} on gpu {gpu} "
                  f"({len(pending)} pending)", flush=True)
        done = [t for t, v in running.items() if v[0].poll() is not None]
        for tag in done:
            proc, j, fh, gpu = running.pop(tag)
            fh.close()
            j["wall_min"] = round((time.monotonic() - j["t0"]) / 60.0, 2)
            j["status"] = "cem_ok" if proc.returncode == 0 else f"cem_fail_rc{proc.returncode}"
            j["log"] = str((log_dir / f"{tag}.log").relative_to(C.REPO))
            j.pop("cmd", None)
            j.pop("t0", None)
            free.append(gpu)
            results.append(j)
            print(f"   done {tag} -> {j['status']} wall={j['wall_min']}min "
                  f"({len(results)}/{len(jobs)})", flush=True)
        if running and not done:
            time.sleep(15)
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=",".join(C.ARMS))
    ap.add_argument("--cases", default="")
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--num-samples", type=int, default=None)
    ap.add_argument("--max-iterations", type=int, default=None)
    ap.add_argument("--seed", type=int, default=C.CEM_SEED)
    ap.add_argument("--limit", type=int, default=0, help="first N cases per arm (smoke)")
    ap.add_argument("--stage", default="full", choices=["full", "smoke"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    adm = admission()
    frozen = adm["frozen"]
    samples = args.num_samples or int(frozen["num_samples"])
    iters = args.max_iterations or int(frozen["max_num_iterations"])
    compile_on = bool(frozen["use_torch_compile"])

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for arm in arms:
        if arm not in C.ARMS:
            raise SystemExit(f"unknown arm {arm!r}")
    cases = priority_order(passing_cases())
    if args.cases:
        want = {c.strip() for c in args.cases.split(",") if c.strip()}
        cases = [c for c in cases if c[0] in want]
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        raise SystemExit("no Stage2b-passing cases")

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    print(f"E206 CEM stage={args.stage} | {len(cases)} cases x {len(arms)} arms "
          f"= {len(cases) * len(arms)} runs | gpus={gpus} | "
          f"samples={samples} iters={iters} seed={args.seed} "
          f"compile={compile_on}", flush=True)

    jobs = [prepare(arm, case_id, samples=samples, iters=iters, seed=args.seed,
                    stage=args.stage, compile_on=compile_on)
            for case_id, _ in cases for arm in arms]
    out_dir = args.out_dir or (C.S6_DIR / "cem" / args.stage)
    results = run_pool(jobs, gpus, out_dir / "logs",
                       dry_run=args.dry_run, compile_on=compile_on)

    tally = collections.Counter(r["status"] for r in results)
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "stage": args.stage,
        "n_cases": len(cases), "arms": arms, "n_runs": len(results),
        "budget": {"num_samples": samples, "max_num_iterations": iters,
                   "seed": args.seed, "use_torch_compile": compile_on},
        "admission_source": "s6_downstream/cem/throughput/admission_decision.json",
        "status_counts": dict(tally),
        "runs": results,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "e206_cem_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    C.write_tsv(out_dir / "e206_cem_summary.tsv", results)
    print(f"\n{dict(tally)}")
    fails = sum(v for k, v in tally.items() if k.startswith("cem_fail"))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
