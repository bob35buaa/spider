#!/usr/bin/env python3
"""E206 P3: measure CEM wall-clock vs object-geom count, and freeze admission.

plan236 called for *synthetic* probe scenes at N in {1,5,9,16}. That is no longer
the cheapest or the most honest option: S3 + P7 have produced REAL noPRG/PRG
scenes whose installed box counts already span N = 2, 5, 7, 9, 10, 12, and 12 is
the true maximum that will ever ship (chair021, the only object that would have
pushed toward 16, was dropped in F12). Measuring the real scenes removes the
"does the probe resemble production?" question entirely, and N=16 is not
extrapolated because nothing will run there.

Probes use the PRG arm: 18N pairs is the expensive arm, so a PRG-derived bound
is conservative for both.

Outputs `e206_throughput_curve.{tsv,md}` + `admission_decision.json` with the
A1/A2/A2b/A3 verdicts. plan236 requires this be written BEFORE the full queue.

Usage:
    .venv/bin/python .../measure_throughput.py --gpus 0,1,2,3,4,5,6,7
    ... --compile-both      # also run compile=on probes for the N spread
    ... --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import e206_common as C  # noqa: E402

PY = str(C.REPO / ".venv/bin/python")
# E176/validate_canary_throughput.py:27-29 — per-record planning time.
PLAN_TIME_RE = re.compile(r"plan time:\s*([0-9]+(?:\.[0-9]+)?)s,.*?opt_steps:\s*(\d+)")

# plan236 P3 admission bars.
A1_MAX_MEDIAN_WALL_MIN = 120.0
A2_QUEUE_RUNS = 130          # 65 case x 2 arm (pre-S3); recomputed from actuals below
A2_GPUS = 8
A2_MAX_HOURS = 48.0


def scene_build_rows() -> list[dict[str, str]]:
    path = C.S5_DIR / "arm_scenes/arm_scene_build.tsv"
    if not path.is_file():
        raise SystemExit(f"missing {path}; run build_arm_scenes.py first")
    return [r for r in C.read_tsv(path) if r.get("status") == "built"]


def pick_probes(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """One case per distinct installed box count, lowest case_id for determinism."""
    by_n: dict[int, dict[str, str]] = {}
    for row in sorted(rows, key=lambda r: r["case_id"]):
        n = int(row["object_geom_count"])
        by_n.setdefault(n, row)
    return [by_n[n] for n in sorted(by_n)]


def cem_env(gpu: str | None, *, compile_on: bool) -> dict:
    env = dict(os.environ)
    env.setdefault("MUJOCO_GL", "disable")
    if not compile_on:
        env["TORCHDYNAMO_DISABLE"] = "1"
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONUNBUFFERED"] = "1"
    return env


def probe_cmd(case_id: str, *, compile_on: bool, samples: int, iters: int,
              out_dir: Path) -> list[str]:
    return [
        PY, "-u", "examples/run_mjwp.py",
        f"+override={C.override_name(case_id, 'prg')}",
        "save_video=false", "video_camera=auto",
        f"+use_torch_compile={'true' if compile_on else 'false'}",
        f"seed={C.CEM_SEED}", f"num_samples={samples}",
        f"max_num_iterations={iters}",
        f"output_dir={out_dir}",
    ]


def run_wave(jobs: list[dict[str, Any]], gpus: list[str], log_dir: Path,
             *, dry_run: bool) -> list[dict[str, Any]]:
    log_dir.mkdir(parents=True, exist_ok=True)
    if dry_run:
        for job in jobs:
            job["status"] = "dry_run"
            job["command"] = " ".join(job["cmd"])
            print(f"   DRY {job['tag']}: {job['command']}", flush=True)
        return jobs

    running: dict[str, tuple] = {}
    pending = list(jobs)
    free = list(gpus)
    while pending or running:
        while pending and free:
            job = pending.pop(0)
            gpu = free.pop(0)
            fh = (log_dir / f"{job['tag']}.log").open("w")
            fh.write(f"# gpu={gpu} command={' '.join(job['cmd'])}\n\n")
            fh.flush()
            job["gpu"] = gpu
            job["t0"] = time.monotonic()
            proc = subprocess.Popen(job["cmd"], cwd=C.REPO, stdout=fh,
                                    stderr=subprocess.STDOUT,
                                    env=cem_env(gpu, compile_on=job["compile_on"]))
            running[job["tag"]] = (proc, job, fh, gpu)
            print(f"   launch {job['tag']} on gpu {gpu} "
                  f"({len(pending)} pending)", flush=True)
        done = [t for t, v in running.items() if v[0].poll() is not None]
        for tag in done:
            proc, job, fh, gpu = running.pop(tag)
            fh.close()
            job["wall_s"] = time.monotonic() - job["t0"]
            job["wall_min"] = job["wall_s"] / 60.0
            job["returncode"] = proc.returncode
            job["status"] = "ok" if proc.returncode == 0 else f"fail_rc{proc.returncode}"
            text = (log_dir / f"{tag}.log").read_text(encoding="utf-8", errors="replace")
            times = [float(m.group(1)) for m in PLAN_TIME_RE.finditer(text)
                     if int(m.group(2)) > 0]
            job["plan_time_records"] = len(times)
            job["plan_time_median_s"] = (
                round(statistics.median(times), 4) if times else ""
            )
            # Trajectory length varies 170-322 steps across probes, so wall clock
            # is driven by BOTH N and episode length. Record the length so the
            # N-scaling fit can use the length-independent per-step plan time.
            steps = re.findall(r"sim_steps: \d+/(\d+)", text)
            job["sim_steps_total"] = int(steps[-1]) if steps else ""
            job.pop("cmd", None)
            job.pop("t0", None)
            free.append(gpu)
            print(f"   done {tag} -> {job['status']} "
                  f"wall={job['wall_min']:.1f}min "
                  f"plan_median={job['plan_time_median_s']}s", flush=True)
        if running and not done:
            time.sleep(15)
    return jobs


def linear_fit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Least squares y = a + b*x; returns (a, b)."""
    n = len(xs)
    if n < 2:
        return (ys[0] if ys else 0.0, 0.0)
    mx, my = sum(xs) / n, sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom if denom else 0.0
    return (my - b * mx, b)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--num-samples", type=int, default=C.CEM_NUM_SAMPLES)
    ap.add_argument("--max-iterations", type=int, default=C.CEM_MAX_ITERATIONS)
    ap.add_argument("--compile-both", action="store_true",
                    help="also run every probe with torch.compile ON")
    ap.add_argument("--compile-on-ns", default="",
                    help="comma-separated N values to ALSO probe with compile ON. "
                         "Cheaper than --compile-both: the compile question is one "
                         "binary decision, so probing it at the large-N end is "
                         "enough and keeps everything inside a single GPU wave.")
    ap.add_argument("--out-dir", type=Path, default=C.S6_DIR / "cem/throughput")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--analyze-only", action="store_true",
                    help="recompute the fit + admission verdict from an existing "
                         "e206_throughput_curve.tsv without re-running any probe")
    args = ap.parse_args()

    rows = scene_build_rows()
    probes = pick_probes(rows)
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]

    compile_on_ns = {int(v) for v in args.compile_on_ns.split(",") if v.strip()}
    jobs: list[dict[str, Any]] = []
    for probe in probes:
        case_id = probe["case_id"]
        n = int(probe["object_geom_count"])
        modes = [False]
        if args.compile_both or n in compile_on_ns:
            modes.append(True)
        for compile_on in modes:
            tag = f"N{n:02d}_{case_id}_compile{'On' if compile_on else 'Off'}"
            out_dir = args.out_dir / "runs" / tag
            jobs.append({
                "tag": tag, "case_id": case_id, "object_key": probe["object_key"],
                "object_geom_count": n,
                "prg_pairs": C.expected_pair_counts(n)["prg"],
                "compile_on": compile_on,
                "num_samples": args.num_samples,
                "max_iterations": args.max_iterations,
                "cmd": probe_cmd(case_id, compile_on=compile_on,
                                 samples=args.num_samples,
                                 iters=args.max_iterations, out_dir=out_dir),
            })

    print(f"E206 P3 throughput: {len(jobs)} probes over {len(gpus)} GPUs "
          f"| budget {args.num_samples}x{args.max_iterations} "
          f"| N spread {sorted({j['object_geom_count'] for j in jobs})}", flush=True)

    if args.analyze_only:
        prev = args.out_dir / "e206_throughput_curve.tsv"
        if not prev.is_file():
            raise SystemExit(f"--analyze-only needs an existing {prev}")
        results = []
        for row in C.read_tsv(prev):
            r = dict(row)
            r["compile_on"] = str(row.get("compile_on", "")).lower() == "true"
            for key in ("object_geom_count", "prg_pairs"):
                r[key] = int(row[key])
            for key in ("wall_min", "wall_s"):
                if row.get(key):
                    r[key] = float(row[key])
            results.append(r)
    else:
        results = run_wave(jobs, gpus, args.out_dir / "logs", dry_run=args.dry_run)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        C.write_tsv(args.out_dir / "e206_throughput_curve.tsv", results)
    if args.dry_run:
        return 0

    ok = [r for r in results if r["status"] == "ok"]
    off = [r for r in ok if not r["compile_on"]]
    on = [r for r in ok if r["compile_on"]]
    best = on if (on and statistics.median([r["wall_min"] for r in on])
                  < statistics.median([r["wall_min"] for r in off] or [1e9])) else off
    use_compile = bool(best and best[0]["compile_on"])

    # N-scaling is fitted on per-step PLAN TIME, not wall clock: episode length
    # varies 170-322 steps across probes, so a wall-clock-vs-N fit would silently
    # mix "more boxes" with "longer clip". Wall clock still drives A1/A2, which
    # are wall-clock bars by definition.
    plan_pts = [(float(r["object_geom_count"]), float(r["plan_time_median_s"]))
                for r in best if r.get("plan_time_median_s") not in ("", None)]
    fit_a, fit_b = (linear_fit([p[0] for p in plan_pts], [p[1] for p in plan_pts])
                    if plan_pts else (0.0, 0.0))
    max_n = max((int(r["object_geom_count"]) for r in rows), default=0)
    median_wall = statistics.median([r["wall_min"] for r in best]) if best else float("nan")
    worst_wall = max((r["wall_min"] for r in best), default=float("nan"))
    # The queue is the real case set x 2 arms, not plan236's pre-S3 estimate.
    queue_runs = len(rows) * len(C.ARMS)
    projected_h = (queue_runs * median_wall) / (A2_GPUS * 60.0) if best else float("nan")
    a2_bound_min = (A2_MAX_HOURS * 60.0 * A2_GPUS) / queue_runs if queue_runs else 0.0

    payload: dict[str, Any] = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "probe_basis": "real installed noPRG/PRG scenes (not synthetic), PRG arm",
        "n_probes": len(results),
        "n_ok": len(ok),
        "budget": {"num_samples": args.num_samples,
                   "max_num_iterations": args.max_iterations,
                   "seed": C.CEM_SEED},
        "frozen": {
            "N_MAX_shipped": max_n,
            "use_torch_compile": use_compile,
            "num_samples": args.num_samples,
            "max_num_iterations": args.max_iterations,
            "queue_priority": ["desk007 (all, C5a lifeline)", "per-object round robin", "remainder"],
        },
        "fit_plan_time_s_vs_N": {
            "intercept_a": round(fit_a, 4), "slope_b_per_box": round(fit_b, 4),
            "note": "per-step plan time; length-independent, unlike wall clock",
        },
        "median_wall_min": round(median_wall, 2) if best else None,
        "worst_wall_min": round(worst_wall, 2) if best else None,
        "A1_single_task": {
            "bar_min": A1_MAX_MEDIAN_WALL_MIN,
            "observed_median_min": round(median_wall, 2) if best else None,
            "verdict": "pass" if best and median_wall <= A1_MAX_MEDIAN_WALL_MIN else "fail",
        },
        "A2_queue": {
            "queue_runs": queue_runs, "gpus": A2_GPUS, "bar_hours": A2_MAX_HOURS,
            "per_task_bound_min": round(a2_bound_min, 1),
            "projected_hours": round(projected_h, 2) if best else None,
            "verdict": "pass" if best and projected_h <= A2_MAX_HOURS else "fail",
        },
        "A3_fallback_needed": not (best and median_wall <= A1_MAX_MEDIAN_WALL_MIN
                                   and projected_h <= A2_MAX_HOURS),
        "probes": results,
    }
    (args.out_dir / "admission_decision.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    md = ["# E206 P3 吞吐实测与准入", "",
          f"预算 {args.num_samples}x{args.max_iterations}, seed={C.CEM_SEED}, "
          f"探针=真实 PRG 场景 (非合成)", "",
          "| N boxes | PRG pairs | case | sim steps | compile | wall (min) | plan median (s) | status |",
          "|---:|---:|---|---:|---|---:|---:|---|"]
    for r in sorted(results, key=lambda x: (x["object_geom_count"], x["compile_on"])):
        wall = f"{r.get('wall_min', float('nan')):.1f}" if "wall_min" in r else "-"
        md.append(f"| {r['object_geom_count']} | {r['prg_pairs']} | {r['case_id']} | "
                  f"{r.get('sim_steps_total','-')} | "
                  f"{'on' if r['compile_on'] else 'off'} | {wall} | "
                  f"{r.get('plan_time_median_s','-')} | {r['status']} |")
    md += ["", f"拟合 (与轨迹长度无关): plan_time_s ≈ {fit_a:.4f} + {fit_b:.4f}·N",
           f"A1 (中位 ≤ {A1_MAX_MEDIAN_WALL_MIN} min): "
           f"{payload['A1_single_task']['verdict']} (实测 {payload['median_wall_min']})",
           f"A2 ({queue_runs} 条 / {A2_GPUS} 卡 ≤ {A2_MAX_HOURS}h): "
           f"{payload['A2_queue']['verdict']} (投影 {payload['A2_queue']['projected_hours']}h)"]
    (args.out_dir / "e206_throughput_curve.md").write_text("\n".join(md) + "\n",
                                                           encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "probes"},
                     ensure_ascii=False, indent=2))
    return 0 if not payload["A3_fallback_needed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
