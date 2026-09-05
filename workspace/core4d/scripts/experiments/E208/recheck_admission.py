#!/usr/bin/env python3
"""E208: inherit E206's admission decision, re-check only the queue arithmetic.

E208 does NOT run its own throughput probes.  The scenes are E206's scenes with
the object's initial pose moved (V5c proves nothing else differs), so E206's
``plan_time_s ~ 19.55 + 0.249*N`` fit and its measured 47.4 min median apply
unchanged.  Probing again would create a second budget authority, and the two
would eventually disagree.

What genuinely changed is the queue size: plan238 sized it at 66 (22 cases x 3
translation variants); it is now 105 (21 cases after the chair005 exclusion x 5
variants after the rotation fix).  So the ``frozen`` block is carried over
verbatim and only A2 is recomputed.

The one part of ``frozen`` that does NOT carry over is ``queue_priority``: E206
gave desk007 a lifeline because its C5a claim needed it, and put chair005 first
because it was the only n=1 object.  Neither applies here -- E208 has no C5a and
no chair005 -- so the E208 payload records its own ordering rule and says why,
rather than shipping an inherited string that no longer describes the queue.

Usage:
    .venv/bin/python .../E208/recheck_admission.py
    ... --n-runs 105        # default: the manifest's row count, else eligible_potential()
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import e208_common as C  # noqa: E402

E208_QUEUE_PRIORITY = [
    "round-robin over (object x variant): the first 20 rows cover all 4x5 cells once",
    "within a round, a diagonal on (object + variant) so consecutive rows share neither",
    "any prefix is therefore a balanced stratified sample, not a truncated one",
]
E208_QUEUE_PRIORITY_NOTE = (
    "E206's 'desk007 lifeline' and 'chair005 first' rules are deliberately NOT "
    "inherited: the lifeline existed for E206's C5a claim, which E208 does not "
    "have, and chair005 is excluded here (decay-degenerate, see log297 F7)."
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-runs", type=int, default=None)
    ap.add_argument("--gpus", type=int, default=None)
    ap.add_argument("--out", type=Path, default=C.ADMISSION_JSON)
    args = ap.parse_args()

    src = C.source_admission()

    n_runs = args.n_runs
    source_of_n = "--n-runs"
    if n_runs is None:
        if C.PRIORITY_MANIFEST.is_file():
            n_runs = len(C.read_tsv(C.PRIORITY_MANIFEST))
            source_of_n = C.rel(C.PRIORITY_MANIFEST)
        else:
            n_runs = C.eligible_potential()
            source_of_n = "eligible_potential()"

    proj = C.queue_projection(n_runs, args.gpus)

    frozen = dict(src["frozen"])
    inherited_priority = frozen.pop("queue_priority", None)
    frozen["queue_priority"] = E208_QUEUE_PRIORITY
    frozen["queue_priority_inherited_from_E206"] = inherited_priority
    frozen["queue_priority_note"] = E208_QUEUE_PRIORITY_NOTE

    payload = {
        "experiment": C.EXP_ID, "run_id": C.RUN_ID, "generated_at": C.now(),
        "inherits_from": C.rel(C.SOURCE_ADMISSION_JSON),
        "inherits_sha256": C.sha256(C.SOURCE_ADMISSION_JSON),
        "inherit_rationale": (
            "E208's scenes are E206's scenes with only the object's initial pose moved "
            "(V5c: masked byte parity across the whole arm chain, 105/105). The timing "
            "fit and the measured median therefore transfer; re-probing would create a "
            "second budget authority."
        ),
        "probe_basis": src["probe_basis"] + " [inherited by E208, not re-measured]",
        "n_probes": src["n_probes"], "n_ok": src["n_ok"],
        "budget": src["budget"],
        "frozen": frozen,
        "fit_plan_time_s_vs_N": src["fit_plan_time_s_vs_N"],
        "median_wall_min": src["median_wall_min"],
        "worst_wall_min": src["worst_wall_min"],
        "A1_single_task": dict(src["A1_single_task"], inherited=True),
        "A2_queue": {
            "queue_runs": proj["n_runs"],
            "queue_runs_source": source_of_n,
            "gpus": proj["gpus"],
            "rounds": proj["rounds"],
            "bar_hours": proj["bar_hours"],
            "per_task_bound_min": proj["per_task_bound_min"],
            "optimistic_hours": proj["optimistic_h"],
            "projected_hours": proj["bound_h"],
            "verdict": proj["verdict"],
            "recomputed": True,
            "note": (
                f"plan238 sized this at {C.EXPECTED_CASES * len(C.TRANS_VARIANTS)} "
                f"(22 cases x 3 trans); actual is {proj['n_runs']} after the 5-variant "
                "amendment and the chair005 exclusion."
            ),
        },
        "A3_fallback_needed": src["A3_fallback_needed"],
        "per_task_timeout_min": C.PER_TASK_TIMEOUT_MIN,
        "gpu_contention_note": (
            "E207 (R293) and E209 (R295) share these 8 GPUs from other sessions. The "
            "memory admission in the runner waits for a free slot and never preempts, so "
            "coexistence is safe, but the realised wall clock will exceed projected_hours "
            "by whatever those queues occupy. Record the actual overlap in log297."
        ),
    }
    C.write_json(args.out, payload)

    print(f"A1 (inherited): median {payload['A1_single_task']['observed_median_min']} min "
          f"vs bar {payload['A1_single_task']['bar_min']} min -> "
          f"{payload['A1_single_task']['verdict']}")
    print(f"A2 (recomputed): {proj['n_runs']} runs / {proj['gpus']} gpus = {proj['rounds']} rounds")
    print(f"   optimistic {proj['optimistic_h']} h, worst-case bound {proj['bound_h']} h "
          f"vs bar {proj['bar_hours']} h -> {proj['verdict']}")
    print(f"   n_runs from {source_of_n}")
    print(f"-> {C.rel(args.out)}")
    return 0 if proj["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
