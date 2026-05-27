#!/usr/bin/env python3
"""Summarize E029 COLA/D6 support-body sanity results.

This evaluator intentionally does not launch full CEM.  Plan 34 requires full
retargeting only after the no-training D6 sanity gate passes.  The script reads
the existing audit, preflight, manifests, sanity summaries, and visualization
artifacts, then writes a single gate decision under results/E029/eval.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


THIS = Path(__file__).resolve()
REPO = THIS.parents[4]
WS = REPO / "workspace/core4d_collab_retarget"
RESULTS = WS / "results/E029"
OUT = RESULTS / "eval"
CANDIDATES = WS / "results/E028/candidates.json"


def _read_csv(path: Path, *, delimiter: str = ",") -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter=delimiter))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in rows for k in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _candidate_count() -> int:
    if not CANDIDATES.is_file():
        return 0
    data = json.loads(CANDIDATES.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return len(data)
    if isinstance(data, dict):
        for key in ("candidates", "rows", "variants"):
            value = data.get(key)
            if isinstance(value, list):
                return len(value)
    return 0


def _manifest_rows(path: Path) -> int:
    rows = _read_csv(path, delimiter="\t")
    return len(rows)


def summarize_sanity_csv(path: Path) -> dict[str, Any]:
    rows = _read_csv(path)
    family = path.relative_to(RESULTS).parts[0]
    pass_count = sum(1 for row in rows if _as_bool(row.get("pass_gate")))
    object_mean = [v for row in rows if (v := _float(row.get("object_pos_err_mean_m"))) is not None]
    object_max = [v for row in rows if (v := _float(row.get("object_pos_err_max_m"))) is not None]
    drift_mean = [v for row in rows if (v := _float(row.get("support_drift_mean_m"))) is not None]
    drift_max = [v for row in rows if (v := _float(row.get("support_drift_max_m"))) is not None]
    force_sat = [v for row in rows if (v := _float(row.get("force_saturation_frac"))) is not None]
    torque_sat = [v for row in rows if (v := _float(row.get("torque_saturation_frac"))) is not None]
    nan_count = sum(int(_float(row.get("nan_count")) or 0) for row in rows)
    mode = path.name.removeprefix("candidates_").removesuffix("_summary.csv")
    gate_eligible = family in {"d6", "freejoint", "connect", "multiconnect"}
    full_gate_pass = gate_eligible and len(rows) == _candidate_count() and pass_count >= 4
    return {
        "family": family,
        "mode_tag": mode,
        "gate_eligible": gate_eligible,
        "summary_csv": str(path.relative_to(REPO)),
        "rows": len(rows),
        "pass_count": pass_count,
        "pass_rate": pass_count / len(rows) if rows else 0.0,
        "full_gate_pass": full_gate_pass,
        "object_mean_avg_m": _mean(object_mean),
        "object_mean_worst_m": max(object_mean) if object_mean else None,
        "object_max_worst_m": max(object_max) if object_max else None,
        "drift_mean_avg_m": _mean(drift_mean),
        "drift_mean_worst_m": max(drift_mean) if drift_mean else None,
        "drift_max_worst_m": max(drift_max) if drift_max else None,
        "force_sat_max": max(force_sat) if force_sat else None,
        "torque_sat_max": max(torque_sat) if torque_sat else None,
        "nan_count_total": nan_count,
    }


def _best_candidate(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    gate_rows = [row for row in rows if _as_bool(row.get("gate_eligible"))]
    rows = gate_rows or rows

    def key(row: dict[str, Any]) -> tuple[float, float, float, float]:
        return (
            float(row.get("pass_count") or 0),
            -float(row.get("nan_count_total") or 0),
            -float(row.get("object_mean_avg_m") or 1e9),
            -float(row.get("drift_mean_avg_m") or 1e9),
        )

    return max(rows, key=key)


def build_summary() -> dict[str, Any]:
    candidate_count = _candidate_count()
    audit_rows = _read_csv(RESULTS / "audit/e028_candidate_modes.csv")
    preflight_rows = _read_csv(RESULTS / "preflight/axis_contact_summary.csv")
    manifest_counts = {
        "d6": _manifest_rows(RESULTS / "d6/manifest.tsv"),
        "freejoint": _manifest_rows(RESULTS / "freejoint/manifest.tsv"),
        "connect": _manifest_rows(RESULTS / "connect/manifest.tsv"),
        "multiconnect": _manifest_rows(RESULTS / "multiconnect/manifest.tsv"),
    }
    candidate_summaries = sorted(RESULTS.glob("*/sanity/candidates_*_summary.csv"))
    sanity_rows = [summarize_sanity_csv(path) for path in candidate_summaries]
    best = _best_candidate(sanity_rows)
    gate_pass = any(_as_bool(row.get("full_gate_pass")) for row in sanity_rows)
    panels = sorted((RESULTS / "preflight").glob("*_axis_contact_panel.jpg"))
    videos = sorted(RESULTS.glob("*/sanity/*.mp4")) + sorted((RESULTS / "d6/sanity_sweep").glob("*.mp4"))
    frames = sorted(RESULTS.glob("*/sanity/*.jpg")) + sorted((RESULTS / "d6/sanity_sweep").glob("*.jpg"))

    audit_proves_not_cola = (
        len(audit_rows) == candidate_count
        and candidate_count > 0
        and all(row.get("support_proxy_mode") == "mocap_pad" for row in audit_rows)
        and all(not _as_bool(row.get("cola_d6_match")) for row in audit_rows)
    )
    preflight_coverage = len(preflight_rows) == candidate_count and candidate_count > 0
    height_axes = sorted({row.get("height_local_axis", "") for row in preflight_rows if row.get("height_local_axis")})
    support_non_mocap_attempted = manifest_counts["d6"] > 0 and manifest_counts["freejoint"] > 0
    bounded_6d_attempted = manifest_counts["connect"] > 0 and manifest_counts["multiconnect"] > 0

    summary = {
        "candidate_count": candidate_count,
        "audit_rows": len(audit_rows),
        "audit_proves_e028_not_cola_d6": audit_proves_not_cola,
        "preflight_rows": len(preflight_rows),
        "preflight_coverage_ok": preflight_coverage,
        "height_local_axes": height_axes,
        "manifest_counts": manifest_counts,
        "num_candidate_sanity_summaries": len(sanity_rows),
        "best_candidate_sanity": best,
        "candidate_full_gate_pass": gate_pass,
        "full_cem_should_run": gate_pass,
        "full_cem_decision": "run_full" if gate_pass else "stop_before_full_sanity_failed",
        "support_non_mocap_attempted": support_non_mocap_attempted,
        "bounded_6d_attempted": bounded_6d_attempted,
        "visualization": {
            "axis_contact_panels": len(panels),
            "sanity_videos": len(videos),
            "sanity_frames": len(frames),
            "coverage_ok": len(panels) >= candidate_count and len(videos) > 0 and len(frames) > 0,
        },
        "claims": {
            "C1_e018_e028_not_cola_dynamic_support": audit_proves_not_cola,
            "C2_e029_dynamic_support_body_attempted": support_non_mocap_attempted,
            "C3_d6_equivalent_connection_attempted_but_not_passed": bounded_6d_attempted and not gate_pass,
            "C4_no_fixed_canonical_z_axis_preflight_done": preflight_coverage and height_axes == ["y"],
            "C5_no_training_load_path_proved": gate_pass,
            "C6_full_retarget_allowed_by_plan": gate_pass,
            "C6_full_retarget_correctly_stopped": not gate_pass,
            "C7_visualization_done": len(panels) >= candidate_count and len(videos) > 0 and len(frames) > 0,
        },
    }
    return summary | {"sanity_overview": sanity_rows}


def write_outputs(summary: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sanity_rows = summary.pop("sanity_overview")
    _write_csv(OUT / "sanity_overview.csv", sanity_rows)
    (OUT / "eval_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    best = summary.get("best_candidate_sanity") or {}
    lines = [
        "# E029 Evaluation Summary",
        "",
        f"- Candidates: `{summary['candidate_count']}`",
        f"- Audit rows: `{summary['audit_rows']}`",
        f"- Preflight rows: `{summary['preflight_rows']}`",
        f"- Candidate sanity summaries: `{summary['num_candidate_sanity_summaries']}`",
        f"- Full CEM decision: `{summary['full_cem_decision']}`",
        "",
        "## Best Candidate Sanity",
        "",
        f"- Summary: `{best.get('summary_csv', '-')}`",
        f"- Pass: `{best.get('pass_count', 0)}/{best.get('rows', 0)}`",
        f"- Object mean avg/worst: `{_fmt(best.get('object_mean_avg_m'))}/{_fmt(best.get('object_mean_worst_m'))}m`",
        f"- Drift mean avg/worst: `{_fmt(best.get('drift_mean_avg_m'))}/{_fmt(best.get('drift_mean_worst_m'))}m`",
        f"- Force saturation max: `{_fmt(best.get('force_sat_max'))}`",
        f"- NaN count total: `{best.get('nan_count_total', 0)}`",
        "",
        "## Claims",
        "",
        "| Claim | Status |",
        "|---|---|",
    ]
    for claim, status in summary["claims"].items():
        lines.append(f"| `{claim}` | `{_fmt(status)}` |")
    lines.extend(
        [
            "",
            "## Manifest Counts",
            "",
            "| Branch | Rows |",
            "|---|---:|",
        ]
    )
    for key, value in summary["manifest_counts"].items():
        lines.append(f"| `{key}` | {value} |")
    lines.extend(
        [
            "",
            "## Decision",
            "",
        ]
    )
    if summary["candidate_full_gate_pass"]:
        lines.append("Candidate D6 sanity passed the plan gate; full CEM may be run.")
    else:
        lines.append(
            "Candidate D6 sanity did not reach `>=4/5`; per plan 34, full CEM remains stopped."
        )
    (OUT / "eval_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true", help="Summarize all E029 sanity outputs")
    ap.add_argument(
        "--fail-on-stopped",
        action="store_true",
        help="Exit non-zero when the plan gate says full CEM should not run",
    )
    args = ap.parse_args()
    if not args.all:
        ap.error("Use --all to summarize E029 outputs")
    summary = build_summary()
    stopped = not summary["candidate_full_gate_pass"]
    write_outputs(summary)
    print((OUT / "eval_summary.md").relative_to(REPO))
    if args.fail_on_stopped and stopped:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
