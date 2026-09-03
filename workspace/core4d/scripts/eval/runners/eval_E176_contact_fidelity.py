#!/usr/bin/env python3
"""Pre-CEM ref-FK contact-target fidelity gate for E176 low-geom proxies."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import yaml


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))

import eval_E175_proxy_fidelity as fidelity  # noqa: E402
from eval.core.core_metrics import object_collision_geoms  # noqa: E402


DEFAULT_MANIFEST = (
    REPO
    / "workspace/core4d/results/E176/s6_downstream/manifests/"
    "lowgeom_full_manifest.tsv"
)
DEFAULT_OUT = (
    REPO
    / "workspace/core4d/results/E176/s2_proxy/contact_fidelity"
)
CONTACT_P90_GATE_M = 0.08


def repo_path(raw: str | Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else REPO / path


def rel(raw: str | Path) -> str:
    path = Path(raw)
    try:
        return str(path.resolve().relative_to(REPO.resolve()))
    except (OSError, ValueError):
        return str(raw)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fields,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        ""
                        if isinstance(row.get(key), float)
                        and not math.isfinite(float(row[key]))
                        else row.get(key, "")
                    )
                    for key in fields
                }
            )


def quantile(values: list[float], q: float) -> float:
    return (
        float(np.quantile(np.asarray(values, dtype=np.float64), q))
        if values
        else math.nan
    )


def summarize(
    scope: str,
    key: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    proxy = [float(row["full_surface_m"]) for row in rows]
    mesh = [float(row["mesh_surface_m"]) for row in rows]
    signed_error = [
        float(row["full_proxy_minus_mesh_surface_m"]) for row in rows
    ]
    abs_error = [abs(value) for value in signed_error]
    mesh_near_rows = [row for row in rows if bool(row["mesh_near_3cm"])]
    return {
        "scope": scope,
        "key": key,
        "active_contact_rows": len(rows),
        "mesh_near_3cm_rows": len(mesh_near_rows),
        "proxy_surface_p50_m": quantile(proxy, 0.50),
        "proxy_surface_p90_m": quantile(proxy, 0.90),
        "proxy_surface_p95_m": quantile(proxy, 0.95),
        "proxy_surface_max_m": max(proxy) if proxy else math.nan,
        "mesh_surface_p90_m": quantile(mesh, 0.90),
        "proxy_minus_mesh_p50_m": quantile(signed_error, 0.50),
        "proxy_minus_mesh_p90_m": quantile(signed_error, 0.90),
        "proxy_mesh_abs_error_p90_m": quantile(abs_error, 0.90),
        "proxy_miss_3cm_frac": (
            float(np.mean([value > 0.03 for value in proxy]))
            if proxy
            else math.nan
        ),
        "proxy_miss_5cm_frac": (
            float(np.mean([value > 0.05 for value in proxy]))
            if proxy
            else math.nan
        ),
        "proxy_blind_when_mesh_near_3cm_frac": (
            float(
                np.mean(
                    [
                        float(row["full_surface_m"]) > 0.03
                        for row in mesh_near_rows
                    ]
                )
            )
            if mesh_near_rows
            else math.nan
        ),
    }


def source_column(row: dict[str, str], suffix: str) -> str:
    """Read a ``source_<experiment>_<suffix>`` manifest column.

    E176 manifests name these ``source_e174_*``; later experiments carry
    their own producer id, so match on the suffix instead of hardcoding.
    """
    for key, value in row.items():
        if key.startswith("source_") and key.endswith(f"_{suffix}"):
            return value
    return ""


def source_config(row: dict[str, str]) -> Path:
    explicit = source_column(row, "config_act")
    if explicit:
        path = repo_path(explicit)
        if path.is_file():
            return path
    outdir = source_column(row, "outdir_npz")
    if outdir:
        fallback = repo_path(outdir).parent / "config_act.yaml"
        if fallback.is_file():
            return fallback
    raise FileNotFoundError(f"source config missing for {row['case_id']}")


def evaluate_case(
    row: dict[str, str],
    *,
    max_object_geoms: int = 9,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scene = repo_path(row["scene_act"])
    trajectory = repo_path(row["trajectory"])
    mask_path = repo_path(row["contact_mask"])
    config_path = source_config(row)
    rollout_path = repo_path(source_column(row, "outdir_npz") or ".")
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    all_gids = object_collision_geoms(model)
    expected_geoms = int(row["object_geom_count"])
    if len(all_gids) != expected_geoms or expected_geoms > max_object_geoms:
        raise AssertionError(
            f"object geom contract {len(all_gids)} != {expected_geoms} "
            f"<= {max_object_geoms}"
        )

    hand_gids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in fidelity.HAND_GEOMS
    ]
    if any(gid < 0 for gid in hand_gids):
        raise ValueError("missing hand collision geoms")
    hand_paired = fidelity.paired_object_geom_ids(
        model,
        [int(gid) for gid in hand_gids],
        all_gids,
    )
    if hand_paired != all_gids:
        raise AssertionError("hand physics pair matrix does not cover all boxes")

    ref_qpos, ref_source = fidelity.load_reference_qpos(
        trajectory,
        rollout_path,
        scene,
        model,
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not bool(config.get("contact_hdmi_dynamic_target", False)):
        raise ValueError("expected contact_hdmi_dynamic_target=true")
    if str(config.get("contact_hdmi_target_source", "")) != "ref_fk":
        raise ValueError("expected contact_hdmi_target_source=ref_fk")
    hand_body_names = list(
        config.get(
            "hand_approach_body_names",
            ["left_wrist_yaw_link", "right_wrist_yaw_link"],
        )
    )
    eef_offset = np.asarray(
        config.get("contact_hdmi_eef_offset", [0.05, 0.0, 0.0]),
        dtype=np.float64,
    )
    uses_eef_offset = bool(
        config.get("contact_hdmi_target_uses_eef_offset", False)
    )
    contact_pos = fidelity.ref_fk_contact_points(
        model,
        ref_qpos,
        hand_body_names,
        eef_offset,
        uses_eef_offset=uses_eef_offset,
    )
    with np.load(mask_path, allow_pickle=True) as payload:
        masks = np.asarray(
            payload["spider_contact_mask_3cm"],
            dtype=np.bool_,
        )
    active_mask = fidelity.resize_time_nearest(
        masks[:, fidelity.person_idx(row["case_id"]), :],
        len(ref_qpos),
    ).astype(np.bool_)
    contact_rows = fidelity.contact_target_rows(
        row=row,
        model=model,
        data=data,
        ref_qpos=ref_qpos,
        contact_pos=contact_pos,
        active_mask=active_mask,
        all_gids=all_gids,
        physics_gids=all_gids,
        prg_gids=all_gids,
    )
    if not contact_rows:
        raise ValueError("contact mask has no active ref-FK rows")
    fidelity.add_visual_mesh_distances(model, contact_rows)
    for contact in contact_rows:
        contact.update(
            {
                "scene_act": rel(scene),
                "object_geom_count": expected_geoms,
                "compiled_pair_count": int(
                    row["compiled_robot_object_pair_count"]
                ),
                "ref_qpos_source": ref_source,
                "target_uses_eef_offset": uses_eef_offset,
            }
        )
    case_summary = summarize(
        "case",
        row["case_id"],
        contact_rows,
    )
    case_summary.update(
        {
            "case_id": row["case_id"],
            "object_key": row["object_key"],
            "object_geom_count": expected_geoms,
            "compiled_pair_count": int(
                row["compiled_robot_object_pair_count"]
            ),
            "ref_qpos_source": ref_source,
            "target_uses_eef_offset": uses_eef_offset,
        }
    )
    return contact_rows, case_summary


def markdown(
    object_rows: list[dict[str, Any]],
    errors: list[dict[str, str]],
    status: str,
    experiment_id: str,
) -> str:
    lines = [
        f"# {experiment_id} ref-FK contact fidelity",
        "",
        f"Gate: grouped target→proxy p90 ≤ {CONTACT_P90_GATE_M:.2f} m.",
        "",
        "| Object | Geoms | Active rows | Proxy p90 | Mesh p90 | |Proxy−mesh| p90 | Gate |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in object_rows:
        lines.append(
            "| {key} | {object_geom_count} | {active_contact_rows} | "
            "{proxy_surface_p90_m:.4f} | {mesh_surface_p90_m:.4f} | "
            "{proxy_mesh_abs_error_p90_m:.4f} | {gate_status} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            f"Overall status: **{status}**",
            f"Errors: `{len(errors)}`",
            "",
        ]
    )
    return "\n".join(lines)


def run(
    manifest_path: Path,
    out_dir: Path,
    *,
    experiment_id: str = "E176",
    expected_cases: int = 39,
    expected_objects: int = 6,
    max_object_geoms: int = 9,
) -> dict[str, Any]:
    rows = read_tsv(manifest_path)
    case_ids = [row["case_id"] for row in rows]
    if len(rows) != expected_cases or len(case_ids) != len(set(case_ids)):
        raise ValueError(
            f"{experiment_id} contact fidelity requires "
            f"{expected_cases} unique rows"
        )

    contact_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=1):
        try:
            contacts, case_summary = evaluate_case(
                row, max_object_geoms=max_object_geoms
            )
            contact_rows.extend(contacts)
            case_rows.append(case_summary)
            print(
                f"[{index:02d}/{len(rows):02d}] {row['case_id']}: "
                f"p90={case_summary['proxy_surface_p90_m']:.4f}m"
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(
                {
                    "case_id": row["case_id"],
                    "object_key": row["object_key"],
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            print(f"[{index:02d}/{len(rows):02d}] {row['case_id']}: {exc}")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in contact_rows:
        grouped[str(row["object_key"])].append(row)
    object_geom_counts = {
        row["object_key"]: int(row["object_geom_count"]) for row in rows
    }
    object_rows = []
    for object_key in sorted(object_geom_counts):
        item = summarize("object", object_key, grouped[object_key])
        item["object_key"] = object_key
        item["object_geom_count"] = object_geom_counts[object_key]
        item["gate_status"] = (
            "pass"
            if item["active_contact_rows"] > 0
            and item["proxy_surface_p90_m"] <= CONTACT_P90_GATE_M
            else "fail"
        )
        object_rows.append(item)

    status = (
        "pass"
        if not errors
        and len(case_rows) == expected_cases
        and len(object_rows) == expected_objects
        and all(row["gate_status"] == "pass" for row in object_rows)
        else "fail"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(out_dir / "contact_target_rows.tsv", contact_rows)
    write_tsv(out_dir / "case_summary.tsv", case_rows)
    write_tsv(out_dir / "object_summary.tsv", object_rows)
    write_tsv(out_dir / "errors.tsv", errors)
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "experiment_id": experiment_id,
        "manifest": rel(manifest_path),
        "status": status,
        "gate": {
            "metric": "grouped_ref_fk_target_to_proxy_surface_p90_m",
            "threshold_m": CONTACT_P90_GATE_M,
        },
        "expected_cases": expected_cases,
        "expected_objects": expected_objects,
        "max_object_geoms": max_object_geoms,
        "evaluated_cases": len(case_rows),
        "error_cases": len(errors),
        "active_contact_rows": len(contact_rows),
        "objects": object_rows,
        "errors": errors,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "summary.md").write_text(
        markdown(object_rows, errors, status, experiment_id),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--experiment-id", default="E176")
    parser.add_argument("--expected-cases", type=int, default=39)
    parser.add_argument("--expected-objects", type=int, default=6)
    parser.add_argument("--max-object-geoms", type=int, default=9)
    args = parser.parse_args()
    payload = run(
        repo_path(args.manifest),
        repo_path(args.out_dir),
        experiment_id=args.experiment_id,
        expected_cases=args.expected_cases,
        expected_objects=args.expected_objects,
        max_object_geoms=args.max_object_geoms,
    )
    return 0 if payload["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
