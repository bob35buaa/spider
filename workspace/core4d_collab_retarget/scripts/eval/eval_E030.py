#!/usr/bin/env python3
"""Evaluate E030 direct-CEM outputs for the three D6-locked sanity-pass cases."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


THIS = Path(__file__).resolve()
REPO = THIS.parents[4]
if str(THIS.parent) not in sys.path:
    sys.path.insert(0, str(THIS.parent))

import eval_E002 as e002  # noqa: E402
import paper_metrics  # noqa: E402


WS = REPO / "workspace/core4d_collab_retarget"
RESULTS = WS / "results/E030/d6_locked_cem"
MANIFEST = WS / "results/E029/d6/manifest.tsv"
BASE = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
SELECTED = {
    "E029_d003_box021_20231011_035_p2_d6_locked",
    "E029_d003_box021_20231018_029_p2_d6_locked",
    "E029_d003_box021_20231020_019_p1_d6_locked",
}


def _read_manifest() -> list[dict[str, str]]:
    with MANIFEST.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def read_variants() -> dict[str, dict[str, str]]:
    variants: dict[str, dict[str, str]] = {}
    for row in _read_manifest():
        variant = row["variant"]
        if variant not in SELECTED:
            continue
        variants[variant] = {
            "name": variant,
            "case": row["derived_task"],
            "source_task": row["source_task"],
            "override": f"core4d_collab_{variant}",
            "split": "E030_d6_locked_cem",
            "role": "main",
            "person_idx": row["person_idx"],
            "scene_name": row["scene_name"],
            "source_e028_variant": row["source_e028_variant"],
            "selected_side_face": row["selected_side_face"],
            "selected_side_frac": row["selected_side_frac"],
            "selected_side_margin": row["selected_side_margin"],
            "endpoint_confidence": row["endpoint_confidence"],
            "mask_slug": row["mask_slug"],
        }
    missing = sorted(SELECTED - set(variants))
    if missing:
        raise FileNotFoundError(f"Missing E030 variants in manifest: {missing}")
    return variants


def _relative_or_abs(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def make_scene_loader(variants: dict[str, dict[str, str]]):
    scenes_by_case: dict[str, str] = {
        str(meta["case"]): str(meta["scene_name"]) for meta in variants.values()
    }

    def load_scene_model(case: str) -> tuple[mujoco.MjModel, Path]:
        scene_name = scenes_by_case.get(case)
        candidates = []
        if scene_name:
            candidates.append(BASE / case / f"{scene_name}.xml")
        candidates.append(BASE / case / "scene.xml")
        for path in candidates:
            if not path.is_file():
                continue
            try:
                return mujoco.MjModel.from_xml_path(str(path)), path
            except Exception:
                text = path.read_text(encoding="utf-8")
                patched = text.replace(
                    "../../../../../../spider/assets",
                    str(REPO / "spider/assets"),
                ).replace(
                    "../../../../../example_datasets",
                    str(REPO / "example_datasets"),
                )
                patched_path = RESULTS / f"_scene_abs_{case}_{path.stem}.xml"
                patched_path.parent.mkdir(parents=True, exist_ok=True)
                patched_path.write_text(patched, encoding="utf-8")
                return mujoco.MjModel.from_xml_path(str(patched_path)), patched_path
        raise FileNotFoundError(f"No loadable E030 scene for {case}")

    return load_scene_model


def _float(summary: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(summary.get(key, default))
    except (TypeError, ValueError):
        return default


def evaluate_variant(variant: str, variants: dict[str, dict[str, str]]) -> dict[str, Any]:
    e002.RESULTS = RESULTS
    e002.VARIANTS_FILE = MANIFEST
    e002.load_scene_model = make_scene_loader(variants)  # type: ignore[method-assign]

    summary = e002.evaluate_variant(variant, variants)
    meta = variants[variant]
    npz_path = RESULTS / f"{variant}.npz"
    qpos_ref, _ctrl_ref, _cfg = e002.load_ref(str(meta["override"]), str(meta["case"]))
    model, scene_used = e002.load_scene_model(str(meta["case"]))
    data_npz = np.load(npz_path, allow_pickle=True)
    qpos = e002.e072.flatten_time_major(data_npz["qpos"])

    summary["scene_used"] = _relative_or_abs(scene_used)
    summary["E030_source_e028_variant"] = meta["source_e028_variant"]
    summary["E030_selected_side_face"] = meta["selected_side_face"]
    summary["E030_selected_side_frac"] = float(meta["selected_side_frac"])
    summary["E030_selected_side_margin"] = float(meta["selected_side_margin"])
    summary["E030_endpoint_confidence"] = meta["endpoint_confidence"]
    summary["E030_mask_slug"] = meta["mask_slug"]
    summary.update(
        paper_metrics.add_paper_metrics(
            summary,
            repo=REPO,
            results_dir=RESULTS,
            model=model,
            qpos=qpos,
            qpos_ref=qpos_ref,
            person_idx=int(meta["person_idx"]),
        )
    )
    summary["E030_contact_object_proxy_success"] = bool(
        summary["E002_success_case_window"]
        and _float(summary, "paper_dynaretarget_object_success") != 0.0
        and _float(summary, "paper_transport_success") != 0.0
    )
    summary["E030_visual_risk_label_proxy"] = (
        "near-clean"
        if summary["E030_contact_object_proxy_success"]
        else (
            "fail"
            if _float(summary, "case_window_sim_contact_frames_pct") < 30.0
            or _float(summary, "case_window_obj_err_mean_m") > 0.20
            or _float(summary, "case_window_pelvis_z_min_m") < 0.45
            else "borderline"
        )
    )
    e002.write_variant_summary(summary)
    return summary


def write_comparison(summaries: list[dict[str, Any]]) -> None:
    keys = sorted({k for row in summaries for k in row})
    path = RESULTS / "comparison.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(summaries)

    aggregate = {
        "num_results": len(summaries),
        "num_contact_object_proxy_success": sum(
            bool(row["E030_contact_object_proxy_success"]) for row in summaries
        ),
        "num_case_window_success": sum(bool(row["E002_success_case_window"]) for row in summaries),
        "mean_case_window_obj_err_m": float(
            np.mean([float(row["case_window_obj_err_mean_m"]) for row in summaries])
        ),
        "mean_case_window_contact_pct": float(
            np.mean([float(row["case_window_sim_contact_frames_pct"]) for row in summaries])
        ),
        "mean_paper_object_Epos_case_m": float(
            np.mean([float(row["paper_object_Epos_case_m"]) for row in summaries])
        ),
        "mean_paper_object_Erot_case_deg": float(
            np.mean([float(row["paper_object_Erot_case_deg"]) for row in summaries])
        ),
        "mean_carry_progress_ratio_case": float(
            np.mean([float(row["paper_carry_progress_ratio_case"]) for row in summaries])
        ),
        "labels": {
            str(row["variant"]): str(row["E030_visual_risk_label_proxy"])
            for row in summaries
        },
    }
    (RESULTS / "aggregate_summary.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(path)
    print(json.dumps(aggregate, indent=2, sort_keys=True))


def main() -> None:
    variants = read_variants()
    selected = sys.argv[1:] or sorted(variants)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "plots").mkdir(parents=True, exist_ok=True)

    summaries = []
    for variant in selected:
        if variant not in variants:
            print(f"[SKIP] unknown E030 variant {variant}")
            continue
        summaries.append(evaluate_variant(variant, variants))
    if not summaries:
        raise SystemExit("No E030 results evaluated.")
    write_comparison(summaries)


if __name__ == "__main__":
    main()
