#!/usr/bin/env python3
"""Benchmark E187 R/G kernels against E186 v4 on the same formal tapes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "workspace/core4d/scripts"))

from eval.core.canonical_distance_query import resolve_geom_ids  # noqa: E402

from spider.rewards.surface_distance import surface_distance_score  # noqa: E402
from spider.simulators.mjwp_object_distance import (  # noqa: E402
    GridObjectDistanceRuntime,
)

RESULTS_E186 = REPO / "workspace/core4d/results/E186"
RESULTS_E187 = REPO / "workspace/core4d/results/E187"
SCENE_MANIFEST = RESULTS_E186 / "s2_compound_physics/compound_scene_manifest.tsv"
DEFAULT_OUTPUT_DIR = RESULTS_E187 / "s2_canonical_grid_sdf/efficiency_v1"
CASE_SPECS = {
    "bucket003_20231018_003_p1": {
        "tape": RESULTS_E186 / "s3_prg_audit/fullbudget_fidelity_v1/raw_chunks/"
        "bucket003_20231018_003_p1/chunk_000000.npz",
        "chosen_grid": RESULTS_E186
        / "s1_canonical_grid_sdf_v4/bucket003/manifest.json",
    },
    "bucket004_20231002_021_p1": {
        "tape": RESULTS_E187 / "s2_canonical_grid_sdf/bucket004_formal_tape/raw_chunks/"
        "bucket004_20231002_021_p1/chunk_000000.npz",
        "chosen_grid": RESULTS_E186
        / "s1_canonical_grid_sdf_v4/bucket004/manifest.json",
    },
    "bucket007_20231020_055_p1": {
        "tape": RESULTS_E186 / "s3_prg_audit/fullbudget_fidelity_v1/raw_chunks/"
        "bucket007_20231020_055_p1/chunk_000000.npz",
        "chosen_grid": RESULTS_E187
        / "s2_canonical_grid_sdf/candidates/bucket007_2p5mm/manifest.json",
    },
}


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_immutable(path: Path, payload: dict[str, Any]) -> None:
    """Create one benchmark result and refuse differing overwrite."""
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    if path.exists():
        if path.read_bytes() != encoded:
            raise RuntimeError(f"immutable efficiency result mismatch: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def read_row(case_id: str) -> dict[str, str]:
    """Load one frozen compound row."""
    with SCENE_MANIFEST.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    matches = [row for row in rows if row["case_id"] == case_id]
    if len(matches) != 1:
        raise RuntimeError(f"efficiency authority row is not unique: {case_id}")
    return matches[0]


def authority(case_id: str) -> dict[str, Any]:
    """Resolve baseline/chosen grids and formal tape without running kernels."""
    if case_id not in CASE_SPECS:
        raise ValueError(f"case is outside E187 efficiency authority: {case_id}")
    spec = CASE_SPECS[case_id]
    row = read_row(case_id)
    paths = {
        "scene": REPO / row["scene_act"],
        "config": (
            spec["tape"].parents[2] / "runs" / f"{case_id}_outdir" / "config_act.yaml"
        ),
        "tape": spec["tape"],
        "baseline_grid": REPO / row["object_distance_manifest"],
        "chosen_grid": spec["chosen_grid"],
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise RuntimeError(f"efficiency artifacts missing: {missing}")
    if sha256(paths["scene"]) != row["effective_scene_sha256"]:
        raise RuntimeError("efficiency scene SHA changed")
    if sha256(paths["baseline_grid"]) != row["object_distance_manifest_sha256"]:
        raise RuntimeError("efficiency baseline grid SHA changed")
    return {"row": row, "paths": paths}


def kernel(
    runtime: GridObjectDistanceRuntime,
    model: mujoco.MjModel,
    ordered_ids: list[int],
    hand_columns: list[int],
    tensors: tuple[torch.Tensor, ...],
    temporal_gate: torch.Tensor,
    temporal_decay: torch.Tensor,
    config: dict[str, Any],
    *,
    continuation: bool,
) -> torch.Tensor:
    """Run production per-geom D_C, conservative G, and surface R work."""
    per_geom = runtime.per_geom_sdf(
        model,
        ordered_ids,
        geom_xpos=tensors[0],
        geom_xmat=tensors[1],
        body_xpos=tensors[2],
        body_xmat=tensors[3],
    )
    conservative = per_geom - runtime.epsilon_grid_m
    gate_checksum = (
        conservative.min(dim=1).values
        + (conservative < float(config["cem_safety_gate_min_sdf_m"]))
        .float()
        .mean(dim=1)
    ).sum()
    hand = per_geom[:, hand_columns].min(dim=1).values
    if continuation:
        score = surface_distance_score(
            hand,
            mode="distance_continuation",
            sigma_m=float(config["surface_band_sigma"]),
        )
    else:
        score = torch.exp(-torch.abs(hand) / float(config["surface_band_sigma"]))
        score = score * (
            (hand >= float(config["surface_band_min_sdf_m"]))
            & (hand <= float(config["surface_band_width_m"]))
        )
    reward = (
        float(config["surface_band_rew_scale"]) * score * temporal_gate * temporal_decay
    )
    return gate_checksum + reward.sum()


def evaluate(
    case_id: str, *, gpu_id: int, batch_size: int, repeats: int
) -> dict[str, Any]:
    """Benchmark baseline and chosen kernels on aligned full-budget transforms."""
    auth = authority(case_id)
    row = auth["row"]
    paths = auth["paths"]
    device = torch.device(f"cuda:{gpu_id}")
    if torch.cuda.get_device_name(device) != "NVIDIA GeForce RTX 5090":
        raise RuntimeError("A1 efficiency benchmark requires local RTX5090 GPU0")
    config = yaml.safe_load(paths["config"].read_text(encoding="utf-8"))
    model = mujoco.MjModel.from_xml_path(str(paths["scene"]))
    object_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")

    def load_runtime(path: Path) -> GridObjectDistanceRuntime:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        return GridObjectDistanceRuntime.load(
            str(path),
            expected_candidate_asset_sha256=row["collider_asset_sha256"],
            expected_error_bound_m=float(manifest["validation"]["epsilon_grid_m"]),
            object_body_id=object_body_id,
        )

    baseline = load_runtime(paths["baseline_grid"])
    chosen = load_runtime(paths["chosen_grid"])
    groups = {
        "body": resolve_geom_ids(model, config["cem_safety_gate_geom_names"]),
        "hand": resolve_geom_ids(model, config["cem_hand_gate_geom_names"]),
        "leg": resolve_geom_ids(model, config["cem_leg_gate_geom_names"]),
    }
    ordered_ids = list(dict.fromkeys(gid for group in groups.values() for gid in group))
    columns = {gid: index for index, gid in enumerate(ordered_ids)}
    hand_columns = [columns[gid] for gid in groups["hand"]]
    with np.load(paths["tape"], allow_pickle=False) as chunk:
        transforms = tuple(
            np.asarray(chunk[key])
            for key in (
                "geometry_geom_xpos",
                "geometry_geom_xmat",
                "geometry_body_xpos",
                "geometry_body_xmat",
            )
        )
        temporal_gate = np.asarray(chunk["reward_trace_surface_band_gate"])
        temporal_decay = np.asarray(chunk["reward_trace_surface_band_decay_factor"])
    samples, horizon = transforms[0].shape[:2]
    if (samples, horizon) != (1024, 48):
        raise RuntimeError("efficiency tape shape mismatch")
    gpu_batches = []
    for start in range(0, samples, batch_size):
        stop = min(start + batch_size, samples)
        flattened = tuple(
            torch.from_numpy(
                value[start:stop].reshape((stop - start) * horizon, *value.shape[2:])
            ).to(device)
            for value in transforms
        )
        gate = torch.from_numpy(temporal_gate[start:stop].reshape(-1)).to(device)
        decay = torch.from_numpy(temporal_decay[start:stop].reshape(-1)).to(device)
        gpu_batches.append((flattened, gate, decay))

    def run(
        runtime: GridObjectDistanceRuntime, continuation: bool
    ) -> tuple[float, float]:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        checksum = 0.0
        for flattened, gate, decay in gpu_batches:
            checksum += float(
                kernel(
                    runtime,
                    model,
                    ordered_ids,
                    hand_columns,
                    flattened,
                    gate,
                    decay,
                    config,
                    continuation=continuation,
                ).item()
            )
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        if not np.isfinite(checksum):
            raise RuntimeError("efficiency kernel checksum is non-finite")
        peak_mib = torch.cuda.max_memory_allocated(device) / (1024**2)
        return elapsed, peak_mib

    run(baseline, False)
    run(chosen, True)
    baseline_seconds: list[float] = []
    chosen_seconds: list[float] = []
    chosen_peak_mib: list[float] = []
    for repeat in range(repeats):
        order = ((baseline, False), (chosen, True))
        if repeat % 2:
            order = tuple(reversed(order))
        measurements: list[tuple[bool, float, float]] = []
        for runtime, continuation in order:
            seconds, peak_mib = run(runtime, continuation)
            measurements.append((continuation, seconds, peak_mib))
        for continuation, seconds, peak_mib in measurements:
            if continuation:
                chosen_seconds.append(seconds)
                chosen_peak_mib.append(peak_mib)
            else:
                baseline_seconds.append(seconds)
    baseline_median = statistics.median(baseline_seconds)
    chosen_median = statistics.median(chosen_seconds)
    ratio = chosen_median / baseline_median
    peak = max(chosen_peak_mib)
    payload = {
        "schema": "e187_reward_grid_efficiency_v1",
        "experiment_id": "E187",
        "stage": "A1_S2_EFFICIENCY",
        "gate0_technical_status": "FAIL",
        "progression_authority": "USER_WAIVED",
        "case_id": case_id,
        "object_key": row["object_key"],
        "device": torch.cuda.get_device_name(device),
        "gpu_id": gpu_id,
        "samples": samples,
        "horizon": horizon,
        "batch_size": batch_size,
        "repeats": repeats,
        "baseline_grid_manifest_sha256": sha256(paths["baseline_grid"]),
        "chosen_grid_manifest_sha256": sha256(paths["chosen_grid"]),
        "tape_sha256": sha256(paths["tape"]),
        "baseline_seconds": baseline_seconds,
        "chosen_seconds": chosen_seconds,
        "baseline_median_seconds": baseline_median,
        "chosen_median_seconds": chosen_median,
        "chosen_over_baseline_ratio": ratio,
        "chosen_peak_allocated_mib": peak,
        "gates": {
            "ratio_le_1p25": ratio <= 1.25,
            "incremental_peak_le_6144mib": peak <= 6144.0,
        },
    }
    payload["status"] = "PASS" if all(payload["gates"].values()) else "FAIL"
    return payload


def parse_args() -> argparse.Namespace:
    """Parse one same-hardware R/G benchmark."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", required=True, choices=sorted(CASE_SPECS))
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--preflight", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Preflight or benchmark one representative formal tape."""
    args = parse_args()
    if args.gpu_id != 0 or args.batch_size <= 0 or args.repeats < 3:
        raise ValueError(
            "efficiency requires GPU0, positive batch, and at least 3 repeats"
        )
    if args.preflight:
        auth = authority(args.case_id)
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "case_id": args.case_id,
                    "paths": {k: str(v) for k, v in auth["paths"].items()},
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    payload = evaluate(
        args.case_id,
        gpu_id=args.gpu_id,
        batch_size=args.batch_size,
        repeats=args.repeats,
    )
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = REPO / output_dir
    write_immutable(output_dir / f"{args.case_id}.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
