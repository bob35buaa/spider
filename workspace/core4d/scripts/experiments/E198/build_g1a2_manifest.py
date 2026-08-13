#!/usr/bin/env python3
"""Build the joint 103-run E198(G1+A2) + E192-ext(A2) priority manifest.

Reuses existing authority and pre-built G1 sidecars (never re-derives inputs):
  * verifies trajectory / scene / override / contact-mask SHA parity vs authority
  * audits every G1 sidecar as a single-variable object-gravcomp diff of its base
  * verifies A2_GATE equals E192 A2_GATE
  * assigns artifact paths, canary/sentinel picks, snapshots scenes, writes manifests
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e198_common as C  # noqa: E402


def git_head() -> str:
    out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=C.REPO, text=True, capture_output=True, check=False)
    return out.stdout.strip() or "unknown"


def signature(element: ET.Element) -> Any:
    return (element.tag, tuple(sorted(element.attrib.items())), (element.text or "").strip(),
            tuple(signature(child) for child in element))


def assert_gravcomp_diff(base: Path, sidecar: Path) -> None:
    """Fail unless sidecar == base with only object body gravcomp 0/absent -> 1."""
    base_root = ET.parse(base).getroot()
    objs = [b for b in base_root.iter("body") if b.get("name") == "object"]
    if len(objs) != 1 or objs[0].get("gravcomp") not in (None, "0", "0.0"):
        raise ValueError(f"unexpected base object gravcomp in {base}")
    expected = ET.parse(base).getroot()
    next(b for b in expected.iter("body") if b.get("name") == "object").set("gravcomp", "1")
    side_root = ET.parse(sidecar).getroot()
    if signature(side_root) != signature(expected):
        raise AssertionError(f"G1 sidecar is not a single-variable gravcomp diff: {sidecar}")


def artifact_paths(experiment: str, case_id: str, arm: str, stage: str) -> dict[str, str]:
    tag = "E198" if experiment == "E198" else "E192ext"
    sub = "canary" if stage == "canary" else "full"
    if experiment == "E198":
        cem_root = f"workspace/core4d/results/E198/s6_downstream/cem/{sub}_g1a2"
        vid_root = f"workspace/core4d/results/E198/s6_downstream/render/{sub}_g1a2"
        log_root = f"logs/E198/cem/{sub}_g1a2"
    else:
        cem_root = f"workspace/core4d/results/E192/s6_downstream/cem/{sub}_a2_expansion"
        vid_root = f"workspace/core4d/results/E192/s6_downstream/render/{sub}_a2_expansion"
        log_root = f"logs/E192/cem/{sub}_a2_expansion"
    variant = f"{tag}_{case_id}_{arm}" + ("_canary" if stage == "canary" else "")
    return {
        "variant": variant,
        "result_npz": f"{cem_root}/{variant}.npz",
        "outdir_npz": f"{cem_root}/{variant}_outdir/trajectory_mjwp_act.npz",
        "config_act": f"{cem_root}/{variant}_outdir/config_act.yaml",
        "video": f"{vid_root}/{variant}.mp4",
        "log": f"{log_root}/{variant}.log",
    }


def finalize_row(cell: dict[str, Any], stage: str, sentinel: bool, canary_rep: bool) -> dict[str, Any]:
    row: dict[str, Any] = dict(cell)
    row.update({
        "kp_pos": C.KP_POS, "kp_rot": C.KP_ROT,
        "cem_samples": C.CANARY_SAMPLES if stage == "canary" else C.FULL_SAMPLES,
        "cem_opt_steps": C.CANARY_OPT_STEPS if stage == "canary" else C.FULL_OPT_STEPS,
        "cem_seed": C.CEM_SEED, "sentinel": sentinel, "canary_representative": canary_rep,
        "gpu_id": "", "status": "READY_FOR_CANARY" if stage == "canary" else "READY_FOR_FULL",
        "failure_mode": "", "execution_mode": stage, "updated_at": C.now(),
    })
    row.update(artifact_paths(cell["experiment"], cell["case_id"], cell["arm"], stage))
    return row


def verify_parity(cell: dict[str, Any]) -> None:
    for label, path_field, sha_field in (
        ("trajectory", "trajectory", "trajectory_sha256"),
        ("override", "override_path", "override_sha256"),
        ("contact", "contact_mask", "contact_mask_sha256"),
    ):
        expected = cell[sha_field]
        if expected:
            actual = C.sha256(cell[path_field])
            if actual != expected:
                raise ValueError(f"{cell['case_id']} {label} SHA drift: {actual} != {expected}")
    # base scene parity (fill sha if authority did not record it)
    base_sha = C.sha256(cell["base_scene_act"])
    if cell.get("base_scene_sha256"):
        if base_sha != cell["base_scene_sha256"]:
            raise ValueError(f"{cell['case_id']} base scene SHA drift")
    else:
        cell["base_scene_sha256"] = base_sha


def build(*, apply: bool, snapshot: bool) -> int:
    # A2 pack must equal E192's frozen A2_GATE.
    sys.path.insert(0, str(C.REPO / "workspace/core4d/scripts/experiments/E192"))
    import e192_common as E192  # noqa: PLC0415
    if C.A2_GATE != E192.A2_GATE:
        raise AssertionError(f"A2_GATE differs from E192: {C.A2_GATE} != {E192.A2_GATE}")

    cells = C.build_cells()
    counts = Counter((c["tier"], c["object_key"], c["arm"]) for c in cells)
    print("tier/object/arm counts:", json.dumps({f"{k[0]}:{k[1]}:{k[2]}": v for k, v in sorted(counts.items())}, indent=2))

    # SHA parity + G1 sidecar single-variable audit
    for cell in cells:
        verify_parity(cell)
        if cell["arm"] == "G1A2":
            assert_gravcomp_diff(C.repo_path(cell["base_scene_act"]), C.repo_path(cell["scene_act"]))
        cell["effective_scene_sha256"] = C.sha256(cell["scene_act"])

    # canary = 1 representative per tier (first case of first object, lexicographic)
    # sentinel = 1 per (experiment, arm, object): lexicographically-first case_id
    canary_keys, sentinel_keys = set(), set()
    for tier, _experiment, _arm, _objects in C.TIERS:
        tier_cells = sorted((c for c in cells if c["tier"] == tier), key=lambda c: (c["object_key"], c["case_id"]))
        canary_keys.add((tier, tier_cells[0]["case_id"]))
    for (exp, arm, obj), grp in _grouped(cells, lambda c: (c["experiment"], c["arm"], c["object_key"])).items():
        sentinel_keys.add((exp, arm, obj, sorted(c["case_id"] for c in grp)[0]))

    full_rows: list[dict[str, Any]] = []
    canary_rows: list[dict[str, Any]] = []
    for cell in cells:
        is_canary = (cell["tier"], cell["case_id"]) in canary_keys
        is_sentinel = (cell["experiment"], cell["arm"], cell["object_key"], cell["case_id"]) in sentinel_keys
        full_rows.append(finalize_row(cell, "full", sentinel=is_sentinel, canary_rep=is_canary))
        if is_canary:
            canary_rows.append(finalize_row(cell, "canary", sentinel=False, canary_rep=True))

    full_rows.sort(key=lambda r: (C.TIER_RANK[r["tier"]], r["object_key"], r["case_id"]))
    canary_rows.sort(key=lambda r: (C.TIER_RANK[r["tier"]], r["object_key"], r["case_id"]))
    for i, r in enumerate(full_rows, 1):
        r["ordinal"] = i
    for i, r in enumerate(canary_rows, 1):
        r["ordinal"] = i

    # cardinality contract
    if len(full_rows) != 103:
        raise ValueError(f"expected 103 full rows, got {len(full_rows)}")
    by_tier = Counter(r["tier"] for r in full_rows)
    if by_tier != Counter({"P0": 9, "P1": 44, "P2": 44, "P3": 6}):
        raise ValueError(f"tier cardinality drift: {dict(by_tier)}")

    if snapshot and apply:
        _snapshot(full_rows)
        _git_add_scenes(full_rows)

    sentinel_rows = [dict(r) for r in full_rows if C.truth(r["sentinel"])]
    C.write_tsv(C.FULL_MANIFEST, full_rows, C.FIELDS)
    C.write_tsv(C.CANARY_MANIFEST, canary_rows, C.FIELDS)
    C.write_tsv(C.SENTINEL_MANIFEST, sentinel_rows, C.FIELDS)
    authority = [{k: r[k] for k in (
        "tier", "experiment", "arm", "object_key", "case_id", "retarget_variant_id", "target_task",
        "trajectory_sha256", "override_sha256", "contact_mask_sha256",
        "base_scene_act", "base_scene_sha256", "scene_act", "scene_name", "effective_scene_sha256",
        "extra_overrides", "gravcomp",
    )} for r in full_rows]
    C.write_tsv(C.AUTHORITY_TSV, authority)
    C.write_json(C.AUTHORITY_TSV.with_suffix(".json"), authority)
    summary = {
        "created_at": C.now(), "git_head": git_head(), "apply": apply, "snapshot": snapshot,
        "full_rows": len(full_rows), "canary_rows": len(canary_rows), "sentinel_rows": len(sentinel_rows),
        "by_tier": dict(by_tier), "by_object": dict(Counter(r["object_key"] for r in full_rows)),
        "by_experiment": dict(Counter(r["experiment"] for r in full_rows)),
        "by_arm": dict(Counter(r["arm"] for r in full_rows)),
        "a2_gate": C.A2_GATE, "budget_full": [C.FULL_SAMPLES, C.FULL_OPT_STEPS],
        "budget_canary": [C.CANARY_SAMPLES, C.CANARY_OPT_STEPS],
    }
    C.write_json(C.MANIFEST_DIR / "e198_build_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _grouped(items, key):
    out: dict[Any, list] = {}
    for it in items:
        out.setdefault(key(it), []).append(it)
    return out


def _snapshot(rows: list[dict[str, Any]]) -> None:
    manifests = {
        "E198": C.RESULTS_E198 / "scene_snapshot/g1a2",
        "E192-ext": C.RESULTS_E192 / "scene_snapshot/a2_expansion",
    }
    records: dict[str, list[str]] = {k: [] for k in manifests}
    for row in rows:
        dest_root = manifests[row["experiment"]] / row["case_id"]
        dest_root.mkdir(parents=True, exist_ok=True)
        for field in ("base_scene_act", "scene_act"):
            src = C.repo_path(row[field])
            dst = dest_root / Path(row[field]).name
            if not dst.exists():
                shutil.copy2(src, dst)
            records[row["experiment"]].append(f"{C.sha256(src)}  {row['case_id']}/{Path(row[field]).name}")
    head = git_head()
    for exp, root in manifests.items():
        (root / "manifest.txt").write_text(
            f"# {exp} scene snapshot\n# git_head={head}\n# created_at={C.now()}\n" +
            "\n".join(sorted(set(records[exp]))) + "\n", encoding="utf-8")


def _git_add_scenes(rows: list[dict[str, Any]]) -> None:
    paths = sorted({C.rel(row[f]) for row in rows for f in ("base_scene_act", "scene_act")})
    subprocess.run(["git", "add", "-f", *paths], cwd=C.REPO, check=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--snapshot", action="store_true")
    args = parser.parse_args()
    return build(apply=args.apply, snapshot=args.snapshot)


if __name__ == "__main__":
    raise SystemExit(main())
