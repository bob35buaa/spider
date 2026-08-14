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


def artifact_paths(experiment: str, case_id: str, arm: str, stage: str, scope: str = "default") -> dict[str, str]:
    tag = "E198" if experiment == "E198" else "E192ext"
    sub = "canary" if stage == "canary" else "full"
    # box001 supplement (plan227) writes to dedicated dirs, isolated from the 4-object run.
    g1a2_slug = "g1a2_box001" if scope == "box001" else "g1a2"
    a2_slug = "a2_box001" if scope == "box001" else "a2_expansion"
    if experiment == "E198":
        cem_root = f"workspace/core4d/results/E198/s6_downstream/cem/{sub}_{g1a2_slug}"
        vid_root = f"workspace/core4d/results/E198/s6_downstream/render/{sub}_{g1a2_slug}"
        log_root = f"logs/E198/cem/{sub}_{g1a2_slug}"
    else:
        cem_root = f"workspace/core4d/results/E192/s6_downstream/cem/{sub}_{a2_slug}"
        vid_root = f"workspace/core4d/results/E192/s6_downstream/render/{sub}_{a2_slug}"
        log_root = f"logs/E192/cem/{sub}_{a2_slug}"
    variant = f"{tag}_{case_id}_{arm}" + ("_canary" if stage == "canary" else "")
    return {
        "variant": variant,
        "result_npz": f"{cem_root}/{variant}.npz",
        "outdir_npz": f"{cem_root}/{variant}_outdir/trajectory_mjwp_act.npz",
        "config_act": f"{cem_root}/{variant}_outdir/config_act.yaml",
        "video": f"{vid_root}/{variant}.mp4",
        "log": f"{log_root}/{variant}.log",
    }


def finalize_row(cell: dict[str, Any], stage: str, sentinel: bool, canary_rep: bool,
                 scope: str = "default") -> dict[str, Any]:
    row: dict[str, Any] = dict(cell)
    row.update({
        "kp_pos": C.KP_POS, "kp_rot": C.KP_ROT,
        "cem_samples": C.CANARY_SAMPLES if stage == "canary" else C.FULL_SAMPLES,
        "cem_opt_steps": C.CANARY_OPT_STEPS if stage == "canary" else C.FULL_OPT_STEPS,
        "cem_seed": C.CEM_SEED, "sentinel": sentinel, "canary_representative": canary_rep,
        "gpu_id": "", "status": "READY_FOR_CANARY" if stage == "canary" else "READY_FOR_FULL",
        "failure_mode": "", "execution_mode": stage, "updated_at": C.now(),
    })
    row.update(artifact_paths(cell["experiment"], cell["case_id"], cell["arm"], stage, scope))
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


# scope -> expected (full row count, per-tier cardinality)
CARDINALITY = {
    "default": (103, Counter({"P0": 9, "P1": 44, "P2": 44, "P3": 6})),
    "box001": (56, Counter({"P0-b1": 28, "P1-b1": 28})),
}


def _meta_path(cell: dict[str, Any]) -> Path:
    """scene_act_meta.json lives in the same task dir as the base PRG scene."""
    return C.repo_path(cell["base_scene_act"]).parent / "scene_act_meta.json"


def verify_box001_meta(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Assert every box001 case runs against the E196-corrected Euler reference.

    For the 21 Euler-mismatch cases the live scene_act_meta.json euler_convention
    MUST equal the compiled_xml_axis_sequence recorded in E196's authority; the
    other 7 (never corrupted) must simply carry a valid convention. Records the
    live meta path/sha/convention + source (corrected/clean) for the authority.
    """
    corrected = C.box001_corrected_meta()  # 21 case_ids -> E196 authority row
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for cell in cells:
        cid = cell["case_id"]
        if cid in seen:
            continue
        seen.add(cid)
        meta = _meta_path(cell)
        if not meta.is_file():
            raise FileNotFoundError(f"{cid}: missing scene_act_meta.json at {meta}")
        conv = json.loads(meta.read_text()).get("euler_convention")
        if not conv:
            raise AssertionError(f"{cid}: scene_act_meta.json has no euler_convention")
        source = "E196_corrected" if cid in corrected else "E194_clean"
        if cid in corrected:
            want = corrected[cid]["compiled_xml_axis_sequence"]
            if conv != want:
                raise AssertionError(
                    f"{cid}: live euler_convention={conv} != compiled {want} "
                    f"(E196 correction NOT applied -> would recontaminate reference)")
        records.append({"case_id": cid, "meta_path": C.rel(meta),
                        "meta_sha256": C.sha256(meta), "euler_convention": conv,
                        "meta_source": source})
    n_cor = sum(1 for r in records if r["meta_source"] == "E196_corrected")
    print(f"[box001 meta] {len(records)} cases verified: {n_cor} E196_corrected, "
          f"{len(records) - n_cor} E194_clean; all conventions match compiled axes")
    return records


def build(*, apply: bool, snapshot: bool, scope: str = "default") -> int:
    # A2 pack must equal E192's frozen A2_GATE.
    sys.path.insert(0, str(C.REPO / "workspace/core4d/scripts/experiments/E192"))
    import e192_common as E192  # noqa: PLC0415
    if C.A2_GATE != E192.A2_GATE:
        raise AssertionError(f"A2_GATE differs from E192: {C.A2_GATE} != {E192.A2_GATE}")

    paths = C.manifest_paths(scope)
    cells = C.build_cells(scope)
    counts = Counter((c["tier"], c["object_key"], c["arm"]) for c in cells)
    print("tier/object/arm counts:", json.dumps({f"{k[0]}:{k[1]}:{k[2]}": v for k, v in sorted(counts.items())}, indent=2))

    # box001 (plan227): fail-closed check that the corrected Euler reference is live.
    meta_records: list[dict[str, Any]] = []
    if scope == "box001":
        meta_records = verify_box001_meta(cells)

    # SHA parity + G1 sidecar single-variable audit
    for cell in cells:
        verify_parity(cell)
        if cell["arm"] == "G1A2":
            assert_gravcomp_diff(C.repo_path(cell["base_scene_act"]), C.repo_path(cell["scene_act"]))
        cell["effective_scene_sha256"] = C.sha256(cell["scene_act"])

    # canary = 1 representative per tier (first case of first object, lexicographic)
    # sentinel = 1 per (experiment, arm, object): lexicographically-first case_id
    canary_keys, sentinel_keys = set(), set()
    for tier, _experiment, _arm, _objects in C.SCOPES[scope]:
        tier_cells = sorted((c for c in cells if c["tier"] == tier), key=lambda c: (c["object_key"], c["case_id"]))
        canary_keys.add((tier, tier_cells[0]["case_id"]))
    for (exp, arm, obj), grp in _grouped(cells, lambda c: (c["experiment"], c["arm"], c["object_key"])).items():
        sentinel_keys.add((exp, arm, obj, sorted(c["case_id"] for c in grp)[0]))

    full_rows: list[dict[str, Any]] = []
    canary_rows: list[dict[str, Any]] = []
    for cell in cells:
        is_canary = (cell["tier"], cell["case_id"]) in canary_keys
        is_sentinel = (cell["experiment"], cell["arm"], cell["object_key"], cell["case_id"]) in sentinel_keys
        full_rows.append(finalize_row(cell, "full", sentinel=is_sentinel, canary_rep=is_canary, scope=scope))
        if is_canary:
            canary_rows.append(finalize_row(cell, "canary", sentinel=False, canary_rep=True, scope=scope))

    full_rows.sort(key=lambda r: (C.TIER_RANK[r["tier"]], r["object_key"], r["case_id"]))
    canary_rows.sort(key=lambda r: (C.TIER_RANK[r["tier"]], r["object_key"], r["case_id"]))
    for i, r in enumerate(full_rows, 1):
        r["ordinal"] = i
    for i, r in enumerate(canary_rows, 1):
        r["ordinal"] = i

    # cardinality contract
    exp_n, exp_tier = CARDINALITY[scope]
    if len(full_rows) != exp_n:
        raise ValueError(f"expected {exp_n} full rows, got {len(full_rows)}")
    by_tier = Counter(r["tier"] for r in full_rows)
    if by_tier != exp_tier:
        raise ValueError(f"tier cardinality drift: {dict(by_tier)}")

    if snapshot and apply:
        _snapshot(full_rows, scope, meta_records)
        _git_add_scenes(full_rows)

    sentinel_rows = [dict(r) for r in full_rows if C.truth(r["sentinel"])]
    C.write_tsv(paths["full"], full_rows, C.FIELDS)
    C.write_tsv(paths["canary"], canary_rows, C.FIELDS)
    C.write_tsv(paths["sentinel"], sentinel_rows, C.FIELDS)
    meta_by_case = {r["case_id"]: r for r in meta_records}
    authority = [{**{k: r[k] for k in (
        "tier", "experiment", "arm", "object_key", "case_id", "retarget_variant_id", "target_task",
        "trajectory_sha256", "override_sha256", "contact_mask_sha256",
        "base_scene_act", "base_scene_sha256", "scene_act", "scene_name", "effective_scene_sha256",
        "extra_overrides", "gravcomp",
    )}, **{f"meta_{k}": meta_by_case.get(r["case_id"], {}).get(k, "")
           for k in ("source", "sha256", "euler_convention")}} for r in full_rows]
    C.write_tsv(paths["authority"], authority)
    C.write_json(paths["authority"].with_suffix(".json"), authority)
    summary = {
        "created_at": C.now(), "git_head": git_head(), "apply": apply, "snapshot": snapshot,
        "scope": scope,
        "full_rows": len(full_rows), "canary_rows": len(canary_rows), "sentinel_rows": len(sentinel_rows),
        "by_tier": dict(by_tier), "by_object": dict(Counter(r["object_key"] for r in full_rows)),
        "by_experiment": dict(Counter(r["experiment"] for r in full_rows)),
        "by_arm": dict(Counter(r["arm"] for r in full_rows)),
        "a2_gate": C.A2_GATE, "budget_full": [C.FULL_SAMPLES, C.FULL_OPT_STEPS],
        "budget_canary": [C.CANARY_SAMPLES, C.CANARY_OPT_STEPS],
        "box001_meta": {"corrected": sum(1 for r in meta_records if r["meta_source"] == "E196_corrected"),
                        "clean": sum(1 for r in meta_records if r["meta_source"] == "E194_clean")},
    }
    C.write_json(C.MANIFEST_DIR / f"e198_build_summary{'' if scope == 'default' else '_' + scope}.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _grouped(items, key):
    out: dict[Any, list] = {}
    for it in items:
        out.setdefault(key(it), []).append(it)
    return out


def _snapshot(rows: list[dict[str, Any]], scope: str = "default",
              meta_records: list[dict[str, Any]] | None = None) -> None:
    g1a2_dir = "g1a2_box001" if scope == "box001" else "g1a2"
    a2_dir = "a2_box001" if scope == "box001" else "a2_expansion"
    manifests = {
        "E198": C.RESULTS_E198 / f"scene_snapshot/{g1a2_dir}",
        "E192-ext": C.RESULTS_E192 / f"scene_snapshot/{a2_dir}",
    }
    meta_by_case = {r["case_id"]: r for r in (meta_records or [])}
    records: dict[str, list[str]] = {k: [] for k in manifests}
    for row in rows:
        dest_root = manifests[row["experiment"]] / row["case_id"]
        dest_root.mkdir(parents=True, exist_ok=True)
        fields = ["base_scene_act", "scene_act"]
        srcs = [C.repo_path(row[f]) for f in fields]
        names = [Path(row[f]).name for f in fields]
        # box001: freeze the exact run-time scene_act_meta.json (Euler reference)
        if scope == "box001" and row["case_id"] in meta_by_case:
            srcs.append(C.repo_path(meta_by_case[row["case_id"]]["meta_path"]))
            names.append("scene_act_meta.json")
        for src, name in zip(srcs, names):
            dst = dest_root / name
            if not dst.exists():
                shutil.copy2(src, dst)
            records[row["experiment"]].append(f"{C.sha256(src)}  {row['case_id']}/{name}")
    head = git_head()
    for exp, root in manifests.items():
        (root / "manifest.txt").write_text(
            f"# {exp} scene snapshot (scope={scope})\n# git_head={head}\n# created_at={C.now()}\n" +
            "\n".join(sorted(set(records[exp]))) + "\n", encoding="utf-8")


def _git_add_scenes(rows: list[dict[str, Any]]) -> None:
    paths = sorted({C.rel(row[f]) for row in rows for f in ("base_scene_act", "scene_act")})
    subprocess.run(["git", "add", "-f", *paths], cwd=C.REPO, check=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--snapshot", action="store_true")
    parser.add_argument("--scope", choices=sorted(C.SCOPES), default="default")
    args = parser.parse_args()
    return build(apply=args.apply, snapshot=args.snapshot, scope=args.scope)


if __name__ == "__main__":
    raise SystemExit(main())
