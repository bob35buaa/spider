#!/usr/bin/env python3
"""E206 P4: install the frozen lowgeom proxy into the SOURCE templates.

Rewrites ONLY the `object_collision*` geoms inside `<body name="object">` of each
in-scope `<obj>_person{1,2}/scene.xml`, so S2 -> S3 -> S4 -> S5 -> S6 all see the
same collision body.  Patching a CEM sidecar instead would leave the S4 target
gate measuring a different geometry than the one CEM runs on.

Safety, in order of application:
  * `stripped_signature` must be IDENTICAL before/after — proves nothing outside
    the collision block moved (E175's own invariant, reused).
  * every emitted geom must compile to `mjGEOM_BOX` — `object_collision_sdf_mode=
    'union'` is fail-closed on anything else (`spider/config.py:69-79`).
  * `nq/nv/nu` must stay 43/41/29 and the model must load.
  * `--apply` is opt-in; the default is a dry run that writes a diff report.

This also repairs the two legacy 1-box templates (M5: `desk005_person2`,
`chair022_person1`), where a desk/chair was modelled as one solid AABB.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e206_common as C  # noqa: E402
import lowgeom_proxy_v2 as L  # noqa: E402

from build_nonbox_multigeom_production import (  # noqa: E402  E175 authority
    object_collision_elements,
    replace_bucket_proxy,
    stripped_signature,
)

EXPECTED_NQ, EXPECTED_NV, EXPECTED_NU = 43, 41, 29


ASSET_ROOT = C.REPO / "example_datasets/processed/core4d/assets/objects"


def ensure_asset_mesh(object_key: str, *, apply: bool) -> dict[str, Any]:
    """Materialise `assets/objects/<key>/<key>_m.obj` from `object_models/`.

    `build_or_audit_templates.py` does this (`shutil.copy2(raw_mesh, asset_mesh)`)
    only when it CREATES a template.  Four in-scope objects (desk020, desk023,
    chair005, chair021) already have templates from an earlier era but their
    asset mesh was never materialised, so their `scene.xml` does not even
    compile.  Same copy, same source, done here so the installer can proceed.

    Verified equivalent: for the three objects that do have assets, the asset
    and `object_models` meshes have identical vertex/face counts, extents and
    centroid (desk021's bytes differ only in file formatting).
    """
    dest = ASSET_ROOT / object_key / f"{object_key}_m.obj"
    src = C.object_mesh_path(object_key)
    if dest.exists():
        return {"object_key": object_key, "asset_action": "present"}
    if not src.exists():
        return {"object_key": object_key, "asset_action": "source_missing", "asset_src": str(src)}
    if apply:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        return {"object_key": object_key, "asset_action": "materialised", "asset_src": str(src)}
    return {"object_key": object_key, "asset_action": "would_materialise", "asset_src": str(src)}


def find_object_body(root: ET.Element) -> ET.Element:
    for body in root.iter("body"):
        if body.get("name") == "object":
            return body
    raise ValueError('no <body name="object"> in scene')


def install_one(
    scene_path: Path,
    object_key: str,
    n_max: int,
    target_cells: int,
    *,
    apply: bool,
) -> dict[str, Any]:
    tree = ET.parse(scene_path)
    root = tree.getroot()
    body = find_object_body(root)

    before_sig = stripped_signature(root)
    before_names = [str(g.get("name")) for g in object_collision_elements(body)]

    boxes, meta = L.build_lowgeom_boxes(
        C.object_mesh_path(object_key), object_key, n_max=n_max, target_cells=target_cells
    )
    proxy_xml, names = L.proxy_geom_xml(boxes)
    replace_bucket_proxy(body, proxy_xml)

    after_sig = stripped_signature(root)
    if before_sig != after_sig:
        raise AssertionError(
            f"{scene_path}: stripped_signature changed — the edit touched something "
            "outside the object_collision block"
        )

    result: dict[str, Any] = {
        "scene": str(scene_path.relative_to(C.REPO)),
        "object_key": object_key,
        "target_cells": target_cells,
        "geoms_before": len(before_names),
        "geoms_after": len(names),
        "signature_stable": True,
        "applied": False,
    }

    # Validate by compiling a temp copy before touching the real file.
    tmp = scene_path.with_suffix(".xml.e206tmp")
    tree.write(tmp, encoding="utf-8", xml_declaration=False)
    try:
        check = L.union_geoms_are_boxes(tmp)
        result.update(
            {
                "all_box": check["all_box"],
                "non_box": check["non_box"],
                "nq": check["nq"],
                "nv": check["nv"],
                "nu": check["nu"],
            }
        )
        if not check["all_box"]:
            raise AssertionError(f"{scene_path}: non-box object geoms {check['non_box']}")
        if (check["nq"], check["nv"], check["nu"]) != (EXPECTED_NQ, EXPECTED_NV, EXPECTED_NU):
            raise AssertionError(
                f"{scene_path}: nq/nv/nu = {check['nq']}/{check['nv']}/{check['nu']}, "
                f"expected {EXPECTED_NQ}/{EXPECTED_NV}/{EXPECTED_NU}"
            )
        if apply:
            shutil.move(str(tmp), str(scene_path))
            result["applied"] = True
    finally:
        if tmp.exists():
            tmp.unlink()

    if apply:
        info_path = scene_path.parent / "task_info.json"
        if info_path.exists():
            info = json.loads(info_path.read_text(encoding="utf-8"))
            info["collision_policy"] = meta["collision_policy"]
            info["object_collision_geom_count"] = len(names)
            info["lowgeom_target_cells"] = target_cells
            info["lowgeom_n_max"] = n_max
            info_path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
            result["task_info_updated"] = True
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contract-tsv", type=Path, default=None,
                    help="lowgeom_contract_n<N>.tsv; default = the frozen N_MAX one")
    ap.add_argument("--n-max", type=int, default=C.N_MAX_TARGET)
    ap.add_argument("--apply", action="store_true", help="write the files (default: dry run)")
    ap.add_argument("--out-dir", type=Path, default=C.S2_TEMPLATE_DIR)
    args = ap.parse_args()

    contract = args.contract_tsv or (C.S2_PROXY_DIR / f"lowgeom_contract_n{args.n_max}.tsv")
    if not contract.exists():
        raise SystemExit(f"missing contract tsv: {contract} (run audit_lowgeom_contract.py first)")

    rows = {
        r["object_key"]: r
        for r in C.read_tsv(contract)
        if r.get("build_ok") == "True" or r.get("build_ok") == "true"
    }
    if not rows:
        raise SystemExit(f"no buildable objects in {contract}")

    results: list[dict[str, Any]] = []
    errors: list[str] = []
    assets: list[dict[str, Any]] = []
    for object_key, row in sorted(rows.items()):
        assets.append(ensure_asset_mesh(object_key, apply=args.apply))
        target_cells = int(row["target_cells"])
        for person in ("person1", "person2"):
            scene = C.PROCESSED_ROOT / f"{object_key}_{person}" / "scene.xml"
            if not scene.exists():
                results.append(
                    {
                        "scene": f"{object_key}_{person}/scene.xml",
                        "object_key": object_key,
                        "status": "missing_template",
                    }
                )
                continue
            try:
                res = install_one(
                    scene, object_key, args.n_max, target_cells, apply=args.apply
                )
                res["status"] = "applied" if res["applied"] else "dry_run_ok"
                results.append(res)
            except Exception as exc:  # noqa: BLE001 - collect, do not abort the batch
                errors.append(f"{object_key}_{person}: {type(exc).__name__}: {exc}")
                results.append(
                    {
                        "scene": f"{object_key}_{person}/scene.xml",
                        "object_key": object_key,
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    C.write_tsv(
        args.out_dir / "lowgeom_install.tsv",
        results,
        ["scene", "object_key", "status", "target_cells", "geoms_before", "geoms_after",
         "signature_stable", "all_box", "nq", "nv", "nu", "applied", "error"],
    )

    by_status: dict[str, int] = {}
    for r in results:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    C.write_tsv(args.out_dir / "lowgeom_assets.tsv", assets,
                ["object_key", "asset_action", "asset_src"])
    summary = {
        "contract": str(contract),
        "assets": {a["asset_action"]: sum(1 for x in assets if x["asset_action"] == a["asset_action"])
                   for a in assets},
        "n_max": args.n_max,
        "apply": args.apply,
        "by_status": by_status,
        "missing_templates": [r["scene"] for r in results if r["status"] == "missing_template"],
        "errors": errors,
    }
    (args.out_dir / "lowgeom_install.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
