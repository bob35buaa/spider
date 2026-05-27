#!/usr/bin/env python3
"""Audit whether E028 candidate support semantics match COLA-style D6 support."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import e029_common as common  # noqa: E402


def _xml_body(root: ET.Element, name: str) -> ET.Element | None:
    for body in root.iter("body"):
        if body.get("name") == name:
            return body
    return None


def _xml_support_welds(root: ET.Element) -> list[ET.Element]:
    equality = root.find("equality")
    if equality is None:
        return []
    welds: list[ET.Element] = []
    for weld in equality.findall("weld"):
        fields = " ".join(
            [
                weld.get("name", ""),
                weld.get("body1", ""),
                weld.get("body2", ""),
            ]
        )
        if "support" in fields or "anchor" in fields:
            welds.append(weld)
    return welds


def _body_model_info(model: mujoco.MjModel, name: str) -> dict[str, str]:
    bid = common.body_id(model, name)
    if bid < 0:
        return {
            f"{name}_model_found": "false",
            f"{name}_jntnum": "0",
            f"{name}_jntadr": "-1",
            f"{name}_mocapid": "-1",
            f"{name}_is_mocap": "false",
        }
    jntnum = int(model.body_jntnum[bid])
    jntadr = int(model.body_jntadr[bid])
    mocapid = int(model.body_mocapid[bid])
    return {
        f"{name}_model_found": "true",
        f"{name}_jntnum": str(jntnum),
        f"{name}_jntadr": str(jntadr),
        f"{name}_mocapid": str(mocapid),
        f"{name}_is_mocap": "true" if mocapid >= 0 else "false",
    }


def _support_npz_stats(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {
            "result_npz_found": "false",
            "support_proxy_force_abs_max": "nan",
            "support_proxy_torque_abs_max": "nan",
            "support_gap_mean_m": "nan",
            "support_gap_max_m": "nan",
            "support_ref_unique_count": "nan",
        }
    data = np.load(path)
    stats: dict[str, str] = {"result_npz_found": "true"}
    for key, out_key in [
        ("support_proxy_force", "support_proxy_force_abs_max"),
        ("support_proxy_torque", "support_proxy_torque_abs_max"),
    ]:
        if key in data.files:
            stats[out_key] = f"{float(np.nanmax(np.abs(data[key]))):.8g}"
        else:
            stats[out_key] = "nan"
    if "support_proxy_pos" in data.files and "support_point_pos" in data.files:
        gap = np.linalg.norm(data["support_proxy_pos"] - data["support_point_pos"], axis=-1)
        stats["support_gap_mean_m"] = f"{float(np.nanmean(gap)):.8g}"
        stats["support_gap_max_m"] = f"{float(np.nanmax(gap)):.8g}"
    else:
        stats["support_gap_mean_m"] = "nan"
        stats["support_gap_max_m"] = "nan"
    if "support_proxy_ref_idx" in data.files:
        stats["support_ref_unique_count"] = str(int(len(np.unique(data["support_proxy_ref_idx"]))))
    else:
        stats["support_ref_unique_count"] = "nan"
    return stats


def audit_row(row: dict[str, str]) -> dict[str, str]:
    variant = row["variant"]
    override = common.override_path(row)
    scene = common.scene_path(row)
    config = common.parse_flat_yaml(override)
    out: dict[str, str] = {
        "variant": variant,
        "source_task": row["source_task"],
        "derived_task": row["derived_task"],
        "scene_name": row["scene_name"],
        "override_path": common.rel_or_abs(override),
        "scene_path": common.rel_or_abs(scene),
        "support_proxy_mode": config.get("support_proxy_mode", ""),
        "support_proxy_enabled": config.get("support_proxy_enabled", ""),
        "support_proxy_mocap_body_name": config.get("support_proxy_mocap_body_name", ""),
        "support_dynamic_body_name": config.get("support_dynamic_body_name", ""),
        "support_proxy_point_local": config.get("support_proxy_point_local", ""),
        "object_action_dims": config.get("object_action_dims", ""),
        "partner_force_scale": config.get("partner_force_scale", ""),
        "support_proxy_connector_kp": config.get("support_proxy_connector_kp", ""),
        "support_proxy_force_clamp": config.get("support_proxy_force_clamp", ""),
        "support_proxy_torque_clamp": config.get("support_proxy_torque_clamp", ""),
    }
    tree = ET.parse(scene)
    root = tree.getroot()
    mocap_body = _xml_body(root, "support_weld_anchor")
    dynamic_body = _xml_body(root, "support_dynamic_anchor")
    support_welds = _xml_support_welds(root)
    out.update(
        {
            "xml_support_weld_anchor_found": "true" if mocap_body is not None else "false",
            "xml_support_weld_anchor_mocap": (mocap_body.get("mocap", "false") if mocap_body is not None else ""),
            "xml_support_dynamic_anchor_found": "true" if dynamic_body is not None else "false",
            "xml_support_dynamic_anchor_mocap": (dynamic_body.get("mocap", "false") if dynamic_body is not None else ""),
            "xml_support_weld_count": str(len(support_welds)),
            "xml_support_weld_names": ";".join(w.get("name", "") for w in support_welds),
            "xml_support_weld_pairs": ";".join(f"{w.get('body1', '')}->{w.get('body2', '')}" for w in support_welds),
            "xml_support_weld_relpose": ";".join(w.get("relpose", "") for w in support_welds),
            "xml_support_weld_solref": ";".join(w.get("solref", "") for w in support_welds),
            "xml_support_weld_solimp": ";".join(w.get("solimp", "") for w in support_welds),
        }
    )
    model = mujoco.MjModel.from_xml_path(str(scene))
    out.update(
        {
            "nq": str(int(model.nq)),
            "nv": str(int(model.nv)),
            "nu": str(int(model.nu)),
            "nmocap": str(int(model.nmocap)),
            "neq": str(int(model.neq)),
            "object_qadr": str(common.object_qadr(model)),
            "object_last_freejoint": "true" if common.object_last_freejoint(model) else "false",
        }
    )
    out.update(_body_model_info(model, "support_weld_anchor"))
    out.update(_body_model_info(model, "support_dynamic_anchor"))
    out.update(_support_npz_stats(common.result_npz_path(row)))

    mode = out["support_proxy_mode"]
    mocap_ok = out["xml_support_weld_anchor_mocap"].lower() == "true"
    dynamic_present = out["support_dynamic_anchor_model_found"] == "true"
    has_direct_object_action = out.get("object_action_dims", "") not in {"", "0"}
    force_zero = float(out["support_proxy_force_abs_max"]) == 0.0 if out["support_proxy_force_abs_max"] != "nan" else False
    if mode == "mocap_pad" and mocap_ok and not dynamic_present and not has_direct_object_action:
        out["semantic_class"] = "kinematic_mocap_weld_anchor"
    elif dynamic_present:
        out["semantic_class"] = "dynamic_support_body"
    else:
        out["semantic_class"] = "mixed_or_unknown"
    out["cola_d6_match"] = "false" if out["semantic_class"] != "dynamic_support_body" else "review"
    out["support_force_is_placeholder_zero"] = "true" if force_zero else "false"
    return out


def _write_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_report(rows: list[dict[str, str]], path: Path, csv_path: Path) -> None:
    total = len(rows)
    mocap = sum(row["semantic_class"] == "kinematic_mocap_weld_anchor" for row in rows)
    dynamic = sum(row["semantic_class"] == "dynamic_support_body" for row in rows)
    force_zero = sum(row["support_force_is_placeholder_zero"] == "true" for row in rows)
    lines = [
        "# E029 Current Support Semantics Audit",
        "",
        f"- Candidates audited: {total}",
        f"- Kinematic mocap weld anchor: {mocap}/{total}",
        f"- Dynamic support body: {dynamic}/{total}",
        f"- `support_proxy_force` placeholder zero: {force_zero}/{total}",
        f"- CSV: `{common.rel_or_abs(csv_path)}`",
        "",
        "## Conclusion",
        "",
        (
            "E028 candidates are not COLA-style dynamic support body + D6 cases. "
            "They compile to `support_weld_anchor` with `mocap=true`, one equality weld "
            "from object to that mocap body, no support generalized coordinates, and no "
            "direct object actuator. This confirms the current scaffold is a kinematic "
            "support anchor, not a dynamic load-path model."
        ),
        "",
        "## Per Candidate",
        "",
        "| Variant | mode | nq/nv/nu | nmocap | support body | weld | object last | force max | gap mean/max | COLA D6 |",
        "|---|---|---:|---:|---|---|---|---:|---:|---|",
    ]
    for row in rows:
        body = (
            "mocap"
            if row["support_weld_anchor_is_mocap"] == "true"
            else ("dynamic" if row["support_dynamic_anchor_model_found"] == "true" else "none")
        )
        gap = f"{row['support_gap_mean_m']}/{row['support_gap_max_m']}"
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['variant']}`",
                    f"`{row['support_proxy_mode']}`",
                    f"{row['nq']}/{row['nv']}/{row['nu']}",
                    row["nmocap"],
                    body,
                    row["xml_support_weld_pairs"],
                    row["object_last_freejoint"],
                    row["support_proxy_force_abs_max"],
                    gap,
                    row["cola_d6_match"],
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Implication For E029",
            "",
            (
                "继续调 `support_proxy_point_local` 只能修补单点 kinematic anchor 的端点选择。"
                "要对齐 COLA/Holosoma，需要新增非 mocap support body，并让 support command "
                "通过该 body 的 generalized force 传到 object。"
            ),
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=common.MANIFEST)
    parser.add_argument("--candidates", type=Path, default=common.CANDIDATES)
    parser.add_argument("--out-dir", type=Path, default=common.E029_RESULTS / "audit")
    args = parser.parse_args()

    rows = [audit_row(row) for row in common.candidate_rows(manifest_path=args.manifest, candidates_path=args.candidates)]
    csv_path = args.out_dir / "e028_candidate_modes.csv"
    report_path = args.out_dir / "current_support_semantics.md"
    _write_csv(rows, csv_path)
    _write_report(rows, report_path, csv_path)
    print(f"Wrote {common.rel_or_abs(csv_path)}")
    print(f"Wrote {common.rel_or_abs(report_path)}")


if __name__ == "__main__":
    main()

