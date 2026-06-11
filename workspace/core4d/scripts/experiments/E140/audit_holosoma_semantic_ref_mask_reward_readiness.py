#!/usr/bin/env python3
"""Audit Holosoma semantic ref-mask reward/config readiness after E139."""

from __future__ import annotations

import csv
import ast
import json
import re
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[4]
HOLOSOMA = Path("/home/ubuntu/Workspace/holosoma")
OUT_DIR = REPO / "workspace/core4d/results/E140/holosoma_semantic_ref_mask_reward_readiness"
OUT_TSV = OUT_DIR / "e140_ref_mask_reward_readiness.tsv"
SUMMARY_JSON = OUT_DIR / "e140_ref_mask_reward_readiness_summary.json"
SUMMARY_MD = OUT_DIR / "e140_ref_mask_reward_readiness_summary.md"

REWARD_TERMS = HOLOSOMA / "src/holosoma/holosoma/managers/reward/terms/wbt.py"
TERMINATION_TERMS = HOLOSOMA / "src/holosoma/holosoma/managers/termination/terms/wbt.py"
REWARD_CONFIG = HOLOSOMA / "src/holosoma/holosoma/config_values/wbt/g1/reward.py"
TERMINATION_CONFIG = HOLOSOMA / "src/holosoma/holosoma/config_values/wbt/g1/termination.py"
EXPERIMENT_CONFIG = HOLOSOMA / "src/holosoma/holosoma/config_values/wbt/g1/experiment.py"
E139_SUMMARY = (
    REPO
    / "workspace/core4d/results/E139/e138_semantic_ref_object_contact_env_probe/"
    / "e139_ref_object_contact_summary.json"
)
E139_PARTNER_MANIFEST = (
    REPO
    / "workspace/core4d/results/E139/e138_semantic_ref_object_contact_env_probe/"
    / "e139_partner_injection_manifest.tsv"
)

TARGET_EXPERIMENTS = {
    "g1_29dof_wbt_w_object_r135_box021_handbox_exp0601_v4_3",
    "g1_29dof_wbt_w_object_r138_box021_r095_loadpath_partner_v4_3",
}

FIELDS = [
    "row_type",
    "name",
    "file",
    "line",
    "status",
    "uses_ref_object_contact",
    "class_or_symbol",
    "reward_config",
    "experiment_config",
    "details",
]


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def line_number(text: str, needle: str) -> int:
    idx = text.find(needle)
    if idx < 0:
        return 0
    return text[:idx].count("\n") + 1


def class_blocks(text: str) -> dict[str, str]:
    matches = list(re.finditer(r"^class\s+(\w+)\b", text, flags=re.MULTILINE))
    out: dict[str, str] = {}
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        out[match.group(1)] = text[start:end]
    return out


def config_blocks(text: str) -> dict[str, str]:
    pattern = re.compile(r"^(g1_29dof_wbt_reward_[A-Za-z0-9_]+)\s*=\s*RewardManagerCfg\(", re.MULTILINE)
    matches = list(pattern.finditer(text))
    out: dict[str, str] = {}
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        out[match.group(1)] = text[start:end]
    return out


def assignment_blocks(text: str, prefix: str) -> dict[str, str]:
    out: dict[str, str] = {}
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        names = [target.id for target in node.targets if isinstance(target, ast.Name)]
        if not names:
            continue
        start = offsets[node.lineno - 1] + node.col_offset
        end = offsets[getattr(node, "end_lineno", node.lineno) - 1] + getattr(node, "end_col_offset", 0)
        segment = text[start:end]
        for name in names:
            if name.startswith(prefix):
                out[name] = segment
    return out


def experiment_blocks(text: str) -> dict[str, str]:
    pattern = re.compile(r"^(g1_29dof_wbt_w_object_[A-Za-z0-9_]+)\s*=\s*replace\(", re.MULTILINE)
    matches = list(pattern.finditer(text))
    out: dict[str, str] = {}
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        out[match.group(1)] = text[start:end]
    return out


def reward_symbol_for_experiment(block: str) -> str:
    matches = re.findall(r"reward\s*=\s*reward\.([A-Za-z0-9_]+)", block)
    return matches[-1] if matches else ""


def write_tsv(rows: list[dict[str, Any]]) -> None:
    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_TSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def read_e139_summary() -> dict[str, Any]:
    if not E139_SUMMARY.exists():
        return {}
    return json.loads(E139_SUMMARY.read_text(encoding="utf-8"))


def partner_rows() -> int:
    if not E139_PARTNER_MANIFEST.exists():
        return 0
    with E139_PARTNER_MANIFEST.open("r", encoding="utf-8", newline="") as f:
        return sum(1 for row in csv.DictReader(f, delimiter="\t") if row.get("partner_injection_status") == "pass")


def main() -> int:
    reward_terms_text = read_text(REWARD_TERMS)
    termination_terms_text = read_text(TERMINATION_TERMS)
    reward_config_text = read_text(REWARD_CONFIG)
    experiment_config_text = read_text(EXPERIMENT_CONFIG)
    termination_config_text = read_text(TERMINATION_CONFIG)

    rows: list[dict[str, Any]] = []

    ref_reward_classes = []
    for cls_name, block in class_blocks(reward_terms_text).items():
        if "ref_object_contact" in block:
            ref_reward_classes.append(cls_name)
            rows.append(
                {
                    "row_type": "reward_term_class",
                    "name": cls_name,
                    "file": str(REWARD_TERMS),
                    "line": line_number(reward_terms_text, f"class {cls_name}"),
                    "status": "present",
                    "uses_ref_object_contact": "true",
                    "class_or_symbol": cls_name,
                    "details": "reward class reads motion_command.ref_object_contact",
                }
            )

    ref_termination_classes = []
    for cls_name, block in class_blocks(termination_terms_text).items():
        if "ref_object_contact" in block:
            ref_termination_classes.append(cls_name)
            rows.append(
                {
                    "row_type": "termination_term_class",
                    "name": cls_name,
                    "file": str(TERMINATION_TERMS),
                    "line": line_number(termination_terms_text, f"class {cls_name}"),
                    "status": "present",
                    "uses_ref_object_contact": "true",
                    "class_or_symbol": cls_name,
                    "details": "termination class reads motion_command.ref_object_contact",
                }
            )

    reward_blocks = config_blocks(reward_config_text)
    helper_blocks = assignment_blocks(reward_config_text, "_")
    ref_helper_symbols = [
        symbol
        for symbol, block in helper_blocks.items()
        if any(cls in block for cls in ref_reward_classes)
    ]
    ref_config_symbols = []
    for symbol, block in reward_blocks.items():
        used = [cls for cls in ref_reward_classes if cls in block]
        helper_used = [helper for helper in ref_helper_symbols if helper in block]
        if helper_used and not used:
            used = ["via:" + ",".join(helper_used)]
        if used:
            ref_config_symbols.append(symbol)
            rows.append(
                {
                    "row_type": "reward_config",
                    "name": symbol,
                    "file": str(REWARD_CONFIG),
                    "line": line_number(reward_config_text, f"{symbol} ="),
                    "status": "uses_ref_mask_reward",
                    "uses_ref_object_contact": "true",
                    "class_or_symbol": ",".join(used),
                    "reward_config": symbol,
                    "details": "reward config includes ref-mask reward class",
                }
            )

    experiment_blocks_map = experiment_blocks(experiment_config_text)
    target_status: dict[str, dict[str, Any]] = {}
    for exp_name in sorted(TARGET_EXPERIMENTS):
        block = experiment_blocks_map.get(exp_name, "")
        reward_symbol = reward_symbol_for_experiment(block)
        reward_block = reward_blocks.get(reward_symbol, "")
        used = [cls for cls in ref_reward_classes if cls in reward_block]
        helper_used = [helper for helper in ref_helper_symbols if helper in reward_block]
        if helper_used and not used:
            used = ["via:" + ",".join(helper_used)]
        status = "uses_ref_mask_reward" if used else "missing_ref_mask_reward"
        target_status[exp_name] = {
            "reward_symbol": reward_symbol,
            "status": status,
            "used_classes": used,
        }
        rows.append(
            {
                "row_type": "target_experiment_config",
                "name": exp_name,
                "file": str(EXPERIMENT_CONFIG),
                "line": line_number(experiment_config_text, f"{exp_name} ="),
                "status": status,
                "uses_ref_object_contact": str(bool(used)).lower(),
                "class_or_symbol": ",".join(used),
                "reward_config": reward_symbol,
                "experiment_config": exp_name,
                "details": "target Box021 experiment config reward wiring",
            }
        )

    termination_has_ref = any(cls in termination_config_text for cls in ref_termination_classes)
    rows.append(
        {
            "row_type": "termination_config_inventory",
            "name": "g1_termination_configs",
            "file": str(TERMINATION_CONFIG),
            "line": 1,
            "status": "uses_ref_mask_termination" if termination_has_ref else "missing_ref_mask_termination",
            "uses_ref_object_contact": str(termination_has_ref).lower(),
            "class_or_symbol": ",".join(ref_termination_classes),
            "details": "inventory-level check for ref-mask termination in g1 termination configs",
        }
    )

    e139 = read_e139_summary()
    e139_ready = e139.get("status") == "pass" and int(e139.get("semantic_ref_mask_runtime_ready_rows", 0)) == 4
    rows.append(
        {
            "row_type": "e139_artifact_gate",
            "name": "E139 semantic runtime artifacts",
            "file": str(E139_SUMMARY),
            "line": 1,
            "status": "pass" if e139_ready and partner_rows() == 4 else "fail",
            "uses_ref_object_contact": "true",
            "details": (
                f"E139 status={e139.get('status')}; "
                f"semantic_ref_mask_runtime_ready_rows={e139.get('semantic_ref_mask_runtime_ready_rows')}; "
                f"partner_injection_pass_rows={partner_rows()}"
            ),
        }
    )

    target_configs_ready = all(item["status"] == "uses_ref_mask_reward" for item in target_status.values())
    recommendation = (
        "existing_config_ready_for_reward_probe"
        if target_configs_ready and e139_ready
        else "needs_ref_mask_reward_config_variant"
    )
    summary = {
        "experiment": "E140",
        "status": "pass",
        "reward_ref_contact_classes": ref_reward_classes,
        "termination_ref_contact_classes": ref_termination_classes,
        "reward_ref_contact_helper_symbols": sorted(ref_helper_symbols),
        "reward_configs_using_ref_mask": sorted(ref_config_symbols),
        "target_experiment_status": target_status,
        "e139_runtime_artifacts_ready": e139_ready,
        "e139_partner_injection_pass_rows": partner_rows(),
        "target_configs_ready": target_configs_ready,
        "recommendation": recommendation,
        "rl_ready_rows": 0,
        "training_launched": False,
        "cem_launched": False,
        "remote_jobs_launched": False,
        "notes": [
            "E140 is a static/source/config audit only.",
            "R135/R138 readiness requires the target reward configs to include ref-mask reward terms.",
            "Static readiness is not PPO readiness.",
        ],
    }

    write_tsv(rows)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# E140 Holosoma Semantic Ref-Mask Reward Readiness Summary",
        "",
        "| metric | value |",
        "|---|---|",
        f"| reward classes using ref_object_contact | `{', '.join(ref_reward_classes)}` |",
        f"| termination classes using ref_object_contact | `{', '.join(ref_termination_classes)}` |",
        f"| helper symbols using ref-mask reward | `{', '.join(sorted(ref_helper_symbols))}` |",
        f"| reward configs using ref-mask reward | `{', '.join(sorted(ref_config_symbols))}` |",
        f"| E139 runtime artifacts ready | `{str(e139_ready).lower()}` |",
        f"| target configs ready | `{str(target_configs_ready).lower()}` |",
        f"| recommendation | `{recommendation}` |",
        "| RL-ready rows | `0` |",
        "| training launched | `false` |",
        "| CEM launched | `false` |",
        "",
        "## Target Experiments",
        "",
        "| experiment | reward config | status | ref-mask classes |",
        "|---|---|---|---|",
    ]
    for exp_name, item in sorted(target_status.items()):
        lines.append(
            f"| `{exp_name}` | `{item['reward_symbol']}` | `{item['status']}` | "
            f"`{', '.join(item['used_classes'])}` |"
        )
    lines.extend(
        [
            "",
            "Interpretation: E140 does not launch PPO/CEM. It only decides whether the post-E139 semantic mask can be used by existing reward configs or whether a new config variant is needed.",
            "",
        ]
    )
    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {SUMMARY_MD} recommendation={recommendation}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
