#!/usr/bin/env python3
"""Add Holosoma partner-hand fields to paired Spider RL exports."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
EXP_WS = SCRIPT_DIR.parents[1]
DEFAULT_MANIFEST = SCRIPT_DIR / "manifest_rl.tsv"
DEFAULT_OUTPUT_DIR = Path("/home/ubuntu/Workspace/holosoma/workspace/data/spider_best_E018b_E022_E025_for_rl_rename")
DEFAULT_PARTNER_SCRIPT = Path("/home/ubuntu/Workspace/holosoma/workspace/v2/scripts/add_partner_hands_to_motion.py")
DEFAULT_LOG = EXP_WS / "results/E021_rl_export_manifest_rename/partner_log.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--partner-script", type=Path, default=DEFAULT_PARTNER_SCRIPT)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_manifest(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as f:
        return {row["case"]: row for row in csv.DictReader(f, delimiter="\t")}


def base_npz(output_dir: Path, row: dict[str, str]) -> Path:
    prefix = row.get("original_prefix") or row["selected_variant"]
    return output_dir / f"{prefix}_v2_mj_w_obj.npz"


def partner_npz(base: Path) -> Path:
    suffix = "_v2_mj_w_obj.npz"
    if not base.name.endswith(suffix):
        raise ValueError(f"unexpected base filename: {base.name}")
    return base.with_name(base.name[: -len(suffix)] + "_v2_mj_w_obj_w_partner.npz")


def paired_other_case(case: str) -> str | None:
    if case.endswith("_p1"):
        return f"{case[:-3]}_p2"
    if case.endswith("_p2"):
        return f"{case[:-3]}_p1"
    return None


def tail(text: str, n: int = 1200) -> str:
    text = text.strip()
    return text if len(text) <= n else text[-n:]


def main() -> int:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    failures = 0

    for case, p1_row in manifest.items():
        partner_case = paired_other_case(case)
        if not partner_case:
            continue

        base_motion_npz = base_npz(args.output_dir, p1_row)
        partner_row = manifest.get(partner_case)
        partner_motion_npz = base_npz(args.output_dir, partner_row) if partner_row else None
        out_npz = partner_npz(base_motion_npz)
        row = {
            "case": case,
            "partner_case": partner_case,
            "person1_npz": str(base_motion_npz),
            "person2_npz": str(partner_motion_npz) if partner_motion_npz else "",
            "output_npz": str(out_npz),
            "status": "",
            "returncode": "",
            "elapsed_seconds": "",
            "stdout_tail": "",
            "stderr_tail": "",
        }

        if not partner_row:
            row["status"] = "missing_partner_manifest"
            rows.append(row)
            continue
        if not base_motion_npz.exists():
            row["status"] = "missing_person1_npz"
            failures += 1
            rows.append(row)
            continue
        if not partner_motion_npz.exists():
            row["status"] = "missing_person2_npz"
            failures += 1
            rows.append(row)
            continue
        if out_npz.exists() and not args.force:
            row["status"] = "skipped_existing"
            rows.append(row)
            continue

        cmd = [
            args.python,
            str(args.partner_script),
            "--person1_npz",
            str(base_motion_npz),
            "--person2_npz",
            str(partner_motion_npz),
            "--output",
            str(out_npz),
        ]
        started = time.perf_counter()
        proc = subprocess.run(cmd, text=True, capture_output=True, check=False, timeout=args.timeout)
        row["returncode"] = str(proc.returncode)
        row["elapsed_seconds"] = f"{time.perf_counter() - started:.3f}"
        row["stdout_tail"] = tail(proc.stdout)
        row["stderr_tail"] = tail(proc.stderr)
        row["status"] = "ok" if proc.returncode == 0 else "failed"
        if proc.returncode != 0:
            failures += 1
        rows.append(row)

    fieldnames = [
        "case",
        "partner_case",
        "person1_npz",
        "person2_npz",
        "output_npz",
        "status",
        "returncode",
        "elapsed_seconds",
        "stdout_tail",
        "stderr_tail",
    ]
    with args.log.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "total_pairs": len(rows),
        "ok": sum(1 for row in rows if row["status"] in {"ok", "skipped_existing"}),
        "failures": failures,
        "log": str(args.log),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
