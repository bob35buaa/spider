#!/usr/bin/env python3
"""Export E026 spider_best_E018b_E022_E025 trajectories for Holosoma RL."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
EXP_WS = SCRIPT_DIR.parents[1]
REPO_ROOT = SCRIPT_DIR.parents[3]
DEFAULT_SELECTOR = EXP_WS / "results/E026_full_eval/best_dynamic_selection.csv"
DEFAULT_MANIFEST = SCRIPT_DIR / "manifest_rl.tsv"
DEFAULT_RESULTS_ROOT = EXP_WS / "results"
DEFAULT_EXPORT_ROOT = EXP_WS / "results/E021_rl_export_manifest_rename"
DEFAULT_HOLOSOMA_ROOT = Path("/home/ubuntu/Workspace/holosoma")
DEFAULT_OUTPUT_DIR = DEFAULT_HOLOSOMA_ROOT / "workspace/data/spider_best_E018b_E022_E025_for_rl_rename"


@dataclass(frozen=True)
class ManifestRow:
    case: str
    selected_method: str
    selected_variant: str
    object_name: str
    original_prefix: str
    recommended_for_rl: str
    reason: str


@dataclass(frozen=True)
class SelectorRow:
    case: str
    selected_method: str
    selected_variant: str
    strict_success: str
    contact_proxy_pct: str
    deep_pen_pct: str
    fall: str
    obj_pos_cm: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selector", type=Path, default=DEFAULT_SELECTOR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--export-root", type=Path, default=DEFAULT_EXPORT_ROOT)
    parser.add_argument("--holosoma-root", type=Path, default=DEFAULT_HOLOSOMA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--input-fps", type=int, default=60)
    parser.add_argument("--output-fps", type=int, default=50)
    parser.add_argument("--env-index", type=int, default=0)
    parser.add_argument(
        "--qpos-layout",
        choices=("flatten", "legacy-env"),
        default="flatten",
        help="How spider_to_rl_shim should interpret 3D qpos arrays",
    )
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--only", nargs="*", default=None, help="Optional case names to export")
    parser.add_argument("--recommended-only", action="store_true", help="Only export manifest rows marked yes/maybe")
    parser.add_argument("--force", action="store_true", help="Re-run even if output exists")
    parser.add_argument(
        "--name-style",
        choices=("original-prefix", "variant"),
        default="original-prefix",
        help="Output filename style",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_manifest(path: Path) -> dict[str, ManifestRow]:
    with path.open(newline="") as f:
        rows = {}
        for row in csv.DictReader(f, delimiter="\t"):
            item = ManifestRow(
                case=row["case"],
                selected_method=row["selected_method"],
                selected_variant=row["selected_variant"],
                object_name=row["object_name"],
                original_prefix=row.get("original_prefix", ""),
                recommended_for_rl=row["recommended_for_rl"],
                reason=row["reason"],
            )
            rows[item.case] = item
    return rows


def read_selector(path: Path) -> dict[str, SelectorRow]:
    with path.open(newline="") as f:
        rows = {}
        for row in csv.DictReader(f):
            item = SelectorRow(
                case=row["case"],
                selected_method=row["selected_method"],
                selected_variant=row["selected_variant"],
                strict_success=row["strict_success"],
                contact_proxy_pct=row["contact_proxy_pct"],
                deep_pen_pct=row["deep_pen_pct"],
                fall=row["fall"],
                obj_pos_cm=row["obj_pos_cm"],
            )
            rows[item.case] = item
    return rows


def source_npz(results_root: Path, variant: str) -> Path:
    experiment = variant.split("_", 1)[0]
    return results_root / experiment / f"{variant}_outdir" / "trajectory_mjwp.npz"


def output_npz(output_dir: Path, manifest_row: ManifestRow, *, name_style: str) -> Path:
    if name_style == "variant":
        return output_dir / f"{manifest_row.selected_variant}_holosoma_rl.npz"
    if not manifest_row.original_prefix:
        raise ValueError(f"{manifest_row.case}: missing original_prefix in manifest")
    return output_dir / f"{manifest_row.original_prefix}_v2_mj_w_obj.npz"


def run_cmd(cmd: list[str], *, cwd: Path | None, timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def tail(text: str, n: int = 1200) -> str:
    text = text.strip()
    if len(text) <= n:
        return text
    return text[-n:]


def write_conversion_log(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case",
        "selected_method",
        "selected_variant",
        "object_name",
        "original_prefix",
        "recommended_for_rl",
        "source_npz",
        "shim_npz",
        "output_npz",
        "status",
        "returncode",
        "elapsed_seconds",
        "stdout_tail",
        "stderr_tail",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    selector = read_selector(args.selector)
    manifest = read_manifest(args.manifest)
    cases = args.only if args.only else list(selector)

    shim = SCRIPT_DIR / "spider_to_rl_shim.py"
    convert_script = (
        args.holosoma_root
        / "src/holosoma_retargeting/holosoma_retargeting/data_conversion/convert_data_format_mj.py"
    )
    retarget_dir = args.holosoma_root / "src/holosoma_retargeting/holosoma_retargeting"
    shim_dir = args.export_root / "shim_inputs"
    conversion_log = args.export_root / "conversion_log.csv"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shim_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    failures = 0

    for case in cases:
        if case not in selector:
            raise KeyError(f"case {case} missing from selector {args.selector}")
        if case not in manifest:
            raise KeyError(f"case {case} missing from manifest {args.manifest}")

        sel = selector[case]
        man = manifest[case]
        if sel.selected_variant != man.selected_variant:
            raise ValueError(f"{case}: selector variant {sel.selected_variant} != manifest {man.selected_variant}")
        if args.recommended_only and man.recommended_for_rl == "no":
            continue

        src = source_npz(args.results_root, sel.selected_variant)
        shim_npz = shim_dir / f"{sel.selected_variant}_qpos_fps.npz"
        out_npz = output_npz(args.output_dir, man, name_style=args.name_style)
        row = {
            "case": case,
            "selected_method": sel.selected_method,
            "selected_variant": sel.selected_variant,
            "object_name": man.object_name,
            "original_prefix": man.original_prefix,
            "recommended_for_rl": man.recommended_for_rl,
            "source_npz": str(src),
            "shim_npz": str(shim_npz),
            "output_npz": str(out_npz),
            "status": "",
            "returncode": "",
            "elapsed_seconds": "",
            "stdout_tail": "",
            "stderr_tail": "",
        }

        started = time.perf_counter()
        if not src.exists():
            row["status"] = "missing_source"
            failures += 1
            rows.append(row)
            continue

        if out_npz.exists() and not args.force:
            row["status"] = "skipped_existing"
            row["elapsed_seconds"] = "0.000"
            rows.append(row)
            continue

        shim_cmd = [
            args.python,
            str(shim),
            "--input",
            str(src),
            "--output",
            str(shim_npz),
            "--env-index",
            str(args.env_index),
            "--layout",
            args.qpos_layout,
            "--fps",
            str(float(args.input_fps)),
        ]
        convert_cmd = [
            args.python,
            str(convert_script),
            "--input_file",
            str(shim_npz),
            "--input_fps",
            str(args.input_fps),
            "--output_fps",
            str(args.output_fps),
            "--output_name",
            str(out_npz),
            "--data_format",
            "smplx",
            "--object_name",
            man.object_name,
            "--has_dynamic_object",
            "--once",
        ]

        if args.dry_run:
            row["status"] = "dry_run"
            row["stdout_tail"] = json.dumps({"shim": shim_cmd, "convert": convert_cmd}, ensure_ascii=True)
            rows.append(row)
            continue

        shim_res = run_cmd(shim_cmd, cwd=REPO_ROOT, timeout=args.timeout)
        if shim_res.returncode != 0:
            row["status"] = "shim_failed"
            row["returncode"] = str(shim_res.returncode)
            row["elapsed_seconds"] = f"{time.perf_counter() - started:.3f}"
            row["stdout_tail"] = tail(shim_res.stdout)
            row["stderr_tail"] = tail(shim_res.stderr)
            failures += 1
            rows.append(row)
            continue

        conv_res = run_cmd(convert_cmd, cwd=retarget_dir, timeout=args.timeout)
        row["returncode"] = str(conv_res.returncode)
        row["elapsed_seconds"] = f"{time.perf_counter() - started:.3f}"
        row["stdout_tail"] = tail(conv_res.stdout)
        row["stderr_tail"] = tail(conv_res.stderr)
        if conv_res.returncode == 0 and out_npz.exists():
            row["status"] = "ok"
        else:
            row["status"] = "convert_failed"
            failures += 1
        rows.append(row)
        write_conversion_log(conversion_log, rows)

    write_conversion_log(conversion_log, rows)
    print(json.dumps({"total": len(rows), "failures": failures, "conversion_log": str(conversion_log)}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
