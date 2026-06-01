#!/usr/bin/env python3
"""Check the minimal environment contract for Core4D data-construction v3."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

SCRIPT_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "data_construction_v3")
for _path in (SCRIPT_ROOT / "lib", SCRIPT_ROOT / "state", SCRIPT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
from typing import Any

from common import (
    command_exists,
    find_spider_repo,
    git_state,
    package_version,
    resolve_holosoma_repo,
    resolve_run_root,
    timestamp,
    write_json,
)


def check_path(label: str, path: Path, required: bool = True) -> dict[str, Any]:
    ok = path.exists()
    return {
        "label": label,
        "path": str(path),
        "required": str(required).lower(),
        "ok": str(ok or not required).lower(),
        "exists": str(ok).lower(),
        "is_file": str(path.is_file()).lower(),
        "is_dir": str(path.is_dir()).lower(),
    }


def check_writable(path: Path) -> dict[str, Any]:
    path.mkdir(parents=True, exist_ok=True)
    probe = path / ".write_probe"
    try:
        probe.write_text("ok\n", encoding="utf-8")
        probe.unlink()
        ok = True
        error = ""
    except Exception as exc:
        ok = False
        error = str(exc)
    return {"label": "run_root_writable", "path": str(path), "ok": str(ok).lower(), "error": error}


def check_mujoco_load(scene: Path, allow_missing: bool) -> dict[str, Any]:
    result: dict[str, Any] = {
        "label": "mujoco_guard_scene_load",
        "scene": str(scene),
        "required": str(not allow_missing).lower(),
        "ok": "false",
        "error": "",
    }
    if not scene.is_file():
        result["error"] = "guard scene missing"
        result["ok"] = str(allow_missing).lower()
        return result
    try:
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(scene))
        result.update({"ok": "true", "nq": model.nq, "nv": model.nv, "nu": model.nu})
    except Exception as exc:
        result["error"] = str(exc)
    return result


def check_retarget_python(holosoma_repo: Path) -> dict[str, Any]:
    setup = holosoma_repo / "scripts/source_retargeting_setup.sh"
    result: dict[str, Any] = {
        "label": "retarget_python_imports",
        "setup": str(setup),
        "ok": "false",
        "python": "",
        "error": "",
    }
    if not setup.is_file():
        result["error"] = "source_retargeting_setup.sh missing"
        return result
    script = f"""
set -euo pipefail
source {str(setup)!r} >/tmp/core4d_dcv3_retarget_env_check.log 2>&1
RETARGET_PY="${{RETARGET_PYTHON_BIN:-}}"
if [ -z "$RETARGET_PY" ] && [ -n "${{CONDA_PREFIX:-}}" ]; then
  RETARGET_PY="$CONDA_PREFIX/bin/python"
fi
if [ -z "$RETARGET_PY" ]; then
  RETARGET_PY="$(command -v python)"
fi
"$RETARGET_PY" - <<'PY'
import sys
import smplx
import holosoma_retargeting
print(sys.executable)
PY
"""
    completed = subprocess.run(["bash", "-lc", script], cwd=holosoma_repo, text=True, capture_output=True, check=False)
    if completed.returncode == 0:
        result["ok"] = "true"
        result["python"] = completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else ""
    else:
        result["error"] = (completed.stderr or completed.stdout).strip()[-1000:]
    return result


def markdown_summary(report: dict[str, Any]) -> str:
    lines = [
        "# Core4D data_construction_v3 环境检查",
        "",
        f"- 时间：{report['created_at']}",
        f"- Spider repo：`{report['spider_repo']}`",
        f"- Holosoma repo：`{report['holosoma_repo']}`",
        f"- CORE4D raw root：`{report['core4d_raw_root']}`",
        f"- run root：`{report['run_root']}`",
        f"- 总体状态：`{report['overall']}`",
        "",
        "## 检查项",
        "",
        "| 项 | 状态 | 路径/说明 |",
        "|---|---|---|",
    ]
    for item in report["checks"]:
        status = "PASS" if item.get("ok") == "true" else "FAIL"
        detail = item.get("path") or item.get("scene") or item.get("error") or ""
        lines.append(f"| {item['label']} | {status} | `{detail}` |")
    if report["warnings"]:
        lines.extend(["", "## 警告", ""])
        lines.extend(f"- {warning}" for warning in report["warnings"])
    if report["errors"]:
        lines.extend(["", "## 错误", ""])
        lines.extend(f"- {error}" for error in report["errors"])
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spider-repo", type=Path, default=None)
    parser.add_argument("--holosoma-repo", type=Path, default=None)
    parser.add_argument("--core4d-raw-root", type=Path, default=None)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--allow-missing-raw", action="store_true")
    parser.add_argument("--allow-missing-guard-scene", action="store_true")
    parser.add_argument(
        "--guard-scene",
        type=Path,
        default=Path("example_datasets/processed/core4d/unitree_g1/humanoid_object/box023_person1/scene.xml"),
    )
    args = parser.parse_args()

    spider_repo = (args.spider_repo or find_spider_repo()).resolve()
    holosoma_repo = (args.holosoma_repo or resolve_holosoma_repo()).resolve()
    core4d_raw_root = (
        args.core4d_raw_root
        or (Path(os.environ["CORE4D_RAW_ROOT"]).expanduser() if os.environ.get("CORE4D_RAW_ROOT") else None)
    )
    run_root = (args.run_root or resolve_run_root(holosoma_repo)).resolve()
    out_dir = (args.out_dir or (run_root / "_environment_check")).resolve()
    guard_scene = args.guard_scene
    if not guard_scene.is_absolute():
        guard_scene = spider_repo / guard_scene

    checks: list[dict[str, Any]] = []
    checks.append(check_path("spider_repo", spider_repo))
    checks.append(check_path("holosoma_repo", holosoma_repo))
    checks.append(
        check_path(
            "core4d_raw_root",
            core4d_raw_root.resolve() if core4d_raw_root else Path("<CORE4D_RAW_ROOT unset>"),
            required=not args.allow_missing_raw,
        )
    )
    checks.append(check_writable(run_root))
    for command in ["ffmpeg", "ffprobe"]:
        checks.append({"label": f"command_{command}", "ok": str(command_exists(command)).lower(), "command": command})
    checks.extend(
        [
            check_path("holosoma_source_retargeting_setup", holosoma_repo / "scripts/source_retargeting_setup.sh"),
            check_path(
                "core4d_to_omniretarget_converter",
                holosoma_repo / "workspace/pipeline/convert_core4d_to_omniretarget.py",
            ),
            check_path("spider_core4d_processor", spider_repo / "spider/process_datasets/core4d.py"),
        ]
    )
    checks.append(check_retarget_python(holosoma_repo))
    checks.append(check_mujoco_load(guard_scene, args.allow_missing_guard_scene))

    packages = {name: package_version(name) for name in ["numpy", "scipy", "trimesh", "mujoco", "torch", "smplx"]}
    warnings = []
    errors = []
    for item in checks:
        if item.get("ok") != "true":
            msg = f"{item['label']} failed: {item.get('path') or item.get('scene') or item.get('command') or ''} {item.get('error', '')}".strip()
            if item.get("required", "true") == "false":
                warnings.append(msg)
            else:
                errors.append(msg)

    report: dict[str, Any] = {
        "created_at": timestamp(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "spider_repo": str(spider_repo),
        "holosoma_repo": str(holosoma_repo),
        "core4d_raw_root": str(core4d_raw_root.resolve()) if core4d_raw_root else "",
        "run_root": str(run_root),
        "out_dir": str(out_dir),
        "overall": "PASS" if not errors else "FAIL",
        "checks": checks,
        "packages": packages,
        "spider_git": git_state(spider_repo),
        "holosoma_git": git_state(holosoma_repo),
        "warnings": warnings,
        "errors": errors,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "environment_check.json", report)
    (out_dir / "environment_check.md").write_text(markdown_summary(report), encoding="utf-8")
    print(json.dumps({"overall": report["overall"], "out_dir": str(out_dir), "errors": errors}, ensure_ascii=False))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
