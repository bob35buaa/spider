#!/usr/bin/env python3
"""Verify an E175 isolated source snapshot against its SHA inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    root = args.root.resolve()
    failures = []
    passed = 0
    for artifact in inventory["artifacts"]:
        relative = Path(artifact["path"])
        if relative.is_absolute() or ".." in relative.parts:
            failures.append(
                {"path": str(relative), "failure": "unsafe_relative_path"}
            )
            continue
        path = root / relative
        if not path.is_file():
            failures.append(
                {"path": str(relative), "failure": "missing"}
            )
            continue
        if path.stat().st_size != int(artifact["bytes"]):
            failures.append(
                {"path": str(relative), "failure": "size_mismatch"}
            )
            continue
        if sha256(path) != artifact["sha256"]:
            failures.append(
                {"path": str(relative), "failure": "sha256_mismatch"}
            )
            continue
        passed += 1

    status = (
        "pass"
        if not failures and passed == int(inventory["files"])
        else "fail"
    )
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "status": status,
        "inventory": str(args.inventory),
        "root": str(root),
        "expected_files": int(inventory["files"]),
        "passed_files": passed,
        "failed_files": len(failures),
        "failures": failures,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "status": status,
                "expected_files": payload["expected_files"],
                "passed_files": passed,
                "failed_files": len(failures),
            },
            sort_keys=True,
        )
    )
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
