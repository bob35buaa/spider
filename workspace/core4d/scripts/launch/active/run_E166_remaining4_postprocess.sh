#!/usr/bin/env bash
# E166 remaining4 CPU-only A_B2_postSmooth generation.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

STAGE="${1:-full}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
BUILDER="workspace/core4d/scripts/experiments/E166/build_remaining4_A_A_B2_manifest.py"
VARIANTS="workspace/core4d/scripts/experiments/E166/remaining4_variants.tsv"
LOGS="logs/E166/remaining4/postprocess/${STAGE}"

if [ "$STAGE" != "full" ]; then
  echo "Invalid STAGE=$STAGE (only full is supported)" >&2
  exit 2
fi

"$PYTHON_BIN" "$BUILDER" >/dev/null
mkdir -p "$LOGS"

"$PYTHON_BIN" - "$VARIANTS" <<'PY'
import csv
import subprocess
import sys
from pathlib import Path

variants = Path(sys.argv[1])
repo = Path.cwd()
rows = list(csv.DictReader(variants.open("r", encoding="utf-8", newline=""), delimiter="\t"))
targets = [row for row in rows if row["arm_kind"] == "postprocess"]
for row in targets:
    src = repo / row["postprocess_source_npz"]
    dst = repo / row["postprocess_output_npz"]
    log = repo / "logs/E166/remaining4/postprocess/full" / f"{row['variant']}.log"
    if dst.is_file():
        print(f"[skip] {row['variant']} -> {dst}")
        continue
    if not src.is_file():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "spider/postprocess/smooth_handoff.py",
        "--input",
        str(src),
        "--output",
        str(dst),
        "--window",
        row["b2_window"] or "7",
        "--polyorder",
        row["b2_polyorder"] or "2",
    ]
    print("[run]", " ".join(cmd))
    with log.open("w", encoding="utf-8") as f:
        subprocess.run(cmd, cwd=repo, check=True, stdout=f, stderr=subprocess.STDOUT)
print(f"E166 remaining4 postprocess complete: rows={len(targets)}")
PY

"$PYTHON_BIN" "$BUILDER"
