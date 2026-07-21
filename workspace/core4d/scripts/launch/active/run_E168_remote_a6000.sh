#!/usr/bin/env bash
# E168 A6000 exact-input launcher. Canary only until local/remote canary passes.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODE="${1:-canary}"
REMOTE="${REMOTE:-spider-remote}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/xiayb/pHRI_workspace/spider}"
SESSION="${SESSION:-E168_a6000_${MODE}_$(date +%Y%m%d_%H%M%S)}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
GPU_MEM_LIMIT_MB="${E168_A6000_GPU_MEM_LIMIT_MB:-5000}"
MANIFEST_DIR="workspace/core4d/results/E168/s6_downstream/cem/manifests"
MANIFEST="${MANIFEST_DIR}/cem_canary_manifest.tsv"
RUNNER="workspace/core4d/scripts/experiments/E168/run_cem_queue.py"
SELECTION_TSV="workspace/core4d/results/E168/s0_environment/a6000_canary_gpu_selection.tsv"

if [ "$MODE" != "canary" ]; then
  echo "E168 A6000 launcher currently only supports canary. Full production needs canary pass first." >&2
  exit 2
fi

retry() {
  local attempt=1
  local max_attempts="${E168_REMOTE_RETRY_MAX:-4}"
  while true; do
    if "$@"; then
      return 0
    fi
    if [ "$attempt" -ge "$max_attempts" ]; then
      echo "Command failed after ${attempt} attempts: $*" >&2
      return 1
    fi
    echo "Retry ${attempt}/${max_attempts}: $*" >&2
    sleep "$((attempt * 3))"
    attempt=$((attempt + 1))
  done
}

select_gpu() {
  ssh "$REMOTE" "E168_GPU_MEM_LIMIT_MB='$GPU_MEM_LIMIT_MB' python3 -" <<'PY'
import os
import subprocess
import sys

LIMIT = int(os.environ.get("E168_GPU_MEM_LIMIT_MB", "5000"))

def cmd(args):
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()

gpu_raw = cmd([
    "nvidia-smi",
    "--query-gpu=index,uuid,memory.used,utilization.gpu",
    "--format=csv,noheader,nounits",
])
try:
    compute_raw = cmd([
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ])
except subprocess.CalledProcessError:
    compute_raw = ""

busy = set()
for line in compute_raw.splitlines():
    parts = [p.strip() for p in line.split(",")]
    if parts and parts[0]:
        busy.add(parts[0])

rows = []
for line in gpu_raw.splitlines():
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 4:
        continue
    rows.append({
        "index": int(parts[0]),
        "uuid": parts[1],
        "memory_used_mb": int(parts[2]),
        "utilization_gpu_pct": int(parts[3]),
        "has_compute_process": parts[1] in busy,
    })

available = [
    row for row in rows
    if row["memory_used_mb"] < LIMIT and not row["has_compute_process"]
]
if not available:
    for row in rows:
        print(
            f"gpu={row['index']} mem={row['memory_used_mb']} util={row['utilization_gpu_pct']} compute={row['has_compute_process']}",
            file=sys.stderr,
        )
    raise SystemExit(1)
available.sort(key=lambda row: (row["memory_used_mb"], row["index"]))
print(available[0]["index"])
PY
}

retry_select_gpu() {
  local attempt=1
  local max_attempts="${E168_REMOTE_RETRY_MAX:-4}"
  while true; do
    if select_gpu; then
      return 0
    fi
    if [ "$attempt" -ge "$max_attempts" ]; then
      echo "GPU selection failed after ${attempt} attempts" >&2
      return 1
    fi
    echo "Retry GPU selection ${attempt}/${max_attempts}" >&2
    sleep "$((attempt * 3))"
    attempt=$((attempt + 1))
  done
}

mapfile -t SYNC_DIRS < <("$PYTHON_BIN" - "$MANIFEST" <<'PY'
import csv
import sys
from pathlib import Path

rows = list(csv.DictReader(open(sys.argv[1], newline="", encoding="utf-8"), delimiter="\t"))
dirs = set()
for row in rows:
    dirs.add(str(Path(row["target_scene"]).parent))
    dirs.add(str(Path(row["contact_mask"]).parent))
for path in [
    "example_datasets/processed/core4d/assets/objects/box004",
    "example_datasets/processed/core4d/assets/objects/box021",
    "example_datasets/processed/core4d/assets/objects/bucket004",
    "spider/assets/robots/unitree_g1",
]:
    dirs.add(path)
for path in sorted(dirs):
    print(path)
PY
)

mapfile -t SYNC_FILES < <("$PYTHON_BIN" - "$MANIFEST" <<'PY'
import csv
import sys

rows = list(csv.DictReader(open(sys.argv[1], newline="", encoding="utf-8"), delimiter="\t"))
files = {
    sys.argv[1],
    "workspace/core4d/results/E168/s6_downstream/cem/manifests/cem_execution_manifest_summary.json",
    "workspace/core4d/scripts/experiments/E168/run_cem_queue.py",
    "examples/run_mjwp.py",
    "spider/config.py",
    "spider/simulators/mjwp.py",
    "spider/optimizers/sampling.py",
    "spider/optimizers/sampling_fast.py",
}
for row in rows:
    files.add(row["override_path"])
for pattern in [
    "examples/config/override/core4d_E15*.yaml",
    "examples/config/override/core4d_E16*.yaml",
    "examples/config/override/core4d_E167_*.yaml",
]:
    # Expanded by bash after this Python block.
    files.add(pattern)
for path in sorted(files):
    print(path)
PY
)

for dir in "${SYNC_DIRS[@]}"; do
  [ -d "$dir" ] || { echo "Missing sync dir: $dir" >&2; exit 1; }
done

GPU_ID="$(retry_select_gpu)"
mkdir -p "$(dirname "$SELECTION_TSV")"
{
  printf 'created_at\tremote\tremote_root\tselected_gpu\tmemory_limit_mb\tsession\n'
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "$REMOTE" "$REMOTE_ROOT" "$GPU_ID" "$GPU_MEM_LIMIT_MB" "$SESSION"
} > "$SELECTION_TSV"

echo "Selected ${REMOTE} GPU ${GPU_ID} for E168 canary"
retry ssh "$REMOTE" "mkdir -p \
  '$REMOTE_ROOT/workspace/core4d/scripts/experiments/E168' \
  '$REMOTE_ROOT/workspace/core4d/results/E168/s6_downstream/cem/manifests' \
  '$REMOTE_ROOT/workspace/core4d/results/E168/s6_downstream/cem/canary' \
  '$REMOTE_ROOT/workspace/core4d/results/E168/s0_environment' \
  '$REMOTE_ROOT/logs/E168/cem/canary' \
  '$REMOTE_ROOT/examples/config/override' \
  '$REMOTE_ROOT/spider/simulators' \
  '$REMOTE_ROOT/spider/optimizers'"

for dir in "${SYNC_DIRS[@]}"; do
  retry ssh "$REMOTE" "mkdir -p '$REMOTE_ROOT/$dir'"
  retry rsync -az "$dir/" "${REMOTE}:${REMOTE_ROOT}/${dir}/"
done

# Intentionally allow shell glob expansion for base override chains.
for file in "${SYNC_FILES[@]}"; do
  retry rsync -az $file "${REMOTE}:${REMOTE_ROOT}/$(dirname "$file")/"
done
retry rsync -az "$SELECTION_TSV" "${REMOTE}:${REMOTE_ROOT}/${SELECTION_TSV}"

retry ssh "$REMOTE" "cd '$REMOTE_ROOT' && \
  '$PYTHON_BIN' -m py_compile \
    workspace/core4d/scripts/experiments/E168/run_cem_queue.py \
    examples/run_mjwp.py spider/config.py spider/simulators/mjwp.py \
    spider/optimizers/sampling.py spider/optimizers/sampling_fast.py && \
  '$PYTHON_BIN' workspace/core4d/scripts/experiments/E168/run_cem_queue.py --mode canary --dry-run --python-bin '$PYTHON_BIN' --gpu-id '$GPU_ID' >/tmp/E168_canary_dryrun.txt && \
  test -s /tmp/E168_canary_dryrun.txt && \
  echo preflight-ok"

retry ssh "$REMOTE" "cd '$REMOTE_ROOT' && \
  tmux new-session -d -s '$SESSION' \"bash -lc 'set -euo pipefail; \
    echo E168 A6000 canary session $SESSION gpu=$GPU_ID; \
    '$PYTHON_BIN' workspace/core4d/scripts/experiments/E168/run_cem_queue.py \
      --mode canary --all --python-bin '$PYTHON_BIN' --gpu-id '$GPU_ID'; \
    echo E168 A6000 canary complete'\""

echo "Started remote tmux session: ${SESSION}"
echo "Monitor: ssh ${REMOTE} \"tmux capture-pane -t ${SESSION} -p | tail -80\""
echo "Pull: REMOTE=${REMOTE} REMOTE_ROOT=${REMOTE_ROOT} bash workspace/core4d/scripts/launch/active/pull_E168_remote_a6000_results.sh canary"
