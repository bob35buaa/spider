#!/usr/bin/env bash
# E179 local RTX 5090 GPU0 launcher: 3-row canary or fixed 4-row Full shard.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODE="${1:-canary}"
if [ "$MODE" != "canary" ] && [ "$MODE" != "full" ]; then
  echo "usage: $0 {canary|full}" >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
LOCAL_GPU="${LOCAL_GPU:-0}"
GPU_MEM_LIMIT_MB="${E179_LOCAL_GPU_MEM_LIMIT_MB:-5000}"
GPU_CHECK_INTERVAL_S="${E179_GPU_CHECK_INTERVAL_S:-5}"
PREP_ONLY="${E179_PREP_ONLY:-0}"
DETACH="${E179_DETACH:-1}"
SESSION="${SESSION:-e179_local_${MODE}_$(date +%Y%m%d_%H%M%S)}"
RESULT_ROOT="workspace/core4d/results/E179"
MANIFEST_ROOT="$RESULT_ROOT/s6_downstream/manifests"
RUNNER="workspace/core4d/scripts/experiments/E179/run_cem_queue.py"
AUDITOR="workspace/core4d/scripts/experiments/E179/audit_e167a_no_prg.py"
if [ "$MODE" = "canary" ]; then
  MANIFEST="$MANIFEST_ROOT/cem_canary_local_gpu0.tsv"
  EXPECTED_ROWS=3
else
  MANIFEST="$MANIFEST_ROOT/cem_full_local_gpu0.tsv"
  EXPECTED_ROWS=4
fi

[ "$LOCAL_GPU" = "0" ] || {
  echo "E179 local allocation is frozen to GPU0; got LOCAL_GPU=$LOCAL_GPU" >&2
  exit 3
}
[ -x "$PYTHON_BIN" ] || {
  echo "missing project Python: $PYTHON_BIN" >&2
  exit 3
}
[ "$PREP_ONLY" = "0" ] || [ "$PREP_ONLY" = "1" ] || {
  echo "E179_PREP_ONLY must be 0 or 1" >&2
  exit 3
}
[ "$DETACH" = "0" ] || [ "$DETACH" = "1" ] || {
  echo "E179_DETACH must be 0 or 1" >&2
  exit 3
}

mapfile -t SNAPSHOT_TASKS < <(
  "$PYTHON_BIN" - <<'PY'
import csv
from pathlib import Path

manifest = Path(
    "workspace/core4d/results/E179/s6_downstream/manifests/"
    "cem_full_manifest.tsv"
)
rows = list(csv.DictReader(manifest.open(encoding="utf-8"), delimiter="\t"))
if len(rows) != 16 or len({row["target_task"] for row in rows}) != 16:
    raise SystemExit("E179 snapshot authority must contain 16 unique tasks")
for row in rows:
    print(row["target_task"])
PY
)

bash workspace/core4d/scripts/convert/snapshot_scenes.sh \
  E179 "${SNAPSHOT_TASKS[@]}" >/dev/null
"$PYTHON_BIN" "$AUDITOR" --require-all >/dev/null
"$PYTHON_BIN" "$RUNNER" \
  --mode "$MODE" \
  --manifest-tsv "$MANIFEST" \
  --python-bin "$PYTHON_BIN" \
  --gpu-id "$LOCAL_GPU" \
  --all \
  --dry-run >/dev/null

if [ "$MODE" = "full" ]; then
  "$PYTHON_BIN" - <<'PY'
import csv
import importlib.util
from pathlib import Path

runner_path = Path(
    "workspace/core4d/scripts/experiments/E179/run_cem_queue.py"
)
spec = importlib.util.spec_from_file_location("e179_queue_gate", runner_path)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
manifest = Path(
    "workspace/core4d/results/E179/s6_downstream/manifests/"
    "cem_canary_local_gpu0.tsv"
)
rows = list(csv.DictReader(manifest.open(encoding="utf-8"), delimiter="\t"))
failures = {
    row["case_id"]: module.validate_runtime_outputs(row)
    for row in rows
}
bad = {
    case_id: issues
    for case_id, issues in failures.items()
    if issues
}
if len(rows) != 3 or bad:
    raise SystemExit(f"E179 Full blocked by canary runtime gate: {bad}")
PY
fi

ENV_ROOT="$RESULT_ROOT/s0_environment/local_${MODE}_${SESSION}"
mkdir -p "$ENV_ROOT"

capture_and_validate_gpu() {
  local label="$1"
  local gpu_file="$ENV_ROOT/${label}_gpu.csv"
  local compute_file="$ENV_ROOT/${label}_compute.csv"
  nvidia-smi \
    --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader,nounits > "$gpu_file"
  nvidia-smi \
    --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory \
    --format=csv,noheader,nounits > "$compute_file" || true
  "$PYTHON_BIN" - "$gpu_file" "$compute_file" "$GPU_MEM_LIMIT_MB" <<'PY'
import csv
import sys
from pathlib import Path

gpu_rows = list(csv.reader(Path(sys.argv[1]).open(encoding="utf-8")))
compute_rows = list(csv.reader(Path(sys.argv[2]).open(encoding="utf-8")))
matches = [row for row in gpu_rows if row and row[0].strip() == "0"]
if len(matches) != 1:
    raise SystemExit(f"expected exactly one local GPU0 row, got {len(matches)}")
row = [value.strip() for value in matches[0]]
uuid = row[1]
memory_used = int(row[3])
limit = int(sys.argv[3])
busy = [
    [value.strip() for value in item]
    for item in compute_rows
    if item and item[0].strip() == uuid
]
if memory_used >= limit:
    raise SystemExit(
        f"local GPU0 memory used {memory_used}MiB >= {limit}MiB"
    )
if busy:
    raise SystemExit(f"local GPU0 has conflicting compute processes: {busy}")
print(f"local GPU0 pass: memory_used={memory_used}MiB compute=0")
PY
}

capture_and_validate_gpu check1
sleep "$GPU_CHECK_INTERVAL_S"
capture_and_validate_gpu check2

RUN_SCRIPT="$ENV_ROOT/run_${SESSION}.sh"
{
  printf '#!/usr/bin/env bash\n'
  printf 'set -euo pipefail\n'
  printf 'cd %q\n' "$(pwd)"
  printf 'export MUJOCO_GL=%q\n' "${MUJOCO_GL:-egl}"
  printf 'export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 '
  printf 'OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1\n'
  printf '%q %q --mode %q --manifest-tsv %q ' \
    "$PYTHON_BIN" "$RUNNER" "$MODE" "$MANIFEST"
  printf -- '--python-bin %q --gpu-id %q --all\n' \
    "$PYTHON_BIN" "$LOCAL_GPU"
} > "$RUN_SCRIPT"
chmod +x "$RUN_SCRIPT"

MODE="$MODE" SESSION="$SESSION" MANIFEST="$MANIFEST" \
ENV_ROOT="$ENV_ROOT" PREP_ONLY="$PREP_ONLY" DETACH="$DETACH" \
EXPECTED_ROWS="$EXPECTED_ROWS" GPU_MEM_LIMIT_MB="$GPU_MEM_LIMIT_MB" \
"$PYTHON_BIN" - <<'PY'
import csv
import hashlib
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

manifest = Path(os.environ["MANIFEST"])
rows = list(csv.DictReader(manifest.open(encoding="utf-8"), delimiter="\t"))
expected = int(os.environ["EXPECTED_ROWS"])
if len(rows) != expected:
    raise SystemExit(f"local manifest rows={len(rows)} expected={expected}")
payload = {
    "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    "experiment_id": "E179",
    "profile": "local-RTX5090",
    "mode": os.environ["MODE"],
    "session": os.environ["SESSION"],
    "gpu_id": "0",
    "policy_source": "user-fixed-local-gpu0",
    "memory_limit_mb": int(os.environ["GPU_MEM_LIMIT_MB"]),
    "manifest": str(manifest),
    "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    "rows": len(rows),
    "git_head": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip(),
    "check1_gpu": Path(
        os.environ["ENV_ROOT"], "check1_gpu.csv"
    ).read_text(encoding="utf-8"),
    "check1_compute": Path(
        os.environ["ENV_ROOT"], "check1_compute.csv"
    ).read_text(encoding="utf-8"),
    "check2_gpu": Path(
        os.environ["ENV_ROOT"], "check2_gpu.csv"
    ).read_text(encoding="utf-8"),
    "check2_compute": Path(
        os.environ["ENV_ROOT"], "check2_compute.csv"
    ).read_text(encoding="utf-8"),
    "status": (
        "preflight_passed_no_launch"
        if os.environ["PREP_ONLY"] == "1"
        else (
            "ready_for_detached_launch"
            if os.environ["DETACH"] == "1"
            else "ready_for_foreground_launch"
        )
    ),
}
path = Path(os.environ["ENV_ROOT"], "execution_manifest.json")
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
latest = Path(
    f"workspace/core4d/results/E179/s0_environment/"
    f"local_{os.environ['MODE']}_latest.json"
)
latest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

if [ "$PREP_ONLY" = "1" ]; then
  echo "E179 local $MODE preflight passed; no CEM launched."
  echo "Execution manifest: $ENV_ROOT/execution_manifest.json"
  exit 0
fi

if [ "$DETACH" = "1" ]; then
  tmux new-session -d -s "$SESSION" -c "$(pwd)" "bash $RUN_SCRIPT"
  echo "Started E179 local $MODE in tmux session=$SESSION GPU0"
else
  bash "$RUN_SCRIPT"
fi
