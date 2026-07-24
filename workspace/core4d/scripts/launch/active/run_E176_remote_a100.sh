#!/usr/bin/env bash
# E176 A100 launcher: dynamic policy allowlist, exact-file sync, serial queues.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODE="${1:-canary}"
if [ "$MODE" != "canary" ] && [ "$MODE" != "full" ]; then
  echo "usage: $0 {canary|full}" >&2
  exit 2
fi

EXPERIMENT_ID="${EXPERIMENT_ID:-E176}"
EXPERIMENT_SLUG="$(
  printf '%s' "$EXPERIMENT_ID" | tr '[:upper:]' '[:lower:]'
)"
REMOTE="${REMOTE:-batchcom@61.172.170.106}"
REMOTE_CANONICAL_ROOT="${REMOTE_CANONICAL_ROOT:-/home/dataset-assist-0/xiayb/workspace/spider}"
REMOTE_SSH_PORT="${REMOTE_SSH_PORT:-30409}"
REMOTE_SSH_KEY="${REMOTE_SSH_KEY:-/home/ubuntu/.ssh/id_rsa_tianyiyun}"
LOCAL_PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
REMOTE_PYTHON_BIN="${REMOTE_PYTHON_BIN:-${REMOTE_CANONICAL_ROOT}/.venv/bin/python}"
A100_GPU_MEM_USED_LIMIT_MB="${A100_GPU_MEM_USED_LIMIT_MB:-5000}"
A100_MAX_GPUS="${A100_MAX_GPUS:-4}"
A100_AVAILABILITY_CMD="${A100_AVAILABILITY_CMD:-}"
A100_POLICY_GPUS="${A100_POLICY_GPUS:-}"
E176_FIXED_GPUS="${E176_FIXED_GPUS:-0}"
E176_PREP_ONLY="${E176_PREP_ONLY:-0}"
E176_SNAPSHOT_ONLY="${E176_SNAPSHOT_ONLY:-0}"
E176_ALLOW_THROUGHPUT_GATE_WAIVER="${E176_ALLOW_THROUGHPUT_GATE_WAIVER:-0}"
E176_THROUGHPUT_GATE_WAIVER_REASON="${E176_THROUGHPUT_GATE_WAIVER_REASON:-}"
SESSION="${SESSION:-${EXPERIMENT_SLUG}_a100_${MODE}_$(date +%Y%m%d_%H%M%S)}"
REMOTE_RUN_PARENT="${REMOTE_RUN_PARENT:-/home/dataset-assist-0/xiayb/workspace/${EXPERIMENT_SLUG}_spider_runs}"
REMOTE_RUN_ROOT="${REMOTE_RUN_ROOT:-${REMOTE_RUN_PARENT}/${SESSION}}"
EXPECTED_CANARY_ROWS="${EXPECTED_CANARY_ROWS:-6}"
EXPECTED_FULL_ROWS="${EXPECTED_FULL_ROWS:-39}"
CANARY_MANIFEST_BASENAME="${CANARY_MANIFEST_BASENAME:-lowgeom_canary_manifest.tsv}"
FULL_MANIFEST_BASENAME="${FULL_MANIFEST_BASENAME:-lowgeom_full_manifest.tsv}"

if [ "$E176_PREP_ONLY" != "0" ] && [ "$E176_PREP_ONLY" != "1" ]; then
  echo "E176_PREP_ONLY must be 0 or 1." >&2
  exit 2
fi
if [ "$E176_FIXED_GPUS" != "0" ] && [ "$E176_FIXED_GPUS" != "1" ]; then
  echo "E176_FIXED_GPUS must be 0 or 1." >&2
  exit 2
fi
if [ "$E176_SNAPSHOT_ONLY" != "0" ] && [ "$E176_SNAPSHOT_ONLY" != "1" ]; then
  echo "E176_SNAPSHOT_ONLY must be 0 or 1." >&2
  exit 2
fi
if [ "$E176_ALLOW_THROUGHPUT_GATE_WAIVER" != "0" ] && \
   [ "$E176_ALLOW_THROUGHPUT_GATE_WAIVER" != "1" ]; then
  echo "E176_ALLOW_THROUGHPUT_GATE_WAIVER must be 0 or 1." >&2
  exit 2
fi
if [ "$E176_ALLOW_THROUGHPUT_GATE_WAIVER" = "1" ] && \
   [ -z "$E176_THROUGHPUT_GATE_WAIVER_REASON" ]; then
  echo "A throughput gate waiver requires E176_THROUGHPUT_GATE_WAIVER_REASON." >&2
  exit 2
fi
if [ "$E176_SNAPSHOT_ONLY" = "1" ] && [ "$E176_PREP_ONLY" = "1" ]; then
  echo "E176_SNAPSHOT_ONLY and E176_PREP_ONLY are mutually exclusive." >&2
  exit 2
fi
if [ "$E176_SNAPSHOT_ONLY" = "0" ] && [ -z "$A100_AVAILABILITY_CMD" ] && [ -z "$A100_POLICY_GPUS" ]; then
  echo "E176 A100 launch blocked: provide A100_AVAILABILITY_CMD or explicit A100_POLICY_GPUS." >&2
  echo "Low memory alone is not reservation/owner authorization." >&2
  exit 3
fi
if [ "$E176_SNAPSHOT_ONLY" = "0" ] && [ -n "$A100_AVAILABILITY_CMD" ] && [ -n "$A100_POLICY_GPUS" ]; then
  echo "Provide only one policy source: A100_AVAILABILITY_CMD or A100_POLICY_GPUS." >&2
  exit 3
fi

RESULT_ROOT="${RESULT_ROOT:-workspace/core4d/results/${EXPERIMENT_ID}}"
MANIFEST_ROOT="${RESULT_ROOT}/s6_downstream/manifests"
if [ "$MODE" = "canary" ]; then
  MANIFEST="${MANIFEST_ROOT}/${CANARY_MANIFEST_BASENAME}"
  CEM_SUBDIR="canary"
  NUM_SAMPLES=64
  NUM_ITERATIONS=4
  EXPECTED_ROWS="$EXPECTED_CANARY_ROWS"
else
  MANIFEST="${MANIFEST_ROOT}/${FULL_MANIFEST_BASENAME}"
  CEM_SUBDIR="full"
  NUM_SAMPLES=1024
  NUM_ITERATIONS=32
  EXPECTED_ROWS="$EXPECTED_FULL_ROWS"
  if [ "$E176_SNAPSHOT_ONLY" = "0" ]; then
    CANARY_GATE="${RESULT_ROOT}/s6_downstream/cem/canary/canary_runtime_gate.json"
    CANARY_THROUGHPUT_GATE="${RESULT_ROOT}/s6_downstream/cem/canary/canary_throughput_gate.json"
    [ -f "$CANARY_GATE" ] && [ -f "$CANARY_THROUGHPUT_GATE" ] || {
      echo "Full blocked: missing canary runtime/throughput gate." >&2
      exit 4
    }
    "$LOCAL_PYTHON_BIN" - "$CANARY_GATE" "$CANARY_THROUGHPUT_GATE" \
      "$EXPECTED_CANARY_ROWS" "$E176_ALLOW_THROUGHPUT_GATE_WAIVER" \
      "$E176_THROUGHPUT_GATE_WAIVER_REASON" <<'PY'
import json, sys
runtime = json.load(open(sys.argv[1], encoding="utf-8"))
throughput = json.load(open(sys.argv[2], encoding="utf-8"))
expected = int(sys.argv[3])
allow_throughput_waiver = sys.argv[4] == "1"
waiver_reason = sys.argv[5]
if runtime.get("status") != "pass" or runtime.get("passed_rows") != expected:
    raise SystemExit(f"Full blocked by canary runtime gate: {runtime}")
if throughput.get("status") != "pass" or throughput.get("passed_rows") != expected:
    if not allow_throughput_waiver:
        raise SystemExit(f"Full blocked by canary throughput gate: {throughput}")
    print(
        "Full throughput gate explicitly waived: "
        f"passed_rows={throughput.get('passed_rows')}/{expected}; "
        f"reason={waiver_reason}"
    )
PY
  fi
fi

RUNNER="${RUNNER:-workspace/core4d/scripts/experiments/E176/run_cem_queue.py}"
VERIFY_SYNC_RUNNER="${VERIFY_SYNC_RUNNER:-workspace/core4d/scripts/experiments/E176/verify_sync_inventory.py}"
EXECUTION_ROOT="${RESULT_ROOT}/s0_environment/a100_${MODE}_${SESSION}"
SHARD_ROOT="${EXECUTION_ROOT}/queues"
SESSION_SCRIPT="${EXECUTION_ROOT}/run_${SESSION}.sh"
EXECUTION_JSON="${EXECUTION_ROOT}/execution_manifest.json"
LATEST_JSON="${RESULT_ROOT}/s0_environment/a100_${MODE}_latest.json"
SYNC_LIST="${EXECUTION_ROOT}/sync_files.txt"
SYNC_INVENTORY="${EXECUTION_ROOT}/sync_inventory.json"
mkdir -p "$SHARD_ROOT"

SSH_OPTS=(
  -o BatchMode=yes
  -o ConnectTimeout=20
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=4
  -p "$REMOTE_SSH_PORT"
  -i "$REMOTE_SSH_KEY"
)
RSYNC_RSH="ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=4 -p ${REMOTE_SSH_PORT} -i ${REMOTE_SSH_KEY}"

ssh_remote() {
  ssh "${SSH_OPTS[@]}" "$REMOTE" "$@"
}

retry() {
  local attempt=1
  local max_attempts="${E176_REMOTE_RETRY_MAX:-4}"
  while true; do
    if "$@"; then
      return 0
    fi
    if [ "$attempt" -ge "$max_attempts" ]; then
      echo "Command failed after ${attempt} attempts: $*" >&2
      return 1
    fi
    sleep "$((attempt * 3))"
    attempt=$((attempt + 1))
  done
}

GPU_SNAPSHOT=""
COMPUTE_SNAPSHOT=""
POLICY_SNAPSHOT=""
SELECTED_GPUS=""
REMOTE_BASE_HEAD=""

select_gpus() {
  GPU_SNAPSHOT="$(ssh_remote "nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits")"
  COMPUTE_SNAPSHOT="$(ssh_remote "nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader,nounits" 2>/dev/null || true)"
  if [ -n "$A100_AVAILABILITY_CMD" ]; then
    POLICY_SNAPSHOT="$(ssh_remote "$A100_AVAILABILITY_CMD")"
  else
    POLICY_SNAPSHOT="$A100_POLICY_GPUS"
  fi
  if [ "$E176_FIXED_GPUS" = "1" ]; then
    SELECTED_GPUS="$(
      POLICY_SNAPSHOT="$POLICY_SNAPSHOT" \
      MAX_GPUS="$A100_MAX_GPUS" \
      "$LOCAL_PYTHON_BIN" - <<'PY'
import os
import re

values = [
    int(value)
    for value in re.findall(
        r"(?<!\d)[0-7](?!\d)", os.environ["POLICY_SNAPSHOT"]
    )
]
values = sorted(set(values))
if not values:
    raise SystemExit("fixed GPU mode received an empty policy set")
if len(values) > int(os.environ["MAX_GPUS"]):
    raise SystemExit(
        f"fixed GPU set has {len(values)} cards, max={os.environ['MAX_GPUS']}"
    )
print(" ".join(map(str, values)))
PY
    )"
    return
  fi
  SELECTED_GPUS="$(
    GPU_SNAPSHOT="$GPU_SNAPSHOT" \
    COMPUTE_SNAPSHOT="$COMPUTE_SNAPSHOT" \
    POLICY_SNAPSHOT="$POLICY_SNAPSHOT" \
    LIMIT_MB="$A100_GPU_MEM_USED_LIMIT_MB" \
    MAX_GPUS="$A100_MAX_GPUS" \
    "$LOCAL_PYTHON_BIN" - <<'PY'
import os, re

gpu_rows = {}
for line in os.environ["GPU_SNAPSHOT"].splitlines():
    fields = [value.strip() for value in line.split(",")]
    if len(fields) < 5:
        continue
    gpu_rows[int(fields[0])] = {
        "uuid": fields[1],
        "memory_used": int(fields[2]),
    }
busy_uuids = set()
for line in os.environ["COMPUTE_SNAPSHOT"].splitlines():
    fields = [value.strip() for value in line.split(",")]
    if fields and fields[0]:
        busy_uuids.add(fields[0])
policy = {
    int(value)
    for value in re.findall(r"(?<!\d)[0-7](?!\d)", os.environ["POLICY_SNAPSHOT"])
}
limit = int(os.environ["LIMIT_MB"])
allowed = [
    index
    for index, row in sorted(gpu_rows.items())
    if index in policy
    and row["memory_used"] < limit
    and row["uuid"] not in busy_uuids
]
allowed = allowed[: int(os.environ["MAX_GPUS"])]
print(" ".join(map(str, allowed)))
PY
  )"
  if [ -z "$SELECTED_GPUS" ]; then
    echo "No A100 GPU satisfies policy ∩ memory<${A100_GPU_MEM_USED_LIMIT_MB}MB ∩ no-compute." >&2
    return 1
  fi
}

write_sync_list() {
  MODE="$MODE" MANIFEST="$MANIFEST" SYNC_LIST="$SYNC_LIST" \
  SYNC_INVENTORY="$SYNC_INVENTORY" \
  EXPERIMENT_ID="$EXPERIMENT_ID" RUNNER="$RUNNER" \
  VERIFY_SYNC_RUNNER="$VERIFY_SYNC_RUNNER" \
    "$LOCAL_PYTHON_BIN" - <<'PY'
import csv
import hashlib
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

import yaml

repo = Path.cwd()
manifest = Path(os.environ["MANIFEST"])
rows = list(csv.DictReader(manifest.open(encoding="utf-8"), delimiter="\t"))
files = {
    manifest,
    Path("examples/run_mjwp.py"),
    Path("spider/config.py"),
    Path("spider/simulators/mjwp.py"),
    Path("workspace/core4d/scripts/experiments/E169/run_cem_queue.py"),
    Path(os.environ["RUNNER"]),
    Path(os.environ["VERIFY_SYNC_RUNNER"]),
}
tracked_runtime = subprocess.check_output(
    ["git", "ls-files", "-z", "spider", "examples/config"],
).decode().split("\0")
files.update(Path(value) for value in tracked_runtime if value)
for row in rows:
    for key in ("target_scene", "trajectory", "scene_act", "contact_mask", "override_path"):
        files.add(Path(row[key]))
    files.add(Path(row["scene_act"]).parent / "scene_act_meta.json")

override_root = Path("examples/config/override")
pending = [Path(row["override_path"]).stem for row in rows]
seen = set()
while pending:
    override_id = pending.pop()
    if override_id in seen:
        continue
    seen.add(override_id)
    path = override_root / f"{override_id}.yaml"
    if not path.is_file():
        raise FileNotFoundError(path)
    files.add(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for item in payload.get("defaults", []):
        if isinstance(item, str) and item != "_self_":
            pending.append(item)
        elif isinstance(item, dict):
            for value in item.values():
                if isinstance(value, str) and value != "_self_":
                    pending.append(value)

for object_key in ("bucket003", "bucket004", "bucket007", "bucket009", "bucket010", "desk007"):
    files.update(
        path for path in
        Path(f"example_datasets/processed/core4d/assets/objects/{object_key}").rglob("*")
        if path.is_file()
    )
files.update(
    path for path in Path("spider/assets/robots/unitree_g1").rglob("*")
    if path.is_file()
)
missing = [str(path) for path in files if not path.is_file()]
if missing:
    raise FileNotFoundError("missing sync files: " + ", ".join(sorted(missing)[:20]))
output = Path(os.environ["SYNC_LIST"])
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text("\n".join(sorted(str(path) for path in files)) + "\n", encoding="utf-8")
inventory = []
for path in sorted(files, key=lambda value: str(value)):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    inventory.append(
        {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": digest.hexdigest(),
        }
    )
Path(os.environ["SYNC_INVENTORY"]).write_text(
    json.dumps(
        {
            "created_at": datetime.now().astimezone().isoformat(
                timespec="seconds"
            ),
            "mode": os.environ["MODE"],
            "files": len(inventory),
            "bytes": sum(item["bytes"] for item in inventory),
            "artifacts": inventory,
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
print(f"{os.environ['EXPERIMENT_ID']} sync files: {len(files)}")
PY
}

build_pool() {
  rm -f "${SHARD_ROOT}"/gpu*.tsv "${SHARD_ROOT}/queue_order.tsv"
  MODE="$MODE" MANIFEST="$MANIFEST" SHARD_ROOT="$SHARD_ROOT" \
  EXPECTED_ROWS="$EXPECTED_ROWS" \
  SELECTED_GPUS="$SELECTED_GPUS" "$LOCAL_PYTHON_BIN" - <<'PY'
import csv
import os
from datetime import datetime
from pathlib import Path

import numpy as np

manifest = Path(os.environ["MANIFEST"])
root = Path(os.environ["SHARD_ROOT"])
gpus = os.environ["SELECTED_GPUS"].split()
rows = list(csv.DictReader(manifest.open(encoding="utf-8"), delimiter="\t"))
fields = list(rows[0]) if rows else []
expected = int(os.environ["EXPECTED_ROWS"])
if len(rows) != expected:
    raise ValueError(f"manifest rows={len(rows)} expected={expected}")
for row in rows:
    with np.load(row["trajectory"], allow_pickle=True) as payload:
        row["_frames"] = int(payload["qpos"].shape[0])
    row["status"] = "not_run"
    row["failure_mode"] = ""
    row["gpu_id"] = ""
    row["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")

queues = {gpu: [] for gpu in gpus}
loads = {gpu: 0 for gpu in gpus}
for row in sorted(rows, key=lambda item: (-item["_frames"], item["case_id"])):
    gpu = min(gpus, key=lambda value: (loads[value], int(value)))
    queues[gpu].append(row)
    loads[gpu] += row["_frames"]

root.mkdir(parents=True, exist_ok=True)
queue_rows = []
for gpu in gpus:
    path = root / f"gpu{gpu}.tsv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in queues[gpu]:
            writer.writerow({key: row.get(key, "") for key in fields})
            queue_rows.append({
                "gpu_id": gpu,
                "queue_index": len(queue_rows) + 1,
                "case_id": row["case_id"],
                "variant": row["variant"],
                "trajectory_frames": row["_frames"],
            })
with (root / "queue_order.tsv").open("w", newline="", encoding="utf-8") as stream:
    fields_out = ["queue_index", "gpu_id", "case_id", "variant", "trajectory_frames"]
    writer = csv.DictWriter(stream, fieldnames=fields_out, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(queue_rows)
print({"gpus": gpus, "loads": loads, "rows": len(rows)})
PY

  cat > "$SESSION_SCRIPT" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$REMOTE_RUN_ROOT"
export MUJOCO_GL=egl
export PYTHONPATH="$REMOTE_RUN_ROOT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
MODE="$MODE"
RUNNER="$RUNNER"
SHARD_ROOT="$SHARD_ROOT"
PYTHON_BIN="$REMOTE_PYTHON_BIN"
GPU_LIST=($SELECTED_GPUS)
NUM_SAMPLES="$NUM_SAMPLES"
NUM_ITERATIONS="$NUM_ITERATIONS"
LOG_DIR="logs/${EXPERIMENT_ID}/cem/$CEM_SUBDIR"
mkdir -p "\$LOG_DIR"
worker_pids=()
for gpu in "\${GPU_LIST[@]}"; do
  shard="\${SHARD_ROOT}/gpu\${gpu}.tsv"
  (
    "\$PYTHON_BIN" "\$RUNNER" \
      --mode "\$MODE" \
      --manifest-tsv "\$shard" \
      --python-bin "\$PYTHON_BIN" \
      --gpu-id "\$gpu" \
      --all \
      --num-samples "\$NUM_SAMPLES" \
      --max-num-iterations "\$NUM_ITERATIONS"
  ) > "\${LOG_DIR}/${SESSION}_gpu\${gpu}.worker.log" 2>&1 &
  worker_pids+=("\$!")
done
status=0
for pid in "\${worker_pids[@]}"; do
  wait "\$pid" || status=1
done
echo "${EXPERIMENT_ID} \$MODE completed status=\$status at \$(date -Is)"
exit "\$status"
EOF
  chmod +x "$SESSION_SCRIPT"
}

write_execution_manifest() {
  local execution_status="ready_to_launch"
  if [ "$E176_PREP_ONLY" = "1" ]; then
    execution_status="preflight_passed_no_launch"
  fi
  GPU_SNAPSHOT="$GPU_SNAPSHOT" COMPUTE_SNAPSHOT="$COMPUTE_SNAPSHOT" \
  POLICY_SNAPSHOT="$POLICY_SNAPSHOT" SELECTED_GPUS="$SELECTED_GPUS" \
  POLICY_SOURCE="$([ -n "$A100_AVAILABILITY_CMD" ] && echo availability_cmd || echo explicit_env)" \
  FIXED_GPUS="$E176_FIXED_GPUS" EXPERIMENT_ID="$EXPERIMENT_ID" \
  THROUGHPUT_GATE_WAIVER="$E176_ALLOW_THROUGHPUT_GATE_WAIVER" \
  THROUGHPUT_GATE_WAIVER_REASON="$E176_THROUGHPUT_GATE_WAIVER_REASON" \
  MODE="$MODE" SESSION="$SESSION" REMOTE="$REMOTE" \
  REMOTE_CANONICAL_ROOT="$REMOTE_CANONICAL_ROOT" \
  REMOTE_RUN_ROOT="$REMOTE_RUN_ROOT" REMOTE_PYTHON_BIN="$REMOTE_PYTHON_BIN" \
  REMOTE_BASE_HEAD="$REMOTE_BASE_HEAD" SYNC_INVENTORY="$SYNC_INVENTORY" \
  SHARD_ROOT="$SHARD_ROOT" MANIFEST="$MANIFEST" EXECUTION_JSON="$EXECUTION_JSON" \
  EXECUTION_STATUS="$execution_status" \
  "$LOCAL_PYTHON_BIN" - <<'PY'
import json, os
import hashlib
import subprocess
from datetime import datetime
from pathlib import Path

inventory_path = Path(os.environ["SYNC_INVENTORY"])
inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
payload = {
    "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    "profile": "A100-8gpu",
    "experiment_id": os.environ["EXPERIMENT_ID"],
    "mode": os.environ["MODE"],
    "session": os.environ["SESSION"],
    "remote": os.environ["REMOTE"],
    "remote_canonical_root": os.environ["REMOTE_CANONICAL_ROOT"],
    "remote_root": os.environ["REMOTE_RUN_ROOT"],
    "remote_python": os.environ["REMOTE_PYTHON_BIN"],
    "remote_pythonpath": os.environ["REMOTE_RUN_ROOT"],
    "remote_canonical_head": os.environ["REMOTE_BASE_HEAD"],
    "local_git_head": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip(),
    "sync_inventory": os.environ["SYNC_INVENTORY"],
    "sync_inventory_sha256": hashlib.sha256(
        inventory_path.read_bytes()
    ).hexdigest(),
    "sync_file_count": inventory["files"],
    "sync_total_bytes": inventory["bytes"],
    "source_manifest": os.environ["MANIFEST"],
    "shard_root": os.environ["SHARD_ROOT"],
    "policy_source": os.environ["POLICY_SOURCE"],
    "fixed_gpus": os.environ["FIXED_GPUS"] == "1",
    "throughput_gate_waiver": os.environ["THROUGHPUT_GATE_WAIVER"] == "1",
    "throughput_gate_waiver_reason": os.environ[
        "THROUGHPUT_GATE_WAIVER_REASON"
    ],
    "policy_snapshot": os.environ["POLICY_SNAPSHOT"],
    "gpu_snapshot": os.environ["GPU_SNAPSHOT"],
    "compute_snapshot": os.environ["COMPUTE_SNAPSHOT"],
    "selected_gpus": os.environ["SELECTED_GPUS"].split(),
    "status": os.environ["EXECUTION_STATUS"],
}
path = Path(os.environ["EXECUTION_JSON"])
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
  cp "$EXECUTION_JSON" "$LATEST_JSON"
}

sync_base_inputs() {
  write_sync_list
  REMOTE_BASE_HEAD="$(ssh_remote "git -C '$REMOTE_CANONICAL_ROOT' rev-parse HEAD")"
  retry ssh_remote "mkdir -p '$REMOTE_RUN_ROOT'"
  retry rsync -azLR --files-from="$SYNC_LIST" -e "$RSYNC_RSH" ./ "${REMOTE}:${REMOTE_RUN_ROOT}/"
}

sync_pool() {
  write_execution_manifest
  retry ssh_remote "mkdir -p '$REMOTE_RUN_ROOT/$EXECUTION_ROOT' '$REMOTE_RUN_ROOT/$(dirname "$LATEST_JSON")'"
  retry rsync -az -e "$RSYNC_RSH" \
    "$EXECUTION_ROOT/" "${REMOTE}:${REMOTE_RUN_ROOT}/${EXECUTION_ROOT}/"
  retry rsync -az -e "$RSYNC_RSH" \
    "$LATEST_JSON" "${REMOTE}:${REMOTE_RUN_ROOT}/${LATEST_JSON}"
}

remote_preflight() {
  local first_gpu first_shard
  first_gpu="${SELECTED_GPUS%% *}"
  first_shard="${SHARD_ROOT}/gpu${first_gpu}.tsv"
  retry ssh_remote "cd '$REMOTE_RUN_ROOT' && \
    PYTHONPATH='$REMOTE_RUN_ROOT' \
    '$REMOTE_PYTHON_BIN' '$VERIFY_SYNC_RUNNER' \
      --inventory '$SYNC_INVENTORY' --root . \
      --output '${EXECUTION_ROOT}/remote_sync_verification.json' && \
    PYTHONPATH='$REMOTE_RUN_ROOT' \
    '$REMOTE_PYTHON_BIN' -m py_compile '$RUNNER' spider/config.py spider/simulators/mjwp.py && \
    PYTHONPATH='$REMOTE_RUN_ROOT' \
    '$REMOTE_PYTHON_BIN' '$RUNNER' --mode '$MODE' --manifest-tsv '$first_shard' \
      --python-bin '$REMOTE_PYTHON_BIN' --gpu-id '$first_gpu' --all --dry-run \
      --num-samples '$NUM_SAMPLES' --max-num-iterations '$NUM_ITERATIONS' \
      > '${EXECUTION_ROOT}/remote_preflight.log' && \
    test -s '${EXECUTION_ROOT}/remote_preflight.log'"
}

if [ "$E176_SNAPSHOT_ONLY" = "1" ]; then
  write_sync_list
  echo "$EXPERIMENT_ID local snapshot inventory complete; no SSH/rsync/tmux."
  echo "Sync list: $SYNC_LIST"
  echo "Sync inventory: $SYNC_INVENTORY"
  exit 0
fi

select_gpus
INITIAL_SELECTED_GPUS="$SELECTED_GPUS"
echo "Initial A100 pool: $INITIAL_SELECTED_GPUS"
sync_base_inputs

pool_attempt=1
while true; do
  select_gpus
  build_pool
  sync_pool
  remote_preflight
  PRELAUNCH_GPUS="$SELECTED_GPUS"
  select_gpus
  if [ "$SELECTED_GPUS" = "$PRELAUNCH_GPUS" ]; then
    break
  fi
  if [ "$pool_attempt" -ge 3 ]; then
    echo "A100 pool changed repeatedly before launch; aborting without tmux." >&2
    exit 5
  fi
  echo "A100 pool changed: '$PRELAUNCH_GPUS' -> '$SELECTED_GPUS'; rebuilding."
  pool_attempt=$((pool_attempt + 1))
done

write_execution_manifest
sync_pool
if [ "$E176_PREP_ONLY" = "1" ]; then
  echo "$EXPERIMENT_ID $MODE remote preflight passed; E176_PREP_ONLY=1, no tmux/CEM launched."
  echo "Execution manifest: $EXECUTION_JSON"
  exit 0
fi
retry ssh_remote "cd '$REMOTE_RUN_ROOT' && tmux new-session -d -s '$SESSION' -c '$REMOTE_RUN_ROOT' 'bash $SESSION_SCRIPT'"

echo "Started $EXPERIMENT_ID $MODE on A100 session=$SESSION GPUs=$SELECTED_GPUS"
echo "Execution manifest: $EXECUTION_JSON"
echo "Monitor: bash workspace/core4d/scripts/launch/active/watch_E176_remote_a100.sh $MODE"
