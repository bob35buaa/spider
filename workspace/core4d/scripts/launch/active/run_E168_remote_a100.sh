#!/usr/bin/env bash
# E168 A100 exact-input launcher.
# Canary must pass before production is launched. Production uses 0,1,2,3 by user allowlist.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODE="${1:-canary}"
if [ "$MODE" != "canary" ] && [ "$MODE" != "production" ]; then
  echo "usage: $0 {canary|production}" >&2
  exit 2
fi

REMOTE="${REMOTE:-batchcom@61.172.170.106}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/dataset-assist-0/xiayb/workspace/spider}"
REMOTE_SSH_PORT="${REMOTE_SSH_PORT:-30409}"
REMOTE_SSH_KEY="${REMOTE_SSH_KEY:-$HOME/.ssh/id_rsa_tianyiyun}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
A100_GPUS="${A100_GPUS:-0 1 2 3}"
SESSION="${SESSION:-E168_a100_${MODE}_$(date +%Y%m%d_%H%M%S)}"
WAIT_FOR_COMPLETION="${WAIT_FOR_COMPLETION:-0}"
A100_SAVE_VIDEO="${E168_A100_SAVE_VIDEO:-0}"

MANIFEST_DIR="workspace/core4d/results/E168/s6_downstream/cem/manifests"
if [ "$MODE" = "canary" ]; then
  MANIFEST="${MANIFEST_DIR}/cem_canary_manifest.tsv"
  CEM_SUBDIR="canary"
else
  MANIFEST="${MANIFEST_DIR}/cem_production_manifest.tsv"
  CEM_SUBDIR="full"
fi
RUNNER="workspace/core4d/scripts/experiments/E168/run_cem_queue.py"
SHARD_ROOT="${MANIFEST_DIR}/a100_${MODE}_${SESSION}"
SESSION_SCRIPT="${SHARD_ROOT}/run_${MODE}_${SESSION}.sh"
SELECTION_TSV="workspace/core4d/results/E168/s0_environment/a100_${MODE}_gpu_selection_${SESSION}.tsv"

SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=4)
if [ -n "$REMOTE_SSH_PORT" ]; then
  SSH_OPTS+=(-p "$REMOTE_SSH_PORT")
fi
if [ -n "$REMOTE_SSH_KEY" ]; then
  SSH_OPTS+=(-i "$REMOTE_SSH_KEY")
fi
RSYNC_RSH="ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=4"
if [ -n "$REMOTE_SSH_PORT" ]; then
  RSYNC_RSH+=" -p $REMOTE_SSH_PORT"
fi
if [ -n "$REMOTE_SSH_KEY" ]; then
  RSYNC_RSH+=" -i $REMOTE_SSH_KEY"
fi

ssh_remote() {
  ssh "${SSH_OPTS[@]}" "$REMOTE" "$@"
}

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

[ -f "$MANIFEST" ] || { echo "missing manifest: $MANIFEST" >&2; exit 1; }
[ -x "$PYTHON_BIN" ] || { echo "missing local python: $PYTHON_BIN" >&2; exit 1; }

gpu_snapshot="$(ssh_remote "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits")"
compute_snapshot="$(ssh_remote "nvidia-smi --query-compute-apps=gpu_bus_id,gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader,nounits" 2>/dev/null || true)"
for gpu in $A100_GPUS; do
  if ! printf '%s\n' "$gpu_snapshot" | awk -F, '{gsub(/ /, "", $1); print $1}' | grep -qx "$gpu"; then
    echo "A100 GPU $gpu not found in nvidia-smi snapshot" >&2
    exit 1
  fi
done

mkdir -p "$(dirname "$SELECTION_TSV")" "$SHARD_ROOT"
{
  printf 'created_at\tremote\tremote_root\tmode\tsession\tallowed_gpus\tgpu_snapshot\tcompute_snapshot\n'
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date -Is)" "$REMOTE" "$REMOTE_ROOT" "$MODE" "$SESSION" "$A100_GPUS" \
    "$(printf '%s' "$gpu_snapshot" | tr '\n' ';')" \
    "$(printf '%s' "$compute_snapshot" | tr '\n' ';')"
} > "$SELECTION_TSV"

MODE="$MODE" MANIFEST="$MANIFEST" SHARD_ROOT="$SHARD_ROOT" A100_GPUS="$A100_GPUS" \
  "$PYTHON_BIN" - <<'PY'
import csv
import os
from datetime import datetime
from pathlib import Path

mode = os.environ["MODE"]
manifest = Path(os.environ["MANIFEST"])
shard_root = Path(os.environ["SHARD_ROOT"])
gpus = os.environ["A100_GPUS"].split()
object_priority = {"box021": 0, "box004": 1, "bucket004": 2}

rows = list(csv.DictReader(manifest.open(newline="", encoding="utf-8"), delimiter="\t"))
fields = list(rows[0].keys()) if rows else []
if mode == "production":
    rows.sort(key=lambda r: (object_priority.get(r["object_key"], 99), int(r["ordinal"]), r["case_id"]))
else:
    rows.sort(key=lambda r: (object_priority.get(r["object_key"], 99), r["case_id"]))
for idx, row in enumerate(rows, 1):
    row["ordinal"] = idx
    row["preferred_pool"] = "a100"
    row["execution_decision"] = "a100_user_allowlist_0_1_2_3"
    row["status"] = "not_run"
    row["failure_mode"] = ""
    row["gpu_id"] = ""
    row["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")

shard_root.mkdir(parents=True, exist_ok=True)
shards = {gpu: [] for gpu in gpus}
for idx, row in enumerate(rows):
    shards[gpus[idx % len(gpus)]].append(row)

for gpu, gpu_rows in shards.items():
    path = shard_root / f"gpu{gpu}.tsv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(gpu_rows)

queue_path = shard_root / "queue_order.tsv"
with queue_path.open("w", newline="", encoding="utf-8") as f:
    fields_out = ["queue_index", "gpu_id", "object_key", "case_id", "variant"]
    writer = csv.DictWriter(f, fieldnames=fields_out, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for idx, row in enumerate(rows, 1):
        writer.writerow({
            "queue_index": idx,
            "gpu_id": gpus[(idx - 1) % len(gpus)],
            "object_key": row["object_key"],
            "case_id": row["case_id"],
            "variant": row["variant"],
        })

print(f"wrote {len(rows)} {mode} rows into {len(gpus)} shards: {shard_root}")
PY

mapfile -t SYNC_DIRS < <("$PYTHON_BIN" - "$MANIFEST" <<'PY'
import csv
import sys
from pathlib import Path

rows = list(csv.DictReader(open(sys.argv[1], newline="", encoding="utf-8"), delimiter="\t"))
dirs = set()
for row in rows:
    for key in ("target_scene", "trajectory", "scene_act", "contact_mask"):
        dirs.add(str(Path(row[key]).parent))
for path in [
    "example_datasets/processed/core4d/assets/objects/box004",
    "example_datasets/processed/core4d/assets/objects/box021",
    "example_datasets/processed/core4d/assets/objects/bucket004",
    "spider/assets/robots/unitree_g1",
    "examples/config/override",
    "workspace/core4d/scripts/experiments/E168",
    "spider/simulators",
    "spider/optimizers",
]:
    dirs.add(path)
for path in sorted(dirs):
    print(path)
PY
)

SYNC_FILES=(
  "$MANIFEST"
  "$SELECTION_TSV"
  "${MANIFEST_DIR}/cem_execution_manifest_summary.json"
  "examples/run_mjwp.py"
  "spider/config.py"
)

for dir in "${SYNC_DIRS[@]}"; do
  [ -d "$dir" ] || { echo "missing sync dir: $dir" >&2; exit 1; }
done
if [ "$MODE" = "canary" ]; then
  EXTRA_ARGS=(--force --num-samples 64 --max-num-iterations 4)
else
  EXTRA_ARGS=()
fi
if [ "$A100_SAVE_VIDEO" = "0" ]; then
  EXTRA_ARGS+=(--save-video false --allow-missing-video)
fi

{
  printf '#!/usr/bin/env bash\n'
  printf 'set -euo pipefail\n'
  printf 'cd %q\n' "$REMOTE_ROOT"
  printf 'MODE=%q\n' "$MODE"
  printf 'PYTHON_BIN=%q\n' "$PYTHON_BIN"
  printf 'SESSION=%q\n' "$SESSION"
  printf 'SHARD_ROOT=%q\n' "$SHARD_ROOT"
  printf 'RUNNER=%q\n' "$RUNNER"
  printf 'LOG_DIR=%q\n' "logs/E168/cem/${CEM_SUBDIR}"
  printf 'GPU_LIST=(%s)\n' "$A100_GPUS"
  printf 'EXTRA_ARGS=('
  for arg in "${EXTRA_ARGS[@]}"; do
    printf '%q ' "$arg"
  done
  printf ')\n'
  cat <<'EOS'
mkdir -p "$LOG_DIR"
echo "E168 A100 ${MODE} session ${SESSION} started at $(date -Is)"
worker_pids=()
for gpu in "${GPU_LIST[@]}"; do
  shard="${SHARD_ROOT}/gpu${gpu}.tsv"
  if [ ! -s "$shard" ] || [ "$(wc -l < "$shard")" -le 1 ]; then
    echo "skip empty shard gpu=${gpu}"
    continue
  fi
  (
    set -euo pipefail
    echo "worker gpu=${gpu} shard=${shard} started at $(date -Is)"
    "$PYTHON_BIN" "$RUNNER" \
      --mode "$MODE" \
      --manifest-tsv "$shard" \
      --all \
      --python-bin "$PYTHON_BIN" \
      --gpu-id "$gpu" \
      "${EXTRA_ARGS[@]}"
    echo "worker gpu=${gpu} done at $(date -Is)"
  ) > "${LOG_DIR}/a100_${SESSION}_gpu${gpu}.worker.log" 2>&1 &
  worker_pids+=("$!")
done

status=0
for pid in "${worker_pids[@]}"; do
  wait "$pid" || status=1
done
echo "E168 A100 ${MODE} session ${SESSION} finished status=${status} at $(date -Is)"
exit "$status"
EOS
} > "$SESSION_SCRIPT"
chmod +x "$SESSION_SCRIPT"
SYNC_FILES+=("$SESSION_SCRIPT")

for file in "${SYNC_FILES[@]}"; do
  [ -f "$file" ] || { echo "missing sync file: $file" >&2; exit 1; }
done

retry ssh_remote "mkdir -p \
  '$REMOTE_ROOT/${MANIFEST_DIR}' \
  '$REMOTE_ROOT/${SHARD_ROOT}' \
  '$REMOTE_ROOT/workspace/core4d/results/E168/s6_downstream/cem/${CEM_SUBDIR}' \
  '$REMOTE_ROOT/workspace/core4d/results/E168/s0_environment' \
  '$REMOTE_ROOT/logs/E168/cem/${CEM_SUBDIR}'"

for dir in "${SYNC_DIRS[@]}"; do
  retry ssh_remote "mkdir -p '$REMOTE_ROOT/$dir'"
  retry rsync -az -e "$RSYNC_RSH" "$dir/" "${REMOTE}:${REMOTE_ROOT}/${dir}/"
done
retry rsync -az -e "$RSYNC_RSH" "$SHARD_ROOT/" "${REMOTE}:${REMOTE_ROOT}/${SHARD_ROOT}/"
for file in "${SYNC_FILES[@]}"; do
  retry ssh_remote "mkdir -p '$REMOTE_ROOT/$(dirname "$file")'"
  retry rsync -az -e "$RSYNC_RSH" "$file" "${REMOTE}:${REMOTE_ROOT}/$(dirname "$file")/"
done

first_gpu="$(printf '%s\n' $A100_GPUS | head -1)"
first_shard="${SHARD_ROOT}/gpu${first_gpu}.tsv"
retry ssh_remote "cd '$REMOTE_ROOT' && \
  '$PYTHON_BIN' -m py_compile '$RUNNER' examples/run_mjwp.py spider/config.py \
    spider/simulators/mjwp.py spider/optimizers/sampling.py spider/optimizers/sampling_fast.py && \
  '$PYTHON_BIN' '$RUNNER' --mode '$MODE' --manifest-tsv '$first_shard' \
    --dry-run --python-bin '$PYTHON_BIN' --gpu-id '$first_gpu' >/tmp/E168_a100_${MODE}_dryrun.txt && \
  test -s /tmp/E168_a100_${MODE}_dryrun.txt && echo preflight-ok"

retry ssh_remote "cd '$REMOTE_ROOT' && tmux new-session -d -s '$SESSION' -c '$REMOTE_ROOT' 'bash ${SESSION_SCRIPT}'"

echo "Started A100 ${MODE} tmux session: ${SESSION}"
echo "Remote: ${REMOTE}:${REMOTE_ROOT}"
echo "GPUs: ${A100_GPUS}"
echo "Shard root: ${SHARD_ROOT}"
echo "Monitor: ssh -p ${REMOTE_SSH_PORT} -i ${REMOTE_SSH_KEY} ${REMOTE} \"tmux capture-pane -t ${SESSION} -p | tail -80\""
echo "Pull: REMOTE=${REMOTE} REMOTE_ROOT=${REMOTE_ROOT} bash workspace/core4d/scripts/launch/active/pull_E168_remote_a100_results.sh ${MODE}"

if [ "$WAIT_FOR_COMPLETION" = "1" ]; then
  echo "Waiting for ${SESSION} to finish..."
  while ssh_remote "tmux has-session -t '$SESSION'" >/dev/null 2>&1; do
    ssh_remote "tmux capture-pane -t '$SESSION' -p | tail -20" || true
    sleep "${E168_A100_WAIT_POLL_SECONDS:-30}"
  done
  echo "A100 ${MODE} session exited."
fi
