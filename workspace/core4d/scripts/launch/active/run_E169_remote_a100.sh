#!/usr/bin/env bash
# E169 fixed A100 GPU0-3 overlap launcher. Never kills existing remote processes.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODE="${1:-canary}"
if [ "$MODE" != "canary" ] && [ "$MODE" != "full" ]; then
  echo "usage: $0 {canary|full}" >&2
  exit 2
fi

REMOTE="${REMOTE:-batchcom@61.172.170.106}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/dataset-assist-0/xiayb/workspace/spider}"
REMOTE_SSH_PORT="${REMOTE_SSH_PORT:-30409}"
REMOTE_SSH_KEY="${REMOTE_SSH_KEY:-$HOME/.ssh/id_rsa_tianyiyun}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
A100_GPUS="0 1 2 3"
SESSION="${SESSION:-E169_a100_${MODE}_$(date +%Y%m%d_%H%M%S)}"
WAIT_FOR_COMPLETION="${WAIT_FOR_COMPLETION:-0}"

MANIFEST_DIR="workspace/core4d/results/E169/manifests"
MANIFEST="${MANIFEST_DIR}/cem_${MODE}_manifest.tsv"
RUNNER="workspace/core4d/scripts/experiments/E169/run_cem_queue.py"
SHARD_ROOT="${MANIFEST_DIR}/a100_${MODE}_${SESSION}"
SESSION_SCRIPT="${SHARD_ROOT}/run_${MODE}_${SESSION}.sh"
SELECTION_TSV="workspace/core4d/results/E169/s0_environment/a100_${MODE}_gpu_selection_${SESSION}.tsv"
LATEST_SESSION="workspace/core4d/results/E169/s0_environment/latest_${MODE}_session.txt"

SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=4 -p "$REMOTE_SSH_PORT" -i "$REMOTE_SSH_KEY")
RSYNC_RSH="ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=4 -p $REMOTE_SSH_PORT -i $REMOTE_SSH_KEY"

ssh_remote() { ssh "${SSH_OPTS[@]}" "$REMOTE" "$@"; }
retry() {
  local attempt=1
  local max_attempts="${E169_REMOTE_RETRY_MAX:-4}"
  while ! "$@"; do
    if [ "$attempt" -ge "$max_attempts" ]; then
      echo "command failed after ${attempt} attempts: $*" >&2
      return 1
    fi
    sleep "$((attempt * 3))"
    attempt=$((attempt + 1))
  done
}

"$PYTHON_BIN" workspace/core4d/scripts/experiments/E169/build_lowerbody_object_factorial_manifest.py --stage preflight
[ -f "$MANIFEST" ] || { echo "missing manifest: $MANIFEST" >&2; exit 1; }

gpu_snapshot="$(retry ssh_remote "nvidia-smi --query-gpu=index,uuid,name,memory.used,utilization.gpu --format=csv,noheader,nounits")"
compute_snapshot="$(retry ssh_remote "nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader,nounits" || true)"
GPU_SNAPSHOT="$gpu_snapshot" A100_GPUS="$A100_GPUS" \
  "$PYTHON_BIN" - <<'PY'
import os
gpus = os.environ["A100_GPUS"].split()
rows = {}
for line in os.environ["GPU_SNAPSHOT"].splitlines():
    parts = [part.strip() for part in line.split(",")]
    if len(parts) >= 5:
        rows[parts[0]] = {"uuid": parts[1], "memory": int(parts[3])}
missing = [gpu for gpu in gpus if gpu not in rows]
if missing:
    raise SystemExit("A100 fixed GPU ids missing: " + ",".join(missing))
print("A100 fixed GPU ids 0,1,2,3 present; overlap launch explicitly allowed")
PY

mkdir -p "$(dirname "$SELECTION_TSV")" "$SHARD_ROOT"
printf '%s\n' "$SESSION" > "$LATEST_SESSION"
{
  printf 'created_at\tremote\tremote_root\tmode\tsession\tallowed_gpus\tpolicy_source\tgpu_snapshot\tcompute_snapshot\n'
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date -Is)" "$REMOTE" "$REMOTE_ROOT" "$MODE" "$SESSION" "$A100_GPUS" \
    "user_explicit_overlap_allowed_no_kill" "$(printf '%s' "$gpu_snapshot" | tr '\n' ';')" \
    "$(printf '%s' "$compute_snapshot" | tr '\n' ';')"
} > "$SELECTION_TSV"

MODE="$MODE" MANIFEST="$MANIFEST" SHARD_ROOT="$SHARD_ROOT" "$PYTHON_BIN" - <<'PY'
import csv, os
from datetime import datetime
from pathlib import Path
mode = os.environ["MODE"]
manifest = Path(os.environ["MANIFEST"])
root = Path(os.environ["SHARD_ROOT"])
rows = list(csv.DictReader(manifest.open(newline="", encoding="utf-8"), delimiter="\t"))
fields = list(rows[0])
root.mkdir(parents=True, exist_ok=True)
for row in rows:
    row["status"] = "not_run"
    row["failure_mode"] = ""
    row["gpu_id"] = ""
    row["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
for gpu in "0123":
    shard = [row for row in rows if row["assigned_gpu"] == gpu]
    expected = 1 if mode == "canary" else 7
    if len(shard) != expected or len({row["case_id"] for row in shard}) != 1:
        raise SystemExit(f"invalid gpu{gpu} shard rows={len(shard)}")
    with (root / f"gpu{gpu}.tsv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader(); writer.writerows(shard)
with (root / "queue_order.tsv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=["gpu_id","case_id","cell_id","variant"], delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row[key if key != "gpu_id" else "assigned_gpu"] for key in writer.fieldnames})
print(f"wrote fixed E169 {mode} shards: {len(rows)} rows")
PY

mapfile -t INPUT_DIRS < <("$PYTHON_BIN" - "$MANIFEST" <<'PY'
import csv, sys
from pathlib import Path
rows = list(csv.DictReader(open(sys.argv[1], newline="", encoding="utf-8"), delimiter="\t"))
dirs = set()
for row in rows:
    for key in ("target_scene", "trajectory", "scene_act", "contact_mask"):
        dirs.add(str(Path(row[key]).parent))
for path in [
    "example_datasets/processed/core4d/assets/objects/box021",
    "spider/assets/robots/unitree_g1",
    "spider/simulators",
    "spider/optimizers",
    "workspace/core4d/scripts/experiments/E169",
]:
    dirs.add(path)
print("\n".join(sorted(dirs)))
PY
)
mapfile -t OVERRIDE_FILES < <("$PYTHON_BIN" - "$MANIFEST" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1], newline="", encoding="utf-8"), delimiter="\t"))
print("\n".join(sorted({row["override_path"] for row in rows})))
PY
)
SYNC_FILES=("$MANIFEST" "$SELECTION_TSV" "$RUNNER" "examples/run_mjwp.py" "spider/config.py" "${OVERRIDE_FILES[@]}")

{
  printf '#!/usr/bin/env bash\nset -euo pipefail\ncd %q\n' "$REMOTE_ROOT"
  printf 'MODE=%q\nPYTHON_BIN=%q\nSHARD_ROOT=%q\nRUNNER=%q\n' "$MODE" "$PYTHON_BIN" "$SHARD_ROOT" "$RUNNER"
  printf 'GPU_LIST=(0 1 2 3)\n'
  cat <<'EOS'
echo "E169 ${MODE} started at $(date -Is)"
pids=()
for gpu in "${GPU_LIST[@]}"; do
  (
    "$PYTHON_BIN" "$RUNNER" --mode "$MODE" --manifest-tsv "${SHARD_ROOT}/gpu${gpu}.tsv" --all --python-bin "$PYTHON_BIN" --gpu-id "$gpu"
  ) > "logs/E169/cem/${MODE}/$(basename "$SHARD_ROOT")_gpu${gpu}.worker.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
echo "E169 ${MODE} finished status=${status} at $(date -Is)"
exit "$status"
EOS
} > "$SESSION_SCRIPT"
chmod +x "$SESSION_SCRIPT"
SYNC_FILES+=("$SESSION_SCRIPT")

for path in "${INPUT_DIRS[@]}"; do [ -d "$path" ] || { echo "missing sync dir: $path" >&2; exit 1; }; done
for path in "${SYNC_FILES[@]}"; do [ -f "$path" ] || { echo "missing sync file: $path" >&2; exit 1; }; done

retry ssh_remote "mkdir -p '$REMOTE_ROOT/$SHARD_ROOT' '$REMOTE_ROOT/workspace/core4d/results/E169/cem/$MODE' '$REMOTE_ROOT/logs/E169/cem/$MODE'"
for path in "${INPUT_DIRS[@]}"; do
  retry ssh_remote "mkdir -p '$REMOTE_ROOT/$path'"
  retry rsync -az -e "$RSYNC_RSH" "$path/" "${REMOTE}:${REMOTE_ROOT}/${path}/"
done
retry rsync -az -e "$RSYNC_RSH" "$SHARD_ROOT/" "${REMOTE}:${REMOTE_ROOT}/${SHARD_ROOT}/"
for path in "${SYNC_FILES[@]}"; do
  retry ssh_remote "mkdir -p '$REMOTE_ROOT/$(dirname "$path")'"
  retry rsync -az -e "$RSYNC_RSH" "$path" "${REMOTE}:${REMOTE_ROOT}/$(dirname "$path")/"
done

retry ssh_remote "cd '$REMOTE_ROOT' && '$PYTHON_BIN' -m py_compile spider/config.py spider/optimizers/sampling.py spider/optimizers/sampling_fast.py spider/simulators/mjwp.py '$RUNNER' && '$PYTHON_BIN' '$RUNNER' --mode '$MODE' --manifest-tsv '$SHARD_ROOT/gpu0.tsv' --python-bin '$PYTHON_BIN' --gpu-id 0 --all --dry-run >/tmp/E169_${MODE}_dryrun.txt && test -s /tmp/E169_${MODE}_dryrun.txt"
retry ssh_remote "cd '$REMOTE_ROOT' && tmux new-session -d -s '$SESSION' -c '$REMOTE_ROOT' 'bash $SESSION_SCRIPT'"

echo "Started E169 ${MODE}: session=${SESSION} GPUs=0,1,2,3"
echo "Monitor: ssh -p ${REMOTE_SSH_PORT} -i ${REMOTE_SSH_KEY} ${REMOTE} \"tmux capture-pane -t ${SESSION} -p | tail -80\""
if [ "$WAIT_FOR_COMPLETION" = "1" ]; then
  while ssh_remote "tmux has-session -t '$SESSION'" >/dev/null 2>&1; do sleep 30; done
fi
