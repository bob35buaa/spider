#!/usr/bin/env bash
# E168 6-GPU full CEM launcher: A6000 GPU0/1 + A100 GPU0/1/2/3.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

SESSION_TAG="${SESSION_TAG:-$(date +%Y%m%d_%H%M%S)}"
SESSION_A100="${SESSION_A100:-E168_6gpu_a100_${SESSION_TAG}}"
SESSION_A6000="${SESSION_A6000:-E168_6gpu_a6000_${SESSION_TAG}}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

A100_REMOTE="${A100_REMOTE:-batchcom@61.172.170.106}"
A100_ROOT="${A100_ROOT:-/home/dataset-assist-0/xiayb/workspace/spider}"
A100_SSH_PORT="${A100_SSH_PORT:-30409}"
A100_SSH_KEY="${A100_SSH_KEY:-$HOME/.ssh/id_rsa_tianyiyun}"
A100_GPUS="${A100_GPUS:-0 1 2 3}"

A6000_REMOTE="${A6000_REMOTE:-spider-remote}"
A6000_ROOT="${A6000_ROOT:-/home/xiayb/pHRI_workspace/spider}"
A6000_GPUS="${A6000_GPUS:-0 1}"

MANIFEST_DIR="workspace/core4d/results/E168/s6_downstream/cem/manifests"
MANIFEST="${MANIFEST_DIR}/cem_production_manifest.tsv"
RUNNER="workspace/core4d/scripts/experiments/E168/run_cem_queue.py"
SHARD_ROOT="${MANIFEST_DIR}/remote6_production_${SESSION_TAG}"
SELECTION_TSV="workspace/core4d/results/E168/s0_environment/remote6_full_gpu_selection_${SESSION_TAG}.tsv"

a100_ssh() {
  ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=4 \
    -p "$A100_SSH_PORT" -i "$A100_SSH_KEY" "$A100_REMOTE" "$@"
}

a6000_ssh() {
  ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=4 \
    "$A6000_REMOTE" "$@"
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

retry_capture() {
  local output
  local attempt=1
  local max_attempts="${E168_REMOTE_RETRY_MAX:-4}"
  while true; do
    if output="$("$@" 2>&1)"; then
      printf '%s\n' "$output"
      return 0
    fi
    if [ "$attempt" -ge "$max_attempts" ]; then
      printf '%s\n' "$output" >&2
      echo "Command failed after ${attempt} attempts: $*" >&2
      return 1
    fi
    printf '%s\n' "$output" >&2
    echo "Retry ${attempt}/${max_attempts}: $*" >&2
    sleep "$((attempt * 3))"
    attempt=$((attempt + 1))
  done
}

rsync_a100() {
  rsync -az -e "ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=4 -p $A100_SSH_PORT -i $A100_SSH_KEY" "$@"
}

rsync_a6000() {
  rsync -az -e "ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=4" "$@"
}

[ -f "$MANIFEST" ] || { echo "missing manifest: $MANIFEST" >&2; exit 1; }
[ -x "$PYTHON_BIN" ] || { echo "missing python: $PYTHON_BIN" >&2; exit 1; }

a100_gpu_snapshot="$(retry_capture a100_ssh "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits")"
a6000_gpu_snapshot="$(retry_capture a6000_ssh "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits")"
mkdir -p "$(dirname "$SELECTION_TSV")" "$SHARD_ROOT"
{
  printf 'created_at\tprofile\tremote\tremote_root\tallowed_gpus\tgpu_snapshot\n'
  printf '%s\ta100\t%s\t%s\t%s\t%s\n' "$(date -Is)" "$A100_REMOTE" "$A100_ROOT" "$A100_GPUS" "$(printf '%s' "$a100_gpu_snapshot" | tr '\n' ';')"
  printf '%s\ta6000\t%s\t%s\t%s\t%s\n' "$(date -Is)" "$A6000_REMOTE" "$A6000_ROOT" "$A6000_GPUS" "$(printf '%s' "$a6000_gpu_snapshot" | tr '\n' ';')"
} > "$SELECTION_TSV"

MANIFEST="$MANIFEST" SHARD_ROOT="$SHARD_ROOT" A100_GPUS="$A100_GPUS" A6000_GPUS="$A6000_GPUS" \
  "$PYTHON_BIN" - <<'PY'
import csv
import os
from datetime import datetime
from pathlib import Path

manifest = Path(os.environ["MANIFEST"])
shard_root = Path(os.environ["SHARD_ROOT"])
a100_gpus = os.environ["A100_GPUS"].split()
a6000_gpus = os.environ["A6000_GPUS"].split()
workers = [("a100", gpu) for gpu in a100_gpus] + [("a6000", gpu) for gpu in a6000_gpus]
object_priority = {"box021": 0, "box004": 1, "bucket004": 2}

rows = list(csv.DictReader(manifest.open(newline="", encoding="utf-8"), delimiter="\t"))
fields = list(rows[0].keys()) if rows else []
def complete(row):
    return all(Path(row[key]).is_file() for key in ("result_npz", "outdir_npz", "config_act"))

remaining = []
for row in rows:
    if complete(row):
        row["status"] = "run_complete_pending_eval"
        row["failure_mode"] = ""
        continue
    row["status"] = "not_run"
    row["failure_mode"] = ""
    row["gpu_id"] = ""
    row["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    remaining.append(row)
remaining.sort(key=lambda r: (object_priority.get(r["object_key"], 99), int(r["ordinal"]), r["case_id"]))

shards = {worker: [] for worker in workers}
for idx, row in enumerate(remaining):
    worker = workers[idx % len(workers)]
    row = dict(row)
    row["preferred_pool"] = worker[0]
    row["execution_decision"] = "remote6_a6000_2gpu_a100_4gpu"
    shards[worker].append(row)

shard_root.mkdir(parents=True, exist_ok=True)
for (profile, gpu), gpu_rows in shards.items():
    path = shard_root / f"{profile}_gpu{gpu}.tsv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(gpu_rows)

with (shard_root / "queue_order.tsv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["queue_index", "profile", "gpu_id", "object_key", "case_id", "variant"],
        delimiter="\t",
        lineterminator="\n",
    )
    writer.writeheader()
    for idx, row in enumerate(remaining, 1):
        profile, gpu = workers[(idx - 1) % len(workers)]
        writer.writerow({
            "queue_index": idx,
            "profile": profile,
            "gpu_id": gpu,
            "object_key": row["object_key"],
            "case_id": row["case_id"],
            "variant": row["variant"],
        })

with manifest.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)

print(f"remote6 remaining_rows={len(remaining)} shard_root={shard_root}")
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

write_session_script() {
  local profile="$1"
  local session="$2"
  local remote_root="$3"
  local gpus="$4"
  local script_path="${SHARD_ROOT}/run_${profile}_${SESSION_TAG}.sh"
  {
    printf '#!/usr/bin/env bash\nset -euo pipefail\n'
    printf 'cd %q\n' "$remote_root"
    printf 'SESSION=%q\n' "$session"
    printf 'PROFILE=%q\n' "$profile"
    printf 'PYTHON_BIN=%q\n' "$PYTHON_BIN"
    printf 'RUNNER=%q\n' "$RUNNER"
    printf 'SHARD_ROOT=%q\n' "$SHARD_ROOT"
    printf 'GPU_LIST=(%s)\n' "$gpus"
    printf 'LOG_DIR=%q\n' "logs/E168/cem/full"
    cat <<'EOS'
mkdir -p "$LOG_DIR"
echo "E168 remote6 ${PROFILE} full session ${SESSION} started at $(date -Is)"
worker_pids=()
for gpu in "${GPU_LIST[@]}"; do
  shard="${SHARD_ROOT}/${PROFILE}_gpu${gpu}.tsv"
  if [ ! -s "$shard" ] || [ "$(wc -l < "$shard")" -le 1 ]; then
    echo "skip empty shard ${shard}"
    continue
  fi
  extra_args=()
  if [ "$PROFILE" = "a100" ]; then
    extra_args+=(--save-video false --allow-missing-video)
  fi
  (
    set -euo pipefail
    echo "worker profile=${PROFILE} gpu=${gpu} shard=${shard} started at $(date -Is)"
    "$PYTHON_BIN" "$RUNNER" \
      --mode production \
      --manifest-tsv "$shard" \
      --all \
      --python-bin "$PYTHON_BIN" \
      --gpu-id "$gpu" \
      "${extra_args[@]}"
    echo "worker profile=${PROFILE} gpu=${gpu} done at $(date -Is)"
  ) > "${LOG_DIR}/remote6_${SESSION}_${PROFILE}_gpu${gpu}.worker.log" 2>&1 &
  worker_pids+=("$!")
done
status=0
for pid in "${worker_pids[@]}"; do
  wait "$pid" || status=1
done
echo "E168 remote6 ${PROFILE} full session ${SESSION} finished status=${status} at $(date -Is)"
exit "$status"
EOS
  } > "$script_path"
  chmod +x "$script_path"
}

write_session_script "a100" "$SESSION_A100" "$A100_ROOT" "$A100_GPUS"
write_session_script "a6000" "$SESSION_A6000" "$A6000_ROOT" "$A6000_GPUS"

for dir in "${SYNC_DIRS[@]}"; do
  [ -d "$dir" ] || { echo "missing sync dir: $dir" >&2; exit 1; }
done
retry rsync_a100 -R "${SYNC_DIRS[@]}" "${A100_REMOTE}:${A100_ROOT}/"
retry rsync_a6000 -R "${SYNC_DIRS[@]}" "${A6000_REMOTE}:${A6000_ROOT}/"

for remote in a100 a6000; do
  if [ "$remote" = "a100" ]; then
    rssh=a100_ssh
    rroot="$A100_ROOT"
    rremote="$A100_REMOTE"
    rsync_fn=rsync_a100
    shard_glob="a100_gpu*.tsv"
  else
    rssh=a6000_ssh
    rroot="$A6000_ROOT"
    rremote="$A6000_REMOTE"
    rsync_fn=rsync_a6000
    shard_glob="a6000_gpu*.tsv"
  fi
  retry "$rssh" "mkdir -p '$rroot/${MANIFEST_DIR}' '$rroot/${SHARD_ROOT}' '$rroot/workspace/core4d/results/E168/s6_downstream/cem/full' '$rroot/logs/E168/cem/full'"
  retry "$rsync_fn" "$MANIFEST" "${rremote}:${rroot}/${MANIFEST_DIR}/"
  retry "$rsync_fn" "$SELECTION_TSV" "${rremote}:${rroot}/$(dirname "$SELECTION_TSV")/"
  retry "$rsync_fn" "${SHARD_ROOT}/queue_order.tsv" "${rremote}:${rroot}/${SHARD_ROOT}/"
  retry "$rsync_fn" ${SHARD_ROOT}/${shard_glob} "${rremote}:${rroot}/${SHARD_ROOT}/"
  retry "$rsync_fn" "${SHARD_ROOT}/run_${remote}_${SESSION_TAG}.sh" "${rremote}:${rroot}/${SHARD_ROOT}/"
  retry "$rsync_fn" examples/run_mjwp.py "${rremote}:${rroot}/examples/"
  retry "$rsync_fn" spider/config.py "${rremote}:${rroot}/spider/"
done

first_a100="$(printf '%s\n' $A100_GPUS | head -1)"
first_a6000="$(printf '%s\n' $A6000_GPUS | head -1)"
retry a100_ssh "cd '$A100_ROOT' && '$PYTHON_BIN' -m py_compile '$RUNNER' examples/run_mjwp.py spider/config.py spider/simulators/mjwp.py spider/optimizers/sampling.py spider/optimizers/sampling_fast.py && '$PYTHON_BIN' '$RUNNER' --mode production --manifest-tsv '${SHARD_ROOT}/a100_gpu${first_a100}.tsv' --dry-run --python-bin '$PYTHON_BIN' --gpu-id '$first_a100' --save-video false --allow-missing-video >/tmp/E168_remote6_a100_dryrun.txt && test -s /tmp/E168_remote6_a100_dryrun.txt && echo a100-preflight-ok"
retry a6000_ssh "cd '$A6000_ROOT' && '$PYTHON_BIN' -m py_compile '$RUNNER' examples/run_mjwp.py spider/config.py spider/simulators/mjwp.py spider/optimizers/sampling.py spider/optimizers/sampling_fast.py && '$PYTHON_BIN' '$RUNNER' --mode production --manifest-tsv '${SHARD_ROOT}/a6000_gpu${first_a6000}.tsv' --dry-run --python-bin '$PYTHON_BIN' --gpu-id '$first_a6000' >/tmp/E168_remote6_a6000_dryrun.txt && test -s /tmp/E168_remote6_a6000_dryrun.txt && echo a6000-preflight-ok"

retry a100_ssh "cd '$A100_ROOT' && tmux new-session -d -s '$SESSION_A100' -c '$A100_ROOT' 'bash ${SHARD_ROOT}/run_a100_${SESSION_TAG}.sh'"
retry a6000_ssh "cd '$A6000_ROOT' && tmux new-session -d -s '$SESSION_A6000' -c '$A6000_ROOT' 'bash ${SHARD_ROOT}/run_a6000_${SESSION_TAG}.sh'"

echo "Started remote6 full sessions:"
echo "  A100 : ${SESSION_A100} (${A100_REMOTE}:${A100_ROOT}) GPUs ${A100_GPUS}"
echo "  A6000: ${SESSION_A6000} (${A6000_REMOTE}:${A6000_ROOT}) GPUs ${A6000_GPUS}"
echo "Shard root: ${SHARD_ROOT}"
echo "Pull: bash workspace/core4d/scripts/launch/active/pull_E168_remote_6gpu_results.sh production ${SESSION_TAG}"
