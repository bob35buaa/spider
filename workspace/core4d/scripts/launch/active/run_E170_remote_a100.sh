#!/usr/bin/env bash
# E170 READY-only A100 GPU0-3 launcher. Never kills or pauses other processes.
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
SESSION="${SESSION:-E170_a100_${MODE}_$(date +%Y%m%d_%H%M%S)}"
WAIT_FOR_COMPLETION="${WAIT_FOR_COMPLETION:-0}"
DRY_RUN="${DRY_RUN:-0}"

RESULT_ROOT="workspace/core4d/results/E170"
MANIFEST="$RESULT_ROOT/s6_downstream/manifests/cem_${MODE}_manifest.tsv"
RUNNER="workspace/core4d/scripts/experiments/E170/run_cem_queue.py"
GENERIC_RUNNER="workspace/core4d/scripts/experiments/E169/run_cem_queue.py"
BUILDER="workspace/core4d/scripts/experiments/E170/build_box021_prg_manifest.py"
SHARD_ROOT="$RESULT_ROOT/s6_downstream/manifests/a100_${MODE}_${SESSION}"
SESSION_SCRIPT="$SHARD_ROOT/run_${MODE}_${SESSION}.sh"
SELECTION_TSV="$RESULT_ROOT/s0_environment/a100_${MODE}_gpu_selection_${SESSION}.tsv"
LATEST_SESSION="$RESULT_ROOT/s0_environment/latest_${MODE}_session.txt"
SYNC_SHA="$RESULT_ROOT/s0_environment/sync_sha256_${SESSION}.txt"

SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=4 -p "$REMOTE_SSH_PORT" -i "$REMOTE_SSH_KEY")
RSYNC_RSH="ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=4 -p $REMOTE_SSH_PORT -i $REMOTE_SSH_KEY"
ssh_remote() { ssh "${SSH_OPTS[@]}" "$REMOTE" "$@"; }
retry() {
  local attempt=1 max_attempts="${E170_REMOTE_RETRY_MAX:-4}"
  while ! "$@"; do
    if [ "$attempt" -ge "$max_attempts" ]; then return 1; fi
    sleep "$((attempt * 3))"; attempt=$((attempt + 1))
  done
}

"$PYTHON_BIN" "$BUILDER" --preflight
[ -f "$MANIFEST" ] || { echo "missing manifest: $MANIFEST" >&2; exit 1; }

MODE="$MODE" MANIFEST="$MANIFEST" SHARD_ROOT="$SHARD_ROOT" "$PYTHON_BIN" - <<'PY'
import csv, os
from datetime import datetime
from pathlib import Path
mode, manifest, root = os.environ["MODE"], Path(os.environ["MANIFEST"]), Path(os.environ["SHARD_ROOT"])
rows = list(csv.DictReader(manifest.open(newline="", encoding="utf-8"), delimiter="\t"))
if not rows: raise SystemExit("no READY rows")
fields = list(rows[0])
if mode == "canary":
    variants = {row["retarget_variant_id"] for row in rows}
    if len(rows) != 2 or variants != {"omnirt_v1", "omnirt_v2"}: raise SystemExit(f"invalid dual canary rows={len(rows)} variants={variants}")
else:
    if len(rows) > 24 or any(row["status"] != "READY_FOR_FULL" for row in rows): raise SystemExit("full manifest contains non-READY rows")
root.mkdir(parents=True, exist_ok=True)
for row in rows:
    row["status"] = "not_run"; row["failure_mode"] = ""; row["gpu_id"] = ""
    row["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
for gpu in "0123":
    shard = [row for row in rows if row["assigned_gpu"] == gpu]
    with (root / f"gpu{gpu}.tsv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n"); writer.writeheader(); writer.writerows(shard)
with (root / "queue_order.tsv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=["gpu_id","case_id","retarget_variant_id","variant"], delimiter="\t", lineterminator="\n"); writer.writeheader()
    for row in rows: writer.writerow({"gpu_id":row["assigned_gpu"],"case_id":row["case_id"],"retarget_variant_id":row["retarget_variant_id"],"variant":row["variant"]})
print(f"E170 {mode}: READY rows={len(rows)}")
PY

for gpu in 0 1 2 3; do
  if [ "$(wc -l < "$SHARD_ROOT/gpu${gpu}.tsv")" -gt 1 ]; then
    "$PYTHON_BIN" "$RUNNER" --mode "$MODE" --manifest-tsv "$SHARD_ROOT/gpu${gpu}.tsv" --all --python-bin "$PYTHON_BIN" --gpu-id "$gpu" --dry-run
  fi
done > "$SHARD_ROOT/local_dry_run_commands.txt"
expected="$(($(wc -l < "$MANIFEST") - 1))"
actual="$(wc -l < "$SHARD_ROOT/local_dry_run_commands.txt")"
[ "$actual" -eq "$expected" ] || { echo "dry-run count $actual != READY rows $expected" >&2; exit 1; }
if [ "$DRY_RUN" = "1" ]; then
  echo "E170 $MODE local dry-run passed: commands=$actual"; exit 0
fi

gpu_snapshot="$(retry ssh_remote "nvidia-smi --query-gpu=index,uuid,name,memory.used,utilization.gpu --format=csv,noheader,nounits")"
compute_snapshot="$(retry ssh_remote "nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader,nounits" || true)"
GPU_SNAPSHOT="$gpu_snapshot" "$PYTHON_BIN" - <<'PY'
import os
ids={line.split(",",1)[0].strip() for line in os.environ["GPU_SNAPSHOT"].splitlines()}
missing=sorted(set("0123")-ids)
if missing: raise SystemExit("fixed A100 ids missing: "+",".join(missing))
PY
mkdir -p "$(dirname "$SELECTION_TSV")"
printf '%s\n' "$SESSION" > "$LATEST_SESSION"
{
  printf 'created_at\tremote\tremote_root\tmode\tsession\tallowed_gpus\tpolicy_source\tgpu_snapshot\tcompute_snapshot\n'
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "$REMOTE" "$REMOTE_ROOT" "$MODE" "$SESSION" "$A100_GPUS" "user_explicit_overlap_allowed_no_kill" "$(printf '%s' "$gpu_snapshot" | tr '\n' ';')" "$(printf '%s' "$compute_snapshot" | tr '\n' ';')"
} > "$SELECTION_TSV"

mapfile -t INPUT_DIRS < <("$PYTHON_BIN" - "$MANIFEST" <<'PY'
import csv, sys
from pathlib import Path
rows=list(csv.DictReader(open(sys.argv[1],newline="",encoding="utf-8"),delimiter="\t")); dirs=set()
for row in rows:
    for key in ("target_scene","trajectory","scene_act","contact_mask"): dirs.add(str(Path(row[key]).parent))
for path in ("example_datasets/processed/core4d/assets/objects/box021","spider/assets/robots/unitree_g1","spider/simulators","spider/optimizers","workspace/core4d/scripts/experiments/E169","workspace/core4d/scripts/experiments/E170"): dirs.add(path)
print("\n".join(sorted(dirs)))
PY
)
mapfile -t OVERRIDE_FILES < <("$PYTHON_BIN" - "$MANIFEST" <<'PY'
import csv,sys
rows=list(csv.DictReader(open(sys.argv[1],newline="",encoding="utf-8"),delimiter="\t"))
print("\n".join(sorted({row["override_path"] for row in rows})))
PY
)
SYNC_FILES=("$MANIFEST" "$SELECTION_TSV" "$RUNNER" "$GENERIC_RUNNER" "$BUILDER" "workspace/core4d/scripts/experiments/E170/e170_common.py" "examples/run_mjwp.py" "spider/config.py" "${OVERRIDE_FILES[@]}")

{
  printf '#!/usr/bin/env bash\nset -euo pipefail\ncd %q\n' "$REMOTE_ROOT"
  printf 'MODE=%q\nPYTHON_BIN=%q\nSHARD_ROOT=%q\nRUNNER=%q\n' "$MODE" "$PYTHON_BIN" "$SHARD_ROOT" "$RUNNER"
  cat <<'EOS'
echo "E170 ${MODE} started at $(date -Is)"
pids=()
for gpu in 0 1 2 3; do
  shard="${SHARD_ROOT}/gpu${gpu}.tsv"
  if [ "$(wc -l < "$shard")" -le 1 ]; then continue; fi
  ( "$PYTHON_BIN" "$RUNNER" --mode "$MODE" --manifest-tsv "$shard" --all --python-bin "$PYTHON_BIN" --gpu-id "$gpu" ) > "logs/E170/cem/${MODE}/$(basename "$SHARD_ROOT")_gpu${gpu}.worker.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
echo "E170 ${MODE} finished status=${status} at $(date -Is)"
exit "$status"
EOS
} > "$SESSION_SCRIPT"
chmod +x "$SESSION_SCRIPT"
SYNC_FILES+=("$SESSION_SCRIPT")

for path in "${INPUT_DIRS[@]}"; do [ -d "$path" ] || { echo "missing sync dir: $path" >&2; exit 1; }; done
for path in "${SYNC_FILES[@]}"; do [ -f "$path" ] || { echo "missing sync file: $path" >&2; exit 1; }; done
INPUT_DIRS_NL="$(printf '%s\n' "${INPUT_DIRS[@]}")" SYNC_FILES_NL="$(printf '%s\n' "${SYNC_FILES[@]}")" "$PYTHON_BIN" - "$SYNC_SHA" <<'PY'
import hashlib,os,sys
from pathlib import Path
paths=[]
for raw in os.environ["INPUT_DIRS_NL"].splitlines():
    paths.extend(path for path in Path(raw).rglob("*") if path.is_file())
paths.extend(Path(raw) for raw in os.environ["SYNC_FILES_NL"].splitlines())
with open(sys.argv[1],"w",encoding="utf-8") as stream:
    for path in sorted(set(paths),key=str):
        digest=hashlib.sha256(path.read_bytes()).hexdigest(); stream.write(f"{digest}  {path}\n")
PY
SYNC_FILES+=("$SYNC_SHA")

retry ssh_remote "mkdir -p '$REMOTE_ROOT/$SHARD_ROOT' '$REMOTE_ROOT/$RESULT_ROOT/s6_downstream/cem/$MODE' '$REMOTE_ROOT/logs/E170/cem/$MODE'"
for path in "${INPUT_DIRS[@]}"; do retry ssh_remote "mkdir -p '$REMOTE_ROOT/$path'"; retry rsync -az -e "$RSYNC_RSH" "$path/" "${REMOTE}:${REMOTE_ROOT}/${path}/"; done
retry rsync -az -e "$RSYNC_RSH" "$SHARD_ROOT/" "${REMOTE}:${REMOTE_ROOT}/${SHARD_ROOT}/"
for path in "${SYNC_FILES[@]}"; do retry ssh_remote "mkdir -p '$REMOTE_ROOT/$(dirname "$path")'"; retry rsync -az -e "$RSYNC_RSH" "$path" "${REMOTE}:${REMOTE_ROOT}/$(dirname "$path")/"; done
retry ssh_remote "cd '$REMOTE_ROOT' && sha256sum -c '$SYNC_SHA'"
retry ssh_remote "cd '$REMOTE_ROOT' && '$PYTHON_BIN' -m py_compile '$RUNNER' '$GENERIC_RUNNER' spider/config.py spider/optimizers/sampling.py spider/optimizers/sampling_fast.py spider/simulators/mjwp.py"
retry ssh_remote "cd '$REMOTE_ROOT' && tmux new-session -d -s '$SESSION' -c '$REMOTE_ROOT' 'bash $SESSION_SCRIPT'"

echo "Started E170 ${MODE}: session=${SESSION} READY=$expected GPUs=0,1,2,3"
if [ "$WAIT_FOR_COMPLETION" = "1" ]; then while ssh_remote "tmux has-session -t '$SESSION'" >/dev/null 2>&1; do sleep 30; done; fi
