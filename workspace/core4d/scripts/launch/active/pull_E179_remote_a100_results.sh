#!/usr/bin/env bash
# Pull and validate only the 12 artifacts registered by E179 A100 execution.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
REMOTE="${REMOTE:-batchcom@61.172.170.106}"
REMOTE_SSH_PORT="${REMOTE_SSH_PORT:-30409}"
REMOTE_SSH_KEY="${REMOTE_SSH_KEY:-/home/ubuntu/.ssh/id_rsa_tianyiyun}"
ALLOW_INCOMPLETE="${ALLOW_INCOMPLETE:-0}"
RSYNC_IO_TIMEOUT="${E179_PULL_IO_TIMEOUT_SECONDS:-180}"
FILE_TRANSPORT="${E179_PULL_FILE_TRANSPORT:-rsync}"
RESULT_ROOT="workspace/core4d/results/E179"
LATEST_JSON="$RESULT_ROOT/s0_environment/a100_full_latest.json"
RUNNER="workspace/core4d/scripts/experiments/E179/run_cem_queue.py"

[ -f "$LATEST_JSON" ] || {
  echo "missing E179 execution pointer: $LATEST_JSON" >&2
  exit 3
}
[[ "$RSYNC_IO_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || {
  echo "E179_PULL_IO_TIMEOUT_SECONDS must be a positive integer" >&2
  exit 3
}
[[ "$FILE_TRANSPORT" =~ ^(rsync|scp)$ ]] || {
  echo "E179_PULL_FILE_TRANSPORT must be rsync or scp" >&2
  exit 3
}

mapfile -t META < <(
  "$PYTHON_BIN" - "$LATEST_JSON" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
if payload.get("selected_gpus") != ["2", "3", "6", "7"]:
    raise SystemExit("E179 execution pointer does not contain fixed GPUs 2,3,6,7")
print(payload["session"])
print(payload["remote_root"])
print(payload["shard_root"])
PY
)
SESSION="${META[0]}"
REMOTE_RUN_ROOT="${META[1]}"
SHARD_ROOT="${META[2]}"
EXECUTION_ROOT="$(dirname "$SHARD_ROOT")"
PULL_LIST="$EXECUTION_ROOT/pull_files.txt"
PULL_SUMMARY="$EXECUTION_ROOT/pull_summary.json"

SSH_OPTS=(
  -o BatchMode=yes
  -o ConnectTimeout=20
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=4
  -p "$REMOTE_SSH_PORT"
  -i "$REMOTE_SSH_KEY"
)
RSYNC_RSH="ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=4 -p ${REMOTE_SSH_PORT} -i ${REMOTE_SSH_KEY}"
retry() {
  local attempt=1
  local max_attempts="${E179_PULL_RETRY_MAX:-6}"
  while true; do
    if "$@"; then
      return 0
    fi
    if [ "$attempt" -ge "$max_attempts" ]; then
      echo "pull failed after ${attempt} attempts: $*" >&2
      return 1
    fi
    sleep "$((attempt * 2))"
    attempt=$((attempt + 1))
  done
}

mkdir -p "$EXECUTION_ROOT"
retry rsync -az --partial -e "$RSYNC_RSH" \
  "${REMOTE}:${REMOTE_RUN_ROOT}/${EXECUTION_ROOT}/" "$EXECUTION_ROOT/"

SHARD_ROOT="$SHARD_ROOT" PULL_LIST="$PULL_LIST" SESSION="$SESSION" \
"$PYTHON_BIN" - <<'PY'
import csv
import os
from pathlib import Path

root = Path(os.environ["SHARD_ROOT"])
rows = []
files = set()
for gpu in ("2", "3", "6", "7"):
    shard = root / f"gpu{gpu}.tsv"
    if not shard.is_file():
        raise FileNotFoundError(shard)
    current = list(
        csv.DictReader(shard.open(encoding="utf-8"), delimiter="\t")
    )
    if len(current) != 3:
        raise SystemExit(f"{shard} must contain exactly 3 rows")
    rows.extend(current)
    files.add(
        f"logs/E179/cem/full/{os.environ['SESSION']}_gpu{gpu}.worker.log"
    )
if len(rows) != 12 or len({row["case_id"] for row in rows}) != 12:
    raise SystemExit("remote execution union must contain 12 unique rows")
for row in rows:
    for key in ("result_npz", "outdir_npz", "config_act", "log", "video"):
        path = Path(row[key])
        if (
            path.is_absolute()
            or ".." in path.parts
            or not str(path).startswith(
                ("workspace/core4d/results/E179/", "logs/E179/")
            )
        ):
            raise SystemExit(f"out-of-scope pull path: {key}={path}")
        files.add(str(path))
Path(os.environ["PULL_LIST"]).write_text(
    "\n".join(sorted(files)) + "\n", encoding="utf-8"
)
PY

while IFS= read -r relative_path; do
  [ -n "$relative_path" ] || continue
  if [[ "$relative_path" == *.mp4 ]] && \
     ! ssh "${SSH_OPTS[@]}" "$REMOTE" \
       "test -f '$REMOTE_RUN_ROOT/$relative_path'"; then
    continue
  fi
  mkdir -p "$(dirname "$relative_path")"
  if [ "$FILE_TRANSPORT" = "scp" ]; then
    retry scp -O \
      -o BatchMode=yes \
      -o ConnectTimeout=20 \
      -o ServerAliveInterval=30 \
      -o ServerAliveCountMax=4 \
      -P "$REMOTE_SSH_PORT" \
      -i "$REMOTE_SSH_KEY" \
      "${REMOTE}:${REMOTE_RUN_ROOT}/${relative_path}" "$relative_path"
  else
    retry rsync -azs --partial --timeout="$RSYNC_IO_TIMEOUT" \
      --ignore-missing-args \
      -e "$RSYNC_RSH" \
      "${REMOTE}:${REMOTE_RUN_ROOT}/${relative_path}" "$relative_path"
  fi
done < "$PULL_LIST"

SHARD_ROOT="$SHARD_ROOT" PULL_SUMMARY="$PULL_SUMMARY" \
LATEST_JSON="$LATEST_JSON" ALLOW_INCOMPLETE="$ALLOW_INCOMPLETE" \
"$PYTHON_BIN" - <<'PY'
import csv
import hashlib
import importlib.util
import json
import os
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

manifest_root = Path(
    "workspace/core4d/results/E179/s6_downstream/manifests"
)
runner_path = Path(
    "workspace/core4d/scripts/experiments/E179/run_cem_queue.py"
)
spec = importlib.util.spec_from_file_location("e179_pull_validation", runner_path)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

rows = []
for gpu in ("2", "3", "6", "7"):
    source = Path(os.environ["SHARD_ROOT"]) / f"gpu{gpu}.tsv"
    destination = manifest_root / f"cem_full_a100_gpu{gpu}.tsv"
    shutil.copy2(source, destination)
    rows.extend(
        csv.DictReader(source.open(encoding="utf-8"), delimiter="\t")
    )
artifacts = []
incomplete = []
for row in rows:
    failures = module.validate_runtime_outputs(row)
    if failures:
        incomplete.append(
            {
                "case_id": row["case_id"],
                "status": row["status"],
                "failures": failures,
            }
        )
    item = {"case_id": row["case_id"], "status": row["status"]}
    for key in ("result_npz", "outdir_npz", "config_act", "log", "video"):
        path = Path(row[key])
        item[f"{key}_exists"] = path.is_file()
        item[f"{key}_bytes"] = path.stat().st_size if path.is_file() else 0
        item[f"{key}_sha256"] = (
            hashlib.sha256(path.read_bytes()).hexdigest()
            if path.is_file()
            else ""
        )
    artifacts.append(item)
payload = {
    "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    "execution_pointer": os.environ["LATEST_JSON"],
    "remote_rows": len(rows),
    "status_counts": dict(Counter(row["status"] for row in rows)),
    "runtime_pass_rows": len(rows) - len(incomplete),
    "incomplete": incomplete,
    "artifacts": artifacts,
    "status": "pass" if not incomplete and len(rows) == 12 else "incomplete",
}
Path(os.environ["PULL_SUMMARY"]).write_text(
    json.dumps(payload, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps({key: payload[key] for key in (
    "remote_rows", "status_counts", "runtime_pass_rows", "status"
)}, indent=2))
if payload["status"] != "pass" and os.environ["ALLOW_INCOMPLETE"] != "1":
    raise SystemExit(2)
PY

"$PYTHON_BIN" "$RUNNER" --merge-worker-shards
echo "E179 remote pull complete: $PULL_SUMMARY"
