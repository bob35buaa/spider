#!/usr/bin/env bash
# Watch/pull E168 remote6 full CEM until all rows finish or sessions stop.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"

SESSION_TAG="${SESSION_TAG:-20260717_173420}"
POLL_INTERVAL="${POLL_INTERVAL:-600}"
PULL_RETRY_MAX="${E168_REMOTE_RETRY_MAX:-8}"
LOG_DIR="logs/E168/cem/monitor"
LOG_FILE="${LOG_DIR}/watch_remote6_${SESSION_TAG}.log"
PULL_SCRIPT="workspace/core4d/scripts/launch/active/pull_E168_remote_6gpu_results.sh"
MANIFEST="workspace/core4d/results/E168/s6_downstream/cem/manifests/cem_production_manifest.tsv"

A100_REMOTE="${A100_REMOTE:-batchcom@61.172.170.106}"
A100_SESSION="${A100_SESSION:-E168_6gpu_a100_${SESSION_TAG}}"
A100_SSH_PORT="${A100_SSH_PORT:-30409}"
A100_SSH_KEY="${A100_SSH_KEY:-$HOME/.ssh/id_rsa_tianyiyun}"
A6000_REMOTE="${A6000_REMOTE:-spider-remote}"
A6000_SESSION="${A6000_SESSION:-E168_6gpu_a6000_${SESSION_TAG}}"

mkdir -p "$LOG_DIR"

ts() {
  date -Is
}

session_alive_a100() {
  ssh -o BatchMode=yes -o ConnectTimeout=10 \
    -p "$A100_SSH_PORT" -i "$A100_SSH_KEY" "$A100_REMOTE" \
    "tmux has-session -t '$A100_SESSION'" >/dev/null 2>&1
}

session_alive_a6000() {
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$A6000_REMOTE" \
    "tmux has-session -t '$A6000_SESSION'" >/dev/null 2>&1
}

summarize() {
  python - "$MANIFEST" <<'PY'
import csv
import sys
from collections import Counter
from pathlib import Path

path = Path(sys.argv[1])
rows = list(csv.DictReader(path.open(newline="", encoding="utf-8"), delimiter="\t"))
status = Counter(row["status"] for row in rows)
objects = {}
for row in rows:
    objects.setdefault(row["object_key"], Counter())[row["status"]] += 1
print("status", dict(status))
for obj in ("box021", "box004", "bucket004"):
    if obj in objects:
        print(obj, dict(objects[obj]))
for label, field in {
    "root_npz": "result_npz",
    "outdir_npz": "outdir_npz",
    "config_act": "config_act",
    "video": "video",
}.items():
    print(f"{label}={sum(1 for row in rows if Path(row[field]).is_file())}/{len(rows)}")
running = [row["case_id"] for row in rows if row["status"] == "running"]
if running:
    print("running", ",".join(running[:12]))
PY
}

echo "[$(ts)] watcher start session_tag=${SESSION_TAG} interval=${POLL_INTERVAL}" | tee -a "$LOG_FILE"

while true; do
  echo "[$(ts)] poll start" | tee -a "$LOG_FILE"
  a100_state=dead
  a6000_state=dead
  if session_alive_a100; then a100_state=alive; fi
  if session_alive_a6000; then a6000_state=alive; fi
  echo "[$(ts)] sessions a100=${a100_state} a6000=${a6000_state}" | tee -a "$LOG_FILE"

  E168_REMOTE_RETRY_MAX="$PULL_RETRY_MAX" "$PULL_SCRIPT" production "$SESSION_TAG" >> "$LOG_FILE" 2>&1
  pull_status=$?
  echo "[$(ts)] pull_status=${pull_status}" | tee -a "$LOG_FILE"
  summarize | tee -a "$LOG_FILE"

  done_count="$(python - "$MANIFEST" <<'PY'
import csv, sys
from pathlib import Path
rows = list(csv.DictReader(Path(sys.argv[1]).open(newline="", encoding="utf-8"), delimiter="\t"))
print(sum(row["status"] == "run_complete_pending_eval" for row in rows))
PY
)"
  if [ "$done_count" = "40" ]; then
    echo "[$(ts)] all rows complete; watcher exit" | tee -a "$LOG_FILE"
    exit 0
  fi

  if [ "$a100_state" = "dead" ] && [ "$a6000_state" = "dead" ]; then
    echo "[$(ts)] both remote sessions are dead before completion; watcher exit" | tee -a "$LOG_FILE"
    exit 1
  fi

  sleep "$POLL_INTERVAL"
done
