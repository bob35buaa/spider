#!/usr/bin/env bash
# E178 local RTX 5090 hybrid worker for race-safe A100 tail offload.
#
# Usage:
#   HYBRID_PREP_ONLY=1 bash .../run_E178_local_hybrid.sh
#   bash workspace/core4d/scripts/launch/active/run_E178_local_hybrid.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
GPU_ID="${GPU_ID:-0}"
REMOTE="${REMOTE:-batchcom@61.172.170.106}"
REMOTE_SSH_PORT="${REMOTE_SSH_PORT:-30409}"
REMOTE_SSH_KEY="${REMOTE_SSH_KEY:-/home/ubuntu/.ssh/id_rsa_tianyiyun}"
HYBRID_PREP_ONLY="${HYBRID_PREP_ONLY:-0}"
LOCAL_SAFETY_FACTOR="${LOCAL_SAFETY_FACTOR:-1.2}"
MINIMUM_DEADLINE_MARGIN_S="${MINIMUM_DEADLINE_MARGIN_S:-300}"
MINIMUM_IMPROVEMENT_PCT="${MINIMUM_IMPROVEMENT_PCT:-5}"
LATEST_EXECUTION_JSON="workspace/core4d/results/E178/s0_environment/a100_full_latest.json"
FULL_MANIFEST="workspace/core4d/results/E178/s6_downstream/manifests/semantic_bucket_full_manifest.tsv"
RUNNER="workspace/core4d/scripts/experiments/E176/run_cem_queue.py"
VALIDATOR="workspace/core4d/scripts/experiments/E176/validate_cem_runtime.py"
BUILDER="workspace/core4d/scripts/experiments/E178/build_hybrid_rebalance.py"

if [ "$HYBRID_PREP_ONLY" != "0" ] && [ "$HYBRID_PREP_ONLY" != "1" ]; then
  echo "HYBRID_PREP_ONLY must be 0 or 1." >&2
  exit 2
fi
[ -x "$PYTHON_BIN" ] || {
  echo "missing project python: $PYTHON_BIN" >&2
  exit 2
}
[ -f "$LATEST_EXECUTION_JSON" ] || {
  echo "missing active E178 Full execution: $LATEST_EXECUTION_JSON" >&2
  exit 2
}

readarray -t EXECUTION_VALUES < <(
  "$PYTHON_BIN" - "$LATEST_EXECUTION_JSON" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
if payload.get("experiment_id") != "E178" or payload.get("mode") != "full":
    raise SystemExit("latest execution is not E178 Full")
print(payload["session"])
print(payload["remote_root"])
print(payload["remote_python"])
print(payload["shard_root"])
PY
)
REMOTE_SESSION="${EXECUTION_VALUES[0]}"
REMOTE_RUN_ROOT="${EXECUTION_VALUES[1]}"
REMOTE_PYTHON_BIN="${EXECUTION_VALUES[2]}"
REMOTE_QUEUE_REL="${EXECUTION_VALUES[3]}"

SSH_OPTS=(
  -o BatchMode=yes
  -o ConnectTimeout=20
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=4
  -p "$REMOTE_SSH_PORT"
  -i "$REMOTE_SSH_KEY"
)
RSYNC_RSH="$(
  printf 'ssh -o BatchMode=yes -o ConnectTimeout=20 '
  printf -- '-o ServerAliveInterval=30 -o ServerAliveCountMax=4 '
  printf -- '-p %s -i %s' "$REMOTE_SSH_PORT" "$REMOTE_SSH_KEY"
)"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
HYBRID_ID="hybrid_local_${RUN_TAG}"
HYBRID_ROOT="workspace/core4d/results/E178/s0_environment/${HYBRID_ID}"
SNAPSHOT_ROOT="${HYBRID_ROOT}/remote_snapshot"
ALLOCATION_ROOT="${HYBRID_ROOT}/allocation"
ROW_STDOUT_ROOT="${HYBRID_ROOT}/row_stdout"
STAGING_REL="${HYBRID_ROOT}/remote_staging"
EVENT_LOG="${HYBRID_ROOT}/hybrid_events.log"
mkdir -p \
  "$SNAPSHOT_ROOT/queues" \
  "$SNAPSHOT_ROOT/case_logs" \
  "$ALLOCATION_ROOT" \
  "$ROW_STDOUT_ROOT"

ssh "${SSH_OPTS[@]}" "$REMOTE" \
  "tmux has-session -t '$REMOTE_SESSION' 2>/dev/null" || {
  echo "active E178 Full tmux is not running: $REMOTE_SESSION" >&2
  exit 3
}

rsync -az -e "$RSYNC_RSH" \
  "${REMOTE}:${REMOTE_RUN_ROOT}/${REMOTE_QUEUE_REL}/" \
  "${SNAPSHOT_ROOT}/queues/"
rsync -az -e "$RSYNC_RSH" \
  "${REMOTE}:${REMOTE_RUN_ROOT}/logs/E178/cem/full/" \
  "${SNAPSHOT_ROOT}/case_logs/"

nvidia-smi \
  --query-gpu=index,name,uuid,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw \
  --format=csv,noheader,nounits > "${HYBRID_ROOT}/local_gpu_before.csv"
nvidia-smi \
  --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory \
  --format=csv,noheader,nounits > "${HYBRID_ROOT}/local_compute_before.csv"

DENSITY_SUMMARY="$(
  find workspace/core4d/results/E178/s6_downstream/benchmark \
    -mindepth 2 -maxdepth 2 -type f \
    -path '*/local_speed_probe_density_*/probe_summary.json' |
    sort |
    tail -1
)"
[ -n "$DENSITY_SUMMARY" ] && [ -f "$DENSITY_SUMMARY" ] || {
  echo "missing successful local density probe summary" >&2
  exit 3
}
LOCAL_FIVE_RATE="$(
  "$PYTHON_BIN" - "$DENSITY_SUMMARY" <<'PY'
import json
import math
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
if payload.get("status") != "pass":
    raise SystemExit("latest density probe did not pass")
median = payload["timing"]["rows"][0]["median_plan_time_s"]
value = float(median) * 16.0
if not math.isfinite(value) or value <= 0:
    raise SystemExit("invalid density-derived Full rate")
print(value)
PY
)"

"$PYTHON_BIN" "$BUILDER" \
  --full-manifest "$FULL_MANIFEST" \
  --queue-dir "${SNAPSHOT_ROOT}/queues" \
  --case-log-dir "${SNAPSHOT_ROOT}/case_logs" \
  --output-root "$ALLOCATION_ROOT" \
  --local-five-geom-seconds-per-record "$LOCAL_FIVE_RATE" \
  --local-safety-factor "$LOCAL_SAFETY_FACTOR" \
  --minimum-deadline-margin-s "$MINIMUM_DEADLINE_MARGIN_S" \
  --minimum-improvement-pct "$MINIMUM_IMPROVEMENT_PCT"

HYBRID_ID="$HYBRID_ID" HYBRID_ROOT="$HYBRID_ROOT" \
REMOTE="$REMOTE" REMOTE_SESSION="$REMOTE_SESSION" \
REMOTE_RUN_ROOT="$REMOTE_RUN_ROOT" REMOTE_QUEUE_REL="$REMOTE_QUEUE_REL" \
DENSITY_SUMMARY="$DENSITY_SUMMARY" LOCAL_FIVE_RATE="$LOCAL_FIVE_RATE" \
ALLOCATION_ROOT="$ALLOCATION_ROOT" HYBRID_PREP_ONLY="$HYBRID_PREP_ONLY" \
"$PYTHON_BIN" - <<'PY'
import hashlib
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

allocation = Path(os.environ["ALLOCATION_ROOT"]) / "allocation_summary.json"
payload = {
    "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    "status": (
        "preflight_passed_no_launch"
        if os.environ["HYBRID_PREP_ONLY"] == "1"
        else "ready_to_launch"
    ),
    "experiment_id": "E178",
    "hybrid_id": os.environ["HYBRID_ID"],
    "profile": "local-RTX5090+A100-8gpu",
    "local_gpu_id": "0",
    "remote": os.environ["REMOTE"],
    "remote_session": os.environ["REMOTE_SESSION"],
    "remote_root": os.environ["REMOTE_RUN_ROOT"],
    "remote_queue_rel": os.environ["REMOTE_QUEUE_REL"],
    "density_probe_summary": os.environ["DENSITY_SUMMARY"],
    "local_five_geom_full_seconds_per_record": float(
        os.environ["LOCAL_FIVE_RATE"]
    ),
    "allocation_summary": str(allocation),
    "allocation_summary_sha256": hashlib.sha256(
        allocation.read_bytes()
    ).hexdigest(),
    "local_git_head": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip(),
}
root = Path(os.environ["HYBRID_ROOT"])
(root / "execution_manifest.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY

if [ "$HYBRID_PREP_ONLY" = "1" ]; then
  echo "E178 hybrid preflight passed; no local production CEM launched."
  echo "Allocation: ${ALLOCATION_ROOT}/allocation_summary.json"
  exit 0
fi

remote_row_status() {
  local case_id="$1"
  ssh "${SSH_OPTS[@]}" "$REMOTE" \
    python3 - "${REMOTE_RUN_ROOT}/${REMOTE_QUEUE_REL}" "$case_id" <<'PY'
import csv
import sys
from pathlib import Path

root = Path(sys.argv[1])
case_id = sys.argv[2]
matches = []
for path in root.glob("gpu*.tsv"):
    for position, row in enumerate(
        csv.DictReader(path.open(encoding="utf-8"), delimiter="\t"),
        start=1,
    ):
        if row["case_id"] == case_id:
            matches.append((row["status"], path.name, position))
if len(matches) != 1:
    raise SystemExit(f"expected one remote row for {case_id}, got {len(matches)}")
print("\t".join(map(str, matches[0])))
PY
}

promote_staged_artifacts() {
  local case_id="$1"
  local sync_list="$2"
  local expected_shard="$3"
  ssh "${SSH_OPTS[@]}" "$REMOTE" \
    python3 - \
      "$REMOTE_RUN_ROOT" \
      "$REMOTE_QUEUE_REL" \
      "$case_id" \
      "$STAGING_REL/$case_id" \
      "$sync_list" \
      "$expected_shard" <<'PY'
import csv
import hashlib
import os
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
queue_rel = Path(sys.argv[2])
case_id = sys.argv[3]
stage_rel = Path(sys.argv[4])
sync_list_rel = Path(sys.argv[5])
expected_shard = sys.argv[6]

matches = []
for path in (run_root / queue_rel).glob("gpu*.tsv"):
    for row in csv.DictReader(path.open(encoding="utf-8"), delimiter="\t"):
        if row["case_id"] == case_id:
            matches.append((row["status"], path.name))
if matches != [("not_run", expected_shard)]:
    raise SystemExit(
        f"promotion blocked by live remote state for {case_id}: {matches}"
    )

stage_root = run_root / stage_rel
sync_list = stage_root / sync_list_rel
paths = [
    Path(value)
    for value in sync_list.read_text(encoding="utf-8").splitlines()
    if value
]
for rel in paths:
    source = stage_root / rel
    if not source.is_file():
        raise FileNotFoundError(source)

# Make the three required artifacts appear in a tight window.  Config is moved
# last because output_complete() requires it before a remote row can skip.
def priority(path: Path) -> tuple[int, str]:
    if path.name == "config_act.yaml":
        return (2, str(path))
    if "probe_manifest" in path.name or "row_manifests" in str(path):
        return (3, str(path))
    return (1, str(path))

for rel in sorted(paths, key=priority):
    source = stage_root / rel
    destination = run_root / rel
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, destination)
print(f"promoted {case_id} files={len(paths)} shard={expected_shard}")
PY
}

for row_manifest in "${ALLOCATION_ROOT}"/row_manifests/*.tsv; do
  readarray -t ROW_VALUES < <(
    "$PYTHON_BIN" - "$row_manifest" <<'PY'
import csv
import sys

row = next(csv.DictReader(open(sys.argv[1], encoding="utf-8"), delimiter="\t"))
print(row["case_id"])
print(row["result_npz"])
print(row["outdir_npz"])
print(row["config_act"])
print(row["log"])
PY
  )
  CASE_ID="${ROW_VALUES[0]}"
  RESULT_NPZ="${ROW_VALUES[1]}"
  OUTDIR_NPZ="${ROW_VALUES[2]}"
  CONFIG_ACT="${ROW_VALUES[3]}"
  CASE_LOG="${ROW_VALUES[4]}"

  REMOTE_STATE="$(remote_row_status "$CASE_ID")"
  IFS=$'\t' read -r REMOTE_STATUS REMOTE_SHARD REMOTE_POSITION \
    <<< "$REMOTE_STATE"
  printf '[%s] pre-run case=%s remote=%s/%s position=%s\n' \
    "$(date -Is)" "$CASE_ID" "$REMOTE_SHARD" "$REMOTE_STATUS" \
    "$REMOTE_POSITION" | tee -a "$EVENT_LOG"
  if [ "$REMOTE_STATUS" != "not_run" ]; then
    echo "Hybrid race gate blocked before local run: $CASE_ID=$REMOTE_STATUS" \
      | tee -a "$EVENT_LOG" >&2
    exit 5
  fi

  MUJOCO_GL="${MUJOCO_GL:-egl}" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    "$PYTHON_BIN" "$RUNNER" \
      --mode full \
      --manifest-tsv "$row_manifest" \
      --python-bin "$PYTHON_BIN" \
      --gpu-id "$GPU_ID" \
      --all \
      --num-samples 1024 \
      --max-num-iterations 32 \
      > "${ROW_STDOUT_ROOT}/${CASE_ID}.runner.log" 2>&1

  "$PYTHON_BIN" "$VALIDATOR" full \
    --manifest "$row_manifest" \
    --output "${ROW_STDOUT_ROOT}/${CASE_ID}.runtime_gate.json" \
    --expected-rows 1 \
    --experiment-id E178-local-hybrid

  REMOTE_STATE="$(remote_row_status "$CASE_ID")"
  IFS=$'\t' read -r REMOTE_STATUS_AFTER REMOTE_SHARD_AFTER REMOTE_POSITION_AFTER \
    <<< "$REMOTE_STATE"
  if [ "$REMOTE_STATUS_AFTER" != "not_run" ] || \
     [ "$REMOTE_SHARD_AFTER" != "$REMOTE_SHARD" ]; then
    echo "Hybrid race gate blocked promotion: $CASE_ID=$REMOTE_STATE" \
      | tee -a "$EVENT_LOG" >&2
    exit 6
  fi

  SYNC_LIST_LOCAL="${ROW_STDOUT_ROOT}/${CASE_ID}.sync_files.txt"
  SYNC_LIST_REL=".sync_files.txt"
  printf '%s\n' \
    "$RESULT_NPZ" \
    "$OUTDIR_NPZ" \
    "$CONFIG_ACT" \
    "$CASE_LOG" \
    "$row_manifest" > "$SYNC_LIST_LOCAL"

  REMOTE_STAGE_CASE="${REMOTE_RUN_ROOT}/${STAGING_REL}/${CASE_ID}"
  ssh "${SSH_OPTS[@]}" "$REMOTE" "mkdir -p '$REMOTE_STAGE_CASE'"
  rsync -azR --files-from="$SYNC_LIST_LOCAL" -e "$RSYNC_RSH" \
    ./ "${REMOTE}:${REMOTE_STAGE_CASE}/"
  rsync -az -e "$RSYNC_RSH" \
    "$SYNC_LIST_LOCAL" \
    "${REMOTE}:${REMOTE_STAGE_CASE}/${SYNC_LIST_REL}"

  promote_staged_artifacts \
    "$CASE_ID" \
    "$SYNC_LIST_REL" \
    "$REMOTE_SHARD"

  REMOTE_GPU="${REMOTE_SHARD#gpu}"
  REMOTE_GPU="${REMOTE_GPU%.tsv}"
  ssh "${SSH_OPTS[@]}" "$REMOTE" \
    "cd '$REMOTE_RUN_ROOT' && \
     PYTHONPATH='$REMOTE_RUN_ROOT' MUJOCO_GL=egl \
     '$REMOTE_PYTHON_BIN' '$RUNNER' \
       --mode full \
       --manifest-tsv '$row_manifest' \
       --python-bin '$REMOTE_PYTHON_BIN' \
       --gpu-id '$REMOTE_GPU' \
       --all \
       --num-samples 1024 \
       --max-num-iterations 32" \
    >> "$EVENT_LOG" 2>&1

  printf '[%s] promoted-and-validated case=%s remote_shard=%s\n' \
    "$(date -Is)" "$CASE_ID" "$REMOTE_SHARD" | tee -a "$EVENT_LOG"
done

nvidia-smi \
  --query-gpu=index,name,uuid,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw \
  --format=csv,noheader,nounits > "${HYBRID_ROOT}/local_gpu_after.csv"
echo "E178 local hybrid rows complete: $HYBRID_ID"
