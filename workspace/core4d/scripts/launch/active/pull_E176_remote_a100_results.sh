#!/usr/bin/env bash
# Pull only artifacts registered by the active E176 A100 execution manifest.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODE="${1:-canary}"
if [ "$MODE" != "canary" ] && [ "$MODE" != "full" ]; then
  echo "usage: $0 {canary|full}" >&2
  exit 2
fi

EXPERIMENT_ID="${EXPERIMENT_ID:-E176}"
REMOTE="${REMOTE:-batchcom@61.172.170.106}"
REMOTE_SSH_PORT="${REMOTE_SSH_PORT:-30409}"
REMOTE_SSH_KEY="${REMOTE_SSH_KEY:-/home/ubuntu/.ssh/id_rsa_tianyiyun}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
RESULT_ROOT="${RESULT_ROOT:-workspace/core4d/results/${EXPERIMENT_ID}}"
EXPECTED_CANARY_ROWS="${EXPECTED_CANARY_ROWS:-6}"
EXPECTED_FULL_ROWS="${EXPECTED_FULL_ROWS:-39}"
CANARY_MANIFEST_BASENAME="${CANARY_MANIFEST_BASENAME:-lowgeom_canary_manifest.tsv}"
FULL_MANIFEST_BASENAME="${FULL_MANIFEST_BASENAME:-lowgeom_full_manifest.tsv}"
EXPECTED_ROWS="$(
  if [ "$MODE" = "canary" ]; then
    printf '%s' "$EXPECTED_CANARY_ROWS"
  else
    printf '%s' "$EXPECTED_FULL_ROWS"
  fi
)"
LATEST_JSON="${RESULT_ROOT}/s0_environment/a100_${MODE}_latest.json"
VALIDATOR="workspace/core4d/scripts/experiments/E176/validate_cem_runtime.py"
THROUGHPUT_VALIDATOR="workspace/core4d/scripts/experiments/E176/validate_canary_throughput.py"
EVALUATOR="${EVALUATOR-workspace/core4d/scripts/eval/runners/eval_E176_lowgeom.py}"
RESULTS_LINK="workspace/core4d/results"
if [ "$MODE" = "canary" ]; then
  MANIFEST_BASENAME="$CANARY_MANIFEST_BASENAME"
else
  MANIFEST_BASENAME="$FULL_MANIFEST_BASENAME"
fi
CANONICAL_MANIFEST="${RESULT_ROOT}/s6_downstream/manifests/${MANIFEST_BASENAME}"
RUNTIME_GATE="${RESULT_ROOT}/s6_downstream/cem/${MODE}/${MODE}_runtime_gate.json"
THROUGHPUT_GATE="${RESULT_ROOT}/s6_downstream/cem/canary/canary_throughput_gate.json"

[ -L "$RESULTS_LINK" ] || {
  echo "refusing pull: $RESULTS_LINK must remain a project-level symlink" >&2
  exit 3
}
[ -f "$CANONICAL_MANIFEST" ] || {
  echo "refusing pull: missing canonical manifest $CANONICAL_MANIFEST" >&2
  exit 3
}

[ -f "$LATEST_JSON" ] || {
  echo "missing $EXPERIMENT_ID execution pointer: $LATEST_JSON" >&2
  exit 3
}

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
  local max_attempts="${E176_PULL_RETRY_MAX:-6}"
  while true; do
    if "$@"; then
      return 0
    fi
    if [ "$attempt" -ge "$max_attempts" ]; then
      echo "pull command failed after ${attempt} attempts: $*" >&2
      return 1
    fi
    echo "transient pull failure; retrying attempt $((attempt + 1))/${max_attempts}" >&2
    sleep "$((attempt * 2))"
    attempt=$((attempt + 1))
  done
}

mapfile -t META < <(
  "$PYTHON_BIN" - "$LATEST_JSON" <<'PY'
import json, sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(payload["session"])
print(payload["remote_root"])
print(Path(payload["shard_root"]).parent)
print(payload["shard_root"])
print(" ".join(payload["selected_gpus"]))
PY
)
SESSION="${META[0]}"
REMOTE_RUN_ROOT="${META[1]}"
EXECUTION_ROOT="${META[2]}"
SHARD_ROOT="${META[3]}"
SELECTED_GPUS="${META[4]}"
PULL_LIST="${EXECUTION_ROOT}/pull_files.txt"
PULL_SUMMARY="${EXECUTION_ROOT}/pull_summary.json"

mkdir -p "$EXECUTION_ROOT"
retry rsync -az --partial -e "$RSYNC_RSH" \
  "${REMOTE}:${REMOTE_RUN_ROOT}/${EXECUTION_ROOT}/" "$EXECUTION_ROOT/"

MODE="$MODE" SESSION="$SESSION" SHARD_ROOT="$SHARD_ROOT" \
SELECTED_GPUS="$SELECTED_GPUS" PULL_LIST="$PULL_LIST" \
EXPECTED_ROWS="$EXPECTED_ROWS" EXPERIMENT_ID="$EXPERIMENT_ID" \
"$PYTHON_BIN" - <<'PY'
import csv
import os
from pathlib import Path

mode = os.environ["MODE"]
root = Path(os.environ["SHARD_ROOT"])
rows = []
for gpu in os.environ["SELECTED_GPUS"].split():
    shard = root / f"gpu{gpu}.tsv"
    if not shard.is_file():
        raise FileNotFoundError(shard)
    rows.extend(csv.DictReader(shard.open(encoding="utf-8"), delimiter="\t"))
expected = int(os.environ["EXPECTED_ROWS"])
cases = [row["case_id"] for row in rows]
if len(rows) != expected or len(set(cases)) != expected:
    raise ValueError(
        f"remote shard union invalid: rows={len(rows)} unique={len(set(cases))} "
        f"expected={expected}"
    )

experiment_id = os.environ["EXPERIMENT_ID"]
allowed_prefixes = (
    f"workspace/core4d/results/{experiment_id}/",
    f"logs/{experiment_id}/",
)
files = set()
for row in rows:
    for key in ("result_npz", "outdir_npz", "config_act", "log", "video"):
        path = row[key]
        parsed = Path(path)
        if (
            parsed.is_absolute()
            or ".." in parsed.parts
            or not path.startswith(allowed_prefixes)
        ):
            raise ValueError(f"out-of-scope pull path {key}={path}")
        files.add(path)
session = os.environ["SESSION"]
for gpu in os.environ["SELECTED_GPUS"].split():
    files.add(
        f"logs/{experiment_id}/cem/{mode}/{session}_gpu{gpu}.worker.log"
    )
Path(os.environ["PULL_LIST"]).write_text(
    "\n".join(sorted(files)) + "\n", encoding="utf-8"
)
print(
    f"{experiment_id} pull rows={len(rows)} "
    f"registered_files={len(files)}"
)
PY

while IFS= read -r relative_path; do
  [ -n "$relative_path" ] || continue
  if [[ "$relative_path" == *.mp4 ]] && \
     ! ssh "${SSH_OPTS[@]}" "$REMOTE" \
       "test -f '$REMOTE_RUN_ROOT/$relative_path'"; then
    echo "optional video absent, skipping: $relative_path"
    continue
  fi
  mkdir -p "$(dirname "$relative_path")"
  retry rsync -azs --partial --timeout=30 --ignore-missing-args \
    -e "$RSYNC_RSH" \
    "${REMOTE}:${REMOTE_RUN_ROOT}/${relative_path}" "$relative_path"
done < "$PULL_LIST"

[ -L "$RESULTS_LINK" ] || {
  echo "pull invariant failed: $RESULTS_LINK is no longer a symlink" >&2
  exit 3
}

MODE="$MODE" SHARD_ROOT="$SHARD_ROOT" SELECTED_GPUS="$SELECTED_GPUS" \
PULL_SUMMARY="$PULL_SUMMARY" LATEST_JSON="$LATEST_JSON" \
CANONICAL_MANIFEST="$CANONICAL_MANIFEST" \
"$PYTHON_BIN" - <<'PY'
import csv
import hashlib
import json
import os
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path

mode = os.environ["MODE"]
manifest = Path(os.environ["CANONICAL_MANIFEST"])
rows = list(csv.DictReader(manifest.open(encoding="utf-8"), delimiter="\t"))
fields = list(rows[0])
by_case = {row["case_id"]: row for row in rows}
remote_rows = []
for gpu in os.environ["SELECTED_GPUS"].split():
    shard = Path(os.environ["SHARD_ROOT"]) / f"gpu{gpu}.tsv"
    remote_rows.extend(
        csv.DictReader(shard.open(encoding="utf-8"), delimiter="\t")
    )
for row in remote_rows:
    case_id = row["case_id"]
    if case_id not in by_case:
        raise ValueError(f"remote shard has unknown case {case_id}")
    by_case[case_id].update({key: row.get(key, "") for key in fields})

fd, temporary = tempfile.mkstemp(
    prefix=f".{manifest.name}.", suffix=".tmp", dir=manifest.parent
)
with open(fd, "w", encoding="utf-8", newline="", closefd=True) as stream:
    writer = csv.DictWriter(
        stream, fieldnames=fields, delimiter="\t", lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
Path(temporary).replace(manifest)

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

artifacts = []
for row in rows:
    item = {"case_id": row["case_id"], "status": row["status"]}
    for key in ("result_npz", "outdir_npz", "config_act", "log", "video"):
        path = Path(row[key])
        item[f"{key}_exists"] = path.is_file()
        item[f"{key}_bytes"] = path.stat().st_size if path.is_file() else 0
        item[f"{key}_sha256"] = sha256(path) if path.is_file() else ""
    artifacts.append(item)
payload = {
    "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    "mode": mode,
    "execution_pointer": os.environ["LATEST_JSON"],
    "rows": len(rows),
    "status_counts": dict(Counter(row["status"] for row in rows)),
    "required_counts": {
        key: sum(item[f"{key}_exists"] for item in artifacts)
        for key in ("result_npz", "outdir_npz", "config_act", "log")
    },
    "optional_video_count": sum(item["video_exists"] for item in artifacts),
    "artifacts": artifacts,
}
Path(os.environ["PULL_SUMMARY"]).write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(payload["status_counts"], sort_keys=True))
PY

"$PYTHON_BIN" "$VALIDATOR" "$MODE" \
  --manifest "$CANONICAL_MANIFEST" \
  --output "$RUNTIME_GATE" \
  --expected-rows "$EXPECTED_ROWS" \
  --experiment-id "$EXPERIMENT_ID"
if [ "$MODE" = "canary" ]; then
  "$PYTHON_BIN" "$THROUGHPUT_VALIDATOR" \
    --manifest "$CANONICAL_MANIFEST" \
    --output "$THROUGHPUT_GATE" \
    --expected-rows "$EXPECTED_ROWS" \
    --experiment-id "$EXPERIMENT_ID"
fi
if [ -n "$EVALUATOR" ]; then
  "$PYTHON_BIN" "$EVALUATOR" "$MODE" --require-all
fi
echo "$EXPERIMENT_ID $MODE pull + runtime gate complete: $PULL_SUMMARY"
