#!/usr/bin/env bash
# Pull only artifacts registered by the active E175 A100 execution manifest.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODE="${1:-canary}"
if [ "$MODE" != "canary" ] && [ "$MODE" != "full" ]; then
  echo "usage: $0 {canary|full}" >&2
  exit 2
fi

REMOTE="${REMOTE:-batchcom@61.172.170.106}"
REMOTE_SSH_PORT="${REMOTE_SSH_PORT:-30409}"
REMOTE_SSH_KEY="${REMOTE_SSH_KEY:-/home/ubuntu/.ssh/id_rsa_tianyiyun}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
RESULT_ROOT="workspace/core4d/results/E175"
LATEST_JSON="${RESULT_ROOT}/s0_environment/a100_${MODE}_latest.json"
VALIDATOR="workspace/core4d/scripts/experiments/E175/validate_cem_runtime.py"
EVALUATOR="workspace/core4d/scripts/eval/runners/eval_E175_nonbox_multigeom.py"
RESULTS_LINK="workspace/core4d/results"
CANONICAL_MANIFEST="${RESULT_ROOT}/s6_downstream/manifests/nonbox_multigeom_${MODE}_manifest.tsv"

[ -L "$RESULTS_LINK" ] || {
  echo "refusing pull: $RESULTS_LINK must remain a project-level symlink" >&2
  exit 3
}
[ -f "$CANONICAL_MANIFEST" ] || {
  echo "refusing pull: missing canonical manifest $CANONICAL_MANIFEST" >&2
  exit 3
}

[ -f "$LATEST_JSON" ] || {
  echo "missing E175 execution pointer: $LATEST_JSON" >&2
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
  local max_attempts="${E175_PULL_RETRY_MAX:-6}"
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
expected = 6 if mode == "canary" else 39
cases = [row["case_id"] for row in rows]
if len(rows) != expected or len(set(cases)) != expected:
    raise ValueError(
        f"remote shard union invalid: rows={len(rows)} unique={len(set(cases))} "
        f"expected={expected}"
    )

allowed_prefixes = ("workspace/core4d/results/E175/", "logs/E175/")
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
    files.add(f"logs/E175/cem/{mode}/{session}_gpu{gpu}.worker.log")
Path(os.environ["PULL_LIST"]).write_text(
    "\n".join(sorted(files)) + "\n", encoding="utf-8"
)
print(f"E175 pull rows={len(rows)} registered_files={len(files)}")
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
  retry rsync -azs --partial --append-verify --ignore-missing-args \
    -e "$RSYNC_RSH" \
    "${REMOTE}:${REMOTE_RUN_ROOT}/${relative_path}" "$relative_path"
done < "$PULL_LIST"

[ -L "$RESULTS_LINK" ] || {
  echo "pull invariant failed: $RESULTS_LINK is no longer a symlink" >&2
  exit 3
}

MODE="$MODE" SHARD_ROOT="$SHARD_ROOT" SELECTED_GPUS="$SELECTED_GPUS" \
PULL_SUMMARY="$PULL_SUMMARY" LATEST_JSON="$LATEST_JSON" \
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
manifest = Path(
    "workspace/core4d/results/E175/s6_downstream/manifests/"
    + (
        "nonbox_multigeom_canary_manifest.tsv"
        if mode == "canary"
        else "nonbox_multigeom_full_manifest.tsv"
    )
)
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

"$PYTHON_BIN" "$VALIDATOR" "$MODE"
"$PYTHON_BIN" "$EVALUATOR" "$MODE" --require-all
echo "E175 $MODE pull + runtime gate complete: $PULL_SUMMARY"
