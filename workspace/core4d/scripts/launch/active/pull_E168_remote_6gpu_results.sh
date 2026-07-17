#!/usr/bin/env bash
# Pull E168 6-GPU full CEM outputs and merge shard statuses.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODE="${1:-production}"
SESSION_TAG="${2:-}"
if [ "$MODE" != "production" ]; then
  echo "usage: $0 production [session_tag]" >&2
  exit 2
fi

A100_REMOTE="${A100_REMOTE:-batchcom@61.172.170.106}"
A100_ROOT="${A100_ROOT:-/home/dataset-assist-0/xiayb/workspace/spider}"
A100_SSH_PORT="${A100_SSH_PORT:-30409}"
A100_SSH_KEY="${A100_SSH_KEY:-$HOME/.ssh/id_rsa_tianyiyun}"
A6000_REMOTE="${A6000_REMOTE:-spider-remote}"
A6000_ROOT="${A6000_ROOT:-/home/xiayb/pHRI_workspace/spider}"

MANIFEST_DIR="workspace/core4d/results/E168/s6_downstream/cem/manifests"
if [ -z "$SESSION_TAG" ]; then
  SHARD_ROOT="$(find "$MANIFEST_DIR" -maxdepth 1 -type d -name 'remote6_production_*' -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)"
else
  SHARD_ROOT="${MANIFEST_DIR}/remote6_production_${SESSION_TAG}"
fi
[ -n "$SHARD_ROOT" ] && [ -d "$SHARD_ROOT" ] || { echo "missing shard root: ${SHARD_ROOT:-<none>}" >&2; exit 1; }

retry() {
  local attempt=1
  local max_attempts="${E168_REMOTE_RETRY_MAX:-4}"
  while true; do
    if "$@"; then return 0; fi
    if [ "$attempt" -ge "$max_attempts" ]; then
      echo "Command failed after ${attempt} attempts: $*" >&2
      return 1
    fi
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

mkdir -p workspace/core4d/results/E168/s6_downstream/cem/full logs/E168/cem/full "$SHARD_ROOT"

retry rsync_a100 "${A100_REMOTE}:${A100_ROOT}/workspace/core4d/results/E168/s6_downstream/cem/full/" \
  "workspace/core4d/results/E168/s6_downstream/cem/full/"
retry rsync_a100 "${A100_REMOTE}:${A100_ROOT}/logs/E168/cem/full/" "logs/E168/cem/full/"
retry rsync_a100 "${A100_REMOTE}:${A100_ROOT}/${SHARD_ROOT}/a100_gpu*.tsv" "$SHARD_ROOT/" || true

retry rsync_a6000 "${A6000_REMOTE}:${A6000_ROOT}/workspace/core4d/results/E168/s6_downstream/cem/full/" \
  "workspace/core4d/results/E168/s6_downstream/cem/full/"
retry rsync_a6000 "${A6000_REMOTE}:${A6000_ROOT}/logs/E168/cem/full/" "logs/E168/cem/full/"
retry rsync_a6000 "${A6000_REMOTE}:${A6000_ROOT}/${SHARD_ROOT}/a6000_gpu*.tsv" "$SHARD_ROOT/" || true

python - "$SHARD_ROOT" <<'PY'
import csv
import sys
from pathlib import Path

shard_root = Path(sys.argv[1])
manifest = Path("workspace/core4d/results/E168/s6_downstream/cem/manifests/cem_production_manifest.tsv")
rows = list(csv.DictReader(manifest.open(newline="", encoding="utf-8"), delimiter="\t"))
fields = list(rows[0].keys()) if rows else []
by_variant = {row["variant"]: row for row in rows}
for shard in sorted(shard_root.glob("*_gpu*.tsv")):
    for row in csv.DictReader(shard.open(newline="", encoding="utf-8"), delimiter="\t"):
        target = by_variant.get(row["variant"])
        if not target:
            continue
        for field in fields:
            if field in row:
                target[field] = row[field]
with manifest.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
print(f"merged remote6 shard statuses from {shard_root}")
PY

python - <<'PY'
import csv
from collections import Counter
from pathlib import Path

manifest = Path("workspace/core4d/results/E168/s6_downstream/cem/manifests/cem_production_manifest.tsv")
rows = list(csv.DictReader(manifest.open(newline="", encoding="utf-8"), delimiter="\t"))
print("status", dict(Counter(row["status"] for row in rows)))
for label, field in {
    "root_npz": "result_npz",
    "outdir_npz": "outdir_npz",
    "config_act": "config_act",
    "video": "video",
}.items():
    print(f"{label}={sum(1 for row in rows if Path(row[field]).is_file())}/{len(rows)}")
PY
