#!/usr/bin/env bash
# Pull E168 A100 CEM artifacts/logs/manifests back to local workspace.
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

if [ "$MODE" = "canary" ]; then
  CEM_SUBDIR="canary"
  EXPECTED=4
else
  CEM_SUBDIR="full"
  EXPECTED=40
fi

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

pull_remote_dir() {
  local rel_path="$1"
  local local_path="$2"
  mkdir -p "$local_path"
  if retry ssh_remote "test -d '${REMOTE_ROOT}/${rel_path}'"; then
    retry rsync -az -e "$RSYNC_RSH" "${REMOTE}:${REMOTE_ROOT}/${rel_path}/" "$local_path/"
  else
    echo "skip missing remote dir: ${REMOTE}:${REMOTE_ROOT}/${rel_path}"
  fi
}

pull_remote_dir "workspace/core4d/results/E168/s6_downstream/cem/${CEM_SUBDIR}" \
  "workspace/core4d/results/E168/s6_downstream/cem/${CEM_SUBDIR}"
pull_remote_dir "workspace/core4d/results/E168/s6_downstream/cem/manifests" \
  "workspace/core4d/results/E168/s6_downstream/cem/manifests"
pull_remote_dir "workspace/core4d/results/E168/s0_environment" \
  "workspace/core4d/results/E168/s0_environment"
pull_remote_dir "logs/E168/cem/${CEM_SUBDIR}" "logs/E168/cem/${CEM_SUBDIR}"

python - "$MODE" <<'PY'
import csv
import sys
from pathlib import Path

mode = sys.argv[1]
manifest_dir = Path("workspace/core4d/results/E168/s6_downstream/cem/manifests")
main_name = "cem_canary_manifest.tsv" if mode == "canary" else "cem_production_manifest.tsv"
main_path = manifest_dir / main_name
shard_dirs = sorted(
    [p for p in manifest_dir.glob(f"a100_{mode}_*") if p.is_dir()],
    key=lambda p: p.stat().st_mtime,
)
if not shard_dirs or not main_path.is_file():
    raise SystemExit(0)

latest = shard_dirs[-1]
main_rows = list(csv.DictReader(main_path.open(newline="", encoding="utf-8"), delimiter="\t"))
fields = list(main_rows[0].keys()) if main_rows else []
by_variant = {row["variant"]: row for row in main_rows}
for shard in sorted(latest.glob("gpu*.tsv")):
    for row in csv.DictReader(shard.open(newline="", encoding="utf-8"), delimiter="\t"):
        if row["variant"] not in by_variant:
            continue
        target = by_variant[row["variant"]]
        for field in fields:
            if field in row:
                target[field] = row[field]

with main_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(main_rows)
print(f"merged latest A100 shard statuses from {latest}")
PY

echo "Pulled E168 A100 ${MODE} artifacts."
python - "$MODE" "$EXPECTED" <<'PY'
import csv
import sys
from pathlib import Path

mode = sys.argv[1]
expected = int(sys.argv[2])
manifest = Path("workspace/core4d/results/E168/s6_downstream/cem/manifests") / (
    "cem_canary_manifest.tsv" if mode == "canary" else "cem_production_manifest.tsv"
)
rows = list(csv.DictReader(manifest.open(newline="", encoding="utf-8"), delimiter="\t"))
checks = {
    "root_npz_count": "result_npz",
    "outdir_npz_count": "outdir_npz",
    "config_act_count": "config_act",
    "video_count": "video",
}
for label, field in checks.items():
    count = sum(1 for row in rows if Path(row[field]).is_file())
    print(f"{label}={count}/{expected}")
PY
