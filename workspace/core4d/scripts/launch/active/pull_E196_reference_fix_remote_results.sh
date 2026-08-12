#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
REQUESTED="${1:-full}"
REMOTE="${E196_ADA_REMOTE:-spider-remote}"
SSH_CONFIG_FILE="${E196_ADA_SSH_CONFIG:-}"
SSH_PORT="${E196_ADA_PORT:-}"
SSH_BIND_INTERFACE="${E196_ADA_BIND_INTERFACE:-}"
ROOT="workspace/core4d/results/E196"
SSH_BASE=(ssh)
[ -n "$SSH_CONFIG_FILE" ] && SSH_BASE+=(-F "$SSH_CONFIG_FILE")
[ -n "$SSH_PORT" ] && SSH_BASE+=(-p "$SSH_PORT")
[ -n "$SSH_BIND_INTERFACE" ] && SSH_BASE+=(-o "BindInterface=$SSH_BIND_INTERFACE")
SSH_BASE+=(-o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=4)
SSH=("${SSH_BASE[@]}" "$REMOTE")
printf -v RSH '%q ' "${SSH_BASE[@]}"
RSH="${RSH% }"

run_ssh() {
  local attempt
  for attempt in 1 2 3 4 5; do
    if "${SSH[@]}" "$@"; then return 0; fi
    echo "E196 pull SSH attempt $attempt/5 failed" >&2
    [ "$attempt" -lt 5 ] && sleep 3
  done
  return 255
}

run_rsync() {
  local attempt
  for attempt in 1 2 3 4 5; do
    if rsync "$@"; then return 0; fi
    echo "E196 pull rsync attempt $attempt/5 failed" >&2
    [ "$attempt" -lt 5 ] && sleep 3
  done
  return 255
}
case "$REQUESTED" in
  wave0) MODES=(wave0) ;;
  remaining) MODES=(remaining) ;;
  full) MODES=(wave0 remaining) ;;
  *) echo "unsupported pull scope: $REQUESTED" >&2; exit 2 ;;
esac

for MODE in "${MODES[@]}"; do
  POINTER="$ROOT/s6_downstream/execution/latest_${MODE}_ada6000.json"
  [ -f "$POINTER" ] || { echo "missing launch pointer: $POINTER" >&2; exit 3; }
  mapfile -t META < <("$PYTHON_BIN" - "$POINTER" <<'PY'
import json,sys
data=json.load(open(sys.argv[1])); print(data["remote_run_root"]); print(data["stage"])
PY
  )
  REMOTE_RUN_ROOT="${META[0]}"; STAGE="${META[1]}"
  mkdir -p "$STAGE"
  run_rsync -az -e "$RSH" "$REMOTE:$REMOTE_RUN_ROOT/$STAGE/" "$STAGE/"
  mapfile -t PATHS < <("$PYTHON_BIN" - "$STAGE" <<'PY'
import csv,sys
from pathlib import Path
files=set()
for path in Path(sys.argv[1]).glob("ada-gpu*.tsv"):
    for row in csv.DictReader(path.open(),delimiter="\t"):
        files.update(row[key] for key in ("result_npz","outdir_npz","config_act","log"))
print("\n".join(sorted(files)))
PY
  )
  FILE_LIST="$(mktemp)"
  trap 'rm -f "$FILE_LIST"' EXIT
  printf '%s\n' "${PATHS[@]}" > "$FILE_LIST"
  run_rsync -az --partial --keep-dirlinks --files-from="$FILE_LIST" -e "$RSH" \
    "$REMOTE:$REMOTE_RUN_ROOT/" ./
  "$PYTHON_BIN" - "$FILE_LIST" <<'PY'
import sys
from pathlib import Path
paths=[Path(line.strip()) for line in Path(sys.argv[1]).read_text().splitlines() if line.strip()]
missing=[str(path) for path in paths if not path.is_file()]
empty=[str(path) for path in paths if path.is_file() and path.stat().st_size == 0]
if missing or empty:
    raise SystemExit(f"pull artifact audit failed: missing={missing} empty={empty}")
print(f"pulled artifact paths={len(paths)} missing=0 empty=0")
PY
  rm -f "$FILE_LIST"
  trap - EXIT
  CANONICAL="$ROOT/s6_downstream/manifests/reference_fix_full_manifest.tsv"
  WAVE="$ROOT/s6_downstream/manifests/reference_fix_${MODE}_manifest.tsv"
  LOCAL="$ROOT/s6_downstream/execution/local/$MODE/manifest.tsv"
  "$PYTHON_BIN" - "$CANONICAL" "$WAVE" "$LOCAL" "$STAGE" <<'PY'
import csv,sys
from collections import Counter
from pathlib import Path
canonical,wave,local,stage=map(Path,sys.argv[1:])
def read(path):
    with path.open() as stream: return list(csv.DictReader(stream,delimiter="\t"))
rows=read(canonical); fields=list(rows[0]); by={row["case_id"]:row for row in rows}
shards=list(stage.glob("ada-gpu*.tsv")); shards += [local] if local.is_file() else []
for shard in shards:
    for row in read(shard):
        if row["case_id"] in by: by[row["case_id"]].update(row)
for target,target_rows in ((canonical,rows),(wave,[row for row in rows if row["wave"]==wave.stem.removeprefix("reference_fix_").removesuffix("_manifest")])):
    with target.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,delimiter="\t",lineterminator="\n",extrasaction="ignore")
        writer.writeheader(); writer.writerows(target_rows)
print(dict(Counter(row["status"] for row in rows)))
PY
done
echo "E196 pull complete: $REQUESTED"
