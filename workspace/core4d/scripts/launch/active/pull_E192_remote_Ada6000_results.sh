#!/usr/bin/env bash
# Pull only E192 artifacts from the recorded Ada6000 execution manifest.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
MODE="${MODE:-${1:-canary}}"; PY="${PYTHON_BIN:-.venv/bin/python}"
ROOT="workspace/core4d/results/E192"
REMOTE="${E192_ADA_REMOTE:-spider-remote}"
POINTER="$ROOT/remote_staging/latest_${MODE}_ada6000.json"
[ -f "$POINTER" ] || { echo "missing pointer: $POINTER" >&2; exit 2; }
mapfile -t META < <("$PY" - "$POINTER" <<'PY'
import json, sys
x=json.load(open(sys.argv[1])); print(x["remote_root"]); print(x["stage"])
PY
)
REMOTE_ROOT="${META[0]}"; STAGE="${META[1]}"
RSH="ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=4"
mkdir -p "$STAGE"
rsync -az -e "$RSH" "$REMOTE:$REMOTE_ROOT/$STAGE/" "$STAGE/"
mapfile -t PATHS < <("$PY" - "$STAGE" <<'PY'
import csv, sys
from pathlib import Path
files=set()
for p in Path(sys.argv[1]).glob("ada-gpu*.tsv"):
  for r in csv.DictReader(p.open(), delimiter="\t"):
    files.update(r[k] for k in ("result_npz","outdir_npz","config_act","video","log"))
print("\n".join(sorted(files)))
PY
)
for path in "${PATHS[@]}"; do
  if ssh -o BatchMode=yes -o ConnectTimeout=20 "$REMOTE" "test -f '$REMOTE_ROOT/$path'"; then
    mkdir -p "$(dirname "$path")"
    rsync -az -e "$RSH" "$REMOTE:$REMOTE_ROOT/$path" "$path"
  else
    echo "E192 Ada pull skip missing artifact: $path"
  fi
done
"$PY" - "$MODE" "$STAGE" <<'PY'
import csv, sys
from pathlib import Path
mode, stage = sys.argv[1], Path(sys.argv[2])
root=Path("workspace/core4d/results/E192/s6_downstream/manifests")
path=root / {"canary":"cem_canary_manifest.tsv","full":"cem_full_manifest.tsv"}[mode]
rows=list(csv.DictReader(path.open(),delimiter="\t")); fields=list(rows[0]); by={r["variant"]:r for r in rows}
for shard in stage.glob("ada-gpu*.tsv"):
  for row in csv.DictReader(shard.open(),delimiter="\t"):
    if row["variant"] in by: by[row["variant"]].update(row)
with path.open("w",newline="") as stream:
  w=csv.DictWriter(stream,fieldnames=fields,delimiter="\t",lineterminator="\n"); w.writeheader(); w.writerows(rows)
print(f"merged Ada6000 state into {path}")
PY
echo "E192 Ada6000 pull complete: mode=$MODE"
