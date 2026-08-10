#!/usr/bin/env bash
# Pull E192 remote artifacts, then merge remote and local worker status.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
MODE="${MODE:-${1:-full}}"
PY="${PYTHON_BIN:-.venv/bin/python}"
ROOT="workspace/core4d/results/E192"
REMOTE="${E192_A100_REMOTE:-batchcom@61.172.170.106}"
PORT="${E192_A100_PORT:-30409}"
KEY="${E192_A100_KEY:-/home/ubuntu/.ssh/id_rsa_tianyiyun}"
POINTER="workspace/core4d/results/E192/remote_staging/latest_${MODE}_a100.json"
[ -f "$POINTER" ] || { echo "missing pointer: $POINTER" >&2; exit 2; }
mapfile -t META < <("$PY" - "$POINTER" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]));print(x['remote_root']);print(x['stage'])
PY
)
REMOTE_ROOT="${META[0]}"; STAGE="${META[1]}"
RSH="ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=4 -p $PORT -i $KEY"
mkdir -p "$STAGE"
rsync -az -e "$RSH" "$REMOTE:$REMOTE_ROOT/$STAGE/" "$STAGE/"
mapfile -t PATHS < <("$PY" - "$STAGE" <<'PY'
import csv,sys
from pathlib import Path
files=set()
for p in Path(sys.argv[1]).glob('a100-gpu*.tsv'):
 for r in csv.DictReader(p.open(),delimiter='\t'):
  for k in ('result_npz','outdir_npz','config_act','video','log'): files.add(r[k])
print('\n'.join(sorted(files)))
PY
)
for path in "${PATHS[@]}"; do
  if ssh -o BatchMode=yes -o ConnectTimeout=20 -p "$PORT" -i "$KEY" "$REMOTE" \
      "test -f '$REMOTE_ROOT/$path'"; then
    mkdir -p "$(dirname "$path")"
    rsync -azs -e "$RSH" "$REMOTE:$REMOTE_ROOT/$path" "$path"
  else
    echo "E192 pull skip missing remote artifact: $path"
  fi
done
LOCAL_SHARD="$ROOT/remote_staging/local-gpu0/$MODE/manifest.tsv"
"$PY" - "$MODE" "$STAGE" "$LOCAL_SHARD" <<'PY'
import csv,sys
from pathlib import Path
mode,stage,local_shard=sys.argv[1],Path(sys.argv[2]),Path(sys.argv[3])
root=Path('workspace/core4d/results/E192/s6_downstream/manifests')
name={'baseline_sentinel':'cem_baseline_sentinel_manifest.tsv','canary':'cem_canary_manifest.tsv','full':'cem_full_manifest.tsv'}[mode]
path=root/name; rows=list(csv.DictReader(path.open(),delimiter='\t'));fields=list(rows[0]);by={r['variant']:r for r in rows}
for shard in [*stage.glob('a100-gpu*.tsv'), local_shard]:
 if not shard.is_file():
  continue
 for r in csv.DictReader(shard.open(),delimiter='\t'):
  if r['variant'] in by: by[r['variant']].update(r)
with path.open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=fields,delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(rows)
print(f'merged remote and local worker state into {path}')
PY
echo "E192 remote pull complete: mode=$MODE"
