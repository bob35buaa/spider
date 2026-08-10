#!/usr/bin/env bash
# E192 fixed A100 GPU4/5 append-only launcher; no process control or migration.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODE="${MODE:-${1:-baseline_sentinel}}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
PREFLIGHT_ONLY="${E192_PREFLIGHT_ONLY:-0}"
REMOTE="${E192_A100_REMOTE:-batchcom@61.172.170.106}"
REMOTE_ROOT="${E192_A100_ROOT:-/home/dataset-assist-0/xiayb/workspace/spider}"
PORT="${E192_A100_PORT:-30409}"
KEY="${E192_A100_KEY:-/home/ubuntu/.ssh/id_rsa_tianyiyun}"
ROOT="workspace/core4d/results/E192"
RUNNER="workspace/core4d/scripts/experiments/E192/run_cem_queue.py"
AUDITOR="workspace/core4d/scripts/experiments/E192/audit_gate_overrides.py"
case "$MODE" in
  baseline_sentinel) MANIFEST="$ROOT/s6_downstream/manifests/cem_baseline_sentinel_manifest.tsv" ;;
  canary) MANIFEST="$ROOT/s6_downstream/manifests/cem_canary_manifest.tsv" ;;
  full) MANIFEST="$ROOT/s6_downstream/manifests/cem_full_manifest.tsv" ;;
  *) echo "invalid MODE=$MODE" >&2; exit 2 ;;
esac

"$PYTHON_BIN" "$AUDITOR" --require-all
SESSION="E192_${MODE}_a100_$(date +%Y%m%d_%H%M%S)"
STAGE="$ROOT/remote_staging/$SESSION"
mkdir -p "$STAGE"
MODE="$MODE" MANIFEST="$MANIFEST" STAGE="$STAGE" "$PYTHON_BIN" - <<'PY'
import csv,os
from pathlib import Path
rows=list(csv.DictReader(open(os.environ['MANIFEST']),delimiter='\t')); fields=list(rows[0])
remote=[r for r in rows if r['host']=='a100']
for worker in ('a100-gpu4','a100-gpu5'):
 grp=[r for r in remote if r['worker_id']==worker]
 if not grp: continue
 p=Path(os.environ['STAGE'])/f'{worker}.tsv'; p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields,delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(grp)
print(f"remote rows={len(remote)} workers={sorted({r['worker_id'] for r in remote})}")
PY

if [ "$PREFLIGHT_ONLY" = "1" ]; then
  echo "E192 remote preflight only: positive audit passed; shards=$STAGE"
  exit 0
fi

SSH=(ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=4 -p "$PORT" -i "$KEY" "$REMOTE")
RSH="ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=4 -p $PORT -i $KEY"
if "${SSH[@]}" "ps -eo args | grep -F '$RUNNER --mode $MODE' | grep -v grep >/dev/null"; then
  echo "E192 remote mode already active: $MODE" >&2
  exit 4
fi
# Sync only E192/code and hash-pinned inputs registered by the manifest.
mapfile -t FILES < <("$PYTHON_BIN" - "$MANIFEST" <<'PY'
import csv,sys
from pathlib import Path
rows=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'))
files={sys.argv[1],'examples/run_mjwp.py','spider/config.py','spider/query_tape.py','spider/optimizers/sampling.py','spider/optimizers/sampling_fast.py','spider/simulators/mjwp.py','spider/simulators/mjwp_object_distance.py','spider/rewards/__init__.py','spider/rewards/surface_distance.py','spider/geometry/__init__.py','spider/geometry/grid_sdf.py'}
for r in rows:
 if r['host']!='a100': continue
 files.add(f"examples/config/override/core4d_{r['target_task']}.yaml")
 files.add(
  f"example_datasets/processed/core4d/assets/objects/{r['object_key']}/"
  f"{r['object_key']}_m.obj"
 )
 for k in ('target_scene','trajectory','contact_mask','override_path','scene_act','scene_snapshot_path'):
  files.add(r[k])
print('\n'.join(sorted(files)))
PY
)
for path in "${FILES[@]}"; do [ -f "$path" ] || { echo "missing sync input: $path" >&2; exit 3; }; done
"${SSH[@]}" "mkdir -p '$REMOTE_ROOT/$STAGE' '$REMOTE_ROOT/logs/E192/cem/$MODE'"
rsync -az -e "$RSH" "$STAGE/" "$REMOTE:$REMOTE_ROOT/$STAGE/"
for path in "${FILES[@]}" "$RUNNER" workspace/core4d/scripts/experiments/E192/e192_common.py; do
  "${SSH[@]}" "mkdir -p '$REMOTE_ROOT/$(dirname "$path")'"
  rsync -az -e "$RSH" "$path" "$REMOTE:$REMOTE_ROOT/$(dirname "$path")/"
done

SESSION_SCRIPT="$STAGE/run_remote.sh"
{
  printf '#!/usr/bin/env bash\nset -euo pipefail\ncd %q\n' "$REMOTE_ROOT"
  printf 'MODE=%q\nPY=%q\nRUNNER=%q\nSTAGE=%q\n' "$MODE" "$PYTHON_BIN" "$RUNNER" "$STAGE"
  cat <<'EOS'
pids=()
for gpu in 4 5; do
  shard="$STAGE/a100-gpu${gpu}.tsv"; [ -f "$shard" ] || continue
  "$PY" "$RUNNER" --mode "$MODE" --manifest-tsv "$shard" --python-bin "$PY" --gpu-id "$gpu" --all \
    >"logs/E192/cem/$MODE/a100-gpu${gpu}.worker.log" 2>&1 &
  pids+=("$!")
done
status=0; for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"
EOS
} > "$SESSION_SCRIPT"
chmod +x "$SESSION_SCRIPT"
rsync -az -e "$RSH" "$SESSION_SCRIPT" "$REMOTE:$REMOTE_ROOT/$STAGE/"
"${SSH[@]}" "cd '$REMOTE_ROOT' && tmux new-session -d -s '$SESSION' -c '$REMOTE_ROOT' 'bash $SESSION_SCRIPT'"
"$PYTHON_BIN" - "$ROOT/remote_staging/latest_${MODE}_a100.json" "$SESSION" "$STAGE" "$REMOTE_ROOT" <<'PY'
import json,sys
from datetime import datetime
from pathlib import Path
p=Path(sys.argv[1]);p.parent.mkdir(parents=True,exist_ok=True)
p.write_text(json.dumps({'created_at':datetime.now().astimezone().isoformat(timespec='seconds'),'session':sys.argv[2],'stage':sys.argv[3],'remote_root':sys.argv[4],'gpus':['4','5']},indent=2)+'\n')
PY
echo "E192 remote append-only session started: $SESSION"
