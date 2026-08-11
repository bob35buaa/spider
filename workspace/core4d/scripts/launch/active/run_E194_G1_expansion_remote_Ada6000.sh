#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
MODE="${MODE:-canary}"
SENTINEL_ONLY="${SENTINEL_ONLY:-0}"
REMOTE="${E194_ADA_REMOTE:-spider-remote}"
REMOTE_ROOT="${E194_ADA_ROOT:-/home/xiayb/pHRI_workspace/spider}"
ROOT="workspace/core4d/results/E194"
RUNNER="workspace/core4d/scripts/experiments/E194/run_g1_expansion_queue.py"
COMMON="workspace/core4d/scripts/experiments/E194/e194_g1_expansion_common.py"
case "$MODE:$SENTINEL_ONLY" in
  canary:*) TAG="canary"; MANIFEST="$ROOT/s6_downstream/manifests/g1_expansion_canary_manifest.tsv" ;;
  full:1) TAG="sentinel"; MANIFEST="$ROOT/s6_downstream/manifests/g1_expansion_sentinel_manifest.tsv" ;;
  full:0) TAG="full"; MANIFEST="$ROOT/s6_downstream/manifests/g1_expansion_full_manifest.tsv" ;;
  *) echo "unsupported MODE=$MODE SENTINEL_ONLY=$SENTINEL_ONLY" >&2; exit 2 ;;
esac
SESSION="E194_G1_${TAG}_ada6000_$(date +%Y%m%d_%H%M%S)"
STAGE="$ROOT/remote_staging/$SESSION"
mkdir -p "$STAGE" "logs/E194/launch"
"$PYTHON_BIN" - "$MANIFEST" "$STAGE" "$TAG" <<'PY'
import csv,sys
from pathlib import Path
rows=list(csv.DictReader(open(sys.argv[1]),delimiter="\t")); fields=list(rows[0]); stage=Path(sys.argv[2]); tag=sys.argv[3]
expected={"canary":{"ada-gpu0":3,"ada-gpu1":3},"sentinel":{"ada-gpu0":2,"ada-gpu1":0},"full":{"ada-gpu0":18,"ada-gpu1":18}}[tag]
for worker,count in expected.items():
 group=[row for row in rows if row["worker"]==worker]
 if len(group)!=count: raise SystemExit(f"{worker} rows={len(group)} expected={count}")
 with (stage/f"{worker}.tsv").open("w",newline="") as stream:
  w=csv.DictWriter(stream,fieldnames=fields,delimiter="\t",lineterminator="\n"); w.writeheader(); w.writerows(group)
print(expected)
PY
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=4 "$REMOTE")
RSH="ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=4"
"${SSH[@]}" "mkdir -p '$REMOTE_ROOT/$STAGE' '$REMOTE_ROOT/logs/E194/cem/${TAG}_g1_expansion'"
mapfile -t FILES < <("$PYTHON_BIN" - "$MANIFEST" <<'PY'
import csv,sys
rows=list(csv.DictReader(open(sys.argv[1]),delimiter="\t"))
files={sys.argv[1],"examples/run_mjwp.py","spider/config.py","spider/io.py","spider/interp.py","spider/query_tape.py",
 "spider/optimizers/sampling.py","spider/optimizers/sampling_fast.py","spider/simulators/mjwp.py","spider/simulators/mjwp_object_distance.py",
 "spider/rewards/__init__.py","spider/rewards/surface_distance.py","spider/geometry/__init__.py","spider/geometry/grid_sdf.py"}
for row in rows:
 files.add(f"examples/config/override/core4d_{row['target_task']}.yaml")
 for key in ("target_scene","trajectory","contact_mask","override_path","scene_act"):
  if row.get(key): files.add(row[key])
print("\n".join(sorted(files)))
PY
)
for path in "${FILES[@]}"; do [ -f "$path" ] || { echo "missing sync input: $path" >&2; exit 3; }; done
rsync -az -e "$RSH" "$STAGE/" "$REMOTE:$REMOTE_ROOT/$STAGE/"
for path in "${FILES[@]}" "$RUNNER" "$COMMON"; do
  "${SSH[@]}" "mkdir -p '$REMOTE_ROOT/$(dirname "$path")'"
  rsync -az -e "$RSH" "$path" "$REMOTE:$REMOTE_ROOT/$(dirname "$path")/"
done
REMOTE_SCRIPT="$STAGE/run_remote.sh"
{
  printf '#!/usr/bin/env bash\nset -euo pipefail\ncd %q\n' "$REMOTE_ROOT"
  printf 'PY=%q\nRUNNER=%q\nSTAGE=%q\nTAG=%q\n' "$PYTHON_BIN" "$RUNNER" "$STAGE" "$TAG"
  cat <<'EOS'
pids=()
for gpu in 0 1; do
  shard="$STAGE/ada-gpu${gpu}.tsv"
  rows=$(($(wc -l < "$shard")-1))
  if [ "$rows" -eq 0 ]; then continue; fi
  "$PY" "$RUNNER" --manifest-tsv "$shard" --python-bin "$PY" --gpu-id "$gpu" \
    >"logs/E194/cem/${TAG}_g1_expansion/ada-gpu${gpu}.worker.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"
EOS
} > "$REMOTE_SCRIPT"
chmod +x "$REMOTE_SCRIPT"
rsync -az -e "$RSH" "$REMOTE_SCRIPT" "$REMOTE:$REMOTE_ROOT/$STAGE/"
"${SSH[@]}" "cd '$REMOTE_ROOT' && tmux new-session -d -s '$SESSION' -c '$REMOTE_ROOT' 'bash $REMOTE_SCRIPT'"
"$PYTHON_BIN" - "$ROOT/remote_staging/latest_${TAG}_ada6000.json" "$SESSION" "$STAGE" "$REMOTE_ROOT" <<'PY'
import json,sys
from datetime import datetime
from pathlib import Path
p=Path(sys.argv[1]); p.parent.mkdir(parents=True,exist_ok=True)
p.write_text(json.dumps({"created_at":datetime.now().astimezone().isoformat(timespec="seconds"),"session":sys.argv[2],"stage":sys.argv[3],"remote_root":sys.argv[4]},indent=2)+"\n")
PY
echo "E194 G1 $TAG Ada6000 session started: $SESSION"
