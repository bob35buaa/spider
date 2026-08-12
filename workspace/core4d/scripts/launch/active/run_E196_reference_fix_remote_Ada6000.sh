#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
MODE="${MODE:-wave0}"
PREPARE_ONLY="${PREPARE_ONLY:-0}"
USE_PREPARED="${USE_PREPARED:-0}"
ALLOW_SUGAR_OVERLAP="${ALLOW_SUGAR_OVERLAP:-0}"
REMOTE="${E196_ADA_REMOTE:-spider-remote}"
SSH_CONFIG_FILE="${E196_ADA_SSH_CONFIG:-}"
SSH_PORT="${E196_ADA_PORT:-}"
SSH_BIND_INTERFACE="${E196_ADA_BIND_INTERFACE:-}"
REMOTE_ROOT="${E196_ADA_SOURCE_ROOT:-/home/xiayb/pHRI_workspace/spider}"
REMOTE_RUN_ROOT="${E196_ADA_RUN_ROOT:-/home/xiayb/pHRI_workspace/spider_e196_reference_fix}"
ROOT="workspace/core4d/results/E196"
RUNNER="workspace/core4d/scripts/experiments/E196/run_reference_fix_queue.py"
COMMON="workspace/core4d/scripts/experiments/E196/e196_reference_fix_common.py"
AUDITOR="workspace/core4d/scripts/experiments/E196/audit_reference_fix.py"
case "$MODE" in
  wave0) MANIFEST="$ROOT/s6_downstream/manifests/reference_fix_wave0_manifest.tsv"; EXPECTED0=1; EXPECTED1=1 ;;
  remaining) MANIFEST="$ROOT/s6_downstream/manifests/reference_fix_remaining_manifest.tsv"; EXPECTED0=9; EXPECTED1=8 ;;
  *) echo "unsupported MODE=$MODE" >&2; exit 2 ;;
esac
POINTER="$ROOT/s6_downstream/execution/latest_prepared_${MODE}_ada6000.json"
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
    echo "Ada SSH attempt $attempt/5 failed" >&2
    [ "$attempt" -lt 5 ] && sleep 3
  done
  return 255
}

run_rsync() {
  local attempt
  for attempt in 1 2 3 4 5; do
    if rsync "$@"; then return 0; fi
    echo "Ada rsync attempt $attempt/5 failed" >&2
    [ "$attempt" -lt 5 ] && sleep 3
  done
  return 255
}

check_remote_gpus() {
  local state
  state="$(run_ssh "nvidia-smi --query-gpu=index,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits")"
  "$PYTHON_BIN" - "$state" "$ALLOW_SUGAR_OVERLAP" <<'PY'
import sys
rows={}
for line in sys.argv[1].splitlines():
    index,total,memory,utilization=(int(value.strip()) for value in line.split(","))
    rows[index]=(total,memory,utilization)
allow_sugar=sys.argv[2]=="1"
for index in (0,1):
    if index not in rows: raise SystemExit(f"Ada GPU{index} missing")
    total,memory,utilization=rows[index]
    if allow_sugar:
        free=total-memory
        if free<24576:
            raise SystemExit(f"Ada GPU{index} lacks overlap headroom: free={free}MB < 24576MB")
    elif memory>=2000 or utilization>=20:
        raise SystemExit(f"Ada GPU{index} busy: memory={memory}MB utilization={utilization}%")
print(f"Ada GPUs resource gate passed: allow_sugar={allow_sugar} rows={rows}")
PY
  local apps
  apps="$(run_ssh "nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader,nounits")"
  if [ -n "$apps" ]; then
    if [ "$ALLOW_SUGAR_OVERLAP" != "1" ]; then
      echo "Ada has compute processes; refusing launch:" >&2
      echo "$apps" >&2
      return 1
    fi
    "$PYTHON_BIN" - "$apps" <<'PY'
import sys
rows=[line.strip() for line in sys.argv[1].splitlines() if line.strip()]
unknown=[line for line in rows if "sugar" not in line.lower()]
if unknown:
    raise SystemExit("non-SUGAR Ada compute process blocks overlap: " + " | ".join(unknown))
print("Ada SUGAR overlap allowlist passed: " + " | ".join(rows))
PY
  fi
}

if [ "$USE_PREPARED" != "1" ]; then
  SESSION="E196_reference_${MODE}_ada"
  STAGE="$ROOT/s6_downstream/execution/deploy_${MODE}_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$STAGE" "logs/E196/launch"
  "$PYTHON_BIN" - "$MANIFEST" "$STAGE" "$EXPECTED0" "$EXPECTED1" <<'PY'
import csv,sys
from pathlib import Path
source,stage=Path(sys.argv[1]),Path(sys.argv[2]); expected={"ada-gpu0":int(sys.argv[3]),"ada-gpu1":int(sys.argv[4])}
rows=list(csv.DictReader(source.open(),delimiter="\t")); fields=list(rows[0])
for worker,count in expected.items():
    group=[row for row in rows if row["worker"]==worker]
    if len(group)!=count: raise SystemExit(f"{worker} rows={len(group)} expected={count}")
    with (stage/f"{worker}.tsv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,delimiter="\t",lineterminator="\n")
        writer.writeheader(); writer.writerows(group)
print(expected)
PY
  run_ssh "if [ ! -d '$REMOTE_RUN_ROOT' ]; then cp -al '$REMOTE_ROOT' '$REMOTE_RUN_ROOT'; printf '%s\n' '$REMOTE_ROOT' > '$REMOTE_RUN_ROOT/.e196_overlay_source'; fi; test \"\$(cat '$REMOTE_RUN_ROOT/.e196_overlay_source')\" = '$REMOTE_ROOT'"
  mapfile -t FILES < <("$PYTHON_BIN" - "$MANIFEST" "$STAGE" <<'PY'
import csv,sys
from pathlib import Path
manifest,stage=Path(sys.argv[1]),Path(sys.argv[2])
rows=[row for row in csv.DictReader(manifest.open(),delimiter="\t") if row["worker"].startswith("ada-")]
files={str(stage/"ada-gpu0.tsv"),str(stage/"ada-gpu1.tsv"),
 "examples/run_mjwp.py","spider/config.py","spider/io.py","spider/interp.py","spider/query_tape.py",
 "spider/postprocess/get_success_rate.py","spider/optimizers/sampling.py","spider/optimizers/sampling_fast.py",
 "spider/simulators/mjwp.py","spider/simulators/mjwp_object_distance.py","spider/simulators/scene_act_reference.py","spider/simulators/hdmi.py",
 "spider/rewards/__init__.py","spider/rewards/surface_distance.py","spider/geometry/__init__.py","spider/geometry/grid_sdf.py",
 "workspace/core4d/scripts/experiments/E196/e196_reference_fix_common.py",
 "workspace/core4d/scripts/experiments/E196/audit_reference_fix.py",
 "workspace/core4d/scripts/experiments/E196/run_reference_fix_queue.py"}
for path in Path("examples/config").rglob("*"):
    if path.is_file() and "override" not in path.parts: files.add(path.as_posix())
for row in rows:
    files.add(f"examples/config/override/core4d_{row['target_task']}.yaml")
    for key in ("target_scene","trajectory","contact_mask","override_path","scene_act","scene_act_meta_path","e194_config_act"):
        if row.get(key): files.add(row[key])
    snapshot=Path("workspace/core4d/results/E196/scene_snapshot/reference_fix")/row["case_id"]
    files.add((snapshot/Path(row["scene_act"]).name).as_posix())
    files.add((snapshot/"scene_act_meta.json").as_posix())
print("\n".join(sorted(files)))
PY
  )
  for path in "${FILES[@]}"; do
    [ -f "$path" ] || { echo "missing E196 deployment input: $path" >&2; exit 3; }
  done
  SHA_MANIFEST="$STAGE/deployment_sha256.tsv"
  "$PYTHON_BIN" - "$SHA_MANIFEST" "${FILES[@]}" <<'PY'
import hashlib,sys
from pathlib import Path
out=Path(sys.argv[1])
with out.open("w") as stream:
    stream.write("path\tsha256\n")
    for raw in sorted(sys.argv[2:]):
        path=Path(raw); stream.write(f"{path.as_posix()}\t{hashlib.sha256(path.read_bytes()).hexdigest()}\n")
PY
  FILES+=("$SHA_MANIFEST")
  run_rsync -azR -e "$RSH" "${FILES[@]}" "$REMOTE:$REMOTE_RUN_ROOT/"
  run_ssh "cd '$REMOTE_RUN_ROOT' && PYTHONPATH='$REMOTE_RUN_ROOT' '$PYTHON_BIN' - '$SHA_MANIFEST' <<'PY'
import csv,hashlib,sys
from pathlib import Path
for row in csv.DictReader(Path(sys.argv[1]).open(),delimiter='\t'):
    path=Path(row['path'])
    if not path.is_file(): raise SystemExit(f'missing deployed file: {path}')
    actual=hashlib.sha256(path.read_bytes()).hexdigest()
    if actual!=row['sha256']: raise SystemExit(f'deployment SHA mismatch: {path}')
print('remote deployment SHA audit passed')
PY"
  for gpu in 0 1; do
    run_ssh "cd '$REMOTE_RUN_ROOT' && PYTHONPATH='$REMOTE_RUN_ROOT' '$PYTHON_BIN' '$AUDITOR' --manifest '$STAGE/ada-gpu${gpu}.tsv' --scope prelaunch --allow-subset --require-all"
  done
  "$PYTHON_BIN" - "$POINTER" "$SESSION" "$STAGE" "$REMOTE_RUN_ROOT" <<'PY'
import json,sys
from datetime import datetime
from pathlib import Path
path=Path(sys.argv[1]); path.parent.mkdir(parents=True,exist_ok=True)
path.write_text(json.dumps({"created_at":datetime.now().astimezone().isoformat(timespec="seconds"),"session":sys.argv[2],"stage":sys.argv[3],"remote_run_root":sys.argv[4]},indent=2)+"\n")
PY
  if [ "$PREPARE_ONLY" = "1" ]; then
    echo "E196 Ada $MODE deployment prepared and audited"
    exit 0
  fi
else
  [ -f "$POINTER" ] || { echo "missing prepared pointer: $POINTER" >&2; exit 3; }
  mapfile -t META < <("$PYTHON_BIN" - "$POINTER" <<'PY'
import json,sys
data=json.load(open(sys.argv[1])); print(data["session"]); print(data["stage"]); print(data["remote_run_root"])
PY
  )
  SESSION="${META[0]}"; STAGE="${META[1]}"; REMOTE_RUN_ROOT="${META[2]}"
fi

check_remote_gpus
GPU_SNAPSHOT="$STAGE/gpu_snapshot_before_launch.txt"
{
  date --iso-8601=seconds
  printf 'remote=%s\nremote_run_root=%s\nmode=%s\nallow_sugar_overlap=%s\n' \
    "$REMOTE" "$REMOTE_RUN_ROOT" "$MODE" "$ALLOW_SUGAR_OVERLAP"
  run_ssh "nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits"
  run_ssh "nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader,nounits" || true
} > "$GPU_SNAPSHOT"
SESSION_STATE="$(run_ssh "if tmux has-session -t '$SESSION' 2>/dev/null; then echo exists; else echo absent; fi")"
if [ "$SESSION_STATE" = "exists" ]; then
  echo "remote tmux session already exists: $SESSION" >&2
  exit 5
fi
for gpu in 0 1; do
  run_ssh "cd '$REMOTE_RUN_ROOT' && PYTHONPATH='$REMOTE_RUN_ROOT' '$PYTHON_BIN' '$AUDITOR' --manifest '$STAGE/ada-gpu${gpu}.tsv' --scope prelaunch --allow-subset --require-all"
done
REMOTE_SCRIPT="$STAGE/run_remote.sh"
{
  printf '#!/usr/bin/env bash\nset -euo pipefail\ncd %q\n' "$REMOTE_RUN_ROOT"
  printf 'PY=%q\nRUNNER=%q\nSTAGE=%q\nMODE=%q\n' "$PYTHON_BIN" "$RUNNER" "$STAGE" "$MODE"
  cat <<'EOS'
export PYTHONPATH="$PWD"
pids=()
for gpu in 0 1; do
  shard="$STAGE/ada-gpu${gpu}.tsv"
  "$PY" "$RUNNER" --manifest-tsv "$shard" --python-bin "$PY" --gpu-id "$gpu" \
    >"logs/E196/launch/${MODE}_ada_gpu${gpu}.worker.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"
EOS
} > "$REMOTE_SCRIPT"
chmod +x "$REMOTE_SCRIPT"
run_rsync -azR -e "$RSH" "$REMOTE_SCRIPT" "$REMOTE:$REMOTE_RUN_ROOT/"
REMOTE_SCRIPT_SHA="$(sha256sum "$REMOTE_SCRIPT" | awk '{print $1}')"
run_ssh "cd '$REMOTE_RUN_ROOT' && test \"\$(sha256sum '$REMOTE_SCRIPT' | awk '{print \$1}')\" = '$REMOTE_SCRIPT_SHA'"
run_ssh "cd '$REMOTE_RUN_ROOT' && mkdir -p logs/E196/launch && tmux new-session -d -s '$SESSION' -c '$REMOTE_RUN_ROOT' 'bash $REMOTE_SCRIPT'"
LAUNCH_POINTER="$ROOT/s6_downstream/execution/latest_${MODE}_ada6000.json"
"$PYTHON_BIN" - "$POINTER" "$LAUNCH_POINTER" "$GPU_SNAPSHOT" "$REMOTE_SCRIPT_SHA" "$ALLOW_SUGAR_OVERLAP" <<'PY'
import json,sys
from datetime import datetime
from pathlib import Path
source,target,snapshot=map(Path,sys.argv[1:4])
data=json.loads(source.read_text())
data.update({
    "launched_at":datetime.now().astimezone().isoformat(timespec="seconds"),
    "gpu_snapshot":snapshot.as_posix(),
    "run_remote_sha256":sys.argv[4],
    "allow_sugar_overlap":sys.argv[5]=="1",
})
target.write_text(json.dumps(data,indent=2,sort_keys=True)+"\n")
PY
echo "E196 Ada $MODE session started: $SESSION"
