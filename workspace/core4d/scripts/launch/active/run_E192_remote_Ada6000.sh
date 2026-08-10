#!/usr/bin/env bash
# E192 remote RTX 6000 Ada two-card append-only launcher.
# This launcher owns only E192_* sessions and never stops unrelated jobs.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODE="${MODE:-${1:-canary}}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
REMOTE="${E192_ADA_REMOTE:-spider-remote}"
REMOTE_ROOT="${E192_ADA_ROOT:-/home/xiayb/pHRI_workspace/spider}"
ROOT="workspace/core4d/results/E192"
MANIFEST="$ROOT/s6_downstream/manifests/cem_${MODE}_manifest.tsv"
RUNNER="workspace/core4d/scripts/experiments/E192/run_cem_queue.py"
AUDITOR="workspace/core4d/scripts/experiments/E192/audit_gate_overrides.py"
case "$MODE" in canary|full) ;; *) echo "MODE must be canary|full" >&2; exit 2 ;; esac

"$PYTHON_BIN" "$AUDITOR" --require-all
SESSION="E192_${MODE}_ada6000_$(date +%Y%m%d_%H%M%S)"
STAGE="$ROOT/remote_staging/$SESSION"
mkdir -p "$STAGE" "logs/E192/launch"

"$PYTHON_BIN" - "$MANIFEST" "$STAGE" <<'PY'
import csv, sys
from pathlib import Path
manifest, stage = Path(sys.argv[1]), Path(sys.argv[2])
rows = list(csv.DictReader(manifest.open(), delimiter="\t")); fields = list(rows[0])
for worker in ("ada-gpu0", "ada-gpu1"):
    group = [r for r in rows if r["worker_id"] == worker and r["host"] == "ada6000"]
    if not group: raise SystemExit(f"empty Ada shard: {worker}")
    out = stage / f"{worker}.tsv"
    with out.open("w", newline="") as stream:
        w = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        w.writeheader(); w.writerows(group)
print(f"Ada rows={len(rows)} workers=ada-gpu0/ada-gpu1")
PY

SSH=(ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=4 "$REMOTE")
RSH="ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=4"
"${SSH[@]}" "mkdir -p '$REMOTE_ROOT/$STAGE' '$REMOTE_ROOT/logs/E192/cem/$MODE'"

mapfile -t FILES < <("$PYTHON_BIN" - "$MANIFEST" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1]), delimiter="\t"))
files = {sys.argv[1], "examples/run_mjwp.py", "spider/config.py", "spider/query_tape.py",
         "spider/optimizers/sampling.py", "spider/optimizers/sampling_fast.py",
         "spider/simulators/mjwp.py", "spider/simulators/mjwp_object_distance.py",
         "spider/rewards/__init__.py", "spider/rewards/surface_distance.py",
         "spider/geometry/__init__.py", "spider/geometry/grid_sdf.py"}
for r in rows:
    files.add(f"examples/config/override/core4d_{r['target_task']}.yaml")
    files.update(r[k] for k in ("target_scene", "trajectory", "contact_mask", "override_path", "scene_act", "scene_snapshot_path") if r[k])
print("\n".join(sorted(files)))
PY
)
for path in "${FILES[@]}"; do [ -f "$path" ] || { echo "missing sync input: $path" >&2; exit 3; }; done
rsync -az -e "$RSH" "$STAGE/" "$REMOTE:$REMOTE_ROOT/$STAGE/"
for path in "${FILES[@]}" "$RUNNER" workspace/core4d/scripts/experiments/E192/e192_common.py; do
  "${SSH[@]}" "mkdir -p '$REMOTE_ROOT/$(dirname "$path")'"
  rsync -az -e "$RSH" "$path" "$REMOTE:$REMOTE_ROOT/$(dirname "$path")/"
done

REMOTE_SCRIPT="$STAGE/run_remote.sh"
{
  printf '#!/usr/bin/env bash\nset -euo pipefail\ncd %q\n' "$REMOTE_ROOT"
  printf 'MODE=%q\nPY=%q\nRUNNER=%q\nSTAGE=%q\n' "$MODE" "$PYTHON_BIN" "$RUNNER" "$STAGE"
  cat <<'EOS'
pids=()
for gpu in 0 1; do
  shard="$STAGE/ada-gpu${gpu}.tsv"; [ -f "$shard" ] || continue
  "$PY" "$RUNNER" --mode "$MODE" --manifest-tsv "$shard" --python-bin "$PY" --gpu-id "$gpu" \
    >"logs/E192/cem/$MODE/ada-gpu${gpu}.worker.log" 2>&1 &
  pids+=("$!")
done
status=0; for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"
EOS
} > "$REMOTE_SCRIPT"
chmod +x "$REMOTE_SCRIPT"
rsync -az -e "$RSH" "$REMOTE_SCRIPT" "$REMOTE:$REMOTE_ROOT/$STAGE/"
"${SSH[@]}" "cd '$REMOTE_ROOT' && tmux new-session -d -s '$SESSION' -c '$REMOTE_ROOT' 'bash $REMOTE_SCRIPT'"
"$PYTHON_BIN" - "$ROOT/remote_staging/latest_${MODE}_ada6000.json" "$SESSION" "$STAGE" "$REMOTE_ROOT" <<'PY'
import json, sys
from datetime import datetime
from pathlib import Path
p = Path(sys.argv[1]); p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps({"created_at": datetime.now().astimezone().isoformat(timespec="seconds"), "session": sys.argv[2], "stage": sys.argv[3], "remote_root": sys.argv[4], "gpus": ["0", "1"]}, indent=2) + "\n")
PY
echo "E192 Ada6000 append-only session started: $SESSION"
