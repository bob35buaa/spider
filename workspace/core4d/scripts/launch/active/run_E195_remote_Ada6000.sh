#!/usr/bin/env bash
# Launch the E195 4/4 queues on remote RTX 6000 Ada GPU0/1 without touching other jobs.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
REMOTE="${E195_ADA_REMOTE:-spider-remote}"
REMOTE_ROOT="${E195_ADA_ROOT:-/home/xiayb/pHRI_workspace/spider}"
ROOT="workspace/core4d/results/E195"
MANIFEST="$ROOT/s6_downstream/manifests/cem_full_manifest.tsv"
RUNNER="workspace/core4d/scripts/experiments/E195/run_cem_queue.py"
COMMON="workspace/core4d/scripts/experiments/E195/e195_common.py"
AUDITOR="workspace/core4d/scripts/experiments/E195/audit_config.py"

"$PYTHON_BIN" "$AUDITOR" --require-all
SESSION="E195_full_ada6000_$(date +%Y%m%d_%H%M%S)"
STAGE="$ROOT/remote_staging/$SESSION"
mkdir -p "$STAGE" "logs/E195/launch"

"$PYTHON_BIN" - "$MANIFEST" "$STAGE" <<'PY'
import csv, sys
from pathlib import Path
manifest, stage = Path(sys.argv[1]), Path(sys.argv[2])
rows = list(csv.DictReader(manifest.open(), delimiter="\t")); fields = list(rows[0])
for worker in ("ada-gpu0", "ada-gpu1"):
    group = [row for row in rows if row["worker_id"] == worker and row["host"] == "ada6000"]
    if len(group) != 4:
        raise SystemExit(f"{worker} rows={len(group)} expected=4")
    with (stage / f"{worker}.tsv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader(); writer.writerows(group)
print("Ada rows=8 workers=4/4")
PY

SSH=(ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=4 "$REMOTE")
RSH="ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=4"
"${SSH[@]}" "mkdir -p '$REMOTE_ROOT/$STAGE' '$REMOTE_ROOT/logs/E195/cem/full'"

mapfile -t FILES < <("$PYTHON_BIN" - "$MANIFEST" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1]), delimiter="\t"))
files = {
    sys.argv[1], "examples/run_mjwp.py", "spider/config.py", "spider/query_tape.py",
    "spider/optimizers/sampling.py", "spider/optimizers/sampling_fast.py",
    "spider/simulators/mjwp.py", "spider/simulators/mjwp_object_distance.py",
    "spider/rewards/__init__.py", "spider/rewards/surface_distance.py",
    "spider/geometry/__init__.py", "spider/geometry/grid_sdf.py",
}
for row in rows:
    files.add(f"examples/config/override/core4d_{row['target_task']}.yaml")
    files.update(row[key] for key in (
        "target_scene", "trajectory", "contact_mask", "override_path", "scene_act"
    ) if row[key])
print("\n".join(sorted(files)))
PY
)
for path in "${FILES[@]}"; do
  [ -f "$path" ] || { echo "missing sync input: $path" >&2; exit 3; }
done
rsync -az -e "$RSH" "$STAGE/" "$REMOTE:$REMOTE_ROOT/$STAGE/"
for path in "${FILES[@]}" "$RUNNER" "$COMMON"; do
  "${SSH[@]}" "mkdir -p '$REMOTE_ROOT/$(dirname "$path")'"
  rsync -az -e "$RSH" "$path" "$REMOTE:$REMOTE_ROOT/$(dirname "$path")/"
done

REMOTE_SCRIPT="$STAGE/run_remote.sh"
{
  printf '#!/usr/bin/env bash\nset -euo pipefail\ncd %q\n' "$REMOTE_ROOT"
  printf 'PY=%q\nRUNNER=%q\nSTAGE=%q\n' "$PYTHON_BIN" "$RUNNER" "$STAGE"
  cat <<'EOS'
pids=()
for gpu in 0 1; do
  shard="$STAGE/ada-gpu${gpu}.tsv"
  "$PY" "$RUNNER" --mode full --manifest-tsv "$shard" --python-bin "$PY" --gpu-id "$gpu" \
    >"logs/E195/cem/full/ada-gpu${gpu}.worker.log" 2>&1 &
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

"$PYTHON_BIN" - "$ROOT/remote_staging/latest_full_ada6000.json" "$SESSION" "$STAGE" "$REMOTE_ROOT" <<'PY'
import json, sys
from datetime import datetime
from pathlib import Path
path = Path(sys.argv[1]); path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps({
    "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    "session": sys.argv[2], "stage": sys.argv[3], "remote_root": sys.argv[4],
    "gpus": ["0", "1"],
}, indent=2) + "\n")
PY
echo "E195 Ada6000 append-only session started: $SESSION"

