#!/usr/bin/env bash
# E179 remote Full launcher: exact user-fixed A100 GPUs 2,3,6,7.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODE="${1:-full}"
[ "$MODE" = "full" ] || {
  echo "E179 remote launcher supports only Full; canary is local GPU0." >&2
  exit 2
}

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
REMOTE="${REMOTE:-batchcom@61.172.170.106}"
REMOTE_CANONICAL_ROOT="${REMOTE_CANONICAL_ROOT:-/home/dataset-assist-0/xiayb/workspace/spider}"
REMOTE_SSH_PORT="${REMOTE_SSH_PORT:-30409}"
REMOTE_SSH_KEY="${REMOTE_SSH_KEY:-/home/ubuntu/.ssh/id_rsa_tianyiyun}"
REMOTE_PYTHON_BIN="${REMOTE_PYTHON_BIN:-${REMOTE_CANONICAL_ROOT}/.venv/bin/python}"
REMOTE_RUN_PARENT="${REMOTE_RUN_PARENT:-/home/dataset-assist-0/xiayb/workspace/e179_spider_runs}"
EXPECTED_GPUS="2 3 6 7"
A100_GPU_MEM_USED_LIMIT_MB="${A100_GPU_MEM_USED_LIMIT_MB:-5000}"
A100_AVAILABILITY_CMD="${A100_AVAILABILITY_CMD:-}"
A100_POLICY_GPUS="${A100_POLICY_GPUS:-}"
ALLOW_COMPUTE_OVERLAP="${E179_ALLOW_COMPUTE_OVERLAP:-1}"
PREP_ONLY="${E179_REMOTE_PREP_ONLY:-0}"
SESSION="${SESSION:-e179_a100_full_$(date +%Y%m%d_%H%M%S)}"
REMOTE_RUN_ROOT="${REMOTE_RUN_ROOT:-${REMOTE_RUN_PARENT}/${SESSION}}"

[ -x "$PYTHON_BIN" ] || {
  echo "missing project Python: $PYTHON_BIN" >&2
  exit 3
}
[ "$PREP_ONLY" = "0" ] || [ "$PREP_ONLY" = "1" ] || {
  echo "E179_REMOTE_PREP_ONLY must be 0 or 1" >&2
  exit 3
}
[ "$ALLOW_COMPUTE_OVERLAP" = "0" ] || \
  [ "$ALLOW_COMPUTE_OVERLAP" = "1" ] || {
  echo "E179_ALLOW_COMPUTE_OVERLAP must be 0 or 1" >&2
  exit 3
}
if [ -z "$A100_AVAILABILITY_CMD" ] && [ -z "$A100_POLICY_GPUS" ]; then
  echo "provide A100_AVAILABILITY_CMD or explicit A100_POLICY_GPUS" >&2
  echo "low memory alone is not reservation/owner authorization" >&2
  exit 3
fi
if [ -n "$A100_AVAILABILITY_CMD" ] && [ -n "$A100_POLICY_GPUS" ]; then
  echo "provide only one policy source" >&2
  exit 3
fi

RESULT_ROOT="workspace/core4d/results/E179"
MANIFEST_ROOT="$RESULT_ROOT/s6_downstream/manifests"
RUNNER="workspace/core4d/scripts/experiments/E179/run_cem_queue.py"
AUDITOR="workspace/core4d/scripts/experiments/E179/audit_e167a_no_prg.py"
EXECUTION_ROOT="$RESULT_ROOT/s0_environment/a100_full_${SESSION}"
QUEUE_ROOT="$EXECUTION_ROOT/queues"
SESSION_SCRIPT="$EXECUTION_ROOT/run_${SESSION}.sh"
EXECUTION_JSON="$EXECUTION_ROOT/execution_manifest.json"
LATEST_JSON="$RESULT_ROOT/s0_environment/a100_full_latest.json"
SYNC_LIST="$EXECUTION_ROOT/sync_files.txt"
SYNC_INVENTORY="$EXECUTION_ROOT/sync_inventory.json"
mkdir -p "$QUEUE_ROOT"

SSH_OPTS=(
  -o BatchMode=yes
  -o ConnectTimeout=20
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=4
  -p "$REMOTE_SSH_PORT"
  -i "$REMOTE_SSH_KEY"
)
RSYNC_RSH="ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=4 -p ${REMOTE_SSH_PORT} -i ${REMOTE_SSH_KEY}"
ssh_remote() {
  ssh "${SSH_OPTS[@]}" "$REMOTE" "$@"
}
retry() {
  local attempt=1
  local max_attempts="${E179_REMOTE_RETRY_MAX:-4}"
  while true; do
    if "$@"; then
      return 0
    fi
    if [ "$attempt" -ge "$max_attempts" ]; then
      echo "command failed after ${attempt} attempts: $*" >&2
      return 1
    fi
    sleep "$((attempt * 3))"
    attempt=$((attempt + 1))
  done
}

"$PYTHON_BIN" "$AUDITOR" --require-all >/dev/null
"$PYTHON_BIN" - <<'PY'
import csv
import importlib.util
from pathlib import Path

runner_path = Path(
    "workspace/core4d/scripts/experiments/E179/run_cem_queue.py"
)
spec = importlib.util.spec_from_file_location("e179_queue_gate", runner_path)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
manifest = Path(
    "workspace/core4d/results/E179/s6_downstream/manifests/"
    "cem_canary_local_gpu0.tsv"
)
rows = list(csv.DictReader(manifest.open(encoding="utf-8"), delimiter="\t"))
bad = {
    row["case_id"]: issues
    for row in rows
    if (issues := module.validate_runtime_outputs(row))
}
if len(rows) != 3 or bad:
    raise SystemExit(f"remote Full blocked by local canary gate: {bad}")
PY

for gpu in 2 3 6 7; do
  source_manifest="$MANIFEST_ROOT/cem_full_a100_gpu${gpu}.tsv"
  target_manifest="$QUEUE_ROOT/gpu${gpu}.tsv"
  [ -f "$source_manifest" ] || {
    echo "missing fixed shard: $source_manifest" >&2
    exit 3
  }
  cp "$source_manifest" "$target_manifest"
  "$PYTHON_BIN" "$RUNNER" \
    --mode full \
    --manifest-tsv "$target_manifest" \
    --python-bin "$PYTHON_BIN" \
    --gpu-id "$gpu" \
    --all \
    --dry-run >/dev/null
done

"$PYTHON_BIN" - "$QUEUE_ROOT" <<'PY'
import csv
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected = {
    "2": {
        "box023_20231008_045_p2",
        "box023_20231020_040_p1",
        "box023_20231020_042_p2",
    },
    "3": {
        "box023_20231011_018_p1",
        "box023_20231020_041_p2",
        "box023_20231020_039_p2",
    },
    "6": {
        "box023_20231011_021_p1",
        "box023_20231020_041_p1",
        "box023_20231020_042_p1",
    },
    "7": {
        "box023_20231011_021_p2",
        "box023_20231020_040_p2",
        "box023_20231008_046_p2",
    },
}
union = []
for gpu, case_ids in expected.items():
    rows = list(
        csv.DictReader(
            (root / f"gpu{gpu}.tsv").open(encoding="utf-8"),
            delimiter="\t",
        )
    )
    actual = {row["case_id"] for row in rows}
    if len(rows) != 3 or actual != case_ids:
        raise SystemExit(f"GPU{gpu} shard drift: {actual}")
    if any(row["assigned_gpu"] != gpu for row in rows):
        raise SystemExit(f"GPU{gpu} assignment field drift")
    union.extend(actual)
if len(union) != 12 or len(set(union)) != 12:
    raise SystemExit("remote shard union must contain 12 unique rows")
PY

POLICY_SNAPSHOT=""
GPU_SNAPSHOT_1=""
COMPUTE_SNAPSHOT_1=""
GPU_SNAPSHOT_2=""
COMPUTE_SNAPSHOT_2=""

capture_policy() {
  if [ -n "$A100_AVAILABILITY_CMD" ]; then
    POLICY_SNAPSHOT="$(retry ssh_remote "$A100_AVAILABILITY_CMD")"
  else
    POLICY_SNAPSHOT="$A100_POLICY_GPUS"
  fi
}

capture_and_validate_remote() {
  local label="$1"
  local gpu_snapshot compute_snapshot
  gpu_snapshot="$(
    retry ssh_remote \
      "nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits"
  )"
  compute_snapshot="$(
    retry ssh_remote \
      "nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader,nounits" \
      2>/dev/null || true
  )"
  GPU_SNAPSHOT="$gpu_snapshot" COMPUTE_SNAPSHOT="$compute_snapshot" \
  POLICY_SNAPSHOT="$POLICY_SNAPSHOT" EXPECTED_GPUS="$EXPECTED_GPUS" \
  LIMIT_MB="$A100_GPU_MEM_USED_LIMIT_MB" \
  ALLOW_COMPUTE_OVERLAP="$ALLOW_COMPUTE_OVERLAP" \
  "$PYTHON_BIN" - <<'PY'
import csv
import io
import os
import re

expected = {int(value) for value in os.environ["EXPECTED_GPUS"].split()}
policy = {
    int(value)
    for value in re.findall(
        r"(?<!\d)[0-7](?!\d)", os.environ["POLICY_SNAPSHOT"]
    )
}
if not expected.issubset(policy):
    raise SystemExit(
        f"fixed GPUs not authorized: expected={sorted(expected)} "
        f"policy={sorted(policy)}"
    )
rows = {}
for raw in csv.reader(io.StringIO(os.environ["GPU_SNAPSHOT"])):
    if not raw:
        continue
    values = [value.strip() for value in raw]
    rows[int(values[0])] = {
        "uuid": values[1],
        "memory_used": int(values[3]),
    }
if not expected.issubset(rows):
    raise SystemExit(f"fixed GPUs missing from nvidia-smi: {sorted(expected-set(rows))}")
busy = {
    raw[0].strip()
    for raw in csv.reader(io.StringIO(os.environ["COMPUTE_SNAPSHOT"]))
    if raw and raw[0].strip()
}
limit = int(os.environ["LIMIT_MB"])
allow_compute_overlap = os.environ["ALLOW_COMPUTE_OVERLAP"] == "1"
failures = []
for gpu in sorted(expected):
    row = rows[gpu]
    if row["memory_used"] >= limit:
        failures.append(f"gpu{gpu}:memory={row['memory_used']}MiB")
    if row["uuid"] in busy and not allow_compute_overlap:
        failures.append(f"gpu{gpu}:compute_process_present")
if failures:
    raise SystemExit("fixed A100 preflight failed: " + ", ".join(failures))
print(
    "fixed A100 pass: "
    + ", ".join(
        f"gpu{gpu}={rows[gpu]['memory_used']}MiB" for gpu in sorted(expected)
    )
    + f"; compute_overlap_authorized={allow_compute_overlap}"
)
PY
  if [ "$label" = "check1" ]; then
    GPU_SNAPSHOT_1="$gpu_snapshot"
    COMPUTE_SNAPSHOT_1="$compute_snapshot"
  else
    GPU_SNAPSHOT_2="$gpu_snapshot"
    COMPUTE_SNAPSHOT_2="$compute_snapshot"
  fi
}

capture_policy
capture_and_validate_remote check1

MODE="$MODE" QUEUE_ROOT="$QUEUE_ROOT" SYNC_LIST="$SYNC_LIST" \
SYNC_INVENTORY="$SYNC_INVENTORY" RUNNER="$RUNNER" \
EXECUTION_ROOT="$EXECUTION_ROOT" SESSION_SCRIPT="$SESSION_SCRIPT" \
"$PYTHON_BIN" - <<'PY'
import csv
import hashlib
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

import yaml

files = {
    Path("examples/run_mjwp.py"),
    Path("workspace/core4d/scripts/experiments/E179/e179_common.py"),
    Path("workspace/core4d/scripts/experiments/E179/run_cem_queue.py"),
    Path("workspace/core4d/scripts/experiments/E179/audit_e167a_no_prg.py"),
    Path(
        "workspace/core4d/results/E168/s0_environment/e167a_profile/"
        "e167a_zonly_profile.json"
    ),
}
tracked = subprocess.check_output(
    ["git", "ls-files", "-z", "spider", "examples/config"]
).decode().split("\0")
files.update(Path(value) for value in tracked if value)
rows = []
queue_root = Path(os.environ["QUEUE_ROOT"])
for gpu in ("2", "3", "6", "7"):
    shard = queue_root / f"gpu{gpu}.tsv"
    files.add(shard)
    rows.extend(
        csv.DictReader(shard.open(encoding="utf-8"), delimiter="\t")
    )
for row in rows:
    for key in (
        "target_scene",
        "trajectory",
        "scene_act",
        "contact_mask",
        "override_path",
    ):
        files.add(Path(row[key]))
    task_dir = Path(row["scene_act"]).parent
    files.update(path for path in task_dir.rglob("*") if path.is_file())

override_root = Path("examples/config/override")
pending = [Path(row["override_path"]).stem for row in rows]
seen = set()
while pending:
    override_id = pending.pop()
    if override_id in seen:
        continue
    seen.add(override_id)
    path = override_root / f"{override_id}.yaml"
    if not path.is_file():
        raise FileNotFoundError(path)
    files.add(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for item in payload.get("defaults", []):
        if isinstance(item, str) and item != "_self_":
            pending.append(item)
        elif isinstance(item, dict):
            pending.extend(
                value
                for value in item.values()
                if isinstance(value, str) and value != "_self_"
            )

object_root = Path(
    "example_datasets/processed/core4d/assets/objects/box023"
)
files.update(path for path in object_root.rglob("*") if path.is_file())
missing = sorted(str(path) for path in files if not path.is_file())
if missing:
    raise FileNotFoundError("missing sync files: " + ", ".join(missing[:20]))
sync_list = Path(os.environ["SYNC_LIST"])
sync_list.write_text(
    "\n".join(sorted(str(path) for path in files)) + "\n",
    encoding="utf-8",
)
artifacts = []
for path in sorted(files, key=str):
    artifacts.append(
        {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    )
Path(os.environ["SYNC_INVENTORY"]).write_text(
    json.dumps(
        {
            "created_at": datetime.now().astimezone().isoformat(
                timespec="seconds"
            ),
            "files": len(artifacts),
            "bytes": sum(item["bytes"] for item in artifacts),
            "artifacts": artifacts,
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY

{
  printf '#!/usr/bin/env bash\n'
  printf 'set -euo pipefail\n'
  printf 'cd %q\n' "$REMOTE_RUN_ROOT"
  printf 'export MUJOCO_GL=egl\n'
  printf 'export PYTHONPATH=%q\n' "$REMOTE_RUN_ROOT"
  printf 'export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 '
  printf 'OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1\n'
  printf 'pids=()\n'
  for gpu in 2 3 6 7; do
    worker_log="logs/E179/cem/full/${SESSION}_gpu${gpu}.worker.log"
    printf 'mkdir -p %q\n' "$(dirname "$worker_log")"
    printf '( %q %q --mode full --manifest-tsv %q ' \
      "$REMOTE_PYTHON_BIN" "$RUNNER" "$QUEUE_ROOT/gpu${gpu}.tsv"
    printf -- '--python-bin %q --gpu-id %q --all ) > %q 2>&1 &\n' \
      "$REMOTE_PYTHON_BIN" "$gpu" "$worker_log"
    printf 'pids+=("$!")\n'
  done
  printf 'status=0\n'
  printf 'for pid in "${pids[@]}"; do wait "$pid" || status=1; done\n'
  printf 'echo "E179 remote Full finished status=$status at $(date -Is)"\n'
  printf 'exit "$status"\n'
} > "$SESSION_SCRIPT"
chmod +x "$SESSION_SCRIPT"

# The session script was created after the initial inventory payload.
"$PYTHON_BIN" - "$SYNC_LIST" "$SYNC_INVENTORY" "$SESSION_SCRIPT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

sync_list, inventory_path, script = map(Path, sys.argv[1:])
paths = [
    Path(value)
    for value in sync_list.read_text(encoding="utf-8").splitlines()
    if value
]
if script not in paths:
    paths.append(script)
sync_list.write_text(
    "\n".join(sorted(str(path) for path in paths)) + "\n",
    encoding="utf-8",
)
payload = json.loads(inventory_path.read_text(encoding="utf-8"))
payload["artifacts"] = [
    {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    for path in sorted(paths, key=str)
]
payload["files"] = len(payload["artifacts"])
payload["bytes"] = sum(item["bytes"] for item in payload["artifacts"])
inventory_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

retry ssh_remote "mkdir -p '$REMOTE_RUN_ROOT'"
retry rsync -azLRL --files-from="$SYNC_LIST" -e "$RSYNC_RSH" \
  ./ "${REMOTE}:${REMOTE_RUN_ROOT}/"
retry ssh_remote \
  "mkdir -p '$REMOTE_RUN_ROOT/$(dirname "$SYNC_INVENTORY")'"
retry rsync -az --partial -e "$RSYNC_RSH" \
  "$SYNC_INVENTORY" \
  "${REMOTE}:${REMOTE_RUN_ROOT}/$(dirname "$SYNC_INVENTORY")/"

retry ssh_remote "cd '$REMOTE_RUN_ROOT' && \
  '$REMOTE_PYTHON_BIN' - '$SYNC_INVENTORY' <<'PY'
import hashlib
import json
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
bad = []
for item in payload['artifacts']:
    path = Path(item['path'])
    if not path.is_file():
        bad.append(item['path'] + ':missing')
    elif hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
        bad.append(item['path'] + ':sha')
if bad:
    raise SystemExit('remote sync verification failed: ' + ', '.join(bad[:20]))
print(f\"remote sync verified files={len(payload['artifacts'])}\")
PY"
retry ssh_remote "cd '$REMOTE_RUN_ROOT' && \
  PYTHONPATH='$REMOTE_RUN_ROOT' '$REMOTE_PYTHON_BIN' -m py_compile \
  '$RUNNER' workspace/core4d/scripts/experiments/E179/e179_common.py && \
  PYTHONPATH='$REMOTE_RUN_ROOT' '$REMOTE_PYTHON_BIN' '$RUNNER' \
  --mode full --manifest-tsv '$QUEUE_ROOT/gpu2.tsv' \
  --python-bin '$REMOTE_PYTHON_BIN' --gpu-id 2 --all --dry-run \
  > '$EXECUTION_ROOT/remote_preflight.log'"

capture_policy
capture_and_validate_remote check2

POLICY_SNAPSHOT="$POLICY_SNAPSHOT" \
GPU_SNAPSHOT_1="$GPU_SNAPSHOT_1" COMPUTE_SNAPSHOT_1="$COMPUTE_SNAPSHOT_1" \
GPU_SNAPSHOT_2="$GPU_SNAPSHOT_2" COMPUTE_SNAPSHOT_2="$COMPUTE_SNAPSHOT_2" \
A100_AVAILABILITY_CMD="$A100_AVAILABILITY_CMD" \
REMOTE="$REMOTE" REMOTE_CANONICAL_ROOT="$REMOTE_CANONICAL_ROOT" \
REMOTE_RUN_ROOT="$REMOTE_RUN_ROOT" REMOTE_PYTHON_BIN="$REMOTE_PYTHON_BIN" \
SESSION="$SESSION" QUEUE_ROOT="$QUEUE_ROOT" SYNC_INVENTORY="$SYNC_INVENTORY" \
EXECUTION_JSON="$EXECUTION_JSON" PREP_ONLY="$PREP_ONLY" \
LIMIT_MB="$A100_GPU_MEM_USED_LIMIT_MB" \
ALLOW_COMPUTE_OVERLAP="$ALLOW_COMPUTE_OVERLAP" \
"$PYTHON_BIN" - <<'PY'
import hashlib
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

inventory = Path(os.environ["SYNC_INVENTORY"])
payload = {
    "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    "experiment_id": "E179",
    "profile": "A100-8gpu",
    "mode": "full",
    "session": os.environ["SESSION"],
    "remote": os.environ["REMOTE"],
    "remote_canonical_root": os.environ["REMOTE_CANONICAL_ROOT"],
    "remote_root": os.environ["REMOTE_RUN_ROOT"],
    "remote_python": os.environ["REMOTE_PYTHON_BIN"],
    "shard_root": os.environ["QUEUE_ROOT"],
    "selected_gpus": ["2", "3", "6", "7"],
    "policy_source": (
        "availability_cmd"
        if os.environ.get("A100_AVAILABILITY_CMD")
        else "explicit_env"
    ),
    "policy_snapshot": os.environ["POLICY_SNAPSHOT"],
    "memory_limit_mb": int(os.environ["LIMIT_MB"]),
    "compute_overlap_authorized": (
        os.environ["ALLOW_COMPUTE_OVERLAP"] == "1"
    ),
    "check1_gpu_snapshot": os.environ["GPU_SNAPSHOT_1"],
    "check1_compute_snapshot": os.environ["COMPUTE_SNAPSHOT_1"],
    "check2_gpu_snapshot": os.environ["GPU_SNAPSHOT_2"],
    "check2_compute_snapshot": os.environ["COMPUTE_SNAPSHOT_2"],
    "sync_inventory": str(inventory),
    "sync_inventory_sha256": hashlib.sha256(
        inventory.read_bytes()
    ).hexdigest(),
    "local_git_head": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip(),
    "status": (
        "preflight_passed_no_launch"
        if os.environ["PREP_ONLY"] == "1"
        else "ready_to_launch"
    ),
}
path = Path(os.environ["EXECUTION_JSON"])
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
cp "$EXECUTION_JSON" "$LATEST_JSON"
retry rsync -az -e "$RSYNC_RSH" \
  "$EXECUTION_JSON" "${REMOTE}:${REMOTE_RUN_ROOT}/${EXECUTION_JSON}"
retry rsync -az -e "$RSYNC_RSH" \
  "$LATEST_JSON" "${REMOTE}:${REMOTE_RUN_ROOT}/${LATEST_JSON}"

if [ "$PREP_ONLY" = "1" ]; then
  echo "E179 remote Full preflight passed; no tmux/CEM launched."
  echo "Execution manifest: $EXECUTION_JSON"
  exit 0
fi

retry ssh_remote "cd '$REMOTE_RUN_ROOT' && \
  tmux new-session -d -s '$SESSION' -c '$REMOTE_RUN_ROOT' \
  'bash $SESSION_SCRIPT'"
echo "Started E179 remote Full: session=$SESSION GPUs=2,3,6,7"
echo "Execution manifest: $EXECUTION_JSON"
