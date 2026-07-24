#!/usr/bin/env bash
# E178 isolated local RTX 5090 speed probe.
#
# Usage:
#   bash workspace/core4d/scripts/launch/active/run_E178_local_speed_probe.sh canary
#   bash workspace/core4d/scripts/launch/active/run_E178_local_speed_probe.sh density
#
# canary: 64 samples x 4 iterations, directly comparable with the A100 canary.
# density: 1024 samples x 2 iterations, estimates production sample-density.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PROBE_MODE="${1:-canary}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
GPU_ID="${GPU_ID:-0}"
SOURCE_MANIFEST="$(
  printf '%s' \
    'workspace/core4d/results/E178/s6_downstream/manifests/' \
    'semantic_bucket_canary_manifest.tsv'
)"
SOURCE_CASE="${SOURCE_CASE:-bucket007_20231020_055_p1}"
RUNNER="workspace/core4d/scripts/experiments/E176/run_cem_queue.py"
RUNTIME_VALIDATOR="workspace/core4d/scripts/experiments/E176/validate_cem_runtime.py"
THROUGHPUT_VALIDATOR="workspace/core4d/scripts/experiments/E176/validate_canary_throughput.py"

case "$PROBE_MODE" in
  canary)
    NUM_SAMPLES=64
    NUM_ITERATIONS=4
    RUNNER_MODE=canary
    EXECUTION_MODE=canary
    ;;
  density)
    NUM_SAMPLES=1024
    NUM_ITERATIONS=2
    RUNNER_MODE=full
    EXECUTION_MODE=production
    ;;
  *)
    echo "usage: $0 {canary|density}" >&2
    exit 2
    ;;
esac

[ -x "$PYTHON_BIN" ] || {
  echo "missing project python: $PYTHON_BIN" >&2
  exit 2
}
[ -f "$SOURCE_MANIFEST" ] || {
  echo "missing source manifest: $SOURCE_MANIFEST" >&2
  exit 2
}

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
PROBE_ID="local_speed_probe_${PROBE_MODE}_${RUN_TAG}"
ENV_ROOT="workspace/core4d/results/E178/s0_environment/${PROBE_ID}"
RESULT_ROOT="workspace/core4d/results/E178/s6_downstream/benchmark/${PROBE_ID}"
LOG_ROOT="logs/E178/benchmark/${PROBE_ID}"
MANIFEST="${ENV_ROOT}/probe_manifest.tsv"
RUNTIME_GATE="${RESULT_ROOT}/runtime_gate.json"
TIMING_JSON="${RESULT_ROOT}/timing_summary.json"
SUMMARY_JSON="${RESULT_ROOT}/probe_summary.json"
mkdir -p "$ENV_ROOT" "$RESULT_ROOT" "$LOG_ROOT"

nvidia-smi \
  --query-gpu=index,name,uuid,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw \
  --format=csv,noheader,nounits > "${ENV_ROOT}/gpu_before.csv"
nvidia-smi \
  --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory \
  --format=csv,noheader,nounits > "${ENV_ROOT}/compute_before.csv"

SOURCE_MANIFEST="$SOURCE_MANIFEST" SOURCE_CASE="$SOURCE_CASE" \
PROBE_ID="$PROBE_ID" ENV_ROOT="$ENV_ROOT" RESULT_ROOT="$RESULT_ROOT" \
LOG_ROOT="$LOG_ROOT" MANIFEST="$MANIFEST" NUM_SAMPLES="$NUM_SAMPLES" \
NUM_ITERATIONS="$NUM_ITERATIONS" EXECUTION_MODE="$EXECUTION_MODE" \
GPU_ID="$GPU_ID" "$PYTHON_BIN" - <<'PY'
import csv
import os
from datetime import datetime
from pathlib import Path

source = Path(os.environ["SOURCE_MANIFEST"])
rows = list(csv.DictReader(source.open(encoding="utf-8"), delimiter="\t"))
fields = list(rows[0]) if rows else []
matches = [row for row in rows if row["case_id"] == os.environ["SOURCE_CASE"]]
if len(matches) != 1:
    raise SystemExit(
        f"expected one source row for {os.environ['SOURCE_CASE']}, got {len(matches)}"
    )
row = dict(matches[0])
variant = (
    row["variant"].removesuffix("_canary")
    + "_"
    + os.environ["PROBE_ID"]
)
result_root = Path(os.environ["RESULT_ROOT"])
log_root = Path(os.environ["LOG_ROOT"])
row.update(
    {
        "assigned_gpu": os.environ["GPU_ID"],
        "gpu_id": os.environ["GPU_ID"],
        "status": "not_run",
        "failure_mode": "",
        "blocker_detail": "",
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "variant": variant,
        "result_npz": str(result_root / f"{variant}.npz"),
        "outdir_npz": str(
            result_root / f"{variant}_outdir" / "trajectory_mjwp_act.npz"
        ),
        "config_act": str(result_root / f"{variant}_outdir" / "config_act.yaml"),
        "video": str(result_root / f"{variant}.mp4"),
        "log": str(log_root / f"{variant}.log"),
        "cem_samples": os.environ["NUM_SAMPLES"],
        "cem_opt_steps": os.environ["NUM_ITERATIONS"],
        "execution_mode": os.environ["EXECUTION_MODE"],
    }
)
manifest = Path(os.environ["MANIFEST"])
manifest.parent.mkdir(parents=True, exist_ok=True)
with manifest.open("w", encoding="utf-8", newline="") as stream:
    writer = csv.DictWriter(
        stream, fieldnames=fields, delimiter="\t", lineterminator="\n"
    )
    writer.writeheader()
    writer.writerow(row)
PY

STARTED_AT="$(date -Is)"
START_EPOCH="$(date +%s)"
set +e
CUDA_VISIBLE_DEVICES="$GPU_ID" \
MUJOCO_GL="${MUJOCO_GL:-egl}" \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
/usr/bin/time -p -o "${ENV_ROOT}/wall_time.txt" \
  "$PYTHON_BIN" "$RUNNER" \
    --mode "$RUNNER_MODE" \
    --manifest-tsv "$MANIFEST" \
    --python-bin "$PYTHON_BIN" \
    --gpu-id "$GPU_ID" \
    --all \
    --num-samples "$NUM_SAMPLES" \
    --max-num-iterations "$NUM_ITERATIONS" \
    > "${ENV_ROOT}/runner.log" 2>&1
RUNNER_RC=$?
set -e
ENDED_AT="$(date -Is)"
END_EPOCH="$(date +%s)"

nvidia-smi \
  --query-gpu=index,name,uuid,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw \
  --format=csv,noheader,nounits > "${ENV_ROOT}/gpu_after.csv"

VALIDATION_RC=0
"$PYTHON_BIN" "$RUNTIME_VALIDATOR" "$RUNNER_MODE" \
  --manifest "$MANIFEST" \
  --output "$RUNTIME_GATE" \
  --expected-rows 1 \
  --experiment-id E178-local-speed-probe || VALIDATION_RC=$?

TIMING_RC=0
"$PYTHON_BIN" "$THROUGHPUT_VALIDATOR" \
  --manifest "$MANIFEST" \
  --output "$TIMING_JSON" \
  --threshold-s 1000000 \
  --expected-rows 1 \
  --experiment-id E178-local-speed-probe || TIMING_RC=$?

PROBE_ID="$PROBE_ID" PROBE_MODE="$PROBE_MODE" MANIFEST="$MANIFEST" \
ENV_ROOT="$ENV_ROOT" RESULT_ROOT="$RESULT_ROOT" SUMMARY_JSON="$SUMMARY_JSON" \
RUNTIME_GATE="$RUNTIME_GATE" TIMING_JSON="$TIMING_JSON" \
NUM_SAMPLES="$NUM_SAMPLES" NUM_ITERATIONS="$NUM_ITERATIONS" \
GPU_ID="$GPU_ID" STARTED_AT="$STARTED_AT" ENDED_AT="$ENDED_AT" \
START_EPOCH="$START_EPOCH" END_EPOCH="$END_EPOCH" RUNNER_RC="$RUNNER_RC" \
VALIDATION_RC="$VALIDATION_RC" TIMING_RC="$TIMING_RC" \
"$PYTHON_BIN" - <<'PY'
import hashlib
import json
import os
import subprocess
from pathlib import Path

def digest(path: Path) -> str:
    value = hashlib.sha256()
    value.update(path.read_bytes())
    return value.hexdigest()

env_root = Path(os.environ["ENV_ROOT"])
manifest = Path(os.environ["MANIFEST"])
runtime = json.loads(Path(os.environ["RUNTIME_GATE"]).read_text())
timing = json.loads(Path(os.environ["TIMING_JSON"]).read_text())
row = timing["rows"][0] if timing.get("rows") else {}
local_median = row.get("median_plan_time_s")
a100_canary_median = 2.9799
payload = {
    "probe_id": os.environ["PROBE_ID"],
    "probe_mode": os.environ["PROBE_MODE"],
    "status": (
        "pass"
        if int(os.environ["RUNNER_RC"]) == 0
        and int(os.environ["VALIDATION_RC"]) == 0
        and int(os.environ["TIMING_RC"]) == 0
        else "fail"
    ),
    "started_at": os.environ["STARTED_AT"],
    "ended_at": os.environ["ENDED_AT"],
    "wall_time_s": int(os.environ["END_EPOCH"]) - int(os.environ["START_EPOCH"]),
    "gpu_id": os.environ["GPU_ID"],
    "num_samples": int(os.environ["NUM_SAMPLES"]),
    "max_num_iterations": int(os.environ["NUM_ITERATIONS"]),
    "manifest": str(manifest),
    "manifest_sha256": digest(manifest),
    "local_git_head": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip(),
    "gpu_before": (env_root / "gpu_before.csv").read_text().strip(),
    "compute_before": (env_root / "compute_before.csv").read_text().strip(),
    "runtime_gate": runtime,
    "timing": timing,
    "a100_canary_median_plan_time_s": a100_canary_median,
    "a100_to_local_median_speed_ratio": (
        a100_canary_median / local_median
        if local_median and os.environ["PROBE_MODE"] == "canary"
        else None
    ),
    "return_codes": {
        "runner": int(os.environ["RUNNER_RC"]),
        "runtime_validator": int(os.environ["VALIDATION_RC"]),
        "timing_parser": int(os.environ["TIMING_RC"]),
    },
}
Path(os.environ["SUMMARY_JSON"]).write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps({
    "status": payload["status"],
    "probe_id": payload["probe_id"],
    "wall_time_s": payload["wall_time_s"],
    "median_plan_time_s": local_median,
    "a100_to_local_median_speed_ratio": payload[
        "a100_to_local_median_speed_ratio"
    ],
    "summary": os.environ["SUMMARY_JSON"],
}, sort_keys=True))
PY

if [ "$RUNNER_RC" -ne 0 ] || [ "$VALIDATION_RC" -ne 0 ] || [ "$TIMING_RC" -ne 0 ]; then
  exit 1
fi
