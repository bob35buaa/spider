#!/usr/bin/env bash
# E188 isolated A100 GPU4/5 deployment and two-worker launch.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODE="${1:-preflight}"
if [[ "$MODE" != "preflight" && "$MODE" != "canary" && "$MODE" != "full" ]]; then
  echo "usage: $0 {preflight|canary|full}" >&2
  exit 2
fi
STAGE="$([[ "$MODE" == full ]] && echo full || echo canary)"
PYTHON_BIN="${E188_PYTHON_BIN:-.venv/bin/python}"
DEPLOYER=workspace/core4d/scripts/experiments/E188/deploy_remote_a100.py
RUNNER=workspace/core4d/scripts/experiments/E188/run_full_queue.py
DEPLOYMENT="workspace/core4d/results/E188/s4_canary/deployment/remote_deployment_manifest.json"
[[ "$STAGE" == full ]] && DEPLOYMENT="workspace/core4d/results/E188/s5_full/deployment/remote_deployment_manifest.json"
HOST="${E188_A100_HOST:-61.172.170.106}"
PORT="${E188_A100_PORT:-30409}"
IDENTITY="${E188_A100_IDENTITY:-/home/ubuntu/.ssh/id_rsa_tianyiyun}"
REMOTE_PYTHON=/home/dataset-assist-0/xiayb/workspace/spider/.venv/bin/python
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=10 -p "$PORT" -i "$IDENTITY" "batchcom@$HOST")

if [[ "$MODE" == preflight || "$MODE" == full ]]; then
  PYTHONPATH=workspace/core4d/scripts/experiments/E188 "$PYTHON_BIN" "$DEPLOYER" overlap-policy --stage "$STAGE"
  PYTHONPATH=workspace/core4d/scripts/experiments/E188 "$PYTHON_BIN" "$DEPLOYER" --host "$HOST" --port "$PORT" --identity "$IDENTITY" freeze --stage "$STAGE"
  PYTHONPATH=workspace/core4d/scripts/experiments/E188 "$PYTHON_BIN" "$DEPLOYER" --host "$HOST" --port "$PORT" --identity "$IDENTITY" deploy --stage "$STAGE"
else
  test -f "$DEPLOYMENT"
  test -f workspace/core4d/results/E188/s0_environment/execution_manifest.json
fi
REMOTE_ROOT="$($PYTHON_BIN -c 'import json,sys; print(json.load(open(sys.argv[1]))["remote_root"])' "$DEPLOYMENT")"
if [[ "$REMOTE_ROOT" != /home/dataset-assist-0/xiayb/workspace/e188_spider_runs/e188_${STAGE}_*/spider ]]; then
  echo "unsafe E188 remote root: $REMOTE_ROOT" >&2
  exit 2
fi
REMOTE_PYTHONPATH="$REMOTE_ROOT:$REMOTE_ROOT/workspace/core4d/scripts/experiments/E188"
QUERY_TAPE_PATH="$("${SSH[@]}" "cd '$REMOTE_ROOT' && PYTHONPATH='$REMOTE_PYTHONPATH' '$REMOTE_PYTHON' -c 'import spider.query_tape; print(spider.query_tape.__file__)'")"
if [[ "$QUERY_TAPE_PATH" != "$REMOTE_ROOT/spider/query_tape.py" ]]; then
  echo "E188 remote runtime import resolved outside snapshot: $QUERY_TAPE_PATH" >&2
  exit 2
fi
for worker in a100-4 a100-5; do
  "${SSH[@]}" "cd '$REMOTE_ROOT' && PYTHONPATH='$REMOTE_PYTHONPATH' MUJOCO_GL=egl '$REMOTE_PYTHON' '$RUNNER' preflight --worker '$worker' --phase '$STAGE' --python-bin '$REMOTE_PYTHON' --device-id 0 --render-mode deferred-local"
done
if [[ "$MODE" == preflight ]]; then
  echo "E188_REMOTE_PREFLIGHT=PASS stage=$STAGE root=$REMOTE_ROOT"
  exit 0
fi

# 2026-08-05 explicit user authority: no memory/compute-process safety gate;
# E188 may overlap external workloads on physical GPU4/5.
for gpu in 4 5; do
  worker="a100-$gpu"
  session="e188_${STAGE}_a100_${gpu}_$(basename "$(dirname "$REMOTE_ROOT")")"
  log_dir="$REMOTE_ROOT/logs/E188"
  log="$log_dir/launcher_${STAGE}_${worker}.log"
  if "${SSH[@]}" "tmux has-session -t '$session'" >/dev/null 2>&1; then
    echo "E188 remote session already exists: $session"
    continue
  fi
  remote_command="cd '$REMOTE_ROOT' && CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH='$REMOTE_PYTHONPATH' MUJOCO_GL=egl '$REMOTE_PYTHON' '$RUNNER' run --worker '$worker' --phase '$STAGE' --python-bin '$REMOTE_PYTHON' --device-id 0 --physical-gpu '$gpu' --render-mode deferred-local > '$log' 2>&1"
  printf -v quoted '%q' "$remote_command"
  "${SSH[@]}" "mkdir -p '$log_dir' && tmux new-session -d -s '$session' -c '$REMOTE_ROOT' $quoted"
  echo "E188_REMOTE_LAUNCHED session=$session phase=$STAGE worker=$worker gpu=$gpu"
done
