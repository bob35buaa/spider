#!/usr/bin/env bash
# E201 · L2/L3-review VLM pre-screen pipeline (Qwen3-VL-235B via ecodata2).
#
#   select L2/L3-review  ->  render 2fps frames (<=32)  ->  build request JSONL
#   ->  call_api_imitate_redaccel (BLACK BOX, path configurable)  ->  parse verdicts
#
# Usage:
#   bash workspace/core4d/scripts/eval/wrappers/run_E201_vlm_review.sh --exp E199
#
# Env / flags (API path is NOT hardcoded — pass it in):
#   ECODATA_CALL_API   path to call_api_imitate_redaccel.py  (or --call-api-path)
#   ECODATA_PYTHON     python that has the ecodata2 lib installed (default: python)
#   MODELS             comma list of models (default Qwen3-VL-235B-A22B-Instruct,gemini-3.5-flash-huangxiaoshuang)
#   NUM_PROC           default 16
#   IMAGE_SIZE         default 1024 (frames rendered large; API downsizes)
#   MAX_FRAMES         default 32
#   FPS                default 5
#   RENDER_PROC        parallel render workers (default 8)
#   STAGES             comma list to run subset: select,render,build,call,parse (default all)
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"

EXP="E199"
CALL_API="${ECODATA_CALL_API:-/mnt/ali-sh-1/usr/xiayibo/ecodata_history/ecodata2_xiayb_rednote_deploy/ecodata2/ecodata/inference_engine/call_api_imitate_redaccel.py}"
ECODATA_PY="${ECODATA_PYTHON:-python}"
MODELS="${MODELS:-Qwen3-VL-235B-A22B-Instruct,gemini-3.5-flash-huangxiaoshuang}"
NUM_PROC="${NUM_PROC:-16}"
IMAGE_SIZE="${IMAGE_SIZE:-1024}"
MAX_FRAMES="${MAX_FRAMES:-32}"
FPS="${FPS:-5}"
RENDER_PROC="${RENDER_PROC:-24}"   # software (llvmpipe) render is CPU-bound; box has 192 cores
STAGES="${STAGES:-select,render,build,call,parse}"
PY="${PYTHON_BIN:-.venv/bin/python}"
# No NVIDIA EGL ICD here -> MuJoCo EGL falls back to llvmpipe software render.
# Cap llvmpipe threads per shard so many shards don't oversubscribe the cores.
export LP_NUM_THREADS="${LP_NUM_THREADS:-4}"
export EGL_LOG_LEVEL="${EGL_LOG_LEVEL:-fatal}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --exp) EXP="$2"; shift 2;;
    --render-proc) RENDER_PROC="$2"; shift 2;;
    --call-api-path) CALL_API="$2"; shift 2;;
    --models) MODELS="$2"; shift 2;;
    --num-proc) NUM_PROC="$2"; shift 2;;
    --image-size) IMAGE_SIZE="$2"; shift 2;;
    --max-frames) MAX_FRAMES="$2"; shift 2;;
    --fps) FPS="$2"; shift 2;;
    --stages) STAGES="$2"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

E201=workspace/core4d/scripts/experiments/E201
VLM=workspace/core4d/results/E201/vlm_review
has(){ [[ ",$STAGES," == *",$1,"* ]]; }

if has select; then
  echo "== [select] L2/L3-review queue =="
  "$PY" "$E201/select_review_queue.py" --exp "$EXP" || exit 1
fi
if has render; then
  echo "== [render] ${FPS}fps frames (<=$MAX_FRAMES) + full mp4, ${RENDER_PROC} shards =="
  pids=()
  for ((s=0; s<RENDER_PROC; s++)); do
    "$PY" "$E201/render_frames_for_vlm.py" --exp "$EXP" --fps "$FPS" --max-frames "$MAX_FRAMES" \
      --shard "$s" --num-shards "$RENDER_PROC" >"/tmp/e201_render_s${s}.log" 2>&1 &
    pids+=($!)
  done
  rc=0
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  echo "== [render] merging shard indices =="
  "$PY" - "$EXP" "$RENDER_PROC" <<'PYMERGE'
import json, sys
from pathlib import Path
exp, n = sys.argv[1], int(sys.argv[2])
root = Path("workspace/core4d/results/E201/vlm_review/frames") / exp
merged = {}
for s in range(n):
    p = root / f"_index.shard{s}.json"
    if p.is_file():
        merged.update(json.loads(p.read_text()))
(root / "_index.json").write_text(json.dumps(merged, ensure_ascii=False, indent=2))
ok = sum(1 for v in merged.values() if "error" not in v)
print(f"[merge] {ok}/{len(merged)} ok -> {root}/_index.json")
PYMERGE
  [[ $rc -eq 0 ]] || { echo "[render] a shard failed, see /tmp/e201_render_s*.log" >&2; exit 1; }
fi
if has build; then
  echo "== [build] request JSONL =="
  "$PY" "$E201/build_vlm_requests.py" --exp "$EXP" || exit 1
fi
IFS=',' read -ra MODEL_ARR <<< "$MODELS"

if has call; then
  [[ -f "$CALL_API" ]] || { echo "call_api not found: $CALL_API (set ECODATA_CALL_API/--call-api-path)" >&2; exit 1; }
  for m in "${MODEL_ARR[@]}"; do
    echo "== [call] $m via $CALL_API (black box), image_size=$IMAGE_SIZE =="
    "$ECODATA_PY" "$CALL_API" \
      --data_path "$VLM/requests/$EXP.jsonl" \
      --output_dir "$VLM/out/$EXP/$m" \
      --model_name "$m" \
      --num_proc "$NUM_PROC" \
      --image_size "$IMAGE_SIZE" \
      --tqdm_desc "E201-$EXP-$m" || exit 1
  done
fi
if has parse; then
  for m in "${MODEL_ARR[@]}"; do
    echo "== [parse] $m verdicts + validate vs human =="
    "$PY" "$E201/parse_vlm_verdicts.py" --exp "$EXP" \
      --predictions "$VLM/out/$EXP/$m/0/generate_predictions.jsonl" \
      --out "$VLM/verdicts/${EXP}_${m}_vlm_verdicts.tsv" || exit 1
  done
fi
echo "== [ok] E201 VLM review done for $EXP (models: $MODELS) =="
