#!/usr/bin/env bash
# Launch the PRG retargeting review player (viser, E170-E199).
#
# Usage:
#   bash workspace/core4d/scripts/eval/wrappers/review_player.sh [E194|E198|E199|E199P] [--port 8080] [--check] [...]
#   bash workspace/core4d/scripts/eval/wrappers/review_player.sh E199,E199P [--port 8080]   # both in one session
# E199 = full-scale box translation augmentation (plan229/log287): 332 rows
# (83 cases x orig+trans0/1/2 as a 4-arm sweep), live qpos playback. `review_player.sh E199`.
# E199P = the E199 pilot (plan228/log286): 8 objects (5 box + bucket003/004/007),
# 31 rows (orig + trans0/1/2 per case). Only E199 source with the bucket cases.
# E194 is the 72-case G1 expansion only (box001/box023/box021), not the older
# E194 15-case arm sweep or its PRG authority rows. Its review index overlays
# only the 29 Euler-mismatch cases with E196 corrected G1 videos/metrics; the
# other 43 cases keep their original E194 videos. Existing annotations for the
# 29-case set are intentionally reset, while all other annotations are kept.
#
# --check runs a headless index audit (case counts vs summary.json) and exits.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
APP=workspace/core4d/scripts/eval/review/viser_review_player.py

# Tracking-error red lines, aligned with the E178 numeric gates.
export CORE4D_REVIEW_ROOT_POS_MAX_CM="${CORE4D_REVIEW_ROOT_POS_MAX_CM:-20}"
export CORE4D_REVIEW_ROOT_ORI_MAX_DEG="${CORE4D_REVIEW_ROOT_ORI_MAX_DEG:-20}"
export CORE4D_REVIEW_HAND_POS_MAX_CM="${CORE4D_REVIEW_HAND_POS_MAX_CM:-20}"
export CORE4D_REVIEW_HAND_ORI_MAX_DEG="${CORE4D_REVIEW_HAND_ORI_MAX_DEG:-20}"
export CORE4D_REVIEW_OBJECT_POS_MAX_CM="${CORE4D_REVIEW_OBJECT_POS_MAX_CM:-20}"
export CORE4D_REVIEW_OBJECT_ORI_MAX_DEG="${CORE4D_REVIEW_OBJECT_ORI_MAX_DEG:-10}"

# viser (and its trimesh dep) are listed in pyproject but not always installed.
if ! "$PYTHON_BIN" -c "import viser, trimesh" 2>/dev/null; then
  echo "[review] installing viser (+ trimesh) ..." >&2
  uv pip install viser trimesh || "$PYTHON_BIN" -m pip install viser trimesh
fi

# First arg may be a single exp id (E199) or a comma-separated list (E199,E199P)
# to load several experiments into one player session (filter by the 实验 dropdown).
if [[ "${1:-}" =~ ^E[0-9]+[A-Za-z_]*(,E[0-9]+[A-Za-z_]*)*$ ]]; then
  exp_id="$1"
  shift
  exec "$PYTHON_BIN" "$APP" --exps "$exp_id" "$@"
fi

exec "$PYTHON_BIN" "$APP" "$@"
