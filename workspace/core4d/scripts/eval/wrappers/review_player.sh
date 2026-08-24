#!/usr/bin/env bash
# Launch the PRG retargeting review player (viser, E170-E200).
#
# Usage:
#   bash workspace/core4d/scripts/eval/wrappers/review_player.sh [E194|E197|E198|E199|E199P|E200N|E200G] [--port 8080] [--check] [...]
# E197 = partner_reexport_v2 RL-export motions (52 target cases, log277): kinematic
#   playback of the 43-dim g1_expansion qpos against each case's source scene.xml.
#   Visual review only — no numeric metrics/gates (this is a kinematic export).
#   bash workspace/core4d/scripts/eval/wrappers/review_player.sh E199,E199P [--port 8080]   # both in one session
#   bash workspace/core4d/scripts/eval/wrappers/review_player.sh E200N,E200G [--port 8080]  # both E200 arms
# E203 = CORE4D v2 human-motion orig retarget + CEM, P1 (box001/box024, 54 cases).
#   Single-arm live-qpos playback of s6_downstream/eval_P1/e203_case_metrics.tsv
#   (built by workspace/core4d/scripts/experiments/E203/run_E203_p1_eval.py).
#   bash workspace/core4d/scripts/eval/wrappers/review_player.sh E203 [--port 8080]
# E204ARM = three-arm ablation on E178's 27 bucket cases (plan234/log292): a 3-arm
#   sweep per case — noPRG(E204) / PRG(E178 reused) / G1A2(E205) — same contactAlignedTop
#   proxy + omnirt_v1 trajectory, only the reward arm differs. Live-qpos playback; use
#   the 实验/arm filters to A/B/C compare. TSV built by build_arm_review_tsv.py (no re-score).
#   bash workspace/core4d/scripts/eval/wrappers/review_player.sh E204ARM [--port 8080]
# E200N = E200 noPRG (E167A) arm (plan230/log289): orig(E190 37) + trans0/1/2 arm sweep.
# E200G = E200 PRG+G1+A2 arm: orig(E198 G1A2 87) + trans0/1/2 arm sweep.
#   The prg_g1a2 CEM re-ran aug only, so orig is pulled inline from the E198
#   factorial G1A2 arm (read-only, no eval re-run) and shown as the orig arm.
#   Both reuse the E199 aug trajectories; live qpos playback. Same join as E199.
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
