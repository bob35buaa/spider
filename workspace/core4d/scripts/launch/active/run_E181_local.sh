#!/usr/bin/env bash
# E181 local stage launcher. It never modifies E178 results.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

STAGE="${1:-preflight}"
SCRIPT_ROOT="workspace/core4d/scripts/experiments/E181"
LOG_ROOT="logs/E181"
mkdir -p "$LOG_ROOT/build"

run_preflight() {
  uv run python "$SCRIPT_ROOT/build_authority.py" \
    2>&1 | tee "$LOG_ROOT/build/s0_authority.log"
  uv run python "$SCRIPT_ROOT/probe_environment.py" \
    --remote-host "${E181_REMOTE_HOST:-spider-remote}" \
    2>&1 | tee "$LOG_ROOT/build/s0_environment.log"
  uv run python "$SCRIPT_ROOT/test_preflight.py" \
    2>&1 | tee "$LOG_ROOT/build/s0_tests.log"
}

run_oracle() {
  uv run python "$SCRIPT_ROOT/build_oracle_meshes.py" \
    2>&1 | tee "$LOG_ROOT/build/s1_oracle.log"
  uv run python "$SCRIPT_ROOT/test_oracle_mesh.py" \
    2>&1 | tee "$LOG_ROOT/build/s1_oracle_tests.log"
}

run_oracle_review() {
  uv run python "$SCRIPT_ROOT/build_oracle_visual_review.py" \
    2>&1 | tee "$LOG_ROOT/build/s1_oracle_visual_review.log"
}

run_coacd() {
  local candidate_workers="${E181_COACD_CANDIDATE_WORKERS:-6}"
  local object_key
  for object_key in bucket003 bucket004 bucket007; do
    uv run python "$SCRIPT_ROOT/build_coacd_candidates.py" \
      --object "$object_key" \
      --parallel-workers "$candidate_workers" \
      2>&1 | tee "$LOG_ROOT/build/s2_coacd_${object_key}.log"
  done
  uv run python "$SCRIPT_ROOT/build_coacd_candidates.py" --summarize \
    2>&1 | tee "$LOG_ROOT/build/s2_coacd_summary.log"
  uv run python "$SCRIPT_ROOT/test_coacd_assets.py" \
    2>&1 | tee "$LOG_ROOT/build/s2_coacd_tests.log"
}

run_asset_eval() {
  uv run python "$SCRIPT_ROOT/evaluate_asset_fidelity.py" \
    2>&1 | tee "$LOG_ROOT/build/s2_asset_fidelity.log"
  set +e
  uv run python "$SCRIPT_ROOT/select_canonical_set.py" \
    2>&1 | tee "$LOG_ROOT/build/s2_canonical_selection.log"
  local selection_status="${PIPESTATUS[0]}"
  set -e
  if [[ "$selection_status" -eq 3 ]]; then
    echo "E181_ASSET_EVAL=ASSET_REJECTED downstream=S3_S6_NOT_AUTHORIZED"
    return 3
  fi
  return "$selection_status"
}

run_asset_visual() {
  # Post-terminal diagnostic only: reads rejected S2 assets and never creates C*.
  uv run python "$SCRIPT_ROOT/render_rejected_candidate_visuals.py" \
    2>&1 | tee "$LOG_ROOT/build/s2_rejected_candidate_visuals.log"
}

case "$STAGE" in
  preflight)
    run_preflight
    ;;
  oracle)
    run_oracle
    ;;
  oracle-review)
    run_oracle_review
    ;;
  coacd)
    run_coacd
    ;;
  asset-eval)
    run_asset_eval
    ;;
  asset-visual)
    run_asset_visual
    ;;
  grid|physics|canary|full)
    echo "E181 stage '$STAGE' is not implemented yet; run preflight first." >&2
    exit 2
    ;;
  *)
    echo "usage: $0 {preflight|oracle|oracle-review|coacd|asset-eval|asset-visual|grid|physics|canary|full}" >&2
    exit 2
    ;;
esac
