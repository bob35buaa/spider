#!/usr/bin/env bash
# Pull E215 remote CEM results + logs back to the local workspace.
#
# If the remote shares this /mnt volume (the E213 setup), it writes the SAME
# manifest/result dirs directly and no pull is needed -- this rsync is then a
# near no-op. For a separate-filesystem remote, it copies the rollouts and logs
# home. It never touches the frozen manifest (results are keyed by variant_id).
#
# Usage: bash workspace/core4d/scripts/launch/active/pull_E215_remote_results.sh
# Env: REMOTE (ssh host), REMOTE_ROOT (repo path on remote)
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

REMOTE="${REMOTE:-spider-remote}"
REMOTE_ROOT="${REMOTE_ROOT:-/mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/spider}"
CEM="workspace/core4d/results/E215/s6_downstream/cem/full"
LOGS="logs/E215/cem/full"

mkdir -p "$CEM" "$LOGS"
rsync -az --ignore-missing-args "${REMOTE}:${REMOTE_ROOT}/${CEM}/" "${CEM}/"
rsync -az --ignore-missing-args "${REMOTE}:${REMOTE_ROOT}/${LOGS}/" "${LOGS}/"
echo "Pulled E215 CEM results + logs from ${REMOTE}"
echo "Now re-run run_e215_cem.py locally to fold the pulled rollouts into the manifest (resume marks them complete)."
