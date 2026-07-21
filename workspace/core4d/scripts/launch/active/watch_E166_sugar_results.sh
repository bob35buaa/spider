#!/usr/bin/env bash
set -euo pipefail

SUGAR_ROOT="${SUGAR_ROOT:-/home/ubuntu/Workspace/Loco-Manipulation/SUGAR}"
POLL_INTERVAL="${POLL_INTERVAL:-600}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-43200}"
OUT_REL="outputs/core4d/e166_foot_smooth_refiner_rl"
PULL_SCRIPT="${SUGAR_ROOT}/scripts/sugar_rl/pull_core4d_e166_refiner_remote.sh"
SUMMARY_SCRIPT="${SUGAR_ROOT}/scripts/sugar_rl/summarize_core4d_e166_failed_windows.py"

started="$(date +%s)"

pull_remote() {
  if [ -x "${PULL_SCRIPT}" ]; then
    bash "${PULL_SCRIPT}" || true
  fi
}

count_done() {
  python - "$SUGAR_ROOT" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1]) / "outputs/core4d/e166_foot_smooth_refiner_rl"
rows = [
    ("box021_035_p2", "A", "box021_035_p2_E166_A_6000"),
    ("box021_035_p2", "A_B2_postSmooth", "box021_035_p2_E166_A_B2_postSmooth_6000"),
    ("box004_082_p1", "A", "box004_082_p1_E166_A_6000"),
    ("box004_082_p1", "A_B2_postSmooth", "box004_082_p1_E166_A_B2_postSmooth_6000"),
    ("box004_083_p2", "A", "box004_083_p2_E166_A_6000"),
    ("box004_083_p2", "A_B2_postSmooth", "box004_083_p2_E166_A_B2_postSmooth_6000"),
]
ckpt = 0
csv = 0
for case, arm, run in rows:
    if (root / case / arm / "train/logs" / run / "model_5999.pt").is_file():
        ckpt += 1
    if (root / case / arm / "eval_staggered_phase_mw30/analysis/success_vs_phase.csv").is_file():
        csv += 1
print(f"{ckpt} {csv}")
PY
}

while true; do
  now="$(date +%s)"
  elapsed="$((now - started))"
  if [ "${elapsed}" -gt "${MAX_WAIT_SECONDS}" ]; then
    echo "[watch] timeout after ${elapsed}s"
    exit 3
  fi

  pull_remote
  read -r ckpt_count csv_count < <(count_done)
  echo "[watch] $(date '+%F %T') final_ckpt=${ckpt_count}/6 success_csv=${csv_count}/6 elapsed=${elapsed}s"

  if [ "${ckpt_count}" = "6" ] && [ "${csv_count}" = "6" ]; then
    python "${SUMMARY_SCRIPT}"
    echo "[watch] complete: ${SUGAR_ROOT}/${OUT_REL}/comparison"
    exit 0
  fi

  sleep "${POLL_INTERVAL}"
done
