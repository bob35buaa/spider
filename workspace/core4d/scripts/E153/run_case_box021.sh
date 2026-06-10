#!/usr/bin/env bash
# E153 case box021_029_p2 — gateA_b1 gate-threshold sweep (3 min_sdf × 2 max_viol).
# Default GPU: local-gpu0. Usage: bash run_case_box021.sh [gpu] [stage]
#   gpu   default 0     stage default full (smoke|full)
# Requires Stage0 decouple (plan 161 §2): config field cem_hand_gate_hard_floor_m.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

export STAGE="${2:-full}"
CASE="box021_029_p2"
TASK="d003_box021_20231018_029_p2_e107_clean"
OVERRIDE="core4d_E152_box021_029_p2_gateA_b1"
export CASE TASK OVERRIDE

source workspace/core4d/scripts/E153/_sweep_lib.sh
e153_run_case "${1:-0}"
