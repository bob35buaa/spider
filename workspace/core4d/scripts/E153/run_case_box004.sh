#!/usr/bin/env bash
# E153 case box004_083_p2 — gateA_b1 gate-threshold sweep (3 min_sdf × 2 max_viol).
# Default GPU: remote-gpu0. Usage: bash run_case_box004.sh [gpu] [stage]
#   gpu   default 0     stage default full (smoke|full)
# Requires Stage0 decouple (plan 161 §2): config field cem_hand_gate_hard_floor_m.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

export STAGE="${2:-full}"
CASE="box004_083_p2"
TASK="e091_box004_20231003_2_083_p2_e092_dyn"
OVERRIDE="core4d_E152_box004_083_p2_gateA_b1"
export CASE TASK OVERRIDE

source workspace/core4d/scripts/E153/_sweep_lib.sh
e153_run_case "${1:-0}"
