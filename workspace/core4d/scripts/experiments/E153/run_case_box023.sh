#!/usr/bin/env bash
# E153 case box023_person2 — gateA_b1 gate-threshold sweep (3 min_sdf × 2 max_viol).
# Default GPU: remote-gpu1. Usage: bash run_case_box023.sh [gpu] [stage]
#   gpu   default 1     stage default full (smoke|full)
# Requires Stage0 decouple (plan 161 §2): config field cem_hand_gate_hard_floor_m.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

export STAGE="${2:-full}"
CASE="box023_person2"
TASK="box023_person2_legobj"
OVERRIDE="core4d_E152_box023_person2_gateA_b1"
export CASE TASK OVERRIDE

source workspace/core4d/scripts/E153/_sweep_lib.sh
e153_run_case "${1:-1}"
