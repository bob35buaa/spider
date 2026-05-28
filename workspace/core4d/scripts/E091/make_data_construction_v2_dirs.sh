#!/usr/bin/env bash
# Create the Holosoma E091 data_construction_v2 workspace.
set -euo pipefail

V2_ROOT="${V2_ROOT:-/home/ubuntu/Workspace/holosoma/workspace/v3/data_construction_v2}"

mkdir -p \
  "$V2_ROOT/inputs" \
  "$V2_ROOT/scripts" \
  "$V2_ROOT/results/medium_manifest" \
  "$V2_ROOT/results/raw_contact_visuals" \
  "$V2_ROOT/results/template_preflight" \
  "$V2_ROOT/results/d005b_g1_feasibility" \
  "$V2_ROOT/visualizations/dashboard" \
  "$V2_ROOT/visualizations/raw_contact" \
  "$V2_ROOT/visualizations/g1_feasibility" \
  "$V2_ROOT/logs" \
  "$V2_ROOT/reports"

printf "data_construction_v2 root: %s\n" "$V2_ROOT"
