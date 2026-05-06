#!/usr/bin/env bash
# E008a: scene_forearm_act with decaying PD guidance (residual_gain_ratio=0)
set -euo pipefail
cd "$(dirname "$0")/../../../.."
uv run examples/run_mjwp.py \
    +override=core4d_box025_forearm_act \
    task=box025_person1 \
    data_id=0 \
    viewer=none
