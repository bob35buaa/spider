#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../../.."
uv run examples/run_mjwp.py \
    +override=core4d_box025_e009a \
    task=box025_person1 \
    data_id=0 \
    viewer=none
