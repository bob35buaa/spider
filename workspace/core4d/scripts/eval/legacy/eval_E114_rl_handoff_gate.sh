#!/usr/bin/env bash
# E114: build strict + contact-alignment RL handoff gate from E113 decisions.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

python3 workspace/core4d/scripts/E114/build_rl_handoff_gate.py "$@"
