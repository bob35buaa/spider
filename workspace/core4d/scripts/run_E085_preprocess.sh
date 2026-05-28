#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

python workspace/core4d/scripts/E085/generate_raw_contact_targets.py "$@"
