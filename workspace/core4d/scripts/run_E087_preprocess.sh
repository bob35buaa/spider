#!/usr/bin/env bash
# E087 preprocessing: mass audit, mass variants, overrides.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

.venv/bin/python workspace/core4d/scripts/E087/mass_audit.py
.venv/bin/python workspace/core4d/scripts/E087/create_mass_variants.py
.venv/bin/python workspace/core4d/scripts/E087/generate_e087_overrides.py

echo "=== E087 preprocess done ==="

