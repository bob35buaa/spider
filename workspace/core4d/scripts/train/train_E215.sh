#!/usr/bin/env bash
# E215 data-build entrypoint (plan247): rot object-augmentation for bucket+box.
#
# Stages (run all, or a subset via STAGES="seed upstream build"):
#   rotfix    verify ../holosoma rot fix under the hsretargeting python (9/9)
#   seed      hardlink _original warm start from E199/E202 into the E215 tree
#   upstream  run pipeline.sh aug (omnirt_v2) -> rot_0/rot_1 npz
#   build     trim rot + build SPIDER task + per-arm CEM scene (+gravcomp)
#   gravcomp  re-verify the G1 gravcomp sidecars (+ tamper self-test)
#   baseline  audit same-arm orig baseline presence (report gaps)
#   snapshot  freeze every built aug task's scene XML into results/E215/scene_snapshot
#   manifest  write overrides + priority manifest + single-variable audit
#   freeze    freeze the manifest (only if the audit passed)
#
# Long steps (upstream, build) are CPU-heavy (~hours). Run in the background:
#   nohup bash workspace/core4d/scripts/train/train_E215.sh > logs/E215/train.log 2>&1 &
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO"

PY="${PY:-$REPO/.venv/bin/python}"
HSPY="${HSPY:-/mnt/ali-sh-1/dataset/zeus/xiayb/.holosoma_deps/miniconda3/envs/hsretargeting/bin/python}"
E215="workspace/core4d/scripts/experiments/E215"
ART="workspace/core4d/results/E215/data_preprocess/manifests/e215_rot_artifacts.tsv"
STAGES="${STAGES:-rotfix seed upstream build gravcomp baseline snapshot manifest freeze}"

run() { echo; echo "=== [E215] stage: $1 ==="; }
has() { [[ " $STAGES " == *" $1 "* ]]; }

mkdir -p logs/E215

if has rotfix; then
  run rotfix
  "$HSPY" "$E215/test_rotation_fix.py"
fi

if has seed; then
  run seed
  "$PY" "$E215/seed_warmstart.py"
fi

if has upstream; then
  run upstream
  # object-serial, up to 5 object groups in parallel; omnirt_v2; never --force
  "$PY" "$E215/run_upstream_retarget.py" --max-workers 5
fi

if has build; then
  run build
  "$PY" "$E215/build_augmented_tasks.py"
fi

if has gravcomp; then
  run gravcomp
  "$PY" "$E215/build_gravcomp_sidecars.py"
fi

if has baseline; then
  run baseline
  MUJOCO_GL=disable "$PY" "$E215/preflight_baseline_audit.py" || true
fi

if has snapshot; then
  run snapshot
  # snapshot every built aug task's scene XML (rule 7/10b) BEFORE any CEM
  if [[ -f "$ART" ]]; then
    mapfile -t TASKS < <("$PY" - "$ART" <<'PYEOF'
import csv, sys
seen = []
with open(sys.argv[1], newline="") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        if r.get("status", "").startswith("built") and r.get("target_task") and r["target_task"] not in seen:
            seen.append(r["target_task"])
print("\n".join(seen))
PYEOF
)
    if [[ ${#TASKS[@]} -gt 0 ]]; then
      bash workspace/core4d/scripts/convert/snapshot_scenes.sh E215 "${TASKS[@]}"
    else
      echo "  no built aug tasks to snapshot yet"
    fi
  else
    echo "  no artifacts TSV yet ($ART); skipping snapshot"
  fi
fi

if has manifest; then
  run manifest
  "$PY" "$E215/build_aug_manifest.py"
fi

if has freeze; then
  run freeze
  "$PY" "$E215/build_aug_manifest.py" --freeze
fi

echo; echo "=== [E215] train_E215.sh done (stages: $STAGES) ==="
