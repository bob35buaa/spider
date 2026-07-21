#!/usr/bin/env bash
# E167: export z-only CEM/postprocess rows into Holosoma/SUGAR refiner inputs.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
HOLOSOMA_REPO="${HOLOSOMA_REPO:-/home/ubuntu/Workspace/holosoma}"
SUGAR_REPO="${SUGAR_REPO:-/home/ubuntu/Workspace/Loco-Manipulation/SUGAR}"
RETARGET_PYTHON="${RETARGET_PYTHON:-/home/ubuntu/.holosoma_deps/miniconda3/envs/hsretargeting/bin/python}"

E167_RL_ROOT="${E167_RL_ROOT:-workspace/core4d/results/E167/holosoma_zonly/rl_export}"
HOLOSOMA_OUT="${HOLOSOMA_OUT:-/home/ubuntu/Workspace/holosoma/workspace/v3/data/R174_E167_zonly_7case_3arm_rl}"
SUGAR_DATA_ROOT="${SUGAR_DATA_ROOT:-/home/ubuntu/Workspace/Loco-Manipulation/SUGAR/data}"
E167_EXPORT_ARMS="${E167_EXPORT_ARMS:-E167A,E167A_B1,E167A_B2}"

"${PYTHON_BIN}" workspace/core4d/scripts/experiments/E167/export_zonly_rl_handoff.py \
  --out-root "${E167_RL_ROOT}" \
  --arms "${E167_EXPORT_ARMS}" \
  --source-ref "E167_zonly_${E167_EXPORT_ARMS//,/_}_7case" \
  --allow-spider-gate-fail

"${PYTHON_BIN}" "${HOLOSOMA_REPO}/workspace/v3/scripts/data/export_rl_motion_from_spider_tsv.py" \
  --input-tsv "${E167_RL_ROOT}/s6_downstream/rl_export/rl_export_input.tsv" \
  --all-ready \
  --target-source cem \
  --partner-source none \
  --include-contact-mask \
  --force \
  --out-dir "${HOLOSOMA_OUT}" \
  --python "${RETARGET_PYTHON}"

"${PYTHON_BIN}" "${SUGAR_REPO}/scripts/data_preprocess/convert_core4d_e167_manifest_to_sugar.py" \
  --manifest "${HOLOSOMA_OUT}/manifest.tsv" \
  --output-root "${SUGAR_DATA_ROOT}"

"${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import numpy as np

root = Path("/home/ubuntu/Workspace/Loco-Manipulation/SUGAR/data")
folders = sorted(p for p in root.glob("Core4D_E167_*") if p.is_dir())
sugar_python = Path("/home/ubuntu/miniconda3/envs/sugar/bin/python")
if not sugar_python.is_file():
    print(f"[warn] SUGAR python not found, skip pkl reserialize: {sugar_python}", file=sys.stderr)
    raise SystemExit(0)
for folder in folders:
    pkl_path = folder / "data_000/obj_motion_global_50hz.pkl"
    if not pkl_path.is_file():
        continue
    with pkl_path.open("rb") as f:
        obj = pickle.load(f)
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    np.savez(tmp_path, **obj)
    subprocess.run([str(sugar_python), "-c", r'''
import pickle
import sys
from pathlib import Path
import numpy as np
npz_path = Path(sys.argv[1])
pkl_path = Path(sys.argv[2])
with np.load(npz_path, allow_pickle=False) as data:
    obj = {key: data[key] for key in data.files}
with pkl_path.open("wb") as f:
    pickle.dump(obj, f)
npz_path.unlink()
''', str(tmp_path), str(pkl_path)], check=True)
print("reserialized E167 SUGAR pkl:", pkl_path)
PY

"${PYTHON_BIN}" - <<'PY'
from pathlib import Path
root = Path("/home/ubuntu/Workspace/Loco-Manipulation/SUGAR/data")
folders = sorted(p for p in root.glob("Core4D_E167_*") if p.is_dir())
missing = []
for folder in folders:
    data = folder / "data_000"
    for name in ("robot_50hz.npz", "obj_motion_global_50hz.pkl", "contact_labels_50hz.npy"):
        if not (data / name).is_file():
            missing.append(str(data / name))
if len(folders) != 21:
    raise SystemExit(f"expected 21 E167 SUGAR folders, got {len(folders)}")
if missing:
    raise SystemExit("missing SUGAR data files:\n" + "\n".join(missing))
print("E167 SUGAR data ready:", len(folders), "folders")
PY
