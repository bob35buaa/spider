#!/usr/bin/env bash
# E166: export selected CEM/postprocess rows into Holosoma/SUGAR refiner inputs.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
PARTNER_PYTHON_BIN="${PARTNER_PYTHON_BIN:-.venv/bin/python}"
CORE4D_RAW_ROOT="${CORE4D_RAW_ROOT:-/mnt/a0ccc676-9496-49f8-a861-f8a1797dec52/mocap_data/CORE4D/CORE4D_Real}"
SMPLX_MODEL_DIR="${SMPLX_MODEL_DIR:-/mnt/a0ccc676-9496-49f8-a861-f8a1797dec52/mocap_data/smplx}"
HOLOSOMA_REPO="${HOLOSOMA_REPO:-/home/ubuntu/Workspace/holosoma}"
SUGAR_REPO="${SUGAR_REPO:-/home/ubuntu/Workspace/Loco-Manipulation/SUGAR}"

E166_RL_ROOT="${E166_RL_ROOT:-workspace/core4d/results/E166/foot_smooth_retarget/rl_export}"
E166_EXPORT_ARMS="${E166_EXPORT_ARMS:-A,A_B2_postSmooth}"
HOLOSOMA_OUT="${HOLOSOMA_OUT:-/home/ubuntu/Workspace/holosoma/workspace/v3/data/R172_E166_A_A_B2_postSmooth_threecase_rl}"
SUGAR_DATA_ROOT="${SUGAR_DATA_ROOT:-/home/ubuntu/Workspace/Loco-Manipulation/SUGAR/data}"

"${PYTHON_BIN}" workspace/core4d/scripts/experiments/E166/export_footSmooth_rl_handoff.py \
  --out-root "${E166_RL_ROOT}" \
  --arms "${E166_EXPORT_ARMS}" \
  --source-ref "E166_${E166_EXPORT_ARMS//,/_}_threecase" \
  --skip-partner \
  --core4d-raw-root "${CORE4D_RAW_ROOT}" \
  --smplx-model-dir "${SMPLX_MODEL_DIR}" \
  --holosoma-repo "${HOLOSOMA_REPO}" \
  --python-bin "${PARTNER_PYTHON_BIN}"

RETARGET_PYTHON="${RETARGET_PYTHON:-/home/ubuntu/.holosoma_deps/miniconda3/envs/hsretargeting/bin/python}"
"${PYTHON_BIN}" "${HOLOSOMA_REPO}/workspace/v3/scripts/data/export_rl_motion_from_spider_tsv.py" \
  --input-tsv "${E166_RL_ROOT}/s6_downstream/rl_export/rl_export_input.tsv" \
  --all-ready \
  --target-source cem \
  --partner-source none \
  --include-contact-mask \
  --force \
  --out-dir "${HOLOSOMA_OUT}" \
  --python "${RETARGET_PYTHON}"

"${PYTHON_BIN}" "${SUGAR_REPO}/scripts/data_preprocess/convert_core4d_e166_manifest_to_sugar.py" \
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
folders = [
    "Core4D_E166_A_Box021_035_p2",
    "Core4D_E166_A_B2_postSmooth_Box021_035_p2",
    "Core4D_E166_A_Box004_082_p1",
    "Core4D_E166_A_B2_postSmooth_Box004_082_p1",
    "Core4D_E166_A_Box004_083_p2",
    "Core4D_E166_A_B2_postSmooth_Box004_083_p2",
]
sugar_python = Path("/home/ubuntu/miniconda3/envs/sugar/bin/python")
if not sugar_python.is_file():
    print(f"[warn] SUGAR python not found, skip pkl reserialize: {sugar_python}", file=sys.stderr)
    raise SystemExit(0)

for folder in folders:
    pkl_path = root / folder / "data_000/obj_motion_global_50hz.pkl"
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
print("reserialized SUGAR pkl:", pkl_path)
PY

"${PYTHON_BIN}" - <<'PY'
from pathlib import Path
root = Path("/home/ubuntu/Workspace/Loco-Manipulation/SUGAR/data")
folders = [
    "Core4D_E166_A_Box021_035_p2",
    "Core4D_E166_A_B2_postSmooth_Box021_035_p2",
    "Core4D_E166_A_Box004_082_p1",
    "Core4D_E166_A_B2_postSmooth_Box004_082_p1",
    "Core4D_E166_A_Box004_083_p2",
    "Core4D_E166_A_B2_postSmooth_Box004_083_p2",
]
missing = []
for folder in folders:
    data = root / folder / "data_000"
    for name in ("robot_50hz.npz", "obj_motion_global_50hz.pkl", "contact_labels_50hz.npy"):
        if not (data / name).is_file():
            missing.append(str(data / name))
if missing:
    raise SystemExit("missing SUGAR data files:\n" + "\n".join(missing))
print("E166 SUGAR data ready:", len(folders), "folders")
PY
