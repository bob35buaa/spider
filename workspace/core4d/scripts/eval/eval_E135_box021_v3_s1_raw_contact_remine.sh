#!/usr/bin/env bash
# E135: bounded Box021 v3 S1 raw-contact remine.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]:-$0}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
CORE4D_RAW_ROOT="${CORE4D_RAW_ROOT:-/mnt/a0ccc676-9496-49f8-a861-f8a1797dec52/mocap_data/CORE4D/CORE4D_Real}"
OUT_ROOT="${E135_OUT_ROOT:-${REPO_ROOT}/workspace/core4d/results/E135/box021_v3_s1_raw_contact_remine}"
INV_DIR="${OUT_ROOT}/s1_raw_contact/inventory"
RAW_DIR="${OUT_ROOT}/s1_raw_contact/raw_contact"
REG_DIR="${OUT_ROOT}/registries_combined_5cm_then_3cm"
REG3_DIR="${OUT_ROOT}/registries_3cm_only"
REG5_DIR="${OUT_ROOT}/registries_5cm_only"
TARGET_INVENTORY_TSV="${INV_DIR}/inventory_box021_bounded.tsv"
TARGET_CASE_IDS="box021_20231011_035_p1,box021_20231011_035_p2,box021_20231018_029_p1,box021_20231018_029_p2"

if [ ! -d "${CORE4D_RAW_ROOT}" ]; then
    echo "ERROR: CORE4D raw root not found: ${CORE4D_RAW_ROOT}" >&2
    exit 1
fi

mkdir -p "${INV_DIR}" "${RAW_DIR}" "${REG_DIR}" "${REG3_DIR}" "${REG5_DIR}" "${OUT_ROOT}/logs"

"${PYTHON_BIN}" "${REPO_ROOT}/workspace/core4d/scripts/data_construction_v3/stages/s1_raw_contact/build_inventory.py" \
    --core4d-raw-root "${CORE4D_RAW_ROOT}" \
    --spider-repo "${REPO_ROOT}" \
    --out-dir "${INV_DIR}" \
    | tee "${OUT_ROOT}/logs/build_inventory_stdout.log"

awk -v ids="${TARGET_CASE_IDS}" '
BEGIN {
    FS = OFS = "\t"
    split(ids, arr, ",")
    for (i in arr) {
        keep[arr[i]] = 1
    }
}
NR == 1 || keep[$1]
' "${INV_DIR}/inventory.tsv" > "${TARGET_INVENTORY_TSV}"

TARGET_COUNT="$(awk 'NR > 1 { n++ } END { print n + 0 }' "${TARGET_INVENTORY_TSV}")"
if [ "${TARGET_COUNT}" -ne 4 ]; then
    echo "ERROR: expected 4 bounded inventory rows, found ${TARGET_COUNT}: ${TARGET_INVENTORY_TSV}" >&2
    exit 1
fi

"${PYTHON_BIN}" "${REPO_ROOT}/workspace/core4d/scripts/data_construction_v3/stages/s1_raw_contact/run_raw_contact.py" \
    --core4d-raw-root "${CORE4D_RAW_ROOT}" \
    --inventory-tsv "${TARGET_INVENTORY_TSV}" \
    --out-dir "${RAW_DIR}" \
    --queue object-key \
    --object-keys box021 \
    --thresholds-m 0.03,0.05 \
    --sample-count "${E135_SAMPLE_COUNT:-12000}" \
    --seed "${E135_SEED:-203}" \
    | tee "${OUT_ROOT}/logs/run_raw_contact_stdout.json"

for registry_dir in "${REG_DIR}" "${REG3_DIR}" "${REG5_DIR}"; do
    "${PYTHON_BIN}" "${REPO_ROOT}/workspace/core4d/scripts/data_construction_v3/state/update_case_state_registry.py" \
        --registry-dir "${registry_dir}" \
        --from-inventory-tsv "${TARGET_INVENTORY_TSV}" \
        --evidence-root "${INV_DIR}" \
        --source-ref E135_S1_inventory_box021_bounded
done

"${PYTHON_BIN}" "${REPO_ROOT}/workspace/core4d/scripts/data_construction_v3/state/update_case_state_registry.py" \
    --registry-dir "${REG_DIR}" \
    --from-raw-contact-tsv "${RAW_DIR}/raw_contact_candidates_5cm.tsv" \
    --raw-contact-label 5cm \
    --evidence-root "${RAW_DIR}" \
    --source-ref E135_S1_raw_contact_5cm

"${PYTHON_BIN}" "${REPO_ROOT}/workspace/core4d/scripts/data_construction_v3/state/update_case_state_registry.py" \
    --registry-dir "${REG_DIR}" \
    --from-raw-contact-tsv "${RAW_DIR}/raw_contact_candidates_3cm.tsv" \
    --raw-contact-label 3cm \
    --evidence-root "${RAW_DIR}" \
    --source-ref E135_S1_raw_contact_3cm

"${PYTHON_BIN}" "${REPO_ROOT}/workspace/core4d/scripts/data_construction_v3/state/update_case_state_registry.py" \
    --registry-dir "${REG3_DIR}" \
    --from-raw-contact-tsv "${RAW_DIR}/raw_contact_candidates_3cm.tsv" \
    --raw-contact-label 3cm \
    --evidence-root "${RAW_DIR}" \
    --source-ref E135_S1_raw_contact_3cm

"${PYTHON_BIN}" "${REPO_ROOT}/workspace/core4d/scripts/data_construction_v3/state/update_case_state_registry.py" \
    --registry-dir "${REG5_DIR}" \
    --from-raw-contact-tsv "${RAW_DIR}/raw_contact_candidates_5cm.tsv" \
    --raw-contact-label 5cm \
    --evidence-root "${RAW_DIR}" \
    --source-ref E135_S1_raw_contact_5cm

"${PYTHON_BIN}" "${REPO_ROOT}/workspace/core4d/scripts/E135/summarize_box021_v3_s1_raw_contact_remine.py" \
    --out-root "${OUT_ROOT}" \
    | tee "${OUT_ROOT}/logs/summarize_stdout.json"
