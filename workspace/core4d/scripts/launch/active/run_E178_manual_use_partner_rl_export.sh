#!/usr/bin/env bash
# Export the final E178 manual-USE set with real opposite-person partners.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SPIDER_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
HOLOSOMA_ROOT="${HOLOSOMA_ROOT:-/home/ubuntu/Workspace/holosoma}"
HS_PYTHON="${HS_PYTHON:-/home/ubuntu/.holosoma_deps/miniconda3/envs/hsretargeting/bin/python}"
MODE="${1:-all}"

CORE_EXPORTER="${SPIDER_ROOT}/workspace/core4d/scripts/experiments/E178/export_manual_use_partner_rl.py"
RL_ROOT="${SPIDER_ROOT}/workspace/core4d/results/E178/s6_downstream/rl_export"
INPUT_TSV="${RL_ROOT}/rl_export_input.tsv"
PARTNER_TSV="${RL_ROOT}/partner_omnirt/rl_partner_omnirt_manifest.tsv"
HS_OUT="${HOLOSOMA_ROOT}/workspace/v3/data/E178_manual_use_partner_rl"
HS_EXPORTER="${HOLOSOMA_ROOT}/workspace/v3/scripts/data/export_rl_motion_from_spider_tsv.py"
HS_GATE="${HOLOSOMA_ROOT}/workspace/v3/scripts/registry/registry_gate.py"
HS_REGISTER="${HOLOSOMA_ROOT}/workspace/v3/scripts/registry/register_motion_manifest.py"
HS_BUILD_REGISTRY="${HOLOSOMA_ROOT}/workspace/v3/scripts/analysis/build_experiment_registry.py"

run_core_export() {
    cd "${SPIDER_ROOT}"
    python "${CORE_EXPORTER}"
}

case_args_for_object() {
    local object_key="$1"
    "${HS_PYTHON}" - "${INPUT_TSV}" "${object_key}" <<'PY'
import csv
import sys
with open(sys.argv[1], newline="", encoding="utf-8") as stream:
    for row in csv.DictReader(stream, delimiter="\t"):
        if row["object_key"] == sys.argv[2] and row["rl_export_decision"] == "RL_EXPORT_READY":
            print(row["case_id"])
PY
}

run_holosoma_export() {
    cd "${HOLOSOMA_ROOT}"
    test -x "${HS_PYTHON}"
    test -f "${INPUT_TSV}"
    test -f "${PARTNER_TSV}"

    mapfile -t all_cases < <("${HS_PYTHON}" - "${INPUT_TSV}" <<'PY'
import csv
import sys
with open(sys.argv[1], newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream, delimiter="\t"))
for row in rows:
    if row["rl_export_decision"] == "RL_EXPORT_READY":
        print(row["case_id"])
PY
    )
    if [ "${#all_cases[@]}" -ne 13 ]; then
        echo "Expected exactly 13 E178 ready cases, got ${#all_cases[@]}" >&2
        exit 2
    fi

    for case_id in "${all_cases[@]}"; do
        "${HS_PYTHON}" "${HS_GATE}" pre-export \
            --input-tsv "${INPUT_TSV}" \
            --partner-omnirt-tsv "${PARTNER_TSV}" \
            --case-id "${case_id}" \
            --target-source both
    done

    local object_key extents
    for object_key in bucket003 bucket004 bucket007; do
        case "${object_key}" in
            bucket003) extents="0.270393755 0.381407245 0.23287703" ;;
            bucket004) extents="0.161566625 0.231058755 0.15230013" ;;
            bucket007) extents="0.273275865 0.2869532 0.284911265" ;;
        esac
        mapfile -t object_cases < <(case_args_for_object "${object_key}")
        if [ "${#object_cases[@]}" -eq 0 ]; then
            echo "No ready case found for ${object_key}" >&2
            exit 2
        fi
        case_args=()
        for case_id in "${object_cases[@]}"; do
            case_args+=(--case-id "${case_id}")
        done
        read -r hx hy hz <<<"${extents}"
        out_dir="${HS_OUT}/${object_key}"

        "${HS_PYTHON}" "${HS_EXPORTER}" \
            --input-tsv "${INPUT_TSV}" \
            --partner-omnirt-tsv "${PARTNER_TSV}" \
            "${case_args[@]}" \
            --target-source both \
            --partner-source omnirt \
            --target-variant ref_fk \
            --out-dir "${out_dir}" \
            --object-half-extents "${hx}" "${hy}" "${hz}" \
            --include-contact-mask \
            --dry-run

        "${HS_PYTHON}" "${HS_EXPORTER}" \
            --input-tsv "${INPUT_TSV}" \
            --partner-omnirt-tsv "${PARTNER_TSV}" \
            "${case_args[@]}" \
            --target-source both \
            --partner-source omnirt \
            --target-variant ref_fk \
            --out-dir "${out_dir}" \
            --object-half-extents "${hx}" "${hy}" "${hz}" \
            --include-contact-mask \
            --force

        "${HS_PYTHON}" "${HS_REGISTER}" \
            --manifest "${out_dir}/manifest.tsv" \
            --owner-scope E178_manual_use_partner_rl \
            --notes "E178 final manual USE ${object_key}; CEM and trajectory targets with real E174 OmniRetarget partners and contact masks."
        "${HS_PYTHON}" "${HS_GATE}" post-export --manifest "${out_dir}/manifest.tsv"
    done

    "${HS_PYTHON}" "${HS_BUILD_REGISTRY}"
    for case_id in "${all_cases[@]}"; do
        "${HS_PYTHON}" "${HS_GATE}" pre-train --motion-id "M_${case_id}_spider_partner_omnirt"
        "${HS_PYTHON}" "${HS_GATE}" pre-train --motion-id "M_${case_id}_omnirt_partner_omnirt"
    done

    "${HS_PYTHON}" - "${HS_OUT}" "${RL_ROOT}/holosoma_downstream_validation.json" <<'PY'
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
output = Path(sys.argv[2])
manifest_info = []
all_rows = []
for object_key in ("bucket003", "bucket004", "bucket007"):
    manifest = root / object_key / "manifest.tsv"
    with manifest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    all_rows.extend(rows)
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    manifest_info.append({"object_key": object_key, "manifest": str(manifest), "rows": len(rows), "sha256": digest})

failures = []
seen = set()
for row in all_rows:
    key = (row["case_id"], row["target_source"])
    if key in seen:
        failures.append(f"duplicate:{key}")
    seen.add(key)
    path = Path(row["export_with_partner_path"])
    if row["decision"] != "export_pass" or not path.is_file():
        failures.append(f"missing_or_failed:{key}")
        continue
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "partner_hand_pos_w": (int(row["output_frames"]), 2, 3),
            "partner_hand_quat_w": (int(row["output_frames"]), 2, 4),
            "object_contact": (int(row["output_frames"]), 2),
        }
        for field, shape in required.items():
            if field not in payload or payload[field].shape != shape:
                failures.append(f"schema:{key}:{field}")
        for field in payload.files:
            value = np.asarray(payload[field])
            if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
                failures.append(f"nonfinite:{key}:{field}")

summary = {
    "experiment": "E178",
    "stage": "S6_holosoma_partner_motion_validation",
    "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    "status": "DOWNSTREAM_RL_INPUT_VALIDATION_PASS" if not failures and len(all_rows) == 26 else "BLOCKED",
    "holosoma_output_root": str(root),
    "owner_scope": "E178_manual_use_partner_rl",
    "manifest_info": manifest_info,
    "rows": len(all_rows),
    "unique_case_target_source": len(seen),
    "case_count": len({row["case_id"] for row in all_rows}),
    "target_source_counts": dict(Counter(row["target_source"] for row in all_rows)),
    "partner_source_counts": dict(Counter(row["partner_source"] for row in all_rows)),
    "decision_counts": dict(Counter(row["decision"] for row in all_rows)),
    "registry_post_export_pass": not failures and len(all_rows) == 26,
    "registry_pre_train_pass": not failures and len(all_rows) == 26,
    "failures": failures,
    "claim_boundary": "input and registry readiness only; no RL policy outcome claim",
}
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
if summary["status"] != "DOWNSTREAM_RL_INPUT_VALIDATION_PASS":
    raise SystemExit(json.dumps(summary, ensure_ascii=False, indent=2))
print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
PY
}

case "${MODE}" in
    export)
        run_core_export
        ;;
    holosoma)
        run_holosoma_export
        ;;
    all)
        run_core_export
        run_holosoma_export
        ;;
    *)
        echo "Usage: $0 [export|holosoma|all]" >&2
        exit 2
        ;;
esac
