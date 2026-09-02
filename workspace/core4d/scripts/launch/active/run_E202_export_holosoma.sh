#!/usr/bin/env bash
# E202-export step 3: Holosoma downstream motion export for the USE13 translation-aug
# paired RL inputs. Partner hands are re-anchored to the SOURCE object (exporter
# default; --no-reanchor NOT passed) -- this is what makes the per-person augmentation
# self-consistent (see plan235 / progress). object_mismatch_* + partner_move_* are
# recorded as diagnostics (matches E200 box practice).
#
# Modes: dry (dry-run only) | export (dry-run + real) | validate (schema + mismatch report).
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SPIDER_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
HOLOSOMA_ROOT="${HOLOSOMA_ROOT:-/mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/holosoma}"
HS_PYTHON="${HS_PYTHON:-/mnt/ali-sh-1/dataset/zeus/xiayb/.holosoma_deps/miniconda3/envs/hsretargeting/bin/python}"
MODE="${1:-export}"

RL_ROOT="${SPIDER_ROOT}/workspace/core4d/results/E202/s6_downstream/rl_export_aug"
INPUT_TSV="${RL_ROOT}/rl_export_input.tsv"
PARTNER_TSV="${RL_ROOT}/partner_omnirt/rl_partner_omnirt_manifest.tsv"
HS_OUT="${HOLOSOMA_ROOT}/workspace/v3/data/E202_use13_aug_partner_rl"
HS_EXPORTER="${HOLOSOMA_ROOT}/workspace/v3/scripts/data/export_rl_motion_from_spider_tsv.py"

test -f "${INPUT_TSV}"; test -f "${PARTNER_TSV}"; test -x "${HS_PYTHON}"

# The exporter resolves SPIDER-relative manifest paths against $SPIDER_REPO
# (defaults to the legacy /home/ubuntu path). Point it at this repo.
export SPIDER_REPO="${SPIDER_ROOT}"

# object half-extents (identical to E178 manual-use export)
half_extents_for() {
    case "$1" in
        bucket003) echo "0.270393755 0.381407245 0.23287703" ;;
        bucket004) echo "0.161566625 0.231058755 0.15230013" ;;
        bucket007) echo "0.273275865 0.2869532 0.284911265" ;;
        *) echo "unknown object $1" >&2; exit 2 ;;
    esac
}

case_ids_for_object() {
    "${HS_PYTHON}" - "${INPUT_TSV}" "$1" <<'PY'
import csv, sys
with open(sys.argv[1], newline="", encoding="utf-8") as s:
    for row in csv.DictReader(s, delimiter="\t"):
        if row["object_key"] == sys.argv[2] and row["rl_export_decision"] == "RL_EXPORT_READY":
            print(row["case_id"])
PY
}

run_export() {
    local dry="$1"
    cd "${HOLOSOMA_ROOT}"
    for object_key in bucket003 bucket004 bucket007; do
        mapfile -t cases < <(case_ids_for_object "${object_key}")
        [ "${#cases[@]}" -gt 0 ] || { echo "no ready case for ${object_key}" >&2; exit 2; }
        local case_args=()
        for c in "${cases[@]}"; do case_args+=(--case-id "${c}"); done
        read -r hx hy hz <<<"$(half_extents_for "${object_key}")"
        local out_dir="${HS_OUT}/${object_key}"
        local extra=(); [ "${dry}" = "1" ] && extra=(--dry-run) || extra=(--force)
        echo "=== ${object_key}: ${#cases[@]} case(s) (dry=${dry}) ==="
        "${HS_PYTHON}" "${HS_EXPORTER}" \
            --input-tsv "${INPUT_TSV}" \
            --partner-omnirt-tsv "${PARTNER_TSV}" \
            "${case_args[@]}" \
            --target-source both \
            --partner-source omnirt \
            --target-variant ref_fk \
            --python "${HS_PYTHON}" \
            --out-dir "${out_dir}" \
            --object-half-extents "${hx}" "${hy}" "${hz}" \
            --include-contact-mask \
            "${extra[@]}"
    done
}

validate() {
    "${HS_PYTHON}" - "${HS_OUT}" "${RL_ROOT}/holosoma_downstream_validation.json" <<'PY'
import csv, json, sys
from collections import Counter
from pathlib import Path
import numpy as np
root, out = Path(sys.argv[1]), Path(sys.argv[2])
rows, mminfo = [], []
for obj in ("bucket003", "bucket004", "bucket007"):
    man = root / obj / "manifest.tsv"
    with man.open(newline="", encoding="utf-8") as s:
        r = list(csv.DictReader(s, delimiter="\t"))
    rows.extend(r)
    mm = [float(x.get("object_mismatch_max_m", "nan") or "nan") for x in r if x.get("object_mismatch_max_m")]
    mv = []
    for x in r:
        for k in ("partner_move_max_l_m", "partner_move_max_r_m"):
            if x.get(k): mv.append(float(x[k]))
    mminfo.append({"object": obj, "rows": len(r),
                   "object_mismatch_max_m": max(mm) if mm else None,
                   "partner_move_max_m": max(mv) if mv else None})
failures, seen = [], set()
for row in rows:
    key = (row["case_id"], row["target_source"])
    if key in seen: failures.append(f"dup:{key}")
    seen.add(key)
    p = Path(row.get("export_with_partner_path", ""))
    if row.get("decision") != "export_pass" or not p.is_file():
        failures.append(f"missing_or_failed:{key}"); continue
    with np.load(p, allow_pickle=False) as z:
        need = {"partner_hand_pos_w": (int(row["output_frames"]), 2, 3),
                "partner_hand_quat_w": (int(row["output_frames"]), 2, 4)}
        for f, shp in need.items():
            if f not in z or z[f].shape != shp: failures.append(f"schema:{key}:{f}")
        for f in z.files:
            v = np.asarray(z[f])
            if np.issubdtype(v.dtype, np.number) and not np.isfinite(v).all():
                failures.append(f"nonfinite:{key}:{f}")
summary = {"experiment": "E202-export", "stage": "S6_holosoma_aug_partner_validation",
           "status": "PASS" if not failures else "BLOCKED", "rows": len(rows),
           "unique_case_target_source": len(seen),
           "target_source_counts": dict(Counter(r["target_source"] for r in rows)),
           "decision_counts": dict(Counter(r.get("decision", "") for r in rows)),
           "reanchor_diagnostics": mminfo,
           "note": "object_mismatch_*/partner_move_* are pre->post reanchor diagnostics; "
                   "final partner_hand_pos_w is anchored to the SOURCE object.",
           "failures": failures}
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
raise SystemExit(0 if not failures else 1)
PY
}

case "${MODE}" in
    dry) run_export 1 ;;
    export) run_export 1; run_export 0; validate ;;
    validate) validate ;;
    *) echo "Usage: $0 [export|dry|validate]" >&2; exit 2 ;;
esac
