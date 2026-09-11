#!/usr/bin/env bash
# E213-export step 4: Holosoma downstream motion export for the selected-arm object-aug
# paired RL inputs (log302 handed off the paired input; this runs the actual motion export).
# Partner hands are re-anchored to the SOURCE object (exporter default; --no-reanchor NOT
# passed) -- this is what makes the per-person augmentation self-consistent for two-person
# objects (see two-person-aug-partner-reanchor / plan235 / log294). object_mismatch_* +
# partner_move_* are recorded as pre->post reanchor diagnostics only (matches E200/E202).
#
# Objects: chair006, desk007, desk021, desk023 (32 units / 20 source cases).
# --object-half-extents is used ONLY for the surface-distance diagnostic (mesh AABB half-size
# in the object canonical frame); it does NOT affect the exported/reanchored motion.
#
# Modes: dry (dry-run only) | export (dry-run + real + validate) | validate (schema + mismatch).
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SPIDER_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
HOLOSOMA_ROOT="${HOLOSOMA_ROOT:-/mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/holosoma}"
HS_PYTHON="${HS_PYTHON:-/mnt/ali-sh-1/dataset/zeus/xiayb/.holosoma_deps/miniconda3/envs/hsretargeting/bin/python}"
MODE="${1:-export}"

RL_ROOT="${SPIDER_ROOT}/workspace/core4d/results/E213/s6_downstream/export"
INPUT_TSV="${RL_ROOT}/rl_export_input.tsv"
PARTNER_TSV="${RL_ROOT}/partner_omnirt/rl_partner_omnirt_manifest.tsv"
HS_OUT="${HOLOSOMA_ROOT}/workspace/v3/data/E213_selected_arm_aug_partner_rl"
HS_EXPORTER="${HOLOSOMA_ROOT}/workspace/v3/scripts/data/export_rl_motion_from_spider_tsv.py"

test -f "${INPUT_TSV}"; test -f "${PARTNER_TSV}"; test -x "${HS_PYTHON}"

OBJECTS=(chair006 desk007 desk021 desk023)

# The exporter resolves SPIDER-relative manifest paths against $SPIDER_REPO
# (defaults to the legacy /home/ubuntu path). Point it at this repo.
export SPIDER_REPO="${SPIDER_ROOT}"

# object mesh AABB half-extents in canonical frame (verified vs built-in for desk021/desk023;
# chair006/desk007 computed from holosoma_retargeting/models/*/*.obj). Diagnostic-only.
half_extents_for() {
    case "$1" in
        chair006) echo "0.297191 0.427942 0.28029" ;;
        desk007)  echo "0.215239 0.349005 0.400791" ;;
        desk021)  echo "0.2544315 0.228550795 0.252833755" ;;
        desk023)  echo "0.2566801 0.23077113 0.40454542" ;;
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
    for object_key in "${OBJECTS[@]}"; do
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
    "${HS_PYTHON}" - "${HS_OUT}" "${RL_ROOT}/holosoma_downstream_validation.json" "${OBJECTS[@]}" <<'PY'
import csv, json, sys
from collections import Counter
from pathlib import Path
import numpy as np
root, out = Path(sys.argv[1]), Path(sys.argv[2])
objects = sys.argv[3:]
rows, mminfo = [], []
for obj in objects:
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
summary = {"experiment": "E213-export", "stage": "S6_holosoma_aug_partner_validation",
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
