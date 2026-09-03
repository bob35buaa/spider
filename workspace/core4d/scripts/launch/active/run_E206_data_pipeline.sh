#!/usr/bin/env bash
# E206 S0-S5 data pipeline (desk+chair move2, lowgeom manual collision proxies).
#
# S0/S1/S2 are ALREADY CLOSED as of commit 128d816 (65 case / 8 objects, contract
# 8/8 all_pass, 15/15 templates installed, 15 review rows approve_clean). Every
# stage below is existence-guarded, so re-running this script resumes at S3.
#
# Differences vs run_E174_data_pipeline.sh:
#   * collision proxies are hand-placed (`manual_boxes.json`), installed into the
#     SOURCE templates by E206/install_lowgeom_templates.py -- not a dcv3 policy
#     flag. S2 here only re-verifies; it never rebuilds a proxy.
#   * chair021 is dropped (F12). The in-scope filter below is the single place
#     that turns the 74-row 3cm pass list into the authoritative 65-row S3 input.
#   * S3 runs SHARDED (run_E206_stage2b_parallel.sh); 65 serial OmniRetarget
#     solves on a 192-core host would waste hours.
#
# P3 (throughput admission) is deliberately NOT a prerequisite here: it gates the
# CEM queue (P8), not retargeting. S3 is CPU-only (CVXPY/IK, no CUDA), so it runs
# fine while the GPUs are busy.
#
# Usage:
#   bash workspace/core4d/scripts/launch/active/run_E206_data_pipeline.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# --- config (env-overridable) ------------------------------------------------
export HOLOSOMA_DEPS_DIR="${HOLOSOMA_DEPS_DIR:-/mnt/ali-sh-1/dataset/zeus/xiayb/.holosoma_deps}"
CORE4D_RAW_ROOT="${CORE4D_RAW_ROOT:-/mnt/ali-sh-1/usr/xiayibo/xyb_data_tidal_alsh/other-datasets/mocap_data/CORE4D/CORE4D_Real}"
SMPLX_MODEL_DIR="${SMPLX_MODEL_DIR:-/mnt/ali-sh-1/usr/xiayibo/xyb_data_tidal_alsh/other-datasets/mocap_data/human_model_files}"
RETARGET_PYTHON_BIN="${RETARGET_PYTHON_BIN:-$HOLOSOMA_DEPS_DIR/miniconda3/envs/hsretargeting/bin/python}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
HOLOSOMA_REPO="${HOLOSOMA_REPO:-../holosoma}"
NSHARDS="${NSHARDS:-24}"
export CORE4D_RAW_ROOT SMPLX_MODEL_DIR RETARGET_PYTHON_BIN

DCV3=workspace/core4d/scripts/data_construction_v3
S3=$DCV3/stages/s3_retarget
S4=$DCV3/stages/s4_gate_visual_qc
S5=$DCV3/stages/s5_handoff
STATE=$DCV3/state/update_case_state_registry.py
E174=workspace/core4d/scripts/experiments/E174
E206=workspace/core4d/scripts/experiments/E206

E=workspace/core4d/results/E206
REG=$E/registries
INV=$E/s1_raw_contact/inventory
RC=$E/s1_raw_contact/raw_contact
TPL=$E/s2_templates
REVIEW=$TPL/review/nonbox_template_review.tsv

reg() { $PYTHON_BIN "$STATE" --registry-dir "$REG" "$@"; }
say() { echo -e "\n=== E206 $* ==="; }

mkdir -p "$REG"

# --- S0/S1/S2 preconditions (must already exist) -----------------------------
say "S0-S2 precondition check"
for f in "$INV/inventory.tsv" "$RC/raw_contact_pass_3cm_move2only.tsv" \
         "$TPL/template_backlog.tsv" "$REVIEW" \
         "$E/s2_proxy/lowgeom_contract.json" "$TPL/lowgeom_install.tsv"; do
  [ -f "$f" ] || { echo "MISSING precondition: $f" >&2; exit 2; }
done
# The contract must be green: shipping a proxy that fails a hard gate into 65
# retargets and 130 CEM runs is exactly the mistake the gates exist to prevent.
$PYTHON_BIN - "$E/s2_proxy/lowgeom_contract.json" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
if not d.get("all_pass"):
    sys.exit(f"lowgeom contract not green: {d.get('hard_gates_pass')}/{d.get('total')}")
print(f"[gate] lowgeom contract {d['hard_gates_pass']}/{d['total']} all_pass -> ok")
PYEOF

[ -f "$REG/retarget_variant_registry.tsv" ] || \
  $PYTHON_BIN $S3/register_retarget_variant.py --registry-dir "$REG" --init-defaults
[ -f "$REG/case_state_registry.tsv" ] || reg --init-empty
reg --from-inventory-tsv "$INV/inventory.tsv" --evidence-root "$INV" --source-ref S1_inventory
reg --from-template-backlog-tsv "$TPL/template_backlog.tsv" --evidence-root "$TPL" \
  --source-ref S2_template_backlog_audit

# --- in-scope S3 input: 74 pass rows -> 65 (drop chair021, F12) --------------
say "S3 input scope filter (DROPPED_OBJECT_KEYS)"
RCPASS=$RC/raw_contact_pass_3cm_move2only_inscope.tsv
$PYTHON_BIN - "$RC/raw_contact_pass_3cm_move2only.tsv" "$RCPASS" <<'PYEOF'
import collections, sys
sys.path.insert(0, "workspace/core4d/scripts/experiments/E206")
import e206_common as C
rows = C.read_tsv(sys.argv[1])
keep = [r for r in rows if C.is_in_scope(r)]
drop = [r for r in rows if not C.is_in_scope(r)]
C.write_tsv(sys.argv[2], keep, list(rows[0].keys()))
print(f"[scope] {len(rows)} -> {len(keep)} in-scope, dropped {len(drop)} "
      f"{dict(collections.Counter(r['object_key'] for r in drop))}")
print(f"[scope] per object: {dict(sorted(collections.Counter(r['object_key'] for r in keep).items()))}")
if len(keep) != C.N_CASES_IN_SCOPE:
    sys.exit(f"expected {C.N_CASES_IN_SCOPE} in-scope cases, got {len(keep)}")
PYEOF

# --- GATE: every template an in-scope case needs must be approve_clean -------
# run_stage2b.py:168 silently holds a case whose template is not clean/
# clean_reviewed, and clean_reviewed only comes from review_decision=
# approve_clean (:77). F13 was exactly this failing quietly for all person2
# rows, so assert it here per TASK rather than trusting a row count.
$PYTHON_BIN - "$RCPASS" "$REVIEW" <<'PYEOF'
import sys
sys.path.insert(0, "workspace/core4d/scripts/experiments/E206")
import e206_common as C
# `person` is already "person1"/"person2", so the template task is a plain join.
needed = {f"{r['object_key']}_{r['person']}" for r in C.read_tsv(sys.argv[1])}
approved = {r["source_scene_task"] for r in C.read_tsv(sys.argv[2])
            if r.get("review_decision") == "approve_clean"}
missing = sorted(t for t in needed if t and t not in approved)
if missing:
    sys.exit(f"[gate] templates not approve_clean, S3 would hold them: {missing}")
print(f"[gate] all {len(needed)} required templates approve_clean -> continue to S3")
PYEOF

# --- S3 v1 primary -----------------------------------------------------------
# run_stage2b.py builds the manifest only (no --execute): execution is the shard
# runner's job, which isolates each case and keeps failure evidence. This mirrors
# run_E174_data_pipeline.sh:151-161; plan236's `--execute` wording predates it.
say "S3 omnirt_v1 primary (65 cases, sharded x$NSHARDS)"
V1=$E/s3_retarget/omnirt_v1/ref_fk
V1M=$V1/stage2b_manifest_omnirt_v1_ref_fk.tsv
[ -f "$V1M" ] || $PYTHON_BIN $S3/run_stage2b.py \
  --raw-contact-tsv "$RCPASS" --template-backlog-tsv "$TPL/template_backlog.tsv" \
  --template-review-tsv "$REVIEW" \
  --retarget-variant-registry "$REG/retarget_variant_registry.tsv" \
  --retarget-variant-id omnirt_v1 --target-variant-id ref_fk \
  --inventory-tsv "$INV/inventory.tsv" --core4d-raw-root "$CORE4D_RAW_ROOT" \
  --smplx-model-dir "$SMPLX_MODEL_DIR" --retarget-python-bin "$RETARGET_PYTHON_BIN" \
  --out-dir "$V1"
VARIANT=omnirt_v1 NSHARDS=$NSHARDS \
  bash workspace/core4d/scripts/launch/active/run_E206_stage2b_parallel.sh
reg --from-stage2b-manifest-tsv "$V1M" --evidence-root "$V1" --source-ref S3_stage2b_omnirt_v1_ref_fk

# --- S3 v2 rescue (only fresh v1 omniretarget_infeasible) --------------------
say "S3 omnirt_v2 rescue (only fresh v1 infeasible)"
V2=$E/s3_retarget/omnirt_v2/ref_fk
RESCUE=$E/s3_retarget/rescue
V2M=$V2/stage2b_manifest_omnirt_v2_ref_fk.tsv
$PYTHON_BIN $E174/build_omnirt_rescue_manifest.py \
  --s3-input-tsv "$RCPASS" --v1-manifest-tsv "$V1M" --out-dir "$RESCUE"
RESCUE_IN=$RESCUE/omnirt_v2_rescue_input.tsv
if [ -f "$RESCUE_IN" ]; then
  echo "[info] v2 rescue set non-empty -> running omnirt_v2"
  [ -f "$V2M" ] || $PYTHON_BIN $S3/run_stage2b.py \
    --raw-contact-tsv "$RESCUE_IN" --template-backlog-tsv "$TPL/template_backlog.tsv" \
    --template-review-tsv "$REVIEW" \
    --retarget-variant-registry "$REG/retarget_variant_registry.tsv" \
    --retarget-variant-id omnirt_v2 --target-variant-id ref_fk \
    --inventory-tsv "$INV/inventory.tsv" --core4d-raw-root "$CORE4D_RAW_ROOT" \
    --smplx-model-dir "$SMPLX_MODEL_DIR" --retarget-python-bin "$RETARGET_PYTHON_BIN" \
    --out-dir "$V2"
  VARIANT=omnirt_v2 NSHARDS=$NSHARDS \
    bash workspace/core4d/scripts/launch/active/run_E206_stage2b_parallel.sh
  reg --from-stage2b-manifest-tsv "$V2M" --evidence-root "$V2" --source-ref S3_stage2b_omnirt_v2_ref_fk
else
  echo "[info] no v1 omniretarget_infeasible rows -> v2 rescue skipped (valid no-op)"
fi

# --- reproducibility cross-check vs E145 (plan236 P5) ------------------------
say "S3 reproducibility cross-check vs E145"
$PYTHON_BIN $E206/check_reproducibility_vs_e145.py \
  --v1-manifest-tsv "$V1M" --out-dir "$E/s3_retarget" \
  || echo "[warn] reproducibility cross-check non-zero (review the tsv)"

# --- S4: target gate + visual QC (per executed variant) ---------------------
say "S4 target gate + visual QC"
run_s4() {
  local variant="$1" manifest="$2"
  [ -f "$manifest" ] || return 0
  local gate="$E/s4_gate_visual_qc/$variant/ref_fk"
  $PYTHON_BIN $S4/run_target_gate.py --stage2b-manifest-tsv "$manifest" --out-dir "$gate" --spider-repo .
  reg --from-target-gate-manifest-tsv "$gate/target_gate_manifest.tsv" --evidence-root "$gate" \
    --source-ref "S4_target_gate_${variant}_ref_fk"
  $PYTHON_BIN $S4/make_visual_qc.py --target-gate-manifest-tsv "$gate/target_gate_manifest.tsv" \
    --out-dir "$gate/visual_qc"
  reg --from-visual-qc-manifest-tsv "$gate/visual_qc/visual_qc_manifest.tsv" --evidence-root "$gate/visual_qc" \
    --source-ref "S4_visual_qc_${variant}_ref_fk"
  $PYTHON_BIN $S4/render_visual_qc_package.py \
    --target-gate-manifest-tsv "$gate/target_gate_manifest.tsv" \
    --out-dir "$gate/visual_qc_render" || echo "[warn] S4 render non-zero for $variant"
}
run_s4 omnirt_v1 "$V1M"
run_s4 omnirt_v2 "$V2M"

# --- S5: handoff --------------------------------------------------------------
say "S5 handoff"
HANDOFF=$E/s5_handoff
S5_ARGS=(--case-state-registry "$REG/case_state_registry.tsv" --out-dir "$HANDOFF")
[ -f "$V1M" ] && S5_ARGS+=(--stage2b-manifest-tsv "$V1M")
[ -f "$V2M" ] && S5_ARGS+=(--stage2b-manifest-tsv "$V2M")
for v in omnirt_v1 omnirt_v2; do
  g=$E/s4_gate_visual_qc/$v/ref_fk/target_gate_manifest.tsv
  [ -f "$g" ] && S5_ARGS+=(--target-gate-manifest-tsv "$g")
done
$PYTHON_BIN $S5/export_handoff.py "${S5_ARGS[@]}"

say "S3-S5 complete. Next: P3 throughput admission, then P7 arm scenes + P8 CEM."
