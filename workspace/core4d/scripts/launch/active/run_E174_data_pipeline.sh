#!/usr/bin/env bash
# E174 S0-S5 local data pipeline (5 bucket + 2 desk, fresh from CORE4D raw).
#
# Canonical reproducible entry for the E174 non-box data-construction chain. First
# run to leave the box topology: object concavity is expressed by the object-side
# collision_policy proxy (bucket_wall / desk_surface_voxel), NOT rubber_hull.
#
# Differences vs E173 (all-box):
#   * action scope is move2-ONLY (move1 excluded, non-move Stage0-rejected);
#   * S2 uses the dcv3 NON-BOX proxy flow: templates are manual_review_required by
#     default and must be lifted to clean_reviewed via an explicit approve step in
#     nonbox_template_review.tsv (object-only mesh/collision overlay evidence);
#     the box mesh-AABB rebuild (--overwrite-existing) is NOT used;
#   * S3 consumes nonbox_template_review.tsv via --template-review-tsv.
#
# Resumable: each stage is guarded by an existence check. S3+ is GATED on all
# in-scope non-box templates being approve_clean; if any are pending the script
# stops cleanly after S2 so the reviewer can approve, then re-run.
#
# Usage:
#   bash workspace/core4d/scripts/launch/active/run_E174_data_pipeline.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# --- config (env-overridable) ------------------------------------------------
export HOLOSOMA_DEPS_DIR="${HOLOSOMA_DEPS_DIR:-/mnt/ali-sh-1/dataset/zeus/xiayb/.holosoma_deps}"
CORE4D_RAW_ROOT="${CORE4D_RAW_ROOT:-/mnt/ali-sh-1/usr/xiayibo/xyb_data_tidal_alsh/other-datasets/mocap_data/CORE4D/CORE4D_Real}"
SMPLX_MODEL_DIR="${SMPLX_MODEL_DIR:-/mnt/ali-sh-1/usr/xiayibo/xyb_data_tidal_alsh/other-datasets/mocap_data/human_model_files}"
RETARGET_PYTHON_BIN="${RETARGET_PYTHON_BIN:-$HOLOSOMA_DEPS_DIR/miniconda3/envs/hsretargeting/bin/python}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
HOLOSOMA_REPO="${HOLOSOMA_REPO:-../holosoma}"
export CORE4D_RAW_ROOT SMPLX_MODEL_DIR RETARGET_PYTHON_BIN

DCV3=workspace/core4d/scripts/data_construction_v3
S1=$DCV3/stages/s1_raw_contact
S2=$DCV3/stages/s2_templates
S3=$DCV3/stages/s3_retarget
S4=$DCV3/stages/s4_gate_visual_qc
S5=$DCV3/stages/s5_handoff
STATE=$DCV3/state/update_case_state_registry.py
E174=workspace/core4d/scripts/experiments/E174

E=workspace/core4d/results/E174
REG=$E/registries
INV=$E/s1_raw_contact/inventory
RC=$E/s1_raw_contact/raw_contact
TPL=$E/s2_templates
REVIEW=$TPL/review/nonbox_template_review.tsv
OBJECT_KEYS=bucket004,bucket009,bucket010,bucket007,bucket003,desk005,desk007

reg() { $PYTHON_BIN "$STATE" --registry-dir "$REG" "$@"; }
say() { echo -e "\n=== E174 $* ==="; }

mkdir -p "$E/s0_environment" "$INV" "$RC" "$TPL" "$REG"

# --- S0: environment + registries -------------------------------------------
say "S0 environment + registries"
$PYTHON_BIN $DCV3/stages/s0_environment/check_environment.py \
  --spider-repo . --holosoma-repo "$HOLOSOMA_REPO" \
  --core4d-raw-root "$CORE4D_RAW_ROOT" --run-root "$E" \
  --out-dir "$E/s0_environment" || echo "[warn] S0 env check non-zero (review s0_environment)"
[ -f "$REG/retarget_variant_registry.tsv" ] || \
  $PYTHON_BIN $S3/register_retarget_variant.py --registry-dir "$REG" --init-defaults
[ -f "$REG/case_state_registry.tsv" ] || reg --init-empty

# --- S1: inventory + raw contact (3cm/5cm) ----------------------------------
say "S1 inventory + raw contact"
[ -f "$INV/inventory.tsv" ] || \
  $PYTHON_BIN $S1/build_inventory.py --core4d-raw-root "$CORE4D_RAW_ROOT" --out-dir "$INV"
reg --from-inventory-tsv "$INV/inventory.tsv" --evidence-root "$INV" --source-ref S1_inventory
[ -f "$RC/raw_contact_candidates_3cm.tsv" ] || \
  $PYTHON_BIN $S1/run_raw_contact.py --core4d-raw-root "$CORE4D_RAW_ROOT" \
    --inventory-tsv "$INV/inventory.tsv" --out-dir "$RC" \
    --queue object-key --object-keys "$OBJECT_KEYS" --thresholds-m 0.03,0.05
for label in 3cm 5cm; do
  reg --from-raw-contact-tsv "$RC/raw_contact_candidates_${label}.tsv" \
    --raw-contact-label "$label" --evidence-root "$RC" --source-ref "S1_raw_contact_${label}"
done

# --- move2-only enforcement (E174): drop non-move2 rows before Stage2b --------
# dcv3 inventory lumps move1_obs0 + move2_obs0 into main_move_obs0 (both survive
# Stage0), so move1 rows can pass raw contact. E174 policy is move2-ONLY: keep
# only rows whose raw action starts with "move2" AND that entered raw contact.
# move1 -> EXCLUDE_ACTION_NOT_MOVE2 (registry only); non-move already Stage0.
RCPASS=$RC/raw_contact_pass_3cm.tsv
RCPASS_MOVE2=$RC/raw_contact_pass_3cm_move2only.tsv
$PYTHON_BIN - "$RCPASS" "$RCPASS_MOVE2" <<'PYEOF'
import csv, sys
src, dst = sys.argv[1], sys.argv[2]
MOVE_OK = {"inventory_pass_to_raw_contact", "inventory_review_to_raw_contact"}
rows = list(csv.DictReader(open(src), delimiter="\t"))
kept, move1, other = [], 0, 0
for r in rows:
    is_move2 = str(r.get("action", "")).startswith("move2")
    is_move1 = str(r.get("action", "")).startswith("move1")
    if r.get("inventory_decision") in MOVE_OK and is_move2:
        kept.append(r)
    elif is_move1:
        move1 += 1
    else:
        other += 1
with open(dst, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys(), delimiter="\t", lineterminator="\n")
    w.writeheader(); w.writerows(kept)
print(f"[move2-only] pass_3cm={len(rows)} -> move2 kept={len(kept)} move1_excluded={move1} other={other}")
PYEOF
RCPASS=$RCPASS_MOVE2

# --- S2: non-box source template proxy build + explicit review ---------------
# NON-BOX flow (dcv3 doc 04): --apply-build creates review-only proxy templates
# for the MISSING dirs (bucket009_person1, desk005_person1) and audits existing
# ones. NO --overwrite-existing: we must NOT rebuild the concave proxies as box
# AABB. All non-box templates come back manual_review_required.
say "S2 non-box template proxy build/audit (apply-build, no overwrite)"
[ -f "$TPL/template_backlog.tsv" ] || \
  $PYTHON_BIN $S2/build_or_audit_templates.py \
    --input-tsv "$RC/raw_contact_candidates_3cm.tsv" \
    --core4d-raw-root "$CORE4D_RAW_ROOT" --out-dir "$TPL" \
    --apply-build
reg --from-template-backlog-tsv "$TPL/template_backlog.tsv" --evidence-root "$TPL" \
  --source-ref S2_template_backlog_audit

# review queue + object-only mesh/collision overlay evidence
[ -f "$REVIEW" ] || \
  $PYTHON_BIN $E174/build_nonbox_template_review.py \
    --template-backlog-tsv "$TPL/template_backlog.tsv" \
    --out-dir "$TPL/review" --render

# GATE: S3+ requires every in-scope non-box template approve_clean.
PENDING=$($PYTHON_BIN - "$REVIEW" <<'PYEOF'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1]), delimiter="\t"))
pending = [r["source_scene_task"] for r in rows if r.get("review_decision") != "approve_clean"]
print(",".join(pending))
PYEOF
)
if [ -n "$PENDING" ]; then
  say "S2 REVIEW REQUIRED -- pending approve_clean: $PENDING"
  echo "Inspect object-only overlays in $TPL/review/object_only_overlay/, then approve:"
  echo "  $PYTHON_BIN $E174/build_nonbox_template_review.py --out-dir $TPL/review \\"
  echo "    --approve <task> [--approve <task> ...] --reviewer codex --notes '...'"
  echo "Then re-run this script to continue S3-S5."
  exit 0
fi
echo "[gate] all in-scope non-box templates approve_clean -> continue to S3"

# --- S3 v1 primary -----------------------------------------------------------
say "S3 omnirt_v1 primary"
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
$PYTHON_BIN $E174/run_stage2b_queue.py --manifest-tsv "$V1M" \
  --holosoma-repo "$HOLOSOMA_REPO" --core4d-raw-root "$CORE4D_RAW_ROOT" \
  --smplx-model-dir "$SMPLX_MODEL_DIR" --retarget-python-bin "$RETARGET_PYTHON_BIN"
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
  $PYTHON_BIN $E174/run_stage2b_queue.py --manifest-tsv "$V2M" \
    --holosoma-repo "$HOLOSOMA_REPO" --core4d-raw-root "$CORE4D_RAW_ROOT" \
    --smplx-model-dir "$SMPLX_MODEL_DIR" --retarget-python-bin "$RETARGET_PYTHON_BIN"
  reg --from-stage2b-manifest-tsv "$V2M" --evidence-root "$V2" --source-ref S3_stage2b_omnirt_v2_ref_fk
else
  echo "[info] no v1 omniretarget_infeasible rows -> v2 rescue skipped (valid no-op)"
fi

# --- S4: target gate + visual QC (per executed variant) ---------------------
say "S4 target gate + visual QC"
run_s4() {
  local variant="$1" mdir="$2" manifest="$3"
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
run_s4 omnirt_v1 "$V1" "$V1M"
run_s4 omnirt_v2 "$V2" "$V2M"

# --- S5: handoff + CEM overrides --------------------------------------------
say "S5 handoff + CEM overrides"
HANDOFF=$E/s5_handoff
S5_ARGS=(--case-state-registry "$REG/case_state_registry.tsv" --out-dir "$HANDOFF")
[ -f "$V1M" ] && S5_ARGS+=(--stage2b-manifest-tsv "$V1M")
[ -f "$V2M" ] && S5_ARGS+=(--stage2b-manifest-tsv "$V2M")
[ -f "$E/s4_gate_visual_qc/omnirt_v1/ref_fk/target_gate_manifest.tsv" ] && \
  S5_ARGS+=(--target-gate-manifest-tsv "$E/s4_gate_visual_qc/omnirt_v1/ref_fk/target_gate_manifest.tsv")
[ -f "$E/s4_gate_visual_qc/omnirt_v2/ref_fk/target_gate_manifest.tsv" ] && \
  S5_ARGS+=(--target-gate-manifest-tsv "$E/s4_gate_visual_qc/omnirt_v2/ref_fk/target_gate_manifest.tsv")
$PYTHON_BIN $S5/export_handoff.py "${S5_ARGS[@]}"
# base-override = E167A_zOnlyBody (plan-frozen E170 PRG base reward; object-agnostic
# zOnlyBody, per-case scene/trajectory injected by handoff -- same as E171/E172/E173)
$PYTHON_BIN $S5/export_cem_overrides.py --handoff-manifest-tsv "$HANDOFF/handoff_manifest.tsv" \
  --out-dir "$HANDOFF/cem_overrides" --spider-repo . \
  --base-override core4d_E167_box004_082_p1_E167A

say "S0-S5 complete. Next: build_pipeline_authority.py + build_prg_cem_manifest.py"
