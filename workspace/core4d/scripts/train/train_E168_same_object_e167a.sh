#!/usr/bin/env bash
# E168: same-object E167A data expansion. GPU execution is delegated to launch scripts.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO_ROOT"

MODE="${1:-}"
RESULT_ROOT="${RESULT_ROOT:-workspace/core4d/results/E168}"
CORE4D_RAW_ROOT="${CORE4D_RAW_ROOT:-/mnt/a0ccc676-9496-49f8-a861-f8a1797dec52/mocap_data/CORE4D/CORE4D_Real}"
SMPLX_MODEL_DIR="${SMPLX_MODEL_DIR:-/mnt/a0ccc676-9496-49f8-a861-f8a1797dec52/mocap_data/smplx}"
HOLOSOMA_REPO="${HOLOSOMA_REPO:-/home/ubuntu/Workspace/holosoma}"
RETARGET_PYTHON="${RETARGET_PYTHON:-/home/ubuntu/.holosoma_deps/miniconda3/envs/hsretargeting/bin/python}"

INVENTORY_DIR="$RESULT_ROOT/s1_raw_contact/inventory"
RECALL_DIR="$RESULT_ROOT/s1_raw_contact/recall"
RAW_CONTACT_DIR="$RESULT_ROOT/s1_raw_contact/raw_contact"
S3_PRODUCTION_INPUT="$RESULT_ROOT/s3_retarget/e168_new_production_raw_contact_pass_3cm.tsv"
REGISTRY_DIR="$RESULT_ROOT/registries"
TEMPLATE_DIR="$RESULT_ROOT/s2_templates"
BUCKET_REVIEW_SNAPSHOT="$RESULT_ROOT/imported_snapshots/E145_bucket004_template_review/nonbox_template_review.tsv"
FORCE_REBUILD="${FORCE_REBUILD:-0}"

run_phase0() {
  python workspace/core4d/scripts/experiments/E168/build_phase0_preflight.py \
    --result-root "$RESULT_ROOT" \
    --holosoma-repo "$HOLOSOMA_REPO" \
    --core4d-raw-root "$CORE4D_RAW_ROOT" \
    --smplx-model-dir "$SMPLX_MODEL_DIR" \
    --retarget-python "$RETARGET_PYTHON"

  python workspace/core4d/scripts/experiments/E168/probe_remote_profiles.py
}

run_recall() {
  if [[ "$FORCE_REBUILD" == "1" || ! -s "$INVENTORY_DIR/inventory.tsv" ]]; then
    python workspace/core4d/scripts/data_construction_v3/stages/s1_raw_contact/build_inventory.py \
      --core4d-raw-root "$CORE4D_RAW_ROOT" \
      --spider-repo "$REPO_ROOT" \
      --out-dir "$INVENTORY_DIR"
  fi

  if [[ "$FORCE_REBUILD" == "1" || ! -s "$RECALL_DIR/e168_same_object_candidates.tsv" ]]; then
    python workspace/core4d/scripts/experiments/E168/build_same_object_recall_manifest.py \
      --inventory-tsv "$INVENTORY_DIR/inventory.tsv" \
      --out-dir "$RECALL_DIR"
  fi
}

run_raw_contact() {
  if [[ "$FORCE_REBUILD" == "1" || ! -s "$RAW_CONTACT_DIR/raw_contact_run_summary.json" ]]; then
    test -f "$RECALL_DIR/e168_same_object_candidates.tsv"
    python workspace/core4d/scripts/data_construction_v3/stages/s1_raw_contact/run_raw_contact.py \
      --core4d-raw-root "$CORE4D_RAW_ROOT" \
      --inventory-tsv "$RECALL_DIR/e168_same_object_candidates.tsv" \
      --out-dir "$RAW_CONTACT_DIR" \
      --queue object-key \
      --object-keys box004,box021,bucket004 \
      --thresholds-m 0.03,0.05 \
      --sample-count 12000 \
      --seed 203
  fi
}

run_registry() {
  python workspace/core4d/scripts/data_construction_v3/stages/s3_retarget/register_retarget_variant.py \
    --registry-dir "$REGISTRY_DIR" \
    --spider-repo "$REPO_ROOT" \
    --holosoma-repo "$HOLOSOMA_REPO" \
    --init-defaults

  python workspace/core4d/scripts/data_construction_v3/state/update_case_state_registry.py \
    --registry-dir "$REGISTRY_DIR" \
    --init-empty

  python workspace/core4d/scripts/data_construction_v3/state/update_case_state_registry.py \
    --registry-dir "$REGISTRY_DIR" \
    --from-inventory-tsv "$RECALL_DIR/e168_same_object_candidates.tsv" \
    --evidence-root "$INVENTORY_DIR" \
    --source-ref E168_S1_exact_candidate_inventory

  for label in 3cm 5cm; do
    python workspace/core4d/scripts/data_construction_v3/state/update_case_state_registry.py \
      --registry-dir "$REGISTRY_DIR" \
      --from-raw-contact-tsv "$RAW_CONTACT_DIR/raw_contact_candidates_${label}.tsv" \
      --raw-contact-label "$label" \
      --evidence-root "$RAW_CONTACT_DIR" \
      --source-ref "E168_S1_raw_contact_${label}"
  done
}

run_templates() {
  python workspace/core4d/scripts/data_construction_v3/stages/s2_templates/build_or_audit_templates.py \
    --input-tsv "$RAW_CONTACT_DIR/raw_contact_pass_3cm.tsv" \
    --core4d-raw-root "$CORE4D_RAW_ROOT" \
    --spider-repo "$REPO_ROOT" \
    --out-dir "$TEMPLATE_DIR" \
    --apply-build

  python workspace/core4d/scripts/experiments/E168/import_bucket004_template_review.py

  python workspace/core4d/scripts/data_construction_v3/state/update_case_state_registry.py \
    --registry-dir "$REGISTRY_DIR" \
    --from-template-backlog-tsv "$TEMPLATE_DIR/template_backlog.tsv" \
    --evidence-root "$TEMPLATE_DIR" \
    --source-ref E168_S2_template_audit

  python workspace/core4d/scripts/data_construction_v3/state/update_case_state_registry.py \
    --registry-dir "$REGISTRY_DIR" \
    --from-template-review-tsv "$BUCKET_REVIEW_SNAPSHOT" \
    --evidence-root "$RESULT_ROOT/imported_snapshots/E145_bucket004_template_review" \
    --source-type experiment_import \
    --source-ref E145_bucket004_template_review

  python workspace/core4d/scripts/data_construction_v3/stages/s2_templates/render_template_review_package.py \
    --template-backlog-tsv "$TEMPLATE_DIR/template_backlog.tsv" \
    --out-dir "$TEMPLATE_DIR/template_visual_review" \
    --overwrite

  python workspace/core4d/scripts/data_construction_v3/stages/s2_templates/render_template_mesh_collision_review_package.py \
    --input-tsv "$TEMPLATE_DIR/template_backlog.tsv" \
    --out-dir "$TEMPLATE_DIR/template_mesh_collision_review" \
    --render-statuses manual_review_required \
    --object-only \
    --overwrite
}

run_stage2b_dry_run() {
  local variant_dir="$RESULT_ROOT/s3_retarget/omnirt_v1/ref_fk"
  local manifest="$variant_dir/stage2b_manifest_omnirt_v1_ref_fk.tsv"

  python workspace/core4d/scripts/experiments/E168/build_s3_production_input.py \
    --raw-contact-pass-tsv "$RAW_CONTACT_DIR/raw_contact_pass_3cm.tsv" \
    --out-tsv "$S3_PRODUCTION_INPUT"

  if [[ ! -s "$manifest" || ! "$(head -1 "$manifest")" =~ execution_attempt ]]; then
    python workspace/core4d/scripts/data_construction_v3/stages/s3_retarget/run_stage2b.py \
      --raw-contact-tsv "$S3_PRODUCTION_INPUT" \
      --template-backlog-tsv "$TEMPLATE_DIR/template_backlog.tsv" \
      --template-review-tsv "$BUCKET_REVIEW_SNAPSHOT" \
      --retarget-variant-registry "$REGISTRY_DIR/retarget_variant_registry.tsv" \
      --retarget-variant-id omnirt_v1 \
      --target-variant-id ref_fk \
      --inventory-tsv "$RECALL_DIR/e168_same_object_candidates.tsv" \
      --core4d-raw-root "$CORE4D_RAW_ROOT" \
      --smplx-model-dir "$SMPLX_MODEL_DIR" \
      --python-bin "$REPO_ROOT/.venv/bin/python" \
      --retarget-python-bin "$RETARGET_PYTHON" \
      --out-dir "$variant_dir"
  fi

  python workspace/core4d/scripts/data_construction_v3/state/update_case_state_registry.py \
    --registry-dir "$REGISTRY_DIR" \
    --from-stage2b-manifest-tsv "$manifest" \
    --evidence-root "$variant_dir" \
    --source-ref E168_S3_omnirt_v1_ref_fk_dry_run
}

run_stage2b_execute() {
  local variant_dir="$RESULT_ROOT/s3_retarget/omnirt_v1/ref_fk"
  python workspace/core4d/scripts/experiments/E168/run_stage2b_queue.py \
    --manifest-tsv "$variant_dir/stage2b_manifest_omnirt_v1_ref_fk.tsv" \
    --holosoma-repo "$HOLOSOMA_REPO" \
    --core4d-raw-root "$CORE4D_RAW_ROOT" \
    --smplx-model-dir "$SMPLX_MODEL_DIR" \
    --python-bin "$REPO_ROOT/.venv/bin/python" \
    --retarget-python-bin "$RETARGET_PYTHON"

  python workspace/core4d/scripts/data_construction_v3/state/update_case_state_registry.py \
    --registry-dir "$REGISTRY_DIR" \
    --from-stage2b-manifest-tsv "$variant_dir/stage2b_manifest_omnirt_v1_ref_fk.tsv" \
    --evidence-root "$variant_dir" \
    --source-ref E168_S3_omnirt_v1_ref_fk_execute
}

case "$MODE" in
  phase0)
    run_phase0
    ;;
  recall)
    run_phase0
    run_recall
    ;;
  raw-contact)
    run_phase0
    run_recall
    run_raw_contact
    ;;
  registry)
    run_phase0
    run_recall
    run_raw_contact
    run_registry
    ;;
  templates)
    run_phase0
    run_recall
    run_raw_contact
    run_registry
    run_templates
    ;;
  stage2b-dry-run)
    run_phase0
    run_recall
    run_raw_contact
    run_registry
    run_templates
    run_stage2b_dry_run
    ;;
  stage2b)
    run_phase0
    run_recall
    run_raw_contact
    run_registry
    run_templates
    run_stage2b_dry_run
    run_stage2b_execute
    ;;
  *)
    echo "usage: $0 {phase0|recall|raw-contact|registry|templates|stage2b-dry-run|stage2b}" >&2
    exit 2
    ;;
esac
