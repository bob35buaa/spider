#!/usr/bin/env bash
# E203 P2/P3/P4 auto: box021,box023,box026 + buckets(001,003,004,005,007,009,010).
# S1 raw-contact -> S2 templates (box auto-build; bucket review-inject) ->
# S3 omnirt_v1 (KEEP_GOING) -> omnirt_v2 rescue -> CEM (8-GPU). All core4d_v2.
# Bucket PRG uses E199 single-object_collision builder (works on reviewed
# bucket templates' object_collision geom; coarser than E202 per-seg union).
set -uo pipefail
cd /mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/spider
export PATH="$PWD/.venv/bin:$PATH"
export HOLOSOMA_REPO=/mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/holosoma
export SMPLX_MODEL_DIR=/mnt/ali-sh-1/usr/xiayibo/xyb_data_tidal_alsh/other-datasets/mocap_data/human_model_files
export E203_MUJOCO_GL=osmesa
export E203_TORCH_COMPILE=1
V2ROOT=/mnt/ali-sh-1/usr/xiayibo/xyb_data_tidal_alsh/other-datasets/mocap_data/CORE4D/CORE4D_Real_v2_materialized
R=$(readlink -f workspace/core4d/results/E203)
PY=.venv/bin/python
OBJS="${OBJS:-box021,box023,box026,bucket001,bucket003,bucket004,bucket005,bucket007,bucket009,bucket010}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
S1OUT="$R/s1_raw_contact/raw_contact_rest"
PASS="$S1OUT/raw_contact_pass_3cm.tsv"
S2OUT="$R/s2_templates"        # shared; audit appends
V1OUT="$R/s3_retarget/omnirt_v1/ref_fk"       # shared out-dir with P1 (task names distinct)
V1RES="$V1OUT/results/omnirt_v1_ref_fk"
V2OUT="$R/s3_retarget/omnirt_v2/ref_fk"
DCV3=workspace/core4d/scripts/data_construction_v3

echo "===== [REST] STEP 1: raw contact (queue=object-key) $OBJS ====="
$PY $DCV3/stages/s1_raw_contact/run_raw_contact.py \
  --core4d-raw-root "$V2ROOT" \
  --inventory-tsv "$R/s1_raw_contact/inventory/inventory.tsv" \
  --out-dir "$S1OUT" --queue object-key --object-keys "$OBJS" --thresholds-m 0.03,0.05

echo "===== [REST] STEP 2: templates (box auto-build; bucket -> manual_review) ====="
$PY $DCV3/stages/s2_templates/build_or_audit_templates.py \
  --input-tsv "$PASS" --core4d-raw-root "$V2ROOT" --out-dir "$S2OUT" --apply-build
$PY $DCV3/state/update_case_state_registry.py \
  --registry-dir "$R/registries" \
  --from-template-backlog-tsv "$S2OUT/template_backlog.tsv" \
  --evidence-root "$S2OUT" --source-ref S2_rest_template_audit || true

echo "===== [REST] STEP 2b: auto-approve existing bucket templates (E177/E202 reviewed) ====="
REVIEW_TSV="$S2OUT/e203_bucket_review.tsv"
$PY - "$S2OUT/template_backlog.tsv" "$REVIEW_TSV" <<'PYEOF'
import sys, os
backlog, out = sys.argv[1:3]
SRC="example_datasets/processed/core4d/unitree_g1/humanoid_object"
rows=[]
with open(backlog) as f:
    hdr=f.readline().rstrip("\n").split("\t")
    ti=hdr.index("source_scene_task") if "source_scene_task" in hdr else 0
    ci=hdr.index("object_category") if "object_category" in hdr else None
    for ln in f:
        c=ln.rstrip("\n").split("\t")
        task=c[ti]
        cat=c[ci] if ci is not None else ("bucket" if "bucket" in task else "")
        if cat=="bucket" and os.path.isfile(f"{SRC}/{task}/scene_act.xml"):
            rows.append(task)
with open(out,"w") as f:
    f.write("source_scene_task\treview_decision\treviewer\treview_notes\n")
    for t in sorted(set(rows)):
        f.write(f"{t}\tapprove_clean\tE203_auto\tE177_E202_reviewed_bucket_proxy_reused\n")
print(f"bucket templates auto-approved: {len(set(rows))}")
PYEOF
$PY $DCV3/state/update_case_state_registry.py \
  --registry-dir "$R/registries" \
  --from-template-review-tsv "$REVIEW_TSV" \
  --evidence-root "$S2OUT" --source-ref E203_bucket_review_reuse || true

echo "===== [REST] STEP 3: omnirt_v1 retarget (KEEP_GOING) ====="
KEEP_GOING=1 $PY $DCV3/stages/s3_retarget/run_stage2b.py \
  --raw-contact-tsv "$PASS" --template-backlog-tsv "$S2OUT/template_backlog.tsv" \
  --template-review-tsv "$REVIEW_TSV" \
  --retarget-variant-registry "$R/registries/retarget_variant_registry.tsv" \
  --retarget-variant-id omnirt_v1 --target-variant-id ref_fk \
  --inventory-tsv "$R/s1_raw_contact/inventory/inventory.tsv" \
  --core4d-raw-root "$V2ROOT" --smplx-model-dir "$SMPLX_MODEL_DIR" \
  --out-dir "$V1OUT" --spider-dataset core4d_v2 --spider-source-dataset core4d \
  --execute --allow-legacy-stage2b-wrapper

echo "===== [REST] STEP 4: omnirt_v2 rescue on v1-infeasible ====="
RESCUE_TSV="$V2OUT/rest_rescue_raw_contact_pass_3cm.tsv"
mkdir -p "$V2OUT"
$PY - "$PASS" "$V1RES/pipeline_failed_cases.tsv" "$RESCUE_TSV" <<'PYEOF'
import sys
pass_tsv, failed_tsv, out_tsv = sys.argv[1:4]
failed=set()
try:
    for ln in open(failed_tsv):
        p=ln.rstrip("\n").split("\t")
        if len(p)>=3: failed.add((p[0],p[1],p[2]))
except FileNotFoundError: pass
with open(pass_tsv) as f:
    hdr=f.readline().rstrip("\n").split("\t")
    di,si,pi=hdr.index("date"),hdr.index("seq"),hdr.index("person")
    rows=[ln for ln in f if (lambda c:(c[di],c[si],c[pi]) in failed)(ln.rstrip("\n").split("\t"))]
open(out_tsv,"w").writelines(["\t".join(hdr)+"\n"]+rows)
print(f"rescue rows: {len(rows)} (from {len(failed)} v1-infeasible)")
PYEOF
if [ "$(($(wc -l < "$RESCUE_TSV") - 1))" -gt 0 ]; then
  KEEP_GOING=1 $PY $DCV3/stages/s3_retarget/run_stage2b.py \
    --raw-contact-tsv "$RESCUE_TSV" --template-backlog-tsv "$S2OUT/template_backlog.tsv" \
    --template-review-tsv "$REVIEW_TSV" \
    --retarget-variant-registry "$R/registries/retarget_variant_registry.tsv" \
    --retarget-variant-id omnirt_v2 --target-variant-id ref_fk \
    --inventory-tsv "$R/s1_raw_contact/inventory/inventory.tsv" \
    --core4d-raw-root "$V2ROOT" --smplx-model-dir "$SMPLX_MODEL_DIR" \
    --out-dir "$V2OUT" --spider-dataset core4d_v2 --spider-source-dataset core4d \
    --execute --allow-legacy-stage2b-wrapper
fi

echo "===== [REST] STEP 5: CEM (8-GPU) on all rest tasks ====="
$PY workspace/core4d/scripts/experiments/E203/run_e203_cem.py \
  --object-keys "$OBJS" --gpus "$GPUS" --max-per-gpu 1 \
  --num-samples 1024 --max-iterations 32 --out-dir "$R/s6_downstream/cem"

echo "===== [REST] DONE ====="
