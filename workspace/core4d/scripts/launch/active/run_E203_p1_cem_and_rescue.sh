#!/usr/bin/env bash
# E203 P1 auto-chain: CEM(omnirt_v1 feasible) -> build rescue set (v1 infeasible)
# -> omnirt_v2 (Phase-4) retarget rescue -> CEM(omnirt_v2). All on core4d_v2.
# Run AFTER P1 S3 (omnirt_v1) retarget completes.
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
OBJS="${OBJS:-box001,box004,box024}"          # P1 objects
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
V1_S3="$R/s3_retarget/omnirt_v1/ref_fk/results/omnirt_v1_ref_fk"
V2_S3="$R/s3_retarget/omnirt_v2/ref_fk"
FAILED="$V1_S3/pipeline_failed_cases.tsv"
RESCUE_TSV="$R/s3_retarget/omnirt_v2/ref_fk_rescue_raw_contact_pass_3cm.tsv"
PASS="$R/s1_raw_contact/raw_contact/raw_contact_pass_3cm.tsv"

echo "===== [E203 P1] STEP 1: CEM on omnirt_v1 feasible tasks ($OBJS) ====="
$PY workspace/core4d/scripts/experiments/E203/run_e203_cem.py \
  --object-keys "$OBJS" --gpus "$GPUS" --max-per-gpu 1 \
  --num-samples 1024 --max-iterations 32 --out-dir "$R/s6_downstream/cem"

echo "===== [E203 P1] STEP 2: build omnirt_v2 rescue set from v1 infeasible ====="
mkdir -p "$V2_S3"
$PY - "$PASS" "$FAILED" "$RESCUE_TSV" <<'PYEOF'
import sys
pass_tsv, failed_tsv, out_tsv = sys.argv[1:4]
# dedup (date,seq,person) of v1 failures
failed = set()
try:
    with open(failed_tsv) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 3:
                failed.add((p[0], p[1], p[2]))
except FileNotFoundError:
    pass
with open(pass_tsv) as f:
    header = f.readline().rstrip("\n").split("\t")
    di, si, pi = header.index("date"), header.index("seq"), header.index("person")
    rows = [ln for ln in f if (lambda c: (c[di], c[si], c[pi]) in failed)(ln.rstrip("\n").split("\t"))]
with open(out_tsv, "w") as f:
    f.write("\t".join(header) + "\n")
    f.writelines(rows)
print(f"rescue set: {len(rows)} rows (from {len(failed)} unique v1-infeasible cases)")
PYEOF

RESCUE_N=$(($(wc -l < "$RESCUE_TSV") - 1))
echo "rescue rows = $RESCUE_N"
if [ "$RESCUE_N" -le 0 ]; then
  echo "no rescue cases; skipping omnirt_v2"
else
  echo "===== [E203 P1] STEP 3: omnirt_v2 (Phase-4) retarget on rescue set ====="
  KEEP_GOING=1 $PY workspace/core4d/scripts/data_construction_v3/stages/s3_retarget/run_stage2b.py \
    --raw-contact-tsv "$RESCUE_TSV" \
    --template-backlog-tsv "$R/s2_templates/template_backlog.tsv" \
    --retarget-variant-registry "$R/registries/retarget_variant_registry.tsv" \
    --retarget-variant-id omnirt_v2 --target-variant-id ref_fk \
    --inventory-tsv "$R/s1_raw_contact/inventory/inventory.tsv" \
    --core4d-raw-root "$V2ROOT" --smplx-model-dir "$SMPLX_MODEL_DIR" \
    --out-dir "$V2_S3" \
    --spider-dataset core4d_v2 --spider-source-dataset core4d \
    --execute --allow-legacy-stage2b-wrapper

  echo "===== [E203 P1] STEP 4: CEM on omnirt_v2 rescued tasks ====="
  # driver derives the v2 contact-mask root from the task name (dcv3_omnirt_v2_...).
  $PY workspace/core4d/scripts/experiments/E203/run_e203_cem.py \
    --object-keys "$OBJS" --gpus "$GPUS" --max-per-gpu 1 \
    --num-samples 1024 --max-iterations 32 --out-dir "$R/s6_downstream/cem"
fi

echo "===== [E203 P1] DONE. CEM summary: ====="
cat "$R/s6_downstream/cem/e203_cem_summary.json" | $PY -c "import sys,json,collections; d=json.load(sys.stdin); print(collections.Counter(r['status'] for r in d))"
