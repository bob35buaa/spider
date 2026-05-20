#!/bin/bash
# Batch holosoma v2 retarget for the 10 remaining cases needed to upgrade
# Tab.5 (E019) from N=3 to N=13. Supports SHARDING across multiple workers
# / multiple machines (output dirs are on shared NFS at /mnt/ali-sh-1/...).
#
# Holosoma retarget is CPU-bound (scipy.sparse + cvxpy + clarabel SOCP, no
# GPU). So "2 machines × 2 GPUs" really means 4 CPU workers. GPU id is just
# a worker tag.
#
# Prereq:
#   - HOLOSOMA_DEPS_DIR set (conda envs root)
#   - CORE4D raw data at /mnt/.../CORE4D_Real/human_object_motions/{date}/{seq}/
#   - g1_29dof_w_{Box021,Box023,Bucket007}.xml templates already generated
#     (sed from Box025; done 2026-05-20).
#
# Usage (3 phases — convert must be done before parallel retarget, trim
# must be done after all retarget finishes):
#
#   # Phase 1: convert + URDF/mesh setup (idempotent, ~30s, run ONCE on any machine)
#   PHASE=convert bash run_holosoma_batch_remaining10.sh
#
#   # Phase 2: parallel retarget — open 4 shells across machines
#   PHASE=retarget SHARD_COUNT=4 SHARD_ID=0 bash run_holosoma_batch_remaining10.sh  # machineA worker0
#   PHASE=retarget SHARD_COUNT=4 SHARD_ID=1 bash run_holosoma_batch_remaining10.sh  # machineA worker1
#   PHASE=retarget SHARD_COUNT=4 SHARD_ID=2 bash run_holosoma_batch_remaining10.sh  # machineB worker0
#   PHASE=retarget SHARD_COUNT=4 SHARD_ID=3 bash run_holosoma_batch_remaining10.sh  # machineB worker1
#
#   # Phase 3: trim leading no-contact frames (run ONCE after all 4 retarget shards done)
#   PHASE=trim bash run_holosoma_batch_remaining10.sh
#
# Runtime per phase (CPU-bound, single worker baseline):
#   convert: ~30s for 10 cases (sequential, mostly IO)
#   retarget: ~1.5min/case × 10 = ~15min single, ~4min with 4 workers
#   trim: ~30s for 10 cases (sequential)
#
# After: pull `CASE_MAP` snippet printed at end of phase=trim into
#   spider/workspace/core4d_collab_retarget/scripts/eval/adapters/kinematic_to_common.py,
# then run eval_holosoma_kinematic.py --all + unified_eval.py.

set -e

: "${HOLOSOMA_DEPS_DIR:?Set HOLOSOMA_DEPS_DIR before running}"
: "${PHASE:?Set PHASE=convert | retarget | trim}"

PYHS="$HOLOSOMA_DEPS_DIR/miniconda3/envs/hsretargeting/bin/python"
HSRT="/mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/holosoma"
RETARGET_DIR="$HSRT/src/holosoma_retargeting/holosoma_retargeting"
OUT_CONVERT="$HSRT/workspace/v2/data/core4d_replace_batch_extra"
OUT_RETARGET="$HSRT/workspace/v2/results/retarget_replace_batch_extra"
OUT_TRIMMED="$HSRT/workspace/v2/results/retarget_replace_batch_extra_trimmed"
DATA_LINK="$RETARGET_DIR/demo_data/core4d_replace_batch_extra"

# Optional: cap OpenMP threads per worker so 4 concurrent processes don't
# fight for cores. Defaults to 4 (assumes >= 16 cores total).
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

mkdir -p "$OUT_CONVERT" "$OUT_RETARGET" "$OUT_TRIMMED" "$DATA_LINK"

# 10 cases: date seq person Object  (-> spider_case)
CASES=(
    "20231018 030 person2 Box021"     # box021_p2
    "20231008 045 person1 Box023"     # box023_p1
    "20231008 045 person2 Box023"     # box023_p2
    "20231030 094 person1 bucket001"  # bucket001_p1
    "20231030 094 person2 bucket001"  # bucket001_p2
    "20231002 004 person1 bucket005"  # bucket005_s2_p1
    "20231002 004 person2 bucket005"  # bucket005_s2_p2
    "20231020 055 person1 Bucket007"  # bucket007_p1
    "20231020 055 person2 Bucket007"  # bucket007_p2
    "20231108 055 person1 Desk021"    # desk021_p1
)

# Shard selection for PHASE=retarget. Worker i gets cases at idx i, i+N, i+2N, ...
SHARD_COUNT="${SHARD_COUNT:-1}"
SHARD_ID="${SHARD_ID:-0}"

my_cases=()
for i in "${!CASES[@]}"; do
    if [ "$PHASE" = "retarget" ]; then
        if (( i % SHARD_COUNT == SHARD_ID )); then
            my_cases+=("${CASES[$i]}")
        fi
    else
        my_cases+=("${CASES[$i]}")
    fi
done

# ============================================================
# Phase 1: convert
# ============================================================
if [ "$PHASE" = "convert" ]; then
    cd "$HSRT"
    echo "=== convert (10 cases, sequential, idempotent) ==="
    for case in "${my_cases[@]}"; do
        read -r date seq person obj <<< "$case"
        tag="${date}-${seq}-${person}-${obj}_with_obj"
        if [ -f "$OUT_CONVERT/${tag}.npz" ] && [ -f "$DATA_LINK/${tag}.npz" ]; then
            echo "  skip $tag (already converted)"
            continue
        fi
        echo "  convert $tag ..."
        $PYHS workspace/pipeline/convert_core4d_to_omniretarget.py \
            --date "$date" --seq "$seq" --person "$person" --with_object \
            --replace_wrist_with_fingertip \
            --output_dir "$OUT_CONVERT" 2>&1 | tail -1
        cp "$OUT_CONVERT/${tag}.npz" "$DATA_LINK/"
    done
    echo "convert done."
    exit 0
fi

# ============================================================
# Phase 2: retarget (per-shard)
# ============================================================
if [ "$PHASE" = "retarget" ]; then
    cd "$RETARGET_DIR"
    echo "=== retarget shard $SHARD_ID/$SHARD_COUNT (${#my_cases[@]} cases on this worker) ==="
    echo "    OMP_NUM_THREADS=$OMP_NUM_THREADS  MKL_NUM_THREADS=$MKL_NUM_THREADS"
    for case in "${my_cases[@]}"; do
        read -r date seq person obj <<< "$case"
        tag="${date}-${seq}-${person}-${obj}_with_obj"
        out_npz="$OUT_RETARGET/${tag}_original.npz"
        if [ -f "$out_npz" ]; then
            echo "  [shard $SHARD_ID] skip $tag (already retargeted)"
            continue
        fi
        # Fallback: auto-convert if NPZ missing in DATA_LINK (e.g. user
        # skipped PHASE=convert, or NFS sync lagged). Convert is idempotent.
        if [ ! -f "$DATA_LINK/${tag}.npz" ]; then
            echo "  [shard $SHARD_ID] convert $tag (fallback, demo_data missing) ..."
            (cd "$HSRT" && $PYHS workspace/pipeline/convert_core4d_to_omniretarget.py \
                --date "$date" --seq "$seq" --person "$person" --with_object \
                --replace_wrist_with_fingertip \
                --output_dir "$OUT_CONVERT" 2>&1 | tail -1)
            cp "$OUT_CONVERT/${tag}.npz" "$DATA_LINK/"
        fi
        echo "  [shard $SHARD_ID] retarget $tag ..."
        time $PYHS examples/robot_retarget.py \
            --data_path demo_data/core4d_replace_batch_extra \
            --task-type object_interaction \
            --task-name "$tag" \
            --data_format smplx \
            --task-config.object-name "$obj" \
            --save_dir "$OUT_RETARGET" 2>&1 | tail -3
    done
    echo "retarget shard $SHARD_ID done."
    exit 0
fi

# ============================================================
# Phase 3: trim
# ============================================================
if [ "$PHASE" = "trim" ]; then
    cd "$HSRT"
    echo "=== trim leading no-contact frames ==="
    $PYHS workspace/pipeline/trim_no_contact.py \
        --input_dir "$OUT_RETARGET" \
        --output_dir "$OUT_TRIMMED" 2>&1 | tail -20

    echo ""
    echo "=== Done. Trimmed outputs: ==="
    ls -la "$OUT_TRIMMED/"
    echo ""
    echo "=== Append these 10 entries to CASE_MAP ==="
    for case in "${CASES[@]}"; do
        read -r date seq person obj <<< "$case"
        sp_p="${person/person/p}"
        obj_lc=$(echo "$obj" | tr '[:upper:]' '[:lower:]')
        case "$obj_lc" in
            box021) sp_case="box021_${sp_p}" ;;
            box023) sp_case="box023_${sp_p}" ;;
            bucket001) sp_case="bucket001_${sp_p}" ;;
            bucket005) sp_case="bucket005_s2_${sp_p}" ;;
            bucket007) sp_case="bucket007_${sp_p}" ;;
            desk021) sp_case="desk021_${sp_p}" ;;
            *) sp_case="UNKNOWN_${obj_lc}_${sp_p}" ;;
        esac
        echo "    \"${sp_case}\": \"${date}-${seq}-${person}-${obj}_with_obj_original.npz\","
    done
    exit 0
fi

echo "ERROR: unknown PHASE=$PHASE (use convert | retarget | trim)"
exit 1
