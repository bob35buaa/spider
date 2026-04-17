#!/bin/bash
# Rerun missing benchmark jobs
export PATH=$HOME/.local/bin:$PATH
export MUJOCO_GL=egl
cd ~/spider

RESULTS_DIR="benchmark_logs"
mkdir -p "$RESULTS_DIR"
RESULTS_FILE="benchmark_results.csv"

run_one() {
    local gpu=$1
    local script=$2
    local override=$3
    local task=$4
    local data_id=$5
    local method=$6
    local dataset=$7
    local logfile="$RESULTS_DIR/${method}_${dataset}_${task}_${data_id}.log"

    echo "=== [GPU$gpu] $method | $dataset/$task/$data_id ==="
    START_T=$(date +%s)
    CUDA_VISIBLE_DEVICES=$gpu uv run "$script" +override="$override" task="$task" data_id="$data_id" \
        robot_type=xhand embodiment_type=bimanual viewer=none save_video=false save_info=false \
        > "$logfile" 2>&1
    END_T=$(date +%s)
    ELAPSED=$((END_T - START_T))

    POS_ERR=$(grep -oP 'pos=\K[0-9.]+' "$logfile" | tail -1)
    QUAT_ERR=$(grep -oP 'quat=\K[0-9.]+' "$logfile" | tail -1)
    RETRY_WARNS=$(grep -c "tracking error exceeded" "$logfile")
    FAIL_ERRS=$(grep -c "Failed to find feasible" "$logfile")
    echo "$method,$dataset,$task,$data_id,${POS_ERR:-NA},${QUAT_ERR:-NA},$ELAPSED,$RETRY_WARNS,$FAIL_ERRS" >> "$RESULTS_FILE"
    echo "  [GPU$gpu] $method $dataset/$task -> pos=$POS_ERR, quat=$QUAT_ERR, time=${ELAPSED}s, retries=$RETRY_WARNS, failures=$FAIL_ERRS"
}

# Batch 1: 4 jobs on 4 GPUs
# GPU0: baseline pour_tube
# GPU1: baseline uncap_alcohol_burner
# GPU2: baseline unplug
# GPU3: fast pick_spoon_bowl
run_one 0 examples/run_mjwp.py oakink_fast pour_tube 0 baseline oakink &
run_one 1 examples/run_mjwp.py oakink_fast uncap_alcohol_burner 0 baseline oakink &
run_one 2 examples/run_mjwp.py oakink_fast unplug 0 baseline oakink &
run_one 3 examples/run_mjwp_fast.py oakink_fast_new pick_spoon_bowl 0 fast oakink &
wait

# Batch 2: 3 fast jobs on 3 GPUs
# GPU0: fast pour_tube
# GPU1: fast uncap_alcohol_burner
# GPU2: fast unplug
run_one 0 examples/run_mjwp_fast.py oakink_fast_new pour_tube 0 fast oakink &
run_one 1 examples/run_mjwp_fast.py oakink_fast_new uncap_alcohol_burner 0 fast oakink &
run_one 2 examples/run_mjwp_fast.py oakink_fast_new unplug 0 fast oakink &
wait

echo ""
echo "=== ALL RESULTS ==="
sort -t',' -k2,2 -k3,3 -k1,1 "$RESULTS_FILE" | column -t -s','
