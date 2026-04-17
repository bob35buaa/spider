#!/bin/bash
# Benchmark run_mjwp.py (baseline) vs run_mjwp_fast.py across all oakink and gigahand tasks
# Parallelized across 4 GPUs
export PATH=$HOME/.local/bin:$PATH
export MUJOCO_GL=egl
cd ~/spider

RESULTS_DIR="benchmark_logs"
mkdir -p "$RESULTS_DIR"
RESULTS_FILE="benchmark_results.csv"
echo "method,dataset,task,data_id,pos_err,quat_err,time_s,retry_warnings,fail_errors" > "$RESULTS_FILE"

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

# Build job list: (gpu script override task data_id method dataset)
# We have 20 jobs (10 tasks x 2 methods). Assign to 4 GPUs round-robin.
JOBS=()
# OakInk tasks
for task in lift_board pick_spoon_bowl pour_tube stir_beaker uncap_alcohol_burner unplug wipe_board; do
    JOBS+=("examples/run_mjwp.py oakink_fast $task 0 baseline oakink")
    JOBS+=("examples/run_mjwp_fast.py oakink_fast_new $task 0 fast oakink")
done
# GigaHand tasks
for task in p36-tea p44-dog p52-instrument; do
    JOBS+=("examples/run_mjwp.py gigahand_fast $task 0 baseline gigahand")
    JOBS+=("examples/run_mjwp_fast.py gigahand_fast_new $task 0 fast gigahand")
done

NUM_GPUS=4

# Run jobs in batches of NUM_GPUS
for ((i=0; i<${#JOBS[@]}; i+=NUM_GPUS)); do
    pids=()
    for ((j=0; j<NUM_GPUS && i+j<${#JOBS[@]}; j++)); do
        gpu=$j
        job="${JOBS[$((i+j))]}"
        # Split job string into args
        read -r script override task data_id method dataset <<< "$job"
        run_one $gpu "$script" "$override" "$task" "$data_id" "$method" "$dataset" &
        pids+=($!)
    done
    # Wait for batch to complete
    for pid in "${pids[@]}"; do
        wait $pid
    done
done

echo ""
echo "=== RESULTS ==="
# Sort by dataset, task, method for easy comparison
sort -t',' -k2,2 -k3,3 -k1,1 "$RESULTS_FILE" | column -t -s','
