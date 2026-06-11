#!/bin/bash
# E046: Batch convert + retarget 5 new CORE4D cases
# Step 1: convert_core4d_to_omniretarget.py (SMPLX → OmniRetarget input)
# Step 2: robot_retarget.py (OmniRetarget → robot qpos)
set -e

HOLOSOMA_DIR="/home/ubuntu/Workspace/holosoma"
CONVERT_SCRIPT="$HOLOSOMA_DIR/workspace/pipeline/convert_core4d_to_omniretarget.py"
RETARGET_DIR="$HOLOSOMA_DIR/src/holosoma_retargeting/holosoma_retargeting"
CORE4D_DIR="/mnt/a0ccc676-9496-49f8-a861-f8a1797dec52/mocap_data/CORE4D/CORE4D_Real/human_object_motions"
SMPLX_MODEL_DIR="/mnt/a0ccc676-9496-49f8-a861-f8a1797dec52/mocap_data/smplx"
OUTPUT_DIR="$HOLOSOMA_DIR/workspace/pipeline/results/e046_new_cases"
CONVERTED_DIR="$OUTPUT_DIR/converted"
RETARGET_OUT="$OUTPUT_DIR/retargeted"

mkdir -p "$CONVERTED_DIR" "$RETARGET_OUT"

# 5 candidate sessions: date/seq object_name
CASES=(
    "20231008 010 Board020"
    "20231030 094 bucket001"
    "20231018 030 Box021"
    "20231023 037 Desk021"
    "20231030 028 desk007"
)

echo "=== Step 1: Convert CORE4D → OmniRetarget format ==="
for case in "${CASES[@]}"; do
    read -r date seq obj <<< "$case"
    echo ">>> Converting $date/$seq ($obj) ..."
    python "$CONVERT_SCRIPT" \
        --core4d_dir "$CORE4D_DIR" \
        --smplx_model_dir "$SMPLX_MODEL_DIR" \
        --output_dir "$CONVERTED_DIR" \
        --date "$date" --seq "$seq" --person person1 \
        --with_object --replace_wrist_with_fingertip
    echo ""
done

echo "=== Step 2: OmniRetarget retarget ==="
# Source the retargeting environment
source "$HOLOSOMA_DIR/scripts/source_retargeting_setup.sh" 2>/dev/null || true

for case in "${CASES[@]}"; do
    read -r date seq obj <<< "$case"
    task_name="${date}-${seq}-person1-${obj}_with_obj"
    echo ">>> Retargeting $task_name ..."

    # Check if object model exists, copy if needed
    obj_lower=$(echo "$obj" | tr '[:upper:]' '[:lower:]')
    cat_lower="${obj_lower//[0-9]/}"
    models_dir="$RETARGET_DIR/models/$obj"
    if [ ! -d "$models_dir" ]; then
        mkdir -p "$models_dir"
        src_mesh="$CORE4D_DIR/../../object_models/$cat_lower/${obj_lower}_m.obj"
        cp "$src_mesh" "$models_dir/$obj.obj"
        echo "  Copied mesh to $models_dir/$obj.obj"
        # Generate URDF
        cat > "$models_dir/$obj.urdf" << URDF
<?xml version="1.0" ?>
<robot name="$obj">
  <link name="base_link">
    <visual>
      <geometry><mesh filename="$obj.obj"/></geometry>
    </visual>
    <collision>
      <geometry><mesh filename="$obj.obj"/></geometry>
    </collision>
  </link>
</robot>
URDF
        echo "  Generated URDF"
    fi

    # Compute relative data_path from RETARGET_DIR
    cd "$RETARGET_DIR"
    python examples/robot_retarget.py \
        --data_path "$CONVERTED_DIR" \
        --task-type object_interaction --task-name "$task_name" \
        --data_format smplx --task-config.object-name "$obj" \
        --save_dir "$RETARGET_OUT" \
        || echo "  FAILED: $task_name"
    cd -
    echo ""
done

echo "=== Done! Results in $RETARGET_OUT ==="
ls -la "$RETARGET_OUT"/*.npz 2>/dev/null || echo "No results generated"
