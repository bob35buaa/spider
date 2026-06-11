#!/usr/bin/env bash
# E090 Phase 1A: current no-fingertip and topface-preIK retarget production.
set -euo pipefail

REPO="${REPO:-$(git rev-parse --show-toplevel)}"
cd "$REPO"

HOLOSOMA_DIR="${HOLOSOMA_DIR:-/home/ubuntu/Workspace/holosoma}"
CORE4D_REAL_ROOT="${CORE4D_REAL_ROOT:-/mnt/a0ccc676-9496-49f8-a861-f8a1797dec52/mocap_data/CORE4D/CORE4D_Real}"
SMPLX_MODEL_DIR="${SMPLX_MODEL_DIR:-/mnt/a0ccc676-9496-49f8-a861-f8a1797dec52/mocap_data/smplx}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
HSRETARGETING_BIN="${HSRETARGETING_BIN:-/home/ubuntu/.holosoma_deps/miniconda3/envs/hsretargeting/bin}"
HS_PYTHON="${HS_PYTHON:-$HSRETARGETING_BIN/python}"
RESULT_ROOT="${RESULT_ROOT:-workspace/core4d/results/E090/retarget}"
CASE_FILE="${CASE_FILE:-workspace/core4d/scripts/E090/canonical_cases.tsv}"
GUARD_CASE_FILE="${GUARD_CASE_FILE:-workspace/core4d/scripts/E090/guard_cases.tsv}"
VARIANTS="${VARIANTS:-current_no_fingertip,topface_preik}"
FORCE=0
DRY_RUN=0
CASE_SET="canonical"
CASE_FILE_ARG=0
INCLUDE_ORIGINAL=0
ONLY_BASE_TASK=""
APPEND_MANIFEST=0

usage() {
  cat <<'USAGE'
Usage:
  bash workspace/core4d/scripts/E090/run_holosoma_retarget_ablation.sh [options]

Options:
  --case-set canonical|guard Case manifest preset. Default: canonical.
  --variants a,b             Comma-separated variants. Default: current_no_fingertip,topface_preik
  --case-file PATH           TSV case manifest. Default: workspace/core4d/scripts/E090/canonical_cases.tsv
  --only-base-task TASK      Run only one base_task from the case manifest.
  --append-manifest          Append to existing variants.tsv instead of overwriting it.
  --force                    Re-run outputs.
  --dry-run                  Print commands only.
  --include-original         Accepted for plan compatibility; Phase 1A ignores original variants.
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    --case-set) CASE_SET="$2"; shift 2 ;;
    --variants) VARIANTS="$2"; shift 2 ;;
    --case-file) CASE_FILE="$2"; CASE_FILE_ARG=1; shift 2 ;;
    --only-base-task) ONLY_BASE_TASK="$2"; shift 2 ;;
    --append-manifest) APPEND_MANIFEST=1; shift ;;
    --force) FORCE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --include-original) INCLUDE_ORIGINAL=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

case "$CASE_SET" in
  canonical) ;;
  guard)
    if [ "$CASE_FILE_ARG" -eq 0 ]; then
      CASE_FILE="$GUARD_CASE_FILE"
    fi
    ;;
  *)
    echo "Only --case-set canonical|guard is implemented in Phase 1A, got: $CASE_SET" >&2
    exit 2
    ;;
esac

run_cmd() {
  echo "+ $*"
  if [ "$DRY_RUN" -eq 0 ]; then
    "$@"
  fi
}

need_file() {
  [ -f "$1" ] || { echo "Missing required file: $1" >&2; exit 1; }
}

need_dir() {
  [ -d "$1" ] || { echo "Missing required directory: $1" >&2; exit 1; }
}

if [ "$DRY_RUN" -eq 0 ]; then
  need_file "$CASE_FILE"
  need_file "$HOLOSOMA_DIR/scripts/source_retargeting_setup.sh"
  need_file "$HS_PYTHON"
  need_dir "$CORE4D_REAL_ROOT"
  need_dir "$SMPLX_MODEL_DIR"
fi

export PATH="$HSRETARGETING_BIN:$PATH"

IFS=',' read -r -a VARIANT_LIST <<< "$VARIANTS"
VARIANT_TSV="workspace/core4d/results/E090/variants.tsv"
if [ "$DRY_RUN" -eq 0 ]; then
  mkdir -p "$RESULT_ROOT" "workspace/core4d/results/E090"
  if [ "$APPEND_MANIFEST" -eq 0 ] || [ ! -f "$VARIANT_TSV" ]; then
    printf "# variant\tbase_task\ttarget_task\tcase_root\tconverted_dir\tretargeted_dir\ttrimmed_dir\tsource_scene_task\n" > "$VARIANT_TSV"
  fi
fi

echo "E090 retarget ablation"
echo "  HOLOSOMA_DIR=$HOLOSOMA_DIR"
echo "  HS_PYTHON=$HS_PYTHON"
echo "  RESULT_ROOT=$RESULT_ROOT"
echo "  CASE_FILE=$CASE_FILE"
echo "  VARIANTS=$VARIANTS"
echo "  ONLY_BASE_TASK=${ONLY_BASE_TASK:-<all>}"
echo "  APPEND_MANIFEST=$APPEND_MANIFEST"

process_variant() {
  local variant=$1
  local date=$2
  local seq=$3
  local person=$4
  local object_name=$5
  local object_model_rel=$6
  local source_scene_task=$7
  local base_task=$8
  local data_id=$9

  local suffix
  case "$variant" in
    current_no_fingertip) suffix="nofing_e090" ;;
    topface_preik) suffix="btop_preik_e090" ;;
    *)
      echo "[SKIP] unsupported Phase 1A variant: $variant"
      return 0
      ;;
  esac

  local target_task="${base_task}_${suffix}"
  local task_name="${date}-${seq}-${person}-${object_name}_with_obj"
  local case_root="${RESULT_ROOT}/holosoma_${target_task}"
  local converted_dir="${case_root}/converted"
  local converted_input_dir="$converted_dir"
  local retargeted_dir="${case_root}/retargeted"
  local trimmed_dir="${case_root}/trimmed"
  local retargeted_npz="${retargeted_dir}/${task_name}_original.npz"
  local trimmed_npz="${trimmed_dir}/${task_name}_original.npz"
  local source_scene="example_datasets/processed/core4d/unitree_g1/humanoid_object/${source_scene_task}/scene.xml"

  echo
  echo "=== E090 ${variant}: ${target_task} ==="
  run_cmd mkdir -p "$converted_dir" "$retargeted_dir" "$trimmed_dir"

  if [ "$FORCE" -eq 1 ] || [ ! -f "${converted_dir}/${task_name}.npz" ]; then
    echo "+ source $HOLOSOMA_DIR/scripts/source_retargeting_setup.sh"
    if [ "$DRY_RUN" -eq 0 ]; then
      # shellcheck disable=SC1090
      source "$HOLOSOMA_DIR/scripts/source_retargeting_setup.sh"
    fi
    convert_args=(
      "$HOLOSOMA_DIR/workspace/pipeline/convert_core4d_to_omniretarget.py"
      --core4d_dir "$CORE4D_REAL_ROOT/human_object_motions"
      --smplx_model_dir "$SMPLX_MODEL_DIR"
      --output_dir "$converted_dir"
      --date "$date"
      --seq "$seq"
      --person "$person"
      --with_object
    )
    if [ "$variant" = "topface_preik" ]; then
      convert_args+=(--replace_wrist_with_fingertip)
    fi
    run_cmd "$HS_PYTHON" "${convert_args[@]}"
  else
    echo "skip convert: ${converted_dir}/${task_name}.npz exists"
  fi

  if [ "$variant" = "topface_preik" ]; then
    local topface_dir="${case_root}/converted_topface"
    if [ "$FORCE" -eq 1 ] || [ ! -f "${topface_dir}/${task_name}.npz" ]; then
      run_cmd "$PYTHON_BIN" workspace/core4d/scripts/E090/rewrite_wrist_top_face_preik.py \
        --input-dir "$converted_dir" \
        --output-dir "$topface_dir" \
        --task-name "$task_name" \
        --scene-xml "$source_scene" \
        --summary-json "${case_root}/topface_rewrite_summary.json"
    else
      echo "skip topface rewrite: ${topface_dir}/${task_name}.npz exists"
    fi
    converted_input_dir="$topface_dir"
  fi

  if [ "$FORCE" -eq 1 ] || [ ! -f "$retargeted_npz" ]; then
    echo "+ source $HOLOSOMA_DIR/scripts/source_retargeting_setup.sh"
    if [ "$DRY_RUN" -eq 0 ]; then
      # shellcheck disable=SC1090
      source "$HOLOSOMA_DIR/scripts/source_retargeting_setup.sh"
    fi
    (
      cd "$HOLOSOMA_DIR/src/holosoma_retargeting/holosoma_retargeting"
      run_cmd "$HS_PYTHON" examples/robot_retarget.py \
        --data_path "$REPO/$converted_input_dir" \
        --task-type object_interaction \
        --task-name "$task_name" \
        --data_format smplx \
        --task-config.object-name "$object_name" \
        --save_dir "$REPO/$retargeted_dir"
    )
  else
    echo "skip retarget: $retargeted_npz exists"
  fi

  if [ "$FORCE" -eq 1 ] || [ ! -f "$trimmed_npz" ]; then
    echo "+ source $HOLOSOMA_DIR/scripts/source_retargeting_setup.sh"
    if [ "$DRY_RUN" -eq 0 ]; then
      # shellcheck disable=SC1090
      source "$HOLOSOMA_DIR/scripts/source_retargeting_setup.sh"
    fi
    run_cmd "$HS_PYTHON" "$HOLOSOMA_DIR/workspace/pipeline/trim_no_contact.py" \
      --input_dir "$REPO/$retargeted_dir" \
      --output_dir "$REPO/$trimmed_dir"
  else
    echo "skip trim: $trimmed_npz exists"
  fi

  if [ "$DRY_RUN" -eq 0 ] && [ ! -f "$trimmed_npz" ]; then
    echo "Missing trimmed output after trim: $trimmed_npz" >&2
    exit 1
  fi

  run_cmd "$PYTHON_BIN" workspace/core4d/data_preprocess/create_spider_scene_from_template.py \
    --source-scene "$source_scene" \
    --task "$target_task" \
    --qpos "$trimmed_npz" \
    --data-id "$data_id" \
    --date "$date" \
    --seq "$seq" \
    --person "$person" \
    --object-name "$object_name" \
    --object-model-rel "$object_model_rel"

  run_cmd "$PYTHON_BIN" spider/process_datasets/core4d.py \
    --source-npz "$trimmed_npz" \
    --task "$target_task" \
    --data-id "$data_id" \
    --no-show-viewer \
    --no-save-video

  run_cmd "$PYTHON_BIN" workspace/core4d/data_preprocess/create_spider_scene_from_template.py \
    --source-scene "$source_scene" \
    --task "$target_task" \
    --qpos "$trimmed_npz" \
    --data-id "$data_id" \
    --date "$date" \
    --seq "$seq" \
    --person "$person" \
    --object-name "$object_name" \
    --object-model-rel "$object_model_rel" \
    --generate-scene-act

  run_cmd "$PYTHON_BIN" workspace/core4d/data_preprocess/verify_processed_case.py \
    --task "$target_task" \
    --source-scene "$source_scene" \
    --trimmed "$trimmed_npz" \
    --out "${case_root}/${target_task}_verify_summary.json"

  if [ "$DRY_RUN" -eq 0 ]; then
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$variant" "$base_task" "$target_task" "$case_root" "$converted_input_dir" "$retargeted_dir" "$trimmed_dir" "$source_scene_task" >> "$VARIANT_TSV"
  fi
}

while IFS=$'\t' read -r enabled date seq person object_name object_model_rel source_scene_task base_task trim_start trim_frames data_id mask_slug; do
  enabled="${enabled//$'\r'/}"
  [ -z "${enabled:-}" ] && continue
  case "$enabled" in
    \#*) continue ;;
  esac
  [ "$enabled" = "1" ] || continue
  if [ -n "$ONLY_BASE_TASK" ] && [ "$base_task" != "$ONLY_BASE_TASK" ]; then
    continue
  fi
  for variant in "${VARIANT_LIST[@]}"; do
    process_variant "$variant" "$date" "$seq" "$person" "$object_name" "$object_model_rel" "$source_scene_task" "$base_task" "$data_id"
  done
done < "$CASE_FILE"

echo
echo "E090 Phase 1A done. Variant manifest: $VARIANT_TSV"
