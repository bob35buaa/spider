#!/usr/bin/env bash
# Batch CORE4D -> Holosoma retarget -> SPIDER preprocessing pipeline.
set -euo pipefail

REPO="${REPO:-$(git rev-parse --show-toplevel)}"
cd "$REPO"

HOLOSOMA_DIR="${HOLOSOMA_DIR:-/home/ubuntu/Workspace/holosoma}"
CORE4D_REAL_ROOT="${CORE4D_REAL_ROOT:-/mnt/a0ccc676-9496-49f8-a861-f8a1797dec52/mocap_data/CORE4D/CORE4D_Real}"
SMPLX_MODEL_DIR="${SMPLX_MODEL_DIR:-/mnt/a0ccc676-9496-49f8-a861-f8a1797dec52/mocap_data/smplx}"
RESULT_ROOT="${RESULT_ROOT:-workspace/core4d/results/data_preprocess}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
RETARGET_PYTHON_BIN="${RETARGET_PYTHON_BIN:-}"
REF_FPS="${REF_FPS:-30.0}"
EVAL_FPS="${EVAL_FPS:-50.0}"
TRIM_MODE="${TRIM_MODE:-holosoma}"
REPLACE_WRIST_WITH_FINGERTIP="${REPLACE_WRIST_WITH_FINGERTIP:-1}"
CASE_FILE="workspace/core4d/data_preprocess/cases_box023.tsv"
FORCE=0
DRY_RUN=0
SKIP_CONTACT=0
SKIP_RETARGET=0
SKIP_SPIDER=0

usage() {
  cat <<'USAGE'
Usage:
  bash workspace/core4d/data_preprocess/pipeline.sh [options]

Options:
  --case-file PATH   TSV case list. Default: workspace/core4d/data_preprocess/cases_box023.tsv
  --force            Re-run steps even if their expected outputs exist.
  --dry-run          Print commands without executing them.
  --skip-contact     Skip 3cm contact mask generation.
  --skip-retarget    Skip Holosoma convert + retarget + trim.
  --skip-spider      Skip SPIDER scene/data generation and verification.
  -h, --help         Show this message.

Environment overrides:
  HOLOSOMA_DIR, CORE4D_REAL_ROOT, SMPLX_MODEL_DIR, RESULT_ROOT, PYTHON_BIN,
  RETARGET_PYTHON_BIN, REF_FPS, EVAL_FPS, TRIM_MODE, REPLACE_WRIST_WITH_FINGERTIP, REPO

External absolute paths:
  HOLOSOMA_DIR       Absolute path to the Holosoma repo.
  CORE4D_REAL_ROOT   Absolute path to CORE4D_Real.
  SMPLX_MODEL_DIR    Absolute path to SMPL-X model files.
  These default to the local workstation paths and should be overridden on
  other machines.

Project-internal paths should stay relative to REPO:
  RESULT_ROOT, CASE_FILE, source_scene_task-derived scene path.

TRIM_MODE:
  holosoma   Use Holosoma workspace/pipeline/trim_no_contact.py (default).

REPLACE_WRIST_WITH_FINGERTIP:
  1          Preserve legacy convert behavior and pass --replace_wrist_with_fingertip (default).
  0          Use wrist targets directly; useful for medium-box H2/E091 ablations.
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    --case-file) CASE_FILE="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --skip-contact) SKIP_CONTACT=1; shift ;;
    --skip-retarget) SKIP_RETARGET=1; shift ;;
    --skip-spider) SKIP_SPIDER=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

run_cmd() {
  echo "+ $*"
  if [ "$DRY_RUN" -eq 0 ]; then
    "$@"
  fi
}

need_file() {
  local path=$1
  if [ ! -f "$path" ]; then
    echo "Missing required file: $path" >&2
    exit 1
  fi
}

need_dir() {
  local path=$1
  if [ ! -d "$path" ]; then
    echo "Missing required directory: $path" >&2
    exit 1
  fi
}

require_set() {
  local name=$1
  local value=$2
  if [ -z "$value" ]; then
    echo "Missing required environment variable: $name" >&2
    exit 2
  fi
}

require_absolute_path() {
  local name=$1
  local value=$2
  case "$value" in
    /*) ;;
    *)
      echo "$name must be an absolute path, got: $value" >&2
      exit 2
      ;;
  esac
}

require_relative_path() {
  local name=$1
  local value=$2
  case "$value" in
    /*)
      echo "$name must be relative to REPO, got absolute path: $value" >&2
      exit 2
      ;;
  esac
}

repo_path() {
  local path=$1
  case "$path" in
    /*) printf "%s\n" "$path" ;;
    *) printf "%s/%s\n" "$REPO" "$path" ;;
  esac
}

retarget_python() {
  if [ -n "${RETARGET_PYTHON_BIN:-}" ]; then
    printf "%s\n" "$RETARGET_PYTHON_BIN"
  elif [ -n "${CONDA_PREFIX:-}" ] && [ -x "$CONDA_PREFIX/bin/python" ]; then
    printf "%s\n" "$CONDA_PREFIX/bin/python"
  else
    command -v python
  fi
}

sync_generated_object_model() {
  local object_name=$1
  local src_dir="$REPO/src/holosoma_retargeting/holosoma_retargeting/models/$object_name"
  local dst_parent="$HOLOSOMA_DIR/src/holosoma_retargeting/holosoma_retargeting/models"
  if [ -d "$src_dir" ] && [ "$REPO" != "$HOLOSOMA_DIR" ]; then
    echo "+ sync generated object model: $src_dir -> $dst_parent/"
    if [ "$DRY_RUN" -eq 0 ]; then
      mkdir -p "$dst_parent"
      cp -a "$src_dir" "$dst_parent/"
    fi
  fi
}

ensure_g1_object_xml() {
  local object_name=$1
  local models_dir="$HOLOSOMA_DIR/src/holosoma_retargeting/holosoma_retargeting/models"
  local g1_dir="$models_dir/g1"
  local target_xml="$g1_dir/g1_29dof_w_${object_name}.xml"
  if [ -f "$target_xml" ]; then
    return 0
  fi
  local seed
  case "$object_name" in
    bucket*) seed="bucket001" ;;
    board*) seed="board007" ;;
    stick*) seed="stick003" ;;
    desk*) seed="desk005" ;;
    chair*) seed="chair006" ;;
    box*) seed="box004" ;;
    *) echo "missing g1 object XML and no template seed for $object_name" >&2; return 1 ;;
  esac
  local seed_xml="$g1_dir/g1_29dof_w_${seed}.xml"
  if [ ! -f "$seed_xml" ]; then
    echo "missing g1 object XML seed: $seed_xml" >&2
    return 1
  fi
  if [ ! -f "$models_dir/$object_name/$object_name.obj" ]; then
    echo "missing generated object mesh for g1 XML: $models_dir/$object_name/$object_name.obj" >&2
    return 1
  fi
  echo "+ generate missing g1 object XML: $target_xml from seed $seed"
  if [ "$DRY_RUN" -eq 0 ]; then
    "$PYTHON_BIN" - "$seed_xml" "$target_xml" "$seed" "$object_name" <<'PY'
from pathlib import Path
import sys

seed_xml = Path(sys.argv[1])
target_xml = Path(sys.argv[2])
seed = sys.argv[3]
name = sys.argv[4]

text = seed_xml.read_text(encoding="utf-8")
for old, new in [
    (seed, name),
    (seed.capitalize(), name.capitalize()),
    (seed.upper(), name.upper()),
]:
    text = text.replace(old, new)
target_xml.write_text(text, encoding="utf-8")
PY
    "$PYTHON_BIN" - "$target_xml" <<'PY'
import sys
import mujoco

mujoco.MjModel.from_xml_path(sys.argv[1])
PY
  fi
}

is_auto_value() {
  case "$1" in
    auto|-|-1) return 0 ;;
    *) return 1 ;;
  esac
}

require_set HOLOSOMA_DIR "$HOLOSOMA_DIR"
require_set CORE4D_REAL_ROOT "$CORE4D_REAL_ROOT"
require_set SMPLX_MODEL_DIR "$SMPLX_MODEL_DIR"
require_absolute_path HOLOSOMA_DIR "$HOLOSOMA_DIR"
require_absolute_path CORE4D_REAL_ROOT "$CORE4D_REAL_ROOT"
require_absolute_path SMPLX_MODEL_DIR "$SMPLX_MODEL_DIR"
require_relative_path RESULT_ROOT "$RESULT_ROOT"
require_relative_path CASE_FILE "$CASE_FILE"

if [ "$DRY_RUN" -eq 0 ]; then
  need_file "$CASE_FILE"
  need_file "$HOLOSOMA_DIR/scripts/source_retargeting_setup.sh"
  need_dir "$CORE4D_REAL_ROOT"
  need_dir "$SMPLX_MODEL_DIR"
fi

process_case() {
  local date=$1
  local seq=$2
  local person=$3
  local object_name=$4
  local object_model_rel=$5
  local source_scene_task=$6
  local target_task=$7
  local trim_start=$8
  local trim_frames=$9
  local data_id=${10}
  local mask_slug=${11}

  local core4d_motion_root="$CORE4D_REAL_ROOT/human_object_motions"
  local seq_dir="$core4d_motion_root/$date/$seq"
  local object_mesh="$CORE4D_REAL_ROOT/object_models/$object_model_rel"
  local case_root="$RESULT_ROOT/holosoma_${target_task}"
  local converted_dir="$case_root/converted"
  local retargeted_dir="$case_root/retargeted"
  local trimmed_dir="$case_root/trimmed"
  local task_name="${date}-${seq}-${person}-${object_name}_with_obj"
  local retargeted_npz="$retargeted_dir/${task_name}_original.npz"
  local trimmed_npz="$trimmed_dir/${task_name}_original.npz"
  local source_scene="example_datasets/processed/core4d/unitree_g1/humanoid_object/${source_scene_task}/scene.xml"
  local mask_out="$RESULT_ROOT/contact_masks/$mask_slug"
  local trim_info="$case_root/trim_window.json"
  local effective_trim_start="$trim_start"
  local effective_trim_frames="$trim_frames"
  local converted_abs
  local retargeted_abs
  local trimmed_abs
  converted_abs=$(repo_path "$converted_dir")
  retargeted_abs=$(repo_path "$retargeted_dir")
  trimmed_abs=$(repo_path "$trimmed_dir")

  echo
  echo "=== ${target_task}: ${date}/${seq} ${person} ${object_name} ==="

  if [ "$SKIP_RETARGET" -eq 0 ]; then
    run_cmd mkdir -p "$converted_dir" "$retargeted_dir" "$trimmed_dir"
    if [ "$FORCE" -eq 1 ] || [ ! -f "$converted_dir/${task_name}.npz" ]; then
      echo "+ source $HOLOSOMA_DIR/scripts/source_retargeting_setup.sh"
      if [ "$DRY_RUN" -eq 0 ]; then
        # shellcheck disable=SC1090
        source "$HOLOSOMA_DIR/scripts/source_retargeting_setup.sh"
      fi
      convert_args=(
        python "$HOLOSOMA_DIR/workspace/pipeline/convert_core4d_to_omniretarget.py" \
        --core4d_dir "$core4d_motion_root" \
        --smplx_model_dir "$SMPLX_MODEL_DIR" \
        --output_dir "$converted_dir" \
        --date "$date" \
        --seq "$seq" \
        --person "$person" \
        --with_object
      )
      if [ "$REPLACE_WRIST_WITH_FINGERTIP" = "1" ]; then
        convert_args+=(--replace_wrist_with_fingertip)
      fi
      convert_args[0]="$(retarget_python)"
      run_cmd "${convert_args[@]}"
    else
      echo "skip convert: $converted_dir/${task_name}.npz exists"
    fi
    sync_generated_object_model "$object_name"
    ensure_g1_object_xml "$object_name"

    if [ "$FORCE" -eq 1 ] || [ ! -f "$retargeted_npz" ]; then
      echo "+ source $HOLOSOMA_DIR/scripts/source_retargeting_setup.sh"
      if [ "$DRY_RUN" -eq 0 ]; then
        # shellcheck disable=SC1090
        source "$HOLOSOMA_DIR/scripts/source_retargeting_setup.sh"
      fi
      (
        cd "$HOLOSOMA_DIR/src/holosoma_retargeting/holosoma_retargeting"
        run_cmd "$(retarget_python)" examples/robot_retarget.py \
          --data_path "$converted_abs" \
          --task-type object_interaction \
          --task-name "$task_name" \
          --data_format smplx \
          --task-config.object-name "$object_name" \
          --save_dir "$retargeted_abs"
      )
    else
      echo "skip retarget: $retargeted_npz exists"
    fi

    if [ "$FORCE" -eq 1 ] || [ ! -f "$trimmed_npz" ]; then
      case "$TRIM_MODE" in
        holosoma)
          echo "+ source $HOLOSOMA_DIR/scripts/source_retargeting_setup.sh"
          if [ "$DRY_RUN" -eq 0 ]; then
            # shellcheck disable=SC1090
            source "$HOLOSOMA_DIR/scripts/source_retargeting_setup.sh"
          fi
          run_cmd "$(retarget_python)" "$HOLOSOMA_DIR/workspace/pipeline/trim_no_contact.py" \
            --input_dir "$retargeted_abs" \
            --output_dir "$trimmed_abs"
          ;;
        *)
          echo "Unsupported TRIM_MODE=$TRIM_MODE" >&2
          exit 2
          ;;
      esac
    else
      echo "skip Holosoma trim: $trimmed_npz exists"
    fi
  fi

  if ! is_auto_value "$effective_trim_start" && ! is_auto_value "$effective_trim_frames"; then
    if [ "$DRY_RUN" -eq 0 ]; then
      run_cmd "$PYTHON_BIN" workspace/core4d/data_preprocess/write_trim_window.py \
        --output-json "$trim_info" \
        --trim-start "$effective_trim_start" \
        --trim-frames "$effective_trim_frames" \
        --source "case-file:$CASE_FILE"
    fi
    echo "Case-file trim window: start=$effective_trim_start frames=$effective_trim_frames"
  elif [ "$DRY_RUN" -eq 0 ] && [ -f "$retargeted_npz" ] && [ -f "$trimmed_npz" ]; then
    infer_args=(
      --untrimmed "$retargeted_npz"
      --trimmed "$trimmed_npz"
      --output-json "$trim_info"
      --print-tsv
    )
    read -r effective_trim_start effective_trim_frames < <(
      "$PYTHON_BIN" workspace/core4d/data_preprocess/infer_holosoma_trim_window.py "${infer_args[@]}"
    )
    echo "Holosoma trim window: start=$effective_trim_start frames=$effective_trim_frames"
  elif is_auto_value "$effective_trim_start" || is_auto_value "$effective_trim_frames"; then
    if [ "$DRY_RUN" -eq 1 ]; then
      echo "dry-run: trim window will be inferred from Holosoma trimmed output"
    else
      echo "Cannot infer trim window; missing $retargeted_npz or $trimmed_npz" >&2
      exit 1
    fi
  fi

  if [ "$SKIP_CONTACT" -eq 0 ]; then
    if is_auto_value "$effective_trim_start" || is_auto_value "$effective_trim_frames"; then
      echo "skip contact mask in dry-run: trim window is auto/inferred"
    elif [ "$FORCE" -eq 1 ] || [ ! -f "$mask_out/raw_contact_mask_3cm.npz" ]; then
      run_cmd "$PYTHON_BIN" \
        workspace/core4d/data_preprocess/generate_core4d_contact_masks.py \
        --seq-dir "$seq_dir" \
        --mesh "$object_mesh" \
        --out-dir "$mask_out" \
        --trim-start "$effective_trim_start" \
        --spider-frames "$effective_trim_frames" \
        --ref-fps "$REF_FPS" \
        --eval-fps "$EVAL_FPS"
    else
      echo "skip contact mask: $mask_out/raw_contact_mask_3cm.npz exists"
    fi
  fi

  if [ "$SKIP_SPIDER" -eq 0 ]; then
    run_cmd "$PYTHON_BIN" \
      workspace/core4d/data_preprocess/create_spider_scene_from_template.py \
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

    run_cmd "$PYTHON_BIN" \
      workspace/core4d/data_preprocess/create_spider_scene_from_template.py \
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

    run_cmd "$PYTHON_BIN" \
      workspace/core4d/data_preprocess/verify_processed_case.py \
      --task "$target_task" \
      --source-scene "$source_scene" \
      --trimmed "$trimmed_npz" \
      --out "$RESULT_ROOT/${target_task}_verify_summary.json"
  fi
}

while IFS=$'\t' read -r enabled date seq person object_name object_model_rel source_scene_task target_task trim_start trim_frames data_id mask_slug; do
  enabled="${enabled//$'\r'/}"
  [ -z "${enabled:-}" ] && continue
  case "$enabled" in
    \#*) continue ;;
  esac
  [ "$enabled" = "1" ] || continue
  process_case "$date" "$seq" "$person" "$object_name" "$object_model_rel" "$source_scene_task" "$target_task" "$trim_start" "$trim_frames" "$data_id" "$mask_slug"
done < "$CASE_FILE"

echo
echo "Pipeline done."
