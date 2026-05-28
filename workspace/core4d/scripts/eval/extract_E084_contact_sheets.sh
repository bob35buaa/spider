#!/usr/bin/env bash
# Build compact E084 frame sheets from extracted keyframes.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

RESULTS="${RESULTS:-workspace/core4d/results/E084}"
VARIANTS_FILE="${VARIANTS_FILE:-workspace/core4d/scripts/E084/variants.tsv}"
OUT="$RESULTS/keyframes/contact_sheets"
mkdir -p "$OUT"

make_sheet() {
  local variant=$1
  local dir="$RESULTS/keyframes/$variant"
  local output="$OUT/${variant}_sheet.jpg"
  [ -d "$dir" ] || return 0
  mapfile -t frames < <(for f in 16 32 50 75 90 100 115 120 125 145 160 180 204; do
    [ -f "$dir/f${f}.jpg" ] && echo "$dir/f${f}.jpg"
  done)
  [ "${#frames[@]}" -gt 0 ] || return 0
  ffmpeg -nostdin -y -loglevel error \
    -pattern_type glob -i "$dir/f*.jpg" \
    -vf "scale=320:-1,tile=5x3:padding=4:margin=4:color=white" \
    -frames:v 1 "$output"
  echo "$output"
}

mapfile -t variants < <(awk -F '\t' 'NF && $1 !~ /^#/ {print $1}' "$VARIANTS_FILE")
for variant in "${variants[@]}"; do
  make_sheet "$variant"
done

if ls "$OUT"/*_sheet.jpg >/dev/null 2>&1; then
  ffmpeg -nostdin -y -loglevel error \
    -pattern_type glob -i "$OUT/*_sheet.jpg" \
    -vf "scale=960:-1,tile=1x6:padding=8:margin=8:color=white" \
    -frames:v 1 "$OUT/E084_all_cases_sheet.jpg" || true
fi

echo "Done. Sheets in $OUT"
