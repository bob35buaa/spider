#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
TAG="${1:-full}"
ROOT="workspace/core4d/results/E194"
REMOTE="${E194_ADA_REMOTE:-spider-remote}"
POINTER="$ROOT/remote_staging/latest_${TAG}_ada6000.json"
case "$TAG" in
  canary) CANONICAL="$ROOT/s6_downstream/manifests/g1_expansion_canary_manifest.tsv" ;;
  sentinel) CANONICAL="$ROOT/s6_downstream/manifests/g1_expansion_sentinel_manifest.tsv" ;;
  full) CANONICAL="$ROOT/s6_downstream/manifests/g1_expansion_full_manifest.tsv" ;;
  *) echo "unsupported tag: $TAG" >&2; exit 2 ;;
esac
[ -f "$POINTER" ] || { echo "missing pointer: $POINTER" >&2; exit 2; }
mapfile -t META < <("$PYTHON_BIN" - "$POINTER" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); print(d["remote_root"]); print(d["stage"])
PY
)
REMOTE_ROOT="${META[0]}"; STAGE="${META[1]}"
RSH="ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=4"
mkdir -p "$STAGE" "logs/E194/cem/${TAG}_g1_expansion"
rsync -az -e "$RSH" "$REMOTE:$REMOTE_ROOT/$STAGE/" "$STAGE/"
mapfile -t PATHS < <("$PYTHON_BIN" - "$STAGE" <<'PY'
import csv,sys
from pathlib import Path
files=set()
for path in Path(sys.argv[1]).glob("ada-gpu*.tsv"):
 for row in csv.DictReader(path.open(),delimiter="\t"):
  files.update(row[key] for key in ("result_npz","outdir_npz","config_act","log"))
print("\n".join(sorted(files)))
PY
)
for path in "${PATHS[@]}"; do
  if ssh -o BatchMode=yes -o ConnectTimeout=20 "$REMOTE" "test -f '$REMOTE_ROOT/$path'"; then
    mkdir -p "$(dirname "$path")"; rsync -az -e "$RSH" "$REMOTE:$REMOTE_ROOT/$path" "$path"
  else
    echo "pull skip missing: $path"
  fi
done
rsync -az -e "$RSH" "$REMOTE:$REMOTE_ROOT/logs/E194/cem/${TAG}_g1_expansion/" "logs/E194/cem/${TAG}_g1_expansion/" || true
"$PYTHON_BIN" - "$STAGE" "$CANONICAL" "$ROOT/remote_staging/local-gpu0/$TAG/manifest.tsv" "$TAG" <<'PY'
import csv,sys
from collections import Counter
from pathlib import Path
stage,canonical,local,tag=Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]),sys.argv[4]
rows=list(csv.DictReader(canonical.open(),delimiter="\t")); fields=list(rows[0]); by={row["variant"]:row for row in rows}
shards=list(stage.glob("ada-gpu*.tsv")); shards += [local] if local.is_file() else []
for shard in shards:
 for row in csv.DictReader(shard.open(),delimiter="\t"):
  if row["variant"] in by: by[row["variant"]].update(row)
with canonical.open("w",newline="") as stream:
 w=csv.DictWriter(stream,fieldnames=fields,delimiter="\t",lineterminator="\n"); w.writeheader(); w.writerows(rows)
print(f"merged {tag}: {dict(Counter(row['status'] for row in rows))}")
if tag=="sentinel":
 full=Path("workspace/core4d/results/E194/s6_downstream/manifests/g1_expansion_full_manifest.tsv")
 frows=list(csv.DictReader(full.open(),delimiter="\t")); ffields=list(frows[0]); done={row["variant"]:row for row in rows}
 for row in frows:
  if row["variant"] in done: row.update(done[row["variant"]])
 with full.open("w",newline="") as stream:
  w=csv.DictWriter(stream,fieldnames=ffields,delimiter="\t",lineterminator="\n",extrasaction="ignore"); w.writeheader(); w.writerows(frows)
 print(f"updated Full sentinel state: {dict(Counter(row['status'] for row in frows))}")
PY
echo "E194 G1 $TAG pull complete"
