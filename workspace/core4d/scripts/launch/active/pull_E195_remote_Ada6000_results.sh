#!/usr/bin/env bash
# Pull E195 Ada artifacts and merge remote/local shard state into the canonical manifest.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
ROOT="workspace/core4d/results/E195"
REMOTE="${E195_ADA_REMOTE:-spider-remote}"
POINTER="$ROOT/remote_staging/latest_full_ada6000.json"
[ -f "$POINTER" ] || { echo "missing pointer: $POINTER" >&2; exit 2; }

mapfile -t META < <("$PYTHON_BIN" - "$POINTER" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
print(data["remote_root"]); print(data["stage"])
PY
)
REMOTE_ROOT="${META[0]}"
STAGE="${META[1]}"
RSH="ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=4"
mkdir -p "$STAGE" "logs/E195/cem/full"
rsync -az -e "$RSH" "$REMOTE:$REMOTE_ROOT/$STAGE/" "$STAGE/"

mapfile -t PATHS < <("$PYTHON_BIN" - "$STAGE" <<'PY'
import csv, sys
from pathlib import Path
files = set()
for path in Path(sys.argv[1]).glob("ada-gpu*.tsv"):
    for row in csv.DictReader(path.open(), delimiter="\t"):
        files.update(row[key] for key in ("result_npz", "outdir_npz", "config_act", "log"))
print("\n".join(sorted(files)))
PY
)
for path in "${PATHS[@]}"; do
  if ssh -o BatchMode=yes -o ConnectTimeout=20 "$REMOTE" "test -f '$REMOTE_ROOT/$path'"; then
    mkdir -p "$(dirname "$path")"
    rsync -az -e "$RSH" "$REMOTE:$REMOTE_ROOT/$path" "$path"
  else
    echo "E195 pull skip missing artifact: $path"
  fi
done
rsync -az -e "$RSH" "$REMOTE:$REMOTE_ROOT/logs/E195/cem/full/" "logs/E195/cem/full/"

"$PYTHON_BIN" - "$STAGE" <<'PY'
import csv, sys
from collections import Counter
from pathlib import Path
stage = Path(sys.argv[1])
canonical = Path("workspace/core4d/results/E195/s6_downstream/manifests/cem_full_manifest.tsv")
rows = list(csv.DictReader(canonical.open(), delimiter="\t")); fields = list(rows[0])
by_variant = {row["variant"]: row for row in rows}
shards = list(stage.glob("ada-gpu*.tsv"))
local = Path("workspace/core4d/results/E195/remote_staging/local-gpu0/full/manifest.tsv")
if local.is_file():
    shards.append(local)
for shard in shards:
    for row in csv.DictReader(shard.open(), delimiter="\t"):
        if row["variant"] in by_variant:
            by_variant[row["variant"]].update(row)
with canonical.open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
counts = Counter(row["status"] for row in rows)
print(f"merged E195 state: {dict(counts)}")
PY
echo "E195 Ada6000 pull complete"

