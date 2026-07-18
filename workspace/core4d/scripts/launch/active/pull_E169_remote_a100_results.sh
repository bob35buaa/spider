#!/usr/bin/env bash
# Pull, merge, and validate E169 A100 results.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODE="${1:-canary}"
if [ "$MODE" != "canary" ] && [ "$MODE" != "full" ]; then echo "usage: $0 {canary|full}" >&2; exit 2; fi
REMOTE="${REMOTE:-batchcom@61.172.170.106}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/dataset-assist-0/xiayb/workspace/spider}"
REMOTE_SSH_PORT="${REMOTE_SSH_PORT:-30409}"
REMOTE_SSH_KEY="${REMOTE_SSH_KEY:-$HOME/.ssh/id_rsa_tianyiyun}"
EXPECTED=4; [ "$MODE" = "full" ] && EXPECTED=28
MANIFEST="workspace/core4d/results/E169/manifests/cem_${MODE}_manifest.tsv"
LATEST_SESSION="workspace/core4d/results/E169/s0_environment/latest_${MODE}_session.txt"
SESSION="${SESSION:-$(cat "$LATEST_SESSION" 2>/dev/null || true)}"
[ -n "$SESSION" ] || { echo "missing E169 ${MODE} session id" >&2; exit 1; }
SHARD_ROOT="workspace/core4d/results/E169/manifests/a100_${MODE}_${SESSION}"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=4 -p "$REMOTE_SSH_PORT" -i "$REMOTE_SSH_KEY")
RSYNC_RSH="ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=4 -p $REMOTE_SSH_PORT -i $REMOTE_SSH_KEY"
ssh_remote() { ssh "${SSH_OPTS[@]}" "$REMOTE" "$@"; }
retry() { local n=1; while ! "$@"; do [ "$n" -ge 4 ] && return 1; sleep "$((n * 3))"; n=$((n + 1)); done; }

mkdir -p "workspace/core4d/results/E169/cem/$MODE" "logs/E169/cem/$MODE" "$SHARD_ROOT"
retry rsync -az -e "$RSYNC_RSH" "${REMOTE}:${REMOTE_ROOT}/workspace/core4d/results/E169/cem/${MODE}/" "workspace/core4d/results/E169/cem/${MODE}/"
retry rsync -az -e "$RSYNC_RSH" "${REMOTE}:${REMOTE_ROOT}/logs/E169/cem/${MODE}/" "logs/E169/cem/${MODE}/"
retry rsync -az -e "$RSYNC_RSH" "${REMOTE}:${REMOTE_ROOT}/${SHARD_ROOT}/" "$SHARD_ROOT/"

".venv/bin/python" - "$MODE" "$EXPECTED" "$MANIFEST" "$SHARD_ROOT" <<'PY'
import csv, hashlib, json, math, sys
from datetime import datetime
from pathlib import Path
import numpy as np
import yaml

mode, expected_s, manifest_s, shard_root_s = sys.argv[1:]
expected = int(expected_s); manifest = Path(manifest_s); shard_root = Path(shard_root_s)
rows = list(csv.DictReader(manifest.open(newline="", encoding="utf-8"), delimiter="\t"))
fields = list(rows[0]) if rows else []
by_variant = {row["variant"]: row for row in rows}
for shard in sorted(shard_root.glob("gpu*.tsv")):
    for row in csv.DictReader(shard.open(newline="", encoding="utf-8"), delimiter="\t"):
        target = by_variant.get(row["variant"])
        if target is not None:
            for field in fields:
                if field in row: target[field] = row[field]
with manifest.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)

def sha(path):
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""): digest.update(chunk)
    return digest.hexdigest()

artifacts=[]; incomplete=[]
for row in rows:
    row_missing=[]
    for key in ("result_npz","outdir_npz","config_act"):
        path=Path(row[key])
        if not path.is_file(): row_missing.append(key); continue
        artifacts.append({"variant":row["variant"],"case_id":row["case_id"],"cell_id":row["cell_id"],"artifact":key,"path":str(path),"size":path.stat().st_size,"sha256":sha(path)})
    if not row_missing:
        try:
            with np.load(row["outdir_npz"],allow_pickle=True) as data:
                if "qpos" not in data or not np.isfinite(np.asarray(data["qpos"],dtype=float)).all(): row_missing.append("valid_qpos")
                if row["g_enabled"]=="true" and "cem_leg_gate_valid_frac" not in data: row_missing.append("leg_gate_diagnostics")
            cfg=yaml.safe_load(Path(row["config_act"]).read_text())
            if cfg.get("cem_leg_gate_enabled") != (row["g_enabled"]=="true"): row_missing.append("config_leg_gate")
        except Exception as exc: row_missing.append(f"validation:{type(exc).__name__}")
    if row_missing: incomplete.append({"variant":row["variant"],"status":row["status"],"missing":row_missing})

out=Path(f"workspace/core4d/results/E169/artifacts/{mode}"); out.mkdir(parents=True,exist_ok=True)
artifact_fields=["variant","case_id","cell_id","artifact","path","size","sha256"]
with (out/"artifact_manifest.tsv").open("w",newline="",encoding="utf-8") as stream:
    writer=csv.DictWriter(stream,fieldnames=artifact_fields,delimiter="\t",lineterminator="\n"); writer.writeheader(); writer.writerows(artifacts)
summary={"created_at":datetime.now().astimezone().isoformat(timespec="seconds"),"mode":mode,"expected_rows":expected,"manifest_rows":len(rows),"complete_rows":len(rows)-len(incomplete),"artifact_files":len(artifacts),"incomplete":incomplete,"status":"pass" if len(rows)==expected and not incomplete else "incomplete"}
(out/"artifact_summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
print(json.dumps(summary,indent=2,sort_keys=True))
if summary["status"] != "pass" and not bool(int(__import__('os').environ.get("ALLOW_INCOMPLETE","0"))): raise SystemExit(2)
PY
