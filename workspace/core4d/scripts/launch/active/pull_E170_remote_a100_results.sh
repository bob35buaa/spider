#!/usr/bin/env bash
# Pull and validate only E170 execution-manifest artifacts.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
MODE="${1:-canary}"
if [ "$MODE" != "canary" ] && [ "$MODE" != "full" ]; then echo "usage: $0 {canary|full}" >&2; exit 2; fi
REMOTE="${REMOTE:-batchcom@61.172.170.106}"; REMOTE_ROOT="${REMOTE_ROOT:-/home/dataset-assist-0/xiayb/workspace/spider}"
REMOTE_SSH_PORT="${REMOTE_SSH_PORT:-30409}"; REMOTE_SSH_KEY="${REMOTE_SSH_KEY:-$HOME/.ssh/id_rsa_tianyiyun}"
RESULT_ROOT="workspace/core4d/results/E170"; MANIFEST="$RESULT_ROOT/s6_downstream/manifests/cem_${MODE}_manifest.tsv"
LATEST_SESSION="$RESULT_ROOT/s0_environment/latest_${MODE}_session.txt"; SESSION="${SESSION:-$(cat "$LATEST_SESSION" 2>/dev/null || true)}"
[ -n "$SESSION" ] || { echo "missing E170 $MODE session id" >&2; exit 1; }
SHARD_ROOT="$RESULT_ROOT/s6_downstream/manifests/a100_${MODE}_${SESSION}"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=4 -p "$REMOTE_SSH_PORT" -i "$REMOTE_SSH_KEY")
RSYNC_RSH="ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=4 -p $REMOTE_SSH_PORT -i $REMOTE_SSH_KEY"
retry() { local n=1; while ! "$@"; do [ "$n" -ge 4 ] && return 1; sleep "$((n * 3))"; n=$((n + 1)); done; }
mkdir -p "$RESULT_ROOT/s6_downstream/cem/$MODE" "logs/E170/cem/$MODE" "$SHARD_ROOT"
retry rsync -az -e "$RSYNC_RSH" "${REMOTE}:${REMOTE_ROOT}/$RESULT_ROOT/s6_downstream/cem/${MODE}/" "$RESULT_ROOT/s6_downstream/cem/${MODE}/"
retry rsync -az -e "$RSYNC_RSH" "${REMOTE}:${REMOTE_ROOT}/logs/E170/cem/${MODE}/" "logs/E170/cem/${MODE}/"
retry rsync -az -e "$RSYNC_RSH" "${REMOTE}:${REMOTE_ROOT}/${SHARD_ROOT}/" "$SHARD_ROOT/"

".venv/bin/python" - "$MODE" "$MANIFEST" "$SHARD_ROOT" <<'PY'
import csv,hashlib,json,os,sys
from datetime import datetime
from pathlib import Path
import numpy as np, yaml
mode,manifest_s,shard_s=sys.argv[1:]; manifest=Path(manifest_s); shard_root=Path(shard_s)
rows=list(csv.DictReader(manifest.open(newline="",encoding="utf-8"),delimiter="\t")); fields=list(rows[0]) if rows else []
by_variant={row["variant"]:row for row in rows}
for shard in sorted(shard_root.glob("gpu*.tsv")):
    for row in csv.DictReader(shard.open(newline="",encoding="utf-8"),delimiter="\t"):
        if row["variant"] in by_variant: by_variant[row["variant"]].update({key:value for key,value in row.items() if key in fields})
with manifest.open("w",newline="",encoding="utf-8") as stream:
    writer=csv.DictWriter(stream,fieldnames=fields,delimiter="\t",lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
def sha(path):
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""): digest.update(chunk)
    return digest.hexdigest()
artifacts=[]; incomplete=[]
for row in rows:
    missing=[]; paths={key:Path(row[key]) for key in ("result_npz","outdir_npz","config_act")}
    for key,path in paths.items():
        if not path.is_file(): missing.append(key)
        else: artifacts.append({"variant":row["variant"],"case_id":row["case_id"],"artifact":key,"path":str(path),"size":path.stat().st_size,"sha256":sha(path)})
    if not missing:
        try:
            with np.load(paths["result_npz"],allow_pickle=True) as root,np.load(paths["outdir_npz"],allow_pickle=True) as out:
                if "qpos" not in root or "qpos" not in out or not np.array_equal(root["qpos"],out["qpos"]): missing.append("root_outdir_qpos_match")
                if "qpos" not in out or not np.isfinite(np.asarray(out["qpos"],dtype=float)).all(): missing.append("finite_qpos")
                required={"cem_leg_gate_valid_frac","cem_leg_gate_selected_valid_frac","cem_leg_gate_fallback_used","sample_leg_gate_min_sdf_min","sample_leg_gate_violation_pct_mean","leg_object_penalty_mean"}
                missing.extend(f"diag:{key}" for key in sorted(required-set(out.files)))
            cfg=yaml.safe_load(paths["config_act"].read_text())
            if cfg.get("scene_name")!="scene_act_E170_lowerbody_physics" or cfg.get("leg_object_penalty_scale")!=2.0 or cfg.get("cem_leg_gate_enabled") is not True: missing.append("effective_config")
            scene=Path(row["scene_act"])
            if not scene.is_file() or sha(scene)!=row["effective_scene_sha256"]: missing.append("effective_scene_sha")
        except Exception as exc: missing.append(f"validation:{type(exc).__name__}:{exc}")
    if missing: incomplete.append({"variant":row["variant"],"status":row["status"],"missing":missing})
out=Path(f"workspace/core4d/results/E170/s6_downstream/artifacts/{mode}"); out.mkdir(parents=True,exist_ok=True)
with (out/"artifact_manifest.tsv").open("w",newline="",encoding="utf-8") as stream:
    fields_a=["variant","case_id","artifact","path","size","sha256"]; writer=csv.DictWriter(stream,fieldnames=fields_a,delimiter="\t",lineterminator="\n"); writer.writeheader(); writer.writerows(artifacts)
complete=len(rows)-len(incomplete); required=2 if mode=="canary" else 24
status="pass" if complete==required and len(rows)==required and not incomplete else ("partial_allow_missing" if complete==len(rows) and not incomplete else "incomplete")
summary={"created_at":datetime.now().astimezone().isoformat(timespec="seconds"),"mode":mode,"required_rows":required,"manifest_rows":len(rows),"complete_rows":complete,"artifact_files":len(artifacts),"incomplete":incomplete,"status":status}
(out/"artifact_summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n"); print(json.dumps(summary,indent=2,sort_keys=True))
if status=="incomplete" and os.environ.get("ALLOW_INCOMPLETE","0")!="1": raise SystemExit(2)
PY
