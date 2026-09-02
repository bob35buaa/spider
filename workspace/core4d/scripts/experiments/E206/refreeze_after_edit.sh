#!/usr/bin/env bash
# E206: re-freeze the collision proxies after a human box edit in review_proxy_3d.py.
#
# Run this AFTER you have deleted boxes in the 3D reviewer and are happy with the
# result.  It re-measures every fidelity/cavity metric on the EDITED box set and
# re-installs the templates, so nothing downstream ever sees the pre-edit proxy.
#
#   bash workspace/core4d/scripts/experiments/E206/refreeze_after_edit.sh
#
# Idempotent: with an empty box_edits.json it reproduces the unedited freeze.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PY=.venv/bin/python
ED=workspace/core4d/scripts/experiments/E206
E=workspace/core4d/results/E206
EDITS=$E/s2_proxy/box_edits.json

echo "== E206 re-freeze after manual box edit =="
if [ -f "$EDITS" ]; then
  echo "-- edits on record --"
  $PY -c "
import json,sys
d=json.load(open('$EDITS'))
if not d: print('  (none)')
for k,v in sorted(d.items()):
    print(f\"  {k:10s} removed={v.get('removed')}  by={v.get('editor','?')}  tc={v.get('target_cells')}  orig_boxes={v.get('n_boxes_original')}\")
"
else
  echo "  (no box_edits.json — nothing edited, this is a plain re-freeze)"
fi

# Freeze the edit file for the duration of the run: the audit and the install
# MUST see the same edits, or the contract ends up describing a proxy that is
# not the one installed (observed 2026-09-03, when edits landed mid-audit).
if [ -f "$EDITS" ]; then cp "$EDITS" "$EDITS.inflight"; fi

echo
echo "== 1/3 re-audit contract on the EDITED box set =="
$PY -u $ED/audit_lowgeom_contract.py --frozen-target-cells || {
  echo "!! contract has failing hard gates after the edit — inspect"
  echo "   $E/s2_proxy/lowgeom_contract.md before continuing" >&2
}

echo
echo "== 2/3 re-install templates from the edited proxies =="
$PY -u $ED/install_lowgeom_templates.py --apply

echo
echo "== 3/3 verify every template loads, is box-only, keeps 43/41/29 =="
$PY - <<'PYEOF'
import sys
sys.path.insert(0, "workspace/core4d/scripts/experiments/E206")
import e206_common as C, lowgeom_proxy_v2 as L

rows = [r for r in C.read_tsv(C.S2_TEMPLATE_DIR / "lowgeom_install.tsv") if r["status"] == "applied"]
bad = []
for r in rows:
    chk = L.union_geoms_are_boxes(C.REPO / r["scene"])
    ok = chk["all_box"] and (chk["nq"], chk["nv"], chk["nu"]) == (43, 41, 29)
    tag = "✎" if r.get("edited") == "true" else " "
    print(f"  {tag} {r['scene'].split('/')[-2]:22s} geoms={chk['object_collision_geom_count']:3d} "
          f"{'OK' if ok else 'FAIL'}")
    if not ok:
        bad.append(r["scene"])
print(f"-> {len(rows)-len(bad)}/{len(rows)} templates OK")
sys.exit(1 if bad else 0)
PYEOF

echo
echo "re-freeze done. Contract: $E/s2_proxy/lowgeom_contract.md"
echo "Review TSV:              $E/s2_templates/review/nonbox_template_review.tsv"
