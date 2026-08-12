#!/usr/bin/env python3
"""Prepare a minimal, identity-complete source TSV for five E197 partner reruns."""
from pathlib import Path
import csv

REPO = Path(__file__).resolve().parents[6]
GATE = REPO / 'workspace/core4d/analysis/E197_full_cem_omnirt_vs_prg_metrics/e197_omni_absolute_wide_gate_filter.tsv'
TARGETS = {
    'box021_20231018_029_p1',
    'box021_20231018_034_p2',
    'box021_20231018_035_p2',
    'box023_20231020_039_p2',
    'box024_20231011_030_p1',
}
OUT = REPO / 'workspace/core4d/results/E197/s6_downstream/rl_export/partner_omnirt_rerun_source.tsv'

def split_case(case_id):
    stem, short = case_id.rsplit('_', 1)
    object_key, date, seq = stem.split('_', 2)
    person = {'p1': 'person1', 'p2': 'person2'}[short]
    return object_key, date, seq, person

rows=[]
for r in csv.DictReader(GATE.open(), delimiter='\t'):
    if r['case_id'] not in TARGETS:
        continue
    object_key, date, seq, person = split_case(r['case_id'])
    object_name = object_key.replace('box', 'Box')
    rows.append({
        'case_id': r['case_id'], 'object_key': object_key, 'object_name': object_name,
        'date': date, 'seq': seq, 'person': person,
        'person_idx': '0' if person == 'person1' else '1',
        'rl_export_decision': 'RL_EXPORT_READY',
        'handoff_decision': 'HANDOFF_READY', 'cem_status': 'pass',
    })
if len(rows) != len(TARGETS):
    raise SystemExit(f'expected {len(TARGETS)} source rows, got {len(rows)}')
fields=list(rows[0])
OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open('w', encoding='utf-8', newline='') as f:
    w=csv.DictWriter(f, fieldnames=fields, delimiter='\t'); w.writeheader(); w.writerows(sorted(rows, key=lambda x:x['case_id']))
print(OUT)
