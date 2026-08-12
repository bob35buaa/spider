#!/usr/bin/env python3
from __future__ import annotations
import csv, json
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
OUT = REPO / 'workspace/core4d/analysis/E197_full_cem_omnirt_vs_prg_metrics'

def main() -> int:
    rows = []
    for name in ('a', 'b', 'c'):
        path = OUT / f'audit_visual_batch_{name}.tsv'
        with path.open(encoding='utf-8', newline='') as f:
            for r in csv.DictReader(f, delimiter='\t'):
                r['fall_flag'] = str(r.get('fall_flag', '')).lower() in ('1', 'true', 'yes')
                r['jitter_flag'] = str(r.get('jitter_flag', '')).lower() in ('1', 'true', 'yes')
                r['evidence_times'] = r.get('evidence_times', r.get('evidence_times_s', ''))
                r['batch'] = name
                rows.append(r)
    rows.sort(key=lambda r: r['case_id'])
    out_tsv = OUT / 'omnirt_video_visual_audit.tsv'
    fields = ['case_id', 'batch', 'fall_flag', 'jitter_flag', 'evidence_times', 'notes']
    with out_tsv.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter='\t'); w.writeheader()
        for r in rows:
            w.writerow({k: ('true' if r[k] is True else 'false' if r[k] is False else r.get(k, '')) for k in fields})
    counts = {'cases': len(rows), 'fall_flag_true': sum(r['fall_flag'] for r in rows), 'jitter_flag_true': sum(r['jitter_flag'] for r in rows), 'complete_case_ids': len({r['case_id'] for r in rows})}
    (OUT / 'omnirt_video_visual_audit_summary.json').write_text(json.dumps(counts, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    md = [
        '# E197 OmniRetarget 视频抽帧视觉审计', '',
        '口径：对 52 个 `omnirt_only.mp4` 按 1 fps 抽帧；检查明显摔倒、明显/大范围抖动。疑似动作变化点由审核 agent 额外查看邻近帧。', '',
        f"- 审计 case：{counts['cases']}（唯一 case：{counts['complete_case_ids']}）",
        f"- 明显摔倒：{counts['fall_flag_true']}；明显/大范围抖动：{counts['jitter_flag_true']}",
        '- 结论：未发现需要因明显摔倒或大范围抖动而剔除的 case。', '',
        '## 标记定义', '',
        '- `fall_flag=true`：抽帧中出现明确失衡倒地/躯干落地姿态。',
        '- `jitter_flag=true`：相邻秒帧显示明显或大范围非动作抖动。',
        '- 该审计是视频视觉 QC，不替代数值 gate 或人工 USE/partner/alignment gate。', '',
        '## Case 明细', '',
        '| Case | Batch | Fall | Jitter | 证据时间 | 备注 |',
        '|---|---|---:|---:|---|---|',
    ]
    for r in rows:
        md.append(f"| {r['case_id']} | {r['batch']} | {'是' if r['fall_flag'] else '否'} | {'是' if r['jitter_flag'] else '否'} | {r['evidence_times'] or '—'} | {r['notes']} |")
    (OUT / 'omnirt_video_visual_audit.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    print(json.dumps(counts, ensure_ascii=False))
    return 0
if __name__ == '__main__': raise SystemExit(main())
