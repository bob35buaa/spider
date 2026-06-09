# E149 E143 clean rubber benchmark results

Date: 2026-06-09 CST

Plan: `workspace/core4d/plan/157_E149_e143_clean_rubber_benchmark_eval_plan.md`

## Scope

E149 is an eval-only clean benchmark view of E148. It uses the E143 manual annotation and failure-analysis report to filter the E148 24-case rubber/sphere/Omni comparison into cleaner benchmark sets.

Inputs:

- E143 annotation workbook: `workspace/core4d/results/E143/raw_mask_ref_fk_24case_omniretarget_comparison/E143_raw_mask_ref_fk_24case_omniretarget_comparison-anno.xlsx`
- E143 failure report: `workspace/core4d/results/E143/spider_contact_failure_analysis/report.md`
- E143 case analysis: `workspace/core4d/results/E143/spider_contact_failure_analysis/data/case_analysis.tsv`
- E148 full comparison: `workspace/core4d/results/E148/e143_24case_rubber_hand_collision/eval/full/e148_case_comparison.tsv`

No CEM, RL, Holosoma job, or remote job was launched.

## Benchmark Sets

`clean6_primary`:

- `box021_035_p1`
- `box021_035_p2`
- `box021_029_p2`
- `box004_083_p1`
- `box004_083_p2`
- `box023_person2`

`relaxed8_valid_like`:

- `box021_035_p2`
- `box023_person2`
- `box021_029_p2`
- `box004_082_p1`
- `box004_083_p2`
- `box021_035_p1`
- `box026_139_p1`
- `box004_083_p1`

The relaxed8 set matches the valid-like list from E143 analysis. The two added cases are useful but mixed: `box004_082_p1` has object jitter, and `box026_139_p1` has mixed object/ref notes.

## Outputs

| Artifact | Path | Count |
|---|---|---:|
| method summary | `workspace/core4d/results/E149/e143_clean_rubber_benchmark/e149_method_summary.tsv` | 6 rows |
| diff summary | `workspace/core4d/results/E149/e143_clean_rubber_benchmark/e149_diff_summary.tsv` | 10 rows |
| case comparison | `workspace/core4d/results/E149/e143_clean_rubber_benchmark/e149_case_comparison.tsv` | 14 rows |
| annotation source | `workspace/core4d/results/E149/e143_clean_rubber_benchmark/e149_case_source_annotations.tsv` | 8 rows |
| markdown summary | `workspace/core4d/results/E149/e143_clean_rubber_benchmark/e149_summary.md` | 1 |
| xlsx | `workspace/core4d/results/E149/e143_clean_rubber_benchmark/E149_e143_clean_rubber_benchmark.xlsx` | 5 sheets |

Workbook sheets:

- `benchmark平均`: 6 data rows
- `benchmark_diff`: 10 data rows
- `clean6逐case`: 6 data rows
- `relaxed8逐case`: 8 data rows
- `case_source_annotations`: 8 data rows

LibreOffice recalc reported `total_errors=0`, `total_formulas=0`.

## Quantitative Result

### clean6_primary

| method | cases | fall | 手物接触 | 5cm | 10cm | 手物穿透 | 深穿透2cm | 腿穿透 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `OmniRetarget` | 6 | 0 | 0.6408 | 0.7076 | 0.7298 | 0.6408 | 0.3210 | 0.0065 |
| `sphere Spider` | 6 | 0 | 0.4813 | 0.6656 | 0.7023 | 0.4813 | 0.0028 | 0.0470 |
| `rubber hand Spider` | 6 | 0 | 0.2833 | 0.6735 | 0.7077 | 0.2520 | 0.0089 | 0.0977 |

Diffs:

| metric | rubber-sphere | OmniRetarget-rubber |
|---|---:|---:|
| 手物接触 | -0.1980 | +0.3575 |
| 5cm | +0.0079 | +0.0341 |
| 10cm | +0.0054 | +0.0221 |
| 手物穿透 | -0.2293 | +0.3888 |
| 腿穿透 | +0.0507 | -0.0913 |

### relaxed8_valid_like

| method | cases | fall | 手物接触 | 5cm | 10cm | 手物穿透 | 深穿透2cm | 腿穿透 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `OmniRetarget` | 8 | 0 | 0.5957 | 0.6924 | 0.7140 | 0.5957 | 0.2860 | 0.0207 |
| `sphere Spider` | 8 | 0 | 0.4635 | 0.6449 | 0.6847 | 0.4635 | 0.0155 | 0.0564 |
| `rubber hand Spider` | 8 | 0 | 0.3070 | 0.6592 | 0.6940 | 0.2777 | 0.0084 | 0.0953 |

Diffs:

| metric | rubber-sphere | OmniRetarget-rubber |
|---|---:|---:|
| 手物接触 | -0.1565 | +0.2886 |
| 5cm | +0.0144 | +0.0331 |
| 10cm | +0.0093 | +0.0200 |
| 手物穿透 | -0.1858 | +0.3179 |
| 腿穿透 | +0.0389 | -0.0746 |

## Interpretation

The clean benchmark confirms the E148 conclusion with less noise:

- Rubber hand Spider gives small positive 5cm/10cm near-contact gains over sphere Spider.
- Rubber hand Spider substantially reduces hand-object penetration versus sphere Spider.
- Rubber hand Spider lowers physics-contact fraction on these clean cases.
- Rubber hand Spider increases leg penetration on these clean cases.

So the cleaner benchmark does not make rubber a clean replacement for sphere. It makes the tradeoff sharper: rubber geometry helps near-distance and penetration metrics, but hurts physical contact and lower-body cleanliness.

## Claims

| Claim | Result | Evidence |
|---|---|---|
| Benchmark cleanliness | PASS | clean6/relaxed8 are taken from E143 annotation/report and exclude object-rotation, non-target, walk-up/ref-bad, and known drift cases from the main average. |
| No rerun | PASS | Script only reads E148 comparison TSV and E143 annotations; no remote/CEM/RL command launched. |
| Rubber tradeoff | PASS | clean6 rubber-sphere: 5cm +0.0079, 10cm +0.0054, hand penetration -0.2293, leg penetration +0.0507, physics contact -0.1980. |

## Validation

- `bash workspace/core4d/scripts/eval/eval_E149_e143_clean_rubber_benchmark.sh`
- `.venv/bin/python -m py_compile workspace/core4d/scripts/eval/eval_E149_e143_clean_rubber_benchmark.py`
- `bash -n workspace/core4d/scripts/eval/eval_E149_e143_clean_rubber_benchmark.sh`
- XLSX recalc: `total_errors=0`, `total_formulas=0`
- Workbook readback: 5 sheets, expected row counts 6/10/6/8/8.
