# E149 — E143 clean benchmark rubber hand collision eval

## Context

E148 finished the full E143 24-case rubber hand collision comparison, but that 24-case aggregate mixes clean carry cases with known object/mocap/ref issues. E143 already has a manual annotation workbook and a failure-analysis report:

- `workspace/core4d/results/E143/raw_mask_ref_fk_24case_omniretarget_comparison/E143_raw_mask_ref_fk_24case_omniretarget_comparison-anno.xlsx`
- `workspace/core4d/results/E143/spider_contact_failure_analysis/report.md`
- `workspace/core4d/results/E143/spider_contact_failure_analysis/data/case_analysis.tsv`
- `workspace/core4d/results/E143/spider_contact_failure_analysis/data/analysis_summary.json`

The report recommends not using all E143 cases as the primary contact benchmark. It defines a cleaner main set where mocap/object/ref semantics are mostly valid and SPIDER's contact-persistence gap can be measured more directly.

This experiment is eval-only. It does not rerun CEM, does not touch Holosoma, and does not change reward/optimizer logic.

## Benchmark Definition

Primary clean benchmark (`clean6_primary`):

- `box021_035_p1`
- `box021_035_p2`
- `box021_029_p2`
- `box004_083_p1`
- `box004_083_p2`
- `box023_person2`

Relaxed valid-like benchmark (`relaxed8_valid_like`):

- all `clean6_primary`
- `box004_082_p1`
- `box026_139_p1`

Rationale:

- `clean6_primary` is the report's recommended main benchmark.
- `relaxed8_valid_like` matches `analysis_summary.json.valid_like_cases`, adding the two mixed but useful cases.
- Dirty/ref-bad/not-target cases remain out of the main average and can be reported separately later as stress tests.

## Claims

1. **Benchmark cleanliness claim**: clean6/relaxed8 exclude the E143 cases marked as object rotation, non-target sequence, walk-up/ref issue, or known visual failure.
2. **No rerun claim**: all metrics are computed from already recovered E148 outputs; no new CEM jobs are launched.
3. **Rubber tradeoff claim**: on clean cases, rubber hand Spider should be judged separately on near-contact, hand penetration, and leg penetration, not by the noisy 24-case mean alone.

## Implementation Plan

Add eval-only fixed entry:

- `workspace/core4d/scripts/eval/eval_E149_e143_clean_rubber_benchmark.py`
- `workspace/core4d/scripts/eval/eval_E149_e143_clean_rubber_benchmark.sh`

Inputs:

- E148 full case comparison: `workspace/core4d/results/E148/e143_24case_rubber_hand_collision/eval/full/e148_case_comparison.tsv`
- E143 annotation/failure analysis TSV and JSON.

Outputs:

- `workspace/core4d/results/E149/e143_clean_rubber_benchmark/e149_method_summary.tsv`
- `workspace/core4d/results/E149/e143_clean_rubber_benchmark/e149_diff_summary.tsv`
- `workspace/core4d/results/E149/e143_clean_rubber_benchmark/e149_case_comparison.tsv`
- `workspace/core4d/results/E149/e143_clean_rubber_benchmark/e149_summary.json`
- `workspace/core4d/results/E149/e143_clean_rubber_benchmark/e149_summary.md`
- `workspace/core4d/results/E149/e143_clean_rubber_benchmark/E149_e143_clean_rubber_benchmark.xlsx`

Workbook sheets:

- `benchmark平均`
- `benchmark_diff`
- `clean6逐case`
- `relaxed8逐case`
- `case_source_annotations`

## Success Criteria

- `clean6_primary` has 6/6 matched rows in E148 comparison.
- `relaxed8_valid_like` has 8/8 matched rows in E148 comparison.
- Each benchmark has OmniRetarget, sphere Spider, and rubber hand Spider averages.
- Diff table includes at least `手物接触`, `5cm`, `10cm`, `手物穿透`, `腿穿透`.
- XLSX opens and has zero formula errors after recalc.
- No remote jobs are launched.

## Commands

```bash
bash workspace/core4d/scripts/eval/eval_E149_e143_clean_rubber_benchmark.sh
python /home/ubuntu/.codex/skills/xlsx/scripts/recalc.py \
  workspace/core4d/results/E149/e143_clean_rubber_benchmark/E149_e143_clean_rubber_benchmark.xlsx 60
```

## Validation

```bash
.venv/bin/python -m py_compile workspace/core4d/scripts/eval/eval_E149_e143_clean_rubber_benchmark.py
bash -n workspace/core4d/scripts/eval/eval_E149_e143_clean_rubber_benchmark.sh
git diff --check -- \
  workspace/core4d/plan/157_E149_e143_clean_rubber_benchmark_eval_plan.md \
  workspace/core4d/scripts/eval/eval_E149_e143_clean_rubber_benchmark.py \
  workspace/core4d/scripts/eval/eval_E149_e143_clean_rubber_benchmark.sh \
  workspace/core4d/log/189_E149_e143_clean_rubber_benchmark_results.md \
  workspace/core4d/EXPERIMENT_TRACKER.md \
  workspace/core4d/progress.md
```
