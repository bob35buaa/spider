# E026 patch: E081 full rerun paper metrics

日期：2026-05-21

## Context

E026 已补齐 `spider_E081_full_rerun` 的 13case rollout 和 legacy E081 comparison，但 summary 表里大量 paper-aligned 指标为空。原因是 `workspace/core4d/scripts/eval/eval_E081.py` 只输出 E081 legacy schema，没有调用 E019 的 `paper_metrics.add_paper_metrics`。

用户指出：如果不接入 paper metrics，这个对比没有意义。

## Claim

`spider_E081_full_rerun` 必须和 E018b / E022-E025 一样输出 E019 paper-aligned metrics，至少包含：

- object orientation / paper object tracking
- 5cm contact preservation
- robot-object deep penetration
- MuJoCo penetration
- smoothness / relative smoothness
- SPIDER FK tracking metrics

## 改动

1. 修改 `workspace/core4d/scripts/eval/eval_E081.py`：
   - 保留 legacy E081 指标。
   - 在生成 `timeseries_*.csv` 和 `legobj_timeseries_*.csv` 后追加 `paper_metrics.add_paper_metrics(...)`。
   - 将 `person_idx` 从 variants TSV 保留下来，传给 mask-gated contact metric。
2. 重跑：
   - `VARIANTS_FILE=workspace/core4d_collab_retarget/scripts/E026/e081_full_variants.tsv RESULTS=workspace/core4d_collab_retarget/results/E026_E081_full .venv/bin/python workspace/core4d/scripts/eval/eval_E081.py`
   - `.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E026_full_eval.py --all`
3. 更新 E026 log / tracker / progress 中的 P0/P1 数字与 caveat。

## 成功标准

- `workspace/core4d_collab_retarget/results/E026_E081_full/comparison.csv` 仍为 `13` rows。
- `spider_E081_full_rerun` rows 的 `schema` 在 E026 normalized output 中变成 `paper_metrics`。
- P0/P1 summary 中 `spider_E081_full_rerun` 不再缺 Obj Ori、Deep Pen、MJ Pen、Smoothness。
