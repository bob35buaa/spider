# E096 P3 Preprocess Retry Summary

日期：2026-05-29

## Case

| item | value |
|---|---|
| original task | `e091_box004_20231003_2_082_p2` |
| retry target task | `e096_box004_20231003_2_082_p2_fingertip` |
| person | `person2` |
| object | `box004` |
| source scene | `box004_person2` |
| setting | `REPLACE_WRIST_WITH_FINGERTIP=1` |

## Result

Retry failed in OmniRetarget before trimmed/SPIDER task generation.

```text
RuntimeError: CVXPY solve failed: infeasible
```

The failure occurs around frame `81/139`, matching the original no-fingertip E095 failure. Therefore P3 remains `preprocess_blocked` and must not be treated as a full-CEM case.

## Paths

| item | path |
|---|---|
| case file | `workspace/core4d/results/E096/preprocess_retry/cases_e096_p3_fingertip_retry.tsv` |
| result root | `workspace/core4d/results/E096/preprocess_retry/results/holosoma_e096_box004_20231003_2_082_p2_fingertip/` |
| run log | `/home/ubuntu/Workspace/holosoma/workspace/v3/data_construction_v2/logs/stage2b_medium_20260529_163812.log` |
