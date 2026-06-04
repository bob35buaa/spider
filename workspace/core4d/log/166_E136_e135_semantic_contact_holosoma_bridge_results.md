# E136 E135 Semantic Contact to Holosoma Bridge Audit Results

Date: 2026-06-03

## Scope

E136 followed
`workspace/core4d/plan/145_E136_e135_semantic_contact_holosoma_bridge_plan.md`.
It audits whether E135 v3 raw-contact masks can safely map onto existing
Holosoma-side exports without resizing or interpolation.

E136 did not write semantic Holosoma `object_contact` exports. It did not update
the global v3 registry and did not launch CEM, PPO, Holosoma training,
checkpoints, or remote jobs.

## Outputs

```text
workspace/core4d/results/E136/e135_semantic_contact_holosoma_bridge/
  e136_e135_semantic_contact_bridge_manifest.tsv
  e136_e135_semantic_contact_bridge_summary.json
  e136_e135_semantic_contact_bridge_summary.md
```

## Result

| metric | value |
|---|---:|
| audit rows | 24 |
| E107 semantic bridge candidate rows | 16 |
| direct raw-axis candidate rows | 8 |
| trim-window slice candidate rows | 8 |
| E126/E131 semantic-ready rows | 0 |
| fragment mapping blocked rows | 8 |
| structural proxy rows | 4 |
| semantic object_contact exports written | 0 |
| training launched | false |
| CEM launched | false |
| remote jobs launched | false |
| status | pass |

Audit table:

| case | threshold | export | raw/export/mapped frames | mapping | candidate | failure |
|---|---|---|---|---|---|---|
| `box021_035_p1` | `3cm` | `E107_retargeted_untrimmed` | 182/182/182 | `direct_raw_axis:pass` | true |  |
| `box021_035_p1` | `3cm` | `E107_trimmed` | 182/127/127 | `trim_window_slice:pass` | true |  |
| `box021_035_p1` | `3cm` | `E126_fragment_export` | 182/214/- | `blocked_missing_raw_window:blocked` | false | `fragment_raw_window_missing` |
| `box021_035_p1` | `3cm` | `E131_geometry_proxy_export` | 182/214/- | `blocked_missing_raw_window:blocked` | false | `fragment_raw_window_missing` |
| `box021_035_p2` | `3cm` | `E107_retargeted_untrimmed` | 182/182/182 | `direct_raw_axis:pass` | true |  |
| `box021_035_p2` | `3cm` | `E107_trimmed` | 182/133/133 | `trim_window_slice:pass` | true |  |
| `box021_035_p2` | `3cm` | `E126_fragment_export` | 182/214/- | `blocked_missing_raw_window:blocked` | false | `fragment_raw_window_missing` |
| `box021_035_p2` | `3cm` | `E131_geometry_proxy_export` | 182/214/- | `blocked_missing_raw_window:blocked` | false | `fragment_raw_window_missing` |
| `box021_029_p1` | `3cm` | `E107_retargeted_untrimmed` | 134/134/134 | `direct_raw_axis:pass` | true |  |
| `box021_029_p1` | `3cm` | `E107_trimmed` | 134/71/71 | `trim_window_slice:pass` | true |  |
| `box021_029_p2` | `3cm` | `E107_retargeted_untrimmed` | 134/134/134 | `direct_raw_axis:pass` | true |  |
| `box021_029_p2` | `3cm` | `E107_trimmed` | 134/75/75 | `trim_window_slice:pass` | true |  |

The same mapping classifications pass for the 5cm threshold. Full per-threshold
rows are in the manifest.

Trimmed E107 candidate active fractions:

| case | threshold | trim window | mapped frames | L/R/both active |
|---|---|---:|---:|---|
| `box021_035_p1` | `3cm` | 55:182 | 127 | 0.763780/0.787402/0.763780 |
| `box021_035_p1` | `5cm` | 55:182 | 127 | 0.779528/0.795276/0.779528 |
| `box021_035_p2` | `3cm` | 49:182 | 133 | 0.751880/0.759398/0.751880 |
| `box021_035_p2` | `5cm` | 49:182 | 133 | 0.766917/0.766917/0.766917 |
| `box021_029_p1` | `3cm` | 63:134 | 71 | 0.676056/0.619718/0.619718 |
| `box021_029_p1` | `5cm` | 63:134 | 71 | 0.676056/0.633803/0.633803 |
| `box021_029_p2` | `3cm` | 59:134 | 75 | 0.680000/0.733333/0.680000 |
| `box021_029_p2` | `5cm` | 59:134 | 75 | 0.680000/0.733333/0.680000 |

## Interpretation

E136 proves two safe E107 bridge routes:

1. E107 retargeted-untrimmed exports have the same frame count as the E135 raw
   masks (`182` or `134`), so a direct raw-axis semantic `object_contact` export
   is a valid candidate.
2. E107 trimmed exports have explicit `trim_window.json` metadata proving the
   raw-to-trimmed slice: `untrimmed_frames == raw_frames`,
   `trimmed_frames == export_frames`, `trim_end - trim_start == export_frames`,
   and valid trim bounds.

E126/E131 fragment exports remain blocked. Their paired exports are `214`
frames and the E126 manifest labels the policy
`min_frames_head_crop_fragment_pair_no_raw_window`. E131 rows still have
`object_contact`, but its source is
`actor_rubber_hand_box021_surface_proxy_5cm`, so it remains structural proxy
evidence rather than semantic raw-contact evidence.

The next bounded step is to write semantic `object_contact` export candidates
only for the E107 direct/trim-window rows and then run the existing Holosoma
loader/runtime probes on those new exports. E126/E131 should not receive
semantic masks until a raw-window mapping is created and audited.

## Claims Verification

| Claim | Result |
|---|---|
| C1: E135 raw-contact masks expose valid 3cm/5cm arrays | pass |
| C2: E107 untrimmed direct raw-axis candidates are detected only on exact frame match | pass |
| C3: E107 trimmed candidates require explicit compatible `trim_window.json` | pass |
| C4: E126/E131 fragment exports remain blocked without raw-window mapping | pass |
| C5: no semantic exports/CEM/PPO/training/remote launched | pass |

## Verification

- `.venv/bin/python -m py_compile workspace/core4d/scripts/E136/audit_e135_semantic_contact_holosoma_bridge.py`
- `bash -n workspace/core4d/scripts/eval/eval_E136_e135_semantic_contact_holosoma_bridge.sh`
- `bash workspace/core4d/scripts/eval/eval_E136_e135_semantic_contact_holosoma_bridge.sh`
