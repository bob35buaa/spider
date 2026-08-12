# E196 reference metadata integrity fix report

_Corrected Euler reference contract for the 29 E194 mismatch cases · 2026-08-12_

---

## 📋 Abstract

Decision: `REFERENCE_FIX_VALIDATED_G1_IMPROVES`. E196 reran exactly 29 affected cases (box001=21, box023=8) with fail-closed metadata resolution. Reference integrity passed `29/29`; performance is reported separately from the technical fix.

## 🔬 Methodology

PRG and contaminated G1 use the frozen E194 public-core authority. Corrected G1 is rescored with the same public core and 12-gate thresholds. All continuous comparisons are paired by case; positive `improvement` always means better.

```mermaid
flowchart LR
    accTitle: E196 Evidence Closure
    accDescr: Corrected metadata passes runtime parity before public-core scoring, paired comparison, and visual review produce the final E196 decision.

    metadata["🔧 Resolve metadata"] --> parity{"🔍 Parity passed?"}
    parity -->|No| stop_run(["❌ Stop experiment"])
    parity -->|Yes| score_cases["📊 Score 29 cases"]
    score_cases --> pair_arms["🔗 Pair three arms"]
    pair_arms --> review_video["🔍 Review videos"]
    review_video --> decision_node(["✅ Record decision"])

    classDef action fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#1e3a5f
    classDef decision fill:#fef9c3,stroke:#ca8a04,stroke-width:2px,color:#713f12
    classDef danger fill:#fee2e2,stroke:#dc2626,stroke-width:2px,color:#7f1d1d
    classDef success fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#14532d

    class metadata,score_cases,pair_arms,review_video action
    class parity decision
    class stop_run danger
    class decision_node success
```

## 📊 Findings

### Reference integrity

| Check | Result | Threshold |
| --- | ---: | ---: |
| Runtime/meta/XML parity | 29/29 | 29/29 |
| Runtime target max | 0.00000296° | <0.0001° |
| Axis target max | 0.00000296° | <0.0001° |
| Public orientation reproduction max | 0.0000000000° | <0.000001° |

### Corrected G1 versus PRG

| Subset | Metric | PRG | Corrected G1 | Raw delta | Improvement | 95% CI |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ALL_AFFECTED_29 | Object orientation error (deg) | 5.8320 | 5.1119 | -0.7201 | +0.7201 | [+0.2016, +1.2477] |
| ALL_AFFECTED_29 | Object 3D position error (cm) | 11.5113 | 9.9528 | -1.5584 | +1.5584 | [+0.8982, +2.2537] |
| ALL_AFFECTED_29 | Object z MAE (cm) | 4.9536 | 3.6012 | -1.3524 | +1.3524 | [+1.0003, +1.7060] |
| BOX001_PRIMARY_20 | Object orientation error (deg) | 6.0756 | 5.2587 | -0.8169 | +0.8169 | [+0.2263, +1.3990] |
| BOX001_PRIMARY_20 | Object 3D position error (cm) | 11.0244 | 9.4733 | -1.5510 | +1.5510 | [+0.7546, +2.3945] |
| BOX001_PRIMARY_20 | Object z MAE (cm) | 4.7911 | 3.1502 | -1.6410 | +1.6410 | [+1.2665, +2.0130] |
| BOX023_8 | Object orientation error (deg) | 5.2708 | 4.8846 | -0.3862 | +0.3862 | [-0.6059, +1.7622] |
| BOX023_8 | Object 3D position error (cm) | 13.1727 | 11.7522 | -1.4205 | +1.4205 | [+0.3068, +2.7995] |
| BOX023_8 | Object z MAE (cm) | 5.5134 | 4.9642 | -0.5492 | +0.5492 | [-0.0398, +1.1247] |

### Twelve-gate result

| Arm | Pass | Total |
| --- | ---: | ---: |
| PRG | 7 | 29 |
| G1_contaminated | 6 | 29 |
| G1_corrected | 10 | 29 |

### New orientation long tails

No corrected G1 case has orientation regression `>5°` versus PRG.

## 💡 Interpretation

The reference contract claim is determined only by runtime/meta/XML/world-pose evidence. Performance claims use corrected G1 versus PRG; contaminated E194 G1 is retained only as historical evidence and is not used for gravcomp causality.

## ⚠️ Limitations

- The rerun set contains only the 29 convention-mismatch cases; the 43 convention-match cases were intentionally not rerun
- Hand tracking is the public core's combined left/right EEF mean; no unsupported per-hand metric is imputed
- Bootstrap intervals quantify paired effect uncertainty but do not replace case-level and visual failure review

## 🔗 Artifacts

- `e196_reference_fix_by_case.tsv`
- `e196_reference_fix_by_object.tsv`
- `e196_reference_integrity_audit.tsv`
- `E196_reference_fix_comparison.xlsx`
- `../../render/full_reference_fix/three_arm_video_manifest.tsv`
