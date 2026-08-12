# E196 reference metadata integrity fix results

_Core4D · Phase 59 · 2026-08-12 · 29 affected E194 mismatch cases · corrected G1 Full rerun_

## Summary

- Question: does fail-closed Euler reference resolution remove the E194 object-orientation corruption without changing the physical/CEM contract?
- Execution: exactly 29 affected cases, `box001=21` and `box023=8`; corrected G1 only; `seed=0`, `1024 samples × 32 opt steps`.
- Artifact closure: `29/29` Full results, `29/29` integrity audit, `0` evaluator errors, `29/29` corrected videos.
- Technical claim: validated. Runtime target/raw and axis target/raw orientation parity stayed below `2.96e-6°`; public metric reproduction error was below `1e-6°`.
- Performance result: corrected G1 improves over contaminated G1 on object orientation and object z/3D position metrics, and improves the strict 12-gate pass count from `6` to `10`; it is not uniformly better on every contact/hand gate.

## 1. Motivation and hypothesis

E194 diagnosis identified a runtime/compiled Euler convention mismatch in 29 G1 expansion rows. The hypothesis was that resolving the convention from compiled hinge axes, requiring sibling metadata, and hard-failing on mismatch would remove the wrong target while leaving raw inputs, scene physics, reward, seed and CEM budget unchanged.

## 2. Setup and reproducibility

| Item | Frozen value |
| --- | --- |
| Case authority | E194 orientation reference conversion audit |
| Case set | 29 (`box001=21`, `box023=8`), SHA `b7255fbb...2941dac` |
| Arm | corrected G1; PRG and contaminated G1 are frozen paired authorities |
| Object control | `gravcomp=1`, `kp_pos=500`, `kp_rot=50` |
| CEM | `seed=0`, `num_samples=1024`, `max_num_iterations=32` |
| Compute | local GPU0 + Ada6000 GPU0/GPU1; Ada/SUGAR overlap allowed only through explicit allowlist/headroom gate |
| Evaluation | public `eval.core.core_metrics` and frozen `core4d-e154-physics-contact-v1` metric standard |

Commands were executed through the canonical E196 wrappers:

```bash
MODE=remaining LOCAL_GPU_ID=0 ALLOW_SUGAR_OVERLAP=1 \
  bash workspace/core4d/scripts/launch/active/run_E196_reference_fix_hybrid_3gpu.sh
bash workspace/core4d/scripts/launch/active/pull_E196_reference_fix_remote_results.sh remaining
bash workspace/core4d/scripts/eval/wrappers/eval_E196_reference_fix.sh full
MUJOCO_GL=egl bash workspace/core4d/scripts/launch/active/run_E196_reference_fix_render_all.sh --require-all
```

## 3. Observed results

### Runtime and artifact integrity

| Check | Observed | Gate |
| --- | ---: | ---: |
| Corrected Full cases | 29/29 | 29/29 |
| Runtime/meta/XML parity | 29/29 | 29/29 |
| Runtime target max | 0.00000296° | <0.0001° |
| Axis target max | 0.00000296° | <0.0001° |
| Public orientation reproduction | 0.0000000000° max | <0.000001° |
| Eval errors | 0 | 0 |
| Corrected MP4 | 29/29 | 29/29 |

### Corrected G1 versus PRG

Positive improvement means lower error or higher contact, according to metric direction.

| Subset | Object orientation error | Object 3D position error | Object z MAE |
| --- | ---: | ---: | ---: |
| All affected 29 | `5.8320 → 5.1119°` (`+0.7201°`) | `11.5113 → 9.9528 cm` (`+1.5584 cm`) | `4.9536 → 3.6012 cm` (`+1.3524 cm`) |
| Box001 primary 20 | `6.0756 → 5.2587°` (`+0.8169°`) | `11.0244 → 9.4733 cm` (`+1.5510 cm`) | `4.7911 → 3.1502 cm` (`+1.6410 cm`) |
| Box023 8 | `5.2708 → 4.8846°` (`+0.3862°`) | `13.1727 → 11.7522 cm` (`+1.4205 cm`) | `5.5134 → 4.9642 cm` (`+0.5492 cm`) |

The corrected-vs-PRG object orientation improvement was positive in `19/29` cases; no case regressed by more than `5°`. Object z MAE improved in `27/29` cases. Strict 12-gate pass counts were PRG `7/29`, contaminated G1 `6/29`, corrected G1 `10/29`.

### Visual observation

All 29 corrected MP4s were rendered. A midpoint contact sheet was inspected for all cases, with priority coverage for all 7 cluster7 cases, representative local/Ada/box001/box023 samples, and gate migration cases. Most corrected midpoints showed the person and box remaining co-visible with no obvious box explosion. `box001_20231023_110_p1` showed apparent object/body fragmentation at the midpoint and is retained as a temporal-review follow-up. This is a midpoint screen, not a claim of full temporal human approval.

## 4. Interpretation

The runtime reference-fix claim is supported: the fail-closed resolver removed the convention mismatch at the actual runtime target and reproduced the public orientation metric. The performance evidence is consistent with the original diagnosis: contaminated G1 had a large object-orientation penalty, while corrected G1 reduced that penalty and improved object z/3D position. The result does not establish that gravcomp improves every hand/contact metric; corrected G1 has both `FAIL_TO_PASS` and `PASS_TO_FAIL` gate migrations. The flagged `box001_20231023_110_p1` requires a full temporal visual review before any case-level promotion decision.

## 5. Limitations and failure record

- Only the 29 convention-mismatch cases were rerun; 43 convention-match cases were not rerun.
- This is one Full seed per case; paired bootstrap intervals quantify effect uncertainty but do not replace additional seeds.
- Public hand metrics are combined left/right EEF means; no unsupported per-hand metric was inferred.
- During recovery, an SSH-reset-heavy pull exposed a local/external `results` directory split. No artifact SHA conflict occurred; 51 split files were merged with `--ignore-existing`, the split tree was retained as backup, and the canonical symlink was restored. The pull wrapper now uses `--keep-dirlinks`.
- Midpoint visual screening is not full temporal human review; the flagged case remains an explicit follow-up.

## 6. Conclusion and next steps

E196 closes the Euler reference metadata integrity bug for the 29 affected rows and validates corrected G1 as the authoritative rerun for this mismatch subset. Keep the resolver and metadata fail-closed contracts in the active path. Before downstream promotion, perform full temporal review of `box001_20231023_110_p1` and review the 10 corrected-vs-PRG `PASS_TO_FAIL` gate migrations case-by-case.

## Result paths

| Artifact | Path |
| --- | --- |
| Evaluation summary | `workspace/core4d/results/E196/s6_downstream/eval/full_reference_fix/e196_reference_fix_summary.json` |
| By-case / by-object TSV | `workspace/core4d/results/E196/s6_downstream/eval/full_reference_fix/e196_reference_fix_by_case.tsv` / `e196_reference_fix_by_object.tsv` |
| Comparison workbook | `workspace/core4d/results/E196/s6_downstream/eval/full_reference_fix/E196_reference_fix_comparison.xlsx` |
| Markdown report | `workspace/core4d/results/E196/s6_downstream/eval/full_reference_fix/E196_reference_metadata_integrity_fix_report.md` |
| Three-arm video manifest | `workspace/core4d/results/E196/s6_downstream/render/full_reference_fix/three_arm_video_manifest.tsv` |
| Visual screening evidence | `workspace/core4d/results/E196/s6_downstream/render/full_reference_fix/visual_review_summary.json` and `visual_review_notes.md` |
| Git-tracked provenance | `workspace/core4d/report/E196/provenance/` |

## Reproducibility notes

The final workbook was recalculated with LibreOffice: `5,308` formulas and `0` formula errors. Provenance SHA manifests were regenerated after eval and visual evidence. The canonical E196 artifacts remain on the external results volume behind `workspace/core4d/results` symlink; NPZ/MP4 are intentionally not duplicated into Git provenance.
