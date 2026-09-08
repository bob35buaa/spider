# 0908 Paper results — SPIDER-CEM vs OmniRetarget

## Purpose
Aggregate/compute a paper-ready SPIDER-CEM vs OmniRetarget comparison over the 52 cases in
`tmp/paper_case_id.txt`, in Markdown / xlsx / LaTeX, at per-case + per-object + overall
granularity (mean ± std + worst).

## Version selection (SPIDER-CEM source per object group)
- chair006 / desk007 / desk021 / desk023 → **E212** (selected arm, `paired_rl_export_input.tsv`)
- box001 / box004 / box024 → **E198 G1A2** (`{obj}_user_approved/rl_export_input.tsv`)
- box021 → **E170 PRG** (`paired_rl_export_input.tsv`; metrics from `e170_case_metrics.tsv`)
  - except `box021_20231018_029_p2` and `box021_20231011_035_p1`, which are **not in E170** →
    per-case override to **E167A** (`E167/holosoma_zonly` export, `E167A_zOnlyBody` variant);
    contact/tracking/health/foot reused from `e167_arm_metrics.tsv` (E167's mask alignment),
    `body_z` recomputed. Their E167-era contact mask is resampled to the rollout timeline
    for the OmniRetarget in-mask metrics (noted `mask_resampled_to_rollout` in provenance).
- box023 → **E190 noPRG** (metrics from **E179** `e179_case_metrics.tsv`)
- bucket003 → **E178** ; bucket007 → **E207**

## Metrics (keys from `eval.core.core_metrics.evaluate_sequence` + `motion_health.run_health`)
- Both methods: raw contact `hand_object_physics_contact_in_mask_frac`, contact@3mm
  `hand_object_physics_contact_3mm_in_mask_frac`, penetration@3mm
  `hand_object_physics_penetration_3mm_frame_frac`, geom@2mm `hand_geom_penetration_2mm_frac`.
- SPIDER-CEM only: 14-gate tracking (root/eef/obj pos+ori), health (fall, body_z_err_p95_m,
  ankle_jerk_p95, obj_speed_max), and foot slide `foot_slip_max_m`.

## Method (reuse precomputed + fill gaps)
Per case: reuse the value from the source `*_case_metrics.tsv` where the column exists;
otherwise recompute from the case's selected `cem_result_npz`. Pure-reuse groups (box021 E170,
bucket003 E178) do no replay. OmniRetarget physics metrics come from MuJoCo position-servo replay
of the SPIDER kinematic reference (`trajectory_kinematic.npz`, = OmniRetarget output in raw
world-freejoint form) under the same `scene_act`, `contact_mask`, and frame window — the E197/E109
protocol. Roundtrip world-pose error < 1e-4°.

Consistency: 291 recompute-vs-reuse overlap comparisons, **0 mismatches > 1e-3**.

## Run commands
```bash
# heavy compute (cached, resumable):
.venv/bin/python workspace/core4d/report/0908/gen_paper_results.py --fresh --compute-only
# path-resolution audit only (no replay):
.venv/bin/python workspace/core4d/report/0908/gen_paper_results.py --audit
# write md/xlsx/latex/tsv/provenance from cache:
.venv/bin/python workspace/core4d/report/0908/gen_paper_results.py
```

## Outputs (`paper_results/`)
`paper_results.md`, `paper_results.xlsx`, `paper_results.tex`, `paper_results_by_case.tsv`,
`provenance.json` (per-metric source flags + xcheck), `_cache_method_metrics.jsonl`,
`omni_scene_act_qpos/` (replayed OmniRetarget qpos for rendering/spot-check).

## Coverage
52/52 cases resolved for both methods (all 15 SPIDER metrics n=52, all 4 OmniRetarget
metrics n=52). 291 recompute-vs-reuse comparisons, 0 mismatches > 1e-3, 0 unresolved.

## Result (overall, 52 cases)
| metric | SPIDER-CEM | OmniRetarget |
|---|---|---|
(Overall numbers are recomputed by the report from the 52-case cache; see `paper_results.md`
for the authoritative per-object and overall tables.) SPIDER-CEM keeps ~59% clean contact@3mm
with ~17% penetration@3mm and ~5% geom@2mm, vs OmniRetarget's ~9% clean contact@3mm with
~54%/~39% penetration — the expected physics-optimized-vs-kinematic gap.

## Notes
- Metric definitions: `workspace/core4d/scripts/eval/core/{core_metrics,motion_health}.py`;
  14-gate authority `workspace/core4d/scripts/experiments/E201/funnel_config.py`.
- Engine adapts `workspace/core4d/scripts/eval/reports/gen_E197_full_cem_omnirt_vs_prg_metrics.py`.
