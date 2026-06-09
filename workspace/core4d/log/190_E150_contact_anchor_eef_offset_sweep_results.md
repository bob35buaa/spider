# E150 contact anchor eef_offset sweep results

Date: 2026-06-09 CST

Plan: `workspace/core4d/plan/158_E150_contact_anchor_eef_offset_sweep_plan.md`

## Scope

E150 tests route A from the contact-anchor diagnosis: keep the rubber hand CEM setup unchanged, but move the shared `contact_hdmi_eef_offset` anchor from the default 0.05m to 0.08m and 0.11m along wrist local x.

Benchmark set: E149 `relaxed8_valid_like`.

Cases:

- `box021_035_p2`
- `box023_person2`
- `box021_029_p2`
- `box004_082_p1`
- `box004_083_p2`
- `box021_035_p1`
- `box026_139_p1`
- `box004_083_p1`

Baseline `off05` reuses E148/E147 rubber trajectories. New full CEM runs are only `off08` and `off11`: 8 cases x 2 offsets = 16 runs.

## Execution

Remote full CEM ran on `spider-remote` in tmux session `E150_full_181630`.

Split:

- GPU0: all `off08` rows, 8 runs
- GPU1: all `off11` rows, 8 runs

Commands:

- manifest: `.venv/bin/python workspace/core4d/scripts/E150/build_eef_offset_sweep_manifest.py`
- launch: `bash workspace/core4d/scripts/run_E150_remote.sh full`
- pull: `bash workspace/core4d/scripts/pull_E150_remote_results.sh full`
- eval: `bash workspace/core4d/scripts/eval/eval_E150_eef_offset_sweep.sh full`
- xlsx recalc: `python /home/ubuntu/.codex/skills/xlsx/scripts/recalc.py workspace/core4d/results/E150/contact_anchor_eef_offset_sweep/comparison/E150_contact_anchor_eef_offset_sweep.xlsx 60`

Smoke note: the first smoke session exposed a `box023_person2` task/sidecar mismatch. E148's E143 row uses `box023_person2_legobj_e026_e081`, but the reused E147 rubber override/sidecar lives under `box023_person2_legobj`. E150 manifest generation was fixed to derive the run-side `derived_task/target_scene/trajectory/base_scene_act` from `rubber_scene_act`'s task directory. The full run then used `task=box023_person2_legobj` and completed.

The tmux session ended naturally. No remote process was killed. SSH status reads intermittently hit kex disconnects, but tmux/GPU/log checks showed healthy execution.

## Artifacts

| Artifact | Path | Count |
|---|---|---:|
| manifest | `workspace/core4d/scripts/E150/variants.tsv` | 24 rows |
| baseline reused rows | off05 rubber paths from E148/E147 | 8 |
| new root NPZ | `workspace/core4d/results/E150/contact_anchor_eef_offset_sweep/cem/full/*.npz` | 16 |
| new MP4 | `workspace/core4d/results/E150/contact_anchor_eef_offset_sweep/cem/full/*_full.mp4` | 16 |
| new outdir trajectory | `workspace/core4d/results/E150/contact_anchor_eef_offset_sweep/cem/full/*_outdir_full/trajectory_mjwp_act.npz` | 16 |
| config | `workspace/core4d/results/E150/contact_anchor_eef_offset_sweep/cem/full/*_outdir_full/config_act.yaml` | 16 |
| run logs | `logs/E150/cem/full/*.log` | 16 |
| keyframe dirs | `workspace/core4d/results/E150/contact_anchor_eef_offset_sweep/cem/full/keyframes/` | 16 dirs |
| visual sheets | `workspace/core4d/results/E150/contact_anchor_eef_offset_sweep/visual_inspection/` | 6 jpg |
| method metrics | `workspace/core4d/results/E150/contact_anchor_eef_offset_sweep/eval/full/e150_method_metrics.tsv` | 24 rows |
| offset delta | `workspace/core4d/results/E150/contact_anchor_eef_offset_sweep/eval/full/e150_offset_delta.tsv` | 16 rows |
| eval summary | `workspace/core4d/results/E150/contact_anchor_eef_offset_sweep/eval/full/e150_summary.md` | 1 |
| comparison XLSX | `workspace/core4d/results/E150/contact_anchor_eef_offset_sweep/comparison/E150_contact_anchor_eef_offset_sweep.xlsx` | 5 sheets |

Workbook sheets after LibreOffice recalculation:

- `offset平均`: 3 data rows
- `offset_delta`: 16 data rows
- `delta平均`: 2 data rows
- `逐case_metrics`: 24 data rows
- `object分组`: 12 data rows

Formula recalculation reported `total_errors=0`, `total_formulas=56`.

Config audit: all 8 `off08` configs contain `contact_hdmi_eef_offset=[0.08,0,0]`; all 8 `off11` configs contain `contact_hdmi_eef_offset=[0.11,0,0]`.

## Quantitative Result

### Offset averages

| offset | cases | fall | 5cm | 10cm | hand pen | deep2 | physics contact | leg pen | obj err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `off05` | 8 | 0 | 0.6592 | 0.6940 | 0.2777 | 0.0084 | 0.3070 | 0.0953 | 0.0083 |
| `off08` | 8 | 0 | 0.6613 | 0.6912 | 0.2846 | 0.0035 | 0.3120 | 0.0956 | 0.0084 |
| `off11` | 8 | 0 | 0.6605 | 0.6928 | 0.3123 | 0.0047 | 0.3800 | 0.0890 | 0.0085 |

### Mean deltas vs off05

| offset | success cases | 5cm delta | 10cm delta | hand pen delta | deep2 delta | physics contact delta | leg pen delta | obj err delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `off08` | 0/8 | +0.0020 | -0.0028 | +0.0069 | -0.0049 | +0.0050 | +0.0003 | +0.0000 |
| `off11` | 0/8 | +0.0012 | -0.0012 | +0.0346 | -0.0038 | +0.0730 | -0.0063 | +0.0001 |

### Per-case deltas

| case | off | 5cm delta | 10cm delta | hand pen delta | deep2 delta | physics contact delta | leg pen delta | success |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `box004_082_p1` | `off08` | +0.0000 | +0.0000 | -0.0183 | +0.0000 | -0.0183 | +0.0000 | false |
| `box004_082_p1` | `off11` | +0.0000 | +0.0092 | -0.0183 | +0.0092 | -0.0275 | +0.0000 | false |
| `box004_083_p1` | `off08` | +0.0196 | +0.0098 | +0.1176 | +0.0000 | +0.1373 | +0.0490 | false |
| `box004_083_p1` | `off11` | +0.0098 | +0.0098 | -0.0686 | +0.0000 | +0.0980 | -0.0490 | false |
| `box004_083_p2` | `off08` | -0.0095 | -0.0190 | +0.0381 | +0.0000 | +0.0381 | -0.0476 | false |
| `box004_083_p2` | `off11` | +0.0000 | +0.0000 | +0.0857 | +0.0000 | +0.0762 | -0.0190 | false |
| `box021_029_p2` | `off08` | +0.0133 | -0.0133 | -0.0267 | -0.0533 | -0.0267 | +0.0000 | false |
| `box021_029_p2` | `off11` | +0.0000 | -0.0133 | +0.0000 | -0.0533 | +0.0667 | +0.0133 | false |
| `box021_035_p1` | `off08` | +0.0000 | +0.0000 | -0.0155 | +0.0000 | -0.0078 | +0.0000 | false |
| `box021_035_p1` | `off11` | +0.0000 | -0.0078 | +0.1628 | +0.0000 | +0.1783 | +0.0078 | false |
| `box021_035_p2` | `off08` | +0.0000 | +0.0000 | -0.0677 | +0.0000 | -0.0677 | +0.0226 | false |
| `box021_035_p2` | `off11` | +0.0000 | -0.0075 | +0.0451 | +0.0000 | +0.1654 | +0.0602 | false |
| `box023_person2` | `off08` | -0.0074 | +0.0000 | -0.0074 | +0.0000 | -0.0221 | -0.0074 | false |
| `box023_person2` | `off11` | +0.0000 | +0.0000 | -0.0147 | +0.0000 | -0.0294 | -0.0074 | false |
| `box026_139_p1` | `off08` | +0.0000 | +0.0000 | +0.0352 | +0.0141 | +0.0070 | -0.0141 | false |
| `box026_139_p1` | `off11` | +0.0000 | +0.0000 | +0.0845 | +0.0141 | +0.0563 | -0.0563 | false |

## Visual Observations

Visual inspection sheets:

- `workspace/core4d/results/E150/contact_anchor_eef_offset_sweep/visual_inspection/box021_035_p1_t24_off05_off08_off11.jpg`
- `workspace/core4d/results/E150/contact_anchor_eef_offset_sweep/visual_inspection/box021_035_p1_t32_off05_off08_off11.jpg`
- `workspace/core4d/results/E150/contact_anchor_eef_offset_sweep/visual_inspection/box004_083_p1_t24_off05_off08_off11.jpg`
- `workspace/core4d/results/E150/contact_anchor_eef_offset_sweep/visual_inspection/box004_083_p1_t32_off05_off08_off11.jpg`
- `workspace/core4d/results/E150/contact_anchor_eef_offset_sweep/visual_inspection/box026_139_p1_t24_off05_off08_off11.jpg`
- `workspace/core4d/results/E150/contact_anchor_eef_offset_sweep/visual_inspection/box026_139_p1_t32_off05_off08_off11.jpg`

Observed:

- `box021_035_p1`: off08 is visually very close to off05. off11 puts the hand/forearm visibly deeper against the box at the later frame, matching `hand_pen_delta=+0.1628` and `physics_contact_delta=+0.1783`.
- `box004_083_p1`: off08/off11 keep a similar bent-over pose. off08 shows more object-edge/foot proximity and is the worst case for off08 penetration increase (`+0.1176`) and leg penetration (`+0.0490`).
- `box026_139_p1`: off08/off11 do not create a clean additional hand-surface grasp. off11 increases hand/object proximity on the side/lid area, but the metric result is penetration growth (`+0.0845`) rather than clean 5cm contact gain.

Overall, the visual check supports the quantitative result: moving the reward anchor forward did not create a robust hand-surface contact improvement. The off11 physical-contact increase is mostly a contact/penetration tradeoff, not a clean contact-quality win.

## Claims

| Claim | Result | Evidence |
|---|---|---|
| Route A pure config sweep | PASS | Only `contact_hdmi_eef_offset=[0.08/0.11,0,0]` was injected at CEM runtime; no reward/optimizer/SPIDER algorithm change. |
| Remote 2-GPU full CEM | PASS | `E150_full_181630` ran off08 on GPU0 and off11 on GPU1, 16/16 results recovered. |
| EEF metric parameterization | PASS | E150 evaluator sets `eval_mod.EEF_OFFSET` per row before sequence eval; full eval produced 24 rows and `missing=0`. |
| Success criterion | FAIL | Required mean 5cm hand contact delta >= +0.03 with no hand-penetration increase. off08: +0.0020 5cm, +0.0069 penetration. off11: +0.0012 5cm, +0.0346 penetration. Per-case successes: 0/16. |

## Decision

E150 does not validate route A as a useful fix.

Forward-moving the shared `contact_hdmi_eef_offset` from 0.05 to 0.08 or 0.11 does not materially improve hand-geometry 5cm/10cm contact on the E149 relaxed8 benchmark. The 0.11 setting increases physical contact fraction, but it also increases hand-object penetration, which is exactly the failure mode this experiment was meant to avoid.

The contact-anchor diagnosis remains plausible as a mechanism, but a pure scalar x-offset is too blunt. The next useful direction should be route B style: reward/eval should reason about rubber-hand surface geometry or multiple contact anchors, rather than pulling a single wrist-derived point farther forward.

## Post-hoc verification: why route A is mathematically inert (2026-06-09)

After the flat result, we asked the deeper question: was the `contact_hdmi` reward even responsive to `eef_offset` during CEM? Rather than re-run GPU, we recomputed the reward **offline on the already-executed trajectories**.

Method (no simulation, deterministic):
- For each of the 8 cases, load the off08 npz (`qpos` stores `[sim, ref]` for every frame) and the case's snapshot rubber scene.
- Replicate `mjwp.py:1033-1044` exactly via MuJoCo FK: `target_world = obj_pos_sim + R(obj_quat_sim)·target_local`, where `target_local = R_obj_ref^T·((ref_wrist + R(ref_wrist_quat)·off) − obj_pos_ref)` (per `run_mjwp.py:1026-1038`, `target_uses_eef_offset=True`); `contact_point = sim_wrist + R(sim_wrist_quat)·off`; `pos_rew = exp(−‖target_world − contact_point‖/σ)`, σ=0.3.
- Sweep `off ∈ {0.05, 0.08, 0.11}` on the **same fixed states** and measure how much the reward moves.

Result — `contact_hdmi` pos_rew is nearly invariant to eef_offset:

| case | rew@0.05 | rew@0.08 | rew@0.11 | Δ(0.05→0.11) | rel% | dist@.05 (cm) | dist@.11 (cm) | coupling ‖·‖₂ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| box021_035_p1 | 0.97689 | 0.97581 | 0.97458 | −0.00232 | −0.24% | 0.719 | 0.794 | 0.0282 |
| box021_035_p2 | 0.97949 | 0.97888 | 0.97805 | −0.00143 | −0.15% | 0.629 | 0.674 | 0.0248 |
| box021_029_p2 | 0.96872 | 0.96820 | 0.96729 | −0.00144 | −0.15% | 0.969 | 1.015 | 0.0476 |
| box004_082_p1 | 0.96739 | 0.96693 | 0.96609 | −0.00130 | −0.13% | 1.011 | 1.051 | 0.0409 |
| box004_083_p1 | 0.97446 | 0.97383 | 0.97299 | −0.00148 | −0.15% | 0.786 | 0.833 | 0.0304 |
| box004_083_p2 | 0.96148 | 0.96034 | 0.95883 | −0.00265 | −0.28% | 1.203 | 1.291 | 0.0449 |
| box023_person2 | 0.97012 | 0.96887 | 0.96735 | −0.00276 | −0.28% | 0.930 | 1.022 | 0.0409 |
| box026_139_p1 | 0.97038 | 0.96893 | 0.96718 | −0.00320 | −0.33% | 0.913 | 1.018 | 0.0457 |
| **MEAN(8)** | **0.97112** | **0.97022** | **0.96904** | **−0.00207** | **−0.21%** | **0.895** | **0.962** | **0.0379** |

Worst-case relative reward change across all 8 cases: **0.33%**.

Two findings:

1. **eef_offset is mathematically inert for this reward.** `off` enters BOTH `target_world` (via ref FK) and `contact_point` (via sim FK), so the net offset term is `[R(sim_wrist) − R_obj_sim·R_obj_ref^T·R(sim... ref_wrist)]·off`. The coupling matrix's spectral norm averages **0.038**, so a 6 cm offset move (0.05→0.11) shifts the tracked distance by only ~0.7 mm — negligible against σ=30 cm. This is why route A's measured 5cm-contact delta (+0.001~+0.002) is in the noise: it is not "under-tuned", the lever is **orthogonal** to the reward. **Route A is falsified.**

2. **The reward is already saturated (~0.97) yet hand contact stays low (5cm 0.46, physics 0.31).** The reward measures "replicate the reference wrist-to-object relative pose" and it *succeeds* (sub-cm distance, 0.97 pos_rew). But (a) the reference (OmniRetarget replay) hand is itself not on the box surface, and (b) the reward never reads the rubber-hand mesh or the box face. So **reward-perfect ≠ hand-on-box**. The ceiling is the reference's own contact quality, not the offset. This is the saturation paradox that motivates route B (read real hand↔box geometry, not a wrist-derived point).

Verification was a pure offline recompute (no GPU, no scene/code change); reproduced across all 8 relaxed8 cases including the E143-mask case box021_029_p2.

## Validation

- Manifest build: 24 rows = 8 off05 reuse + 16 to-run; split 8 off08 / 8 off11.
- Static checks before run: `py_compile`, shell `bash -n`, `git diff --check`.
- Remote full: 16/16 root NPZ, 16/16 MP4, 16/16 outdir trajectory, 16 logs.
- Config audit: off08/off11 offsets present in 16/16 configs.
- Full eval: `method_rows=24`, `delta_rows=16`, `missing=0`.
- XLSX recalculation: `total_errors=0`, `total_formulas=56`.
- Workbook readback: sheets `offset平均`, `offset_delta`, `delta平均`, `逐case_metrics`, `object分组` with row counts 3/16/2/24/12.
- Visual inspection: six generated side-by-side sheets checked and summarized above.
