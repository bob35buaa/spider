# E143 Raw-Mask Ref-FK 24-Case OmniRetarget Comparison Plan

## Context

The current objective is to improve and audit retargeting hand-object contact against OmniRetarget, not Holosoma or RL. E142 showed that existing E112/E113 contact-aware outputs improved over Spider/ref-FK on selected cases but did not exceed OmniRetarget under the same-case physics contact metric.

E143 runs the narrow requested sweep: `raw_mask_ref_fk` on the E109 expanded 24-case workset, reusing completed raw-mask runs where they already exist and launching only missing cases.

## Claims

1. The E109 expanded 24-case workset can be converted into an E143 `raw_mask_ref_fk` manifest with task, person index, mask path, baseline Spider result, and OmniRetarget reference metrics.
2. Previously completed `raw_mask_ref_fk` full outputs are identified and reused instead of rerun.
3. Missing full outputs can be split across local GPU0 plus `spider-remote` GPU0/GPU1 without killing unrelated processes.
4. The final E143 evaluation reports per-case and aggregate comparison against OmniRetarget for hand-object contact, 5cm/10cm proximity, hand-object penetration, and leg penetration.

## Scope

- Input workset:
  - `workspace/core4d/results/E109/expanded_24_work_cases/expanded_24_case_comparison.tsv`
  - `workspace/core4d/results/E109/expanded_24_work_cases/expanded_24_method_metrics.tsv`
  - `workspace/core4d/results/E110/contact_metric_audit/contact_band_metrics.tsv`
- Existing reusable raw-mask outputs:
  - E112 full `raw_mask_ref_fk` rows, when case/task-compatible.
- New E143 outputs:
  - manifest and overrides under `workspace/core4d/scripts/E143/` and `examples/config/override/core4d_E143_*`
  - full CEM outputs under `workspace/core4d/results/E143/cem/full/`
  - evaluation and xlsx comparison under `workspace/core4d/results/E143/raw_mask_ref_fk_24case_omniretarget_comparison/`

## Method

Use the E112 raw-mask configuration pattern:

- `contact_hdmi_target_source: ref_fk`
- `contact_hdmi_gain: 5.0`
- `contact_hdmi_mask_source: core4d_3cm`
- `contact_hdmi_mask_time_axis: auto`
- no hold-band reward

For E104 raw proxy masks, convert the 3cm threshold to MJWP-compatible `raw_contact_mask_3cm.npz` by nearest-neighbor resizing to each task trajectory length, matching E112/E113 practice.

## Parallel Execution

- local GPU0: assigned E143 split `local-gpu0`
- remote GPU0: assigned E143 split `remote-gpu0`
- remote GPU1: assigned E143 split `remote-gpu1`

Follow `experiment-planning-zh/remote-execution.md`. Because the repository currently contains many untracked experiment scripts and overrides, E143 may use `rsync` for untracked E143 artifacts if git push/pull is not sufficient; this must be recorded in the result log.

## Success Criteria

- E143 manifest contains exactly 24 E109 cases.
- Every manifest row is either:
  - `already_done` with existing full raw-mask NPZ/MP4/outdir trajectory, or
  - `to_run` with a valid override, task scene, task trajectory, baseline path, and contact mask.
- Only `to_run` cases are launched.
- After run recovery, 24/24 raw-mask cases have full output artifacts or are explicitly reported as failed with logs.
- Evaluation joins every E143 case to OmniRetarget metrics.
- Final summary and `.xlsx` contain:
  - method-level aggregate rows for OmniRetarget, Spider/ref-FK baseline, and E143 `raw_mask_ref_fk`;
  - case-by-case rows with hand-object contact, 5cm/10cm proximity, hand-object penetration, and leg penetration;
  - deltas of E143 `raw_mask_ref_fk` vs OmniRetarget.

## Fixed Commands

```bash
python workspace/core4d/scripts/E143/build_raw_mask_ref_fk_24case_manifest.py
bash workspace/core4d/scripts/train/train_E143_raw_mask_ref_fk_24case.sh list full local
bash workspace/core4d/scripts/run_E143_remote.sh full
bash workspace/core4d/scripts/pull_E143_remote_results.sh full
bash workspace/core4d/scripts/eval/eval_E143_raw_mask_ref_fk_24case.sh full
```

## Non-Goals

- No Holosoma changes.
- No RL.
- No new reward family beyond E112-style `raw_mask_ref_fk`.
- No rerun of cases that already have compatible full raw-mask outputs.
