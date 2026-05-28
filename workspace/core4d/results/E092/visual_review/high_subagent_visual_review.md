# E092 High-Reasoning Visual Review

Reviewer: `019e6fc7-1fa9-7442-af87-b56c8f376236`

Inputs:

- `workspace/core4d/results/E092/visual_review/contact_sheets/E092D1_box004_083_p2_dyn_smoke_sheet.jpg`
- `workspace/core4d/results/E092/visual_review/contact_sheets/E092D2_box026_039_p2_dyn_smoke_sheet.jpg`
- `workspace/core4d/results/E092/visual_review/contact_sheets/E092D3_box026_135_p2_dyn_smoke_sheet.jpg`
- `workspace/core4d/results/E092/visual_review/contact_sheets/E092O1_box004_083_p2_omni_smoke_sheet.jpg`
- `workspace/core4d/results/E092/visual_review/contact_sheets/E092O2_box026_039_p2_omni_smoke_sheet.jpg`
- `workspace/core4d/results/E092/visual_review/contact_sheets/E092O3_box026_135_p2_omni_smoke_sheet.jpg`

## Per-Case Observations

- D1/C1 box004 spider_dyn: clear failed posture. The robot bends toward the box early, then enters severe pelvis/hip collapse and falls backward. Hand-floor support is not visually dominant, matching RH floor `1.0%`. No clear head or upper-body box penetration is visible. The box stays visually stable, but useful hand-object contact is essentially absent, matching contact `3.8%`.
- D2/C2 box026_039 spider_dyn: the robot sits/kneels/leans near the box edge. It does not fully lie flat like C1, but the hips are low and the posture is unstable. No obvious hand-floor support appears, matching `0%` hand-floor. No clear head or upper-body penetration is visible. Object tracking looks stable, but the motion semantics fail.
- D3/C3 box026_135 spider_dyn: clear collapse. In the later frames the robot falls sideways/backward, with the right hand/arm close to or supporting on the floor, matching RH floor `39.0%`. The pelvis/hips drop clearly. No clear head or upper-body penetration is visible. The box rotates/lifts unnaturally in later frames.
- O1/C1 box004 rl_from_omni: visually almost identical to D1. Severe pelvis collapse and backward fall; only brief right-hand floor proximity if any. No clear head or upper-body box penetration. The box is stable, but contact semantics fail.
- O2/C2 box026_039 rl_from_omni: close to D2, but the later hips/torso appear lower, consistent with pelvis min dropping from `0.191m` to `0.135m`. No obvious hand-floor support. No clear head or upper-body penetration. Object tracking is mostly stable, but contact/posture is not credible.
- O3/C3 box026_135 rl_from_omni: highly similar to D3. Later collapse and right hand/arm floor contact are clear, matching RH floor `40.2%`. No clear head or upper-body penetration. The box undergoes large rotation/lift, making the interaction visually untrustworthy.

## Route Difference

The spider_dyn and rl_from_omni smoke videos show no meaningful visual improvement from either route. C1 is essentially the same, C3 is essentially the same, and C2 only shows slightly worse low-hip collapse on rl_from_omni.

## Recommendation

Do not continue Stage A full, Stage B rl_from_spider, or Stage C main from these smoke results. The failures are visually real, not only metric artifacts. The next experiment should first address pelvis/upright stability, hand-object contact semantics, and hand-floor/fall shortcuts before spending main-run compute.
