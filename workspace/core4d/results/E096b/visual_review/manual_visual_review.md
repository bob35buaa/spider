# E096b manual visual review

Date: 2026-05-29

Inputs:

- `workspace/core4d/results/E096b/visual_review/frame_sheets/cem_p1_sheet.jpg`
- `workspace/core4d/results/E096b/visual_review/frame_sheets/cem_p2_sheet.jpg`
- `workspace/core4d/results/E096b/cem/full/keyframes/E096bP1_box004_083_p1_mask_cem/`
- `workspace/core4d/results/E096b/cem/full/keyframes/E096bP2_box004_082_p1_mask_cem/`

## Summary

Both E096b mask-on CEM variants visually support the numeric `WORK` decision.

| variant | visual decision | notes |
|---|---|---|
| `E096bP1_box004_083_p1_mask_cem` | WORK | Full body remains upright enough through the sequence; no obvious fall, prone-on-box failure, head/upper-body box penetration, or hand-floor support. Box stays close to reference and the final pose is consistent with the E096 positive pattern. |
| `E096bP2_box004_082_p1_mask_cem` | WORK | More visible box orientation change than P1, but the robot stays standing/bent rather than collapsing; no clear head/upper-body/hand-floor safety failure. Object motion remains visually aligned with the reference and matches the low object-error metrics. |

## Frame Checks

P1 inspected frames:

- `f40.jpg`: sim follows the bent reach pose with hands near the box; no visible upper-body compression into the box.
- `f85.jpg` / `f100.jpg`: carrying phase is upright enough, with hands near object sides/top and no hand-floor contact.
- `f140.jpg` / `f160.jpg`: late sequence remains stable; box is still close to the robot and no fall mode appears.

P2 inspected frames:

- `f40.jpg`: initial contact pose tracks the reference; hands/markers are around the box, not on the floor.
- `f85.jpg` / `f100.jpg`: carrying phase stays upright; no head/torso support on the box.
- `f140.jpg` / `f160.jpg`: robot remains standing/bent; box is tilted relative to the reference in late frames, but not enough to contradict the object mean/max error or safety metrics.

## Decision

Use E096b P1/P2 as the cleaner mask-on evidence for the box004 positive set. E096 remains useful as the prior run, but E096b is the run to cite when the contact-mask configuration matters.
