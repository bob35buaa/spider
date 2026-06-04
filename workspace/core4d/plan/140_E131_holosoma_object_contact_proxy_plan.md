# E131 Holosoma Object-Contact Proxy Plan

Date: 2026-06-03

## Context

E130 confirmed that the E126 paired Holosoma fragment exports can support offline
Box021 surface-distance inspection, but both exports lack `object_contact`. In
Holosoma, `MotionLoader` accepts `object_contact` only when it has shape
`(num_frames, 2)` and casts it to bool. Without this key, ref-masked contact
reward branches see an all-false mask.

No upstream real raw/trimmed per-hand mask is currently present for the E126
fragment exports. E131 therefore must not claim raw-contact semantics. Its scope
is to create a clearly labeled proxy artifact that closes the structural loader
contract for bounded reward/debug inspection only.

## Claims

1. A separate E131 export can add `object_contact (T,2)` to each E126 paired
   fragment without mutating E126 artifacts.
2. The mask source can be explicitly labeled as `actor_rubber_hand_box021_surface_proxy_5cm`.
3. The proxy can structurally unblock Holosoma ref-mask loader checks while
   preserving `semantic_ref_mask_ready=false`, `rl_ready_rows=0`,
   `training_launched=false`, and `cem_launched=false`.

## Method

- Load the two E126 paired exports.
- Use actor-side `left_rubber_hand_link/right_rubber_hand_link` positions because
  the E126 body list does not contain Holosoma `left/right_handbox_link`.
- Compute signed distance from each hand point to the Box021 oriented bounding
  box using half extents `[0.195495, 0.15155, 0.25004]`.
- Set proxy `object_contact[:, hand] = surface_distance <= 0.05`.
- Store masked exports under E131, not E126.
- Write manifest fields that distinguish:
  - structural readiness for loader/ref-mask smoke
  - semantic readiness for real ref-contact training

## Outputs

```text
workspace/core4d/results/E131/holosoma_object_contact_proxy/
  exports/*_object_contact_proxy5cm.npz
  e131_object_contact_proxy_manifest.tsv
  e131_object_contact_proxy_summary.json
  e131_object_contact_proxy_summary.md
```

## Success Criteria

- 2/2 paired exports produce proxy-contact NPZs.
- Each output has `object_contact` shape `(T,2)` and dtype bool-compatible.
- Manifest records active fractions and longest runs per hand.
- Summary has `structural_ref_mask_ready_rows=2`, `semantic_ref_mask_ready_rows=0`,
  `rl_ready_rows=0`, `training_launched=false`, `cem_launched=false`.
- No CEM, PPO, Holosoma training, checkpoint creation, or remote jobs are launched.

## Guardrails

- Do not rewrite E126 source exports.
- Do not use the proxy as raw-contact ground truth.
- Do not claim the fragments solve the main `box021_029_p2` gate.
- Do not start PPO or full CEM from these artifacts.
