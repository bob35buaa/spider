# E015 Visual Evaluation

Montage: `workspace/core4d_collab_retarget/results/E015/keyframes/E015_visual_montage.jpg`

## Observations

| Variant | Visual conclusion |
|---------|-------------------|
| `E015_box025_p2_m2_kp500` | Object does move with the reference direction, but the box rotates upward/sideways through the carry window and does not stay aligned like E014. The robot keeps some hand contact, but support lag is visible as a delayed, over-rotated carry rather than the tight E014 weld behavior. |
| `E015_box025_p2_m1_kp500` | The rollout becomes numerically invalid after the early frames; later montage cells render black. Before failure, the object already shows large orientation drift. This matches the all-sample NaN reward warnings and NaN object metrics. |
| `E015_box025_p2_m2_kp1000` | The rollout also becomes numerically invalid, with black later frames. Early frames show aggressive object tilt and shadow/floor interaction before the renderer loses valid frames. This is not an effort-tuned improvement over default. |
| `E015_box023_p2_m2_kp500` | Guard remains upright/stable overall, but the motion looks like weak robot-side contact around a small object rather than a clean carry. The object tracks more closely than main, but hand contact is visually intermittent and below guard target. |

## Summary

E015 dynamic support does not visually reproduce E014's tight kinematic support behavior. The default main is dominated by support lag and rotation, while lower mass and higher kp variants become numerically unstable. Guard stability is preserved, but it is not sufficient to establish COLA A+B.
