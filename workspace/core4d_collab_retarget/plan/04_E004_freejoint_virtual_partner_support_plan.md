# E004 Plan: true-freejoint virtual partner support

日期：2026-05-17

## Context

E002 showed that true-freejoint object transport fails when the object has no actuator guidance. E003 showed that passive mass/friction tuning is not enough: the best main result (`box025_m1_f4`) improves object mean error to `0.400m`, but remains floor-supported and far worse than E081; the `box023` guard fails and can fall over.

This matches the task-level assumption in `workspace/exp_task.md`: large CORE4D objects are collaborative, and sim2real will have a real human supporting the other side. The next experiment should test explicit virtual partner support while preserving true-freejoint evaluation.

The code already has a freejoint-compatible external-force path:

- `spider.config.Config.partner_force_scale`
- `partner_force_spring_kp/kd`
- `partner_force_spring_kp_rot/kd_rot`
- `spider/simulators/mjwp.py::_apply_partner_force`
- `examples/run_mjwp.py` precomputes `partner_force_ref_pos/ref_quat` from freejoint qpos when spring or weld is enabled.

This applies `xfrc_applied` to the object body; it is not `scene_act` actuator control and keeps `nu=29`.

## Claims

| Claim | Minimum evidence |
|-------|------------------|
| C1 virtual partner support is the missing collaborative factor | `box025_p2` object mean/max improves below E003, ideally near E081 case-window, while `contact_guidance=false`, `nu=29`, `nq_obj=7`. |
| C2 support must not become unreported GT object actuation | Log force settings explicitly; eval must still use freejoint `scene.xml`, no object actuators, no `object_pd_override`, and videos must show robot/object physics rather than `scene_act` tracking. |
| C3 guard remains stable | `box023_p2` should not fall over and leg interference should remain `<=5%`; otherwise the support model is too aggressive. |

## Variants

Start from E002 original freejoint leg-object tasks, not E003 lightweight tasks, so the experiment answers whether collaborator support can make the physically realistic true-freejoint setup usable.

| Variant | Source task | Support | Role |
|---------|-------------|---------|------|
| `E004_box025_p2_g05` | `box025_person2_freejoint_legobj` | partner gravity compensation `0.5`, no spring | main sanity |
| `E004_box025_p2_g05_s20_r2` | `box025_person2_freejoint_legobj` | gravity `0.5`, position spring `kp=20`, auto kd, rot spring `kp=2`, rot clamp `0.5` | main |
| `E004_box023_p2_g05_s10_r1` | `box023_person2_freejoint_legobj` | gravity `0.5`, position spring `kp=10`, auto kd, rot spring `kp=1`, rot clamp `0.5` | guard |

Optional follow-up if main improves but guard destabilizes: lower `box023` spring to `kp=5` or use gravity-only guard.

## Implementation

新增：

- `workspace/core4d_collab_retarget/scripts/E004/variants.tsv`
- `workspace/core4d_collab_retarget/scripts/E004/generate_e004_overrides.py`
- `workspace/core4d_collab_retarget/scripts/run_E004_preprocess.sh`
- `workspace/core4d_collab_retarget/scripts/train/train_E004.sh`
- `workspace/core4d_collab_retarget/scripts/run_E004_remote.sh`
- `workspace/core4d_collab_retarget/scripts/pull_E004_remote_results.sh`
- `workspace/core4d_collab_retarget/scripts/eval/eval_E004.py`
- `workspace/core4d_collab_retarget/log/04_E004_freejoint_virtual_partner_support_results.md`

No new dataset scene is required unless a follow-up wants a force-site/visual marker. E004 should reuse E002 tracked freejoint tasks and generate only Hydra overrides.

## Commands

```bash
bash workspace/core4d_collab_retarget/scripts/run_E004_preprocess.sh
bash workspace/core4d_collab_retarget/scripts/train/train_E004.sh smoke 0
bash workspace/core4d_collab_retarget/scripts/run_E004_remote.sh
bash workspace/core4d_collab_retarget/scripts/pull_E004_remote_results.sh
```

## Success Criteria

| Metric | Threshold |
|--------|-----------|
| `box025_p2` case-window object mean | `<0.30m` useful, `<0.20m` strong |
| `box025_p2` floor contact | lower than E003 best `76.9%`; visual not merely dragging on floor |
| `box023_p2` leg interference | `<=5%` |
| `box023_p2` visual stability | no fallover in f100/f125/f160/f204 |
| model/config parity | `contact_guidance=false`, `object_action_dims=0`, `nu=29`, `nq_obj=7` |

## Decision Rule

If virtual partner support improves `box025` and keeps `box023` stable, continue with partner-force scheduling and support-point modeling. If it still fails, move to a more explicit dual-agent/contact-constraint formulation instead of tuning passive physics or single-robot reward weights.
