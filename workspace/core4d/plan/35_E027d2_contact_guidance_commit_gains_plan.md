# E027d2: Contact Guidance — Commit Phase Gain Restore

## Context

E027d identified 3 root causes for object not moving with contact_guidance:
1. ✅ Object ctrl not reset to ref (fixed)
2. ✅ noise_scale=0 (by design — correct)
3. ❓ Object still doesn't move despite fixes

## New Root Cause Found

After tracing the full CEM pipeline:

1. `optimize()` runs 32 iterations with **decaying gains**: `kp * decay^i`
2. Last iteration (`i=31`) sets `kp=0, kd=0` because `residual_gain_ratio=0.0`
3. `load_env_params` in rollout's reset phase doesn't restore gains (no kp/kd keys in init_env_param)
4. **Commit phase** calls `step_env` with ctrl containing ref positions, but gains remain at 0
5. PD actuator formula: `force = kp * (target - current) - kd * velocity` → with kp=0, force=0
6. Object never moves during commit

## Claims

1. Setting `residual_gain_ratio > 0` (or restoring gains before commit) will make the object move during the commit phase
2. With proper gains during commit, object will track ref trajectory
3. Robot body tracking should remain stable (CEM still optimizes robot channels)

## Solution

**Option A** (simplest): Set `residual_gain_ratio: 1.0` in config → last CEM iteration keeps full gains → gains remain active during commit.

**Option B**: After `optimize()` returns, explicitly set full gains before the commit loop.

**Choosing Option A** — it's a single config change. The contact_guidance design for OMOMO works because the robot physically holds the object. For CORE4D (no grasp), we need the actuator to keep driving the object.

But wait — if last iteration has full gains, the CEM optimization in that iteration evaluates "what happens when object is fully actuated" which is exactly what happens during commit. This makes the evaluation honest.

Actually there's a subtlety: with `guidance_decay_ratio=0.8` and 32 iterations, by iteration 31 the decay is `0.8^31 = 0.001` → effectively zero anyway. The issue is that the CEM uses decaying gains to gradually transfer from "object PD helps" to "robot alone carries". But for CORE4D, we want the actuator to ALWAYS help (like E027b's object_pd_override).

**Better solution**: Set `guidance_decay_ratio: 1.0` (no decay) so gains stay constant through all CEM iterations AND during commit. This turns contact_guidance into a persistent object PD driver — essentially E027b but through the contact_guidance mechanism.

## Changes

| File | Change |
|------|--------|
| `examples/config/override/core4d_e027d.yaml` | `guidance_decay_ratio: 1.0`, higher gains (kp=500, kd=500) |
| `examples/run_mjwp.py` | After optimize(), restore gains before commit loop |

## Verification

1. Run desk005: object should move > 0.5m along ref
2. Check robot stability (pelvis > 0.65m)
3. Compare with E027b results (pos_err ~0.10, rot_err ~8.8°)

## Risk

- High constant gains might make CEM optimization meaningless (object moves regardless of robot)
- This is essentially equivalent to E027b's object_pd_override — may not add value beyond that
- But validates the contact_guidance code path works correctly
