# E004 Results: true-freejoint virtual partner support sweep

日期：2026-05-17

## Status

E004 Wave A/B completed. Result: COM-level virtual partner support does not recover true-freejoint collaborative transport. All 9 evaluated variants kept `scene.xml` freejoint parity (`contact_guidance=false`, `nu=29`, `nq_obj=7`, no object actuators), but no main `box025_p2` variant reached useful proxy.

## Commands

```bash
bash workspace/core4d_collab_retarget/scripts/run_E004_preprocess.sh
bash workspace/core4d_collab_retarget/scripts/train/train_E004.sh smoke 0
bash workspace/core4d_collab_retarget/scripts/run_E004_remote.sh
bash workspace/core4d_collab_retarget/scripts/train/train_E004.sh local_wave 0
bash workspace/core4d_collab_retarget/scripts/train/train_E004.sh one 0 E004_box023_p2_s20
bash workspace/core4d_collab_retarget/scripts/train/train_E004.sh one 0 E004_box023_p2_s10_hc
bash workspace/core4d_collab_retarget/scripts/pull_E004_remote_results.sh
```

Remote GPU1 hung twice near the end of `box023` runs (`box023_s20` at `250/272`, `box023_s10_hc` at `194/272`) with zero GPU utilization and no trajectory written. Both were terminated and rerun locally. Remote GPU0 completed all `box025` variants.

## Setup / Smoke

Nine Wave A/B variants were generated. Wave C rotation probes remain commented out and were not run.

Smoke passed for all 9 variants with `max_sim_steps=4`; eval showed `E004_freejoint_parity_ok=True` for all variants.

## Full Metrics

| Variant | Role | Wave | kp | hold | obj mean/max | hand contact | floor contact | leg intf | bottom mean | pelvis min | Proxy |
|---------|------|------|----|------|--------------|--------------|---------------|----------|-------------|------------|-------|
| `E004_box025_p2_g05` | control | A | 0 | 0 | `0.660/1.288m` | `77.5%` | `71.1%` | `0.0%` | `-0.068m` | `0.770m` | control failed |
| `E004_box025_p2_s10` | main | A | 10 | 0 | `0.770/1.566m` | `91.9%` | `83.8%` | `9.2%` | `-0.074m` | `0.778m` | useful False |
| `E004_box025_p2_s20` | main | A | 20 | 0 | `0.771/1.577m` | `90.2%` | `82.1%` | `6.9%` | `-0.075m` | `0.775m` | useful False |
| `E004_box025_p2_s40` | main | A | 40 | 0 | `0.769/1.551m` | `90.2%` | `82.1%` | `0.6%` | `-0.077m` | `0.772m` | useful False |
| `E004_box025_p2_s20_hc` | main | B | 20 | 1 | `0.770/1.552m` | `89.0%` | `79.2%` | `10.4%` | `-0.076m` | `0.775m` | useful False |
| `E004_box025_p2_s40_hc` | main | B | 40 | 1 | `0.768/1.549m` | `91.9%` | `68.2%` | `15.6%` | `-0.068m` | `0.779m` | useful False |
| `E004_box023_p2_s10` | guard | A | 10 | 0 | `0.814/1.494m` | `65.3%` | `78.7%` | `0.0%` | `0.024m` | `0.687m` | guard stable True |
| `E004_box023_p2_s20` | guard | A | 20 | 0 | `0.806/1.505m` | `60.7%` | `70.0%` | `0.0%` | `0.040m` | `0.684m` | guard stable True |
| `E004_box023_p2_s10_hc` | guard | B | 10 | 1 | `0.812/1.471m` | `67.3%` | `62.7%` | `10.7%` | `0.018m` | `0.040m` | guard stable False |

Aggregate:

```json
{
  "num_results": 9,
  "num_freejoint_parity_ok": 9,
  "num_main_useful_proxy": 0,
  "num_main_strong_proxy": 0,
  "num_guard_stable_proxy": 2
}
```

## Visual Observations

Keyframe sheets:

- `workspace/core4d_collab_retarget/results/E004/contact_sheet_main.jpg`
- `workspace/core4d_collab_retarget/results/E004/contact_sheet_guard.jpg`

Main `box025` observations:

- Gravity-only (`g05`) does not create transport; the simulated object still lags far behind ref and remains effectively floor-supported.
- Translation springs (`s10/s20/s40`) keep the robot near the box with high hand contact, but the box stays in the same qualitative failure mode: it is dragged/pushed around the floor rather than carried along the ref trajectory.
- Hold-contact (`s20_hc/s40_hc`) improves neither object tracking nor carry semantics. `s40_hc` reduces floor-contact metric, but increases leg interference to `15.6%`, so it is not a clean improvement.

Guard `box023` observations:

- `s10/s20` remain upright and avoid leg interference, but object tracking remains poor (`~0.81m` mean). The object is set down/left behind rather than transported.
- `s10_hc` is visually invalid: the robot falls by f204, matching `post2_pelvis_z_min=0.040m` and `guard stable False`.

## Claims

| Claim | Result |
|-------|--------|
| C1 virtual translational support is the missing factor | Rejected for current COM-force implementation. No main variant beat E003 best (`0.400/0.795m`); all main springs stayed around `0.77/1.55m`. |
| C2 gravity-only is insufficient | Supported. `g05` failed (`0.660/1.288m`) and remained floor-supported. |
| C3 useful translational spring range is `kp=10-40` | Rejected. `kp=10/20/40` were all similarly bad; higher kp mainly changed artifacts, not transport. |
| C4 virtual partner must not do the task alone | Supported as a diagnostic requirement. Here it also did not solve the task; high hand contact did not imply causal carrying. |
| C5 rotation torque should not be mainline | Supported. Position failed badly, so Wave C rotation probes were not justified; E030-style torque sweeps should not be repeated here. |

## Interpretation

E004 indicates that applying a virtual partner force at object COM is too weak/ill-posed for CORE4D collaborative transport. It can add support, but it does not create the missing interaction geometry: the robot still lacks a stable grasp/support relation that can transmit the object trajectory.

The main failure is not hand proximity. Several failed main variants have `89-92%` hand contact. The missing piece is force/contact semantics: where the partner supports, where the robot should support, and how object orientation/support points couple to the hands.

## Next

Do not run Wave C rotation probes for E004. The position/support objective is already failed, and prior E030 showed torque can create NaN/positive feedback.

Next experiment should move away from COM xfrc and toward one of:

1. explicit support-point forces/sites on the partner side of the object, not COM-only;
2. dual-agent/proxy partner with contact/connect constraints;
3. equality/contact constraint formulation that gives the object a physically meaningful second support point while keeping the RL target single-robot + human partner.

E005 should prioritize explicit support geometry or dual-agent proxy rather than further tuning `partner_force_spring_kp`.
