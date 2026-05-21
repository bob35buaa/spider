# E030 Case Scope

E030 follows the E027 multi-evidence data protocol: algorithm failures are not data-discard evidence.

| Variant | Case | Role | Source | Quality Label | Baseline Contact | Baseline Deep Pen | Rationale |
|---|---|---|---|---|---:|---:|---|
| E030_box025_p1_tinygeom_surface_gate | `box025_p1` | `target_geometry` | `E018b:E018b_box025_p1_canonical_t02` | `retarget_questionable` | 66.07142857142857 | 5.056179775280898 | Tiny lower-body proxy plus surface/contact gate for near-pass box025 p1 |
| E030_bucket007_p2_tinygeom_surface_gate | `bucket007_p2` | `target_geometry` | `E018b:E018b_bucket007_p2_canonical_t02` | `retarget_questionable` | 29.749103942652326 | 18.421052631578945 | Tiny lower-body proxy plus surface/contact gate for retarget-questionable bucket007 p2 |
| E030_box023_p1_surface_hold_gate | `box023_p1` | `diagnostic_surface` | `E022:E022_box023_p1_raw3_eval_axis` | `usable_with_caveat` | 25.301204819277107 | 0.0 | Surface/side-control diagnostic after E027 ruled out phase shift |
| E030_box023_p2_surface_hold_gate | `box023_p2` | `diagnostic_surface` | `E018b:E018b_box023_p2_canonical_t02` | `usable_with_caveat` | 28.57142857142857 | 3.3333333333333335 | Surface/side-control diagnostic after E027 ruled out phase shift |
| E030_bucket005_s2_p1_leg_guard_surface | `bucket005_s2_p1` | `shortcut_guard` | `E018b:E018b_bucket005_s2_p1_canonical_t02` | `usable_algorithmic_failure` | 97.59358288770053 | 88.15165876777252 | Shortcut guard: high contact only counts if penetration and leg artifact are controlled |
| E030_box025_p2_guard_surface | `box025_p2` | `clean_guard` | `E018b:E018b_box025_p2_canonical_t02` | `usable_algorithmic_failure` | 86.9281045751634 | 0.0 | Clean strict-pass guard; surface controls must not collapse box025 p2 |

Excluded from E030 full variants:

- `desk021_p1`: only case with `discard_from_success_denominator=True`; retained in P1 caveat tables.
- `bucket001_p1`: E029 fixed fall but contact stayed `0%`; next step is reachability/support timing audit, not another stability sweep.
- `bucket001_p2`: retained as usable-with-caveat bucket penetration evidence; E030 uses `bucket005_s2_p1` as the shortcut guard to keep the batch bounded.
