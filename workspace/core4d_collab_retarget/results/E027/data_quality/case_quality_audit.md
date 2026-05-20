# E027 Data / Retarget Quality Audit

| Case | Label | Drop P0 | Drop Denom | Failed Evidence | Primary Evidence | Counter Evidence |
|---|---|---:|---:|---:|---|---|
| box021_p1 | `usable_algorithmic_failure` | True | False | 0 | E020 root_cause=algo_stability; mask_overclaim=35.2%; ref_leg_interference=0.0% | object tracking ok; holosoma available |
| box021_p2 | `usable_algorithmic_failure` | True | False | 0 | E020 root_cause=algo_stability; mask_overclaim=22.7%; ref_leg_interference=9.3% | object tracking ok; holosoma available |
| box023_p1 | `usable_with_caveat` | False | False | 1 | E020 root_cause=contact_mask; mask_overclaim=54.4%; ref_leg_interference=0.0% | object tracking ok; no fall; holosoma available |
| box023_p2 | `usable_with_caveat` | False | False | 1 | E020 root_cause=algo_contact; mask_overclaim=53.7%; ref_leg_interference=0.0% | object tracking ok; no fall; holosoma available |
| box025_p1 | `retarget_questionable` | False | False | 1 | E020 root_cause=retarget_kinematic; mask_overclaim=43.1%; ref_leg_interference=66.5% | object tracking ok; no fall; holosoma available |
| box025_p2 | `usable_algorithmic_failure` | False | False | 0 | E020 root_cause=pass; mask_overclaim=37.9%; ref_leg_interference=23.0% | object tracking ok; no fall; strict pass guard; holosoma available |
| bucket001_p1 | `usable_with_caveat` | True | False | 1 | E020 root_cause=algo_stability; mask_overclaim=71.5%; ref_leg_interference=0.0% | object tracking ok; holosoma available |
| bucket001_p2 | `usable_with_caveat` | False | False | 1 | E020 root_cause=algo_stability; mask_overclaim=63.5%; ref_leg_interference=0.0% | object tracking ok; no fall; holosoma available |
| bucket005_s2_p1 | `usable_algorithmic_failure` | False | False | 0 | E020 root_cause=algo_contact; mask_overclaim=36.8%; ref_leg_interference=30.4% | object tracking ok; no fall; holosoma available |
| bucket005_s2_p2 | `usable_algorithmic_failure` | False | False | 0 | E020 root_cause=algo_contact; mask_overclaim=38.5%; ref_leg_interference=5.1% | object tracking ok; no fall; holosoma available |
| bucket007_p1 | `usable_algorithmic_failure` | False | False | 0 | E020 root_cause=algo_contact; mask_overclaim=44.2%; ref_leg_interference=14.0% | object tracking ok; no fall; holosoma available |
| bucket007_p2 | `retarget_questionable` | False | False | 1 | E020 root_cause=retarget_kinematic; mask_overclaim=26.3%; ref_leg_interference=66.3% | object tracking ok; no fall; holosoma available |
| desk021_p1 | `discard_from_success_denominator` | True | True | 3 | E020 root_cause=raw_data; mask_overclaim=49.6%; ref_leg_interference=4.9%; holosoma_missing | object tracking ok; no fall |

## Label Counts

- `discard_from_success_denominator`: 1
- `retarget_questionable`: 2
- `usable_algorithmic_failure`: 6
- `usable_with_caveat`: 4

All cases remain in the P1/caveat table. `discard_from_success_denominator` only affects the main optimization denominator.