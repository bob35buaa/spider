# E142 OmniRetarget Contact Exceedance Audit Results

## Summary

E142 answered the corrected question:

Do the existing contact-aware retarget/CEM candidates from E112/E113 exceed OmniRetarget on same-case hand-object contact?

Result: no, under the primary physics-contact metric.

Primary metric:

`candidate hand_object_contact_physics_frac > E110 OmniRetarget hand_object_physics_contact_frac`

## Result Paths

| artifact | path |
|---|---|
| all candidate rows | `workspace/core4d/results/E142/omniretarget_contact_exceedance_audit/e142_contact_exceedance_rows.tsv` |
| best by case | `workspace/core4d/results/E142/omniretarget_contact_exceedance_audit/e142_best_by_case.tsv` |
| summary JSON | `workspace/core4d/results/E142/omniretarget_contact_exceedance_audit/e142_contact_exceedance_summary.json` |
| summary MD | `workspace/core4d/results/E142/omniretarget_contact_exceedance_audit/e142_contact_exceedance_summary.md` |

## Key Counts

| metric | value |
|---|---:|
| candidate rows | 12 |
| joined to E110 OmniRetarget | 12 |
| best case rows | 7 |
| candidate rows exceeding OmniRetarget physics contact | 0 |
| best cases exceeding OmniRetarget physics contact | 0 |
| best cases exceeding OmniRetarget with lower-body guard | 0 |

No CEM, RL, remote job, or Holosoma command was launched.

## Best By Case

| case | best candidate | candidate physics | Omni physics | delta vs Omni | delta vs E110 Spider | lower-body | decision |
|---|---|---:|---:|---:|---:|---|---|
| `e091_box004_20231003_2_082_p1` | E112 `raw_mask_ref_fk` | 0.467890 | 0.477064 | -0.009174 | +0.091743 | WORK | `contact_not_exceed_omni` |
| `e091_box004_20231003_2_083_p1` | E113 `hold_band` | 0.352941 | 0.441176 | -0.088235 | +0.029412 | FAIL | `contact_not_exceed_omni` |
| `e091_box004_20231003_2_083_p2` | E113 `hold_band` | 0.523810 | 0.676190 | -0.152381 | +0.066667 | WORK | `contact_not_exceed_omni` |
| `d003_box021_20231011_035_p1` | E112 `raw_mask_ref_fk` | 0.751938 | 0.790698 | -0.038760 | +0.046512 | WORK | `contact_not_exceed_omni` |
| `d003_box021_20231011_035_p2` | E113 `hold_band` | 0.736842 | 0.751880 | -0.015038 | +0.060150 | FAIL | `contact_not_exceed_omni` |
| `d003_box021_20231018_029_p2` | E113 `hold_band` | 0.653333 | 0.706667 | -0.053333 | +0.200000 | FAIL | `contact_not_exceed_omni` |
| `e091_box026_20231020_135_p1` | E112 `raw_mask_ref_fk` | 0.341463 | 0.597561 | -0.256098 | -0.134146 | FAIL | `contact_not_exceed_omni` |

## Interpretation

E112/E113 were useful because they improved contact relative to the Spider/E110 or experiment-local baselines. However, they did not yet beat OmniRetarget on physics hand-object contact.

The closest cases are:

- `box004_082_p1`: only -0.9pp below OmniRetarget, lower-body guard pass.
- `box021_035_p2`: -1.5pp below OmniRetarget, but lower-body guard fail.
- `box021_035_p1`: -3.9pp below OmniRetarget, lower-body guard pass.

The main diagnostic `box021_029_p2` improves strongly over E110 Spider (+20.0pp), but remains -5.3pp below OmniRetarget and has lower-body failure.

## Claims Verification

| Claim | Result |
|---|---|
| C1: E110 OmniRetarget rows join to E112/E113 candidates | pass, 12/12 joined |
| C2: compute per-candidate contact deltas vs OmniRetarget | pass |
| C3: identify cases already exceeding OmniRetarget | pass, 0 cases |
| C4: no CEM/RL/Holosoma launch | pass |

## Verification

```bash
python -m py_compile workspace/core4d/scripts/E142/audit_omniretarget_contact_exceedance.py
bash -n workspace/core4d/scripts/eval/eval_E142_omniretarget_contact_exceedance_audit.sh
bash workspace/core4d/scripts/eval/eval_E142_omniretarget_contact_exceedance_audit.sh
```

## Next Step

Continue retarget/contact-aware CEM work, not Holosoma/RL. The immediate target should be to push the closest cases over OmniRetarget physics contact first:

- `box004_082_p1` needs roughly +1pp physics contact without losing lower-body guard.
- `box021_035_p1` needs roughly +4pp with lower-body guard preserved.
- `box021_035_p2` and `box021_029_p2` need contact improvement plus lower-body guard repair.
