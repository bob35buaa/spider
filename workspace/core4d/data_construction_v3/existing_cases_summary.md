# Existing Cases Seed Summary

本文件由 `build_existing_cases_seed.py` 从可信历史结果生成。旧污染 scene 的 Box021/Box026 结果不纳入；E105 ref-FK 与 E106 重复的 Box026 case 由 E106 覆盖。

- rows: `57`

## Counts

### object

| value | count |
|---|---:|
| `box004` | 4 |
| `box021` | 15 |
| `box023` | 2 |
| `box025` | 2 |
| `box026` | 34 |

### target_variant

| value | count |
|---|---:|
| `ref_fk` | 52 |
| `adaptive` | 3 |
| `fingertip_aware` | 2 |

### current_decision

| value | count |
|---|---:|
| `VISUAL_QC_PASS` | 44 |
| `REJECT_OMNIRETARGET` | 4 |
| `TARGET_GATE_PASS` | 9 |

### cem_status

| value | count |
|---|---:|
| `pass` | 10 |
| `fail` | 34 |
| `not_run` | 13 |

### downstream_decision

| value | count |
|---|---:|
| `DOWNSTREAM_CEM_PASS` | 10 |
| `DOWNSTREAM_MOTION_BINDING_FAIL` | 29 |
| `` | 13 |
| `DOWNSTREAM_CEM_FAIL` | 2 |
| `DOWNSTREAM_POSTURE_FAIL` | 3 |

## Sources

| source | status |
|---|---|
| `E092_box004_ref_fk` | `included_box004_only` |
| `E094_box004_adaptive` | `included_box004_only` |
| `E096b_box004_ref_fk` | `included` |
| `E105_box026_clean` | `included_nonduplicate_routes; duplicated ref_fk superseded by E106` |
| `E106_box026_clean_batch` | `included` |
| `E107_box021_gate_and_selected4` | `included` |
| `E079_E080_E081_box023_box025` | `included_as_legacy_cem_cache; E081 leg-object rows supersede E079/E080 person2 rows` |
| `pre_E103_box021_box026` | `excluded_invalidated_by_template_inertial_bug` |
