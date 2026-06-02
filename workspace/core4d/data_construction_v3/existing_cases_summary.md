# Existing Cases Seed Summary

本文件由 `build_existing_cases_seed.py` 从可信历史结果生成。旧污染 scene 的 Box021/Box026 结果不纳入；E105 ref-FK 与 E106 重复的 Box026 case 由 E106 覆盖。

- rows: `1931`

## Counts

### object

| value | count |
|---|---:|
| `board005` | 14 |
| `board007` | 92 |
| `board020` | 126 |
| `board021` | 16 |
| `box001` | 102 |
| `box004` | 24 |
| `box020` | 18 |
| `box021` | 65 |
| `box022` | 8 |
| `box023` | 48 |
| `box024` | 46 |
| `box025` | 44 |
| `box026` | 86 |
| `bucket001` | 80 |
| `bucket003` | 38 |
| `bucket004` | 39 |
| `bucket005` | 8 |
| `bucket006` | 34 |
| `bucket007` | 141 |
| `bucket008` | 34 |
| `bucket009` | 17 |
| `bucket010` | 37 |
| `chair005` | 24 |
| `chair006` | 60 |
| `chair020` | 24 |
| `chair021` | 118 |
| `chair022` | 66 |
| `desk001` | 48 |
| `desk005` | 18 |
| `desk007` | 74 |
| `desk020` | 36 |
| `desk021` | 120 |
| `desk023` | 112 |
| `stick001` | 18 |
| `stick003` | 36 |
| `stick006` | 38 |
| `stick008` | 22 |

### target_variant

| value | count |
|---|---:|
| `ref_fk` | 1926 |
| `adaptive` | 3 |
| `fingertip_aware` | 2 |

### current_decision

| value | count |
|---|---:|
| `REJECT_RAW_INVENTORY` | 1154 |
| `INVENTORY_READY` | 219 |
| `REJECT_TEMPLATE_BACKLOG` | 338 |
| `VISUAL_QC_PASS` | 47 |
| `REJECT_OMNIRETARGET` | 4 |
| `TARGET_GATE_PASS` | 9 |
| `REJECT_TEMPLATE_AUDIT` | 150 |
| `RAW_CONTACT_READY` | 4 |
| `REJECT_VISUAL_QC` | 1 |
| `REJECT_RAW_CONTACT` | 5 |

### cem_status

| value | count |
|---|---:|
| `not_run` | 1884 |
| `pass` | 12 |
| `fail` | 35 |

### downstream_decision

| value | count |
|---|---:|
| `` | 1884 |
| `DOWNSTREAM_CEM_PASS` | 11 |
| `DOWNSTREAM_MOTION_BINDING_FAIL` | 29 |
| `DOWNSTREAM_CEM_FAIL` | 3 |
| `DOWNSTREAM_POSTURE_FAIL` | 3 |
| `DOWNSTREAM_RL_PASS` | 1 |

## Sources

| source | status |
|---|---|
| `E092_box004_ref_fk` | `included_box004_only` |
| `E094_box004_adaptive` | `included_box004_only` |
| `E096b_box004_ref_fk` | `included` |
| `E105_box026_clean` | `included_nonduplicate_routes; duplicated ref_fk superseded by E106` |
| `E106_box026_clean_batch` | `included` |
| `E107_box021_gate_and_selected4` | `included` |
| `E108_nonbox_candidate_mining` | `included_full_candidate_state_cache; final bucket004 registry supersedes matching rows` |
| `E108_nonbox_bucket004` | `included_final_registry; 1 RL smoke pass, 1 CEM pass, 1 CEM fail, 1 visual reject` |
| `E079_E080_E081_box023_box025` | `included_as_legacy_cem_cache; E081 leg-object rows supersede E079/E080 person2 rows` |
| `pre_E103_box021_box026` | `excluded_invalidated_by_template_inertial_bug` |
