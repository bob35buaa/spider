# E092 Route Decision Tree

```text
Stage A spider_dyn smoke (3 cases, local + remote GPU0/GPU1)
  ├─ C1 FAIL: pelvis 0.074m
  ├─ C2 FAIL: pelvis 0.191m
  └─ C3 FAIL: pelvis 0.186m + RH floor 39.0%
      ↓
  No Stage A full, because no smoke PASS / REVIEW+.
      ↓
  No Stage B rl_from_spider, because no Stage A full WORK sequence exists.

Stage C rl_from_omni smoke (3 cases, local + remote GPU0/GPU1)
  ├─ C1 FAIL: pelvis 0.073m
  ├─ C2 FAIL: pelvis 0.135m
  └─ C3 FAIL: pelvis 0.189m + RH floor 40.2%
      ↓
  No Stage C main, because all smoke rollouts fail the pelvis and visual gates.

Next branch:
  Open a focused pelvis/upright/floor/contact-constraint experiment before
  spending full/main compute or expanding more medium-box data.
```
