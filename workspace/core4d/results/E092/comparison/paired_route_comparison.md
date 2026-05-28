# E092 Paired Route Comparison

## Smoke Metrics

| case | role | spider_dyn status | spider pelvis | spider obj | spider contact | spider RH floor | rl_from_omni status | omni pelvis | omni obj | omni contact | omni RH floor | decision |
|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---|
| C1 box004 | positive seed | FAIL | `0.074m` | `0.006m` | `3.8%` | `1.0%` | FAIL | `0.073m` | `0.006m` | `3.8%` | `1.0%` | skip full/main |
| C2 Box026 039 | low-support near-pass | FAIL | `0.191m` | `0.008m` | `58.5%` | `0.0%` | FAIL | `0.135m` | `0.008m` | `57.7%` | `0.0%` | skip full/main |
| C3 Box026 135 | inside-risk near-pass | FAIL | `0.186m` | `0.006m` | `69.5%` | `39.0%` | FAIL | `0.189m` | `0.006m` | `69.5%` | `40.2%` | skip full/main |

## Route Decision

- Stage A produced no `WORK` dynamic sequence. All three smoke rollouts failed the pelvis criterion by a large margin (`0.074-0.191m` vs `>=0.55m` WORK threshold).
- Stage B is skipped because the plan requires Stage A full `WORK` output before building `rl_from_spider` inputs.
- Stage C direct OmniRetarget smoke also failed all three cases (`0.073-0.189m` vs `>=0.45m` smoke threshold).
- Visual review confirms that the failures are real posture failures: pelvis/hip collapse dominates, and C3 has a clear right-hand floor shortcut.
- Neither route shows credible improvement over the other at smoke scale. The next experiment should address pelvis/upright/contact/floor constraints before any full/main run.
