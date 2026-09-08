# CORE4D-G1 Retargeting: Four-Method Comparison

Comparison of four humanoid+object retargeting methods on the **same 52 sequences spanning 11 objects** (Unitree G1). All aggregates are the unweighted mean over cases, recomputed from per-case data so the methodology is identical across methods.

| Method | Type | Physics rollout | Tracking metrics |
|---|---|---|---|
| **SPIDER-CEM** (ours) | physics-based retargeting | yes | yes |
| SBTO | sampling-based trajectory optimization | yes | yes |
| OmniRetarget | kinematic retarget | no | no (contact only) |
| GMR | general motion retargeting (kinematic) | no | no (distinct stability set) |

## 1. Overall — shared contact-fidelity metrics (all four methods)

| Metric | Ours | SBTO | OmniRt | GMR |
|---|---|---|---|---|
| Raw contact *(diagnostic — see §1.1)* ↑ | 0.824 | 0.807 | 0.853 | 0.047 |
| Phys. contact@3mm ↑ | **0.588** | 0.172 | 0.101 | 0.001 |
| Phys. pen@3mm ↓ | 0.176 | 0.479 | 0.536 | **0.046** |
| Geom. pen@2mm ↓ | **0.052** | 0.427 | 0.381 | 0.059 |
| **Clean-contact ratio** (contact@3mm / raw) ↑ | **0.713** | 0.213 | 0.118 | 0.025 |

Values are per-case means (std / worst-case in `paper_comparison.xlsx`). SPIDER-CEM leads on physically-realised contact and both penetration metrics by a wide margin.

### 1.1 Why "raw contact" is misleading (and what to report instead)

`raw contact` (`hand_object_physics_contact_in_mask_frac`) counts a reference-contact frame as "in contact" whenever MuJoCo reports **any** hand–object contact — **including deep interpenetration**. In `core_metrics.py` the per-frame flag is simply `hand_contact = (any hand–object contact exists)` (line ~1294), with **no penetration-depth gate**. `Phys. contact@3mm` uses the same contacts but additionally requires the minimum contact distance `≥ −3 mm` (line ~1302), i.e. a *clean surface touch* rather than the hand stabbed through the object.

Consequence: a method that shoves the hand through the object (a kinematic retarget with no collision resolution) racks up `raw contact` on nearly every frame, while a physically-correct method that keeps a clean surface touch scores marginally lower on `raw contact` but far higher on every penetration-aware metric. This is exactly the pattern in the table:

- **OmniRetarget** has the *highest* raw contact (0.853) yet the *worst* penetration (phys pen@3mm 0.536, geom pen@2mm 0.381) and only 0.101 clean contact.
- **SPIDER-CEM** has slightly lower raw contact (0.824) but **5.8× more clean contact** (0.588) and **3–7× less penetration**.
- The **clean-contact ratio** row above (clean contact / raw contact) exposes this directly: 0.71 for ours vs 0.21 / 0.12 / 0.02 — i.e. ~71% of our contacts are clean touches, vs ≤21% for every baseline.

> **Symmetric caveat — read penetration *jointly* with contact.** GMR appears to "win" the two penetration columns, but only because it barely touches the object at all (clean contact 0.001). Near-zero penetration is trivial for a method that never makes contact. The fair reading: among methods that actually establish the reference contact (ours, SBTO, OmniRetarget), **SPIDER-CEM has both the highest clean contact and the lowest penetration**. This is why contact and penetration must be reported together — neither is meaningful alone.

**Recommendation (adopted in the LaTeX table):** drop `raw contact` from the headline and report the penetration-aware trio — **Phys. contact@3mm ↑, Phys. pen@3mm ↓, Geom. pen@2mm ↓** — optionally with the clean-contact ratio. `raw contact` is retained only as a labelled diagnostic (it is *evidence* that the baselines' contact is illusory, not a quality metric). All four methods have the penetration-aware metrics, so the comparison stays fully 4-way.

## 2. Overall — tracking error (physics methods only)

| Metric | Ours (SPIDER-CEM) | SBTO |
|---|---|---|
| Root pos err (cm) ↓ | 13.69 | **6.42** |
| Root ori err (deg) ↓ | **9.12** | 13.60 |
| EEF pos err (cm) ↓ | 13.20 | **8.19** |
| EEF ori err (deg) ↓ | **17.39** | 20.40 |
| Obj pos err (cm) ↓ | 10.36 | **6.95** |
| Obj ori err (deg) ↓ | **5.13** | 8.84 |

> SBTO optimizes tracking directly and wins on most kinematic-tracking terms, but at the cost of far worse contact fidelity (Table 1) and larger jerk/foot-slip. SPIDER-CEM trades a little tracking error for markedly better physical contact and penetration.

## 3. By-object — shared contact metrics

### Raw contact (↑ higher better)

| Object | n | Ours | SBTO | OmniRt | GMR |
|---|---|---|---|---|---|
| box001 | 6 | 0.835 | 0.707 | **0.879** | 0.009 |
| box004 | 3 | 0.684 | 0.679 | **0.958** | 0.000 |
| box021 | 11 | 0.785 | 0.702 | **0.906** | 0.092 |
| box023 | 7 | 0.757 | 0.704 | **0.836** | 0.013 |
| box024 | 4 | 0.865 | 0.890 | **1.000** | 0.222 |
| bucket003 | 3 | **0.875** | 0.861 | 0.601 | 0.021 |
| bucket007 | 4 | 0.819 | **0.937** | 0.929 | 0.048 |
| chair006 | 4 | 0.925 | **0.980** | 0.803 | 0.000 |
| desk007 | 3 | 0.862 | 0.879 | **1.000** | 0.000 |
| desk021 | 4 | 0.857 | **0.941** | 0.636 | 0.010 |
| desk023 | 3 | 0.921 | **0.941** | 0.705 | 0.033 |
| **Overall** | 52 | 0.824 | 0.807 | **0.853** | 0.047 |

### Phys. contact@3mm (↑ higher better)

| Object | n | Ours | SBTO | OmniRt | GMR |
|---|---|---|---|---|---|
| box001 | 6 | **0.612** | 0.208 | 0.015 | 0.000 |
| box004 | 3 | **0.486** | 0.005 | 0.064 | 0.000 |
| box021 | 11 | **0.552** | 0.145 | 0.149 | 0.000 |
| box023 | 7 | **0.583** | 0.194 | 0.281 | 0.000 |
| box024 | 4 | **0.510** | 0.289 | 0.000 | 0.006 |
| bucket003 | 3 | **0.564** | 0.259 | 0.009 | 0.002 |
| bucket007 | 4 | **0.492** | 0.156 | 0.062 | 0.000 |
| chair006 | 4 | **0.711** | 0.145 | 0.038 | 0.000 |
| desk007 | 3 | **0.735** | 0.263 | 0.014 | 0.000 |
| desk021 | 4 | **0.619** | 0.100 | 0.149 | 0.004 |
| desk023 | 3 | **0.685** | 0.136 | 0.101 | 0.005 |
| **Overall** | 52 | **0.588** | 0.172 | 0.101 | 0.001 |

### Phys. pen@3mm (↓ lower better)

| Object | n | Ours | SBTO | OmniRt | GMR |
|---|---|---|---|---|---|
| box001 | 6 | 0.168 | 0.366 | 0.607 | **0.009** |
| box004 | 3 | 0.123 | 0.448 | 0.571 | **0.000** |
| box021 | 11 | 0.180 | 0.454 | 0.579 | **0.092** |
| box023 | 7 | 0.110 | 0.327 | 0.337 | **0.013** |
| box024 | 4 | 0.264 | 0.480 | 0.751 | **0.216** |
| bucket003 | 3 | 0.250 | 0.452 | 0.544 | **0.019** |
| bucket007 | 4 | 0.202 | 0.552 | 0.522 | **0.048** |
| chair006 | 4 | 0.173 | 0.633 | 0.472 | **0.000** |
| desk007 | 3 | 0.154 | 0.596 | 0.842 | **0.000** |
| desk021 | 4 | 0.184 | 0.654 | 0.346 | **0.006** |
| desk023 | 3 | 0.170 | 0.553 | 0.421 | **0.028** |
| **Overall** | 52 | 0.176 | 0.479 | 0.536 | **0.046** |

### Geom. pen@2mm (↓ lower better)

| Object | n | Ours | SBTO | OmniRt | GMR |
|---|---|---|---|---|---|
| box001 | 6 | 0.038 | 0.302 | 0.507 | **0.009** |
| box004 | 3 | 0.054 | 0.448 | 0.541 | **0.000** |
| box021 | 11 | **0.062** | 0.426 | 0.225 | 0.092 |
| box023 | 7 | 0.076 | 0.327 | 0.272 | **0.013** |
| box024 | 4 | **0.037** | 0.194 | 0.526 | 0.222 |
| bucket003 | 3 | 0.038 | 0.433 | 0.378 | **0.019** |
| bucket007 | 4 | 0.055 | 0.542 | 0.514 | **0.048** |
| chair006 | 4 | **0.045** | 0.536 | 0.367 | 0.068 |
| desk007 | 3 | 0.062 | 0.670 | 0.814 | **0.000** |
| desk021 | 4 | **0.026** | 0.553 | 0.180 | 0.102 |
| desk023 | 3 | 0.044 | 0.489 | 0.285 | **0.040** |
| **Overall** | 52 | **0.052** | 0.427 | 0.381 | 0.059 |

## 4. By-object — tracking error (SPIDER-CEM vs SBTO)

| Object | n | Root cm (C/S) | Root deg (C/S) | EEF cm (C/S) | EEF deg (C/S) | Obj cm (C/S) | Obj deg (C/S) |
|---|---|---|---|---|---|---|---|
| box001 | 6 | 11.7/**4.8** | **6.1**/10.4 | 11.8/**7.0** | **15.9**/19.2 | 9.6/**5.9** | **4.9**/8.9 |
| box004 | 3 | 15.9/**9.1** | **13.5**/17.9 | 14.4/**11.6** | **21.6**/25.0 | **10.7**/11.8 | **10.7**/35.1 |
| box021 | 11 | 15.4/**8.2** | **12.4**/15.8 | 15.3/**10.0** | **18.9**/22.0 | 13.1/**9.8** | **5.5**/11.2 |
| box023 | 7 | 13.5/**5.9** | **11.8**/13.5 | 13.5/**7.1** | **18.8**/20.6 | 10.9/**6.0** | **4.8**/5.3 |
| box024 | 4 | 13.8/**5.9** | **7.1**/12.6 | 13.4/**8.2** | **17.1**/19.5 | 10.8/**6.3** | **4.1**/4.8 |
| bucket003 | 3 | 13.9/**6.6** | **7.0**/15.7 | 13.3/**7.2** | **17.2**/19.3 | 9.4/**5.1** | **4.8**/5.1 |
| bucket007 | 4 | 14.1/**5.7** | **7.7**/11.9 | 12.9/**7.2** | 17.8/**17.8** | 7.0/**4.3** | **4.5**/6.9 |
| chair006 | 4 | 9.4/**3.8** | **4.8**/10.5 | 9.2/**6.7** | **11.1**/17.0 | 9.0/**7.4** | **3.3**/5.2 |
| desk007 | 3 | 10.5/**4.7** | **5.8**/10.7 | 9.5/**7.2** | **13.8**/21.2 | 6.3/**4.3** | **3.5**/5.4 |
| desk021 | 4 | 15.4/**6.5** | **9.6**/13.2 | 14.4/**8.0** | **17.9**/19.9 | 11.2/**6.0** | **6.2**/6.3 |
| desk023 | 3 | 15.6/**8.1** | **7.8**/16.9 | 13.8/**9.0** | **18.4**/22.3 | 9.6/**5.5** | **4.7**/5.5 |
| **Overall** | 52 | 13.7/**6.4** | **9.1**/13.6 | 13.2/**8.2** | **17.4**/20.4 | 10.4/**7.0** | **5.1**/8.8 |

(C = SPIDER-CEM, S = SBTO; bold = better.)

## 5. Method-specific stability metrics (heterogeneous — not cross-compared)

| Method | Metric | mean | worst |
|---|---|---|---|
| SPIDER-CEM | Fall | 0.00 | 0.00 |
| SPIDER-CEM | Body-z err p95 (m) | 0.09 | 0.17 |
| SPIDER-CEM | Ankle jerk p95 (m/s^3) | 653.80 | 981.83 |
| SPIDER-CEM | Obj speed max (m/s) | 1.29 | 1.95 |
| SPIDER-CEM | Foot slip max (m) | 0.87 | 2.33 |
| SBTO | Fall | 0.00 | 0.00 |
| SBTO | Body-z err p95 (m) | 0.08 | 0.15 |
| SBTO | Ankle jerk p95 (m/s^3) | 1059.42 | 1567.84 |
| SBTO | Obj speed max (m/s) | 1.83 | 3.24 |
| SBTO | Foot slip max (m) | 1.16 | 2.38 |
| GMR | Foot slip max (m) | 0.96 | 2.22 |
| GMR | Foot grnd dev (m) | 0.05 | 0.10 |
| GMR | Grounded frac | 0.957 | 0.762 |
| GMR | qpos accel p95 | 53.40 | 955.66 |
| GMR | qpos jerk p95 | 1136.38 | 20066.50 |
| GMR | Ankle jerk p95 | 170.37 | 502.69 |
| GMR | Trackbody spd max | 2.33 | 6.06 |
| GMR | Ankle spd max | 2.17 | 6.06 |

## Notes & caveats

- **Fair 4-way comparison = the four shared contact metrics** (Section 1/3). These are the only metrics computed identically for all four methods.
- **Tracking metrics** (root/eef/obj error) require a physics rollout with a tracked reference; OmniRetarget and GMR are kinematic retargets and do not produce them, so tracking is a **2-way** SPIDER-CEM vs SBTO comparison only.
- **Stability metrics are not aligned** across methods (SPIDER/SBTO report body-z / ankle-jerk / foot-slip on the physics rollout; GMR reports qpos accel/jerk, grounded-frac, foot-ground-dev on the kinematic output). They are listed per method for reference, **not** cross-compared.
- **Raw vs physical contact:** 'raw contact' measures contact against the kinematic reference mask; 'physical contact@3mm' measures contact actually realized under physics within a 3 mm band. A high raw / low physical gap (OmniRetarget, GMR) indicates contact that does not survive physics.
- All numbers are means over cases; per-case distributions and std/worst are in `paper_comparison.xlsx`.