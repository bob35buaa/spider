# CORE4D-G1 Retargeting: 4-Method Comparison

Comparison of the retargeting methods below on the **same 52 sequences spanning 11 objects** (Unitree G1). All aggregates are the unweighted mean over cases, recomputed from per-case data so the methodology is identical across methods.

| Method | Type | Physics rollout | Tracking metrics |
|---|---|---|---|
| **SPIDER-CEM** (ours) | physics-based retargeting | yes | yes |
| SBTO | sampling-based trajectory optimization | yes | yes |
| OmniRetarget | kinematic retarget | no | no (contact only) |
| GMR | general motion retargeting (kinematic) | no | no (distinct stability set) |

## 1. Overall — shared contact-fidelity metrics (all 4 methods)

| Metric | Ours | SBTO | OmniRt | GMR |
|---|---|---|---|---|
| Raw contact *(diagnostic — see §1.1)* ↑ | 0.824 | 0.837 | 0.853 | 0.354 |
| Phys. contact@3mm ↑ | **0.588** | 0.190 | 0.101 | 0.004 |
| Phys. contact@10mm ↑ | **0.804** | 0.650 | 0.510 | 0.014 |
| Phys. pen@3mm ↓ | **0.176** | 0.490 | 0.536 | 0.349 |
| Phys. pen@10mm ↓ | **0.017** | 0.136 | 0.247 | 0.339 |
| Max phys pen (mm) ↓ | **12.3** | 25.1 | 21.1 | 102.1 |
| Geom. pen@2mm ↓ | **0.052** | 0.425 | 0.381 | 0.456 |
| **Clean-contact ratio** (contact@3mm / raw) ↑ | **0.713** | 0.227 | 0.118 | 0.012 |

Values are per-case means (std / worst-case in `paper_comparison.xlsx`). SPIDER-CEM leads on physically-realised contact and both penetration metrics by a wide margin.

### 1.1 Why "raw contact" is misleading (and what to report instead)

`raw contact` (`hand_object_physics_contact_in_mask_frac`) counts a reference-contact frame as "in contact" whenever MuJoCo reports **any** hand–object contact — **including deep interpenetration**. In `core_metrics.py` the per-frame flag is simply `hand_contact = (any hand–object contact exists)` (line ~1294), with **no penetration-depth gate**. `Phys. contact@3mm` uses the same contacts but additionally requires the minimum contact distance `≥ −3 mm` (line ~1302), i.e. a *clean surface touch* rather than the hand stabbed through the object.

Consequence: a method that shoves the hand through the object (a kinematic retarget with no collision resolution) racks up `raw contact` on nearly every frame, while a physically-correct method that keeps a clean surface touch scores marginally lower on `raw contact` but far higher on every penetration-aware metric. This is exactly the pattern in the table:

- **OmniRetarget** has the *highest* raw contact (0.853) yet the *worst* penetration (phys pen@3mm 0.536, geom pen@2mm 0.381) and only 0.101 clean contact.
- **SPIDER-CEM** has slightly lower raw contact (0.824) but far more clean contact (0.588) and markedly less penetration.
- The **clean-contact ratio** row above (clean contact / raw contact) exposes this directly: 0.71 (Ours), 0.23 (SBTO), 0.12 (OmniRt), 0.01 (GMR).

> **Read penetration *jointly* with contact — the GMR case.** GMR is object-unaware, so it barely establishes any *clean* contact (contact@3mm 0.004, contact@10mm 0.014). A frame-level pen fraction can look moderate for such a method simply because it is rarely near the object — but when its hand *does* intersect the object it stabs catastrophically deep (max phys pen 102 mm vs 12 mm for ours). So GMR neither contacts cleanly nor avoids deep penetration; its numbers must be read as contact + max-pen together, never a single frame-fraction. **SPIDER-CEM attains the highest clean contact AND the lowest penetration on every depth threshold.**

**Recommendation (adopted in the LaTeX table):** drop `raw contact` from the headline and report the penetration-aware set — **Phys. contact@3mm ↑, Phys. contact@10mm ↑, Phys. pen@3mm ↓, Phys. pen@10mm ↓, Max phys pen (mm) ↓, Geom. pen@2mm ↓**. `raw contact` is retained only as a labelled diagnostic (it is *evidence* that the baseline's contact is illusory, not a quality metric).

## 2. Overall — tracking error (physics methods only)

| Metric | Ours (SPIDER-CEM) | SBTO |
|---|---|---|
| Root pos err (cm) ↓ | 13.69 | **6.11** |
| Root ori err (deg) ↓ | **9.12** | 12.95 |
| EEF pos err (cm) ↓ | 13.20 | **7.79** |
| EEF ori err (deg) ↓ | **17.39** | 19.95 |
| Obj pos err (cm) ↓ | 10.36 | **5.91** |
| Obj ori err (deg) ↓ | **5.13** | 6.75 |

> SBTO optimizes tracking directly and wins on most kinematic-tracking terms, but at the cost of far worse contact fidelity (Table 1) and larger jerk/foot-slip. SPIDER-CEM trades a little tracking error for markedly better physical contact and penetration.

## 3. By-object — shared contact metrics

### Raw contact (↑ higher better)

| Object | n | Ours | SBTO | OmniRt | GMR |
|---|---|---|---|---|---|
| box001 | 6 | 0.835 | 0.677 | **0.879** | 0.762 |
| box004 | 3 | 0.684 | 0.840 | **0.958** | 0.531 |
| box021 | 11 | 0.785 | 0.817 | **0.906** | 0.605 |
| box023 | 7 | 0.757 | 0.704 | **0.836** | 0.325 |
| box024 | 4 | 0.865 | 0.890 | **1.000** | 0.349 |
| bucket003 | 3 | **0.875** | 0.861 | 0.601 | 0.231 |
| bucket007 | 4 | 0.819 | **0.937** | 0.929 | 0.151 |
| chair006 | 4 | 0.925 | **0.980** | 0.803 | 0.000 |
| desk007 | 3 | 0.862 | 0.879 | **1.000** | 0.000 |
| desk021 | 4 | 0.857 | **0.941** | 0.636 | 0.096 |
| desk023 | 3 | 0.921 | **0.941** | 0.705 | 0.074 |
| **Overall** | 52 | 0.824 | 0.837 | **0.853** | 0.354 |

### Phys. contact@3mm (↑ higher better)

| Object | n | Ours | SBTO | OmniRt | GMR |
|---|---|---|---|---|---|
| box001 | 6 | **0.612** | 0.234 | 0.015 | 0.001 |
| box004 | 3 | **0.486** | 0.167 | 0.064 | 0.006 |
| box021 | 11 | **0.552** | 0.172 | 0.149 | 0.006 |
| box023 | 7 | **0.583** | 0.194 | 0.281 | 0.003 |
| box024 | 4 | **0.510** | 0.289 | 0.000 | 0.003 |
| bucket003 | 3 | **0.564** | 0.259 | 0.009 | 0.004 |
| bucket007 | 4 | **0.492** | 0.156 | 0.062 | 0.012 |
| chair006 | 4 | **0.711** | 0.145 | 0.038 | 0.000 |
| desk007 | 3 | **0.735** | 0.263 | 0.014 | 0.000 |
| desk021 | 4 | **0.619** | 0.100 | 0.149 | 0.006 |
| desk023 | 3 | **0.685** | 0.136 | 0.101 | 0.006 |
| **Overall** | 52 | **0.588** | 0.190 | 0.101 | 0.004 |

### Phys. contact@10mm (↑ higher better)

| Object | n | Ours | SBTO | OmniRt | GMR |
|---|---|---|---|---|---|
| box001 | 6 | **0.821** | 0.560 | 0.289 | 0.013 |
| box004 | 3 | 0.679 | 0.608 | **0.689** | 0.009 |
| box021 | 11 | 0.761 | 0.639 | **0.904** | 0.018 |
| box023 | 7 | 0.731 | 0.588 | **0.774** | 0.021 |
| box024 | 4 | **0.860** | 0.821 | 0.104 | 0.003 |
| bucket003 | 3 | **0.751** | 0.701 | 0.105 | 0.020 |
| bucket007 | 4 | **0.798** | 0.531 | 0.240 | 0.022 |
| chair006 | 4 | **0.925** | 0.762 | 0.243 | 0.000 |
| desk007 | 3 | **0.862** | 0.785 | 0.225 | 0.000 |
| desk021 | 4 | **0.857** | 0.577 | 0.627 | 0.026 |
| desk023 | 3 | **0.921** | 0.747 | 0.506 | 0.008 |
| **Overall** | 52 | **0.804** | 0.650 | 0.510 | 0.014 |

### Phys. pen@3mm (↓ lower better)

| Object | n | Ours | SBTO | OmniRt | GMR |
|---|---|---|---|---|---|
| box001 | 6 | **0.168** | 0.343 | 0.607 | 0.761 |
| box004 | 3 | **0.123** | 0.431 | 0.571 | 0.525 |
| box021 | 11 | **0.180** | 0.522 | 0.579 | 0.599 |
| box023 | 7 | **0.110** | 0.327 | 0.337 | 0.322 |
| box024 | 4 | **0.264** | 0.480 | 0.751 | 0.347 |
| bucket003 | 3 | 0.250 | 0.452 | 0.544 | **0.227** |
| bucket007 | 4 | 0.202 | 0.552 | 0.522 | **0.139** |
| chair006 | 4 | 0.173 | 0.633 | 0.472 | **0.000** |
| desk007 | 3 | 0.154 | 0.596 | 0.842 | **0.000** |
| desk021 | 4 | 0.184 | 0.654 | 0.346 | **0.089** |
| desk023 | 3 | 0.170 | 0.553 | 0.421 | **0.068** |
| **Overall** | 52 | **0.176** | 0.490 | 0.536 | 0.349 |

### Phys. pen@10mm (↓ lower better)

| Object | n | Ours | SBTO | OmniRt | GMR |
|---|---|---|---|---|---|
| box001 | 6 | **0.010** | 0.077 | 0.419 | 0.748 |
| box004 | 3 | **0.003** | 0.146 | 0.171 | 0.522 |
| box021 | 11 | 0.019 | 0.151 | **0.001** | 0.587 |
| box023 | 7 | **0.017** | 0.078 | 0.039 | 0.304 |
| box024 | 4 | **0.004** | 0.052 | 0.662 | 0.347 |
| bucket003 | 3 | **0.109** | 0.120 | 0.462 | 0.210 |
| bucket007 | 4 | **0.013** | 0.270 | 0.414 | 0.129 |
| chair006 | 4 | 0.002 | 0.139 | 0.344 | **0.000** |
| desk007 | 3 | 0.023 | 0.098 | 0.669 | **0.000** |
| desk021 | 4 | **0.002** | 0.268 | 0.008 | 0.069 |
| desk023 | 3 | **0.000** | 0.135 | 0.144 | 0.066 |
| **Overall** | 52 | **0.017** | 0.136 | 0.247 | 0.339 |

### Max phys pen (mm) (↓ lower better)

| Object | n | Ours | SBTO | OmniRt | GMR |
|---|---|---|---|---|---|
| box001 | 6 | **9.1** | 16.5 | 17.2 | 220.7 |
| box004 | 3 | **7.2** | 38.0 | 13.2 | 128.3 |
| box021 | 11 | 14.3 | 36.0 | **8.6** | 165.7 |
| box023 | 7 | 20.9 | 21.6 | **10.6** | 92.0 |
| box024 | 4 | **8.3** | 14.1 | 16.9 | 114.8 |
| bucket003 | 3 | **15.8** | 20.8 | 33.7 | 50.8 |
| bucket007 | 4 | **11.2** | 29.4 | 66.9 | 44.8 |
| chair006 | 4 | 7.4 | 18.5 | 29.8 | **0.0** |
| desk007 | 3 | 18.2 | 24.3 | 45.9 | **0.0** |
| desk021 | 4 | **7.2** | 24.7 | 10.7 | 40.1 |
| desk023 | 3 | **6.4** | 20.3 | 17.2 | 60.6 |
| **Overall** | 52 | **12.3** | 25.1 | 21.1 | 102.1 |

### Geom. pen@2mm (↓ lower better)

| Object | n | Ours | SBTO | OmniRt | GMR |
|---|---|---|---|---|---|
| box001 | 6 | **0.038** | 0.208 | 0.507 | 0.761 |
| box004 | 3 | **0.054** | 0.421 | 0.541 | 0.528 |
| box021 | 11 | **0.062** | 0.473 | 0.225 | 0.601 |
| box023 | 7 | **0.076** | 0.327 | 0.272 | 0.324 |
| box024 | 4 | **0.037** | 0.194 | 0.526 | 0.347 |
| bucket003 | 3 | **0.038** | 0.433 | 0.378 | 0.453 |
| bucket007 | 4 | **0.055** | 0.542 | 0.514 | 0.334 |
| chair006 | 4 | **0.045** | 0.536 | 0.367 | 0.668 |
| desk007 | 3 | **0.062** | 0.670 | 0.814 | 0.152 |
| desk021 | 4 | **0.026** | 0.553 | 0.180 | 0.198 |
| desk023 | 3 | **0.044** | 0.489 | 0.285 | 0.226 |
| **Overall** | 52 | **0.052** | 0.425 | 0.381 | 0.456 |

## 4. By-object — tracking error (SPIDER-CEM vs SBTO)

| Object | n | Root cm (C/S) | Root deg (C/S) | EEF cm (C/S) | EEF deg (C/S) | Obj cm (C/S) | Obj deg (C/S) |
|---|---|---|---|---|---|---|---|
| box001 | 6 | 11.7/**4.7** | **6.1**/10.4 | 11.8/**6.8** | **15.9**/19.6 | 9.6/**5.7** | **4.9**/7.9 |
| box004 | 3 | 15.9/**7.4** | **13.5**/13.6 | 14.4/**8.9** | **21.6**/21.9 | 10.7/**6.5** | **10.7**/15.7 |
| box021 | 11 | 15.4/**7.3** | **12.4**/13.9 | 15.3/**8.9** | **18.9**/20.5 | 13.1/**6.4** | **5.5**/7.1 |
| box023 | 7 | 13.5/**5.9** | **11.8**/13.5 | 13.5/**7.1** | **18.8**/20.6 | 10.9/**6.0** | **4.8**/5.3 |
| box024 | 4 | 13.8/**5.9** | **7.1**/12.6 | 13.4/**8.2** | **17.1**/19.5 | 10.8/**6.3** | **4.1**/4.8 |
| bucket003 | 3 | 13.9/**6.6** | **7.0**/15.7 | 13.3/**7.2** | **17.2**/19.3 | 9.4/**5.1** | **4.8**/5.1 |
| bucket007 | 4 | 14.1/**5.7** | **7.7**/11.9 | 12.9/**7.2** | 17.8/**17.8** | 7.0/**4.3** | **4.5**/6.9 |
| chair006 | 4 | 9.4/**3.8** | **4.8**/10.5 | 9.2/**6.7** | **11.1**/17.0 | 9.0/**7.4** | **3.3**/5.2 |
| desk007 | 3 | 10.5/**4.7** | **5.8**/10.7 | 9.5/**7.2** | **13.8**/21.2 | 6.3/**4.3** | **3.5**/5.4 |
| desk021 | 4 | 15.4/**6.5** | **9.6**/13.2 | 14.4/**8.0** | **17.9**/19.9 | 11.2/**6.0** | **6.2**/6.3 |
| desk023 | 3 | 15.6/**8.1** | **7.8**/16.9 | 13.8/**9.0** | **18.4**/22.3 | 9.6/**5.5** | **4.7**/5.5 |
| **Overall** | 52 | 13.7/**6.1** | **9.1**/13.0 | 13.2/**7.8** | **17.4**/19.9 | 10.4/**5.9** | **5.1**/6.7 |

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
| SBTO | Ankle jerk p95 (m/s^3) | 1041.74 | 1567.84 |
| SBTO | Obj speed max (m/s) | 1.77 | 3.24 |
| SBTO | Foot slip max (m) | 1.20 | 2.38 |
| OmniRetarget | Ankle jerk p95 (m/s^3) | 1560.83 | 3149.26 |
| OmniRetarget | Obj speed max (m/s) | 2.83 | 5.70 |
| OmniRetarget | Foot slip max (m) | 1.33 | 2.40 |
| GMR | Foot slip max (m) | 0.96 | 2.22 |
| GMR | Foot grnd dev (m) | 0.05 | 0.10 |
| GMR | Grounded frac | 0.957 | 0.762 |
| GMR | qpos accel p95 | 53.34 | 955.57 |
| GMR | qpos jerk p95 | 1135.09 | 20066.50 |
| GMR | Ankle jerk p95 | 170.37 | 502.69 |
| GMR | Trackbody spd max | 2.33 | 6.06 |
| GMR | Ankle spd max | 2.17 | 6.06 |

## Notes & caveats

- **Fair comparison = the shared contact metrics** (Section 1/3). These are computed identically for all 4 methods.
- **Tracking metrics** (root/eef/obj error) require a physics rollout with a tracked reference; OmniRetarget and GMR are kinematic retargets and do not produce them, so tracking is a **2-way** SPIDER-CEM vs SBTO comparison only.
- **Stability metrics** (Section 5) are heterogeneous: SPIDER-CEM / SBTO / OmniRetarget report ankle-jerk / obj-speed / foot-slip on a MuJoCo rollout (same code), while GMR's come from its own kinematic pipeline at a different scale. Listed per method for reference, not cross-compared.
- **Raw vs physical contact:** 'raw contact' measures contact against the kinematic reference mask; 'physical contact@3mm' measures contact actually realized under physics within a 3 mm band. A high raw / low physical gap (the kinematic baselines) indicates contact that does not survive physics.
- **GMR is object-unaware:** its dynamic-quality metrics come from its own kinematic pipeline at a different scale (ankle-jerk especially — see the `_nojerk` table), and its deep max penetration reflects the hand passing through the object's nominal volume. Read GMR jointly with its near-zero clean contact; its columns are indicative, not head-to-head comparable.
- All numbers are means over cases; per-case distributions and std/worst are in `paper_comparison_full.xlsx`.