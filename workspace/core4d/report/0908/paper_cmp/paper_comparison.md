# CORE4D-G1 Retargeting: SPIDER-CEM vs OmniRetarget

Comparison of the retargeting methods below on the **same 52 sequences spanning 11 objects** (Unitree G1). All aggregates are the unweighted mean over cases, recomputed from per-case data so the methodology is identical across methods.

| Method | Type | Physics rollout | Tracking metrics |
|---|---|---|---|
| **SPIDER-CEM** (ours) | physics-based retargeting | yes | yes |
| OmniRetarget | kinematic retarget | no | no (contact only) |

## 1. Overall — shared contact-fidelity metrics (both methods)

| Metric | Ours | OmniRt |
|---|---|---|
| Raw contact *(diagnostic — see §1.1)* ↑ | 0.824 | 0.853 |
| Phys. contact@3mm ↑ | **0.588** | 0.101 |
| Phys. contact@10mm ↑ | **0.804** | 0.510 |
| Phys. pen@3mm ↓ | **0.176** | 0.536 |
| Phys. pen@10mm ↓ | **0.017** | 0.247 |
| Max phys pen (mm) ↓ | **12.3** | 21.1 |
| Geom. pen@2mm ↓ | **0.052** | 0.381 |
| **Clean-contact ratio** (contact@3mm / raw) ↑ | **0.713** | 0.118 |

Values are per-case means (std / worst-case in `paper_comparison.xlsx`). SPIDER-CEM leads on physically-realised contact and both penetration metrics by a wide margin.

### 1.1 Why "raw contact" is misleading (and what to report instead)

`raw contact` (`hand_object_physics_contact_in_mask_frac`) counts a reference-contact frame as "in contact" whenever MuJoCo reports **any** hand–object contact — **including deep interpenetration**. In `core_metrics.py` the per-frame flag is simply `hand_contact = (any hand–object contact exists)` (line ~1294), with **no penetration-depth gate**. `Phys. contact@3mm` uses the same contacts but additionally requires the minimum contact distance `≥ −3 mm` (line ~1302), i.e. a *clean surface touch* rather than the hand stabbed through the object.

Consequence: a method that shoves the hand through the object (a kinematic retarget with no collision resolution) racks up `raw contact` on nearly every frame, while a physically-correct method that keeps a clean surface touch scores marginally lower on `raw contact` but far higher on every penetration-aware metric. This is exactly the pattern in the table:

- **OmniRetarget** has the *highest* raw contact (0.853) yet the *worst* penetration (phys pen@3mm 0.536, geom pen@2mm 0.381) and only 0.101 clean contact.
- **SPIDER-CEM** has slightly lower raw contact (0.824) but far more clean contact (0.588) and markedly less penetration.
- The **clean-contact ratio** row above (clean contact / raw contact) exposes this directly: 0.71 (Ours), 0.12 (OmniRt).

> **Read penetration *jointly* with contact.** A low penetration number is only meaningful if the method actually establishes the reference contact: SPIDER-CEM attains both the higher clean contact *and* the lower penetration than the kinematic OmniRetarget baseline.

**Recommendation (adopted in the LaTeX table):** drop `raw contact` from the headline and report the penetration-aware set — **Phys. contact@3mm ↑, Phys. contact@10mm ↑, Phys. pen@3mm ↓, Phys. pen@10mm ↓, Max phys pen (mm) ↓, Geom. pen@2mm ↓**. `raw contact` is retained only as a labelled diagnostic (it is *evidence* that the baseline's contact is illusory, not a quality metric).

## 3. By-object — shared contact metrics

### Raw contact (↑ higher better)

| Object | n | Ours | OmniRt |
|---|---|---|---|
| box001 | 6 | 0.835 | **0.879** |
| box004 | 3 | 0.684 | **0.958** |
| box021 | 11 | 0.785 | **0.906** |
| box023 | 7 | 0.757 | **0.836** |
| box024 | 4 | 0.865 | **1.000** |
| bucket003 | 3 | **0.875** | 0.601 |
| bucket007 | 4 | 0.819 | **0.929** |
| chair006 | 4 | **0.925** | 0.803 |
| desk007 | 3 | 0.862 | **1.000** |
| desk021 | 4 | **0.857** | 0.636 |
| desk023 | 3 | **0.921** | 0.705 |
| **Overall** | 52 | 0.824 | **0.853** |

### Phys. contact@3mm (↑ higher better)

| Object | n | Ours | OmniRt |
|---|---|---|---|
| box001 | 6 | **0.612** | 0.015 |
| box004 | 3 | **0.486** | 0.064 |
| box021 | 11 | **0.552** | 0.149 |
| box023 | 7 | **0.583** | 0.281 |
| box024 | 4 | **0.510** | 0.000 |
| bucket003 | 3 | **0.564** | 0.009 |
| bucket007 | 4 | **0.492** | 0.062 |
| chair006 | 4 | **0.711** | 0.038 |
| desk007 | 3 | **0.735** | 0.014 |
| desk021 | 4 | **0.619** | 0.149 |
| desk023 | 3 | **0.685** | 0.101 |
| **Overall** | 52 | **0.588** | 0.101 |

### Phys. contact@10mm (↑ higher better)

| Object | n | Ours | OmniRt |
|---|---|---|---|
| box001 | 6 | **0.821** | 0.289 |
| box004 | 3 | 0.679 | **0.689** |
| box021 | 11 | 0.761 | **0.904** |
| box023 | 7 | 0.731 | **0.774** |
| box024 | 4 | **0.860** | 0.104 |
| bucket003 | 3 | **0.751** | 0.105 |
| bucket007 | 4 | **0.798** | 0.240 |
| chair006 | 4 | **0.925** | 0.243 |
| desk007 | 3 | **0.862** | 0.225 |
| desk021 | 4 | **0.857** | 0.627 |
| desk023 | 3 | **0.921** | 0.506 |
| **Overall** | 52 | **0.804** | 0.510 |

### Phys. pen@3mm (↓ lower better)

| Object | n | Ours | OmniRt |
|---|---|---|---|
| box001 | 6 | **0.168** | 0.607 |
| box004 | 3 | **0.123** | 0.571 |
| box021 | 11 | **0.180** | 0.579 |
| box023 | 7 | **0.110** | 0.337 |
| box024 | 4 | **0.264** | 0.751 |
| bucket003 | 3 | **0.250** | 0.544 |
| bucket007 | 4 | **0.202** | 0.522 |
| chair006 | 4 | **0.173** | 0.472 |
| desk007 | 3 | **0.154** | 0.842 |
| desk021 | 4 | **0.184** | 0.346 |
| desk023 | 3 | **0.170** | 0.421 |
| **Overall** | 52 | **0.176** | 0.536 |

### Phys. pen@10mm (↓ lower better)

| Object | n | Ours | OmniRt |
|---|---|---|---|
| box001 | 6 | **0.010** | 0.419 |
| box004 | 3 | **0.003** | 0.171 |
| box021 | 11 | 0.019 | **0.001** |
| box023 | 7 | **0.017** | 0.039 |
| box024 | 4 | **0.004** | 0.662 |
| bucket003 | 3 | **0.109** | 0.462 |
| bucket007 | 4 | **0.013** | 0.414 |
| chair006 | 4 | **0.002** | 0.344 |
| desk007 | 3 | **0.023** | 0.669 |
| desk021 | 4 | **0.002** | 0.008 |
| desk023 | 3 | **0.000** | 0.144 |
| **Overall** | 52 | **0.017** | 0.247 |

### Max phys pen (mm) (↓ lower better)

| Object | n | Ours | OmniRt |
|---|---|---|---|
| box001 | 6 | **9.1** | 17.2 |
| box004 | 3 | **7.2** | 13.2 |
| box021 | 11 | 14.3 | **8.6** |
| box023 | 7 | 20.9 | **10.6** |
| box024 | 4 | **8.3** | 16.9 |
| bucket003 | 3 | **15.8** | 33.7 |
| bucket007 | 4 | **11.2** | 66.9 |
| chair006 | 4 | **7.4** | 29.8 |
| desk007 | 3 | **18.2** | 45.9 |
| desk021 | 4 | **7.2** | 10.7 |
| desk023 | 3 | **6.4** | 17.2 |
| **Overall** | 52 | **12.3** | 21.1 |

### Geom. pen@2mm (↓ lower better)

| Object | n | Ours | OmniRt |
|---|---|---|---|
| box001 | 6 | **0.038** | 0.507 |
| box004 | 3 | **0.054** | 0.541 |
| box021 | 11 | **0.062** | 0.225 |
| box023 | 7 | **0.076** | 0.272 |
| box024 | 4 | **0.037** | 0.526 |
| bucket003 | 3 | **0.038** | 0.378 |
| bucket007 | 4 | **0.055** | 0.514 |
| chair006 | 4 | **0.045** | 0.367 |
| desk007 | 3 | **0.062** | 0.814 |
| desk021 | 4 | **0.026** | 0.180 |
| desk023 | 3 | **0.044** | 0.285 |
| **Overall** | 52 | **0.052** | 0.381 |

## 5. Method-specific stability metrics (heterogeneous — not cross-compared)

| Method | Metric | mean | worst |
|---|---|---|---|
| SPIDER-CEM | Fall | 0.00 | 0.00 |
| SPIDER-CEM | Body-z err p95 (m) | 0.09 | 0.17 |
| SPIDER-CEM | Ankle jerk p95 (m/s^3) | 653.80 | 981.83 |
| SPIDER-CEM | Obj speed max (m/s) | 1.29 | 1.95 |
| SPIDER-CEM | Foot slip max (m) | 0.87 | 2.33 |
| OmniRetarget | Ankle jerk p95 (m/s^3) | 1560.83 | 3149.26 |
| OmniRetarget | Obj speed max (m/s) | 2.83 | 5.70 |
| OmniRetarget | Foot slip max (m) | 1.33 | 2.40 |

## Notes & caveats

- **Fair comparison = the shared contact metrics** (Section 1/3). These are computed identically for both methods.
- **Tracking metrics** (root/eef/obj error) require a tracked physics rollout; OmniRetarget is a kinematic retarget and does not produce them, so they are reported for SPIDER-CEM only (see `paper_results`).
- **Stability metrics**: SPIDER-CEM and OmniRetarget (physics-replay, E197 protocol) report ankle-jerk / obj-speed / foot-slip on the same MuJoCo rollout; listed for reference.
- **Raw vs physical contact:** 'raw contact' measures contact against the kinematic reference mask; 'physical contact@3mm' measures contact actually realized under physics within a 3 mm band. A high raw / low physical gap (OmniRetarget) indicates contact that does not survive physics.
- **Interim scope:** SBTO and GMR are temporarily excluded while their upstream workbooks are regenerated; this report compares SPIDER-CEM vs OmniRetarget only.
- All numbers are means over cases; per-case distributions and std/worst are in `paper_comparison.xlsx`.