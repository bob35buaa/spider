# Phase 6 Plan: Hand Contact Guidance (E025-E027)

## Context

E024 (Phase 5 final) established that SPIDER CEM **cannot** produce contact-rich manipulation through:
- Partner force (gravity compensation)
- Body tracking alone
- Object reward alone

Root cause: CEM joint-space sampling doesn't discover "reach → contact → push" causal sequences.

## Key Data Discovery (Pre-Plan Analysis)

Hand-to-object-**surface** distance in anchored reference trajectories:

| Case | Mean dist | <5cm frames | <10cm frames | Feasibility |
|------|-----------|-------------|--------------|-------------|
| bucket010 | 0.074m | **50%** | **75%** | **Primary target** |
| desk005 | 0.145m | 0% | 0% | Secondary (needs approach) |
| box025 | 0.175m | 0% | 2% | Excluded (arm span) |

**bucket010 is uniquely favorable**: the reference trajectory already places G1 wrists near the object surface. A hand-approach reward can provide gradient to close the remaining ~5-7cm gap.

## Experiment Series

### E025: Hand Approach Reward (Primary)

**Hypothesis**: Adding an explicit hand-to-object-surface distance reward provides CEM with gradient toward contact, overcoming the "blind sampling" limitation.

**Approach**:
```python
# New reward component: hand_approach_rew
hand_approach_rew = hand_approach_rew_scale * exp(-sigma * min_hand_surface_dist)
```

Unlike `contact_rew` (which only fires AFTER contact), this reward is **always active** — it provides gradient from any distance. Unlike `task_body_rew` (which tracks reference positions), this directly minimizes hand-object distance regardless of reference.

**Implementation**:
1. Add `hand_approach_rew_scale` and `hand_approach_sigma` to Config
2. In `get_reward()`, compute hand-to-object distance and add exponential reward
3. Use object geom bounding box (from `model.geom_size`) for surface distance approximation
4. Add partner force (50% gravity comp) to reduce object weight requirement

**Test Matrix** (bucket010 anchored + partner force 50%):
- E025-a: hand_approach=5.0, sigma=5.0, partner_force=0.5
- E025-b: hand_approach=10.0, sigma=10.0, partner_force=0.5
- E025-c: hand_approach=5.0, sigma=5.0, partner_force=0.0 (no partner, pure approach)
- E025-d: hand_approach=5.0 + task_body_rew=1.0 (hand weight=20.0), partner_force=0.5

**Claims (strict)**:
- C1: Min hand-surface dist in sim < 0.03m for ≥30 consecutive frames (sustained proximity)
- C2: Video confirms hands visibly reaching toward object (not staying at sides)
- C3: Any obj_z increase correlates temporally with hand proximity (causation check)
- C4: pelvis_z ≥ 0.50m ≥95% frames (stability preserved)
- C5: Improvement over E024 baseline (hand dist in sim < hand dist in E024-a1)

**Success criteria**: C1 + C2 + C4 all pass → proceed to E026 (approach + push)

---

### E026: Hand Approach + Object Spring (Direction 3)

**Conditional on**: E025 achieves hand proximity (C1 pass) but object doesn't move.

**Hypothesis**: If hands reach the object but can't generate enough force, a weak spring pulling the object toward reference provides the "initial push" — CEM only needs to maintain contact, not initiate lift.

**Approach**:
```python
# Combine: hand_approach (guides hands near) + spring (assists object motion)
partner_force_spring_kp = 50.0  # weak spring toward ref pos
hand_approach_rew_scale = 5.0
```

**Implementation**: Uses existing `partner_force_spring_kp` infrastructure from E024.

**Test Matrix** (bucket010 anchored):
- E026-a: approach=5.0 + spring_kp=50 (weak spring)
- E026-b: approach=5.0 + spring_kp=100 (moderate spring)
- E026-c: approach=10.0 + spring_kp=50 + obj_rew=3.0

**Claims**:
- C1: obj_z > init+0.05m sustained ≥30 frames (0.6s at 50fps ctrl)
- C2: Hand-object contact during object motion (not pure spring-driven)
- C3: Video confirms hand is on object when it moves
- C4: Stability preserved

---

### E027: Multi-Case Validation + High-Weight Hand Tracking

**Conditional on**: E025 or E026 works on bucket010.

**Hypothesis**: Successful approach transfers to desk005 with adjusted parameters.

**Test Matrix**:
- E027-a: Best E025/E026 config on desk005 (longer approach distance)
- E027-b: Combined hand_approach + task_body_rew (hand_weight=30.0) on bucket010
- E027-c: Combined approach on desk005

---

## Code Changes Required

### `spider/config.py` additions:
```python
# E025: Hand approach reward — guides hands toward object surface
hand_approach_rew_scale: float = 0.0
hand_approach_sigma: float = 5.0  # steepness of exponential decay
hand_approach_body_names: list[str] = field(default_factory=lambda: ["left_wrist_yaw_link", "right_wrist_yaw_link"])
hand_approach_body_ids: list[int] = field(default_factory=list)  # resolved at runtime
```

### `spider/simulators/mjwp.py` additions in `get_reward()`:
```python
# E025: hand approach reward — exp decay of hand-to-object-surface distance
hand_approach_rew = torch.zeros(N, device=config.device)
if config.hand_approach_rew_scale > 0.0 and config.hand_approach_body_ids:
    xpos_sim = wp.to_torch(env.data_wp.xpos)  # (N, nbody, 3)
    hand_pos = xpos_sim[:, config.hand_approach_body_ids]  # (N, 2, 3)
    obj_pos = xpos_sim[:, obj_body_id:obj_body_id+1]  # (N, 1, 3)
    # Surface distance approximation using object geom half-extents
    delta = torch.abs(hand_pos - obj_pos)  # (N, 2, 3)
    half_ext = torch.tensor(obj_half_extents, device=config.device)  # (3,)
    surface_dist = torch.clamp(delta - half_ext, min=0.0)  # (N, 2, 3)
    min_dist = surface_dist.norm(dim=-1).min(dim=1).values  # (N,) — min over hands
    hand_approach_rew = config.hand_approach_rew_scale * torch.exp(-config.hand_approach_sigma * min_dist)
```

### Config YAML (`core4d_box025_e025.yaml` template):
```yaml
dataset_name: core4d
task: bucket010_person1
data_id: 0
robot_type: unitree_g1
embodiment_type: humanoid_object
data_path: "...trajectory_kinematic_anchored.npz"

# Hand approach reward
hand_approach_rew_scale: 5.0
hand_approach_sigma: 5.0
hand_approach_body_names: ["left_wrist_yaw_link", "right_wrist_yaw_link"]

# Partner force (50% grav comp)
partner_force_scale: 0.5

# Body tracking (moderate)
base_pos_rew_scale: 10.0
joint_rew_scale: 3.0
```

## Implementation Order

1. Add config fields → `spider/config.py`
2. Add `hand_approach_rew` to `get_reward()` → `spider/simulators/mjwp.py`
3. Resolve `hand_approach_body_ids` in `process_config()` → `spider/config.py`
4. Create config YAML for E025 variants
5. Run E025-a through E025-d on bucket010
6. Analyze results (video + metrics)
7. If E025 passes → E026, else → reassess

## Data Paths

| Item | Path |
|------|------|
| Input trajectory | `example_datasets/processed/core4d/.../bucket010_person1/0/trajectory_kinematic_anchored.npz` |
| Results | `workspace/core4d/results/E025_hand_approach/bucket010/{a,b,c,d}.{npz,mp4}` |
| Config | `examples/config/override/core4d_bucket010_e025.yaml` |
| Log | `workspace/core4d/log/23_E025_hand_approach_results.md` |

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| CEM still ignores approach reward (too weak vs body tracking) | Sweep scales: 5/10/20 |
| Hands approach but can't generate contact force | E026 spring assists |
| Object half-extent approximation inaccurate (non-box shapes) | Use actual geom size from model |
| bucket010 success doesn't transfer to other cases | E027 validates generalization |
| Reward conflicts (approach pulls hand away from body reference) | E025-d tests combined weights |
