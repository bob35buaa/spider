# E003 Results: freejoint physics feasibility sweep

日期：2026-05-17

## Status

E003 full CEM completed. Result: lowering mass and increasing hand/object friction does not recover true-freejoint success.

## Setup

四个 variant 均基于 E002 true-freejoint + leg-object collision task 派生，只修改派生 `scene.xml` 中的 object mass/inertia 和接触 friction：

| Variant | Source | Mass | Hand-object friction | Object-floor friction | Role |
|---------|--------|------|----------------------|-----------------------|------|
| `E003_box025_p2_m1` | `box025_person2_freejoint_legobj` | `1.0kg` | `2.0` | `1.0` | main |
| `E003_box025_p2_m1_f4` | `box025_person2_freejoint_legobj` | `1.0kg` | `4.0` | `0.5` | main |
| `E003_box023_p2_m1` | `box023_person2_freejoint_legobj` | `1.0kg` | `2.0` | `1.0` | guard |
| `E003_box023_p2_m1_f4` | `box023_person2_freejoint_legobj` | `1.0kg` | `4.0` | `0.5` | guard |

## Commands

```bash
bash workspace/core4d_collab_retarget/scripts/run_E003_preprocess.sh
bash workspace/core4d_collab_retarget/scripts/train/train_E003.sh smoke 0
bash workspace/core4d_collab_retarget/scripts/run_E003_remote.sh
bash workspace/core4d_collab_retarget/scripts/pull_E003_remote_results.sh
```

## Results

### Preprocess / config audit

`bash workspace/core4d_collab_retarget/scripts/run_E003_preprocess.sh` 已生成四个派生 task 和四个 override。

| Variant | Scene check | Config check |
|---------|-------------|--------------|
| `E003_box025_p2_m1` | object mass `1kg`, inertia `0.03 0.03 0.02`, hand/object friction `2`, object/floor friction `1` | `contact_guidance=False`, `scene_name=""`, `nq/nv/nu/nq_obj=43/41/29/7`, `ctrl_ref=29` |
| `E003_box025_p2_m1_f4` | object mass `1kg`, inertia `0.03 0.03 0.02`, hand/object friction `4`, object/floor friction `0.5` | `contact_guidance=False`, `scene_name=""`, `nq/nv/nu/nq_obj=43/41/29/7`, `ctrl_ref=29` |
| `E003_box023_p2_m1` | object mass `1kg`, inertia `0.03 0.03 0.02`, hand/object friction `2`, object/floor friction `1` | `contact_guidance=False`, `scene_name=""`, `nq/nv/nu/nq_obj=43/41/29/7`, `ctrl_ref=29` |
| `E003_box023_p2_m1_f4` | object mass `1kg`, inertia `0.03 0.03 0.02`, hand/object friction `4`, object/floor friction `0.5` | `contact_guidance=False`, `scene_name=""`, `nq/nv/nu/nq_obj=43/41/29/7`, `ctrl_ref=29` |

### Smoke

GPU smoke passed:

```bash
bash workspace/core4d_collab_retarget/scripts/train/train_E003.sh smoke 0
.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E003.py
```

Smoke uses `max_sim_steps=4`, so the table below only verifies execution/eval plumbing.

| Variant | T | nu | nq_obj | obj mean/max | hand contact | leg intf | floor contact |
|---------|---|----|--------|--------------|--------------|----------|---------------|
| `E003_box025_p2_m1` | 4 | 29 | 7 | `0.048/0.061m` | `0.0%` | `0.0%` | `100.0%` |
| `E003_box025_p2_m1_f4` | 4 | 29 | 7 | `0.049/0.062m` | `0.0%` | `0.0%` | `100.0%` |
| `E003_box023_p2_m1` | 4 | 29 | 7 | `0.010/0.013m` | `0.0%` | `0.0%` | `100.0%` |
| `E003_box023_p2_m1_f4` | 4 | 29 | 7 | `0.010/0.013m` | `0.0%` | `0.0%` | `100.0%` |

### Full metrics

Full CEM was launched on `spider-remote` in tmux session `E003`.

| Variant | Role | mass/friction | obj mean/max | hand contact | leg intf | floor contact | bottom mean | Success |
|---------|------|---------------|--------------|--------------|----------|---------------|-------------|---------|
| `E003_box025_p2_m1` | main | `1kg`, hand `2`, floor `1` | `0.622/1.190m` | `82.7%` | `0.0%` | `70.5%` | `-0.072m` | case-window False, strict False |
| `E003_box025_p2_m1_f4` | main | `1kg`, hand `4`, floor `0.5` | `0.400/0.795m` | `93.6%` | `1.7%` | `76.9%` | `-0.083m` | case-window False, strict False |
| `E003_box023_p2_m1` | guard | `1kg`, hand `2`, floor `1` | `0.831/1.512m` | `60.7%` | `11.3%` | `98.0%` | `0.040m` | case-window False, strict False |
| `E003_box023_p2_m1_f4` | guard | `1kg`, hand `4`, floor `0.5` | `0.947/1.678m` | `64.7%` | `10.0%` | `68.0%` | `0.005m` | case-window False, strict False |

Comparison to E002/E081:

| Case | E081 actuator-guided baseline | E002 true-freejoint | Best E003 true-freejoint | Interpretation |
|------|-------------------------------|---------------------|--------------------------|----------------|
| `box025_p2` | obj `0.143/0.271m`, hand `89.0%`, leg `7.5%`, floor `59.5%`, case-window True | obj `0.703/1.356m`, hand `89.6%`, leg `0.0%`, floor `85.5%` | `m1_f4`: obj `0.400/0.795m`, hand `93.6%`, leg `1.7%`, floor `76.9%` | lighter/high-friction improves transport but remains far from E081 and still floor-supported |
| `box023_p2` | obj `0.164/0.317m`, hand `66.7%`, leg `2.7%`, floor `34.7%`, strict True | obj `0.830/1.488m`, hand `72.0%`, leg `0.0%`, floor `88.7%` | `m1`: obj `0.831/1.512m`; `m1_f4`: obj `0.947/1.678m` | physics sweep does not recover guard; high friction destabilizes/falls |

### Frame-level check

| Variant | Frame | obj err | sim/ref obj z | hand contact | floor contact | leg contact | observation |
|---------|-------|---------|---------------|--------------|---------------|-------------|-------------|
| `box025_m1` | f125 | `0.705m` | `0.407/0.482m` | 1 | 0 | 0 | sim box remains near robot and lags ref translation |
| `box025_m1` | f160 | `1.178m` | `0.379/0.413m` | 0 | 2 | 0 | hand contact lost; box remains floor-supported |
| `box025_m1_f4` | f125 | `0.466m` | `0.417/0.482m` | 1 | 0 | 0 | better than E002 but still clearly behind ref |
| `box025_m1_f4` | f160 | `0.779m` | `0.383/0.413m` | 2 | 0 | 0 | hand contact persists, but object still not transported far enough |
| `box023_m1` | f125 | `1.370m` | `0.244/0.469m` | 1 | 1 | 3 | small box is supported/perturbed by leg contacts, not clean hand carry |
| `box023_m1_f4` | f125 | `1.574m` | `0.156/0.469m` | 0 | 4 | 0 | robot has fallen over the box; visual failure despite reduced floor friction |
| `box023_m1_f4` | f204 | `0.312m` | `0.246/0.149m` | 0 | 0 | 0 | object happens to be closer, but robot is fully fallen and task is invalid |

Keyframes:

- `workspace/core4d_collab_retarget/results/E003/keyframes/E003_box025_p2_m1_f4/f125.jpg`
- `workspace/core4d_collab_retarget/results/E003/keyframes/E003_box025_p2_m1_f4/f160.jpg`
- `workspace/core4d_collab_retarget/results/E003/keyframes/E003_box023_p2_m1/f125.jpg`
- `workspace/core4d_collab_retarget/results/E003/keyframes/E003_box023_p2_m1_f4/f125.jpg`
- `workspace/core4d_collab_retarget/results/E003/keyframes/E003_box023_p2_m1_f4/f204.jpg`

## Claims

| Claim | Status |
|-------|--------|
| C1 物理参数是主要瓶颈 | Rejected. `box025_m1_f4` improves over E002, but still fails by a wide margin; `box023` guard does not recover. |
| C2 当前 reward/optimizer 是主要瓶颈 | Supported. True-freejoint failure persists after mass/friction changes, so the current single-agent CEM reward/contact model is insufficient for object transport. |
| C3 接触增强没有通过腿/地板作弊 | Mixed/failed. Main stays mostly clean, but guard has `10-11%` leg-box interference and visual robot fallover. |

## Next

E004 should stop tuning passive physics alone. The next useful experiment is an explicit assistance/contact model:

1. Add a virtual collaborator/support term for freejoint object transport, aligned with the eventual sim2real setup where a human supports the other side.
2. Keep object true-freejoint in evaluation; do not return to `scene_act` object actuator tracking as the claimed result.
3. Evaluate against the same E081/E002/E003 metrics and keyframes.
