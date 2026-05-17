# E002 Results: freejoint leg-object control audit

日期：2026-05-17

## Status

E002 full CEM 已完成；结论是当前 E081 reward/control 口径在真 freejoint 物体下失败。

## Setup

- main: `E002_box025_p2_freejoint` -> `box025_person2_freejoint_legobj`
- guard: `E002_box023_p2_freejoint` -> `box023_person2_freejoint_legobj`
- 关键约束：`scene.xml` freejoint object, `contact_guidance=false`, `object_action_dims=0`, no object actuator ids。

## Commands

```bash
bash workspace/core4d_collab_retarget/scripts/run_E002_preprocess.sh
bash workspace/core4d_collab_retarget/scripts/train/train_E002.sh smoke 0
bash workspace/core4d_collab_retarget/scripts/train/train_E002.sh local 0
```

## Results

Full run:

```bash
bash workspace/core4d_collab_retarget/scripts/train/train_E002.sh local 0
```

输出：

- `workspace/core4d_collab_retarget/results/E002/comparison.csv`
- `workspace/core4d_collab_retarget/results/E002/E002_box025_p2_freejoint.{npz,mp4}`
- `workspace/core4d_collab_retarget/results/E002/E002_box023_p2_freejoint.{npz,mp4}`
- keyframes: `workspace/core4d_collab_retarget/results/E002/keyframes/`

### Full metrics

| Variant | Role | contact_guidance | nu | nq_obj | obj mean/max | hand contact | leg intf | floor contact | bottom mean | Success |
|---------|------|------------------|----|--------|--------------|--------------|----------|---------------|-------------|---------|
| `E002_box025_p2_freejoint` | main | False | 29 | 7 | `0.703/1.356m` | `89.6%` | `0.0%` | `85.5%` | `-0.073m` | numeric False, case-window False, strict False |
| `E002_box023_p2_freejoint` | guard | False | 29 | 7 | `0.830/1.488m` | `72.0%` | `0.0%` | `88.7%` | `0.026m` | numeric False, case-window False, strict False |

E081 baseline 同口径：

| Variant | Role | obj mean/max | hand contact | leg intf | floor contact | bottom mean | Success |
|---------|------|--------------|--------------|----------|---------------|-------------|---------|
| `E081_box025_p2_legobj` | main | `0.143/0.271m` | `89.0%` | `7.5%` | `59.5%` | `-0.075m` | numeric False, case-window True, strict False |
| `E081_box023_p2_legobj` | guard | `0.164/0.317m` | `66.7%` | `2.7%` | `34.7%` | `0.144m` | numeric True, case-window True, strict True |

### Frame-level check

| Variant | Frame | obj err | sim/ref obj z | hand contact | floor contact | note |
|---------|-------|---------|---------------|--------------|---------------|------|
| box025 | f125 | `0.848m` | `0.404/0.482m` | 2 | 2 | sim box remains floor-supported while ref has translated away |
| box025 | f160 | `1.334m` | `0.383/0.413m` | 0 | 2 | hands lose contact; object remains on floor |
| box023 | f125 | `1.387m` | `0.225/0.469m` | 2 | 1 | small box is tilted/contacted but far from ref |
| box023 | f204 | `1.463m` | `0.157/0.149m` | 0 | 4 | object has fallen/settled; hand contact gone |

### Interpretation

E002 answers the original freejoint question directly:

- The current E081-style reward stack is strongly dependent on `scene_act` object actuator guidance.
- Removing object actuator guidance does not merely degrade the large-box main case; it also breaks the small-box guard that E081 passed.
- The optimizer still finds hand-object contacts (`89.6%` and `72.0%`) and keeps leg/box interference at `0%`, so failure is not missing hand proximity or leg collision. It is the inability to generate stable object transport through physical contact alone.
- Visual/keyframe checks match the metrics: sim objects remain floor-supported or tilt/fall while the ref object translates/lifts away.

### Follow-up

E003 should not just rerun the same freejoint setting. The next useful experiment is a physics-feasibility sweep: object mass/contact/friction sensitivity under the same true-freejoint evaluation. If lighter/higher-friction objects still fail, we should move to explicit virtual grasp/contact constraints or dual-agent support rather than expecting the current single-agent CEM reward to solve freejoint transport.

## E070 parity bug 复查

用户提醒复查 `workspace/core4d/log/90_E070_mjwarp_ref_control_parity_results.md`。结论：

- E070 的根因是 `contact_guidance`/`scene_act` 下曾经把 `qpos_ref[:, :nu]` 当作 actuator `ctrl_ref`，导致 floating-base qpos 被送进 robot actuators。
- 当前 E002 freejoint 数据本身有 `ctrl` 字段，shape 是 29，和 freejoint model `nu=29` 一致；不会触发旧的 qpos-as-ctrl 路径。
- 但 `spider/io.py` 的缺失 `ctrl` fallback 仍有同类隐患：`humanoid_object` 会把 object qpos 拼进 ctrl。已修成只取 robot actuator 维度。
- `examples/run_mjwp.py` / `examples/run_mjwp_fast.py` 已增加进入 MJWarp 前的维度断言：`qpos_ref==nq`、`qvel_ref==nv`、`ctrl_ref==nu`，避免静默错跑。
- `spider/simulators/mjwp.py` local-frame joint reward 已从硬编码 `7:-7` 改成按 `config.nq_obj` 切分；freejoint 仍是 7，scene_act 是 6。

验证：

| Check | 结果 |
|-------|------|
| E002 `box025_person2_freejoint_legobj` | model `(nq,nv,nu)=(43,41,29)`；raw/post-convert ref `ctrl=29` |
| E081 `box025_person2_legobj` | model `(42,41,35)`；raw `ctrl=29`，scene_act post-convert `ctrl=35` |
| 模拟 NPZ 缺 `ctrl` | E002 fallback `ctrl=29`；E081 fallback post-convert `ctrl=35` |
| 静态检查 | `py_compile`、`bash -n`、`git diff --check` 均通过 |

## Smoke

- 沙箱内直接 GPU smoke 失败：PyTorch/Warp 看不到 CUDA。
- CPU 极小 run 已通过 ref 维度断言阶段，随后失败于 Warp graph capture：`RuntimeError: Must be a CUDA device`。这说明当前环境必须用提升权限访问 GPU 才能跑 MJWarp smoke/full CEM。
- 提升权限 GPU smoke 已通过：

```bash
bash workspace/core4d_collab_retarget/scripts/train/train_E002.sh smoke 0
.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E002.py \
  E002_box025_p2_freejoint E002_box023_p2_freejoint
```

Smoke 只跑 `max_sim_steps=4`，量化指标不代表实验结论；只用于验证脚本/维度/freejoint eval。

| Variant | T | contact_guidance | nq_obj | nu | npair | obj mean | hand contact | leg intf |
|---------|---|------------------|--------|----|-------|----------|--------------|----------|
| `E002_box025_p2_freejoint` | 4 | False | 7 | 29 | 42 | `0.048m` case-window | `0.0%` | `0.0%` |
| `E002_box023_p2_freejoint` | 4 | False | 7 | 29 | 42 | `0.010m` case-window | `0.0%` | `0.0%` |

下一步运行 full CEM 后再判定 E002 成败。
