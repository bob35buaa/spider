# E002 Results: freejoint leg-object control audit

日期：2026-05-17

## Status

E002 setup 已实现；正在等待 GPU smoke/full CEM。

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
