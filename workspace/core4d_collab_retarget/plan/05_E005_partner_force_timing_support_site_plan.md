# E005 Plan: corrected partner-force timing and support-site geometry

日期：2026-05-17

状态：计划中。E004 完成后立即启动；先修 `partner_force` 的 ref timing，再比较 corrected COM force 与 off-COM support-site force。

## Context

E004 的 9 组 true-freejoint 虚拟协作者实验全部保持 `nu=29`、`nq_obj=7`、no object actuator，但 main `box025_p2` 全部失败，视觉上主要表现为 sim object 滞后 ref、仍被拖/推在地面附近。

E005 前代码复查发现一个必须先处理的问题：`spider/simulators/mjwp.py::_apply_partner_force` 用硬编码 `dt = 1.0 / 30.0` 把 sim time 映射到 `partner_force_ref_pos` frame。`box025_person2_freejoint_legobj` 的 `task_info.json` 确实是 30Hz，因此 E004 main 的 COM spring timing 不被这个问题推翻；但 `box023_person2_freejoint_legobj` 没有 `ref_dt` 字段，会使用 config 默认 50Hz (`0.02`)。所以 E004 guard spring timing 存在滞后风险，且代码本身不应写死 30Hz。

同时，E004 也暴露出 COM-level force 的语义不足：即使 hand contact 很高，物体不一定获得可传力的双端支撑。因此 E005 把两个因素拆开验证：

1. per-task ref_dt 的 COM spring 是否已经足够；
2. 如果仍不足，off-COM support-site force 是否能提供更接近双人协作的支撑几何。

## Claims

| Claim | 验证证据 |
|-------|----------|
| C1 partner-force timing 必须按 task 显式处理 | box025 使用 `0.0333333`，box023 使用 `0.02`；所有 eval parity 记录 `partner_force_ref_dt` |
| C2 仅修 timing 仍不足以恢复协作搬运 | corrected COM 仍未达到 useful proxy：obj mean `<0.30m`、max `<0.70m`、floor contact `<76.9%` |
| C3 off-COM support-site 比 COM force 更有物理语义 | 同 kp 下 support-site 的 obj mean/floor contact/视觉 carry 语义优于 corrected COM |
| C4 support-site 方向必须可解释 | `local -Y` 与 `local +Y` side ablation 至少一个明显更合理；若二者都差，说明不是简单侧向 support point |
| C5 guard 不应被 support-site 破坏 | `box023_p2` guard pelvis stable、leg interference `<=5%`，否则该支撑几何不可作为主线 |

## Code Changes

1. `spider/config.py`
   - 新增 `partner_force_ref_dt`，默认 `-1.0` 表示使用 `config.ref_dt`，E005 override 按 task 显式写入。
   - 新增 `partner_force_point_local`，空列表表示旧 COM mode；长度 3 表示 object-local support point。
   - 新增 force/torque clamp，防止 off-COM wrench 在 batch 中爆炸。

2. `spider/simulators/mjwp.py`
   - partner force ref index 从硬编码 `1/30` 改为 `env.partner_force_ref_dt`。
   - precompute ref 时同时存 `env.partner_force_ref_dt`。
   - 当 `partner_force_point_local` 有效时，把 local point 旋到 world：
     - current point = object COM + `R_obj * point_local`
     - target point = ref COM + `R_ref * point_local`
     - point velocity = linear vel + `omega x r`
     - `F = gravity + kp*(target-current) - kd*point_vel`
     - 写入等效 wrench：force 到 `xfrc[:3]`，torque 到 `xfrc[3:6] += r x F`
   - `kp_rot` 仍保持 0；off-COM torque 只来自点力矩臂，不再引入 E030-style orientation PD。

## Variant Grid

所有 variant 使用 E002 true-freejoint leg-object tasks，保持 `scene_name=""`、`contact_guidance=false`、`object_action_dims=0`、`object_actuator_ids=[]`。

| Variant | Task | Mode | point local | kp | hold | Role |
|---------|------|------|-------------|----|------|------|
| `E005_box025_p2_com_s20` | `box025_person2_freejoint_legobj` | COM, explicit 30Hz | empty | 20 | 0 | COM baseline |
| `E005_box025_p2_com_s40` | same | COM, explicit 30Hz | empty | 40 | 0 | stronger COM baseline |
| `E005_box025_p2_yneg_s20` | same | support-site | `[0,-0.38,0.30]` | 20 | 0 | partner-side hypothesis |
| `E005_box025_p2_yneg_s40` | same | support-site | `[0,-0.38,0.30]` | 40 | 0 | stronger support-site |
| `E005_box025_p2_ypos_s20` | same | support-site | `[0,0.38,0.30]` | 20 | 0 | side ablation |
| `E005_box025_p2_yneg_s20_hc` | same | support-site | `[0,-0.38,0.30]` | 20 | 1 | robot participation |
| `E005_box023_p2_com_s10` | `box023_person2_freejoint_legobj` | COM, explicit 50Hz | empty | 10 | 0 | guard timing baseline |
| `E005_box023_p2_xneg_s10` | same | support-site | `[-0.16,0,0.10]` | 10 | 0 | guard partner-side |
| `E005_box023_p2_xpos_s10` | same | support-site | `[0.16,0,0.10]` | 10 | 0 | guard side ablation |

## Parallel Execution

| GPU | Queue |
|-----|-------|
| local GPU0 | `box025_com_s20` -> `box025_yneg_s20` -> `box025_yneg_s20_hc` |
| remote GPU0 | `box025_com_s40` -> `box025_yneg_s40` -> `box025_ypos_s20` |
| remote GPU1 | `box023_com_s10` -> `box023_xneg_s10` -> `box023_xpos_s10` |

即使 COM 已明显成功，本轮仍保留 support-site side ablation，避免把 timing/ref_dt 口径误判为 geometry 贡献。

## Commands

```bash
bash workspace/core4d_collab_retarget/scripts/run_E005_preprocess.sh
bash workspace/core4d_collab_retarget/scripts/train/train_E005.sh smoke 0
git add <E005 files> && git commit -m "exp(core4d_collab_retarget): E005 support-site setup" && git push
bash workspace/core4d_collab_retarget/scripts/train/train_E005.sh local_wave 0
bash workspace/core4d_collab_retarget/scripts/run_E005_remote.sh
bash workspace/core4d_collab_retarget/scripts/pull_E005_remote_results.sh
.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E005.py --all
```

## Success Criteria

| Area | Useful | Strong |
|------|--------|--------|
| COM baseline | box025 does not regress from E004 same-30Hz `s20/s40`; box023 guard timing uses 50Hz explicitly | reaches E003 best or better (`<=0.40m` mean, `<=0.80m` max) |
| support-site contribution | same kp support-site beats corrected COM by `>=0.10m` mean or materially reduces floor contact | main proxy: mean `<0.30m`, max `<0.70m`, floor contact `<76.9%` |
| robot participation | hand contact `>=80%` or HC variant preserves visible hand-object support | video shows robot and virtual partner share load, not object-only tracking |
| guard | pelvis min `>=0.55m`, leg interference `<=5%` | stable and obj mean `<0.45m` |
| parity | every variant `nu=29`, `nq_obj=7`, `contact_guidance=false` | same plus no object actuators and `kp_rot=0` |

## Decision Rules

- If COM baseline succeeds, E006 should focus on making the virtual partner physically realizable/sim2real-friendly rather than more support-site sweeps.
- If corrected COM improves but support-site improves further, E006 should refine support geometry and eventually replace hand-tuned point offsets with contact-derived site selection.
- If corrected COM and support-site both fail, stop tuning `partner_force_*`; move to explicit dual-agent/proxy body or equality/contact constraint support.
- If any off-COM support creates falls/NaNs, reduce force/torque clamp before increasing stiffness.
