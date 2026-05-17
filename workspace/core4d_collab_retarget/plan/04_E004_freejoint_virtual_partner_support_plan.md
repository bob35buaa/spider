# E004 Plan: true-freejoint large-scale virtual partner support sweep

日期：2026-05-17

状态：计划重写版。E004 作为一个较大规模实验执行，先不节省 GPU；采用本地 1 卡 + 远程 2 卡的多轮并行。

## Context

E002 证明：E081 风格的单机器人 reward/control 在 `scene.xml` 真 freejoint object 下不能通过真实接触完成运输。E003 继续证明：降低质量、提高摩擦只能把 `box025_p2` 最好 case-window obj mean 从 `0.703m` 改到 `0.400m`，但仍然 floor-supported；`box023_p2` guard 仍失败，并出现 `10-11%` 腿/箱干涉。

这说明当前瓶颈不再是单纯物理参数，而是 CORE4D 大物体任务本来是双人协作。sim2real 目标也是单机器人策略 + 真实人类在另一端支撑。因此 E004 专门验证 H003：在保持 object 真 freejoint 的前提下，用虚拟协作者外力模拟另一端人类支持，观察是否能把 object transport 从“单机器人推/拖”推进到“有协作者支撑的可搬运”。

E004 使用现有 freejoint 外力路径：

- `partner_force_scale`
- `partner_force_spring_kp`
- `partner_force_spring_kd`
- `partner_force_spring_kp_rot`
- `partner_force_spring_kd_rot`
- `partner_force_rot_clamp`
- `mjwp.py::_apply_partner_force`

这个路径写入 object body 的 `xfrc_applied`，不使用 `scene_act.xml` object actuators。主线必须保持 `scene_name: ""`、`contact_guidance: false`、`object_action_dims: 0`、`nu=29`、`nq_obj=7`。

## Historical References

旧实验只作为先验，不作为当前结论。原因是当时存在后续已修复或已审计的问题，包括 E070 ctrl mapping parity、E048 collision boxes、person2 ref/leg-object pair、以及旧版本 reward/control 口径不统一。

| 来源 | 历史观察 | E004 采用方式 |
|------|----------|---------------|
| `core4d` E024 | 50-90% gravity compensation 不能让 G1 主动接近并搬动物体；90% 更像物体变轻/漂浮 | 保留一个 gravity-only control 复验，但不把它当主线 |
| `core4d` E028 | position-only damped spring 稳定、能产生手接触，但 orientation 会翻/漂 | 主线采用 translation spring；rotation 不进首轮主线 |
| `core4d` E029 | `scene_act` actuator / kinematic override 与 freejoint CEM 目标不一致 | E004 禁止 object actuator、禁止 kinematic override |
| `core4d` E030 | CPU 单环境 torque PD 可行；Warp batch 中 position spring + orientation torque 出现正反馈和大量 NaN，weld 也不可靠 | 主线 `partner_force_spring_kp_rot=0`；rotation torque 只允许作为后置隔离 probe |
| E004 v2 draft | 建议 `g=0.5` + translational spring `kp=10/20/40`，并把 contact reward 与 force support 分开 | 采用参数区间，但扩成多轮 sweep，加入 control、guard、hold-contact 变体 |

## Claims

| Claim | 验证证据 |
|-------|----------|
| C1 虚拟协作者 translational support 是缺失因素之一 | `box025_p2` obj mean/max 明显低于 E003 最好 `0.400/0.795m`，且仍为 `scene.xml` freejoint |
| C2 gravity-only 不是充分方案 | `E004_box025_p2_g05` 若仍失败，就停止把 gravity compensation 当主调参方向 |
| C3 translational spring 的有效区间在 `kp=10-40` 附近 | `s10/s20/s40` 形成单调或局部最优趋势；高 kp 若引入摔倒/拖地/抖动，需要记录为不可行边界 |
| C4 不能让虚拟协作者独自完成任务 | 成功变体必须保持合理 hand contact / hold-contact，视频中机器人不能只是旁观 |
| C5 rotation torque 不是 E004 主线 | 首轮所有 full variants `kp_rot=0`；只有 translation 成功但 orientation 成为唯一瓶颈时，才做小样本 torque probe |

## Non-negotiables

- 使用 E002 的原始质量 true-freejoint leg-object tasks，不使用 E003 lightweight 派生 task。
- 主线所有 full variants：`contact_guidance=false`、`scene_name=""`、`object_pd_override=false`、`object_action_dims=0`、`partner_force_spring_kp_rot=0`。
- “contact/hold reward” 只允许作为 reward-only 参数启用，不能继承会打开 `scene_act` 的 E078/E041 override defaults。
- 每次正式训练前执行 scene snapshot，记录当前 `scene.xml` 与 `task_info.json` sha256。
- 评估必须对齐 E081/E002/E003：object mean/max、hand contact、leg-object interference、object-floor contact、bottom mean、pelvis stability、关键帧视频观察。

## Variant Grid

### Wave A: support force sweep

第一轮重新跑旧方向的关键控制组，同时扫 translation spring。全部保持 hold-contact off，用于分离“力支持”本身的作用。

| Variant | Source task | Force config | Reward config | Role |
|---------|-------------|--------------|---------------|------|
| `E004_box025_p2_g05` | `box025_person2_freejoint_legobj` | `partner_force_scale=0.5`, `kp=0`, `kp_rot=0` | E002 freejoint reward, `hold_contact=0` | gravity-only control |
| `E004_box025_p2_s10` | `box025_person2_freejoint_legobj` | `scale=0.5`, `kp=10`, `kd=-1`, `kp_rot=0` | E002 freejoint reward, `hold_contact=0` | low spring |
| `E004_box025_p2_s20` | `box025_person2_freejoint_legobj` | `scale=0.5`, `kp=20`, `kd=-1`, `kp_rot=0` | E002 freejoint reward, `hold_contact=0` | main anchor |
| `E004_box025_p2_s40` | `box025_person2_freejoint_legobj` | `scale=0.5`, `kp=40`, `kd=-1`, `kp_rot=0` | E002 freejoint reward, `hold_contact=0` | aggressive boundary |
| `E004_box023_p2_s10` | `box023_person2_freejoint_legobj` | `scale=0.5`, `kp=10`, `kd=-1`, `kp_rot=0` | E002 freejoint reward, `hold_contact=0` | guard stable |
| `E004_box023_p2_s20` | `box023_person2_freejoint_legobj` | `scale=0.5`, `kp=20`, `kd=-1`, `kp_rot=0` | E002 freejoint reward, `hold_contact=0` | guard stress |

### Wave B: robot participation / hold-contact reward

第二轮只在 Wave A 显示 object transport 有改善时执行。它不是 `contact_guidance=true`，而是复用 E075B/E078 中 reward-only 的 hold/contact 思路，强制机器人在 ref 接触窗口保持手-object 接近。

| Variant | Base | Additional reward-only params | Role |
|---------|------|-------------------------------|------|
| `E004_box025_p2_s20_hc` | `s20` | `hold_contact_rew_scale=1.0`, `hold_contact_sigma=0.05`, window `1.8-2.5s`, require ref contact | main participation check |
| `E004_box025_p2_s40_hc` | `s40` | same | aggressive participation check |
| `E004_box023_p2_s10_hc` | `box023 s10` | same | guard participation check |

如果 Wave A 中 `s10` 明显优于 `s20/s40`，则把 `box025_s20_hc` 替换为 `box025_s10_hc`，不要机械跑坏参数。

### Wave C: optional orientation isolation

只有当 Wave A/B 同时满足：

- object position mean 已经进入 `<0.30m`；
- floor contact 和 hand contact 合理；
- 视频显示主要剩余问题是 object orientation；
- 没有 NaN、没有摔倒。

才允许单独做 rotation probe：

| Variant | Config | Scope |
|---------|--------|-------|
| `E004_box025_p2_s20_rot025_probe` | `scale=0.5`, `kp=20`, `kp_rot=0.25`, `kd_rot=-1`, `rot_clamp=0.10` | smoke + short full only |
| `E004_box025_p2_s20_rot050_probe` | `scale=0.5`, `kp=20`, `kp_rot=0.50`, `kd_rot=-1`, `rot_clamp=0.10` | only if `rot025` no NaN |

任何 rotation probe 一旦出现 NaN 或 object 爆飞，立即停止，不把 rotation torque 纳入 E004 结论主线。

## Parallel Execution Plan

先本地 smoke 全部配置，确认 override/model parity。正式 Wave A/B 用本地 1 卡 + 远程 2 卡：

| GPU | Queue |
|-----|-------|
| local GPU0 | `E004_box025_p2_s20` -> `E004_box025_p2_s20_hc` |
| remote GPU0 | `E004_box025_p2_g05` -> `E004_box025_p2_s10` -> `E004_box025_p2_s40` |
| remote GPU1 | `E004_box023_p2_s10` -> `E004_box023_p2_s20` -> `E004_box023_p2_s10_hc` |

如果 local full 先完成且 `s20` 已经明显失败，则本地后续不跑 `s20_hc`，改跑当前最有希望的 hold-contact 变体。

## Implementation Plan

新增或更新：

- `workspace/core4d_collab_retarget/scripts/E004/variants.tsv`
- `workspace/core4d_collab_retarget/scripts/E004/generate_e004_overrides.py`
- `workspace/core4d_collab_retarget/scripts/run_E004_preprocess.sh`
- `workspace/core4d_collab_retarget/scripts/train/train_E004.sh`
- `workspace/core4d_collab_retarget/scripts/run_E004_remote.sh`
- `workspace/core4d_collab_retarget/scripts/pull_E004_remote_results.sh`
- `workspace/core4d_collab_retarget/scripts/eval/eval_E004.py`
- `workspace/core4d_collab_retarget/log/04_E004_freejoint_virtual_partner_support_results.md`

实现细节：

1. 从 E002 `box025_person2_freejoint_legobj` / `box023_person2_freejoint_legobj` 生成 override，不再改 dataset XML。
2. 每个 override 显式写入 parity guard 字段：`scene_name=""`、`contact_guidance=false`、`object_pd_override=false`、`object_action_dims=0`、`object_actuator_ids=[]`。
3. `generate_e004_overrides.py` 复制 E002 contact masks 到 `results/E004/contact_masks/`，并重新计算 palm normal。
4. CR/HC variants 只改 `hold_contact_*` 和必要 contact mask 参数，不继承 `core4d_e078*` 的 `scene_act/contact_guidance` defaults。
5. `train_E004.sh` 入口支持 `smoke`、`local_wave`、`one <variant> <gpu>`。
6. `run_E004_remote.sh` 在远端创建 tmux 友好的 GPU0/GPU1 队列；每个 variant 的 `output_dir`、`video_output_path`、log path 必须唯一。
7. `eval_E004.py` 复用 E003 指标，并新增 force-config audit、estimated support force range、NaN/frame-count guard。

## Commands

```bash
bash workspace/core4d_collab_retarget/scripts/run_E004_preprocess.sh
bash workspace/core4d_collab_retarget/scripts/train/train_E004.sh smoke 0
git add <E004 files> && git commit -m "exp(core4d_collab_retarget): E004 partner support setup" && git push
bash workspace/core4d_collab_retarget/scripts/train/train_E004.sh local_wave 0
bash workspace/core4d_collab_retarget/scripts/run_E004_remote.sh
bash workspace/core4d_collab_retarget/scripts/pull_E004_remote_results.sh
uv run python workspace/core4d_collab_retarget/scripts/eval/eval_E004.py --all
```

## Success Criteria

| Area | Useful | Strong |
|------|--------|--------|
| `box025_p2` object tracking | case-window mean `<0.30m`, max `<0.70m` | mean `<0.20m`, max near E081 `0.27m` |
| `box025_p2` floor/lift | floor contact below E003 best `76.9%`, bottom mean better than `-0.083m` | floor contact near/below E081 `59.5%`, visual not dragged |
| robot participation | hand contact remains `>=80%` or hold window visibly maintained | video shows robot hand-object contact through lift/transport |
| `box023_p2` guard | no fallover, pelvis stable, leg interference `<=5%` | obj mean `<0.45m`, floor contact below E002 `88.7%` |
| model parity | all variants `nu=29`, `nq_obj=7`, `contact_guidance=false` | same plus eval confirms no object actuator ids |

## Decision Rules

- If `g05` fails and `s10/s20/s40` improve, gravity-only is ruled out and E005 should focus on support geometry/contact semantics, not gravity tuning.
- If all spring variants remain `obj mean >0.50m`, partner xfrc at object COM is insufficient; next step should move to explicit support-point forces, dual-agent proxy, or connect/equality constraint experiments.
- If `s40` improves object but increases fall/leg/floor artifacts, use the best lower kp and add scheduling/force caps rather than increasing stiffness.
- If object tracks well but hand contact collapses, Wave B hold-contact variants become the main result; otherwise E004 is not RL-demo-ready.
- If only orientation is bad after position success, run Wave C rotation probe once; do not repeat E030-style torque sweeps after NaN.

## Expected Log Content

The E004 result log must include:

- config parity table for every variant;
- Wave A/B metrics table against E081/E002/E003;
- keyframe observations at f100/f125/f160/f204 or closest valid frames;
- force/support interpretation, including whether object was carried, floated, dragged, or pushed by legs;
- claim verification C1-C5;
- recommendation for E005, with a clear stop/continue decision for virtual partner xfrc.
