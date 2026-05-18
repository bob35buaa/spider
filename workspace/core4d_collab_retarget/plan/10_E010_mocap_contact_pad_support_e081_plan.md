# E010 计划：mocap contact pad 虚拟协作者支持

日期：2026-05-18

## Context

E006 视频和后验指标确认了“只旋转、不平移”的失败形态：参考 `box025_p2` object 水平位移约 `1.57m`，E006 main 实际只有 `0.21-0.50m`，但起终旋转达到 `20-48deg`。根因分两层：

1. E006 写死 `support_proxy_ref_dt=0.0333`，与插值后的 `sim_dt` 不一致，proxy target 自身没有走完整参考；
2. direct off-COM wrench 的 `r x F` 会自然产生力矩；当 robot hand/support 没有闭环、object 仍高比例贴地时，系统更容易用绕地面/支撑点旋转来降低误差，而不是形成 COM 水平运输。

E007/E008 已经修正第 1 层：E008 best `ypos_k20_vmax2` 的 proxy ratio `0.998`，object xy ratio 提升到 `0.724`，rot 降到 `13.6deg`。E009 进一步证明第 2 层不能靠 hold-contact reward 解决：4/4 main proxy gate 通过，但 `num_main_reaches_E081_transport_proxy=0`、`num_main_improves_E008_best=0`；强 HC 会引入 `38deg` 级旋转或 `15-19%` 腿/箱干涉。

E010 因此转向结构性接触：用一个由 support proxy trajectory 驱动的 mocap/contact pad 作为虚拟协作者手/支撑面，通过 MuJoCo contact 或软约束把力传给 object，而不是继续直接写 `xfrc_applied`。

## Claims

| Claim | 验证方式 |
|-------|----------|
| C1 contact pad 能避免 direct wrench 的自由力矩捷径 | 与 E009 比，object rotation 不升高，xy ratio 不靠 `>30deg` 旋转达成 |
| C2 contact pad 能给 object 提供更物理的 partner-side 反作用 | pad-object contact pct、object floor pct、connector/pad gap 与视频同步改善 |
| C3 如果 robot-side 仍不闭环，pad 会暴露为“虚拟人独自搬” | hand contact、pad contact、floor contact 分开记录；hand 不足则不判 work |
| C4 eval 继续对齐 E081，而不是 E005/E008 | 保留 E009 的 E081 transport gate 和 majority gate |
| C5 true-freejoint parity 不破坏 | `nu=29`、`nq_obj=7`、object actuator empty；新增 mocap body 不进入 qpos/ctrl |

## 实现改动

1. Config / MJWarp
   - 新增 `support_proxy_mode`，默认 `"wrench"`，E010 使用 `"mocap_pad"`。
   - 复用 `_load_support_proxy()` 生成 proxy trajectory，但在 `"mocap_pad"` 下不对 object 写 direct wrench。
   - 新增 `_update_support_proxy_mocap_pad()`：按当前 sim time 取 `support_proxy_ref_pos[idx]`，写入 XML 中的 mocap body `support_proxy_pad`。
   - `get_support_proxy_state()` 继续记录 proxy/support point；若可行，额外记录 pad pose 或 pad-object contact 诊断。

2. Scene XML
   - 为 `box025_person2_freejoint_legobj` 和 `box023_person2_freejoint_legobj` 生成 `scene_contact_pad.xml`：
     - 保持 object true-freejoint；
     - 新增 `body name="support_proxy_pad" mocap="true"`；
     - 新增 pad geom，例如 sphere/ellipsoid，使用接触对 `support_proxy_pad_geom` ↔ `object_collision`；
     - pad 不与 robot/floor 接触，避免引入额外腿部碰撞。
   - 新增 scene XML 必须 `git add -f`，因为 `example_datasets/` 默认被 ignore。

3. E010 scripts
   - `workspace/core4d_collab_retarget/scripts/E010/variants.tsv`
   - `workspace/core4d_collab_retarget/scripts/E010/generate_e010_overrides.py`
   - `workspace/core4d_collab_retarget/scripts/run_E010_preprocess.sh`
   - `workspace/core4d_collab_retarget/scripts/train/train_E010.sh`
   - `workspace/core4d_collab_retarget/scripts/train/train_E010_remote_tmux.sh`
   - `workspace/core4d_collab_retarget/scripts/run_E010_remote.sh`
   - `workspace/core4d_collab_retarget/scripts/pull_E010_remote_results.sh`
   - `workspace/core4d_collab_retarget/scripts/eval/eval_E010.py`

## Variant Grid

首轮不贪多，先验证 contact pad 是否比 E009 direct wrench 更像搬运。

| Variant | Task | point local | pad size | vmax | HC | Role |
|---------|------|-------------|----------|------|----|------|
| `E010_box025_p2_ypos_pad10_vmax2` | `box025_person2_freejoint_legobj` | `[0, 0.38, 0.30]` | `0.10` | `2.0` | `0` | main anchor |
| `E010_box025_p2_ypos_pad16_vmax2` | same | `[0, 0.38, 0.30]` | `0.16` | `2.0` | `0` | contact area |
| `E010_box025_p2_ypos_pad10_vmax0` | same | `[0, 0.38, 0.30]` | `0.10` | `0.0` | `0` | no speed clamp |
| `E010_box025_p2_ypos_pad10_vmax2_hc05` | same | `[0, 0.38, 0.30]` | `0.10` | `2.0` | `0.5` | light robot participation |
| `E010_box025_p2_yneg_pad10_vmax2` | same | `[0, -0.38, 0.30]` | `0.10` | `2.0` | `0` | side ablation |
| `E010_box023_p2_xpos_pad08_vmax0` | `box023_person2_freejoint_legobj` | `[0.16, 0, 0.10]` | `0.08` | `0.0` | `0` | guard |
| `E010_box023_p2_xpos_pad12_vmax0` | same | `[0.16, 0, 0.10]` | `0.12` | `0.0` | `0.5` | guard + light HC |

## Parallel Execution

| GPU | Queue |
|-----|-------|
| local GPU0 | `E010_box025_p2_ypos_pad10_vmax2` -> `E010_box025_p2_ypos_pad10_vmax2_hc05` |
| remote GPU0 | `E010_box025_p2_ypos_pad16_vmax2` -> `E010_box025_p2_ypos_pad10_vmax0` -> `E010_box025_p2_yneg_pad10_vmax2` |
| remote GPU1 | `E010_box023_p2_xpos_pad08_vmax0` -> `E010_box023_p2_xpos_pad12_vmax0` |

## 成功标准

继承 E009 的 E081 transport gate：

- `E010_proxy_support_tracking_ok=true`
- `case_window_obj_err_mean_m <= 0.20`
- `case_window_obj_err_max_m <= 0.40`
- `case_window_sim_contact_frames_pct >= 80%`
- `case_window_sim_object_floor_contact_frames_pct <= 75%`
- `E010_object_xy_disp_ratio >= 0.75`
- `E010_object_rot_deg <= 15deg`
- `E010_freejoint_parity_ok=true`

E010 额外关注：

- pad-object contact pct 不能为 0；
- 若 object 指标改善但 hand contact 低于 E009/E081，则判为“虚拟 pad 独自搬”，不算 work；
- 若 rotation `>30deg` 或 leg interference `>12.5%`，即使 obj mean 降低也不算有效。

## 预授权命令

```bash
bash workspace/core4d_collab_retarget/scripts/train/train_E010.sh __codex_auth_probe__ 0
bash workspace/core4d_collab_retarget/scripts/run_E010_remote.sh __codex_auth_probe__
bash workspace/core4d_collab_retarget/scripts/pull_E010_remote_results.sh __codex_auth_probe__
```

## 执行命令

```bash
bash workspace/core4d_collab_retarget/scripts/run_E010_preprocess.sh
bash workspace/core4d_collab_retarget/scripts/train/train_E010.sh smoke 0
.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E010.py --all

# full
bash workspace/core4d_collab_retarget/scripts/run_E010_remote.sh
bash workspace/core4d_collab_retarget/scripts/train/train_E010.sh local_wave 0
bash workspace/core4d_collab_retarget/scripts/pull_E010_remote_results.sh
.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E010.py --all
```

## 决策规则

- 如果 contact pad 明显降低 object error 且 rotation/leg interference 受控，但 hand contact 仍低：E011 做 robot-side hand pose/contact shaping，而不是再调 pad。
- 如果 contact pad 只让 pad 独自搬 object：E011 改成双人/真实 partner mocap hand，而不是单 pad。
- 如果 contact pad 不改善或造成不稳定：回到 E008 best，尝试 soft equality/weld 的极弱约束作为 diagnostic，而不是继续加直接 wrench。
