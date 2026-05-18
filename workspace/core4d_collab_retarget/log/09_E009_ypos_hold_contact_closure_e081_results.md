# E009 结果：ypos best + robot-side hold-contact 闭环

日期：2026-05-18

## 状态

E009 plan、脚本、预授权、smoke、本地 full、远程 full、回收和显式 6-variant eval 均已完成。结论：在 E008 best `ypos_k20_vmax2` 上加 robot-side hold-contact 没有达到 E081，也没有改善 E008 best。proxy timebase/speed 已不再是瓶颈，4/4 main 都完整跟随 reference support point；失败来自 robot-side 支撑闭环仍未建立，hold-contact reward 越强越容易换来 object 旋转、腿/箱干涉或手接触下降。

## 执行命令

```bash
bash workspace/core4d_collab_retarget/scripts/run_E009_preprocess.sh
bash workspace/core4d_collab_retarget/scripts/train/train_E009.sh __codex_auth_probe__ 0
bash workspace/core4d_collab_retarget/scripts/run_E009_remote.sh __codex_auth_probe__
bash workspace/core4d_collab_retarget/scripts/pull_E009_remote_results.sh __codex_auth_probe__
bash workspace/core4d_collab_retarget/scripts/train/train_E009.sh smoke 0
.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E009.py --all
bash workspace/core4d_collab_retarget/scripts/run_E009_remote.sh
bash workspace/core4d_collab_retarget/scripts/train/train_E009.sh local_wave 0
bash workspace/core4d_collab_retarget/scripts/pull_E009_remote_results.sh
.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E009.py \
  E009_box025_p2_ypos_k20_vmax2_hc05 \
  E009_box025_p2_ypos_k20_vmax2_hc1 \
  E009_box025_p2_ypos_k20_vmax2_hc2 \
  E009_box025_p2_ypos_k20_vmax0_hc1 \
  E009_box023_p2_xpos_k10_vmax0_hc1 \
  E009_box023_p2_xpos_k10_vmax0_hc2
```

`pull_E009_remote_results.sh` 自动 eval 只覆盖了 4 个远程变体；最终结果以上面显式 6 个 full NPZ 重评为准。

## 结果路径

| 产物 | 路径 |
|------|------|
| Plan | `workspace/core4d_collab_retarget/plan/09_E009_ypos_hold_contact_closure_e081_plan.md` |
| Results | `workspace/core4d_collab_retarget/results/E009/` |
| Logs | `logs/core4d_collab_retarget/E009/` |
| Comparison | `workspace/core4d_collab_retarget/results/E009/comparison.csv` |
| Aggregate | `workspace/core4d_collab_retarget/results/E009/aggregate_summary.json` |
| Videos | `workspace/core4d_collab_retarget/results/E009/*.mp4` |
| Keyframes | `workspace/core4d_collab_retarget/results/E009/keyframes/` |

## Full 汇总

最终 full aggregate：

```json
{
  "num_results": 6,
  "num_main_results": 4,
  "num_guard_results": 2,
  "num_freejoint_parity_ok": 6,
  "num_support_proxy_metrics_present": 6,
  "num_main_proxy_timebase_ok": 4,
  "num_main_proxy_support_tracking_ok": 4,
  "num_main_reaches_E081_transport_proxy": 0,
  "num_main_beats_or_matches_E081_majority": 0,
  "num_main_improves_E008_best": 0,
  "num_guard_stable_proxy": 1
}
```

关键指标：

| Variant | Role | HC / vmax | obj mean/max (m) | hand % | floor % | leg obj % | xy ratio | rot deg | proxy ratio | vs E008 obj mean | 结论 |
|---------|------|-----------|------------------|--------|---------|-----------|----------|---------|-------------|------------------|------|
| `E009_box025_p2_ypos_k20_vmax2_hc05` | main | `0.5 / 2` | `0.395 / 0.726` | `76.9` | `62.4` | `3.5` | `0.710` | `18.2` | `0.998` | `+0.032` | 最稳定但差于 E008 best |
| `E009_box025_p2_ypos_k20_vmax2_hc1` | main | `1.0 / 2` | `0.441 / 0.825` | `74.6` | `63.0` | `19.1` | `0.737` | `15.2` | `0.998` | `+0.078` | 手接触下降且腿干涉升高 |
| `E009_box025_p2_ypos_k20_vmax2_hc2` | main | `2.0 / 2` | `0.344 / 0.695` | `66.5` | `70.5` | `15.6` | `0.671` | `38.5` | `0.998` | `-0.019` | mean 略好但靠旋转/地面，失败 |
| `E009_box025_p2_ypos_k20_vmax0_hc1` | main | `1.0 / 0` | `0.382 / 0.719` | `55.5` | `69.4` | `16.8` | `0.789` | `38.6` | `1.000` | `+0.019` | xy 足够但手接触崩、旋转过大 |
| `E009_box023_p2_xpos_k10_vmax0_hc1` | guard | `1.0 / 0` | `0.848 / 1.475` | `72.7` | `93.3` | `24.7` | `0.317` | `142.3` | `1.000` | n/a | 摔倒/干涉，不 stable |
| `E009_box023_p2_xpos_k10_vmax0_hc2` | guard | `2.0 / 0` | `0.444 / 0.739` | `12.7` | `67.3` | `5.3` | `0.851` | `90.2` | `1.000` | n/a | stable guard，但无手端协作 |

E081 main baseline：obj `0.143/0.271m`，hand `89.0%`，floor `59.5%`，leg interference `7.5%`。E009 最接近的 `hc05` 仍比 E081 obj mean 高 `0.252m`，hand contact 低约 `12pp`，rotation 超过 `15deg`，因此不能判定为基本搬运。

## 可视化观察

已用 video-frames 提取关键帧：

- `workspace/core4d_collab_retarget/results/E009/keyframes/E009_hc05_t2.jpg`
- `workspace/core4d_collab_retarget/results/E009/keyframes/E009_hc05_t4.jpg`
- `workspace/core4d_collab_retarget/results/E009/keyframes/E009_hc2_t4.jpg`
- `workspace/core4d_collab_retarget/results/E009/keyframes/E009_vmax0_hc1_t4.jpg`

实际观察：

- `hc05` 的 sim box 有明显水平移动，但仍落在 ref 后方，机器人手臂贴近箱体侧面而不是稳定托举；这与 hand `76.9%`、obj `0.395/0.726m` 一致。
- `hc1` 没有把 hand contact 推到 `>=80%`，反而增加腿/箱干涉到 `19.1%`。
- `hc2` 和 `vmax0_hc1` 出现更典型的“旋转替代平移”：xy ratio 可以接近或超过阈值，但 object rotation 达 `38deg`，手接触下降，地面接触升高。
- guard `hc1` 摔倒/干涉，guard `hc2` 姿态稳定但 hand contact 只有 `12.7%`，不能说明真实协作成立。

## 与 E006 失败现象的关系

E006 视频里的“物体几乎不平移、主要旋转”有两层原因：

1. E006 的 `support_proxy_ref_dt=0.0333` 与插值后的 `sim_dt` 不一致，proxy 自己只走了约半程；这已在 E007/E008 修正。
2. off-COM support force 的 `r x F` 会天然生成力矩；当 robot 手端没有闭环、object 仍大比例贴地时，系统更容易绕地面或局部支撑点转动，而不是形成 COM 水平运输。

E009 说明第 1 层已解决：proxy ratio `0.998-1.000`。但第 2 层仍在：hold-contact reward 不能把机器人手端变成可靠物理支撑，强 reward 反而引入旋转和腿干涉。

## Claims 验证

| Claim | 结论 | 证据 |
|-------|------|------|
| C1 E008 best 的剩余瓶颈是 robot-side hand/support 闭环 | ✅ 基本确认 | proxy gate 全过，但 hand 仍 `55-77%`，obj error 不降 |
| C2 hold-contact 不能以腿/箱干涉或摔倒换 tracking | ❌ 未通过 | main `hc1/hc2/vmax0` leg intf `15-19%`，guard `hc1` pelvis min `0.281m` |
| C3 speed/timebase 不再是主因 | ✅ 通过 | 4/4 main proxy support tracking ok |
| C4 eval 继续对齐 E081 | ✅ 通过 | 输出 E081 delta、majority score、transport gate |
| C5 若 hold-contact 无法改善，则 E010 进入 contact pad/soft constraint | ✅ 触发 | `num_main_improves_E008_best=0`，`num_main_reaches_E081_transport_proxy=0` |

## 结论

E009 不是 work。最有价值的结果是排除了“只要把手接触 reward 加强就能到 E081”的假设。当前路线的限制已经很清楚：support proxy 能把虚拟协作者端带到正确位置，但 direct wrench 没有给 robot 手端形成可用的物理反作用闭环；hold-contact reward 在动作空间里只能改变姿态搜索偏好，不能稳定地产生双端托举。

下一步 E010 应停止继续扫 hold-contact scale，转向结构性接触机制：

1. 在 object 远端加入 mocap/contact pad，让 partner support 通过 MuJoCo contact/soft constraint 作用到 object；
2. 保留 E008 best 的 `ypos` 方向作为主线，对比 pad radius/stiffness/height；
3. 仍以 E081 transport gate 为验收，不回退到 E005。
