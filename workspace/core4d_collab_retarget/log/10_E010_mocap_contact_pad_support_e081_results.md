# E010 结果：mocap contact pad 虚拟协作者支持

日期：2026-05-18

## 状态

E010 plan、contact-pad scene 生成、代码接入、脚本、预授权、smoke、本地 full、远程 full、回收和显式 7-variant eval 均已完成。结论：`mocap_pad` 工程路径成立，并且 7/7 结果保持 true-freejoint parity；但 contact pad 没有把虚拟协作者端的运动有效传给 object。5 个 main 变体全部通过 proxy tracking，却没有一个达到 E081 transport gate，也没有改善 E008 best。

这轮最关键的负结果是：只把 support point 换成一个 mocap contact pad，不足以让物体从“地面支撑/低位滑动”切换到“机器人手端 + partner 端共同搬运”。视频里 sim 物体明显滞后 ref，pad/reference 点走完整路径，但 object COM 只完成很小比例的水平位移。

## 执行命令

```bash
bash workspace/core4d_collab_retarget/scripts/run_E010_preprocess.sh
bash workspace/core4d_collab_retarget/scripts/train/train_E010.sh __codex_auth_probe__ 0
bash workspace/core4d_collab_retarget/scripts/run_E010_remote.sh __codex_auth_probe__
bash workspace/core4d_collab_retarget/scripts/pull_E010_remote_results.sh __codex_auth_probe__
bash workspace/core4d_collab_retarget/scripts/train/train_E010.sh smoke 0
.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E010.py --all
bash workspace/core4d_collab_retarget/scripts/run_E010_remote.sh
bash workspace/core4d_collab_retarget/scripts/train/train_E010.sh local_wave 0
bash workspace/core4d_collab_retarget/scripts/pull_E010_remote_results.sh
.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E010.py \
  E010_box025_p2_ypos_pad10_vmax2 \
  E010_box025_p2_ypos_pad16_vmax2 \
  E010_box025_p2_ypos_pad10_vmax0 \
  E010_box025_p2_ypos_pad10_vmax2_hc05 \
  E010_box025_p2_yneg_pad10_vmax2 \
  E010_box023_p2_xpos_pad08_vmax0 \
  E010_box023_p2_xpos_pad12_vmax0
```

最终结果以上面显式 7 个 full NPZ 重评为准。main 的 NPZ 形状为 `(124,2,43)`，guard 为 `(136,2,43)`，均不是 4-step smoke。

## 结果路径

| 产物 | 路径 |
|------|------|
| Plan | `workspace/core4d_collab_retarget/plan/10_E010_mocap_contact_pad_support_e081_plan.md` |
| Results | `workspace/core4d_collab_retarget/results/E010/` |
| Logs | `logs/core4d_collab_retarget/E010/` |
| Comparison | `workspace/core4d_collab_retarget/results/E010/comparison.csv` |
| Aggregate | `workspace/core4d_collab_retarget/results/E010/aggregate_summary.json` |
| Videos | `workspace/core4d_collab_retarget/results/E010/*.mp4` |
| Keyframes | `workspace/core4d_collab_retarget/results/E010/keyframes/` |
| Scene snapshots | `workspace/core4d_collab_retarget/results/E010/scene_snapshot/` |

## Full 汇总

最终 full aggregate：

```json
{
  "num_results": 7,
  "num_main_results": 5,
  "num_guard_results": 2,
  "num_freejoint_parity_ok": 7,
  "num_support_proxy_metrics_present": 7,
  "num_main_proxy_timebase_ok": 5,
  "num_main_proxy_support_tracking_ok": 5,
  "num_main_reaches_E081_transport_proxy": 0,
  "num_main_beats_or_matches_E081_majority": 0,
  "num_main_improves_E008_best": 0,
  "num_guard_stable_proxy": 1
}
```

关键指标：

| Variant | Role | pad / HC | obj mean/max (m) | hand % | floor % | leg obj % | xy ratio | rot deg | proxy ratio | gap mean (m) | 结论 |
|---------|------|----------|------------------|--------|---------|-----------|----------|---------|-------------|--------------|------|
| `E010_box025_p2_ypos_pad10_vmax2` | main | `0.10 / 0` | `0.714 / 1.376` | `90.8` | `92.5` | `0.6` | `0.172` | `13.3` | `0.998` | `0.658` | 手接触高但物体几乎不运输 |
| `E010_box025_p2_ypos_pad16_vmax2` | main | `0.16 / 0` | `0.698 / 1.334` | `87.3` | `84.4` | `1.2` | `0.326` | `8.9` | `0.998` | `0.650` | pad 更大略增平移，仍远低于 E008/E081 |
| `E010_box025_p2_ypos_pad10_vmax0` | main | `0.10 / 0` | `0.720 / 1.408` | `89.6` | `89.0` | `0.0` | `0.225` | `11.7` | `1.000` | `0.665` | 不限速不解决接触传力 |
| `E010_box025_p2_ypos_pad10_vmax2_hc05` | main | `0.10 / 0.5` | `0.680 / 1.295` | `83.8` | `83.8` | `2.9` | `0.276` | `17.4` | `0.998` | `0.620` | 轻 HC 略改善 error，但仍不是搬运 |
| `E010_box025_p2_yneg_pad10_vmax2` | main | `0.10 / 0` | `0.703 / 1.328` | `89.0` | `82.7` | `0.0` | `0.301` | `16.6` | `0.992` | `0.660` | side ablation 仍失败 |
| `E010_box023_p2_xpos_pad08_vmax0` | guard | `0.08 / 0` | `0.859 / 1.440` | `67.3` | `94.0` | `0.0` | `0.133` | `23.5` | `1.000` | `0.820` | stable guard，但无 transport |
| `E010_box023_p2_xpos_pad12_vmax0` | guard | `0.12 / 0.5` | `0.907 / 1.563` | `56.0` | `88.0` | `2.7` | `0.179` | `7.3` | `1.000` | `0.889` | pelvis min `0.071m`，不 stable |

E081 main baseline：obj `0.143/0.271m`，hand `89.0%`，floor `59.5%`，leg interference `7.5%`。E010 最好 main 仍比 E081 obj mean 高约 `0.54m`，floor contact 高 `23-33pp`，xy ratio 只有 `0.17-0.33`，因此不能判定为基本搬运。

## 可视化观察

已检查 E010 keyframes：

- `workspace/core4d_collab_retarget/results/E010/keyframes/E010_box025_p2_ypos_pad10_vmax2/f50.jpg`
- `workspace/core4d_collab_retarget/results/E010/keyframes/E010_box025_p2_ypos_pad10_vmax2/f204.jpg`
- `workspace/core4d_collab_retarget/results/E010/keyframes/E010_box025_p2_ypos_pad16_vmax2/f204.jpg`
- `workspace/core4d_collab_retarget/results/E010/keyframes/E010_box025_p2_yneg_pad10_vmax2/f204.jpg`

实际观察：

- ref 侧 box 已经沿参考方向完成大幅水平位移，但 sim 侧 box 仍靠近起点，视觉上更像机器人贴着箱体侧面站住，而不是把箱体搬走。
- `pad10_vmax2` 的 hand contact 达 `90.8%`，但 object xy ratio 只有 `0.172`，floor contact 达 `92.5%`；这说明“手贴住”没有转化成托举/运输。
- `pad16_vmax2` 把 xy ratio 提到 `0.326`、rot 降到 `8.9deg`，但 gap mean 仍 `0.650m`，物体仍明显滞后。pad 尺寸不是首要瓶颈。
- `yneg` 与 `ypos` 都没有恢复 E008 的平移水平；这排除了单侧方向选择作为主因。

## 与 E006 失败现象的关系

用户指出 E006 视频像“物体不平移，只旋转”，后验指标确认这是 E006 的核心失败模式：参考 `box025_p2` object 水平净位移约 `1.57m`、旋转约 `2deg`，而 E006 main 实际水平净位移只有 `0.21-0.50m`，旋转却达到 `20.6-47.6deg`。

E007/E008 已经证明 E006 的一层原因是 proxy timebase/speed gate：修正后 E008 best 能到 xy ratio `0.724`、rot `13.6deg`。E010 则说明另一个方向也失败：把 direct off-COM wrench 换成一个单点 contact pad 后，旋转确实没有爆炸，但平移也几乎消失。换句话说：

1. direct wrench 容易走“旋转替代平移”的捷径；
2. 单个 mocap contact pad 又太弱或接触几何不对，无法形成可靠水平/竖直传力；
3. 真正缺的是“双端闭合结构”：虚拟协作者端、object、机器人手端之间需要有能传力且不靠地面的约束/接触闭环。

## Claims 验证

| Claim | 结论 | 证据 |
|-------|------|------|
| C1 contact pad 能避免 direct wrench 的自由力矩捷径 | ⚠️ 部分成立 | rotation 多数低于 E006，但 object xy ratio 也降到 `0.17-0.33` |
| C2 contact pad 能给 object 提供更物理的 partner-side 反作用 | ❌ 未通过 | proxy gate 全过，但 gap mean `0.62-0.66m`，object floor `82.7-92.5%` |
| C3 如果 robot-side 仍不闭环，pad 会暴露为“虚拟人独自搬” | ✅ 暴露为另一种失败 | 不是 pad 独自搬，而是 pad 未能搬；hand contact 高也无 transport |
| C4 eval 继续对齐 E081 | ✅ 通过 | 输出 E081 majority / transport gate；5/5 main gate 均 false |
| C5 true-freejoint parity 不破坏 | ✅ 通过 | 7/7 `nu=29`、`nq_obj=7`、object actuator empty，`support_proxy_mode=mocap_pad` |

## 结论

E010 不是 work。它排除了“单个 mocap contact pad 就能替代 direct wrench”的假设。与 E006/E008/E009 串起来看，失败链条已经更清楚：

- E006：timebase/speed 截断 + off-COM wrench，表现为平移不足、旋转过量；
- E008：修正 timebase/speed 后平移显著恢复，但 robot-side 闭环不足；
- E009：hold-contact reward 不能补上 robot-side 闭环；
- E010：单 contact pad 不足以通过真实接触传力，物体重新变成高 floor contact、低 xy transport。

下一步 E011 不应继续只扫 pad size、vmax 或 HC。更有信息量的方向是做 soft equality/weld diagnostic：在 E008 best 基础上给 object 一个弱 target/constraint，定量测出“还差多少外部约束/partner coupling 才能达到 E081 transport”。如果极弱约束就能恢复搬运，再把约束换成更物理的双手/双点 partner mocap；如果弱约束也不行，则说明 robot-side retarget/search 目标本身需要重做。
