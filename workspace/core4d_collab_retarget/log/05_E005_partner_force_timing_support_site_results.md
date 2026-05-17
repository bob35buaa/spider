# E005 结果：partner-force ref_dt 显式化与 support-site 几何

日期：2026-05-17

## 状态

Setup、smoke、`box025` main full sweep 已完成；远程结果已回收。本次在 E004 基础上做一个代码级修正：`partner_force` 的 reference frame indexing 不再写死为 30Hz，而是使用每个 task 显式给出的 `partner_force_ref_dt`。`box025` 使用 task_info 中的 `0.03333333333333333`；`box023` 使用默认 `0.02`。

本实验还新增 off-COM support-site：在 object-local 支撑点施加等效 wrench `F, r x F`，用来测试“协作者支撑点几何”是否比 COM-only force 更符合双人搬运语义。

远程 `box023_p2_xneg_s10` 在 `sim_steps=34/272` 后日志不再更新，GPU 利用率为 0%，进程仍占 CPU；已判定为卡住并终止。`box023_p2_xpos_s10` 因同一远程序列未继续启动。因此最终 full 分析只纳入 7 个实际完整结果：6 个 `box025` main + 1 个 `box023` COM guard。两个 `box023` support-site 文件仅有 smoke 级 4-step NPZ，已从 full 分析中排除。

## 计划执行命令

```bash
bash workspace/core4d_collab_retarget/scripts/run_E005_preprocess.sh
bash workspace/core4d_collab_retarget/scripts/train/train_E005.sh smoke 0
bash workspace/core4d_collab_retarget/scripts/train/train_E005.sh local_wave 0
bash workspace/core4d_collab_retarget/scripts/run_E005_remote.sh
bash workspace/core4d_collab_retarget/scripts/pull_E005_remote_results.sh
.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E005.py --all
```

## 配置

变体设计：

- corrected/explicit-ref-dt COM spring (`com_s20/s40`)：作为 E004 COM-force 的对照；
- off-COM support-site force：`box025` 使用 object-local `-Y/+Y`，`box023` 使用 object-local `-X/+X`，方向来自 preprocess 计算的 ref contact normal；
- `box023_p2` guard 变体用于检查稳定性；
- 1 个 hold-contact 变体用于检查机器人参与度。

所有变体必须保持 true-freejoint parity：`scene.xml`、`contact_guidance=false`、`scene_name=""`、`object_action_dims=0`、`object_actuator_ids=[]`、`kp_rot=0`。

## Smoke

执行命令：

```bash
bash workspace/core4d_collab_retarget/scripts/run_E005_preprocess.sh
bash workspace/core4d_collab_retarget/scripts/train/train_E005.sh smoke 0
bash workspace/core4d_collab_retarget/scripts/train/train_E005.sh eval --all
```

Smoke 使用 `max_sim_steps=4`，只验证 wiring、CUDA/Warp 路径与 parity，不作为任务成功依据。

汇总：

```json
{
  "num_results": 9,
  "num_freejoint_parity_ok": 9,
  "num_main_results": 6,
  "num_guard_results": 3,
  "num_guard_stable_proxy": 3
}
```

## 结果路径

| 产物 | 路径 |
|------|------|
| Results | `workspace/core4d_collab_retarget/results/E005/` |
| Logs | `logs/core4d_collab_retarget/E005/` |
| Overrides | `examples/config/override/core4d_collab_E005_*.yaml` |
| Variants | `workspace/core4d_collab_retarget/scripts/E005/variants.tsv` |

## Full 结果

评估命令：

```bash
bash workspace/core4d_collab_retarget/scripts/pull_E005_remote_results.sh
.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E005.py \
  E005_box025_p2_com_s20 \
  E005_box025_p2_yneg_s20 \
  E005_box025_p2_yneg_s20_hc \
  E005_box025_p2_com_s40 \
  E005_box025_p2_yneg_s40 \
  E005_box025_p2_ypos_s20 \
  E005_box023_p2_com_s10
```

汇总：

```json
{
  "num_results": 7,
  "num_main_results": 6,
  "num_guard_results": 1,
  "num_freejoint_parity_ok": 7,
  "num_timing_improved_proxy": 0,
  "num_main_useful_proxy": 0,
  "num_main_strong_proxy": 0,
  "num_guard_stable_proxy": 1
}
```

关键指标：

| Variant | Mode | kp | ref_dt | obj mean/max (m) | hand contact % | floor % | bottom mean (m) | leg obj % | pelvis min (m) | 结论 |
|---------|------|----|--------|------------------|----------------|---------|-----------------|-----------|----------------|------|
| `E005_box025_p2_com_s20` | COM | 20 | 0.0333 | `0.678 / 1.325` | `85.5` | `71.7` | `-0.068` | `2.3` | `0.776` | 本轮 main 最好但仍远未 useful |
| `E005_box025_p2_com_s40` | COM | 40 | 0.0333 | `0.729 / 1.412` | `88.4` | `83.2` | `-0.073` | `0.0` | `0.777` | 更大 kp 没有改善 |
| `E005_box025_p2_yneg_s20` | support-site | 20 | 0.0333 | `0.760 / 1.545` | `68.8` | `96.5` | `-0.015` | `0.0` | `0.772` | 远端点力更差，地面依赖更强 |
| `E005_box025_p2_yneg_s40` | support-site | 40 | 0.0333 | `0.766 / 1.521` | `83.2` | `96.0` | `-0.036` | `0.0` | `0.776` | 加大 kp 无效 |
| `E005_box025_p2_ypos_s20` | support-site | 20 | 0.0333 | `0.710 / 1.422` | `85.5` | `94.2` | `-0.062` | `0.0` | `0.776` | side ablation 也未优于 COM |
| `E005_box025_p2_yneg_s20_hc` | support-site + HC | 20 | 0.0333 | `0.763 / 1.499` | `75.1` | `97.1` | `-0.037` | `0.0` | `0.771` | hold-contact 未恢复机器人参与 |
| `E005_box023_p2_com_s10` | COM guard | 10 | 0.02 | `0.821 / 1.448` | `64.7` | `87.3` | `0.031` | `0.0` | `0.660` | 稳定但不运输，50Hz timing 未改善 |

对比基线：

- E004 `box025_p2_g05`: `0.660 / 1.288m`，floor `71.1%`。E005 最好 `com_s20` 与其接近但没有超过。
- E004 `box025_p2_s20/s40`: 约 `0.77 / 1.55m`。E005 corrected COM 有小幅改善，说明 timing/参数整理有帮助，但不是关键瓶颈。
- E003 `box025_m1_f4`: `0.400 / 0.795m`，floor `76.9%`。E005 全部远差于该物理参数 sweep 最好结果。
- E002 `box023_p2_freejoint`: `0.830 / 1.488m`。E005 `box023_com_s10` 几乎相同，说明 box023 guard 的 E004 问题不是 ref_dt 一项能解决。

## 可视化

关键帧拼图：

`workspace/core4d_collab_retarget/results/E005/e005_full_keyframe_montage.jpg`

实际观察：

- `box025` COM 与 support-site 的 sim object 都没有形成真实搬运，箱体多数阶段仍贴近地面或被地面隐式支撑。
- support-site 变体在视觉上没有表现出“双端支撑后箱体被托起”；量化上 floor contact 反而升至 `94-97%`。
- `yneg_s20_hc` 虽然加入 hold-contact，但手-物参与没有变成稳定托举，视觉上仍是虚拟力/地面主导。
- `box023_com_s10` 机器人姿态稳定，但物体明显落后/偏离参考，属于 guard stable 而非 transport。

## Claims 验证

| Claim | 结论 | 证据 |
|-------|------|------|
| C1 partner-force timing 必须按 task 显式处理 | ✅ 通过 | 所有 full 结果 parity ok；`box025 ref_dt=0.0333`，`box023 ref_dt=0.02` |
| C2 仅修 timing 仍不足以恢复协作搬运 | ✅ 通过 | corrected COM 最好 `0.678/1.325m`，未达到 useful proxy |
| C3 off-COM support-site 比 COM force 更有物理语义 | ❌ 当前实现不成立 | 同 kp 下 `yneg_s20` 比 `com_s20` 更差，floor `96.5%` vs `71.7%` |
| C4 support-site 方向必须可解释 | ❌ main side ablation 未找到好方向 | `ypos_s20` 比 `yneg_s20` 好一些，但仍差于 COM 且 floor `94.2%` |
| C5 guard 不应被 support-site 破坏 | ⚠️ 未完成 | 远程 `box023_xneg_s10` 卡住，`xpos_s10` 未启动；COM guard 稳定但不运输 |

## 结论与下一步

E005 说明：把 partner-force ref_dt 显式化是必要的代码修正，但不是 CORE4D true-freejoint 搬运失败的主因。当前“直接对 object COM 或 object-local support site 施加等效外力”的路线已经显示出上限：它能轻微改善某些 tracking 数值，但不会生成真实的双端协作搬运，并且 support-site 版本更容易变成地面/虚拟力主导。

下一步不建议继续加密 `partner_force_point_local` / `kp` sweep。结合 Du et al. COLA 的论文笔记，E006 应转向结构性 proxy：增加独立 dynamic support body，通过 soft constraint/contact 与 object support site 传力，并记录 proxy-object interaction force。这样才能区分“另一端真实人类提供的支撑”与“直接把 object 轨迹 actuator 化”的假象。
