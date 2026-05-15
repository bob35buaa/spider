# E080 Results: box025 大物体边界/负控复查

日期：2026-05-15

对应计划：`workspace/core4d/plan/85_E080_box025_boundary_control_plan.md`

## 结论摘要

E080 按 E079 的 no-hold + 3cm per-EEF mask + case-specific window 口径跑了 `box025_person1/person2`。两个 case 都完成了 CEM、视频、统一评估和视觉复核。

核心结论：

- 数据与运行：`box025_person1/person2` 都用显式 trim `38/124` 生成了 3cm mask；p1 本地 RTX5090 跑完，p2 远程 `spider-remote` GPU1 跑完。
- 数值：case-specific window 下 p1/p2 都被三阈值判为 True，即 `2/2=100%`；fixed `post2` 旧口径均 False，即 `0/2=0%`。
- 视觉复核修正：用户指出 p2 视觉上“挺像搬箱子”，该观察成立。更准确的判断是：p2 是 partial positive / near-usable，可视上明显接近搬/扶箱；p1 仍是 false positive，因为腿/箱几何干涉更重、物体跟踪更差。
- 几何复核：scene 只定义了 `left_hand_object`、`right_hand_object`、`object_floor` 三类与箱子相关的 contact pair，没有腿/脚-箱子接触 pair。因此腿即使视觉靠近或穿入箱体，也不会在 MuJoCo 中给箱子提供物理支撑。
- 解释：`case-window=True` 在 box025 上说明窗口内手-箱接触 proxy、稳定性和均值物体误差达标；但它还不能区分“真实抬搬/扶搬”与“靠近箱体但存在腿部穿模/箱体高度不足”。E080 因此确认了 E079 之后的主要风险：当前三阈值需要补充 leg-box interference、object lift/floor-contact、object max/orientation 等指标。

## 实验配置

| 项 | 值 |
|----|----|
| p1 | `E080_box025_p1`, `task=box025_person1`, 本地 RTX5090 |
| p2 | `E080_box025_p2`, `task=box025_person2`, 远程 `spider-remote` GPU1 |
| base override | `core4d_e074a_box023` |
| contact mask | `contact_hdmi_mask_source=core4d_3cm` |
| hold contact | `hold_contact_rew_scale=0.0` |
| trim | p1/p2 均为 `trim_start=38, trim_frames=124` |
| palm normal | p1/p2 均自动为 left `[0,-1,0]`, right `[0,1,0]` |

## 结果路径

| 类型 | 路径 |
|------|------|
| 结果汇总 | `workspace/core4d/results/E080/comparison.csv` |
| 聚合摘要 | `workspace/core4d/results/E080/aggregate_summary.json` |
| NPZ/MP4 | `workspace/core4d/results/E080/E080_box025_p{1,2}.npz`, `.mp4` |
| timeseries | `workspace/core4d/results/E080/timeseries_E080_box025_p{1,2}.csv` |
| eval summary | `workspace/core4d/results/E080/eval_summary_E080_box025_p{1,2}.{json,csv}` |
| 关键帧 | `workspace/core4d/results/E080/keyframes/E080_box025_p{1,2}/` |
| plot | `workspace/core4d/results/E080/plots/E080_box025_p{1,2}_post2_failure_timeline.png` |
| scene snapshot | `workspace/core4d/results/E080/scene_snapshot/` |
| logs | `logs/E080/E080_box025_p1.log`, `logs/E080/E080_box025_p2.log`, `logs/E080/remote_gpu1.log` |

## 量化指标

Case-specific window 仍按 E079 口径：从当前 variant 的 `eval_contact_mask_3cm` 中取目标 person 左右手 OR 的 first/last active frame，前后 padding 10 frame。

| Variant | Split | CaseWin | Fixed post2 | Window s | Obj mean/max m | Pelvis min m | Ref/Sim contact % | Hand SDF mean m | Ctrl Linf | Visual |
|---|---|---|---|---|---|---|---|---|---|---|
| `E080_box025_p1` | local | True | False | 0.66-4.20 | 0.184/0.336 | 0.747 | 97.8/75.3 | 0.058 | 0.712 | false positive：腿/箱几何干涉重，物体偏差大 |
| `E080_box025_p2` | remote | True | False | 0.64-4.08 | 0.146/0.289 | 0.759 | 97.1/90.8 | 0.047 | 0.473 | partial positive：视觉上接近搬/扶箱，但仍需检查腿干涉和箱体高度 |

聚合结果：

```json
{
  "num_results": 2,
  "num_main_results": 2,
  "num_main_case_window_success": 2,
  "main_case_window_success_pct": 100.0,
  "num_main_numeric_success": 0,
  "main_numeric_success_pct": 0.0,
  "guard_results": []
}
```

补充诊断：

| 指标 | p1 | p2 | 解读 |
|------|----|----|------|
| `obj_err > 0.30m` 帧比例 | 17.3% | 0.0% | p2 物体跟踪明显更好 |
| `obj_err > 0.25m` 帧比例 | 24.2% | 12.1% | 两者都有中后段物体误差，p1 更重 |
| ref active 但 sim 无接触最长连续帧 | 22 帧 | 5 帧 | p2 接触连续性好于 p1 |
| full pelvis min | 0.747m | 0.759m | 都没有摔倒；稳定性指标无法区分真假搬运 |

二次复核补充：

| 指标 | p1 | p2 | 解读 |
|------|----|----|------|
| 腿/脚-箱 adjusted SDF 最小值 | `-0.137m` | `-0.046m` | 负值表示腿/脚几何体穿入箱体碰撞盒；p1 明显更严重。 |
| 腿/脚-箱 adjusted SDF `<0` 帧比例 | `101/248 = 40.7%` | `50/248 = 20.2%` | p2 仍有局部干涉，但强度和持续时间低于 p1。 |
| 手-箱 MuJoCo contact 帧数 | `142/248` | `157/248` | p2 的手-箱接触连续性更好。 |
| 腿/脚-箱 MuJoCo contact 帧数 | `0/248` | `0/248` | scene 没有腿/脚-箱 contact pair；腿部不会物理支撑箱子。 |
| sim f100 箱底高度 proxy | `-0.017m` | `-0.039m` | 用 `object_z - half_z` 粗略估计；同期 ref 约 `+0.050m/+0.053m`，sim 抬箱高度不足。 |

关键帧腿/箱几何干涉：

| Variant | f100 | f115 | f125 | f160 | f180 |
|---|---|---|---|---|---|
| p1 | left_shin `-0.121m` | left_thigh `-0.029m` | left_thigh `-0.025m` | left_thigh `+0.004m` | lf2 `-0.021m` |
| p2 | right_linkage_brace `-0.024m` | right_thigh `-0.020m` | rf3 `-0.010m` | rf3 `+0.054m` | rf2 `+0.153m` |

## 可视化观察

视频复核由 subagent 只读观察完成，结论保守汇总如下。

### `E080_box025_p1`

- `f50 / 1.00s`：ref/sim 都是上身前倾贴近箱体，手在箱体上缘或侧面附近；不像清楚抱持，更像身体靠近箱体建立支撑。
- `f75 / 1.50s`：sim 粗略贴近 ref，但手/脚附近有接触标记，姿态像贴箱移动，没有稳定抓握。
- `f100 / 2.00s`：ref 中箱体已经明显转/移；sim 与 ref 的物体位置语义开始偏离。此时 `obj_err=0.307m`，已经超过可靠跟踪范围。
- `f115-f125 / 2.30-2.50s`：sim 手多为贴近或浮在侧面/上缘，手指没有包住箱体，更像推/靠而非搬运。
- `f160 / 3.20s`：视觉仍靠近箱体，但 timeseries 显示 `sim_total_contact_count=0`，说明接触代理已经断过。
- `f180 / 3.60s`：sim 又回到弯腰贴箱状态，像重新靠上箱体，不像自然释放/放下。

结论：p1 是明确 false positive。原因不是“完全不像搬箱子”，而是有明显腿/箱穿入和较大物体偏差；case-window True 不能单独支持“真实搬运成功”。

### `E080_box025_p2`

- `f50 / 1.00s`：ref/sim 都从箱体侧后方接触，粗语义比 p1 更接近。
- `f75 / 1.50s`：sim 与 ref 都像扶着箱体走，但不像稳定抬搬。
- `f100 / 2.00s`：sim 大体跟随 ref，但 `obj_err=0.238m`，已有明显物体偏差。
- `f115-f125 / 2.30-2.50s`：sim 手保持在箱侧，更像推/扶；该段右脚步态也没有跟上 ref。
- `f145 / 2.90s`：sim 接触连续性好于 p1，但没有明显抓握或箱体离地证据。
- `f160-f180 / 3.20-3.60s`：后段释放不清楚，sim 到 `f180` 手仍在箱侧附近。
- `f204 / 4.08s`：timeseries 显示 `sim_total_contact_count=0`、`sim_min_hand_sdf=0.114m`，说明窗口末端已脱离。

结论修正：p2 视觉上确实接近搬/扶箱，应标为 partial positive / near-usable，而不是简单否定。它仍不能直接作为严格成功样例，原因是 sim 箱体高度低于 ref，且 f100-f145 仍有右腿/脚与箱体的局部几何干涉；这些问题需要进入下一版指标。

## Claims 验证

| Claim | 结果 | 说明 |
|------|------|------|
| C1 pipeline 能处理 box025 p1/p2 | 通过 | 两个 mask 均生成成功，`trim_start=38`、`T_spider=124`、`T_eval=207`。 |
| C2 no-hold CEM 能直接跑 box025 p1/p2 | 通过 | p1 本地、p2 远程 GPU1 均完整产出 `.npz/.mp4/eval summary`。 |
| C3 box025 不应被误读为泛化成功 | 部分通过，需修正表述 | p1 是 false positive；p2 视觉上接近搬/扶箱，不能简单归为失败。当前三阈值仍不足，因为没有 leg-box interference 和 object lift/floor-contact。 |
| C4 p1/p2 差异帮助判断失败来源 | 通过 | p2 比 p1 明显更好，说明 person/case 质量差异很关键；box025 不应只按 Tier3/drop 一刀切，需要区分 p1 false positive 与 p2 partial positive。 |
| C5 远程只用 GPU1 | 通过 | 远程脚本以 `REMOTE_GPU=1` 启动；tmux session `E080` 完成 p2。 |

## 分析

### 1. box025 不是单纯负例，p2 是有价值的边界正信号

E054 已判定 box025 是 Tier3/drop，`dim_max=0.89m` 超出单 G1 的可行范围。E080 二次复核后需要更细分：p1 支持负控判断；p2 视觉上更接近真实搬/扶箱，说明该 case 不是完全不可用，而是处在“可视上接近、物理/几何指标仍不充分”的边界区域。

### 2. case-window 三阈值在大物体上会误判

两个 case 都满足：

```text
case_window_pelvis_z_min_m >= 0.55
case_window_sim_contact_frames_pct >= 50
case_window_obj_err_mean_m <= 0.20
```

但视觉仍失败。原因是：

- pelvis 高只能说明没摔倒，不能说明搬运成功；
- sim contact % 是几何接触 proxy，大物体上容易通过贴/靠/推得到；
- object mean error 会被窗口均值掩盖，p1 `obj_err_max=0.336m` 已经明显偏离；
- 当前指标没有物体姿态/旋转、接触力学方向、释放语义和“抓持 vs 推扶”的判别。

### 3. p2 比 p1 好，且应改变原始文字结论

p2 的 `obj_err_mean/max`、contact continuity、ctrl deviation 都优于 p1；视觉也更接近 ref。原日志里“p1/p2 都不应判为真实搬运”的表述过强。更准确的说法是：p1 是 false positive；p2 是 partial positive / near-usable，需要用腿/箱干涉、箱体离地高度、物体姿态误差进一步判定能否进入正样本集。

## 下一步建议

1. 将 E080 作为 E079 成功判据的修正证据：case-window 三阈值不能单独作为 C3 成功标准，但 p2 不能被简单当作失败。
2. 在下一版 eval 中加入：
   - `case_window_obj_err_max_m` 阈值；
   - object orientation / rotation error；
   - contact continuity/dropout；
   - early contact establishment；
   - release/putdown 阶段语义；
   - `visual_usable` / `semantic_success` 标签。
3. 后续泛化验证对 Tier3/drop case 分开统计：p1 作为负控，p2 作为边界 partial positive，不能混在普通成功率里。
4. 若继续研究 box025，下一步优先做 leg-box interference/lift-aware eval，而不是直接调 reward。

## 复现命令

```bash
bash workspace/core4d/scripts/run_E080_preprocess.sh
bash workspace/core4d/scripts/train/train_E080.sh local 0
bash workspace/core4d/scripts/run_E080_remote.sh
REMOTE_HOST=spider-remote REMOTE_REPO=/home/xiayb/pHRI_workspace/spider bash workspace/core4d/scripts/pull_E080_remote_results.sh
.venv/bin/python workspace/core4d/scripts/eval/eval_E080.py
```
