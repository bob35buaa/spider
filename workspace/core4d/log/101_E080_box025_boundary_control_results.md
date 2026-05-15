# E080 Results: box025 大物体边界/负控复查

日期：2026-05-15

对应计划：`workspace/core4d/plan/85_E080_box025_boundary_control_plan.md`

## 结论摘要

E080 按 E079 的 no-hold + 3cm per-EEF mask + case-specific window 口径跑了 `box025_person1/person2`。两个 case 都完成了 CEM、视频、统一评估和视觉复核。

核心结论：

- 数据与运行：`box025_person1/person2` 都用显式 trim `38/124` 生成了 3cm mask；p1 本地 RTX5090 跑完，p2 远程 `spider-remote` GPU1 跑完。
- 数值：case-specific window 下 p1/p2 都被三阈值判为 True，即 `2/2=100%`；fixed `post2` 旧口径均 False，即 `0/2=0%`。
- 视觉：subagent 复核认为 p2 明显好于 p1，但两者都不应判为“真实搬运 box025”。p1 更像趴箱/贴箱/推箱，存在物体支撑和假接触嫌疑；p2 更接近“扶着箱体移动”，但仍更像推/扶，不是稳定抓持或抬搬。
- 解释：`case-window=True` 在 box025 上只说明窗口内几何接触 proxy 和均值物体误差过阈值；它不能证明大物体语义搬运成功。E080 因此确认了 E079 之后的主要风险：当前三阈值会把大物体边界/负控 case 误读为成功。

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
| `E080_box025_p1` | local | True | False | 0.66-4.20 | 0.184/0.336 | 0.747 | 97.8/75.3 | 0.058 | 0.712 | false positive：趴箱/贴箱/推箱，不是真实搬运 |
| `E080_box025_p2` | remote | True | False | 0.64-4.08 | 0.146/0.289 | 0.759 | 97.1/90.8 | 0.047 | 0.473 | partial/false positive：比 p1 好，像扶/推箱，不是稳定抓持搬运 |

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

## 可视化观察

视频复核由 subagent 只读观察完成，结论保守汇总如下。

### `E080_box025_p1`

- `f50 / 1.00s`：ref/sim 都是上身前倾贴近箱体，手在箱体上缘或侧面附近；不像清楚抱持，更像身体靠近箱体建立支撑。
- `f75 / 1.50s`：sim 粗略贴近 ref，但手/脚附近有接触标记，姿态像贴箱移动，没有稳定抓握。
- `f100 / 2.00s`：ref 中箱体已经明显转/移；sim 与 ref 的物体位置语义开始偏离。此时 `obj_err=0.307m`，已经超过可靠跟踪范围。
- `f115-f125 / 2.30-2.50s`：sim 手多为贴近或浮在侧面/上缘，手指没有包住箱体，更像推/靠而非搬运。
- `f160 / 3.20s`：视觉仍靠近箱体，但 timeseries 显示 `sim_total_contact_count=0`，说明接触代理已经断过。
- `f180 / 3.60s`：sim 又回到弯腰贴箱状态，像重新靠上箱体，不像自然释放/放下。

结论：p1 是明确视觉 false positive。case-window True 不能支持“真实搬运成功”。

### `E080_box025_p2`

- `f50 / 1.00s`：ref/sim 都从箱体侧后方接触，粗语义比 p1 更接近。
- `f75 / 1.50s`：sim 与 ref 都像扶着箱体走，但不像稳定抬搬。
- `f100 / 2.00s`：sim 大体跟随 ref，但 `obj_err=0.238m`，已有明显物体偏差。
- `f115-f125 / 2.30-2.50s`：sim 手保持在箱侧，更像推/扶；该段右脚步态也没有跟上 ref。
- `f145 / 2.90s`：sim 接触连续性好于 p1，但没有明显抓握或箱体离地证据。
- `f160-f180 / 3.20-3.60s`：后段释放不清楚，sim 到 `f180` 手仍在箱侧附近。
- `f204 / 4.08s`：timeseries 显示 `sim_total_contact_count=0`、`sim_min_hand_sdf=0.114m`，说明窗口末端已脱离。

结论：p2 比 p1 更好，可视为“接触/推扶箱体的粗阶段匹配”，但不能作为严格真实搬运成功样例。

## Claims 验证

| Claim | 结果 | 说明 |
|------|------|------|
| C1 pipeline 能处理 box025 p1/p2 | 通过 | 两个 mask 均生成成功，`trim_start=38`、`T_spider=124`、`T_eval=207`。 |
| C2 no-hold CEM 能直接跑 box025 p1/p2 | 通过 | p1 本地、p2 远程 GPU1 均完整产出 `.npz/.mp4/eval summary`。 |
| C3 box025 不应被误读为泛化成功 | 通过但暴露判据问题 | 人工视觉没有把 p1/p2 判为真实搬运；但 case-window 三阈值把两者都判 True，说明该数值口径仍会误判大物体负控。 |
| C4 p1/p2 差异帮助判断失败来源 | 通过 | p2 比 p1 明显更好，但两者都不是稳定抓持搬运；这支持 box025 主要是大物体结构性边界，而不是单个 person 数据质量问题。 |
| C5 远程只用 GPU1 | 通过 | 远程脚本以 `REMOTE_GPU=1` 启动；tmux session `E080` 完成 p2。 |

## 分析

### 1. box025 是成功判据的负控，不是算法正例

E054 已判定 box025 是 Tier3/drop，`dim_max=0.89m` 超出单 G1 的可行范围。E080 的视觉结果与这一历史判断一致：机器人能稳定站住，也能贴近/推扶箱体，但不能形成可信的抓持搬运。

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

### 3. p2 比 p1 好，但不足以改变 box025 结论

p2 的 `obj_err_mean/max`、contact continuity、ctrl deviation 都优于 p1；视觉也更接近 ref。这个差异说明 person2 数据或动作更适合当前方法。但 p2 仍只是扶/推箱体的粗阶段匹配，不是严格搬运成功。因此它不能推翻 box025 Tier3/drop 的历史结论。

## 下一步建议

1. 将 E080 作为 E079 成功判据的负控证据：case-window 三阈值不能单独作为 C3 成功标准。
2. 在下一版 eval 中加入：
   - `case_window_obj_err_max_m` 阈值；
   - object orientation / rotation error；
   - contact continuity/dropout；
   - early contact establishment；
   - release/putdown 阶段语义；
   - `visual_usable` / `semantic_success` 标签。
3. 后续泛化验证不要把 Tier3/drop case 纳入成功率正样本；box025 只用于边界/负控或双机器人/大 humanoid 路线。
4. 若要进一步研究 box025，应转向双人/双机器人或更大机器人，不应继续在单 G1 + 当前 CEM reward 上调参。

## 复现命令

```bash
bash workspace/core4d/scripts/run_E080_preprocess.sh
bash workspace/core4d/scripts/train/train_E080.sh local 0
bash workspace/core4d/scripts/run_E080_remote.sh
REMOTE_HOST=spider-remote REMOTE_REPO=/home/xiayb/pHRI_workspace/spider bash workspace/core4d/scripts/pull_E080_remote_results.sh
.venv/bin/python workspace/core4d/scripts/eval/eval_E080.py
```
