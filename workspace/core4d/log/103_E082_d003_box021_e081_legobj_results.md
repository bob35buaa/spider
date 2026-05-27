# E082 Results: E081 leg-object route on 3 D003 Box021 cases

日期：2026-05-27

## 结论

E082 按 E081 的非-freejoint `scene_act` + leg/foot-object collision 路线完整跑完 3 个 D003 Box021 case，结果是 **0/3 通过**。三条都有 `.npz/.mp4/keyframes/eval`，数据与运行链路本身跑通；失败主要不是预处理没有加载，而是这条 CEM/scene_act 方法在 D003 Box021 上没有产生可靠的物理接触搬运。

核心现象：

- 物体误差很大：case-window object mean `0.457-0.849m`，max `1.138-1.480m`。
- 接触不稳定：sim contact 只有 `14.7% / 37.4% / 69.6%`，且高接触的 p1 case 是推/压/倒伏，不是有效搬运。
- 腿/箱干涉仍明显：case-window leg interference `31.8-69.8%`。
- 机器人明显倒伏：pelvis z min `0.183-0.355m`。
- 物体仍贴地/低于 ref：object bottom mean gap vs ref `-9.7/-15.4/-11.8cm`，floor-contact `75-92%`。

因此，**不建议把 E082 输出作为后续 RL seed**。如果必须在 D003 Box021 上接后续 CEM/RL，当前应优先用原始 OmniRetarget/运动学结果作为参考或重建 RL 数据目标，而不是用 E082 CEM 输出。

## 运行

| Split | Variant | GPU | 状态 | run final object err |
|---|---|---:|---|---:|
| local | `E082_d003_box021_20231018_029_p2_legobj` | local GPU0 | 完成 | pos `0.5928`, quat `0.2631` |
| remote-gpu0 | `E082_d003_box021_20231011_035_p2_legobj` | remote GPU0 | 完成 | pos `0.4112`, quat `0.2128` |
| remote-gpu1 | `E082_d003_box021_20231020_019_p1_legobj` | remote GPU1 | 完成 | pos `0.7659`, quat `0.1605` |

运行时三张卡上同时存在 R108/R109/R110 Holosoma RL 训练负载，因此耗时偏长；但三条均正常完成，没有 OOM。远程渲染结束时出现 EGL destructor warning，不影响 `.npz/.mp4` 落盘和 eval。

## 量化结果

`workspace/core4d/results/E082/comparison.csv`

| Case | Eval T | final obj pos | case obj mean/max | sim/ref contact | leg interference | pelvis z min | bottom gap vs ref | Success |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `20231018_029_p2` | 150 | `0.976m` | `0.680 / 1.138m` | `14.7% / 86.8%` | `69.8%` | `0.355m` | `-9.7cm` | False |
| `20231011_035_p2` | 266 | `0.266m` | `0.457 / 1.162m` | `37.4% / 37.9%` | `52.9%` | `0.183m` | `-15.4cm` | False |
| `20231020_019_p1` | 196 | `1.154m` | `0.849 / 1.480m` | `69.6% / 25.7%` | `31.8%` | `0.188m` | `-11.8cm` | False |

Aggregate:

```json
{
  "num_results": 3,
  "num_main_results": 3,
  "num_main_case_window_success": 0,
  "main_case_window_success_pct": 0.0,
  "num_main_legobj_strict_proxy_success": 0,
  "main_legobj_strict_proxy_success_pct": 0.0
}
```

## 可视化

视频与 keyframe sheet:

- `workspace/core4d/results/E082/E082_d003_box021_20231018_029_p2_legobj.mp4`
- `workspace/core4d/results/E082/E082_d003_box021_20231011_035_p2_legobj.mp4`
- `workspace/core4d/results/E082/E082_d003_box021_20231020_019_p1_legobj.mp4`
- `workspace/core4d/results/E082/keyframes/contact_sheets/E082_all_cases_sheet.jpg`

实际观察：

- `20231018_029_p2`: ref 是弯腰扶/搬箱，sim 很快进入低姿态，箱体倾斜和偏移明显；后段机器人趴/跪在箱子旁，手-箱接触断续，腿/脚与箱体干涉明显，不是搬运。
- `20231011_035_p2`: sim 在早中段出现站/坐/压在箱体上的姿态，箱体相对 ref 大幅漂移；后段机器人和箱体分离并倒伏，视觉上完全不是协作搬箱。
- `20231020_019_p1`: sim 有较多表面接触，但主要是推箱/压箱和倒伏；物体被推偏、翻斜，机器人最终倒在箱旁，不能作为可用动作种子。

## Claims

| Claim | 结果 |
|---|---|
| C1 在 `workspace/core4d` 完成 | 通过 |
| C2 复用 E077-E081 数据链路 | 通过 |
| C3 不污染原始 D003 source task | 通过；新增 `*_legobj_e082` 派生目录 |
| C4 腿/脚-物体 contact pair 生效 | 通过；每个派生 scene 加 16 个 pair，eval 有 leg-object 指标 |
| C5 三个 case 完成 full CEM | 通过；3/3 `.npz/.mp4/eval/keyframes` |
| C6 本地 1 卡 + 远程 2 卡并行 | 通过；但与 R108/R109/R110 共享 GPU |
| C7 后续 RL 价值判断基于视觉 + E081 指标 | 通过；结论为不建议使用 E082 输出 |

## 诊断

预处理层面没有发现阻塞性错误：mask 能加载并 resize，派生 `scene_act.xml` 能由 MuJoCo 加载，16 个 leg/foot-object pair 生效，smoke 和 full run 都完整输出。`eval_E082.py` 初始存在相对路径 bug，已修复并重跑合并评估；该 bug 只影响 eval，不影响 CEM 输出。

更可能的根因是方法层面：

- E081 的 leg-object collision 只修了“腿/脚可以参与碰撞”的物理缺口，不能自动产生 sustained hand support。
- D003 Box021 的 ref 需要持续协作支撑和接触语义，CEM 在当前 reward 下更容易找到推箱、压箱、趴箱、倒伏这些局部最优。
- 非-freejoint `scene_act` 仍带 object actuator/kinematic-object 口径，对后续 RL seed 的价值有限；一旦 CEM 输出本身倒伏/漂移，不能靠 RL 后处理自然修好。

## 下一步

- 对 D003 Box021，不再沿 E082/E081 路线继续小调参。
- 后续若要接 RL，优先使用原始 OmniRetarget/运动学 reference 或重新构造 RL target，而不是用 E082 CEM 输出。
- 如果继续做 SPIDER 前置优化，应另起实验，明确加入 support/contact semantic objective 或回到更接近 RL 训练数据需求的 kinematic/hybrid 数据路线。
