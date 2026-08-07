# E189 结果：box004/box024/box001 E167A no-PRG 与 E172/E173 PRG 三物体配对对比

_Core4D Phase 52 · 2026-08-06/07 · 43-case single-seed Full CEM ablation ·
[实验计划](../plan/215_E189_box004_box024_box001_e167a_no_prg_full_cem_plan.md)_

---

## 📋 Summary

- 问题：在 E172 box004（6条）+ E173 box024（9条）+ E173 box001（28条）共 `43` 条
  CEM-eligible authority 上（box023 已由 E179 单独回答，本轮不重跑），只移除
  E170 lower-body PRG（P/R/G 全部 `false`），三个物体各自能否保持或改善 Full
  CEM 结果，尤其是大箱（box024/box001）是否与 box023 的 `PRG_BETTER` 一致，
  还是随尺寸出现方向反转。
- 执行：E189 使用 E167A `zOnlyBody` base、关闭 PRG，固定 `seed=0`、
  `1024 samples × 32 iterations`；本机 8×L20Y（jzsy-11 主跑 + ditg-12 补跑 3
  条 NFS 中断行 + 1 条 EDGE 中断收尾）跨机执行，最终 `43/43` 完成，无终态失败。
- 主结果（十二门，按物体独立判定，禁止合并成单一结论）：

  | 物体 | n | PRG 12门 | E189 no-PRG 12门 | 结论 |
  |---|---:|---:|---:|---|
  | box004 | 6 | 2/6 | 2/6 | `NO_PRG_NONINFERIOR` |
  | box024 | 9 | 2/9 | 0/9 | `PRG_BETTER` |
  | box001 | 28 | 5/28 | 6/28 | `NO_PRG_NONINFERIOR` |

- 关键发现：**三个物体的 lower-body gate 无一例外全部退化**
  （box004 `5/6→3/6`、box024 `7/9→1/9`、box001 `19/28→15/28`），方向与 PRG
  设计初衷（下肢防穿透）完全一致，物理上可信。但**"PRG 随物体体积单调退化"
  的假设在 no-PRG 侧不成立**：体积最大的 box001（0.256m³）是
  `NO_PRG_NONINFERIOR`，体积更小的 box024（0.253m³，仅比 box001 小
  0.003m³）反而是三者中唯一明显回归的 `PRG_BETTER`；同尺寸区间内方向相反，
  说明退化幅度更依赖具体物体几何/抓取拓扑，而非单调依赖体积本身。
- 视频关键帧复核（11 条 pass-migration case 中抽样 6 条，见 §5.5）：
  box024 的 lower-body 回归有清晰的可见证据（小腿明显穿入箱体），与数值
  结论强吻合；其余 case 差异更细微但方向一致，未发现渲染损坏或明显 reward
  hacking 迹象。
- 决策：box004/box001 范围内 no-PRG 与 PRG 打平，不构成升级 no-PRG 的理由
  （仍是 non-inferior 而非 better）；box024 明确保留 PRG。三物体均不进入
  RL export，box023(E179)/box024/box001 三者共同证据表明 PRG 仍是当前default。

## 🎯 1. Experiment motivation

E172 box004、E173 box024/box001 三个物体历史上只用 `E167A + E170 PRG` 跑过
Full CEM，从未与纯 `E167A_zOnlyBody`（关闭 PRG）做过同 case 配对对比——
box023 是唯一的例外（E179，`PRG_BETTER`，16/16）。E173 报告了一个关键线索：
PRG numeric pass 率随物体体积单调退化（box004 83% → box023 81% → box021
64% → box026 42% → box024 33% → box001 36%）。E179 在小箱 box023 上验证了
移除 PRG 会让 lower-body 更差、object tracking 略好；本实验回答的是这个
trade-off 在大箱（box024/box001，体积 6-7 倍于 box023）和另一小箱（box004）
上是否成立，还是随尺寸出现方向反转。

与 E179 相同，十二门口径同时回答两个问题：

| 口径 | 回答的问题 |
|---|---|
| Physics 六门 | 动作是否满足基础物理与接触约束 |
| Physics + tracking 十二门 | 物理约束与目标动作跟踪是否同时满足 |

C0–C6（plan/215）要求的是实验闭合与证据完整，不预设 no-PRG 必须获胜，也不
要求三个物体得到统一结论——结论允许是 `NO_PRG_BETTER` /
`NO_PRG_NONINFERIOR` / `PRG_BETTER` / `MIXED` 的任意组合。

## ⚙️ 2. Experiment setup

### Paired authority

- Case set：box004(6，来自 E172) + box024(9，来自 E173) + box001(28，来自
  E173) = `43`，显式排除 box023。
- v1/v2 retarget variant 分布：box004 `5/1`、box024 `5/4`、box001 `21/7`，
  与冻结记录一致。
- E172 manifest SHA `c53d0619...`、E173 manifest SHA `2128deb8...`（与 E179
  记录一致），只读引用，未回写任何 E172/E173 结果。
- E172/E173 十二门 baseline 由 Phase 0 只读重算（复用 E179 同一 scoring
  adapter）：box004 `2/6`(33%)、box024 `2/9`(22%)、box001 `5/28`(18%)——均
  比历史六门更低，降幅比例上比 box023 的先例（六门13/16→十二门7/16）更大。

### CEM and execution

- CEM budget：seed `0`、`1024 samples × 32 opt steps`（Full），与 E172/E173
  完全一致。
- 场景：43 条各自从原始 `scene_act.xml` fresh 生成
  `scene_act_E189_rubberHull.xml`（不复用 `scene_act_{E172,E173}_..._PRG.xml`），
  PRG 负向审计 43/43 pass（`p/r/g=false`，无下肢 pair、无 leg gate、无 PRG
  诊断字段泄漏）。
- 执行环境：本机 8×L20Y（jzsy-11 主跑 43 条中的 40 条 + ditg-12 补跑
  3 条 gpu7 队列 NFS `OSError` 中断行 + ditg-12 复核 1 条被误判丢失的行
  `box024_20231011_028_p2`——该行 CEM 本身已算完（`Total time=2470s`），
  只是 `run_cem_queue.py` 的 copy 步骤撞上瞬时 NFS I/O error，未 try/except
  导致队列崩溃；已修复 `copy_with_retry`（3次重试）+ 逐行异常隔离，用
  `validate_runtime_outputs` 直接补齐缺失的 `result_npz`，**未重新计算**）。
- 最后 1 条（`box001_20231003_2_037_p2`）跨机运行到 2026-08-07 凌晨才收尾，
  `Final object tracking error: pos=0.1391, quat=0.1113`，43/43 全部有终止
  行、无 NaN、无爆炸迹象。

## 🔧 3. Core algorithm or method

`E167A_zOnlyBody` 冻结 profile（SHA
`666c302dcf549e517ff02b50c139551cf98635b7b951872284d6a39766548c17`，与
E168/E179 一致）：`e167_body_z_enabled=true`、`e167_ground_z_enabled=true`、
`cem_hand_gate_enabled=true`、`surface_band_*`、`cem_posture_gate_enabled=true`
等原生组件全部保留，只关闭 E170/E171/E172/E173 引入的 P（下肢-物理对）/
R（repulsion 软惩罚）/G（candidate gate）三个 PRG 组件。"不要 PRG" 不等于
"不要 E167A 原生的下肢/姿态约束"。

## 📊 4. Metrics

十二门 gate contract 与 E179 完全一致（`core4d-e189-12gate-tracking-v1`）：

| Failure mode | Metric | Pass |
|---|---|---:|
| fall | `fall_flag` | `false` |
| body_z | `body_z_err_p95_m` | `≤0.20m` |
| contact | `hand_object_physics_contact_in_mask_frac` | `≥0.50` |
| release | `hand_object_release_false_contact_3mm_frac` | `≤0.30` |
| hand_penetration | `hand_object_physics_penetration_3mm_frame_frac` | `≤0.30` |
| lower_body | `leg_penetration_frac` | `≤0.10` |
| root_pos/ori | `track_root_{pos_cm,ori_deg}_err_mean` | `≤20/≤20` |
| hand_pos/ori | `track_eef_{pos_cm,ori_deg}_err_mean` | `≤20/≤20` |
| object_pos/ori | `track_obj_{pos_cm,ori_deg}_err_mean` | `≤20/≤10` |

Evaluator 直接 import `eval.core.core_metrics.evaluate_sequence`，不动态
加载 E172/E173 的 evaluator；E172/E173 与 E189 的 raw metric values 用同一
E189 scoring adapter 重新打分，不能直接比较历史六门数字。三物体各自独立
统计 + 43 条合并表（合并表仅完整性校验，不作结论依据）。

## 📈 5. Results

### 5.1 Headline pass rates

| 物体 | n | Physics 6门 (PRG→E189) | 十二门 (PRG→E189) | 结论 |
|---|---:|---|---|---|
| box004 | 6 | `5/6→3/6` | `2/6→2/6` | `NO_PRG_NONINFERIOR` |
| box024 | 9 | `3/9→0/9` | `2/9→0/9` | `PRG_BETTER` |
| box001 | 28 | `10/28→9/28` | `5/28→6/28` | `NO_PRG_NONINFERIOR` |

合并 43 条（仅完整性参考）：`9/43→8/43`。

### 5.2 十二门逐门通过数（Delta = E189−PRG）

| Gate | box004 Δ | box024 Δ | box001 Δ |
|---|---:|---:|---:|
| fall | +0 | +0 | +0 |
| body-z | +0 | +0 | +0 |
| contact | +0 | +0 | +0 |
| release | +1 | +0 | +0 |
| hand penetration | +0 | -1 | +2 |
| **lower-body** | **-2** | **-6** | **-4** |
| root pos | +0 | +1 | +2 |
| root ori | +0 | -1 | +0 |
| EEF pos | +0 | +1 | +2 |
| EEF ori | +0 | -2 | -1 |
| object pos | +0 | +0 | +0 |
| object ori | +1 | +0 | +0 |

三物体 lower-body 无一例外退化，是本实验最一致的信号；hand_penetration 和
tracking 门则方向不一，box024 普遍变差、box001 普遍略好，box004 基本持平。

### 5.3 Pass migration

| 物体 | PASS→PASS | PASS→FAIL | FAIL→PASS | FAIL→FAIL |
|---|---:|---:|---:|---:|
| box004 | 1 | 1 | 1 | 3 |
| box024 | 0 | 2 | 0 | 7 |
| box001 | 2 | 3 | 4 | 19 |

box024 只有净退化、没有任何救回（`FAIL→PASS=0`），是三者中方向最一致的
负面信号；box001 反而有 `4` 条救回、只 `3` 条退化，抵消后略有净改善。

### 5.4 连续指标 paired 统计（节选，完整表见 xlsx）

- `leg_penetration_frac` 的 mean improvement（PRG−E189，负值=E189更差）：
  box004 `-0.073`、box024 `-0.238`、box001 `-0.109`——三者全部方向一致地
  变差，box024 幅度最大（bootstrap 95% CI `[0.113, 0.374]` 不跨0，统计上
  显著）。
- `track_obj_pos_err_cm_mean` improvement：box004 `+0.299`、box024
  `+0.169`、box001 `+0.451`——object tracking 三物体全部方向一致地
  略微改善（但 box024 的 CI 跨 0，不显著）。
- 完整 11 个连续指标（mean/median/IQR/bootstrap CI）见
  `E189_vs_PRG_boxes_report.md` 各物体章节与 xlsx `Summary`/三个物体 sheet。

### 5.5 Video review observations（11 条 pass-migration case 抽样 6 条关键帧）

用 `ffmpeg` 在每条 paired 对照视频（`render/paired_e172_e173/{object}/`）的
10%/50%/90% 时刻抽帧，人工核对：

- **`box024_20231011_028_p2`（PASS→FAIL, lower_body）**：50% 抓取蹲姿帧
  显示极清晰的差异——PRG sim 侧小腿在箱体外侧、姿态干净；no-PRG sim 侧
  右小腿明显穿入箱体体积内（腿部轮廓被箱体遮挡的方式与"腿在箱子后面"不
  一致，是真实几何穿透而非视角遮挡）。这是本轮最强的视觉证据，直接支持
  box024 的 `PRG_BETTER` 数值结论，排除了"lower_body 数值退化是评测口径
  问题"的可能性。
- **`box024_20231011_027_p2`（PASS→FAIL, lower_body）**：同一物体的另一条
  case，50% 帧同样显示 no-PRG 侧膝盖/小腿贴近箱体下沿，穿透迹象比
  `028_p2` 略弱但方向一致。
- **`box004_20231003_2_083_p1`（PASS→FAIL, lower_body）**：50% 蹲姿帧下
  no-PRG 侧的支撑腿脚踝位置比 PRG 侧更靠近箱体边缘，90% 站起后两侧姿态
  基本一致——差异集中在蹲取的中间阶段，与"抱起时腿更贴近物体"的
  lower_body 定义吻合，但视觉上不如 box024 剧烈。
- **`box004_20231003_2_083_p2`（FAIL→PASS, hand_ori 修复）**：两侧动作
  整体相似，未观察到明显的手部朝向差异在此分辨率下可辨识；数值改善幅度
  本身也较小。
- **`box001_20231003_2_037_p1`（FAIL→PASS, hand_penetration 修复）**、
  **`box001_20231003_2_038_p2`（PASS→FAIL, lower_body）**：均为搬箱倚靠
  姿态，两侧整体动作合理、无穿模/掉落/抖动等明显 artifact；case 间差异
  比 box024 更细微，与其数值 delta 幅度更小一致。
- **未发现的问题**：43 条视频渲染 0 失败、0 近零字节文件、0 corrupt
  文件；抽样复核的 6 条 paired 视频均无 reward hacking 迹象（动作在两侧
  都是"够物体—搬起—站立"的合理物理过程，没有诸如穿地板、瞬移等异常）。
- **未覆盖的 5 条 migration case**（`box001_20231003_2_038_p1`、
  `box001_20231003_2_039_p1`、`box001_20231003_1_040_p2`、
  `box001_20231003_1_042_p2`、`box001_20231023_110_p2`）未逐条抽帧，仅
  做了文件完整性检查（非零字节、时长正常），如需更细致的逐帧复核可在
  `render/paired_e172_e173/box001/` 下用 `/video-frames` 补做。

视觉判定：`SUPPORTS_MIXED`（box024 视觉强支持 `PRG_BETTER`；box004/box001
视觉上大体支持 `NONINFERIOR`，未见系统性视觉退化，与数值结论一致）。

## 🔍 6. How to read the tables and videos

- `E189_vs_PRG_boxes_report.md`／`.xlsx`：人类可读报告，按物体分节，
  **结论以按物体表为准，43 条合并表仅完整性参考**。
- xlsx 三个 sheet（`box004`/`box024`/`box001`）：`case_id` + 两版是否通过
  + 两版失败模式 + 12 个门各自 (PRG值/E189值/Delta) 三列一组，Delta 表头
  橙色区分，**Delta = PRG − noPRG**（与内部 tsv/json 的 `E189−PRG` 方向
  相反，仅展示层翻转，不影响原始数据文件定义）。
- `e189_vs_prg_paired.tsv`／`e189_vs_prg_gate_matrix.tsv`／
  `e189_eval_summary.json`：原始 paired 数据，`delta_{metric}` 定义为
  `E189−PRG`（与 xlsx 展示层相反，使用前注意区分）。
- 视频：`render/full/*.mp4` 为 E189 自身 43 条；
  `render/paired_e172_e173/{object}/{case_id}.mp4` 为左（PRG）右（E189）
  对照，`render_manifest.tsv` 记录每条渲染状态（43/43 rendered，0 missing）。

## 🔬 7. Interpretation

### Observed evidence

1. lower-body 门在三个物体上**方向完全一致**地退化，且 box024 有直接可见
   的腿部穿透证据——这是"PRG 移除后下肢约束消失"这一因果机制在三个不同
   物体几何上的重复验证，不是随机噪声。
2. "PRG 随体积单调退化"假设在 no-PRG 侧**未被复现**：box001（体积最大）
   反而是 non-inferior，box024（体积第二大，仅比 box001 小 1%）却是唯一
   明显回归的物体。结合 E173 报告的历史六门数字（box024 33% vs box001
   36%，两者本就非常接近），说明"体积"本身可能只是一个粗代理变量，真正
   决定 PRG 依赖程度的更可能是抓取姿态/接触拓扑等物体几何细节，而非单纯
   尺寸。
3. box004 样本量小（n=6），单条 case 迁移就能让 pass 比例摆动
   `±17%`（plan/215 已预注册此风险）；`NO_PRG_NONINFERIOR` 结论应结合
   leg_penetration 连续指标（仍然退化 `-0.073`）审慎解读，不能只看 2/6
   打平的表面数字。

### Inference

三物体证据合起来看，PRG 的下肢保护效果具有跨物体一致性（因果头稳固），
但其"值不值得保留"的判断因物体而异——box024 上代价明显超过收益，box001/
box004 上大致打平。这与其说是"体积假设"的反例，不如说是提醒：物体几何
（而非单一标量体积）才是决定是否需要 PRG 的关键变量，需要更细的物体特征
（如箱体宽高比、抓取面到地面距离）才能预测。

## ✅ 8. Conclusion and discussion

- box004：`NO_PRG_NONINFERIOR`——无充分证据支持切换到 no-PRG，也无需强制
  保留 PRG；鉴于 n=6 样本量小 + leg_penetration 仍退化，**保守选择继续
  保留 PRG 作为 box004 default**。
- box024：`PRG_BETTER`——十二门 `2/9→0/9`、lower-body `7/9→1/9`、且视频
  证据最强，**明确保留 PRG**。
- box001：`NO_PRG_NONINFERIOR`（略偏好 no-PRG，`5/28→6/28`
  且净迁移 `+1`）——但 leg_penetration 连续指标仍退化，**暂不建议切换**，
  维持 PRG default，待更多证据（如更大样本或专门的 leg_penetration 消融）
  再决定。
- 三个物体均不进入 RL/SUGAR/Holosoma export，不改变 E172/E173/E179 的任何
  历史结果、指标或人工标签，chair(E175 等) 之后再议。

### Claims verification（plan/215 C0-C6）

| Claim | 状态 | 证据 |
|---|---|---|
| C0-authority | ✅ PASS | 43=6+9+28，不含 box023；SHA 逐条核对通过（Phase 0） |
| C1-method-fidelity | ✅ PASS | 43/43 E167A profile SHA 匹配，B1/B2 未启用 |
| C2-no-prg | ✅ PASS | 43/43 `p/r/g=false`，无 PRG 诊断字段泄漏（Phase 1 审计） |
| C3-full-closure | ✅ PASS | 43/43 complete，0 terminal_failed，0 missing |
| C4-paired-metrics | ✅ PASS | 43/43 paired，`43×12=516` gate cells 全部显式 |
| C5-visual | ✅ PASS（抽样） | 43/43 self+paired 视频已渲染；11 条 migration case 中 6 条已逐帧复核，其余 5 条仅完整性检查（非零字节+正常时长），未逐帧看 |
| C6-reproducibility | ✅ PASS | config/scene/trajectory/mask/manifest/GPU/命令/SHA 全部保留在 `results/E189/` |

C5 未 100% 逐帧覆盖 11 条 migration case（6/11），属于本轮已知的抽样局限，
不影响 C0-C4/C6 的完整闭合，也不改变按物体的数值结论。

## ⚠️ 9. Limitations and caveats

- **视觉复核抽样而非全覆盖**：11 条 pass-migration case 中只有 6 条做了
  逐帧人工复核，另外 5 条（均为 box001）只做了文件完整性检查。
- **单 seed**：所有 43 条 CEM 都是 `seed=0` 单次结果，McNemar/bootstrap CI
  仅供参考，不是稳健性证据；box004 (n=6) 尤其对单 case 迁移敏感。
- **合并 43 条数字不能替代按物体结论**：`combined_43_row_secondary_only`
  中 `9/43→8/43` 看起来只是小幅下降，但掩盖了 box024 的明显回归和 box001
  的净改善两个相反方向的信号，绝不能作为跨物体统一叙事使用。
- **"体积假设"证伪范围有限**：本实验只能说明 box004/box024/box001 三点
  上体积不能单调预测 no-PRG 表现，不能推广到未测试的其它物体几何。
- **跨机执行的操作性风险已消解但记录在案**：gpu7 队列因共享 NFS 瞬时
  I/O error 中断过一次（`box024_20231011_028_p2` 影响后续 3 条排队），
  已用 `copy_with_retry` 修复并补齐，不影响本次数字，但该修复应该保留进
  后续实验的 `run_cem_queue.py` 基线。

## ✍️ 10. Next steps

1. 补齐 box001 剩余 5 条 pass-migration case 的逐帧视觉复核（可选，用
   `/video-frames` 对 `render/paired_e172_e173/box001/` 下对应文件跑）。
2. 如需回答"物体几何的哪个维度决定 PRG 依赖程度"，需要专门设计一个跨
   物体特征（宽高比、抓取面到地面距离等）与 lower_body pass 率的回归/
   相关性分析，而不是继续用体积做单一代理变量。
3. box024 的 PRG_BETTER 结论已经足够明确，不建议在该物体上重复消融；
   box001/box004 若要进一步验证 non-inferior 结论的稳健性，需要额外 seed
   而不是重跑同一 seed。
4. 本实验不改变任何 RL export 计划；chair(E175 等更极端凹几何) 优先级由
   用户决定，不受本轮 box004/box024/box001 结果直接影响（不同物体类别）。

## 📦 Reproducibility notes

### Canonical artifacts

```text
workspace/core4d/results/E189/
├── input_authority/
│   ├── input_authority.tsv
│   ├── e172_e173_baseline_12gate.tsv          # PRG 十二门 baseline（Phase 0）
│   └── hash_audit.json
├── scene_snapshot/                             # Safeguard-2，含 manifest.txt (git HEAD + sha256)
├── s5_handoff/overrides/                       # 43 条 Hydra override
└── s6_downstream/
    ├── manifests/cem_full_manifest.tsv         # 43 条权威 manifest
    ├── cem/full/                               # 43 条 result_npz + outdir_full/
    ├── render/full/                             # 43 条 E189 self MP4 + render_manifest.tsv
    ├── render/paired_e172_e173/{box004,box024,box001}/  # 43 条 paired 对照 MP4
    └── eval/full/
        ├── e189_case_metrics.tsv
        ├── e189_vs_prg_paired.tsv
        ├── e189_vs_prg_gate_matrix.tsv
        ├── e189_eval_summary.json
        ├── E189_vs_PRG_boxes_report.md
        └── E189_vs_PRG_boxes_report.xlsx
```

场景 XML（`scene_act_E189_rubberHull.xml` ×43）已 `git add -f`
（Safeguard-1，随本轮提交一并入库）。

### Canonical commands

```bash
# Phase 0: 冻结 paired authority + baseline
python3 workspace/core4d/scripts/experiments/E189/build_paired_authority.py

# Phase 1: 生成 no-PRG sidecar/override + 审计
python3 workspace/core4d/scripts/experiments/E189/build_e167a_no_prg_manifest.py --apply --snapshot
python3 workspace/core4d/scripts/experiments/E189/audit_e167a_no_prg.py --require-all

# Phase 2-3: canary + full CEM（本机 8×L20Y）
MODE=canary bash workspace/core4d/scripts/launch/active/run_E189_local_8gpu.sh
MODE=full   bash workspace/core4d/scripts/launch/active/run_E189_local_8gpu.sh

# Phase 4: 十二门 eval + paired 报表（含 xlsx）
bash workspace/core4d/scripts/eval/wrappers/eval_E189_boxes_e167a_vs_prg.sh full

# Phase 5: 渲染 self + paired 对照
bash workspace/core4d/scripts/launch/active/run_E189_render_all.sh
```

### Execution issues and resolutions

| 问题 | 根因 | 处理 |
|---|---|---|
| `import mujoco` 崩溃（CEM 阶段） | `MUJOCO_GL=osmesa/egl` 均可能触发未被保护的 `AttributeError` | CEM/审计脚本 `unset MUJOCO_GL`，落到 glfw 分支（`save_video=false` 不需要真实 GL） |
| `mujoco.Renderer()` 崩溃（渲染阶段） | `unset MUJOCO_GL` 时 glfw 因缺 `DISPLAY` 抛 `FatalError`，渲染阶段确实需要出帧 | 渲染脚本改用显式 `MUJOCO_GL=egl`（与 CEM 阶段的 GL 需求不同，不要混用同一条规则） |
| gpu7 队列崩溃 | `run_cem_queue.py` 的 npz copy 步骤撞上共享 NFS 瞬时 `OSError`，未加保护导致后续排队 case 从未尝试 | 新增 `copy_with_retry`（3次重试）+ 逐 case 异常隔离；已完成计算的 case 用 `validate_runtime_outputs` 直接补齐 `result_npz`，未重新计算 |
| 跨机（jzsy-11 + ditg-12）状态可见性 | 两台机器共享 NFS 但进程/GPU 互不可见 | 完全依赖共享 manifest/shard/log 文件的内容和 mtime 判断进度，不依赖 `ps`/`nvidia-smi` |
